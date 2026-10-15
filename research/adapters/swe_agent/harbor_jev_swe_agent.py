"""Harbor agent: SWE-agent (v1.1.0) with optional per-step routing by a decision model (E25, harness B).

Harbor's built-in ``swe-agent`` agent installs SWE-agent from git into ``/opt/sweagent-venv`` in the task container and
runs ``sweagent run ... --env.deployment.type=local``. This subclass changes three things, the same in EVERY E25 arm:

1. litellm is pinned (``--ak litellm_version=1.82.6``, the version of the project's other Gemini runs) inside the venv;
2. the SWE-agent config is ``adapters/swe_agent/e25_default.yaml`` (SWE-agent's default without the Anthropic
   cache-control processor, plus the registered caps and Gemini's minimal thinking), uploaded as
   ``/opt/jev-swe/e25_default.yaml`` and passed through Harbor's ``SWEAGENT_CONFIG``;
3. the container's ``sweagent`` wrapper runs ``/opt/jev-swe/jev_sweagent.py`` on SWE-agent's venv interpreter without
   activating the venv (``WRAPPER``: the agent's shell keeps the task's testbed Python), and that starts SWE-agent unchanged when
   ``--ak jev_router=off`` (``baseline``, ``always_cheap``) and routes each step between the strong model and
   ``JEV_ROUTER_CHEAP`` when ``--ak jev_router=trigger`` (``jev_router``; settings ``JEV_*`` through ``--ae``).

E29 (``--ak llm_proxy=http://host.docker.internal:8091/e29/<arm>``): every arm runs with ``jev_router=off`` and SWE-agent's
model posts to the host-side routing proxy (``jevharness.adapters.route_proxy``) at ``<llm_proxy>/<trial>/v1beta``; the
proxy holds the Gemini key, so Harbor passes only a dummy ``GEMINI_API_KEY`` into the container.

It also fills Harbor's ``cost_usd`` from SWE-agent's trajectory (``info.model_stats.instance_cost``; Harbor reads
``info.total_cost`` / ``info.cost``, which SWE-agent 1.1.0 does not write), so every arm's result.json has its cost.

    PYTHONPATH=adapters/swe_agent harbor run -d swebench-verified@1.0 -a harbor_jev_swe_agent:JevSweAgent \\
        -m gemini/gemini-3.5-flash --ak version=v1.1.0 --ak jev_router=trigger \\
        --ae JEV_BASE_URL=http://host.docker.internal:8090 --ae JEV_ROUTER_CHEAP=gemini/gemini-3.5-flash-lite \\
        --extra-docker-compose adapters/pi/harbor_host_gateway.yaml -i 'django__django-11099'
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path, PurePosixPath
from typing import Any, override

from pydantic import Field

from harbor.agents.installed.swe_agent import SweAgent, SweAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

HERE = Path(__file__).resolve().parent
REMOTE_DIR = PurePosixPath("/opt/jev-swe")
RUNNER = "jev_sweagent.py"
CONFIG = "e25_default.yaml"
ROUTER_MODES = ("off", "trigger", "shadow")
# The container's `sweagent` command. SWE-agent runs on its own venv's interpreter by absolute path, WITHOUT activating
# the venv: Harbor's wrapper sources the venv's activate script, which puts the venv (Python 3.12) first on PATH after
# Harbor's `conda activate testbed`, so the agent's `python` was SWE-agent's interpreter instead of the task's testbed
# environment (found by E25's first SWE smoke: `which python` -> /opt/sweagent-venv/bin/python; Harbor's own comment in
# SweAgent.run says the agent's python must be the testbed's). Same wrapper in every arm.
WRAPPER = ("#!/bin/bash\n"
           f'exec /opt/sweagent-venv/bin/python {(REMOTE_DIR / RUNNER).as_posix()} "$@"\n')


class JevSweAgentOptions(SweAgentOptions):
    jev_router: str = Field(default="off", description="Per-step routing: off | trigger | shadow.")
    litellm_version: str = Field(default="1.82.6", description="litellm pinned in SWE-agent's venv ('' = as installed).")
    llm_proxy: str = Field(default="", description="E29: base URL of the host-side routing proxy for this arm "
                                                   "(jevharness.adapters.route_proxy); empty = litellm calls Gemini.")


def episode_name(logs_dir: Path | str) -> str:
    """The trial's name (the parent of the agent's logs dir), safe for a URL path segment."""
    name = Path(logs_dir).parent.name or "episode"
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name)


def trajectory_cost(traj: dict[str, Any]) -> float | None:
    """SWE-agent's cost of an instance from its .traj (``info.model_stats.instance_cost``)."""
    info = traj.get("info") or {}
    stats = info.get("model_stats") or {}
    for v in (info.get("total_cost"), info.get("cost"), stats.get("instance_cost")):
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


class JevSweAgent(SweAgent):
    options_model = JevSweAgentOptions
    options: JevSweAgentOptions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        mode = (self.options.jev_router or "off").strip()
        if mode not in ROUTER_MODES:
            raise ValueError(f"jev_router must be one of {ROUTER_MODES}, got {mode!r}")
        # Trial wires extra_env into the agent's execs (and _get_env reads it): config path and router mode only
        self._extra_env["SWEAGENT_CONFIG"] = str(REMOTE_DIR / CONFIG)
        self._extra_env["JEV_ROUTER_MODE"] = mode
        self._extra_env.setdefault("JEV_LOG", "/logs/agent/jev-decisions.jsonl")
        self._extra_env.setdefault("JEV_CALLS_LOG", "/logs/agent/jev-calls.jsonl")
        proxy = (self.options.llm_proxy or "").strip().rstrip("/")
        if proxy:   # E29: SWE-agent's model talks to the proxy (jev_sweagent.proxy_args adds --agent.model.api_base)
            if mode != "off":
                raise ValueError("llm_proxy routes in the proxy: run it with jev_router=off")
            self._extra_env["E29_API_BASE"] = f"{proxy}/{episode_name(self.logs_dir)}/v1beta"

    @staticmethod
    @override
    def name() -> str:
        return "swe-agent-jev"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        pin = (self.options.litellm_version or "").strip()
        if pin:
            await self.exec_as_root(
                environment,
                command=('if [ -f "$HOME/.local/bin/env" ]; then source "$HOME/.local/bin/env"; fi && '
                         f"uv pip install --python /opt/sweagent-venv/bin/python -q {shlex.quote('litellm==' + pin)}"),
            )
        await self.exec_as_root(environment, command=f"mkdir -p {shlex.quote(REMOTE_DIR.as_posix())}")
        for filename in (RUNNER, CONFIG):
            await environment.upload_file(HERE / filename, (REMOTE_DIR / filename).as_posix())
        await self.exec_as_root(
            environment,
            command=(
                f"chmod 644 {shlex.quote((REMOTE_DIR / RUNNER).as_posix())} "
                f"{shlex.quote((REMOTE_DIR / CONFIG).as_posix())} && "
                "cat > /usr/local/bin/sweagent << 'WRAPPER'\n"
                f"{WRAPPER}"
                "WRAPPER\n"
                "chmod +x /usr/local/bin/sweagent"
            ),
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        super().populate_context_post_run(context)
        if context.cost_usd:
            return
        path = self._find_trajectory_file() or (self.logs_dir / "swe-agent.trajectory.json")
        try:
            cost = trajectory_cost(json.loads(Path(path).read_text()))
        except (OSError, ValueError, TypeError):
            cost = None
        if cost is not None:
            context.cost_usd = cost
