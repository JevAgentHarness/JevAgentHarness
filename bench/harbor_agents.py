"""Harbor agents for the JevAgentHarness benchmarks (Harbor 0.24.0; see ``bench/README.md``).

Harbor (github.com/laude-institute/harbor) ships Codex, pi, mini-swe-agent and Terminus-2 agents; each reaches the
proxy through its own custom-endpoint options (see ``bench/run_harbor.sh``). The subclasses here only add the Harbor
trial name to the proxy URL (``/s/<trial>/v1``), so the proxy logs every step under the trial and
``bench/summarize.py`` can join the two. DeepSeek Harness (``dsh``) has no Harbor agent, so it is defined here::

    PYTHONPATH=bench harbor run -d terminal-bench@2.0 -a harbor_agents:Dsh -m jah/jah-auto \\
        --ak proxy_url=http://host.docker.internal:8787/v1 --extra-docker-compose bench/host_gateway.yaml

The agent installs dsh with Node 22, writes a Cordis patch that points dsh's pi-ai adapter at the proxy (an
OpenAI-compatible route; the proxy holds every provider key, dsh gets a dummy one), and runs one headless turn with
full permissions inside the task container. dsh's NDJSON event stream is kept as ``/logs/agent/dsh.jsonl``.
"""

from __future__ import annotations

import base64
import json
import shlex
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, override

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.codex import Codex
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.agents.installed.pi import Pi
from harbor.agents.options import InstalledAgentOptions
from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from pydantic import Field

DSH_PACKAGE = "@deepseek-ai/dsh"
DSH_VERSION = "0.2.1-alpha.2"
REMOTE = PurePosixPath("/tmp/jah-dsh")
DUMMY_KEY = "jah-proxy-holds-the-keys"


def trial_name(logs_dir: Any) -> str:
    """The Harbor trial (the parent of the agent's logs dir), safe for a URL path segment."""
    name = Path(logs_dir).parent.name or "trial"
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name)


def with_session(url: str, session: str) -> str:
    """``http://host:8787/v1`` -> ``http://host:8787/s/<session>/v1``: the proxy logs every step under the trial."""
    base = url.rstrip("/")
    root, sep, tail = base.rpartition("/v1")
    return f"{root}/s/{session}/v1{tail}" if sep else f"{base}/s/{session}"


CODEX_CATALOG = "/tmp/jah-codex/catalog.json"


class JahCodex(Codex):
    """Harbor's Codex agent; the proxy URL in ``--ak config=...`` gets the trial name as the proxy session, and the
    model names jah-* get Codex model metadata (``codex_catalog.json``: GPT-5.5's tools and prompts), so Codex
    offers its freeform apply_patch tool instead of the unknown-model fallback."""

    @staticmethod
    @override
    def name() -> str:
        return "codex-jah"

    @property
    @override
    def config_source(self) -> Any:
        cfg = super().config_source
        if isinstance(cfg, dict):
            for prov in (cfg.get("model_providers") or {}).values():
                if isinstance(prov, dict) and prov.get("base_url"):
                    prov["base_url"] = with_session(prov["base_url"], trial_name(self.logs_dir))
            cfg.setdefault("model_catalog_json", CODEX_CATALOG)
        return cfg

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        await self.exec_as_agent(environment, command=f"mkdir -p {PurePosixPath(CODEX_CATALOG).parent}")
        await self._upload_agent_owned_file(environment, Path(__file__).resolve().parent / "codex_catalog.json",
                                            CODEX_CATALOG)


class JahPi(Pi):
    """Harbor's pi agent with a custom OpenAI-compatible endpoint; the trial name becomes the proxy session."""

    @staticmethod
    @override
    def name() -> str:
        return "pi-jah"

    @override
    def _build_custom_models_json(self, access: Any, model_id: str) -> dict[str, Any] | None:
        mj = super()._build_custom_models_json(access, model_id)
        for prov in ((mj or {}).get("providers") or {}).values():
            if prov.get("baseUrl"):
                prov["baseUrl"] = with_session(prov["baseUrl"], trial_name(self.logs_dir))
        return mj


class JahMiniSwe(MiniSweAgent):
    """Harbor's mini-swe-agent with ``OPENAI_BASE_URL`` pointing at the proxy; the trial name becomes the proxy
    session."""

    @staticmethod
    @override
    def name() -> str:
        return "mini-swe-jah"

    @property
    @override
    def model_connection(self) -> Any:
        access = super().model_connection
        url = access.configured_base_url
        if not url:
            return access
        new = with_session(url, trial_name(self.logs_dir))
        return replace(access, configured_base_url=new,
                       env={k: (new if v == url else v) for k, v in access.env.items()})

    @property
    @override
    def extra_env(self) -> dict[str, str]:
        # Harbor applies the agent's --ae values to every exec last, so the session URL must be set here too
        env = super().extra_env
        for k in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
            if env.get(k):
                env[k] = with_session(env[k], trial_name(self.logs_dir))
        return env


class JahTerminus2(Terminus2):
    """Harbor's Terminus-2 (it calls the model from the Harbor process, so ``api_base`` is the proxy on localhost);
    the trial name becomes the proxy session."""

    @staticmethod
    @override
    def name() -> str:
        return "terminus-2-jah"

    def __init__(self, logs_dir: Any, *args: Any, api_base: str | None = None, **kwargs: Any) -> None:
        if api_base:
            api_base = with_session(api_base, trial_name(logs_dir))
        super().__init__(logs_dir, *args, api_base=api_base, **kwargs)


def dsh_patch(proxy_url: str, model: str) -> list[dict[str, Any]]:
    """The Cordis patch: one OpenAI-compatible route to the proxy, the model as the default, extras off."""
    return [
        {"id": "llm-pi-ai", "config": {"providers": {"jah": {
            "api": "openai-completions", "baseURL": proxy_url.rstrip("/"), "apiKeyEnv": "JAH_API_KEY",
            "compat": {"supportsStore": False},
            "models": [{"id": model, "contextWindow": 1000000, "maxTokens": 65536}]}}}},
        {"id": "agent-default-model", "config": {"provider": "jah", "model": model}},
        {"id": "session-title-llm", "disabled": True},      # one extra LLM call per session otherwise
        {"id": "tool-web", "disabled": True},               # needs a DeepSeek key; not part of the comparison
        {"id": "web-search-deepseek", "disabled": True},
        {"id": "web-fetch-http", "disabled": True},
    ]


class DshOptions(InstalledAgentOptions):
    proxy_url: str = Field(default="http://host.docker.internal:8787/v1",
                           description="The JevAgentHarness proxy as seen from the task container.")


class Dsh(BaseInstalledAgent):
    options_model = DshOptions
    options: DshOptions
    _OUTPUT = "dsh.jsonl"

    @staticmethod
    @override
    def name() -> str:
        return "dsh"

    @override
    def get_version_command(self) -> str | None:
        return ". ~/.nvm/nvm.sh; dsh --version"

    @override
    def parse_version(self, stdout: str) -> str:
        return stdout.strip().splitlines()[-1].strip()

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("curl",))
        version = self._version or DSH_VERSION
        await self.exec_as_agent(environment, command=(
            "set -euo pipefail; "
            f"{nvm_node_install_snippet()} && "
            f"npm install -g {DSH_PACKAGE}@{version} && dsh --version"))

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        model = (self.model_name or "jah/jah-auto").split("/", 1)[-1]
        url = with_session(self.options.proxy_url, trial_name(self.logs_dir))
        patch = json.dumps(dsh_patch(url, model))     # JSON is valid YAML
        b64 = base64.b64encode(patch.encode()).decode()
        out = (self.environment_logs_dir / self._OUTPUT).as_posix()
        await self.exec_as_agent(environment, command=(
            f"mkdir -p {REMOTE} && echo {b64} | base64 -d > {REMOTE}/jah.patch.yml"))
        env = {"JAH_API_KEY": DUMMY_KEY, "DSH_HOME": f"{REMOTE}/home", "DSH_TELEMETRY_MODE": "DISABLED",
               "DSH_PERMISSION_MODE": "danger-full-access", "NO_PROXY": "host.docker.internal,localhost,127.0.0.1"}
        await self.exec_as_agent(environment, env=env, command=(
            ". ~/.nvm/nvm.sh; "
            f"mkdir -p {shlex.quote(str(self.environment_logs_dir))} && "
            f"dsh --profile headless --patch {REMOTE}/jah.patch.yml {shlex.quote(instruction)} --json "
            f"2>{shlex.quote(out)}.stderr </dev/null | stdbuf -oL tee {shlex.quote(out)}"))

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        """Token totals from dsh's step-end events (the proxy's log is the source of truth for cost)."""
        path = self.logs_dir / self._OUTPUT
        if not path.exists():
            return
        n_in = n_out = n_cache = 0
        for line in path.read_text(errors="replace").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = ev.get("usage") if ev.get("type") == "status" and ev.get("phase") == "step_end" else None
            if isinstance(u, dict):
                n_in += int(u.get("inputTokens") or 0) + int(u.get("cacheReadTokens") or 0)
                n_out += int(u.get("outputTokens") or 0)
                n_cache += int(u.get("cacheReadTokens") or 0)
        context.n_input_tokens, context.n_output_tokens, context.n_cache_tokens = n_in, n_out, n_cache
