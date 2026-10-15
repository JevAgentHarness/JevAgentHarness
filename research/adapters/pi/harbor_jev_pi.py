"""Harbor agent: pi with the Jev System-One extensions in this directory.

Harbor's built-in ``pi`` agent only loads its own extensions (MCP adapter, max-turns). This subclass
uploads the selected ``jev-*.ts`` files (and ``jev-common.ts``) into the task container and adds one
``--extension`` flag per file. Everything else -- install, model credentials, ATIF trajectory
conversion -- is Harbor's ``Pi`` agent unchanged.

Run from the repository root, with this directory importable::

    PYTHONPATH=adapters/pi harbor run -d terminal-bench@2.0 \\
        -a harbor_jev_pi:JevPi -m google/gemini-3.5-flash \\
        --ak version=0.99.2 --ak jev_extensions=tool-gate,stop-verify \\
        --ae TYPESAFE_API_KEY="$TYPESAFE_API_KEY" --ae JEV_TAU=0.5

Agent kwargs (``--ak``): everything Harbor's ``pi`` accepts (``version``, ``thinking``,
``max_turns``, ...) plus ``jev_extensions``, a comma-separated subset of ``EXTENSIONS``, and ``llm_proxy`` (E29): the
base URL of the host-side routing proxy for one arm (``jevharness.adapters.route_proxy``, e.g.
``http://host.docker.internal:8091/e29/gated``). With it, pi's ``models.json`` points the built-in ``google`` provider at
``<llm_proxy>/<trial name>/v1beta`` (its models, pricing metadata and thought-signature handling stay pi's own), so every
request goes through the proxy, which holds the real key; Harbor then needs only a dummy ``GEMINI_API_KEY`` and no key
enters the task container.
Pass ``JEV_*`` settings and decision-model credentials with ``--ae``. Decisions are logged to
``/logs/agent/jev-decisions.jsonl`` (downloaded with the trial's agent logs) unless ``JEV_LOG`` is set.
"""

from __future__ import annotations

import base64
import json
import shlex
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, override

from packaging.version import InvalidVersion, Version
from pydantic import Field

from harbor.agents.installed.pi import Pi, PiOptions
from harbor.agents.model_connection import ResolvedModelConnection, resolve_model_connection
from harbor.environments.base import BaseEnvironment

HERE = Path(__file__).resolve().parent
REMOTE_DIR = PurePosixPath("/tmp/harbor-pi-jev")
SHARED = "jev-common.ts"
EXTENSIONS: dict[str, str] = {
    "tool-gate": "jev-tool-gate.ts",
    "active-tools": "jev-active-tools.ts",
    "stop-verify": "jev-stop-verify.ts",
    "router": "jev-router.ts",
    "memory": "jev-memory.ts",
    "reflect": "jev-reflect.ts",
    "compact": "jev-compact.ts",
}
# Classifier models, virtual models, and actionable turn_end/agent_before_settle boundaries.
MIN_PI_VERSION = Version("0.99.0")
# jev-router.ts's virtual model lives under this provider (JEV_ROUTER_PROVIDER); its routed models are Google's
ROUTER_PROVIDER = "s1"
ROUTED_PROVIDER = "google"


class JevPiOptions(PiOptions):
    jev_extensions: str = Field(
        default="",
        description="Comma-separated Jev extensions to load: " + ", ".join(EXTENSIONS) + ". Empty runs plain pi.",
    )
    llm_proxy: str = Field(
        default="",
        description="E29: base URL of the host-side routing proxy for this arm; empty = pi talks to the provider.",
    )


def episode_name(logs_dir: Path | str) -> str:
    """The trial's name (the parent of the agent's logs dir), safe for a URL path segment."""
    name = Path(logs_dir).parent.name or "episode"
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name)


def proxy_models_json(llm_proxy: str, episode: str, provider: str = ROUTED_PROVIDER) -> dict[str, Any]:
    """pi's ``models.json``: only the built-in provider's ``baseUrl`` changes (pi keeps its own model list)."""
    return {"providers": {provider: {"baseUrl": f"{llm_proxy.rstrip('/')}/{episode}/v1beta"}}}


class JevPi(Pi):
    options_model = JevPiOptions
    options: JevPiOptions

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        names = [name.strip() for name in self.options.jev_extensions.split(",") if name.strip()]
        unknown = sorted(set(names) - set(EXTENSIONS))
        if unknown:
            raise ValueError(f"Unknown jev_extensions {unknown}; choose from {sorted(EXTENSIONS)}")
        version = self.options.version
        if names and version not in (None, "latest"):
            try:
                too_old = Version(version) < MIN_PI_VERSION
            except InvalidVersion:
                too_old = False
            if too_old:
                raise ValueError(f"Jev extensions need pi >= {MIN_PI_VERSION}; got {version}")
        self._jev_files = [EXTENSIONS[name] for name in dict.fromkeys(names)]
        # Trial reads extra_env when the agent runs, so this reaches the pi process.
        self._extra_env.setdefault("JEV_LOG", str(self.environment_logs_dir / "jev-decisions.jsonl"))

    @staticmethod
    @override
    def name() -> str:
        return "pi-jev"

    @property
    @override
    def model_connection(self) -> ResolvedModelConnection:
        """Harbor resolves no credentials for the virtual ``s1`` provider of ``jev-router.ts`` (``-m s1/router``):
        give it the routed provider's (Google's) key the way Harbor resolves it for ``google/...`` models, so the key
        reaches pi through Harbor's own environment resolution and never through ``--ae`` or a job file (E25)."""
        conn = super().model_connection
        if (self.model_name or "").split("/", 1)[0] != ROUTER_PROVIDER or conn.env:
            return conn
        routed = resolve_model_connection(f"{ROUTED_PROVIDER}/router", self.MODEL_CONNECTION, self._resolve_env)
        return replace(conn, provider=ROUTER_PROVIDER, env=dict(routed.env))

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        proxy = (self.options.llm_proxy or "").strip()
        if proxy:
            content = json.dumps(proxy_models_json(proxy, episode_name(self.logs_dir)), indent=1)
            b64 = base64.b64encode(content.encode()).decode()
            await self.exec_as_agent(
                environment,
                command=('mkdir -p "$HOME/.pi/agent" && chmod 700 "$HOME/.pi/agent" && '
                         f'echo {b64} | base64 -d > "$HOME/.pi/agent/models.json" && '
                         'chmod 600 "$HOME/.pi/agent/models.json"'),
            )
        if not self._jev_files:
            return
        await self.exec_as_agent(environment, command=f"mkdir -p {shlex.quote(REMOTE_DIR.as_posix())}")
        for filename in (SHARED, *self._jev_files):
            await self._upload_agent_owned_file(environment, HERE / filename, (REMOTE_DIR / filename).as_posix())

    @override
    def build_cli_flags(self) -> str:
        extensions = " ".join(
            f"--extension {shlex.quote((REMOTE_DIR / filename).as_posix())}" for filename in self._jev_files
        )
        return " ".join(part for part in (super().build_cli_flags(), extensions) if part)
