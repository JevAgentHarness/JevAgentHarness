"""``jah``: run the proxy, write a starter config, print harness setup snippets, summarize logs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from importlib import resources
from pathlib import Path

from . import __version__
from .config import DEFAULT_CONFIG, load_config

DEFAULT_PATH = "~/.jah/config.yaml"
HARNESSES = ("codex", "pi", "dsh", "mini-swe-agent", "openai-sdk")


def _snippet(harness: str, base: str) -> str:
    name = harness.replace("-", "_")
    text = resources.files("jevagentharness").joinpath(f"snippets/{name}.md").read_text()
    return text.replace("{BASE}", base)


def cmd_init(a: argparse.Namespace) -> int:
    p = Path(a.path).expanduser()
    if p.exists() and not a.force:
        print(f"{p} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(DEFAULT_CONFIG)
    print(f"wrote {p}")
    return 0


def _config_path(a: argparse.Namespace) -> str | None:
    """-c if given, else the file `jah init` writes when it exists, else the built-in starter config."""
    if a.config:
        return a.config
    p = Path(DEFAULT_PATH).expanduser()
    return str(p) if p.exists() else None


def cmd_serve(a: argparse.Namespace) -> int:
    from .server import run
    cfg = load_config(_config_path(a), a.set)
    if a.port:
        cfg.port = a.port
    if a.host:
        cfg.host = a.host
    if a.log_dir:
        cfg.log_dir = a.log_dir
    missing = sorted({u.key_env for u in cfg.upstreams.values()
                      if u.key_env and not os.environ.get(u.key_env)
                      and any(t.upstream == u.name for t in cfg.tiers.values())})
    if missing:
        print(f"refusing to start: set {', '.join(missing)} in the environment", file=sys.stderr)
        return 2

    def banner(proxy) -> None:  # type: ignore[no-untyped-def]
        t = cfg.tiers
        print(f"jah {__version__} on http://{cfg.host}:{cfg.port}/v1  "
              f"strong={t[cfg.router.strong].model}  fast={t[cfg.router.fast].model}  "
              f"decision={proxy.backend.describe()}  tau={cfg.router.tau}  logs={proxy.log_dir}", flush=True)
    run(cfg, banner)
    return 0


def cmd_setup(a: argparse.Namespace) -> int:
    print(_snippet(a.harness, a.base.rstrip("/")))
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from .report import read_steps, summarize, table
    s = summarize(read_steps(a.log), by=a.by)
    print(json.dumps(s, indent=1) if a.json else table(s))
    return 0


def cmd_ask(a: argparse.Namespace) -> int:
    """One request through a running proxy (a smoke test)."""
    import urllib.request
    body = {"model": a.model, "messages": [{"role": "user", "content": a.prompt}]}
    req = urllib.request.Request(a.base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer jah"})
    with urllib.request.urlopen(req, timeout=300) as r:  # noqa: S310 - local proxy
        d = json.load(r)
    print(d["choices"][0]["message"].get("content"))
    print(json.dumps(d.get("usage")), file=sys.stderr)
    return 0


def cmd_decide(a: argparse.Namespace) -> int:
    """Ask the configured decision model about a state (a quick check of the key and the backend)."""
    from .decision import make_backend
    from .router import QUESTION
    cfg = load_config(_config_path(a), a.set)
    be = make_backend(cfg.router.decision)

    async def go() -> None:
        try:
            d = await be.decide(Path(a.state).read_text() if a.state else "TASK: fix the failing test\n\n"
                                "[1] ACTION: bash(pytest -q)\n    RESULT: 1 failed, 12 passed", QUESTION)
            print(json.dumps({"p": d.p, "model": d.model, "cost_usd": d.cost_usd, "latency_s": round(d.latency_s, 3)}))
        finally:
            await be.close()
    asyncio.run(go())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jah", description="JevAgentHarness: decision-model routing for agent harnesses")
    ap.add_argument("--version", action="version", version=f"jah {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="write a starter config")
    p.add_argument("path", nargs="?", default=DEFAULT_PATH)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("serve", help="run the proxy")
    p.add_argument("--config", "-c", default=None,
                   help="YAML config (default: ~/.jah/config.yaml if it exists, else the built-in config)")
    p.add_argument("--set", action="append", default=[], help="override a value: dotted.key=YAML")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--log-dir")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("setup", help="print the setup for a harness")
    p.add_argument("harness", choices=HARNESSES)
    p.add_argument("--base", default="http://127.0.0.1:8787/v1")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("report", help="summarize a proxy log")
    p.add_argument("log", nargs="?", default="~/.jah/logs")
    p.add_argument("--by", default="model_name")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("ask", help="send one chat request through a running proxy")
    p.add_argument("prompt")
    p.add_argument("--model", default="jah-auto")
    p.add_argument("--base", default="http://127.0.0.1:8787/v1")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("decide", help="ask the configured decision model about a state")
    p.add_argument("--config", "-c", default=None)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--state", help="a text file with the state (default: a built-in example)")
    p.set_defaults(fn=cmd_decide)

    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
