"""Terminal-Bench-Science helpers: survey a checkout, list task subsets, patch tasks for a TLS-intercepting sandbox.

    python adapters/pi/tbsci.py survey <clone> [--json out.json]   # Markdown table of every task (+ JSON)
    python adapters/pi/tbsci.py list <clone> hard_pool             # slugs of a subset: all | hard_pool | sandbox
    python adapters/pi/tbsci.py manifest <clone>                   # slugs in tasks/dataset.toml (the allowlist)
    python adapters/pi/tbsci.py patch-ca <task dir> <CA bundle>    # sandbox deviation (used by tbsci_tasks.sh)
    python adapters/pi/tbsci.py select runs/tbsci_cal --clone <clone> [--out adapters/pi/tbsci_hard.txt]
                                                                   # the hard subset from the calibration pass

Survey, per task (task.toml, Dockerfiles, build scripts): domain / field, agent / verifier / build timeouts,
CPUs / memory / storage / GPUs, the expert time estimate, verifier mode and network, base images, whether a build
runs ``apt-get`` on a Debian-family image, the hosts the image build downloads from, and whether the build pulls
data from Hugging Face (TB-science keeps large inputs in the ``harborframework/terminal-bench-science-lfs`` HF
dataset, not in git LFS).

Subsets (the committed lists are ``adapters/pi/tbsci_<name>.txt``, regenerated with ``list``):

* ``hard_pool``: static stage of our hard subset. Tasks a pi agent can attempt on a standard Docker host under a
  capped budget (no GPU, <= 4 CPUs / 16 GB / 20 GB disk, image build <= 30 min, verifier <= 30 min) whose expert
  estimate is 4-24 h: hard, but not so far out that every arm scores 0. The final hard subset is picked from this
  pool by a calibration pass (adapters/pi/README.md, "Terminal-Bench-Science").
* ``sandbox``: tasks whose images should build behind the dev sandbox's container egress (Ubuntu mirrors, PyPI,
  GitHub, npm and conda reachable; Debian mirrors, Hugging Face, the PyTorch index, Posit / CRAN, Julia and rustup
  blocked). A heuristic for smoke tests only.

``select`` (PREREGISTRATION A6.22 (d) 2) applies the registered selection rule to the Harbor job directories of ONE
calibration pass (``oracle``, ``nop`` and ``baseline`` under ``<calibration dir>``, run at the registered caps: 1 h
of agent time, 100 turns, one attempt) over the 37-task pool -- validity, floor, signal, fill, freeze (see
:func:`select_hard`) -- and writes ``tbsci_hard.txt``. It reads results only and never launches a run; a task whose
only trial of a stage is an infrastructure error is *pending* (retry it once), and nothing is frozen while any task
is pending.

``patch-ca`` copies a CA bundle next to every Dockerfile of a task (and into the environment/ and tests/ build
contexts) and inserts, after each FROM instruction, a COPY of it plus ENV variables that point pip, requests,
curl, git, node, conda and cargo at it. Nothing else changes. Results on patched tasks are a sandbox deviation.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Any

HARD_POOL_RULES: dict[str, float] = {
    "max_cpus": 4, "max_memory_mb": 16384, "max_storage_mb": 20480, "max_build_timeout_s": 1800,
    "max_verifier_timeout_s": 1800, "min_expert_hours": 4, "max_expert_hours": 24, "max_gpus": 0,
}
# Container egress of the dev sandbox (probed 2026-10-01). Unlisted hosts count as blocked.
SANDBOX_REACHABLE = ("pypi.org", "files.pythonhosted.org", "github.com", "githubusercontent.com",
                     "registry.npmjs.org", "conda.anaconda.org", "archive.ubuntu.com", "security.ubuntu.com")
UBUNTU_HINTS = ("ubuntu", "rocker/", "condaforge/", "miniforge", "mambaorg/")
DEBIAN_HINTS = ("debian", "bookworm", "bullseye", "trixie", "python:", "node:")
HF_HINTS = ("terminal-bench-science-lfs", "huggingface_hub", "huggingface.co", "hf_hub_download",
            "snapshot_download")
URL_HOST = re.compile(r"https?://([A-Za-z0-9.\-]+)")
SCRIPT_SUFFIXES = {".py", ".sh", ".R", ".r", ".jl", ".yaml", ".yml"}
CA_NAME, CA_TARGET = "harbor-sandbox-ca.crt", "/usr/local/share/harbor-sandbox-ca.crt"
CA_VARS = ("PIP_CERT", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE", "GIT_SSL_CAINFO",
           "NODE_EXTRA_CA_CERTS", "CONDA_SSL_VERIFY", "CARGO_HTTP_CAINFO")
CA_MARK = "# [sandbox patch: TLS-intercepting egress]"


# ------------------------------------------------------------------ Dockerfile parsing
def instructions(text: str) -> list[tuple[int, int, str]]:
    """Dockerfile instructions as (first line, last line, joined text): continuation lines and heredoc bodies
    belong to their instruction, so a Python ``from x import y`` inside a ``RUN`` is never taken for ``FROM``."""
    lines = text.splitlines()
    out: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s or s.startswith("#"):
            i += 1
            continue
        start, parts = i, [lines[i].rstrip()]
        while parts[-1].endswith("\\") and i + 1 < len(lines):
            i += 1
            parts.append(lines[i].rstrip())
        joined = " ".join(p[:-1] if p.endswith("\\") else p for p in parts)
        for tag in re.findall(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", joined):
            while i + 1 < len(lines) and lines[i + 1].strip() != tag:  # heredoc body up to its terminator
                i += 1
            i += 1
        out.append((start, i, joined))
        i += 1
    return out


def image_family(image: str) -> str:
    """'ubuntu' | 'debian' | 'other' for a FROM image reference."""
    ref = image.lower().split("@")[0]
    if any(h in ref for h in UBUNTU_HINTS):
        return "ubuntu"
    if any(h in ref for h in DEBIAN_HINTS):
        return "debian"
    return "other"


def dockerfile_stages(text: str) -> list[dict[str, Any]]:
    """One entry per FROM stage: base image (stage aliases resolved), family, runs apt-get, installs curl."""
    stages: list[dict[str, Any]] = []
    aliases: dict[str, dict[str, Any]] = {}
    for _, _, ins in instructions(text):
        m = re.match(r"FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?\s*$", ins, re.I)
        if m:
            parent = aliases.get(m.group(1).lower())
            st = {"image": parent["image"] if parent else m.group(1),
                  "family": parent["family"] if parent else image_family(m.group(1)),
                  "apt": False, "curl": bool(parent and parent["curl"])}
            stages.append(st)
            if m.group(2):
                aliases[m.group(2).lower()] = st
        elif stages and re.match(r"RUN\s", ins, re.I) and "apt-get" in ins:
            stages[-1]["apt"] = True
            if re.search(r"apt-get\s+install[^&;|]*\bcurl\b", ins):
                stages[-1]["curl"] = True
    return stages


def patch_dockerfile_text(text: str) -> tuple[str, int]:
    """Insert the CA block after every FROM instruction. Returns (new text, number of FROMs patched)."""
    lines = text.splitlines(keepends=True)
    after = [last for first, last, ins in instructions(text) if re.match(r"FROM\s", ins, re.I)]
    block = (f"{CA_MARK} trust the proxy CA bundle (not part of the task)\n"
             f"COPY {CA_NAME} {CA_TARGET}\n" + "ENV " + " ".join(f"{v}={CA_TARGET}" for v in CA_VARS) + "\n")
    for last in reversed(after):
        if not lines[last].endswith("\n"):
            lines[last] += "\n"
        lines.insert(last + 1, block)
    return "".join(lines), len(after)


def is_dockerfile(p: Path) -> bool:
    return p.is_file() and (p.name == "Dockerfile" or p.name.startswith("Dockerfile.") or p.name.endswith(".Dockerfile"))


def patch_ca(task: Path, ca_bundle: Path) -> int:
    """Sandbox deviation: make every image build of ``task`` trust ``ca_bundle``. Idempotent per copy."""
    contexts = {task / "environment", task / "tests"}
    n = 0
    for sub in ("environment", "tests"):
        for df in sorted(p for p in (task / sub).rglob("*") if is_dockerfile(p)):
            text = df.read_text()
            if CA_MARK in text:
                continue
            new, k = patch_dockerfile_text(text)
            df.write_text(new)
            n += k
            contexts.add(df.parent)
    for c in sorted(contexts):  # the bundle must be in whichever directory is the build context
        if c.is_dir():
            shutil.copyfile(ca_bundle, c / CA_NAME)
            ign = c / ".dockerignore"
            if ign.exists() and f"!{CA_NAME}" not in ign.read_text():
                ign.write_text(ign.read_text().rstrip("\n") + f"\n!{CA_NAME}\n")
    return n


# ------------------------------------------------------------------ survey
def manifest_slugs(clone: Path) -> list[str]:
    raw = tomllib.loads((clone / "tasks" / "dataset.toml").read_text())
    return [t["name"].split("/", 1)[1] for t in raw.get("tasks", [])]


def task_dirs(clone: Path) -> list[tuple[str, Path]]:
    main = sorted(p.parent for p in clone.glob("tasks/*/*/*/task.toml"))
    lite = sorted(p.parent for p in clone.glob("lite/*/task.toml"))
    return [("main", d) for d in main] + [("lite", d) for d in lite]


def survey_task(task: Path, set_name: str = "main", in_manifest: bool = True) -> dict[str, Any]:
    c = tomllib.loads((task / "task.toml").read_text())
    md, ag, vf, en = (c.get(k, {}) for k in ("metadata", "agent", "verifier", "environment"))
    files = [p for sub in ("environment", "tests") for p in sorted((task / sub).rglob("*")) if p.is_file()]
    dockerfiles = [p for p in files if is_dockerfile(p)]
    scripts = [p for p in files if p.suffix in SCRIPT_SUFFIXES or is_dockerfile(p)]
    stages = {sub: dockerfile_stages(p.read_text()) if (p := task / sub / "Dockerfile").exists() else []
              for sub in ("environment", "tests")}
    docker_text = "\n".join(p.read_text(errors="replace") for p in dockerfiles)
    if set_name == "main":
        domain, field = task.parent.parent.name, task.parent.name
    else:
        domain, field = (str(md.get(k, "")).replace(" ", "-") for k in ("domain", "field"))
    env_final = stages["environment"][-1] if stages["environment"] else {}
    row: dict[str, Any] = {
        "slug": task.name, "set": set_name, "in_manifest": in_manifest, "domain": domain, "field": field,
        "agent_timeout_s": ag.get("timeout_sec"), "verifier_timeout_s": vf.get("timeout_sec"),
        "build_timeout_s": en.get("build_timeout_sec"), "cpus": en.get("cpus", 1),
        "memory_mb": en.get("memory_mb", 2048), "storage_mb": en.get("storage_mb", 10240),
        "gpus": en.get("gpus", 0), "expert_hours": md.get("expert_time_estimate_hours"),
        "verifier_mode": vf.get("environment_mode"),
        "verifier_network": (vf.get("environment") or {}).get("network_mode"),
        "agent_network": en.get("network_mode"), "n_artifacts": len(c.get("artifacts", [])),
        "compose": (task / "environment" / "docker-compose.yaml").exists(),
        "env_image": env_final.get("image"), "env_family": env_final.get("family"),
        "test_image": stages["tests"][-1]["image"] if stages["tests"] else None,
        "debian_apt": any(s["apt"] and s["family"] == "debian" for ss in stages.values() for s in ss),
        "env_has_curl": bool(env_final.get("curl")),
        "build_hosts": sorted(set(URL_HOST.findall(docker_text))),
        "hf_data": any(h in p.read_text(errors="replace") for p in scripts for h in HF_HINTS),
    }
    row["hard_pool"] = in_hard_pool(row)
    row["sandbox_ok"] = sandbox_ok(row)
    return row


def in_hard_pool(r: dict[str, Any], rules: dict[str, float] = HARD_POOL_RULES) -> bool:
    eh = r.get("expert_hours")
    return bool(r["set"] == "main" and r["in_manifest"] and (r["gpus"] or 0) <= rules["max_gpus"]
                and r["cpus"] <= rules["max_cpus"] and r["memory_mb"] <= rules["max_memory_mb"]
                and r["storage_mb"] <= rules["max_storage_mb"]
                and (r["build_timeout_s"] or 0) <= rules["max_build_timeout_s"]
                and (r["verifier_timeout_s"] or 0) <= rules["max_verifier_timeout_s"]
                and eh is not None and rules["min_expert_hours"] <= float(eh) <= rules["max_expert_hours"])


def sandbox_ok(r: dict[str, Any]) -> bool:
    """Heuristic: the images build and pi installs behind the dev sandbox's container egress."""
    hosts_ok = all(any(h == ok or h.endswith("." + ok) for ok in SANDBOX_REACHABLE) for h in r["build_hosts"])
    pi_ok = r["env_family"] == "ubuntu" or r["env_has_curl"]  # pi's installer apt-gets curl if it is missing
    return bool(not r["debian_apt"] and not r["hf_data"] and hosts_ok and pi_ok and not r["compose"]
                and (r["gpus"] or 0) == 0)


def survey(clone: Path) -> list[dict[str, Any]]:
    allow = set(manifest_slugs(clone))
    return [survey_task(d, s, s == "main" and d.name in allow) for s, d in task_dirs(clone)]


def select(rows: list[dict[str, Any]], subset: str) -> list[str]:
    keys = {"all": None, "hard_pool": "hard_pool", "sandbox": "sandbox_ok"}
    if subset not in keys:
        raise ValueError(f"unknown subset {subset!r} ({', '.join(keys)})")
    key = keys[subset]
    return [r["slug"] for r in rows if r["set"] == "main" and r["in_manifest"] and (key is None or r[key])]


# ------------------------------------------------------------------ the hard subset (PREREGISTRATION A6.22 (d) 2)
REGISTERED_POOL_SHA256 = "f131ee5289003d95facdb055bb308a18a9d68280ddabb2aeb1c9fd122c9261fe"  # tbsci_hard_pool.txt
HARD_N, HARD_PER_DOMAIN = 20, 4
DOMAIN_ORDER = ("earth", "engineering", "life", "mathematical", "physical")
ORACLE_MAX_RUN_S = 1800.0          # half the registered 1 h agent cap (the cap is not raised)
CAL_JOBS = ("oracle", "nop", "baseline")


def short_domain(domain: str) -> str:
    """The task directory's domain as the rule names it: ``earth-sciences`` -> ``earth``."""
    return str(domain).split("-", 1)[0].lower()


def subset_slugs(path: Path) -> list[str]:
    return [ln.split("#")[0].strip() for ln in path.read_text().splitlines() if ln.split("#")[0].strip()]


def sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def decisive_trial(trials: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """The trial that decides one (job, task) under "an infrastructure failure is retried once": trials in start order;
    an infrastructure error is any Harbor exception except ``AgentTimeoutError`` (a timed-out trial is scored as it
    stands). Returns (the first non-errored trial among the first two, "") or (None, why): ``missing`` (no trial),
    ``retry needed`` (the only trial errored), ``infrastructure failed twice``."""
    ts = sorted(trials, key=lambda r: (r.get("started_at") is None, str(r.get("started_at")), r["trial"]))
    if not ts:
        return None, "missing"
    for r in ts[:2]:
        if not r.get("infra_error"):
            return r, ""
    if len(ts) == 1:
        return None, f"retry needed ({ts[0].get('error_class')})"
    return None, f"infrastructure failed twice ({ts[0].get('error_class')}; {ts[1].get('error_class')})"


def select_hard(cal_dir: Path, info: dict[str, dict[str, Any]], pool: list[str], n: int = HARD_N,
                per_domain: int = HARD_PER_DOMAIN) -> dict[str, Any]:
    """A6.22 (d) 2 on one calibration pass (Harbor jobs ``oracle``, ``nop``, ``baseline`` in ``cal_dir``):

    * validity -- drop a task whose decisive ``oracle`` trial does not reach reward 1 or whose oracle agent-run time
      exceeds 1,800 s, or whose build / verifier fails twice (:func:`decisive_trial`; also for ``nop``);
    * floor -- ``nop``'s tests passed with no agent output, f;
    * signal -- one ``baseline`` attempt: keep a task iff it has reward 1 or passes more tests than f (an errored
      attempt is retried once, then the task is dropped; a timed-out attempt is scored as it stands);
    * fill -- ``n`` = 20 tasks, ``per_domain`` = 4 slots per domain (earth, engineering, life, mathematical, physical:
      the task directory's domain); within a domain by ``expert_time_estimate_hours`` ascending, ties by slug; slots a
      domain cannot fill go one at a time, round-robin in that domain order, to the domains with eligible tasks left,
      until ``n`` are chosen or none is left.

    ``info``: slug -> survey row (``domain``, ``expert_hours``). Returns per-task statuses (``eligible`` | ``dropped``
    | ``pending``), the chosen slugs in domain order, and the counts."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from summarize_harbor import summarize  # noqa: PLC0415  (adapters/pi; the Harbor result reader)

    jobs = [cal_dir / j for j in CAL_JOBS if (cal_dir / j).is_dir()]
    by: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in summarize(jobs):
        by.setdefault((r["job"], r["task"]), []).append(r)
    tasks: dict[str, dict[str, Any]] = {}
    for slug in pool:
        row = info.get(slug)
        if row is None:
            raise ValueError(f"pool task {slug!r} is not in the survey (wrong clone / commit?)")
        dom = short_domain(row["domain"])
        if dom not in DOMAIN_ORDER:
            raise ValueError(f"{slug}: domain {row['domain']!r} is not one of {DOMAIN_ORDER}")
        st: dict[str, Any] = {"slug": slug, "domain": dom, "expert_hours": row.get("expert_hours")}
        tasks[slug] = st

        def stop(stage: str, why: str, pending: bool = False) -> None:
            st.update(status="pending" if pending else "dropped", stage=stage, reason=why)  # noqa: B023

        o, why = decisive_trial(by.get(("oracle", slug), []))
        if o is None:
            stop("validity", f"oracle: {why}", pending=not why.startswith("infrastructure failed"))
            continue
        st["oracle_run_s"] = o.get("agent_run_s")
        if o.get("reward") is None or float(o["reward"]) < 1.0:
            stop("validity", f"oracle reward {o.get('reward')} < 1")
            continue
        if o.get("agent_run_s") is None or float(o["agent_run_s"]) > ORACLE_MAX_RUN_S:
            stop("validity", f"oracle agent-run time {o.get('agent_run_s')} s > {ORACLE_MAX_RUN_S:.0f} s")
            continue
        f, why = decisive_trial(by.get(("nop", slug), []))
        if f is None:
            stop("floor", f"nop: {why}", pending=not why.startswith("infrastructure failed"))
            continue
        st["floor"] = f.get("tests_passed")
        b, why = decisive_trial(by.get(("baseline", slug), []))
        if b is None:
            stop("signal", f"baseline: {why}", pending=not why.startswith("infrastructure failed"))
            continue
        st.update(baseline_reward=b.get("reward"), baseline_tests_passed=b.get("tests_passed"),
                  baseline_timed_out=bool(b.get("timed_out")))
        solved = b.get("reward") is not None and float(b["reward"]) >= 1.0
        above = (b.get("tests_passed") is not None and st["floor"] is not None
                 and int(b["tests_passed"]) > int(st["floor"]))
        if not (solved or above):
            stop("signal", f"baseline reward {b.get('reward')}, tests passed {b.get('tests_passed')} <= floor "
                           f"{st['floor']}" + (" (no test counts)" if st["floor"] is None
                                               or b.get("tests_passed") is None else ""))
            continue
        st.update(status="eligible", reason="reward 1" if solved else "tests passed above the nop floor")
    # fill
    elig = {d: sorted((t for t in tasks.values() if t["status"] == "eligible" and t["domain"] == d),
                      key=lambda t: (float("inf") if t["expert_hours"] is None else float(t["expert_hours"]),
                                     t["slug"])) for d in DOMAIN_ORDER}
    chosen = {d: elig[d][:per_domain] for d in DOMAIN_ORDER}
    rest = {d: elig[d][per_domain:] for d in DOMAIN_ORDER}
    total = sum(len(v) for v in chosen.values())
    while total < n and any(rest.values()):
        for d in DOMAIN_ORDER:
            if total < n and rest[d]:
                chosen[d].append(rest[d].pop(0))
                total += 1
    if total > n:   # fewer than per_domain x 5 slots: keep the first n in domain order (never with N = 20)
        keep = [t["slug"] for d in DOMAIN_ORDER for t in chosen[d]][:n]
        chosen = {d: [t for t in chosen[d] if t["slug"] in keep] for d in DOMAIN_ORDER}
    for d in DOMAIN_ORDER:
        for t in chosen[d]:
            t["selected"] = True
    counts = {s: sum(1 for t in tasks.values() if t["status"] == s) for s in ("eligible", "dropped", "pending")}
    return {"rule": "PREREGISTRATION A6.22 (d) 2", "calibration_dir": str(cal_dir), "n_target": n,
            "per_domain": per_domain, "selected": [t["slug"] for d in DOMAIN_ORDER for t in chosen[d]],
            "by_domain": {d: [t["slug"] for t in chosen[d]] for d in DOMAIN_ORDER}, "counts": counts,
            "pending": sorted(t["slug"] for t in tasks.values() if t["status"] == "pending"),
            "jobs_found": [j.name for j in jobs], "tasks": tasks}


def freeze_text(sel: dict[str, Any], pool: Path, run_id: str, commit: str, tbsci_commit: str) -> str:
    """``tbsci_hard.txt``: a header (pool hash, calibration run id, commit) and one slug per line, in domain order."""
    k = len(sel["selected"])
    head = [f"# Terminal-Bench-Science hard subset ({sel['rule']}), written by `adapters/pi/tbsci.py select`; "
            "do not edit by hand.",
            f"# pool: {pool.name} sha256 {sha256_of(pool)} ({pool.stat().st_size} bytes)",
            f"# calibration run: {run_id}",
            f"# commit: {commit}; terminal-bench-science: {tbsci_commit}",
            f"# N = {k} of {sel['n_target']}" + ("" if k == sel["n_target"] else " (a smaller subset, reported as such)")
            + "; per domain: " + ", ".join(f"{d} {len(v)}" for d, v in sel["by_domain"].items())]
    return "\n".join(head + sel["selected"]) + "\n"


def _git_head(path: Path) -> str:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def markdown(rows: list[dict[str, Any]]) -> str:
    cols = ["slug", "domain", "expert_hours", "cpus", "memory_mb", "verifier_timeout_s", "build_timeout_s",
            "env_family", "debian_apt", "hf_data", "hard_pool", "sandbox_ok"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: (r["domain"], r["slug"])):
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey", help="Markdown table of every task")
    s.add_argument("clone", type=Path)
    s.add_argument("--json", type=Path, default=None, help="also write the per-task rows as JSON")
    ls = sub.add_parser("list", help="slugs of a subset")
    ls.add_argument("clone", type=Path)
    ls.add_argument("subset", choices=["all", "hard_pool", "sandbox"])
    mf = sub.add_parser("manifest", help="slugs listed in tasks/dataset.toml")
    mf.add_argument("clone", type=Path)
    p = sub.add_parser("patch-ca", help="sandbox deviation: trust a CA bundle in every image build of a task")
    p.add_argument("task", type=Path)
    p.add_argument("ca_bundle", type=Path)
    se = sub.add_parser("select", help="A6.22 (d) 2: the hard subset from one calibration pass (reads results only)")
    se.add_argument("cal_dir", type=Path, help="the calibration pass's OUT (Harbor jobs oracle, nop, baseline)")
    src = se.add_mutually_exclusive_group(required=True)
    src.add_argument("--clone", type=Path, help="terminal-bench-science checkout (domains, expert estimates)")
    src.add_argument("--survey", type=Path, help="per-task rows written by `survey --json`")
    se.add_argument("--pool", type=Path, default=Path(__file__).resolve().parent / "tbsci_hard_pool.txt")
    se.add_argument("--allow-other-pool", action="store_true", help="TESTS ONLY: a pool other than the registered one")
    se.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "tbsci_hard.txt")
    se.add_argument("--force", action="store_true", help="overwrite an existing --out (a frozen list is never redone)")
    se.add_argument("--run-id", default=None, help="calibration run id for the header (default: the directory name)")
    se.add_argument("--json", type=Path, default=None, help="also write the per-task selection report as JSON")
    a = ap.parse_args(argv)
    if a.cmd == "select":
        raise SystemExit(_cmd_select(a))
    if a.cmd == "patch-ca":
        print(f"patched {patch_ca(a.task, a.ca_bundle)} FROM instruction(s)")
        return
    if a.cmd == "manifest":
        print("\n".join(manifest_slugs(a.clone)))
        return
    rows = survey(a.clone)
    if a.cmd == "list":
        print("\n".join(select(rows, a.subset)))
        return
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))
    print(markdown(rows))
    print(f"\n{len(rows)} tasks ({sum(r['set'] == 'lite' for r in rows)} lite); "
          f"hard_pool {sum(r['hard_pool'] for r in rows)}; sandbox_ok {sum(r['sandbox_ok'] for r in rows)}",
          file=sys.stderr)


def _cmd_select(a: argparse.Namespace) -> int:
    pool_sha = sha256_of(a.pool)
    if pool_sha != REGISTERED_POOL_SHA256 and not a.allow_other_pool:
        print(f"!! {a.pool}: sha256 {pool_sha} is not the registered pool ({REGISTERED_POOL_SHA256})", file=sys.stderr)
        return 2
    rows = survey(a.clone) if a.clone else json.loads(a.survey.read_text())
    info = {r["slug"]: r for r in rows if r.get("set", "main") == "main"}
    sel = select_hard(a.cal_dir, info, subset_slugs(a.pool))
    if a.json:
        a.json.write_text(json.dumps(sel, indent=1))
    for t in sel["tasks"].values():
        print(f"{t['slug']:40s} {t['domain']:13s} {str(t['expert_hours']):>6s} h  {t['status']:9s}"
              f"{' SELECTED' if t.get('selected') else ''}  {t.get('stage', '')} {t.get('reason', '')}",
              file=sys.stderr)
    if sel["pending"]:
        print(f"!! {len(sel['pending'])} task(s) pending -- retry the errored calibration trial once, in the same "
              f"calibration directory, then select again: {', '.join(sel['pending'])}", file=sys.stderr)
        return 1
    if a.out.exists() and not a.force:
        print(f"!! {a.out} exists: the hard subset is frozen once (pass --force only to redo an unused freeze)",
              file=sys.stderr)
        return 1
    text = freeze_text(sel, a.pool, a.run_id or a.cal_dir.name, _git_head(Path(__file__).resolve().parent),
                       _git_head(a.clone) if a.clone else "unknown")
    a.out.write_text(text)
    print(f"{a.out}: {len(sel['selected'])} task(s); sha256 {sha256_of(a.out)}, {a.out.stat().st_size} bytes -- record "
          "both in the Deviations section before any evaluation trial (A6.22 (d) 2)")
    return 0


if __name__ == "__main__":
    main()
