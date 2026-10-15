"""On-host checks of the Miles image before any training step (``scripts/aws/gpu_round6_stages.sh``).

  python -m jevharness.training.miles_port.smoke env
      versions (torch / CUDA / transformers / sglang / miles), GPU names and compute capability, whether the fla and
      causal-conv1d kernels import and run on this GPU (Qwen3.5-family linear attention; without them the packed path
      drops per-document resets, MILES_PORT.md §4 S3).
  python -m jevharness.training.miles_port.smoke engine --model DIR --dev data/train_v1/dev.jsonl.gz --n 32 --out X.json
      starts an SGLang server on GPU 0, scores N v1-dev decisions through ``SGLangClient`` (the rollout path), then the
      HF model in-process through ``decision_model.option_logits`` (the training / serving readout) on the same token
      ids; writes max / mean |delta| of the restricted log-softmax, argmax agreement and both accuracies (S2).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def env_report() -> dict[str, Any]:
    rep: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["cuda"] = torch.version.cuda
        rep["gpus"] = [{"name": torch.cuda.get_device_name(i), "cc": list(torch.cuda.get_device_capability(i)),
                        "mem_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
                       for i in range(torch.cuda.device_count())]
    except Exception as e:  # noqa: BLE001
        rep["torch_error"] = repr(e)
    for mod in ("transformers", "sglang", "miles", "flash_attn", "fla", "causal_conv1d", "ray"):
        try:
            m = __import__(mod)
            rep[mod] = getattr(m, "__version__", "present")
        except Exception as e:  # noqa: BLE001
            rep[mod] = f"missing: {type(e).__name__}: {str(e)[:200]}"
    rep["kernels"] = kernel_check()
    return rep


def kernel_check() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"skipped": "no CUDA"}
        dev = "cuda"
        try:
            from causal_conv1d import causal_conv1d_fn

            x = torch.randn(1, 64, 32, device=dev, dtype=torch.bfloat16)
            w = torch.randn(64, 4, device=dev, dtype=torch.bfloat16)
            y = causal_conv1d_fn(x, w, None, activation="silu")
            out["causal_conv1d"] = f"ok {tuple(y.shape)}"
        except Exception as e:  # noqa: BLE001
            out["causal_conv1d"] = f"FAIL {type(e).__name__}: {str(e)[:300]}"
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            B, T, H, K = 1, 64, 2, 64
            q = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16)
            k = torch.nn.functional.normalize(torch.randn(B, T, H, K, device=dev, dtype=torch.float32), dim=-1).bfloat16()
            v = torch.randn(B, T, H, K, device=dev, dtype=torch.bfloat16)
            g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=dev, dtype=torch.float32))
            beta = torch.rand(B, T, H, device=dev, dtype=torch.bfloat16)
            o, _ = chunk_gated_delta_rule(q, k, v, g, beta)
            out["fla_gated_delta_rule"] = f"ok {tuple(o.shape)}"
        except Exception as e:  # noqa: BLE001
            out["fla_gated_delta_rule"] = f"FAIL {type(e).__name__}: {str(e)[:300]}"
        try:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

            out["transformers_qwen3_5_fast_path"] = bool(getattr(mq, "is_fast_path_available", False))
        except Exception as e:  # noqa: BLE001
            out["transformers_qwen3_5_fast_path"] = f"n/a {type(e).__name__}"
    except Exception as e:  # noqa: BLE001
        out["error"] = repr(e)
    return out


def _wait(url: str, proc: subprocess.Popen, timeout: float = 900.0) -> None:
    import httpx

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"sglang server exited with {proc.returncode}")
        try:
            if httpx.get(url + "/health", timeout=5).status_code == 200:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    raise TimeoutError("sglang server did not come up")


def engine_check(model: str, dev: str, n: int, out: str, port: int = 30011, mem: float = 0.6,
                 attention_backend: str = "flashinfer") -> dict[str, Any]:
    import torch

    from ..data import permuted, read_jsonl
    from ..decision_model import DecisionCollator, option_logits
    from .rollout import load_tokenizer
    from .sglang_client import SGLangClient

    tok = load_tokenizer(model)
    col = DecisionCollator(tok, max_len=4096)
    rows = [r for r in read_jsonl(dev) if 2 <= len(r["options"]) <= 52][:n]
    prompts, tgts = [], []
    for r in rows:
        p, t, _ = permuted(r, None)
        prompts.append(p)
        tgts.append(t)
    ids = [col.encode(p) for p in prompts]
    codes = [col.codes[:len(t)] for t in tgts]
    rep: dict[str, Any] = {"model": model, "n": len(rows), "codes_single_token": True,
                           "prompt_tokens_mean": sum(map(len, ids)) / max(1, len(ids))}
    env = {**os.environ, "SGLANG_RETURN_ORIGINAL_LOGPROB": "1", "CUDA_VISIBLE_DEVICES": "0"}
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--host", "127.0.0.1", "--port",
           str(port), "--tp", "1", "--mem-fraction-static", str(mem), "--attention-backend", attention_backend,
           "--dtype", "bfloat16", "--log-level", "warning"]
    rep["sglang_cmd"] = " ".join(cmd)
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    try:
        url = f"http://127.0.0.1:{port}"
        _wait(url, proc)
        cl = SGLangClient(url)
        t0 = time.time()
        sg = []
        for s in range(0, len(ids), 8):
            sg += cl.option_logprobs(ids[s:s + 8], codes[s:s + 8])
        rep["sglang_s"] = round(time.time() - t0, 2)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(5)
    from ..train_decision import load_causal_lm

    def hf_scores(dtype: Any, device: str) -> list[list[float]]:
        net = load_causal_lm(model, dtype).to(device).eval()
        out_: list[list[float]] = []
        with torch.no_grad():
            for p, t in zip(prompts, tgts):
                enc = col([{"prompt": p, "target": t}])
                z = option_logits(net, enc["input_ids"].to(device), enc["attention_mask"].to(device),
                                  enc["code_ids"].to(device))
                out_.append(z[0, :len(t)].float().cpu().tolist())
        del net
        torch.cuda.empty_cache()
        return out_

    t0 = time.time()
    hf32 = hf_scores(torch.float32, "cuda:0")      # the reference
    hf16 = hf_scores(torch.bfloat16, "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")   # trainer precision
    rep["hf_s"] = round(time.time() - t0, 2)

    def compare_scores(xs: list[list[float]], ys: list[list[float]]) -> dict[str, Any]:
        dmax, dmain, tvs, agree = [], [], [], 0
        for a, b in zip(xs, ys):
            la = torch.log_softmax(torch.tensor(a, dtype=torch.float64), -1)
            lb = torch.log_softmax(torch.tensor(b, dtype=torch.float64), -1)
            fin = torch.isfinite(la) & torch.isfinite(lb)
            dmax.append(float((la - lb)[fin].abs().max()) if fin.any() else float("inf"))
            main = fin & ((la.exp() >= 0.01) | (lb.exp() >= 0.01))   # options that carry probability
            dmain.append(float((la - lb)[main].abs().max()) if main.any() else 0.0)
            tvs.append(0.5 * float((la.exp() - lb.exp()).abs().sum()))
            agree += int(int(la.argmax()) == int(lb.argmax()))
        m = max(1, len(tvs))
        return {"max_abs_dlogp_all": max(dmax), "max_abs_dlogp_p>=0.01": max(dmain),
                "mean_abs_dlogp_p>=0.01": sum(dmain) / m, "tv_mean": sum(tvs) / m, "tv_max": max(tvs),
                "argmax_agree": agree / m}

    def acc(xs: list[list[float]]) -> float:
        return sum(int(max(range(len(a)), key=lambda i: a[i]) == max(range(len(t)), key=lambda i: t[i]))
                   for a, t in zip(xs, tgts)) / max(1, len(xs))

    rep["sglang_vs_hf_fp32"] = compare_scores(sg, hf32)
    rep["hf_bf16_vs_hf_fp32"] = compare_scores(hf16, hf32)
    rep["sglang_vs_hf_bf16"] = compare_scores(sg, hf16)
    rep.update({"acc_sglang": acc(sg), "acc_hf_fp32": acc(hf32), "acc_hf_bf16": acc(hf16)})
    s = rep["sglang_vs_hf_bf16"]
    # the manifest's S2 as first written: max |delta log p| over every option, SGLang vs the HF forward (bf16), <= 0.1
    rep["max_abs_dlogp"] = s["max_abs_dlogp_all"]
    rep["pass_s2_as_written"] = s["max_abs_dlogp_all"] <= 0.1
    # S2 as amended (manifest §9, 2026-10-03): distributions over the options, SGLang vs the fp32 HF reference
    r = rep["sglang_vs_hf_fp32"]
    rep["pass_s2"] = r["argmax_agree"] >= 0.95 and r["tv_mean"] <= 0.02 and r["tv_max"] <= 0.10
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


def replay_check(model: str, config: str, out: str, n_tasks: int = 8, group: int = 2, port: int = 30012,
                 mean_tol: float = 0.02, max_tol: float = 0.2) -> dict[str, Any]:
    """A7.10 (f) on the host: sampled GRPO events from the SGLang engine (the behaviour policy as Miles serves it; the
    simulated System-2, so no API cost; the GRPO stage's arms and samplers, training seeds of its pool) are replayed
    through the trainer's model class (``export.model_class``, bf16) on the identical token ids; the stored behaviour
    log-probability of every trainable event must be reproduced within ``mean_tol`` (mean |delta|) and ``max_tol``."""
    import torch
    import yaml

    from ...llm import make_llm
    from ..grpo_online import event_logp
    from .config import StageConfig
    from .export import model_class
    from .rollout import Harness
    from .sglang_client import SGLangClient

    raw = yaml.safe_load(Path(config).read_text())
    raw.update(base=model, init=model, out=str(Path(out).parent / "replay_check_run"))
    raw["rollout"].update(tasks_per_iter=n_tasks, group_size=group, iterations=1, workers=8)
    cfg = StageConfig.model_validate(raw).check()
    m = cfg.miles
    env = {**os.environ, "SGLANG_RETURN_ORIGINAL_LOGPROB": "1", "CUDA_VISIBLE_DEVICES": "0"}
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--host", "127.0.0.1", "--port",
           str(port), "--tp", "1", "--mem-fraction-static", str(m.sglang_mem_fraction), "--attention-backend",
           m.sglang_attention_backend, "--dtype", "bfloat16", "--log-level", "warning"]
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    try:
        url = f"http://127.0.0.1:{port}"
        _wait(url, proc)
        h = Harness(cfg, url, client=SGLangClient(url),
                    executor=make_llm({"provider": "simulated", "model": "scripted", "error_rate": 0.1}))
        episodes, row = h.run(0)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(5)
    recs = [r for e in episodes for r in e.records if r.train]
    net = model_class(model).from_pretrained(model, dtype=torch.bfloat16).to("cuda:0").eval()
    T = max(cfg.rollout.temperature, 1e-6) if cfg.rollout else 1.0
    gaps = []
    with torch.no_grad():
        for r in recs:
            ids = h.collator.encode(r.prompt)
            logits = net(input_ids=torch.tensor([ids], device="cuda:0")).logits[0, -1].float()
            z = logits.index_select(0, torch.tensor(h.collator.codes[:r.n], device="cuda:0"))
            lp = torch.log_softmax(z.double() / T, -1)
            gaps.append(abs(float(event_logp(lp, r.event, r.ev)) - r.old_logp))
    rep = {"model": model, "episodes": len(episodes), "events": len(recs), "by_ev": {},
           "mean_abs_dlogp": sum(gaps) / len(gaps) if gaps else None, "max_abs_dlogp": max(gaps) if gaps else None,
           "mean_tol": mean_tol, "max_tol": max_tol, "rollout": {k: v for k, v in row.items()
                                                                 if isinstance(v, (int, float, str))}}
    for r in recs:
        rep["by_ev"][r.ev] = rep["by_ev"].get(r.ev, 0) + 1
    rep["pass"] = bool(gaps) and rep["mean_abs_dlogp"] <= mean_tol and rep["max_abs_dlogp"] <= max_tol
    Path(out).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))
    return rep


def regress(model: str, dev: str, n: int, out: str, calibration: str | None = None) -> dict[str, Any]:
    """The frozen model's option probabilities on the first ``n`` rows of ``dev`` through ``serve_jev.Engine`` (the
    serving path): a regression corpus to compare two environments (Miles image vs project venv) or a repair."""
    from ..data import read_jsonl
    from ..serve_jev import Engine

    rows = [r for r in read_jsonl(dev) if 2 <= len(r["options"]) <= 52][:n]
    eng = Engine(model, None, calibration, name="regress")
    probs = eng._probs([(r["state"], r["question"], r["options"], r.get("kind", "choice")) for r in rows])
    import transformers

    rep = {"model": model, "dev": dev, "n": len(rows), "transformers": transformers.__version__,
           "ids": [r.get("id") for r in rows], "probs": [[float(x) for x in p] for p in probs]}
    Path(out).write_text(json.dumps(rep))
    return rep


def compare(a: str, b: str, tol: float = 0.02) -> dict[str, Any]:
    A, Bj = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    assert A["ids"] == Bj["ids"], "different corpora"
    dmax, agree = 0.0, 0
    for p, q in zip(A["probs"], Bj["probs"]):
        dmax = max(dmax, max(abs(x - y) for x, y in zip(p, q)))
        agree += int(max(range(len(p)), key=lambda i: p[i]) == max(range(len(q)), key=lambda i: q[i]))
    rep = {"a": a, "b": b, "n": len(A["probs"]), "max_abs_dprob": dmax, "argmax_agree": agree / max(1, len(A["probs"])),
           "tol": tol, "pass": dmax <= tol}
    print(json.dumps(rep, indent=1))
    return rep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    rg = sub.add_parser("regress")
    rg.add_argument("--model", required=True)
    rg.add_argument("--dev", required=True)
    rg.add_argument("--n", type=int, default=64)
    rg.add_argument("--calibration", default=None)
    rg.add_argument("--out", required=True)
    cp = sub.add_parser("compare")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--tol", type=float, default=0.02)
    cp.add_argument("--out", default=None)
    rp = sub.add_parser("replay")
    rp.add_argument("--model", required=True)
    rp.add_argument("--config", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--n-tasks", type=int, default=8)
    e = sub.add_parser("engine")
    e.add_argument("--model", required=True)
    e.add_argument("--dev", required=True)
    e.add_argument("--n", type=int, default=32)
    e.add_argument("--out", required=True)
    e.add_argument("--port", type=int, default=30011)
    e.add_argument("--mem", type=float, default=0.6)
    e.add_argument("--attention-backend", default="flashinfer")
    a = ap.parse_args(argv)
    if a.cmd == "env":
        print(json.dumps(env_report(), indent=1))
    elif a.cmd == "replay":
        rep = replay_check(a.model, a.config, a.out, a.n_tasks)
        sys.exit(0 if rep["pass"] else 1)
    elif a.cmd == "regress":
        regress(a.model, a.dev, a.n, a.out, a.calibration)
    elif a.cmd == "compare":
        rep = compare(a.a, a.b, a.tol)
        if a.out:
            Path(a.out).write_text(json.dumps(rep, indent=1))
        sys.exit(0 if rep["pass"] else 1)
    else:
        engine_check(a.model, a.dev, a.n, a.out, a.port, a.mem, a.attention_backend)


if __name__ == "__main__":
    main()
