"""On-policy distillation (training/opd.py, scripts/train/65_opd.sh). Offline: tiny local Qwen3-style model, the
simulated System-2 executor, simulated / local (in-process Engine) teachers; CPU, about a minute in total."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

ROOT = Path(__file__).resolve().parents[1]
SIM_TEACHER = 'sim={"type": "simulated", "accuracy": 0.9, "seed": 1, "latency_ms": 0}'


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    sys.path.insert(0, str(ROOT / "scripts/train"))
    from make_tiny_model import make_tiny_model

    return make_tiny_model(tmp_path_factory.mktemp("tiny"))


def _args(tiny: Path, out: Path, *extra: str) -> list[str]:
    return ["--student", str(tiny), "--teacher", SIM_TEACHER, "--executor", "sim", "--iterations", "1",
            "--tasks-per-iter", "2", "--n-tasks", "4", "--suite", "{n_distractors: 3}", "--max-steps", "6",
            "--workers", "2", "--lora-r", "4", "--lora-alpha", "8", "--max-len", "512", "--lr", "0.01",
            "--no-bf16", "--out", str(out), *extra]


def _rows(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]


def _lora_b_abs_max(path: Path) -> float:
    from safetensors.torch import load_file

    w = load_file(str(path / "adapter_model.safetensors"))
    return max(float(t.abs().max()) for k, t in w.items() if "lora_B" in k)


# ------------------------------------------------------------------ loss
def test_opd_loss_variants_and_relation_to_soft_ce():
    from jevharness.training.decision_model import decision_loss
    from jevharness.training.opd import LOSSES, forward_kl, opd_loss

    torch.manual_seed(0)
    z = torch.randn(3, 4)
    z[1, 3] = float("-inf")  # row 1 has 3 options (the collator's padding convention)
    t = torch.softmax(torch.randn(3, 4), -1)
    t[1, 3] = 0.0
    t = t / t.sum(-1, keepdim=True)
    mask = torch.isfinite(z)
    fkl, rkl, jsd = (opd_loss(z, t, k) for k in LOSSES)
    assert torch.all(fkl >= -1e-6) and torch.all(rkl >= -1e-6) and torch.all(jsd >= -1e-6)
    assert torch.all(jsd <= math.log(2) + 1e-6)  # JSD (base e) is bounded by log 2
    # forward KL = soft CE - H(teacher): identical gradient to the stages-20/40/80 objective
    batch = {"code_ids": torch.where(mask, torch.ones_like(z, dtype=torch.long), torch.full_like(z, -1).long()),
             "target": t}
    ce = decision_loss(z, batch, "soft_ce")
    ent = -(torch.where(t > 0, t * torch.log(t.clamp(min=1e-12)), torch.zeros_like(t))).sum(-1)
    assert torch.allclose(fkl.mean(), ce - ent.mean(), atol=1e-5)
    # every divergence is 0 when the student equals the teacher
    zt = torch.log(t.clamp(min=1e-12)).masked_fill(~mask, float("-inf"))
    for k in LOSSES:
        assert float(opd_loss(zt, t, k).abs().max()) < 1e-5, k
    # gradients pull the student towards the teacher (forward KL: d/dz = p - t)
    zg = z.clone().requires_grad_(True)
    opd_loss(zg, t, "forward_kl").sum().backward()
    p = torch.softmax(z, -1).masked_fill(~mask, 0.0)
    assert torch.allclose(zg.grad.masked_fill(~mask, 0.0), (p - t).masked_fill(~mask, 0.0), atol=1e-5)
    assert forward_kl([0.2, 0.8], [0.0, 0.0]) == pytest.approx(0.2 * math.log(0.4) + 0.8 * math.log(1.6), abs=1e-6)
    with pytest.raises(ValueError):
        opd_loss(z, t, "soft_ce")


# ------------------------------------------------------------------ teacher questions
def test_teacher_question_restricts_to_the_pool_and_hides_gold_from_real_teachers():
    from jevharness.decision import Choice, Noul
    from jevharness.decision.types import QKind
    from jevharness.training.opd import Visit, teacher_question

    q = Choice("which?", ["a", "b", "c"], ["A", "B", "C"], decision_point="tool_select", gold="b")
    v = Visit(prompt="p", n=2, event=[0], ev="choice", old_logp=0.0, dp="tool_select", step=0, key="k", sampled=0,
              greedy=0, gold=1, conf=0.5, entropy=0.7, options=["c -- C", "b -- B"], labels=["c", "b"], pool=[2, 1],
              question=q)
    real = teacher_question(v, expose_gold=False)
    assert real.kind == QKind.CHOICE and real.options == ["c", "b"] and real.option_descriptions == ["C", "B"]
    assert real.meta == {}  # gold never goes on the wire
    assert teacher_question(v, expose_gold=True).meta == {"gold": "b"}
    v.pool, v.labels = [0, 2], ["a", "c"]  # knockout pool without the gold: nothing to expose
    assert teacher_question(v, expose_gold=True).meta == {}
    n = Noul("done?", decision_point="stop", gold=False, criteria={"true": "t", "false": "f"})
    vn = Visit(prompt="p", n=2, event=[0], ev="gate", old_logp=0.0, dp="stop", step=0, key="k", sampled=0, greedy=0,
               gold=1, conf=0.5, entropy=0.7, kind="noul", options=["yes -- t", "no -- f"], labels=["yes", "no"],
               pool=[0, 1], question=n)
    tq = teacher_question(vn, expose_gold=False)
    assert tq.kind == QKind.NOUL and tq.meta == {"criteria": {"true": "t", "false": "f"}}
    assert teacher_question(vn, expose_gold=True).meta["gold"] is False


def test_real_teacher_backends_never_see_gold(tiny: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from jevharness.decision.simulated import SimulatedBackend
    from jevharness.training import opd

    seen: list[dict] = []

    class Recorder(SimulatedBackend):
        batched_questions = True

        def _decide(self, state, questions):  # noqa: ANN001, ANN202
            seen.extend(q.meta for q in questions)
            return super()._decide(state, questions)

    monkeypatch.setattr(opd, "make_teacher", lambda spec, cache: Recorder(latency_ms=0))
    out = tmp_path / "real"
    opd.main(_args(tiny, out, "--teacher", '{"type": "jev", "provider": "openrouter"}'))
    assert seen and all("gold" not in m for m in seen)
    assert _rows(out)[0]["labelled"] == len(seen) > 0


# ------------------------------------------------------------------ end to end
def test_opd_sim_one_iteration_trains_and_writes_outputs(tiny: Path, tmp_path: Path):
    from jevharness.training.data import read_jsonl, render
    from jevharness.training.opd import main

    out = tmp_path / "r9"
    main(_args(tiny, out, "--eval-every", "1", "--eval-n", "1", "--log-rollouts"))
    rows = _rows(out)
    assert [r["iter"] for r in rows] == [0, 1] and "eval/opd_replace/success" in rows[0]
    r = rows[1]
    assert r["episodes"] == 2 and r["decisions"] > 0 and r["labelled"] == r["decisions"] == r["fresh_n"]
    assert r["replay_n"] == 0 and r["train_n"] == r["fresh_n"] and r["optimizer_steps"] >= 1
    for k in ("success", "cost_usd", "llm_calls", "agree_greedy", "agree_sampled", "teacher_acc",
              "kl_teacher_student", "loss", "teacher_cost_iter", "teacher_cost_total", "teacher_calls_iter"):
        assert k in r and (r[k] is None or math.isfinite(r[k])), k
    assert r["teacher_calls_iter"] > 0 and r["teacher_cost_total"] == 0.0 and "eval/opd_replace/success" in r
    assert r["agree_greedy/tool_select"] is not None and r["labelled/tool_select"] > 0
    assert _lora_b_abs_max(out) > 0 and (out / "ckpt-00001" / "trainer_state.pt").exists()
    assert (out / "rollouts.jsonl").exists() and (out / "opd_config.yaml").exists()
    cfg = json.loads((out / "decision_config.json").read_text())
    assert cfg["loss"] == "opd" and cfg["opd_loss"] == "forward_kl" and cfg["lora"]
    summ = json.loads((out / "summary.json").read_text())
    assert summ["stop_reason"] == "iterations" and summ["teacher_ledger"]["calls"] == r["teacher_calls_iter"]
    # the labelled rows are training-format rows whose prompt is the student's rollout prompt, byte for byte
    lab = read_jsonl(out / "labelled" / "iter-00001.jsonl")
    assert len(lab) == r["labelled"]
    for e in lab:
        assert e["meta"]["label_source"] == "opd_teacher" and set(e["target"]) == set(e["options"])
        assert math.isclose(sum(e["target"].values()), 1.0, rel_tol=1e-6) and 2 <= len(e["options"]) <= 52
        assert e["id"].split("/")[0].startswith("jevgym") and int(e["id"].split("-")[2]) >= 10000
        render(e)  # renderable as served
    assert {e["dp"] for e in lab} == {"tool_select", "tool_params"}


def test_opd_prompt_tokens_identical_to_render_and_chat_text(tiny: Path, tmp_path: Path,
                                                             monkeypatch: pytest.MonkeyPatch):
    """The tokens fed to the student in the update are exactly ``chat_text(render(row))`` of the labelled rows, and
    without permutation the update prompt IS the rollout prompt the student was scored with."""
    from transformers import AutoTokenizer

    from jevharness.training import opd
    from jevharness.training.data import read_jsonl, render
    from jevharness.training.decision_model import chat_text

    tok = AutoTokenizer.from_pretrained(tiny)
    fed: list[list[int]] = []
    real = opd.option_logits
    rollout_prompts: list[str] = []
    real_act = opd.OPDDecider._act

    def spy_logits(model, input_ids, attention_mask, code_ids):  # noqa: ANN001, ANN202
        if torch.is_grad_enabled():  # the update (rollout scoring runs under inference_mode)
            for ids, m in zip(input_ids.tolist(), attention_mask.tolist()):
                fed.append([t for t, keep in zip(ids, m) if keep])
        return real(model, input_ids, attention_mask, code_ids)

    def spy_act(self, q, ex, pool, prompt, z):  # noqa: ANN001, ANN202
        ans, v = real_act(self, q, ex, pool, prompt, z)
        rollout_prompts.append(v.prompt)
        return ans, v

    monkeypatch.setattr(opd, "option_logits", spy_logits)
    monkeypatch.setattr(opd.OPDDecider, "_act", spy_act)
    out = tmp_path / "bytes"
    opd.main(_args(tiny, out, "--no-permute"))
    lab = read_jsonl(out / "labelled" / "iter-00001.jsonl")
    assert fed and len(fed) == len(lab)
    expected = sorted(tok(chat_text(tok, render(e)[0]), add_special_tokens=False).input_ids[-512:] for e in lab)
    assert sorted(fed) == expected  # byte-identical prompts -> identical token ids (minibatch order differs)
    assert {render(e)[0] for e in lab} <= set(rollout_prompts)  # the update prompt is the rollout prompt


def test_opd_replay_mixing(tiny: Path, tmp_path: Path):
    from jevharness.training.opd import main

    out = tmp_path / "replay"
    main(_args(tiny, out, "--iterations", "2", "--replay-frac", "1.0"))
    r1, r2 = _rows(out)
    assert r1["replay_n"] == 0 and r2["replay_n"] == min(r1["fresh_n"], r2["fresh_n"]) > 0
    assert r2["train_n"] == r2["fresh_n"] + r2["replay_n"]
    out0 = tmp_path / "noreplay"
    main(_args(tiny, out0, "--iterations", "2", "--replay-frac", "0"))
    assert all(r["replay_n"] == 0 and r["train_n"] == r["fresh_n"] for r in _rows(out0))


def test_opd_budget_stop(tiny: Path, tmp_path: Path):
    from jevharness.training.opd import main

    out = tmp_path / "budget"
    priced = 'sim={"type": "simulated", "accuracy": 0.9, "latency_ms": 0, "price_per_m_input": 1000.0}'
    main(_args(tiny, out, "--iterations", "3", "--teacher", priced, "--budget-usd", "0.001"))
    rows = _rows(out)
    assert len(rows) == 1 and rows[0]["stop_reason"] == "budget" and rows[0]["teacher_cost_total"] >= 0.001
    summ = json.loads((out / "summary.json").read_text())
    assert summ["stop_reason"] == "budget" and summ["iterations_done"] == 1
    assert (out / "ckpt-00001").exists() and _lora_b_abs_max(out) > 0  # the paid labels were still trained on


def test_opd_resume(tiny: Path, tmp_path: Path):
    from jevharness.training.opd import main

    out = tmp_path / "a"
    main(_args(tiny, out, "--iterations", "1"))
    st = torch.load(out / "ckpt-00001" / "trainer_state.pt", weights_only=False)
    assert st["iteration"] == 1 and st["teacher_calls_total"] == _rows(out)[0]["teacher_calls_iter"]
    out_b = tmp_path / "b"
    main(_args(tiny, out_b, "--iterations", "2", "--resume", str(out / "ckpt-00001"), "--replay-frac", "1.0"))
    rows = _rows(out_b)
    assert [r["iter"] for r in rows] == [2] and rows[0]["replay_n"] > 0  # replay reloaded from the checkpoint
    assert rows[0]["teacher_calls_total"] == st["teacher_calls_total"] + rows[0]["teacher_calls_iter"]
    # `--resume auto` picks the latest checkpoint under --out and runs nothing when iterations are done
    main(_args(tiny, out_b, "--iterations", "2", "--resume", "auto"))
    assert [r["iter"] for r in _rows(out_b)] == [2]


def test_opd_local_engine_teacher_and_losses(tiny: Path, tmp_path: Path):
    """A checkpoint read out in-process by serve_jev.Engine is a teacher; every loss option trains."""
    from jevharness.training.opd import main

    local = json.dumps({"type": "local", "model": str(tiny), "max_len": 512, "batch_size": 4})
    for loss in ("reverse_kl", "jsd"):
        out = tmp_path / loss
        main(_args(tiny, out, "--teacher", f"t={local}", "--loss", loss, "--rollout-temperature", "0"))
        r = _rows(out)[0]
        assert r["labelled"] > 0 and math.isfinite(r["loss"]) and r["teacher_input_tokens_iter"] > 0
        assert json.loads((out / "decision_config.json").read_text())["opd_loss"] == loss
        assert _lora_b_abs_max(out) > 0


def test_opd_rejects_evaluation_seeds(tiny: Path, tmp_path: Path):
    from jevharness.training.opd import parse_args

    with pytest.raises(SystemExit):
        parse_args(_args(tiny, tmp_path, "--seeds", "0"))
    with pytest.raises(SystemExit):
        parse_args(_args(tiny, tmp_path, "--seeds", "5000"))
    with pytest.raises(SystemExit):  # dev seeds overlapping the training pool
        parse_args(_args(tiny, tmp_path, "--eval-every", "1", "--eval-seed0", "10002"))
    a = parse_args(_args(tiny, tmp_path))
    assert a.teacher_name == "sim" and a.teacher_spec["type"] == "simulated" and a.suite["strict"]


def test_opd_config_yaml_and_redaction(tiny: Path, tmp_path: Path):
    from jevharness.training.opd import parse_args, redact

    a = parse_args(["--config", str(ROOT / "configs/train/opd_jevgym.yaml"), "--student", str(tiny)])
    assert a.seeds >= 10000 and a.eval_seed0 >= 1000 and a.loss == "forward_kl" and a.teacher_spec["type"] == "jev"
    assert a.executor == "jevgym" and a.llm["provider"] == "gemini" and a.budget_usd > 0
    s = parse_args(["--config", str(ROOT / "configs/train/opd_smoke.yaml"), "--student", str(tiny)])
    assert s.executor == "sim" and s.iterations == 1 and s.tasks_per_iter == 2
    (tmp_path / "bad.yaml").write_text("studnet: x\n")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(tmp_path / "bad.yaml"), "--student", str(tiny)])
    assert redact({"type": "jev", "api_key": "sk-1", "extra": {"token": "t", "model": "m"}}) == \
        {"type": "jev", "extra": {"model": "m"}}


# ------------------------------------------------------------------ stage script
def test_65_opd_dry_run(tiny: Path, tmp_path: Path):
    env = {**os.environ, "DRY_RUN": "1", "MODEL": str(tiny), "INIT": "", "EXECUTOR": "sim", "LOSS": "jsd",
           "BUDGET_USD": "2", "TEACHER": SIM_TEACHER, "OUT": str(tmp_path / "o"), "RESUME": "auto",
           "CONFIG": "configs/train/opd_smoke.yaml"}
    r = subprocess.run(["bash", str(ROOT / "scripts/train/65_opd.sh")], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    cmds = [c for c in r.stdout.splitlines() if c.startswith("+ ")]
    assert len(cmds) == 1 and "jevharness.training.opd" in cmds[0]
    for part in ("--executor sim", "--loss jsd", "--budget-usd 2", "--resume auto", "opd_smoke.yaml"):
        assert part in cmds[0], part
    assert "loss: jsd" in r.stdout and "budget_usd: 2.0" in r.stdout  # --print-config under DRY_RUN
    r2 = subprocess.run(["bash", str(ROOT / "scripts/train/65_opd.sh")], cwd=ROOT, capture_output=True, text=True,
                        env={**os.environ, "DRY_RUN": "1", "GPUS": "8", "EXECUTOR": "gemini-lite"}, timeout=120)
    assert r2.returncode == 0 and "gemini-3.5-flash-lite" in r2.stdout and "jev-openrouter" in r2.stdout


def test_opd_round5_options_named_executors_budget_gold_fallback(tiny: Path, tmp_path: Path):
    """Round-5 paths: per-arm executors (`llm: <name of --llms>`), --eval-arms, --strict-seeds, token-budget update
    micro-batches, --gold-fallback (target = gold one-hot where the teacher's argmax is not the gold)."""
    from jevharness.training.opd import parse_args, train

    out = tmp_path / "r5"
    arms = ("[{name: rep, tool_select: {mode: replace}, tool_params: {mode: replace}}, "
            "{name: sh, llm: other, tool_select: {mode: shadow}, stop: {mode: shadow}}]")
    a = parse_args(_args(tiny, out, "--arms", arms, "--llms", "{other: {provider: simulated, model: scripted, "
                         "error_rate: 0.3, simulated_latency_ms: 0}}", "--eval-every", "1", "--eval-n", "2",
                         "--eval-seed0", "1000", "--eval-arms", "rep", "--strict-seeds", "--max-batch-tokens", "300",
                         "--gold-fallback", "--gold-alpha", "0.3", "--teacher",
                         'sim={"type": "simulated", "accuracy": 0.5, "seed": 3, "latency_ms": 0}'))
    train(a)
    rows = _rows(out)
    it = [r for r in rows if r["iter"] == 1][0]
    assert "eval/rep/success" in rows[0] and not any(k.startswith("eval/sh/") for k in rows[0])
    assert any(k.startswith("arm/rep/") for k in it) and any(k.startswith("arm/sh/") for k in it)
    assert it["optimizer_steps"] > 0 and it["gold_fallback_n"] >= 0
    lab = [json.loads(line) for line in (out / "labelled" / "iter-00001.jsonl").read_text().splitlines()]
    for r in lab:
        g = r["meta"]["gold"]
        if r["meta"]["label_source"] == "gold_fallback":
            assert r["target"][next(o for o in r["options"] if o == g or o.startswith(f"{g} --") or o.startswith(f"{g}"))] == 1.0
        elif g is not None and r["meta"]["label_source"] == "opd_teacher":
            best = max(r["target"], key=r["target"].get)
            assert best == g or best.startswith(f"{g} --") or best.split(" -- ")[0] == g


def test_opd_behaviour_frozen_control(tiny: Path, tmp_path: Path):
    """A7.13 (iii): rollouts by a frozen copy of --init; only the trained adapter moves."""
    from safetensors.torch import load_file

    from jevharness.training.opd import parse_args, train

    init = tmp_path / "init"
    train(parse_args(_args(tiny, init)))
    out = tmp_path / "frozen"
    train(parse_args(_args(tiny, out, "--init", str(init), "--behaviour-frozen", "--iterations", "2")))
    a, b = load_file(str(init / "adapter_model.safetensors")), load_file(str(out / "adapter_model.safetensors"))
    assert set(a) == set(b) and any(float((a[k] - b[k]).abs().max()) > 0 for k in a)
    assert all(r.get("decisions", 1) > 0 for r in _rows(out) if r["iter"] > 0)
    with pytest.raises(SystemExit):
        train(parse_args(_args(tiny, tmp_path / "bad", "--behaviour-frozen")))
