"""Exploratory row E28 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09 "E28: our recipe on StartLux-Decision-0.8B"):
the data build (E12c's mix split, E27's augmentation on the agent states only, the public items untouched, 1 : 1,
E27's disjointness check and seed guard), the configurations and profile as registered, the pinned base revision and
the driver's guards. Offline; no model or API call."""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

import explore_e28_sl08b as X  # noqa: E402

CFG = ROOT / "configs/explore/e28_sl08b"
DRIVER = ROOT / "scripts/aws/e28_sl08b.sh"


def _cfg(name: str) -> dict:
    return yaml.safe_load((CFG / name).read_text())


def _mix(tmp_path: Path, n_agent: int = 300) -> Path:
    rows = []
    with gzip.open(ROOT / "data/train_v1/train.jsonl.gz", "rt") as f:
        for i, line in enumerate(f):
            if i >= n_agent:
                break
            r = json.loads(line)
            r["meta"] = {**(r.get("meta") or {}), "e12": "replay_v1"}
            rows.append(r)
    for i in range(n_agent):
        rows.append({"id": f"w2ctrain-{i:04d}/tool", "dp": "tool_select", "kind": "choice", "state": "USER: hi",
                     "question": "Which tool?", "options": ["a -- x", "b -- y"], "target": {"a -- x": 1.0, "b -- y": 0.0},
                     "q_values": None, "source": "w2c", "meta": {"e12": "w2c_call"}})
    p = tmp_path / "train_mix.jsonl.gz"
    X.write_jl_gz(rows, p)
    return p


def test_build_augments_agent_states_only_and_keeps_the_ratio(tmp_path, monkeypatch):
    import e27_clef as T

    p = _mix(tmp_path)
    monkeypatch.setattr(T, "E12_MIX_SHA", X.sha256_file(p))
    info = X.build(p, tmp_path / "out")
    assert info["vocab_check"]["ok"] and info["vocab_check"]["overlap_count"] == 0
    assert info["agent_rows"] == info["public_rows"] == 300 and info["mix"]["rows"] == 600
    assert 0 < info["augmentation"]["augmented_states"] < info["augmentation"]["states"]
    rows = X.read_jl(tmp_path / "out" / "train_mix_e28.jsonl.gz")
    src = {r["id"]: r for r in X.read_jl(p)}
    assert sorted(r["id"] for r in rows) == sorted(src)
    n_aug = 0
    for r in rows:
        s = src[r["id"]]
        if r["meta"]["e12"] != "replay_v1":
            assert r == s                                    # public items untouched
            continue
        if "e27_aug" in r["meta"]:
            n_aug += 1
            assert r["state"] != s["state"]
            # the answer key is only moved: the same multiset of nonzero targets
            assert sorted(v for v in r["target"].values() if v) == sorted(v for v in s["target"].values() if v)
        else:
            assert r == s
    assert n_aug == info["augmentation"]["augmented_rows"] > 0
    # deterministic (gzip mtime 0, seeded shuffle)
    info2 = X.build(p, tmp_path / "out2")
    assert info2["mix"]["sha256"] == info["mix"]["sha256"]
    assert json.loads((tmp_path / "out" / "vocab_check.json").read_text())["overlap_count"] == 0


def test_build_refuses_a_different_mix_and_unknown_rows(tmp_path):
    p = _mix(tmp_path, 20)
    with pytest.raises(SystemExit):
        X.build(p, tmp_path / "out")                     # sha256 is not E12's
    with pytest.raises(SystemExit):
        X.split_mix([{"id": "x", "meta": {}}])


def test_build_refuses_guarded_seeds(tmp_path, monkeypatch):
    import e27_clef as T

    rows = X.read_jl(_mix(tmp_path, 5))
    rows[0]["id"] = re.sub(r"-(\d+)-", "-330150-", rows[0]["id"], count=1)   # an E21 (evaluation) seed
    p = tmp_path / "bad.jsonl.gz"
    X.write_jl_gz(rows, p)
    monkeypatch.setattr(T, "E12_MIX_SHA", X.sha256_file(p))
    assert T.seed_guard([rows[0]])
    with pytest.raises(SystemExit):
        X.build(p, tmp_path / "out")


def test_base_revision_is_the_logged_rows():
    m = json.loads((ROOT / "results/explore/gpu/manifest_e2_0.8B.json").read_text())
    rev = [s for s in m["sizes"] if s["size"] == "0.8B"][0]["weights_revision"]
    pin = json.loads((CFG / "startlux_pin.json").read_text())
    assert pin["sizes"] == [{"size": "0.8B", "repo": X.BASE_REPO, "weights_revision": rev}]
    assert X.BASE_REV == rev and f"BASE_REV={rev}" in DRIVER.read_text()


def test_configs_as_registered():
    e21 = yaml.safe_load((ROOT / "configs/explore/e21_shift/live_e14a.yaml").read_text())
    e27 = yaml.safe_load((ROOT / "configs/explore/e27_small_shift/live_in.yaml").read_text())
    e27f = yaml.safe_load((ROOT / "configs/explore/e27_small_shift/final_in.yaml").read_text())
    for name, seed0, n, workers in (("live_shift", 330100, 200, 4), ("smoke_shift", 339048, 2, 2),
                                    ("live_shift_base", 330100, 200, 4), ("smoke_shift_base", 339050, 2, 2),
                                    ("final_shift", 330100, 200, 1), ("smoke_final_shift", 339052, 2, 1)):
        c = _cfg(f"{name}.yaml")
        assert {k: v for k, v in c["suite"].items() if k not in ("n", "seed0")} == \
               {k: v for k, v in e21["suite"].items() if k not in ("n", "seed0")}
        assert c["suite"]["seed0"] == seed0 and c["suite"]["n"] == n and c["n_tasks"] == n and c["workers"] == workers
        assert c["llm"] == e21["llm"]
    for name, seed0, n, workers in (("live_in", 300000, 200, 4), ("smoke_in", 944, 2, 2)):
        c = _cfg(f"{name}.yaml")
        assert {k: v for k, v in c["suite"].items() if k not in ("n", "seed0")} == \
               {k: v for k, v in e27["suite"].items() if k not in ("n", "seed0")}
        assert c["suite"]["seed0"] == seed0 and c["n_tasks"] == n and c["workers"] == workers
    assert _cfg("live_in.yaml")["suite"] == e27["suite"]
    for name, seed0 in (("final_in", 300500), ("smoke_final_in", 946)):
        c = _cfg(f"{name}.yaml")
        assert c["suite"]["seed0"] == seed0
    assert _cfg("final_in.yaml")["suite"] == e27f["suite"]
    for name in ("final_in", "final_shift", "smoke_final_in", "smoke_final_shift"):
        c = _cfg(f"{name}.yaml")
        assert c["workers"] == 1 and c["schedule"] == {"mode": "task_major", "seed": 732}
        arm, = c["arms"]
        assert arm["profile"] == "configs/harness/explore_e28_final.yaml" and arm["profile_backend"] == "e28"
        assert arm["name"] in ("e28_final_in", "e28_final_shift")
    for name in ("live_shift", "smoke_shift", "live_in", "smoke_in", "final_in", "final_shift"):
        (be, spec), = _cfg(f"{name}.yaml")["s1_backends"].items()
        assert spec["model"] == "startlux-0.8b-mixed" and spec["base_url"].endswith(":8081") and spec["cache"] is None
    for name in ("live_shift_base", "smoke_shift_base"):
        c = _cfg(f"{name}.yaml")
        (be, spec), = c["s1_backends"].items()
        assert spec["model"] == "StartLux-Decision-0.8B" and spec["base_url"].endswith(":8092")
        assert c["s1_cache"].startswith("cache/s1_e28_")
    # smokes on development seeds only, never the registered evaluation tasks
    for name in ("smoke_shift", "smoke_shift_base", "smoke_final_shift"):
        s0 = _cfg(f"{name}.yaml")["suite"]["seed0"]
        assert 339040 <= s0 <= 339098 and not (339040 <= s0 <= 339047)       # E27 used 339040-339047
    for name in ("smoke_in", "smoke_final_in"):
        s0 = _cfg(f"{name}.yaml")["suite"]["seed0"]
        assert 944 <= s0 and s0 + 1 <= 947
    e1 = _cfg("e1_e28.yaml")
    e12c = yaml.safe_load((ROOT / "configs/explore/e12_train/e1_e12c.yaml").read_text())
    for k in ("suites", "exclude_ids_file", "batch_by_state", "workers"):
        assert e1[k] == e12c[k]
    assert set(e1["backends"]) == {"e28"} and e1["s1_cache"] != e12c["s1_cache"]
    assert e1["name"] == X.E1_RUN


def test_profile_pending_or_fitted():
    from jevharness.harness.profile import load_profile, resolve_arm

    prof = load_profile(X.PROFILE)
    base = load_profile(X.FINAL_PROFILE)
    sec, bsec = prof["system2"]["gemini-3.5-flash"], base["system2"]["gemini-3.5-flash"]
    assert sec["decision_points"] == bsec["decision_points"]
    assert sec["backends"]["jevagent_rl"] == bsec["backends"]["jevagent_rl"]
    c = _cfg("final_in.yaml")
    out, meta = resolve_arm(c["arms"][0], c)
    if prof["status"] == "pending":
        assert meta["skip"]
        return
    blk = sec["backends"]["e28"]
    fits = json.loads((X.CAL_OUT / "fits.json").read_text())
    for dp in X.DPS:
        f = blk["fitted_from"][dp]
        assert X.sha256_file(ROOT / f["path"]) == f["sha256"]
        t = fits["fits"][dp]["tau_hat"]
        if blk[dp].get("always_defer"):
            assert t is None or t > 1.0
        else:
            assert blk[dp]["tau"] == pytest.approx(float(t))


def test_driver_syntax_and_guards():
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)
    txt = DRIVER.read_text()
    forb = set(re.findall(r"(?m)^(i-0[0-9a-f]{16})$", (CFG / "forbidden_iids.txt").read_text()))
    for iid in ("<instance-id>", "<instance-id>", "<instance-id>", "<instance-id>",
                "<instance-id>"):
        assert iid in forb
    others = set()
    for d in (ROOT / "scripts/aws").glob("*.sh"):
        if d.name != DRIVER.name:
            others |= set(re.findall(r"(?<![A-Za-z0-9])i-0[0-9a-f]{16}", d.read_text()))
    assert others <= forb
    assert not re.findall(r"(?<![A-Za-z0-9])i-0[0-9a-f]{16}", txt)        # the list lives in the config file
    assert "TAG=jev-e28-sl08b" in txt and "GUARD_MIN=${GUARD_MIN:-470}" in txt and "CAP_USD=${CAP_USD:-30}" in txt
    assert "types=(g6e.2xlarge g5.2xlarge)" in txt
    assert X.CAL_REQ_SHA in txt and "OPENROUTER" not in txt and "OPENAI_API" not in txt
    assert "--lr 1e-4 --epochs 2 --max-len 4096 --seed 0" in txt and "timeout 150m" in txt
    assert "--lora-r 32 --lora-alpha 64" in txt and "--loss soft_ce_brier --brier-weight 0.5" in txt


def test_merge_vendor_keeps_the_layout_and_adds_scaled_lora(tmp_path):
    import torch
    from safetensors.torch import load_file, save_file

    base, ad, out = tmp_path / "base", tmp_path / "ad", tmp_path / "out"
    base.mkdir(), ad.mkdir()
    g = torch.Generator().manual_seed(0)
    w = {"model.language_model.layers.0.mlp.up_proj.weight": torch.randn(6, 4, generator=g).to(torch.bfloat16),
         "model.language_model.layers.0.input_layernorm.weight": torch.ones(4, dtype=torch.bfloat16),
         "model.visual.blocks.0.attn.qkv.weight": torch.randn(3, 3, generator=g).to(torch.bfloat16),
         "mtp.layers.0.mlp.up_proj.weight": torch.randn(6, 4, generator=g).to(torch.bfloat16)}
    save_file(w, str(base / "model-00001-of-00001.safetensors"))
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors"
                                                                                  for k in w}}))
    for f in ("config.json", "decision_config.json", "tokenizer.json"):
        (base / f).write_text("{}")
    (base / ".cache").mkdir()
    A, B = torch.randn(2, 4, generator=g), torch.randn(6, 2, generator=g)
    save_file({"base_model.model.model.layers.0.mlp.up_proj.lora_A.weight": A,
               "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight": B}, str(ad / "adapter_model.safetensors"))
    (ad / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "use_rslora": False}))
    rec = X.merge_vendor(base, ad, out)
    assert rec["modules_merged"] == 1 and rec["scale"] == 2.0 and rec["base_tensors"] == 4
    m = load_file(str(out / "model-00001-of-00001.safetensors"))
    assert set(m) == set(w)
    k = "model.language_model.layers.0.mlp.up_proj.weight"
    want = (w[k].float() + 2.0 * (B @ A)).to(torch.bfloat16)
    assert torch.equal(m[k], want) and m[k].dtype == torch.bfloat16
    for kk in w:
        if kk != k:
            assert torch.equal(m[kk], w[kk])                 # mtp / visual / norms untouched
    assert (out / "decision_config.json").exists() and not (out / ".cache").exists()
    # an adapter module without a base weight is refused
    save_file({"base_model.model.model.layers.9.mlp.up_proj.lora_A.weight": A,
               "base_model.model.model.layers.9.mlp.up_proj.lora_B.weight": B}, str(ad / "adapter_model.safetensors"))
    with pytest.raises(SystemExit):
        X.merge_vendor(base, ad, tmp_path / "out2")


def test_vendor_timing_config_is_the_ladder_s():
    v = _cfg("e1_e28_vendor.yaml")
    lad = yaml.safe_load((ROOT / "configs/decision/explore_bfcl_when2call_startlux_ladder.yaml").read_text())
    for k in ("suites", "exclude_ids_file", "batch_by_state", "workers"):
        assert v[k] == lad[k]
    (be, spec), = v["backends"].items()
    assert be == "e28v" and spec["base_url"].endswith(":8092") and v["name"] == X.E1_VENDOR_RUN
    assert spec["price_model"] == lad["backends"]["startlux_0_8b"]["price_model"]
    assert "stage_vendor_e1()" in DRIVER.read_text()
