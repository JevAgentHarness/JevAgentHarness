"""scripts/leakage_audit.py (offline): task-id and seed handling, the trainers' rollout-seed reconstruction, the
noise-free state normalisation and entity masking, MinHash estimates against the exact Jaccard (including the
monotone-hash regression), JevGym task-spec fingerprints, task-weighted corpus accuracy, and -- when the local data
are present -- the teacher-cache re-keying and the reconstruction check against the logged v2 GRPO seeds."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import leakage_audit as LA  # noqa: E402


def test_task_ids_seeds_and_ranges():
    p = LA.parse_task_id("jevgym-clarify-300352-d60-near-strict/nearmiss_replace:10/stop:shadow")
    assert p == {"family": "clarify", "seed": 300352, "n_distractors": 60, "distractor_style": "near", "strict": True,
                 "task_id": "jevgym-clarify-300352-d60-near-strict"}
    assert LA.parse_task_id("jevgym-stock-12002-d30-mixed-strict/0/tool_select")["distractor_style"] == "mixed"
    assert LA.parse_task_id("w2ctrain-81e246ee34f1/when2call") is None
    assert LA.seeds_in_bytes(b'{"task_id": "jevgym-multi-200001-d60-near-strict"} jevgym-cancel-7-d10') == {200001, 7}
    assert LA.ranges([5, 1, 2, 3, 9, 10]) == [[1, 3], [5, 5], [9, 10]]
    assert LA.ranges_text(range(10)) == "0-9"
    assert [LA.classify_seed(s) for s in (12, 5000, 10000, 100049, 200000, 300499, 300500)] == \
        ["development", "calibration", "training", "retired", "stage1_eval", "stage2_eval", "stage2_spare"]


def test_rollout_seed_reconstruction_matches_the_trainers_sampler():
    # opd.py / grpo_online.py: Random(seed).shuffle(range(n)), a cursor that reshuffles at the end of the order
    rng = random.Random(0)
    order = list(range(20))
    rng.shuffle(order)
    expect = [100 + i for i in order]
    rng.shuffle(order)
    expect += [100 + i for i in order[:4]]
    assert LA.rollout_seeds(100, 20, 8, 3, style="cursor") == expect
    # Miles: order[(rid * k + j) % n] without reshuffling
    assert LA.rollout_seeds(100, 20, 8, 3, style="modular")[:20] == expect[:20]
    # the first v2 GRPO rollout as logged in results/gpu_round6_miles/grpo_rollout_metrics.jsonl
    assert LA.rollout_seeds(70000, 3000, 8, 1, style="modular") == [72639, 70247, 72120, 70824, 72013, 70712, 71871,
                                                                       70561]


def test_noise_free_normalisation_and_masking():
    a = ("TASK: Cancel Sven Kim's order.\n\nHISTORY:\n[0] find_user({\"name\": \"Sven Kim\"}) -> {\"user_id\": \"U8399\"}\n"
         "log: heartbeat ok #52342\nlog: heartbea\n...[truncated]...\nartbeat ok #448029\nlog: heartbeat ok #1\n\n"
         "CURRENT OBSERVATION:\nok")
    b = a.replace("#52342", "#895403").replace("#448029", "#7").replace("log: heartbea\n", "log: h\n")
    assert LA.normalise_state(a) == LA.normalise_state(b)
    assert "heartbeat" not in LA.normalise_state(a) and "u8399" in LA.normalise_state(a)
    assert LA.normalise_state(a) != LA.normalise_state(a.replace("U8399", "U8398"))
    for line in ("log: heartbeat ok #12", "log: hear", "beat ok #4480", "#12", "12", "...[truncated]..."):
        assert LA.is_noise_line(line), line
    for line in ("[1] get_orders()", "U8399", '{"user_id": "U1"}', "ok"):
        assert not LA.is_noise_line(line), line
    m = LA.mask_entities("Refund Ivan Wang's order ORD-4822 of user U1701 for 227.09")
    assert m == "refund name name's order ordid of user userid for num"
    assert LA.mask_entities("Refund Dana Kim's order ORD-1 of user U2 for 3") == m


def test_minhash_tracks_the_exact_jaccard():
    mh = LA.MinHasher()
    rng = random.Random(1)
    words = [f"w{i}" for i in range(300)]
    errs = []
    for _ in range(60):
        t1 = " ".join(rng.choice(words) for _ in range(80))
        k = rng.randint(0, 80)
        t2 = " ".join(t1.split()[:k] + [rng.choice(words) for _ in range(80 - k)])
        est = float((mh.signature(LA.shingle_hashes(t1)) == mh.signature(LA.shingle_hashes(t2))).mean())
        errs.append(est - LA.jaccard(t1, t2))
    assert abs(np.mean(errs)) < 0.03 and max(map(abs, errs)) < 0.2
    t = "the same text with several words in it for shingles"
    assert (mh.signature(LA.shingle_hashes(t)) == mh.signature(LA.shingle_hashes(t))).all()
    # regression: two short texts sharing one shingle used to collide on (almost) every permutation when the
    # modulus left a*x + b monotone in x
    a = "i want to retrieve the details of a youtube community post."
    b = "user: retrieve the details of a supreme court case titled 'roe v. wade'.include dissent information."
    est = float((mh.signature(LA.shingle_hashes(a)) == mh.signature(LA.shingle_hashes(b))).mean())
    assert est < 0.3 and LA.jaccard(a, b) < 0.1


def test_near_scores_verifies_flagged_pairs_exactly():
    base = " ".join(f"tok{i}" for i in range(60))
    near = base.replace("tok30", "other30")
    far = " ".join(f"zz{i}" for i in range(60))
    score, est, cal = LA.near_scores([near, far], [base])
    assert score[0] == pytest.approx(LA.jaccard(near, base)) and score[1] < 0.2
    assert cal["verified"] == 1
    # within-corpus baseline: the same group is never its own neighbour
    s2, _, _ = LA.near_scores([base, near], [base, near], ["t1", "t1"], ["t1", "t1"])
    assert (s2 <= 0).all()


def test_task_specs_and_overlap():
    a = LA.make_env("jevgym-clarify-300000-d60-near-strict")
    b = LA.make_env("jevgym-clarify-300000-d60-near-strict")
    c = LA.make_env("jevgym-multi-300001-d60-near-strict")
    sa, sb, sc = LA.task_spec(a), LA.task_spec(b), LA.task_spec(c)
    assert sa == sb and sa["plan"] != sc["plan"] and sa["world"] != sc["world"]
    with pytest.raises(ValueError):
        LA.make_env("jevgym-multi-300000-d60-near-strict")      # seed 300000 is a clarify task in the v2 round-robin
    ov = LA.spec_overlap({"e": sa}, {"t1": sb, "t2": sc})
    assert ov["plan_dup"] == 1 and ov["world_dup"] == 1 and ov["instruction_rate"] == 1.0
    ov = LA.spec_overlap({"e": sc}, {"t1": sa})
    assert ov["plan_dup"] == 0 and ov["world_dup"] == 0


def test_corpus_accuracy_is_task_weighted():
    occ = [{"task": "t1", "qhash": "q1", "gold": "a", "gold_aux": {"acceptable_tools": ["a"]}, "gold_status": "ok"},
           {"task": "t1", "qhash": "q2", "gold": "b", "gold_aux": {"acceptable_tools": ["b"]}, "gold_status": "ok"},
           {"task": "t2", "qhash": "q3", "gold": "c", "gold_aux": {"acceptable_tools": ["c", "d"]}, "gold_status": "ok"}]
    ans = {"q1": {"answer": "a"}, "q2": {"answer": "x"}, "q3": {"answer": "d"}}
    r = LA.corpus_accuracy(occ, ans)
    assert r["acc"] == pytest.approx((0.5 + 1.0) / 2) and r["n_occ"] == 3 and r["n_tasks"] == 2
    assert LA.corpus_accuracy(occ, ans, [1])["acc"] == 0.0
    assert LA.corpus_accuracy(occ, {"q1": {"answer": "a", "error": "x"}}, [0])["acc"] == 0.0   # failed = incorrect


@pytest.mark.skipif(not (LA.R6 / "grpo_rollout_metrics.jsonl").exists(), reason="v2 GRPO log not present")
def test_reconstruction_reproduces_the_logged_v2_grpo_seeds():
    rc = LA.validate_reconstruction()
    assert rc["checked"] and rc["identical"] and rc["n_logged"] == 240


@pytest.mark.skipif(not (LA.TEACHER_CACHE.exists() and (LA.V1_CACHE / "_work/train_gold.jsonl").exists()),
                    reason="data v1 teacher cache / work files not present")
def test_teacher_cache_holds_exactly_the_training_requests():
    tc = LA.teacher_cache_check()
    assert tc["matched"] == tc["cache_entries"] and tc["cache_only"] == 0 and tc["rows_only"] == 0
    assert tc["row_seeds_in_eval_ranges"] == 0 and tc["row_seed_min"] >= 10000


def test_wire_canary_offline():
    wc = LA.wire_canary()
    assert wc["canary_in_meta"] and not wc["canary_in_wire"] and not wc["canary_in_serve_prompt"]
