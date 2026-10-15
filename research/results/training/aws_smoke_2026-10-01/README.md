# First GPU smoke on AWS, 2026-10-01

**Pipeline check plus zero-shot R0 baselines; nothing here is a training result.**

- Host: AWS `g6.4xlarge` (1x NVIDIA L4, 22 GB usable, 16 vCPU, 64 GiB RAM), us-east-1a, on-demand $1.32/h.
  Deep Learning AMI PyTorch 2.14 (Ubuntu 26.04), driver 595.91 (CUDA 13.2). g6e.2xlarge (L40S) had no capacity
  in any us-east-1 zone at launch time.
- Guard rails: 12 h hard power-off, stop after 60 min of GPU idle (cron watchdog), shutdown behaviour = stop, no
  inbound ports (SSM only), instance role limited to SSM + the project bucket. The instance was stopped by hand right
  after the run (about 40 min of uptime, about $0.9).
- Code: git bundle of the branch via a private S3 bucket (no keys on the box; the smoke needs none).

## Results

| Step | Commit | Outcome |
|---|---|---|
| `scripts/train/setup_gpu.sh` | `69d4912` | ok in 2 min: pinned stack (torch 2.14.1+cu130, transformers 5.18.0, TRL 1.14.1, PEFT 0.21.1), CUDA + bf16, TRL fields, 52 single-token codes, 1-step fwd/bwd, full offline pytest green (`env_report.json`) |
| `scripts/train/smoke_gpu.sh`, first run | `69d4912` | stages 00-55 ok, **60-grpo-trl OOM**: every stage ran on Qwen3-4B, not 0.6B, because `_common.sh`'s 4B default won (`smoke_report_4B_unintended.json`) |
| `scripts/train/smoke_gpu.sh`, after the fix | `a95834d` | **all 14 stages ok on Qwen3-0.6B in about 5 min** (`smoke_report_0.6B.json`) |

Bugs found and fixed: `setup_gpu.sh` could not install uv on PEP 668 distros (`69d4912`); `smoke_gpu.sh` ran on 4B
instead of 0.6B (`a95834d`, regression test in `tests/test_training_kit.py`).

**Memory note for planning (from the unintended 4B run):** Qwen3-4B LoRA SFT (239 s for 10 steps), local teacher
labelling, KL, DPO and PG all fit on a 24 GB L4; TRL GRPO at batch 8 x 4 generations does not. For 4B GRPO use a
48 GB GPU (L40S, `g6e`) or smaller GRPO batches.

## R0: zero-shot readout of the base models (no training)

`python -m jevharness.training.eval_decisions --model <m> --dev data/train_v0/dev.jsonl data/train_v0/dev_when2call.jsonl.gz`

| Model | v0 dev acc (n=210) | ECE | AUROC | tool_select | stop | tool_params | When2Call dev acc (n=300) |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 37.6 | 0.51 | 0.65 | 24.2 | 24.2 | 57.0 | 7.7 |
| Qwen3-1.7B | 63.8 | 0.34 | 0.72 | 27.4 | 95.2 | 67.4 | 10.3 |
| Qwen3-4B | 72.9 | 0.26 | 0.75 | 56.5 | 85.5 | 75.6 | 46.3 |

Zero-shot small models are poor tool selectors and very poorly calibrated on When2Call, which is the headroom the
SFT -> KL -> GRPO recipe targets (TRAINING_RECIPES). These dev sets are training-side splits; do not compare them with
the E1 numbers (BFCL / When2Call test).

Next on GPU (not started; GPUs stay off until then): R2 = Qwen3-4B LoRA SFT on v0 (single L40S or L4), then
calibrate and serve through `training/serve_jev.py` and run E1 + a JevGym slice with the trained decider.
