# StartLux-Decision GPU smoke test (A7.18), 2026-10-03

Development only, not an evaluation. Stage logs (`*.log`, gitignored) are on S3 under `startlux_smoke/`. No task success or accuracy number was computed, read or recorded; the dev run's
trajectories stay on S3 (`s3://<bucket>/startlux_smoke/dev_run/`) and are not in the repository.

## Host
- `<instance-id>`, g6e.xlarge (1x NVIDIA L40S, 46068 MiB; 4 vCPU, 30 GiB RAM), us-east-1a; Deep Learning OSS
  Nvidia Driver AMI GPU PyTorch 2.14 (Ubuntu 26.04) `<aws-resource-id>`, driver 595.91.07; SSM only
  (`jev-train-noingress`, `<instance-profile>`); Name=jev-startlux-smoke, Project=jev; shutdown behaviour stop, 4 h guard.
- Launched 02:58:30Z, stop initiated 03:58:42Z (1.00 instance-hour, $1.861/h on demand: about $1.87); state `stopped`.
  The 200 GB gp3 root volume (250 MB/s) persists while stopped (about $0.70/day until the instance is terminated).
- Driver: `scripts/aws/startlux_smoke.sh` (launch / push / send / stop; box stages). Harness commit on the host:
  `b7f731c` (this branch). Serving env: Python 3.11.17 (uv), torch 2.14.1 (cu130), transformers 5.8.1, accelerate
  1.15.0, flash-linear-attention 0.5.2, causal-conv1d 1.7.0 (built from source against the AMI's pip CUDA 13.0 toolkit
  `/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13`, nvcc 13.0.88; about 17 min on 4 vCPU; the manifest shows
  `cuda_home: null` because the final invocations found the kernels already built), triton 3.8.0. Package
  `StartLuxLabs/startlux-decision` at `a9101e9`.

## Results per size (`startlux_manifest.json`; one GPU, so `serve_startlux.sh` ran once per size, 4B stopped first)
| size | mode | weights revision | kernels | server | smoke (3 questions / 69 options) |
|---|---|---|---|---|---|
| 4B | original weights, bf16 | `3890ae0c174215e1555cc94c882ef45c5fe87cad` | `fast kernels: active`; 40 CUDA graphs; graph self-test max diff 0.0 | started | ok / ok (69 of 69 probabilities) |
| 27B (Q8_0) | `StartLux-Decision-27B-Q8_0-GGUF` via llama.cpp + `gguf_server` (no GPU >= 64 GiB) | `a54b311e3aede0c869bf006116bcfdddb1ef466f` | n/a (llama.cpp CUDA: `ghcr.io/ggml-org/llama.cpp:server-cuda`, build 11347, commit 5fc4f3c8c, digest in `llama_image_digest.txt`) | started | ok / ok (69 of 69) |

Repeated smoke requests, one at a time after one warm-up (`probe_*.json`, informational):
- 4B: 3 questions p50 26.4 ms / p95 26.6 ms (n = 20); 69 options p50 100.1 / p95 100.3 ms (n = 10).
- 27B (Q8_0): 3 questions p50 569 ms / p95 615 ms; 69 options p50 1545 / p95 1564 ms.

## Development end-to-end check (4B)
`jevbench run configs/experiments/stage2_startlux_states.yaml` with `-o name=startlux_smoke_dev_s0 suite.seed0=0
suite.n=50 n_tasks=20 workers=1 s1_cache=null`, the `startlux_4b` Jev wire backend on `http://127.0.0.1:8090` (no
placeholder) and one arm `dev_sl4b_ts_replace_f` (tool_select replace); Flash System-2; JevGym seeds 0-19;
`JEV_EVAL_RUNS` unset. Summary (`e2e_summary.json`, written by a reader that never opens `result`):
- the pipeline runs end to end: 20 episodes, 0 episodes with an error, 130 System-1 calls (no cache), all answered by
  `StartLux-Decision-4B` (`/v1/models`, `/health`: fast kernels, 40 graphs);
- System-1 latency per decision, single concurrency, client side: p50 268.8 ms, p95 598.3 ms, mean 456.1 ms, max
  10828 ms. The two calls over 1 s (10.3 s, 10.8 s; calls 3 and 14 in time order) are first requests at a new padded
  length, whose linear-attention kernels compile on first use (vendor docs); after the first 20 calls p50 269.0 ms, p95
  598.3 ms, max 602.3 ms. Input tokens per request (all knockout prompts of the 69-option question): p50 4702, max 8415;
- API spend (ledger): $0.25.

## Script fixes (this branch)
1. `serve_startlux.sh`, Q8_0 path: waited only for `gguf_server`'s `/health`, which answers before llama-server has
   loaded the GGUF; it now waits for llama-server's own `/health` (`START_TIMEOUT_S`, 900 s). The weights download is
   pinned to the revision recorded in the manifest; the plan loop reads from fd 3.
2. Kernels were installed only with a new venv, so the failed causal-conv1d build (no nvcc on PATH on this AMI; no
   prebuilt wheel for torch 2.14) was never retried; the kernels now install whenever they do not import, and the build
   finds a CUDA toolkit of torch's CUDA major version (`/usr/local/cuda*` or the AMI's pip toolkit) unless `CUDA_HOME`
   is set.
3. `transformers>=5.8` installed 5.18.0, which no longer defines `modeling_qwen3_5.is_fast_path_available`; the
   vendor's check reads it, so it printed "NOT active" with both kernels importable and the server would refuse to
   start. Pinned to the vendor-tested 5.8.1 (`TRANSFORMERS_SPEC`; A7.18 (b) asks >= 5.8).
4. `accelerate` was missing (`from_pretrained(device_map=...)` requires it; the vendor gets it via peft); added.
5. `startlux_plan.py smoke`: each answer is now checked against its question (every option has a probability, noul in
   [0, 1], one probability per score level, sums to 1).

## Open points for the main session
- **27B (Q8_0) context per request.** The vendor's llama-server command (`-c 16384 --parallel 4`) gives each slot 4096
  tokens: a 13216-token single-question request returned HTTP 400 ("request (13216 tokens) exceeds the available
  context size (4096 tokens)" in the llama-server log, `s3://<bucket>/startlux_smoke/logs/llama_27B.log`). The 4B (transformers) answered the same request. Any StartLux-27B
  prompt above 4096 tokens (one question with the full state) would fail. Not changed here (untested); options are a
  larger `-c` (e.g. 65536 with `--parallel 4`) or `--parallel 1`, decided before the first 27B request (A7.18 (g)).
- On a one-GPU host the plan gives the GPU to the 27B (Q8_0) and does not run the 4B (one size per GPU); run the sizes
  in separate invocations (`SIZES=4B`, then `SIZES=27B`) or on separate GPUs.
- First requests at each new padded input length take about 10 s on the 4B (kernel compilation); a latency or tariff
  window should start after a warm-up that covers the lengths it will see.
