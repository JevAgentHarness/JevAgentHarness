# E5 small decision models: execution notes

Exploratory and descriptive (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-06 "Exploratory row E5" and "E5: three hosts
instead of one"). The numbers are in `summary.md` / `summary.json`; this file records only how the run went.

## Hosts and GPU time (all g5.8xlarge, one A10G 24 GB, $2.448 / h, launched from one image of the E3 host)

| host | instance | work | stage GPU-h (cap) | instance up (launch → power-off) |
|---|---|---|---|---|
| A | <instance-id> | Qwen3-1.7B; the latency session (all models + frozen 4B); the 4B calibration-state replay | 4.92 | 06:10:41 → about 11:10 (5.0 h) |
| B | <instance-id> | Qwen3-0.6B | 2.54 (3) | 06:27:47 → about 09:55 (3.5 h) |
| C | <instance-id> | MiniCPM5-1B | 2.17 (4) | 06:27:50 → about 09:33 (3.1 h) |
| total | | | 9.64 (14) | about 11.6 h, about $28 |

- Hosts B and C were sent with `CAP_H` 4.5 / 5.5: the driver's cap check always reserves 1.5 h for a latency stage those
  hosts do not run, so the effective limit for their model was the registered 3 / 4 h.
- B and C's first send (06:32) stopped in `e5_inputs` after 7 s (the driver looked for the shared inputs under the part
  prefix; fixed in 46b8051) and both hosts sat idle until 07:23, when they were relaunched. This idle time is in the
  instance hours, not in the stage GPU hours.
- Host A was told by hand (06:32) that the 0.6B and MiniCPM5-1B slots were done there (`done/REMOTE_NOTE.txt`); their
  adapters and temperatures were copied from S3 onto A at about 09:57, before A's latency session (10:12), and checked
  by sha256 against the S3 copies.
- Every model fit the recipe's memory setting (3,072-token micro-batches, no activation checkpointing); no fallback
  and no backup swap was needed. Every training mix matched SFT v1's (sha256 `9fe1088d…`).
- API cost of the offline block: $0 (every replay and calibration-replay record has cost 0 and a local backend).

## Live row

- 15:00 UTC: first attempt on host A. Both servers came up (frozen 4B on :8081, MiniCPM5-1B on :8082), but the
  Gemini 3.5 Flash direct API answered HTTP 503 ("high demand"); smoke episodes took 90–350 s instead of about 8 s and
  one errored after its retries. The smoke was stopped by hand at 15:12 (development seeds 900–904 only; records kept
  under `s3://…/e5_small/runs/e5_small_smoke_attempt1`) and host A was stopped. The live row waits for the API to
  answer normally (a one-token probe every 10 min), then reruns the unchanged stages.
- 15:32–17:46 UTC: one-token probes of the direct API every 10 min found no healthy round (HTTP 503 or 9–16 s for a
  one-word answer); the OpenRouter route to the same model answered 5 of 5 in 0.7–1.5 s at 17:47. By the Deviations
  line of about 18:00 UTC the live row's System-2 is Gemini 3.5 Flash through OpenRouter for every arm (key from SSM on
  the host). Dollar figures of the live row are within-E5 only (this route reports about half the direct API's input
  tokens); success and time comparisons are within E5 (one route for every arm). The driver pauses the row if an
  episode errors or more than 4 LLM calls need a retry (`e5_box.py health`).
- 18:04 UTC: live row started after a clean smoke (15 episodes, no error, no retry). 18:05: paused by the watchdog after
  one `e5_final` episode errored: the MiniCPM5-1B server ran out of GPU memory (the frozen 4B's server on the same A10G
  held about 18 GB of session cache). Both servers now run with `--cache-tokens 16384` (Deviations about 18:10 UTC);
  the errored episode is re-run by the registered passes. While diagnosing, the success field of the first five
  episodes was printed by mistake (recorded in the same Deviations line; nothing changed on that basis).
- 19:09 UTC: OpenRouter refused every call with HTTP 403 (the organisation's monthly OpenRouter budget was exhausted);
  the watchdog paused the row with tasks 300700–300789 complete in all three arms (no route or System-1 error and no
  retried LLM call before that; logged spend $3.18). Tasks 300790–300799 (30 episodes) were not run; the 4 episodes
  that errored on the 403 are excluded as not run (Deviations about 19:30 UTC). The analysis uses the 90 tasks
  complete in every arm. Host A was stopped at 19:10.
