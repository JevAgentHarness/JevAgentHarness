#!/usr/bin/env bash
# Exploratory row E4 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-05) on the hard-task SWE box (hard_setup.sh):
# JevAgentHarness on mini-SWE-agent 2.4.6, per-step routing between DeepSeek V4 Pro (reasoning on) and V4 Flash
# (reasoning off) through OpenRouter, Jev via OpenRouter as the decision model. Runs unattended: every stage logs to
# $S/e4.txt and syncs status + outputs to S3; a spend watchdog and an idle stop guard run beside it.
#   PHASE=smoke bash scripts/aws/e4_swe.sh   # E4 smoke: the H5 smoke pair x 3 arms
#   PHASE=dev   bash scripts/aws/e4_swe.sh   # 45 dev instances x 5 arms -> threshold.json (the registered rule)
#   PHASE=eval  bash scripts/aws/e4_swe.sh   # 100 eval instances x 3 arms (refuses without the committed threshold)
# Every phase: pre-pull images, agent runs (one shared task-major pool), ONE re-run of infrastructure-errored
# instances, local SWE-bench evaluation (swebench 4.1.0), collect, summary, leak check, S3 sync; STOP_WHEN_DONE=1 powers
# the box off at the end. Keys come from SSM Parameter Store into this process's environment only.
# Detached: nohup setsid env PHASE=dev bash scripts/aws/e4_swe.sh > /opt/jev/hard_status/e4_dev.log 2>&1 < /dev/null &
# Local mode (E4_LOCAL=1; handoff/experiments/e4r/run.sh, a collaborator's Linux machine with Docker): the repo and status
# directories come from E4_REPO / E4_STATUS_DIR, the key from the environment (OPENROUTER_API_KEY; no SSM), and there is
# no S3 sync, no power-off or idle guard and no shutdown. E4_PREPULL=0 skips the pre-pull (images are pulled on demand by
# `docker run`); E4_PULL_MIRROR=mirror.gcr.io pulls through that registry and tags the image with its Docker Hub name.
set -uo pipefail
PHASE=${PHASE:?PHASE=smoke|dev|eval}
EXP=${EXP:-e4}                          # e4 (DeepSeek via OpenRouter) | e4b (Gemini 3.8 Flash, direct API; e4b_chain.sh)
                                        # | e4r (E4's arms again on its own box; scripts/aws/e4r_swe.sh)
PY_EXP="--exp $EXP"
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}; B="s3://$BUCKET"; REGION=${AWS_REGION:-us-east-1}
LOCAL=${E4_LOCAL:-0}
R=${E4_REPO:-/opt/jev/JevAgentHarness/research}; S=${E4_STATUS_DIR:-/opt/jev/hard_status}; mkdir -p "$S"
# status prefix in S3: E4 / E4b share e4/status; E4r (its own box) keeps its own, so E4's log there is never overwritten
case "$EXP" in e4r) STATUS_S3=${STATUS_S3:-$B/e4r/status} ;; *) STATUS_S3=${STATUS_S3:-$B/e4/status} ;; esac
export STATUS_S3
OUT=${OUT:-$R/runs/${EXP}_swe/$PHASE}
case "$PHASE" in
  smoke) W=${WORKERS:-6};  EW=${EVAL_WORKERS:-2}; PHASE_COST_LIMIT=${PHASE_COST_LIMIT:-6} ;;
  dev)   W=${WORKERS:-25}; EW=${EVAL_WORKERS:-4}; PHASE_COST_LIMIT=${PHASE_COST_LIMIT:-60} ;;
  eval)  W=${WORKERS:-24}; EW=${EVAL_WORKERS:-5}; PHASE_COST_LIMIT=${PHASE_COST_LIMIT:-80} ;;
  *) echo "unknown PHASE $PHASE" >&2; exit 2 ;;
esac
CAP_USAGE=${CAP_USAGE:?absolute OpenRouter key usage (USD) at which the watchdog stops E4 (start usage + cap)}
GUARD_MIN=${GUARD_MIN:-720}
cd "$R" || exit 1
export PATH="$HOME/.local/bin:$PATH"
st() { echo "$(date -u +%FT%TZ) [$EXP:$PHASE] $*" | tee -a "$S/e4.txt"
       [ "$LOCAL" = 1 ] || aws s3 sync "$S" "$STATUS_S3" --only-show-errors --region "$REGION" --exclude '*' --include 'e4*' || true; }
sync_out() { [ "$LOCAL" = 1 ] && return 0
             aws s3 sync "$OUT" "$B/$EXP/runs/$PHASE" --only-show-errors --region "$REGION" --exclude '*.sqlite' || true; }

if [ "$LOCAL" != 1 ]; then
  val=$(aws ssm get-parameter --region "$REGION" --name ${SSM_PREFIX:?set SSM_PREFIX}/OPENROUTER_API_KEY --with-decryption \
        --query Parameter.Value --output text 2>/dev/null) || val=""
  [ -n "$val" ] && [ "$val" != None ] && export OPENROUTER_API_KEY="$val"
  unset val
fi
[ -n "${OPENROUTER_API_KEY:-}" ] || { st "refusing: OPENROUTER_API_KEY missing"; exit 2; }
if [ "$EXP" = e4b ]; then
  val=$(aws ssm get-parameter --region "$REGION" --name ${SSM_PREFIX:?set SSM_PREFIX}/GEMINI_API_KEY --with-decryption \
        --query Parameter.Value --output text 2>/dev/null) || val=""
  [ -n "$val" ] && [ "$val" != None ] && export GEMINI_API_KEY="$val"
  unset val
  [ -n "${GEMINI_API_KEY:-}" ] || { st "refusing: GEMINI_API_KEY missing"; exit 2; }
fi
# the key goes to curl and grep through stdin / a file descriptor, never through argv
or_usage() { printf 'Authorization: Bearer %s\n' "$OPENROUTER_API_KEY" | curl -s -m 20 -H @- https://openrouter.ai/api/v1/key \
             | jq -r '.data.usage // "?"' 2>/dev/null; }
leak_check() { local n; n=$(grep -rlF -f <(printf '%s\n' "$OPENROUTER_API_KEY") "$OUT" 2>/dev/null | wc -l)
  [ -n "${OPENROUTER_API_KEY_JEV:-}" ] && n=$((n + $(grep -rlF -f <(printf '%s\n' "$OPENROUTER_API_KEY_JEV") "$OUT" 2>/dev/null | wc -l)))
  [ -n "${GEMINI_API_KEY:-}" ] && n=$((n + $(grep -rlF -f <(printf '%s\n' "$GEMINI_API_KEY") "$OUT" 2>/dev/null | wc -l))); echo "$n"; }

# power-off guard (re-armed per phase) and the idle stop guard (no E4 process and no container for 30 min -> off)
if [ "$LOCAL" = 1 ]; then STOP_WHEN_DONE=0; else
  [ "${NO_GUARD_REARM:-0}" = 1 ] || { sudo shutdown -c 2>/dev/null; sudo shutdown -h +"$GUARD_MIN" "E4 guard" 2>/dev/null; }
fi
PIDF="$S/${EXP}_${PHASE}.pid"; echo $$ > "$PIDF"
[ "$LOCAL" = 1 ] || pgrep -f e4_idle_guard.sh >/dev/null || nohup setsid bash scripts/aws/e4_idle_guard.sh > /dev/null 2>&1 < /dev/null &

mkdir -p "$OUT"
U0=$(or_usage)
st "start $(git log --oneline -1 | cut -c1-60) workers=$W eval_workers=$EW out=$OUT openrouter_usage=$U0 cap_usage=$CAP_USAGE phase_llm_limit=\$$PHASE_COST_LIMIT"
case "$U0" in ''|'?') st "refusing: OpenRouter usage unreadable"; exit 2 ;; esac
if awk -v u="$U0" -v c="$CAP_USAGE" 'BEGIN{exit !(u + 0 >= c - 2)}'; then st "refusing: usage $U0 within \$2 of the cap $CAP_USAGE"; exit 3; fi

IDSF=configs/explore/${EXP}_swe/${PHASE}_ids.txt
st "instances: $(wc -l < "$IDSF") (sha256 $(sha256sum "$IDSF" | cut -c1-16))"
if [ "$PHASE" = eval ]; then
  .venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); import explore_e4_swe as E; E.set_exp('$EXP'); \
print('tau', E.check_eval_gate(E.IDS_DIR / 'threshold.json', E.IDS_DIR / 'eval_ids.txt'))" > "$S/e4_gate.txt" 2>&1 \
    || { st "refusing the eval phase: $(tail -1 "$S/e4_gate.txt")"; exit 4; }
  st "eval gate: $(cat "$S/e4_gate.txt")"
fi

# pre-pull the instance images (anonymous Docker Hub: 100 pulls/h/IP; mirror.gcr.io first, see hard_setup.sh)
HUB_RESERVE=${HUB_RESERVE:-10}; export HUB_RESERVE
hub_remaining() { local tok
  tok=$(curl -s -m 20 "https://auth.docker.io/token?service=registry.docker.io&scope=repository:ratelimitpreview/test:pull" \
        | jq -r .token 2>/dev/null)
  curl -s -m 20 --head -H "Authorization: Bearer $tok" https://registry-1.docker.io/v2/ratelimitpreview/test/manifests/latest \
    | tr -d '\r' | awk -F'[ ;]' 'tolower($1)=="ratelimit-remaining:"{print $2}'; }
pull() { local img="docker.io/swebench/sweb.eval.x86_64.$(echo "${1//__/_1776_}" | tr 'A-Z' 'a-z'):latest" i r
         docker image inspect "$img" >/dev/null 2>&1 && { echo "CACHED $img"; return 0; }
         if [ -n "${E4_PULL_MIRROR:-}" ] && docker pull -q "$E4_PULL_MIRROR/${img#docker.io/}" >/dev/null 2>&1; then
           docker tag "$E4_PULL_MIRROR/${img#docker.io/}" "$img" && docker rmi "$E4_PULL_MIRROR/${img#docker.io/}" >/dev/null 2>&1
           echo "PULLED $img (via $E4_PULL_MIRROR)"; return 0; fi
         for i in $(seq 1 30); do r=$(hub_remaining); { [ -z "$r" ] || [ "$r" -ge "$HUB_RESERVE" ]; } && break
           echo "WAIT hub_remaining=$r $img"; sleep 120; done
         for i in 1 2 3 4 5; do docker pull -q "$img" >/dev/null 2>&1 && { echo "PULLED $img"; return 0; }; sleep $((i * 20)); done
         echo "PULL FAILED $img"; }
export -f pull hub_remaining
export E4_PULL_MIRROR=${E4_PULL_MIRROR:-}
t0=$(date +%s)
if [ "${E4_PREPULL:-1}" = 1 ]; then
  sed 's/#.*//; /^\s*$/d' "$IDSF" | xargs -P 6 -I{} bash -c 'pull "$@"' _ {} > "$S/e4_pull_$PHASE.txt" 2>&1
else
  echo "pre-pull skipped (E4_PREPULL=0): images are pulled on demand" > "$S/e4_pull_$PHASE.txt"
fi
st "pre-pull done in $(( $(date +%s) - t0 )) s; cached $(grep -c ^CACHED "$S/e4_pull_$PHASE.txt"), pulled $(grep -c ^PULLED "$S/e4_pull_$PHASE.txt"), failures $(grep -c 'PULL FAILED' "$S/e4_pull_$PHASE.txt"); disk $(df -BG / | awk 'NR==2{print $3"/"$2}')"
# (no background pre-pull of the next phase's images: on the dev launch, pulling beside 25 container starts saturated
# the gp3 volume, ~125 MB/s, and held every `docker run` for ~10 min; each phase pre-pulls its own images first)
# spend watchdog: stops the runner if the key's usage (shared key: includes any other spend) reaches the cap
( while sleep 60; do u=$(or_usage); case "$u" in ''|'?') continue ;; esac
    if awk -v u="$u" -v c="$CAP_USAGE" 'BEGIN{exit !(u + 0 >= c - 1)}'; then
      st "WATCHDOG: OpenRouter usage $u reached cap $CAP_USAGE - 1: stopping the runner"
      pkill -f "explore_e4_swe.py $PY_EXP run --phase $PHASE"; fi
  done ) & WD=$!
( while sleep 600; do sync_out; done ) & SYNC=$!

run() {  # run [extra flags]
  MSWEA_GLOBAL_COST_LIMIT=$PHASE_COST_LIMIT MSWEA_SILENT_STARTUP=1 .venv/bin/python scripts/explore_e4_swe.py $PY_EXP run \
    --phase "$PHASE" -o "$OUT" -w "$W" -c environment.pull_timeout=900 ${RUN_FLAGS:-} "$@" >> "$OUT/run.$PHASE.log" 2>&1
  echo $?
}
st "agent runs: $(run)"
nerr=$(.venv/bin/python - "$OUT" <<'PY'
import glob, json, sys
print(sum(bool(json.load(open(f)).get("error")) for f in glob.glob(f"{sys.argv[1]}/*/*/*.jev.json")))
PY
)
st "errored records after the main pass: $nerr"
if [ "$nerr" -gt 0 ]; then st "re-run of errored instances (once): $(run --rerun-errored)"; fi
ARMS=$(.venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); import explore_e4_swe as E; E.set_exp('$EXP'); print(' '.join(E.PHASES['$PHASE']['arms']))")
for arm in $ARMS; do
  cnt=$(.venv/bin/python - "$OUT/$arm" <<'PY'
import glob, json, sys
recs = [json.load(open(f)) for f in glob.glob(f"{sys.argv[1]}/*/*.jev.json")]
print(f"instances={len(recs)} errors={sum(bool(r.get('error')) for r in recs)} cost=${sum(float(((r.get('usage') or {}).get('total') or {}).get('cost_usd') or 0) for r in recs):.3f}")
PY
)
  st "$arm: $cnt"
done
kill "$WD" 2>/dev/null

# local SWE-bench evaluation, arms in parallel (EW containers each), then collect
epids=()
for arm in $ARMS; do
  ( d="$OUT/$arm"; mkdir -p "$d/eval"
    cd "$d/eval" && "$R/.venv-eval/bin/python" -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified \
      --split test --predictions_path "$d/preds.json" --max_workers "$EW" --run_id "${EXP}_${PHASE}_$arm" --report_dir "$d/eval" \
      > "$d/eval/eval.log" 2>&1
    rep=$(ls "$d"/eval/*."${EXP}_${PHASE}_$arm".json 2>/dev/null | head -1)
    cd "$R" && .venv/bin/python -m jevharness.adapters.mini_swe collect "$d" ${rep:+--report "$rep"} -o "$OUT/$arm.jsonl" \
      > "$d/collect.log" 2>&1 ) &
  epids+=($!)
done
for p in "${epids[@]}"; do wait "$p"; done
for arm in $ARMS; do st "$arm eval: $(ls "$OUT/$arm"/eval/*."${EXP}_${PHASE}_$arm".json 2>/dev/null | wc -l) report(s), $(grep -o '"resolved_instances": [0-9]*' "$OUT/$arm"/eval/*."${EXP}_${PHASE}_$arm".json 2>/dev/null | head -1)"; done
kill "$SYNC" 2>/dev/null

.venv/bin/python scripts/explore_e4_swe.py $PY_EXP summarize "$OUT" --phase "$PHASE" --out-dir "$OUT/summary" \
  > "$OUT/summary.log" 2>&1; st "summary exit=$?"
if [ "$PHASE" = dev ]; then
  .venv/bin/python scripts/explore_e4_swe.py $PY_EXP threshold "$OUT" > "$OUT/threshold.log" 2>&1
  st "threshold exit=$? $(.venv/bin/python -c "import json; d=json.load(open('$OUT/threshold.json')); print('tau', d['tau'], d['case'], d['resolved'], {k: round(v, 3) if v is not None else None for k, v in d['fast_share'].items()})" 2>&1 | tail -1)"
fi
st "leak check: $(leak_check) file(s) under $OUT contain a key value"
sync_out
st "done; openrouter_usage=$(or_usage) (phase start $U0); disk $(df -BG / | awk 'NR==2{print $3"/"$2}')"
rm -f "$PIDF"
if [ "${STOP_WHEN_DONE:-1}" = 1 ]; then st "stopping the box"; sudo shutdown -h +1 "E4 $PHASE done"; fi
