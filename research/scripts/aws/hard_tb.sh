#!/usr/bin/env bash
# H5 (exploratory) on Terminal-Bench 2.0 and Terminal-Bench-Science: pi 0.99.2 + the Jev extensions through Harbor
# at git 4f848e3 (adapters/pi/run_tb2.sh, run_tbsci.sh), on a hard-task box prepared by hard_setup.sh. System-2 Gemini 3.5 Flash
# (--thinking minimal); System-1 real Jev via OpenRouter behind the host decision server (jevharness.decision.server,
# {"type":"jev","provider":"openrouter"}, name jev-1.13; the containers reach it as host.docker.internal:8090).
#   PHASE=smoke_tb2     2 TB2 tasks (SMOKE_TB2) x baseline, jev_stop, jev_gate_stop, k = 1        -> runs/hard_smoke/tb2
#   PHASE=smoke_tbsci   2 TB-science tasks outside the hard pool (SMOKE_TBSCI) x oracle, nop, baseline,
#                       jev_stop_reflect, jev_gate_stop, k = 1, registered caps (1 h, 100 turns)  -> runs/hard_smoke/tbsci
#   PHASE=tbsci_cal     A6.22 (d) 2 calibration pass on the 37-task pool: oracle then nop (one process) next to one
#                       baseline attempt per task (another process); no automatic retries           -> runs/tbsci_cal
#   PHASE=tbsci_cal_floor   only its API-free stages (oracle, then nop) -> runs/tbsci_cal
#   PHASE=tbsci_cal_signal  only its baseline stage: one attempt per pool task that passed validity and has a floor
#                       (after tbsci_cal_floor; same pass, same directory)                          -> runs/tbsci_cal
#   PHASE=tbsci_select  `tbsci.py select` -> adapters/pi/tbsci_hard.txt (+ select.json), synced to the bucket
#   PHASE=tbsci_eval    nop warm-up (k = 1, own job) then baseline, jev_stop_reflect, jev_gate_stop in parallel on
#                       the frozen list, k = 3, errored trials re-run up to twice by Harbor            -> runs/tbsci
#   PHASE=tbsci_eval_pass PASS=<p>  the same evaluation, one attempt per arm per pass (budget-gated; see the phase)
#                                                                                                -> runs/tbsci/pass<p>
#   PHASE=tb2_full      the 89 TB2 tasks x baseline, jev_stop, jev_gate_stop in parallel, k = 3, each task's own
#                       agent timeout, no turn cap, errored trials re-run up to twice by Harbor        -> runs/tb2
# Retries (A6.22 (e) 4): Harbor `--max-retries 2 --retry-exclude AgentTimeoutError` (every exception except the agent
# timeout is an infrastructure error, A6.22 (e) 1); Harbor deletes the errored attempt's directory and job.log keeps
# the "Retrying" line. Calibration keeps errored trials (tbsci.py select needs them): retry those by hand once.
# Arms of one suite run as parallel Harbor jobs so they share the box and the API over the same period.
# Keys come from SSM Parameter Store into this process's environment only (Harbor hands GEMINI_API_KEY to pi through
# its own environment resolution, never through --ae or --env-file, so no key reaches a job's config files).
# Run detached:  nohup setsid env PHASE=tb2_full bash hard_tb.sh > /opt/jev/hard_status/tb_tb2_full.log 2>&1 < /dev/null &
set -uo pipefail
PHASE=${PHASE:?PHASE=smoke_tb2|smoke_tbsci|tbsci_cal|tbsci_cal_floor|tbsci_cal_signal|tbsci_select|tbsci_eval|tbsci_eval_pass|tb2_full}
STOP_WHEN_DONE=${STOP_WHEN_DONE:-0}   # 1: power the box off (shutdown behaviour = stop) after the final sync
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}; B="s3://$BUCKET"; REGION=${AWS_REGION:-us-east-1}
BOX=${BOX:-tb}
J=/opt/jev; R=$J/JevAgentHarness/research; S=$J/hard_status; mkdir -p "$S"
export HARBOR=${HARBOR:-$J/harbor-venv/bin/harbor}
export JEV_WIRE_MODEL=jev-1.13
SMOKE_TB2=${SMOKE_TB2:-"*prove-plus-comm *overfull-hbox"}
SMOKE_TBSCI=${SMOKE_TBSCI:-}
N_TB2=${N_TB2:-6}            # concurrent trials per TB2 arm (3 arms in parallel)
N_TBSCI=${N_TBSCI:-3}        # concurrent trials per TB-science arm (3 arms in parallel; tasks ask for <= 4 CPUs each)
N_CAL=${N_CAL:-4}            # concurrent trials per calibration process (2 processes)
RETRY="--max-retries 2 --retry-exclude AgentTimeoutError"
cd "$R" || exit 1
export PATH="$HOME/.local/bin:$PATH"
st() { echo "$(date -u +%FT%TZ) [tb:$PHASE] $*" | tee -a "$S/tb.txt"
       aws s3 sync "$S" "$B/hard/$BOX/status" --only-show-errors --region "$REGION" || true; }
sync_dir() { aws s3 sync "$1" "$B/hard/$BOX/runs/$2" --only-show-errors --region "$REGION" || true; }

for v in GEMINI_API_KEY OPENROUTER_API_KEY; do
  val=$(aws ssm get-parameter --region "$REGION" --name "${SSM_PREFIX:?set SSM_PREFIX}/$v" --with-decryption --query Parameter.Value \
        --output text 2>/dev/null) || val=""
  [ -n "$val" ] && [ "$val" != None ] && export "$v=$val"
done
unset val
[ -n "${GEMINI_API_KEY:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ] || { st "refusing: GEMINI_API_KEY and OPENROUTER_API_KEY needed"; exit 2; }
or_usage() { curl -s -m 20 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
             | jq -r '.data.usage // "?"' 2>/dev/null; }
leak_check() {  # files under $1 that contain a key value (count only; values never printed)
  local n=0 k
  for k in "$GEMINI_API_KEY" "$OPENROUTER_API_KEY"; do n=$((n + $(grep -rlF -- "$k" "$1" 2>/dev/null | wc -l))); done
  echo "$n"
}
ensure_server() {
  if ! curl -s --noproxy '*' -m 5 http://127.0.0.1:8090/health > /dev/null; then
    mkdir -p cache
    nohup setsid .venv/bin/python -m jevharness.decision.server --host 0.0.0.0 --port 8090 --name jev-1.13 \
      --cache "cache/s1_pi_jev_$BOX.sqlite" --s1 '{"type":"jev","provider":"openrouter"}' \
      >> "$S/decision_server_$BOX.log" 2>&1 < /dev/null &
    for _ in $(seq 1 30); do curl -s --noproxy '*' -m 2 http://127.0.0.1:8090/health > /dev/null && break; sleep 2; done
  fi
  st "decision server: $(curl -s --noproxy '*' -m 5 http://127.0.0.1:8090/health || echo DOWN)"
}
# one Harbor job per arm, all in parallel; each in its own log; waits for all
run_arms() {  # run_arms <script> <out> <arms...>   (env: N_CONCURRENT, N_ATTEMPTS, SUBSET/INCLUDE, HARBOR_ARGS ...)
  local script=$1 out=$2; shift 2
  local pids=() a
  for a in "$@"; do
    ( OUT="$out" ENV_FILE=/nonexistent bash "adapters/pi/$script" "$a" > "$out/run_$a.log" 2>&1; echo "exit=$?" >> "$out/run_$a.log" ) &
    pids+=($!); sleep 15
  done
  for a in "${pids[@]}"; do wait "$a"; done
}
summ() {  # summ <out> <name> <arms...>: summarize_harbor (full set + complete case), leak check, sync
  local out=$1 name=$2; shift 2
  local dirs=() a
  for a in "$@"; do [ -d "$out/$a" ] && dirs+=("$out/$a"); done
  .venv/bin/python adapters/pi/summarize_harbor.py "${dirs[@]}" --s1-price-per-m 0.042 \
    ${BASE:+--baseline "$BASE"} --budgets 600,1800,3600 --json "$out/summary.json" > "$out/summary.md" 2> "$out/summary.err"
  st "$name summary exit=$? -> $out/summary.md; leak check: $(leak_check "$out") file(s) contain a key value"
  for a in "$@"; do
    [ -d "$out/$a" ] || continue
    st "$name/$a: trials=$(ls -d "$out/$a"/*/ 2>/dev/null | wc -l) $(tail -1 "$out/run_$a.log" 2>/dev/null)" \
       "exceptions=[$(jq -r '.exception_info.exception_type // empty' "$out/$a"/*/result.json 2>/dev/null | sort | uniq -c | tr -s ' ' | tr '\n' ',')]" \
       "retries_logged=$(grep -c 'Retrying in' "$out/$a/job.log" 2>/dev/null)"
  done
  sync_dir "$out" "$name"
}

st "start $(git log --oneline -1) openrouter_usage=$(or_usage) disk $(df -BG / | awk 'NR==2{print $3"/"$2}')"
case "$PHASE" in
  smoke_tb2)
    ensure_server; out=$R/runs/hard_smoke/tb2; mkdir -p "$out"
    N_CONCURRENT=2 N_ATTEMPTS=1 INCLUDE="$SMOKE_TB2" run_arms run_tb2.sh "$out" baseline jev_stop jev_gate_stop
    BASE=baseline summ "$out" hard_smoke_tb2 baseline jev_stop jev_gate_stop ;;
  smoke_tbsci)
    [ -n "$SMOKE_TBSCI" ] || { st "SMOKE_TBSCI (2 slugs outside the pool) is required"; exit 2; }
    ensure_server; out=$R/runs/hard_smoke/tbsci; mkdir -p "$out"
    inc=""; for s in $SMOKE_TBSCI; do inc+="${inc:+ }terminal-bench-science/$s"; done
    N_CONCURRENT=2 N_ATTEMPTS=1 INCLUDE="$inc" run_arms run_tbsci.sh "$out" oracle nop
    N_CONCURRENT=2 N_ATTEMPTS=1 INCLUDE="$inc" run_arms run_tbsci.sh "$out" baseline jev_stop_reflect jev_gate_stop
    BASE=baseline summ "$out" hard_smoke_tbsci oracle nop baseline jev_stop_reflect jev_gate_stop ;;
  tbsci_cal|tbsci_cal_floor|tbsci_cal_signal)
    out=$R/runs/tbsci_cal; mkdir -p "$out"
    [ "$(sha256sum adapters/pi/tbsci_hard_pool.txt | cut -d' ' -f1)" = f131ee5289003d95facdb055bb308a18a9d68280ddabb2aeb1c9fd122c9261fe ] \
      || { st "refusing: the pool's sha256 is not the registered one"; exit 2; }
    pids=()
    if [ "$PHASE" != tbsci_cal_signal ]; then
      ( OUT="$out" ENV_FILE=/nonexistent SUBSET=hard_pool N_CONCURRENT=$N_CAL N_ATTEMPTS=1 \
          bash adapters/pi/run_tbsci.sh oracle > "$out/run_oracle.log" 2>&1; echo "exit=$?" >> "$out/run_oracle.log"
        OUT="$out" ENV_FILE=/nonexistent SUBSET=hard_pool N_CONCURRENT=$N_CAL N_ATTEMPTS=1 \
          bash adapters/pi/run_tbsci.sh nop > "$out/run_nop.log" 2>&1; echo "exit=$?" >> "$out/run_nop.log" ) &
      pids+=($!); sleep 30
    fi
    cal_subset=hard_pool
    if [ "$PHASE" = tbsci_cal_signal ]; then
      # README "one capped baseline attempt per remaining task": only the tasks that passed validity and have a floor,
      # i.e. those `select` leaves pending at the signal stage (it never writes the list while any task is pending)
      .venv/bin/python adapters/pi/tbsci.py select "$out" --clone "$J/tbsci" --json "$out/pre_signal.json" \
        --out "$out/pre_signal_list_unused.txt" > "$out/pre_signal.txt" 2>&1
      rm -f "$out/pre_signal_list_unused.txt"
      .venv/bin/python -c "import json,sys; d=json.load(open(sys.argv[1])); print('\n'.join(sorted(t['slug'] for t in d['tasks'].values() if t.get('stage') == 'signal' and t.get('status') == 'pending')))" \
        "$out/pre_signal.json" > "$out/signal_tasks.txt"
      st "signal stage: $(grep -c . "$out/signal_tasks.txt") task(s) passed validity + floor; other pre-signal statuses:" \
         "$(.venv/bin/python -c "import json,sys,collections; d=json.load(open(sys.argv[1])); print(dict(collections.Counter((t.get('stage'),t.get('status')) for t in d['tasks'].values())))" "$out/pre_signal.json")"
      [ -s "$out/signal_tasks.txt" ] || { st "no task to attempt"; exit 1; }
      cal_subset="$out/signal_tasks.txt"
    fi
    if [ "$PHASE" != tbsci_cal_floor ]; then
      ensure_server
      ( OUT="$out" ENV_FILE=/nonexistent SUBSET="$cal_subset" N_CONCURRENT=$N_CAL N_ATTEMPTS=1 \
          bash adapters/pi/run_tbsci.sh baseline > "$out/run_baseline.log" 2>&1; echo "exit=$?" >> "$out/run_baseline.log" ) &
      pids+=($!)
    fi
    ( while sleep 900; do sync_dir "$out" tbsci_cal; done ) & SY=$!
    for p in "${pids[@]}"; do wait "$p"; done; kill $SY 2>/dev/null
    summ "$out" tbsci_cal oracle nop baseline ;;
  tbsci_select)
    out=$R/runs/tbsci_cal
    .venv/bin/python adapters/pi/tbsci.py select "$out" --clone "$J/tbsci" --run-id "tbsci_cal@$(hostname -s)" \
      --json "$out/select.json" > "$out/select.txt" 2>&1
    rc=$?; st "select exit=$rc: $(tail -1 "$out/select.txt")"
    [ -f adapters/pi/tbsci_hard.txt ] && cp adapters/pi/tbsci_hard.txt "$out/tbsci_hard.txt"
    sync_dir "$out" tbsci_cal ;;
  tbsci_eval)
    [ -f adapters/pi/tbsci_hard.txt ] || { st "refusing: adapters/pi/tbsci_hard.txt is missing (run tbsci_select)"; exit 2; }
    ensure_server; out=$R/runs/tbsci; mkdir -p "$out" "$R/runs/tbsci_warm"
    st "frozen list sha256 $(sha256sum adapters/pi/tbsci_hard.txt | cut -d' ' -f1), $(wc -c < adapters/pi/tbsci_hard.txt) bytes"
    OUT="$R/runs/tbsci_warm" ENV_FILE=/nonexistent SUBSET=hard N_CONCURRENT=6 N_ATTEMPTS=1 \
      bash adapters/pi/run_tbsci.sh nop > "$R/runs/tbsci_warm/run_nop.log" 2>&1
    st "nop warm-up exit=$?"
    ( while sleep 900; do sync_dir "$out" tbsci; done ) & SY=$!
    SUBSET=hard N_CONCURRENT=$N_TBSCI N_ATTEMPTS=3 HARBOR_ARGS="$RETRY" \
      run_arms run_tbsci.sh "$out" baseline jev_stop_reflect jev_gate_stop
    kill $SY 2>/dev/null
    BASE=baseline summ "$out" tbsci baseline jev_stop_reflect jev_gate_stop ;;
  tbsci_eval_pass)
    # Budget-gated form of tbsci_eval: the same 3 arms x frozen list x k = 3, run as one attempt per arm per pass
    # (PASS=1, 2, 3; Harbor's own -k 3 also runs attempt-major: every task's attempt 1, then attempt 2, ...), so that
    # a spend cap can stop between passes instead of cancelling trials. Pass p of arm a is the Harbor job
    # runs/tbsci/pass<p>/<a>; summarize_harbor groups trials by job name, so the summary merges the passes (attempts
    # per task = passes run). Same caps, arms, retries (Harbor --max-retries 2) and concurrency as tbsci_eval.
    PASS=${PASS:?PASS=1|2|3}
    [ -f adapters/pi/tbsci_hard.txt ] || { st "refusing: adapters/pi/tbsci_hard.txt is missing (run tbsci_select)"; exit 2; }
    ensure_server; out=$R/runs/tbsci/pass$PASS; mkdir -p "$out" "$R/runs/tbsci_warm"
    st "pass $PASS; frozen list sha256 $(sha256sum adapters/pi/tbsci_hard.txt | cut -d' ' -f1), $(wc -c < adapters/pi/tbsci_hard.txt) bytes"
    if [ ! -d "$R/runs/tbsci_warm/nop" ]; then
      OUT="$R/runs/tbsci_warm" ENV_FILE=/nonexistent SUBSET=hard N_CONCURRENT=6 N_ATTEMPTS=1 \
        bash adapters/pi/run_tbsci.sh nop > "$R/runs/tbsci_warm/run_nop.log" 2>&1
      st "nop warm-up exit=$?"
    fi
    ( while sleep 900; do sync_dir "$R/runs/tbsci" tbsci; done ) & SY=$!
    SUBSET=hard N_CONCURRENT=$N_TBSCI N_ATTEMPTS=1 HARBOR_ARGS="$RETRY" \
      run_arms run_tbsci.sh "$out" baseline jev_stop_reflect jev_gate_stop
    kill $SY 2>/dev/null
    for a in baseline jev_stop_reflect jev_gate_stop; do
      st "tbsci pass $PASS/$a: trials=$(ls -d "$out/$a"/*/ 2>/dev/null | wc -l) $(tail -1 "$out/run_$a.log" 2>/dev/null)" \
         "exceptions=[$(jq -r '.exception_info.exception_type // empty' "$out/$a"/*/result.json 2>/dev/null | sort | uniq -c | tr -s ' ' | tr '\n' ',')]" \
         "retries_logged=$(grep -c 'Retrying in' "$out/$a/job.log" 2>/dev/null)" \
         "cost=\$$(cat "$out/$a"/*/result.json 2>/dev/null | jq -s '[.[].agent_result.cost_usd // 0] | add // 0')"
    done
    o=$R/runs/tbsci; dirs=(); for a in baseline jev_stop_reflect jev_gate_stop; do dirs+=("$o"/pass*/"$a"); done
    .venv/bin/python adapters/pi/summarize_harbor.py "${dirs[@]}" --s1-price-per-m 0.042 --baseline baseline \
      --budgets 600,1800,3600 --json "$o/summary.json" > "$o/summary.md" 2> "$o/summary.err"; rc=$?
    st "tbsci summary over pass(es) $(ls -d "$o"/pass* | xargs -n1 basename | tr '\n' ' ')exit=$rc -> $o/summary.md;" \
       "leak check: $(leak_check "$o") file(s) contain a key value"
    sync_dir "$o" tbsci ;;
  tb2_full)
    ensure_server; out=$R/runs/tb2; mkdir -p "$out"
    ( while sleep 900; do sync_dir "$out" tb2; done ) & SY=$!
    N_CONCURRENT=$N_TB2 N_ATTEMPTS=3 HARBOR_ARGS="$RETRY" run_arms run_tb2.sh "$out" baseline jev_stop jev_gate_stop
    kill $SY 2>/dev/null
    BASE=baseline summ "$out" tb2 baseline jev_stop jev_gate_stop ;;
  *) st "unknown PHASE"; exit 2 ;;
esac
st "done; openrouter_usage=$(or_usage); disk $(df -BG / | awk 'NR==2{print $3"/"$2}')"
if [ "$STOP_WHEN_DONE" = 1 ]; then st "stopping the box (STOP_WHEN_DONE=1)"; sudo shutdown -h +1 "jev-hard: $PHASE done"; fi
