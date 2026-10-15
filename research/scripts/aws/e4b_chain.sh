#!/usr/bin/env bash
# Exploratory row E4b (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-05) on the SWE box, unattended end to end:
#   smoke (H5 smoke pair x 3 arms; may overlap the tail of E4's eval) -> cost projection gate -> wait for E4's eval
#   driver to exit (and cancel its power-off) -> dev (30 x 5 arms) -> threshold by the registered rule (no human step)
#   -> eval projection gate -> eval (100 x 3 arms) -> summary -> S3 -> power off.
# Budget: $400 of Gemini API (direct) in total, from the logged per-call costs (litellm's Gemini 3.8 Flash prices);
# each phase process also gets MSWEA_GLOBAL_COST_LIMIT = what is left, so the cap binds on the real spend in flight.
# Registered cost rule: if the smoke projection of dev + eval exceeds $300, always_fast runs on the first 50 eval
# instances only; if it exceeds $400, E4b stops after the smoke.
#   nohup setsid env E4_PID=<pid of the running E4 eval driver> bash scripts/aws/e4b_chain.sh > /opt/jev/hard_status/e4b_chain.log 2>&1 &
set -uo pipefail
R=/opt/jev/JevAgentHarness/research; S=/opt/jev/hard_status; cd "$R" || exit 1
B=s3://$BUCKET; REGION=us-east-1
BUDGET=${BUDGET:-400}; CUT_AT=${CUT_AT:-300}; SAFETY=${SAFETY:-1.5}
E4_PID=${E4_PID:-}
echo $$ > "$S/e4b_driver.pid"
st() { echo "$(date -u +%FT%TZ) [e4b:chain] $*" | tee -a "$S/e4.txt"
       aws s3 sync "$S" "$B/e4/status" --only-show-errors --region "$REGION" --exclude '*' --include 'e4*' || true; }
spent() { .venv/bin/python - <<'PY'
import glob, json
tot = 0.0
for f in glob.glob("runs/e4b_swe/*/*/*/*.jev.json") + glob.glob("runs/e4b_swe/*/*/_errored/*/*.jev.json"):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    for c in r.get("calls") or []:
        if c.get("component") == "s2":
            tot += float(c.get("cost_usd") or 0)
print(f"{tot:.4f}")
PY
}
# per-arm mean Gemini $ per instance of a finished phase directory -> projection for the given job counts
project() { .venv/bin/python - "$@" <<'PY'
import glob, json, sys
run, spec = sys.argv[1], sys.argv[2:]          # spec: arm=count (arm may be a prefix: jevagent=...)
mean = {}
for arm_dir in glob.glob(f"{run}/*/"):
    arm = arm_dir.rstrip("/").split("/")[-1]
    cs = []
    for f in glob.glob(f"{arm_dir}*/*.jev.json"):
        r = json.load(open(f))
        cs.append(sum(float(c.get("cost_usd") or 0) for c in r.get("calls") or [] if c.get("component") == "s2"))
    if cs:
        mean[arm] = sum(cs) / len(cs)
tot = 0.0
for s in spec:
    a, n = s.split("=")
    m = [v for k, v in mean.items() if k == a or k.startswith(a)]
    tot += (max(m) if m else max(mean.values())) * int(n)
print(f"{tot:.2f}")
PY
}
gt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a + 0 > b + 0)}'; }

st "start $(git log --oneline -1 | cut -c1-60); budget \$$BUDGET (Gemini, logged), cut rule at \$$CUT_AT, safety x$SAFETY; E4 eval driver pid ${E4_PID:-none}"
# keep the box up when E4's eval driver ends (it schedules a power-off 1 min after its last line)
if [ -n "$E4_PID" ]; then
  ( while kill -0 "$E4_PID" 2>/dev/null; do sleep 3; done
    sudo shutdown -c 2>/dev/null; sudo shutdown -h +720 "E4b guard" 2>/dev/null
    echo "$(date -u +%FT%TZ) [e4b:chain] E4 eval driver exited; its power-off cancelled, guard re-armed (+12 h)" >> "$S/e4.txt" ) &
  WAITER=$!
fi

# 1. smoke (overlaps the tail of E4's eval: 6 containers). SKIP_SMOKE=1: the smoke already ran (wait for its driver)
if [ "${SKIP_SMOKE:-0}" = 1 ]; then
  sp=$(cat "$S/e4b_smoke.pid" 2>/dev/null); while [ -n "$sp" ] && kill -0 "$sp" 2>/dev/null; do sleep 10; done
else
  env EXP=e4b PHASE=smoke STOP_WHEN_DONE=0 NO_GUARD_REARM=1 CAP_USAGE=245 PHASE_COST_LIMIT=20 \
    bash scripts/aws/e4_swe.sh > "$S/e4b_smoke.log" 2>&1
fi
SP=$(spent)
# the registered cut first (projection > $CUT_AT -> always_fast on the first 50 eval instances), then the $BUDGET gate on
# the run as it will be run (amendment of 2026-10-06: dev candidates tau 0.5 / 0.6)
D=30; [ "${SKIP_DEV:-0}" = 1 ] && D=0
P=$(project runs/e4b_swe/smoke always_deep=$((D + 100)) always_fast=$((D + 100)) jevagent=$((2 * D + 100)))
PS=$(awk -v p="$P" -v k="$SAFETY" 'BEGIN{printf "%.2f", p * k}')
ARM_LIMIT=""; NF=$((D + 100))
if gt "$PS" "$CUT_AT"; then
  ARM_LIMIT="--arm-limit always_fast=50"; NF=$((D + 50))
  st "registered cost rule: projection \$$PS > \$$CUT_AT -> always_fast runs on the first 50 eval instances"
  P=$(project runs/e4b_swe/smoke always_deep=$((D + 100)) always_fast=$NF jevagent=$((2 * D + 100)))
  PS=$(awk -v p="$P" -v k="$SAFETY" 'BEGIN{printf "%.2f", p * k}')
fi
echo "${ARM_LIMIT:-none}" > "$S/e4b_arm_limit.txt"
st "smoke done: logged Gemini \$$SP; projection dev + eval \$$P (x$SAFETY = \$$PS) [dev $D per arm, 2 tau; eval deep 100, fast $((NF - D)), jevagent 100]"
if gt "$(awk -v a="$SP" -v b="$PS" 'BEGIN{print a + b}')" "$BUDGET"; then
  st "STOP: projected spend over the \$$BUDGET budget; E4b ends after the smoke"; rm -f "$S/e4b_driver.pid"
  [ -n "${WAITER:-}" ] && wait "$WAITER"; sudo shutdown -h +5 "E4b stopped by its budget gate"; exit 0
fi

# 2. wait for E4's eval driver to finish (it owns the box until then; dev and eval wall-clock stay unshared)
if [ -n "${WAITER:-}" ]; then st "waiting for E4's eval driver (pid $E4_PID)"; wait "$WAITER"; fi
sudo shutdown -c 2>/dev/null; sudo shutdown -h +720 "E4b guard" 2>/dev/null

# 3. dev (the registered rule writes threshold.json; no human step). SKIP_DEV=1 (only under a registered deviation):
# no dev, eval at the registered route tau 0.5
if [ "${SKIP_DEV:-0}" = 1 ]; then
  .venv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, "scripts")
import explore_e4_swe as E
E.set_exp("e4b")
sel = json.loads((E.IDS_DIR / "selection.json").read_text())
th = {"tau": E.REGISTERED_ROUTE_TAU, "case": "no_dev", "reason": "dev skipped (registered deviation): registered route tau",
      "eval_ids_sha256": sel["eval"]["sha256"]}
(E.IDS_DIR / "threshold.json").write_text(json.dumps(th, indent=1) + "\n")
PY
  TAU=0.5; TARM=always_deep; SP=$(spent); NF=100; [ -n "$ARM_LIMIT" ] && NF=50
  P=$(project runs/e4b_swe/smoke always_deep=100 always_fast=$NF jevagent=100)
  PS=$(awk -v p="$P" -v k="$SAFETY" 'BEGIN{printf "%.2f", p * k}')
  st "dev skipped (registered deviation): tau 0.5; logged Gemini \$$SP; eval projection from the smoke \$$P (x$SAFETY = \$$PS)"
  if gt "$(awk -v a="$SP" -v b="$PS" 'BEGIN{print a + b}')" "$BUDGET"; then
    st "STOP: eval projection over the \$$BUDGET budget"; rm -f "$S/e4b_driver.pid"; sudo shutdown -h +5; exit 0
  fi
else
LEFT=$(awk -v b="$BUDGET" -v s="$(spent)" 'BEGIN{printf "%.2f", b - s}')
  env EXP=e4b PHASE=dev STOP_WHEN_DONE=0 NO_GUARD_REARM=1 CAP_USAGE=245 PHASE_COST_LIMIT="$LEFT" \
    bash scripts/aws/e4_swe.sh > "$S/e4b_dev.log" 2>&1
  if [ ! -s runs/e4b_swe/dev/threshold.json ]; then st "STOP: no dev threshold (see e4b_dev.log)"; rm -f "$S/e4b_driver.pid"; sudo shutdown -h +5; exit 1; fi
  cp runs/e4b_swe/dev/threshold.json configs/explore/e4b_swe/threshold.json
  TAU=$(.venv/bin/python -c "import json; print(json.load(open('configs/explore/e4b_swe/threshold.json'))['tau'])")
  TARM=$(.venv/bin/python -c "print('jevagent_t%03d' % round(float('$TAU') * 100))")
  SP=$(spent)
  NF=100; [ -n "$ARM_LIMIT" ] && NF=50
  P=$(project runs/e4b_swe/dev always_deep=100 always_fast=$NF "$TARM=100")
  PS=$(awk -v p="$P" -v k="$SAFETY" 'BEGIN{printf "%.2f", p * k}')
  st "dev done: tau $TAU ($(.venv/bin/python -c "import json; d=json.load(open('configs/explore/e4b_swe/threshold.json')); print(d['case'], d['resolved'])")); logged Gemini \$$SP; eval projection \$$P (x$SAFETY = \$$PS)"
  if gt "$(awk -v a="$SP" -v b="$PS" 'BEGIN{print a + b}')" "$BUDGET"; then
    st "STOP: eval projection over the \$$BUDGET budget; E4b ends after dev"; rm -f "$S/e4b_driver.pid"; sudo shutdown -h +5; exit 0
  fi
  
fi
# 4. eval
LEFT=$(awk -v b="$BUDGET" -v s="$SP" 'BEGIN{printf "%.2f", b - s}')
env EXP=e4b PHASE=eval STOP_WHEN_DONE=0 NO_GUARD_REARM=1 CAP_USAGE=245 PHASE_COST_LIMIT="$LEFT" RUN_FLAGS="$ARM_LIMIT" \
  bash scripts/aws/e4_swe.sh > "$S/e4b_eval.log" 2>&1
st "eval done; logged Gemini \$$(spent) in total (smoke + dev + eval)"
aws s3 cp configs/explore/e4b_swe/threshold.json "$B/e4b/threshold.json" --only-show-errors --region "$REGION" || true
rm -f "$S/e4b_driver.pid"
st "chain done; stopping the box"
sudo shutdown -h +2 "E4b done"
