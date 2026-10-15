#!/usr/bin/env bash
# Terminal-Bench 2.0 arms of pi (+ the Jev extensions) through Harbor; one Harbor job per arm.
#
#   adapters/pi/run_tb2.sh                                   # arms baseline + jev_gate_stop on terminal-bench@2.0
#   adapters/pi/run_tb2.sh baseline jev_shadow jev_gate_stop # chosen arms
#   TASKS_DIR=runs/dev/tb2/tasks N_CONCURRENT=2 adapters/pi/run_tb2.sh   # local task dir (tb2_sandbox_tasks.sh)
#   DRY_RUN=1 adapters/pi/run_tb2.sh                         # print the harbor commands only
#
# Arms (pi extensions; tau defaults as in the pilot):
#   baseline       plain pi through the same wrapper (harbor_jev_pi:JevPi, no extensions)
#   jev_shadow     tool-gate + stop-verify in log mode: S1 asked and logged, never acts (calibration data)
#   jev_gate       tool-gate (tool_select/guard, JEV_GATE_TAU)
#   jev_stop       stop-verify (stop/guard, JEV_STOP_TAU)
#   jev_gate_stop  both
#   jev_reflect    reflect (reflection/trigger, JEV_REFLECT_TAU): after a failed or repeated call, S1 decides whether
#                  the agent should stop and reflect (targets the loop-until-timeout failure mode)
#   jev_stop_reflect  stop-verify + reflect
#   jev_router     pi's virtual model s1/router (jev-router.ts): before every request the decision model chooses the
#                  strong LLM (MODEL) or the fast one (CHEAP_MODEL) at JEV_ROUTER_TAU (E25, harness A)
#   always_cheap   plain pi with CHEAP_MODEL on every step (E25's descriptive lower bound)
#   oracle, nop    Harbor's built-in agents, no API calls: the reference solution (checks that a task builds and
#                  its verifier passes in this environment) and do-nothing (the verifier's floor: tests that pass
#                  without any agent output)
# Env:
#   HARBOR (harbor binary; Python >= 3.12 venv), MODEL (google/gemini-3.5-flash), CHEAP_MODEL
#   (google/gemini-3.5-flash-lite; jev_router's fast LLM and always_cheap's LLM), JEV_ROUTER_TAU (0.5), PI_VERSION (0.99.2),
#   THINKING (minimal), DATASET (terminal-bench@2.0) or TASKS_DIR (local dataset dir -> `harbor run -p`),
#   INCLUDE (space-separated task-name globs), N_CONCURRENT (4), N_ATTEMPTS (1), OUT (runs/tb2), ENV_FILE (.env);
#   relative OUT / TASKS_DIR / ENV_FILE / EXTRA_CA are taken from the current directory and made absolute (Harbor's
#   `docker compose cp` resolves a relative path against the task's environment dir),
#   AGENT_TIMEOUT_MULT (unset = task timeouts; e.g. 0.125 -> --agent-timeout-multiplier 0.125),
#   MAX_TURNS (unset; pi turn cap -> --ak max_turns=N; a capped run is a budget deviation, report it),
#   JEV_BASE_URL (http://host.docker.internal:8090: decision server on the host, see harbor_host_gateway.yaml),
#   JEV_WIRE_MODEL (gemini-s1), JEV_GATE_TAU (0.5), JEV_STOP_TAU (0.5), JEV_REFLECT_TAU (0.5),
#   EXTRA_CA (host path of a CA bundle the task containers must trust: TLS-intercepting egress, e.g. the agent
#   proxy's /path/to/extra-ca-bundle.crt; mounted read-only and exported to pi's install + run and to the verifier),
#   EXTRA_CA_VERIFIER (1; 0 = do not export the EXTRA_CA variables to the verifier: Harbor does not mount the bundle
#   into separate-mode verifier containers, e.g. Terminal-Bench-Science's no-network verifiers),
#   HARBOR_ARGS (extra `harbor run` args, split on whitespace: no quoting, no globbing, e.g. "-x qemu-* -l 10").
# Start the decision server first (bind a non-loopback address so containers reach it):
#   python -m jevharness.decision.server --host 0.0.0.0 --port 8090 --name gemini-s1 --cache cache/s1_pi_gemini.sqlite \
#     --s1 '{"type":"llm_generative","llm":{"provider":"gemini","model":"gemini-3.5-flash-lite"},"reasoning_effort":"minimal"}'
# Summarise: python adapters/pi/summarize_harbor.py $OUT/<arm> ...
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARBOR="${HARBOR:-harbor}"
MODEL="${MODEL:-google/gemini-3.5-flash}"
CHEAP_MODEL="${CHEAP_MODEL:-google/gemini-3.5-flash-lite}"
JEV_ROUTER_TAU="${JEV_ROUTER_TAU:-0.5}"
PI_VERSION="${PI_VERSION:-0.99.2}"
THINKING="${THINKING:-minimal}"
DATASET="${DATASET:-terminal-bench@2.0}"
TASKS_DIR="${TASKS_DIR:-}"
INCLUDE="${INCLUDE:-}"
N_CONCURRENT="${N_CONCURRENT:-4}"
N_ATTEMPTS="${N_ATTEMPTS:-1}"
OUT="${OUT:-$REPO/runs/tb2}"
ENV_FILE="${ENV_FILE:-$REPO/.env}"
JEV_BASE_URL="${JEV_BASE_URL:-http://host.docker.internal:8090}"
JEV_WIRE_MODEL="${JEV_WIRE_MODEL:-gemini-s1}"
JEV_GATE_TAU="${JEV_GATE_TAU:-0.5}"
JEV_STOP_TAU="${JEV_STOP_TAU:-0.5}"
JEV_REFLECT_TAU="${JEV_REFLECT_TAU:-0.5}"
AGENT_TIMEOUT_MULT="${AGENT_TIMEOUT_MULT:-}"
MAX_TURNS="${MAX_TURNS:-}"
EXTRA_CA="${EXTRA_CA:-}"
EXTRA_CA_VERIFIER="${EXTRA_CA_VERIFIER:-1}"
HARBOR_ARGS="${HARBOR_ARGS:-}"
DRY_RUN="${DRY_RUN:-}"
abspath() { case "$1" in ""|/*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "${1#./}" ;; esac; }
OUT="$(abspath "$OUT")"; TASKS_DIR="$(abspath "$TASKS_DIR")"; ENV_FILE="$(abspath "$ENV_FILE")"
EXTRA_CA="$(abspath "$EXTRA_CA")"
ARMS=("$@")
[ "${#ARMS[@]}" -gt 0 ] || ARMS=(baseline jev_gate_stop)

pi_opts=(--ak "version=$PI_VERSION")
[ -n "$THINKING" ] && pi_opts+=(--ak "thinking=$THINKING")
[ -n "$MAX_TURNS" ] && pi_opts+=(--ak "max_turns=$MAX_TURNS")
common=(-o "$OUT" -n "$N_CONCURRENT" -k "$N_ATTEMPTS" -y
        --extra-docker-compose "$REPO/adapters/pi/harbor_host_gateway.yaml")
[ -n "$AGENT_TIMEOUT_MULT" ] && common+=(--agent-timeout-multiplier "$AGENT_TIMEOUT_MULT")
[ -f "$ENV_FILE" ] && common+=(--env-file "$ENV_FILE")
if [ -n "$TASKS_DIR" ]; then common+=(-p "$TASKS_DIR"); else common+=(-d "$DATASET"); fi
read -r -a include <<< "$INCLUDE"   # no pathname expansion: the globs are for Harbor, not the shell
for g in ${include[@]+"${include[@]}"}; do common+=(-i "$g"); done
if [ -n "$EXTRA_CA" ]; then
  ca=/etc/harbor-extra-ca/ca-bundle.crt
  common+=(--mounts-json "[{\"type\":\"bind\",\"source\":\"$EXTRA_CA\",\"target\":\"$ca\",\"read_only\":true}]")
  for v in NODE_EXTRA_CA_CERTS CURL_CA_BUNDLE GIT_SSL_CAINFO SSL_CERT_FILE; do common+=(--ae "$v=$ca"); done
  if [ "$EXTRA_CA_VERIFIER" = "1" ]; then for v in CURL_CA_BUNDLE SSL_CERT_FILE; do common+=(--ve "$v=$ca"); done; fi
fi
jev_env=(--ae "JEV_BASE_URL=$JEV_BASE_URL" --ae "JEV_WIRE_MODEL=$JEV_WIRE_MODEL"
         --ae "JEV_GATE_TAU=$JEV_GATE_TAU" --ae "JEV_STOP_TAU=$JEV_STOP_TAU" --ae "JEV_REFLECT_TAU=$JEV_REFLECT_TAU")

if [ -z "$DRY_RUN" ]; then
  case "$JEV_BASE_URL" in
    http://host.docker.internal:*)
      port="${JEV_BASE_URL##*:}"
      curl -s --noproxy '*' -m 5 -o /dev/null "http://127.0.0.1:$port/health" \
        || echo "WARNING: no decision server on host port $port (Jev arms fail open = plain pi)" >&2 ;;
  esac
fi

for arm in "${ARMS[@]}"; do
  agent_model="$MODEL"
  case "$arm" in
    jev_router)   agent_model="s1/router" ;;   # jev-router.ts's virtual model (JEV_ROUTER_PROVIDER / JEV_ROUTER_ID)
    always_cheap) agent_model="$CHEAP_MODEL" ;;
  esac
  agent=(-a harbor_jev_pi:JevPi -m "$agent_model" "${pi_opts[@]}")
  case "$arm" in
    oracle|nop)    agent=(-a "$arm"); extra=() ;;
    baseline|always_cheap) extra=() ;;
    jev_router)    extra=(--ak jev_extensions=router "${jev_env[@]}" --ae "JEV_ROUTER_STRONG=$MODEL"
                          --ae "JEV_ROUTER_CHEAP=$CHEAP_MODEL" --ae "JEV_ROUTER_TAU=$JEV_ROUTER_TAU") ;;
    jev_gate)      extra=(--ak jev_extensions=tool-gate "${jev_env[@]}") ;;
    jev_stop)      extra=(--ak jev_extensions=stop-verify "${jev_env[@]}") ;;
    jev_gate_stop) extra=(--ak jev_extensions=tool-gate,stop-verify "${jev_env[@]}") ;;
    jev_shadow)    extra=(--ak jev_extensions=tool-gate,stop-verify "${jev_env[@]}"
                          --ae JEV_GATE_MODE=log --ae JEV_STOP_MODE=log) ;;
    jev_reflect)   extra=(--ak jev_extensions=reflect "${jev_env[@]}") ;;
    jev_stop_reflect) extra=(--ak jev_extensions=stop-verify,reflect "${jev_env[@]}") ;;
    *) echo "unknown arm $arm (baseline jev_shadow jev_gate jev_stop jev_gate_stop jev_reflect jev_stop_reflect" \
            "jev_router always_cheap oracle nop)" >&2
       exit 2 ;;
  esac
  cmd=("$HARBOR" run "${agent[@]}" "${common[@]}" --job-name "$arm" ${extra[@]+"${extra[@]}"})
  read -r -a hargs <<< "$HARBOR_ARGS"   # word-split, no pathname expansion
  cmd+=(${hargs[@]+"${hargs[@]}"})
  echo "+ PYTHONPATH=adapters/pi ${cmd[*]}" >&2
  [ -n "$DRY_RUN" ] && continue
  (cd "$REPO" && PYTHONPATH="$REPO/adapters/pi${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}")
done
[ -n "$DRY_RUN" ] || echo ">> summarise: python adapters/pi/summarize_harbor.py ${ARMS[*]/#/$OUT/}" >&2
