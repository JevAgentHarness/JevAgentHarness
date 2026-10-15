#!/usr/bin/env bash
# SWE-bench Verified arms of SWE-agent v1.1.0 (+ per-step routing by a decision model) through Harbor; one Harbor job per
# arm. E25 (docs/PREREG_A7_STAGE2.md, Deviations 2026-10-09, "E25"), harness B.
#
#   adapters/swe_agent/run_swe.sh baseline jev_router always_cheap        # every arm on IDS_FILE
#   DRY_RUN=1 IDS_FILE=configs/explore/e25_harnesses/swe_smoke_ids.txt adapters/swe_agent/run_swe.sh baseline
#
# Arms (every arm runs harbor_jev_swe_agent:JevSweAgent with adapters/swe_agent/e25_default.yaml):
#   baseline      SWE-agent unchanged, MODEL on every step
#   jev_router    before every step the decision model chooses MODEL or CHEAP_MODEL (p(hard) >= JEV_ROUTER_TAU -> MODEL)
#   always_cheap  SWE-agent unchanged, CHEAP_MODEL on every step (descriptive lower bound)
# Env:
#   HARBOR (harbor binary, Python >= 3.12 venv), MODEL (gemini/gemini-3.5-flash), CHEAP_MODEL (gemini/gemini-3.5-flash-lite),
#   SWE_AGENT_VERSION (v1.1.0), LITELLM_VERSION (1.82.6), DATASET (swebench-verified@1.0), IDS_FILE (instance ids, one
#   per line; empty = the whole dataset), N_CONCURRENT (5), N_ATTEMPTS (1), OUT (runs/e25/swe), JEV_BASE_URL
#   (http://host.docker.internal:8090, the host's decision server), JEV_WIRE_MODEL (jev-1.13), JEV_ROUTER_TAU (0.5),
#   HARBOR_ARGS (extra `harbor run` args, word-split). Keys: Harbor resolves GEMINI_API_KEY from its own environment.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HARBOR="${HARBOR:-harbor}"
MODEL="${MODEL:-gemini/gemini-3.5-flash}"
CHEAP_MODEL="${CHEAP_MODEL:-gemini/gemini-3.5-flash-lite}"
SWE_AGENT_VERSION="${SWE_AGENT_VERSION:-v1.1.0}"
LITELLM_VERSION="${LITELLM_VERSION:-1.82.6}"
DATASET="${DATASET:-swebench-verified@1.0}"
IDS_FILE="${IDS_FILE:-}"
N_CONCURRENT="${N_CONCURRENT:-5}"
N_ATTEMPTS="${N_ATTEMPTS:-1}"
OUT="${OUT:-$REPO/runs/e25/swe}"
JEV_BASE_URL="${JEV_BASE_URL:-http://host.docker.internal:8090}"
JEV_WIRE_MODEL="${JEV_WIRE_MODEL:-jev-1.13}"
JEV_ROUTER_TAU="${JEV_ROUTER_TAU:-0.5}"
HARBOR_ARGS="${HARBOR_ARGS:-}"
DRY_RUN="${DRY_RUN:-}"
case "$OUT" in /*) ;; *) OUT="$PWD/${OUT#./}" ;; esac
case "$IDS_FILE" in ""|/*) ;; *) IDS_FILE="$PWD/${IDS_FILE#./}" ;; esac
ARMS=("$@")
[ "${#ARMS[@]}" -gt 0 ] || ARMS=(baseline jev_router)

common=(-d "$DATASET" -o "$OUT" -n "$N_CONCURRENT" -k "$N_ATTEMPTS" -y
        --extra-docker-compose "$REPO/adapters/pi/harbor_host_gateway.yaml")
if [ -n "$IDS_FILE" ]; then
  while IFS= read -r id; do
    id="${id%%#*}"; id="$(echo "$id" | tr -d '[:space:]')"
    [ -n "$id" ] && common+=(-i "*$id")
  done < "$IDS_FILE"
fi
opts=(--ak "version=$SWE_AGENT_VERSION" --ak "litellm_version=$LITELLM_VERSION")

for arm in "${ARMS[@]}"; do
  case "$arm" in
    baseline)     model="$MODEL"; extra=(--ak jev_router=off) ;;
    always_cheap) model="$CHEAP_MODEL"; extra=(--ak jev_router=off) ;;
    jev_router)   model="$MODEL"; extra=(--ak jev_router=trigger --ae "JEV_BASE_URL=$JEV_BASE_URL"
                    --ae "JEV_WIRE_MODEL=$JEV_WIRE_MODEL" --ae "JEV_ROUTER_TAU=$JEV_ROUTER_TAU"
                    --ae "JEV_ROUTER_CHEAP=$CHEAP_MODEL") ;;
    *) echo "unknown arm $arm (baseline jev_router always_cheap)" >&2; exit 2 ;;
  esac
  cmd=("$HARBOR" run -a harbor_jev_swe_agent:JevSweAgent -m "$model" "${opts[@]}" "${common[@]}" --job-name "$arm"
       "${extra[@]}")
  read -r -a hargs <<< "$HARBOR_ARGS"
  cmd+=(${hargs[@]+"${hargs[@]}"})
  echo "+ PYTHONPATH=adapters/swe_agent ${cmd[*]}" >&2
  [ -n "$DRY_RUN" ] && continue
  (cd "$REPO" && PYTHONPATH="$REPO/adapters/swe_agent${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}")
done
