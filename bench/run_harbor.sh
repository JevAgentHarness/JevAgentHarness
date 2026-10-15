#!/usr/bin/env bash
# One Harbor job: one harness x one proxy model name (arm) on a task list.
#
#   HARNESS=codex|pi|dsh|mswe|t2 ARM=jah-auto|jah-strong|jah-fast TASKS=bench/splits/tb2_dev.txt bash bench/run_harbor.sh
#
# The proxy (`jah serve --host 0.0.0.0`) must run on this host; task containers reach it as host.docker.internal.
# Every agent gets a dummy key: the proxy holds the provider keys. Variables (defaults):
#   DATASET (terminal-bench@2.0)  N (4 concurrent trials)  JOBS_DIR (runs/harbor)  PORT (8787)  JOB (<harness>-<arm>)
#   CODEX_VERSION (0.162.1)  PI_VERSION (1.2.0)  DSH_VERSION (0.2.1-alpha.2)  HARBOR (harbor)  EXTRA (more harbor args)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS=${HARNESS:?codex|pi|dsh|mswe|t2}
ARM=${ARM:?jah-auto|jah-strong|jah-fast}
TASKS=${TASKS:?a file with one task name per line}
DATASET=${DATASET:-terminal-bench@2.0}
N=${N:-4}
PORT=${PORT:-8787}
JOBS_DIR=${JOBS_DIR:-runs/harbor}
JOB=${JOB:-$HARNESS-$ARM}
HARBOR=${HARBOR:-harbor}
PROXY="http://host.docker.internal:$PORT/v1"
DUMMY=jah-proxy-holds-the-keys

includes=()
while read -r t; do [[ -n "$t" && "$t" != \#* ]] && includes+=(-i "$t"); done < "$TASKS"
common=(-d "$DATASET" -n "$N" --jobs-dir "$JOBS_DIR" --job-name "$JOB" --extra-docker-compose "$HERE/host_gateway.yaml"
        "${includes[@]}")

case "$HARNESS" in
  codex)
    cfg=$(printf '{"model_provider":"jah","web_search":"disabled","agents":{"enabled":false},"analytics":{"enabled":false},"model_providers":{"jah":{"name":"jah","base_url":"%s","env_key":"JAH_API_KEY","wire_api":"responses","supports_websockets":false,"stream_idle_timeout_ms":900000,"request_max_retries":4,"stream_max_retries":3}}}' "$PROXY")
    agent=(-a harbor_agents:JahCodex -m "openai/$ARM" --ak "version=${CODEX_VERSION:-0.162.1}" --ak "config=$cfg"
           --ae "OPENAI_API_KEY=$DUMMY" --ae "JAH_API_KEY=$DUMMY") ;;
  pi)
    agent=(-a harbor_agents:JahPi -m "openai/$ARM" --ak "version=${PI_VERSION:-1.2.0}" --ak model_api=openai-completions
           --ae "OPENAI_BASE_URL=$PROXY" --ae "OPENAI_API_KEY=$DUMMY") ;;
  dsh)
    agent=(-a harbor_agents:Dsh -m "jah/$ARM" --ak "version=${DSH_VERSION:-0.2.1-alpha.2}" --ak "proxy_url=$PROXY") ;;
  mswe)
    agent=(-a harbor_agents:JahMiniSwe -m "openai/$ARM" ${MSWE_VERSION:+--ak "version=$MSWE_VERSION"}
           --ae "OPENAI_BASE_URL=$PROXY" --ae "OPENAI_API_KEY=$DUMMY" --ae "MSWEA_API_KEY=$DUMMY") ;;
  t2)
    # Terminus-2 calls the model from the Harbor process on this host, not from the task container
    export OPENAI_API_KEY=$DUMMY
    agent=(-a harbor_agents:JahTerminus2 -m "openai/$ARM" --ak "api_base=http://localhost:$PORT/v1"
           --ak 'model_info={"max_input_tokens":1048576,"max_output_tokens":65536,"input_cost_per_token":0,"output_cost_per_token":0}') ;;
  *) echo "HARNESS must be codex, pi, dsh, mswe or t2" >&2; exit 2 ;;
esac

set -x
# shellcheck disable=SC2086
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" "$HARBOR" run "${common[@]}" "${agent[@]}" ${EXTRA:-}
