#!/usr/bin/env bash
# Copy Terminal-Bench-Science tasks into a flat local Harbor dataset dir, optionally patched for a restricted sandbox.
#
#   adapters/pi/tbsci_tasks.sh <out dir> [task | @subset ...]       # default: @all
#   TBSCI_CLONE=<clone> adapters/pi/tbsci_tasks.sh runs/tbsci/tasks @hard_pool
#   PATCH_CA=/path/to/extra-ca-bundle.crt adapters/pi/tbsci_tasks.sh <out dir> @sandbox    # sandbox deviation
#
# Why: TB-science nests tasks as tasks/<domain>/<field>/<slug>/. Harbor's local datasets (`harbor run -p <dir>`)
# and its git "implicit" datasets only look one level down, so `-p tasks/` finds no task. With Harbor Hub access,
# prefer the published dataset (`-d terminal-bench-science/terminal-bench-science@0.1.0`, run_tbsci.sh's default).
# Use this script for a pinned git commit, for a dataset dir with only some tasks, or for the sandbox patch below.
#
# Subsets: @all = every task in tasks/dataset.toml (the published allowlist, 70 tasks at the pinned commit);
# @<name> = adapters/pi/tbsci_<name>.txt (one slug per line, `#` comments), e.g. @hard_pool, @sandbox.
# A bare slug may also name a task under lite/ (the "too easy" set; empty at the pinned commit).
#
# PATCH_CA=<host CA bundle> (sandbox only): unlike TB2, which pulls prebuilt images, every TB-science image is BUILT
# from environment/Dockerfile and tests/Dockerfile, and their pip / conda / curl / git downloads fail certificate
# checks behind a TLS-intercepting egress proxy. `tbsci.py patch-ca` copies the bundle next to every Dockerfile (and
# into the environment/ and tests/ build contexts) and inserts, after each FROM, a COPY of it plus ENV variables that
# point pip, requests, curl, git, node, conda and cargo at it. Instruction, data, tests and reward logic are
# untouched, but the bundle and the variables stay in the agent and verifier images: report results on patched
# tasks as a sandbox deviation, never as official TB-science numbers.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:?usage: $0 <out dir> [task | @subset ...]}"
shift || true
ARGS=("$@")
[ "${#ARGS[@]}" -gt 0 ] || ARGS=(@all)
TBSCI_REPO="${TBSCI_REPO:-https://github.com/harbor-framework/terminal-bench-science.git}"
TBSCI_COMMIT="${TBSCI_COMMIT:-cfff307772949a2ef90aab1aada9597e92a6ad1e}"   # 2026-09-28; 67/70 task digests = Hub 0.1.0
TBSCI_CLONE="${TBSCI_CLONE:-}"           # existing clone (skips the git clone; used as checked out)
PATCH_CA="${PATCH_CA:-}"

if [ -z "$TBSCI_CLONE" ]; then
  TBSCI_CLONE="$(mktemp -d)/terminal-bench-science"
  git clone -q --filter=blob:none "$TBSCI_REPO" "$TBSCI_CLONE"
  git -C "$TBSCI_CLONE" checkout -q "$TBSCI_COMMIT"
fi
[ -f "$TBSCI_CLONE/tasks/dataset.toml" ] || { echo "!! $TBSCI_CLONE is not a terminal-bench-science checkout" >&2; exit 1; }
[ -z "$PATCH_CA" ] || [ -f "$PATCH_CA" ] || { echo "!! PATCH_CA=$PATCH_CA: no such file" >&2; exit 1; }

slugs=()
for a in "${ARGS[@]}"; do
  case "$a" in
    @all) while IFS= read -r s; do slugs+=("$s"); done < <(python3 "$HERE/tbsci.py" manifest "$TBSCI_CLONE") ;;
    @*)   f="$HERE/tbsci_${a#@}.txt"
          [ -f "$f" ] || { echo "!! unknown subset $a (no $f)" >&2; exit 1; }
          while IFS= read -r s; do slugs+=("$s"); done < <(sed 's/#.*//; s/[[:space:]]//g; /^$/d' "$f") ;;
    *)    slugs+=("$a") ;;
  esac
done

mkdir -p "$OUT"
for s in "${slugs[@]}"; do
  found=()
  while IFS= read -r d; do [ -f "$d/task.toml" ] && found+=("$d"); done < <(
    find "$TBSCI_CLONE/tasks" "$TBSCI_CLONE/lite" -mindepth 1 -maxdepth 3 -type d -name "$s" 2>/dev/null)
  [ "${#found[@]}" -eq 1 ] || { echo "!! task $s: ${#found[@]} matches in $TBSCI_CLONE" >&2; exit 1; }
  rm -rf "${OUT:?}/$s"
  cp -r "${found[0]}" "$OUT/$s"
  if [ -n "$PATCH_CA" ]; then
    python3 "$HERE/tbsci.py" patch-ca "$OUT/$s" "$PATCH_CA" >/dev/null
    echo ">> $s: copied + CA patch"
  else
    echo ">> $s: copied"
  fi
done
echo ">> dataset dir: $OUT (${#slugs[@]} tasks; harbor run -p $OUT ..., or TASKS_DIR=$OUT adapters/pi/run_tbsci.sh)"
