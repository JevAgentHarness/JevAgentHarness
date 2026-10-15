#!/usr/bin/env bash
# Copy Terminal-Bench 2.0 tasks into a local Harbor dataset dir, optionally patched for a restricted sandbox.
#
#   adapters/pi/tb2_sandbox_tasks.sh <out dir> [task ...]          # default tasks: prove-plus-comm overfull-hbox
#   PATCH_UV=0 adapters/pi/tb2_sandbox_tasks.sh <out dir> ...       # plain copies (open egress: use -d terminal-bench@2.0)
#
# Why: Harbor 0.23 resolves `-d terminal-bench@2.0` through its hosted registry (*.supabase.co) and every TB2
# verifier installs uv with `curl -LsSf https://astral.sh/uv/<ver>/install.sh | sh`. Where those hosts are
# blocked (the L3 smoke sandbox, docs/HARNESS_INTEGRATION.md) the tasks come from the pinned git commit of
# laude-institute/terminal-bench-2 (the one in Harbor's registry.json) and PATCH_UV=1 (default) replaces ONLY the
# astral.sh installer line with the same uv release fetched from GitHub. Instruction, environment image, tests
# and reward logic are untouched. Report results of patched tasks as a sandbox deviation, never as official TB2.
set -euo pipefail

OUT="${1:?usage: $0 <out dir> [task ...]}"
shift || true
TASKS=("$@")
[ "${#TASKS[@]}" -gt 0 ] || TASKS=(prove-plus-comm overfull-hbox)
TB2_REPO="${TB2_REPO:-https://github.com/laude-institute/terminal-bench-2.git}"
TB2_COMMIT="${TB2_COMMIT:-69671fbaac6d67a7ef0dfec016cc38a64ef7a77c}"   # harbor registry.json, terminal-bench@2.0
TB2_CLONE="${TB2_CLONE:-}"               # existing clone (skips the git clone)
PATCH_UV="${PATCH_UV:-1}"

if [ -z "$TB2_CLONE" ]; then
  TB2_CLONE="$(mktemp -d)/terminal-bench-2"
  git clone -q --filter=blob:none "$TB2_REPO" "$TB2_CLONE"
fi
git -C "$TB2_CLONE" checkout -q "$TB2_COMMIT"
mkdir -p "$OUT"
for t in "${TASKS[@]}"; do
  [ -f "$TB2_CLONE/$t/task.toml" ] || { echo "!! no task $t in $TB2_CLONE" >&2; exit 1; }
  rm -rf "${OUT:?}/$t"
  cp -r "$TB2_CLONE/$t" "$OUT/$t"
  test_sh="$OUT/$t/tests/test.sh"
  if [ "$PATCH_UV" = "1" ] && [ -f "$test_sh" ] && grep -q 'astral.sh/uv/' "$test_sh"; then
    python3 - "$test_sh" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
pat = re.compile(r'^curl -LsSf https://astral\.sh/uv/([0-9.]+)/install\.sh \| sh[ \t]*$', re.M)
m = pat.search(s)
if not m:
    sys.exit(f"unexpected uv install line in {p}")
ver = m.group(1)
repl = (f'# [sandbox patch: astral.sh blocked] same uv {ver} release, from GitHub\n'
        f'mkdir -p "$HOME/.local/bin" && curl -LsSf https://github.com/astral-sh/uv/releases/download/{ver}/'
        f'uv-x86_64-unknown-linux-gnu.tar.gz | tar -xz --strip-components=1 -C "$HOME/.local/bin" && '
        f'printf \'export PATH="$HOME/.local/bin:$PATH"\\n\' > "$HOME/.local/bin/env"')
open(p, "w").write(pat.sub(lambda _: repl, s, count=1))
PY
    echo ">> $t: patched uv install in tests/test.sh"
  else
    echo ">> $t: copied"
  fi
done
echo ">> dataset dir: $OUT (harbor run -p $OUT ...)"
