#!/usr/bin/env bash
# One-time setup of a hard-task (H5) CPU box: Ubuntu 24.04, 32 vCPU, 500 GB gp3, instance profile <instance-profile>,
# security group jev-train-noingress (no inbound; SSM only). Idempotent: re-running skips finished steps.
# Runs as ubuntu (sudo for system packages), e.g. through SSM:
#   aws ssm send-command --document-name AWS-RunShellScript --instance-ids <id> --parameters \
#     'commands=["sudo -u ubuntu -H bash -lc \"aws s3 cp s3://<bucket>/code/hard_setup.sh /tmp/hard_setup.sh && BOX=swe bash /tmp/hard_setup.sh\""]'
# Installs: Docker CE (+ compose / buildx; address pool 10.128.0.0/12 /24, mirror.gcr.io as Docker Hub mirror), uv,
# the repo from s3://<bucket>/code/jev.bundle (branch main) plus the optional working-tree
# overlay s3://<bucket>/code/hard_overlay.tgz (files not yet committed), its venv (.[all] + mini-swe-agent 2.4.6),
# .venv-eval (swebench 4.1.0), Node 22 + pi 0.99.2, Harbor at git 4f848e3 (pyproject 0.23.0; Python 3.12 venv /opt/jev/harbor-venv) and a
# terminal-bench-science checkout at cfff307 (/opt/jev/tbsci; for `tbsci.py select`).
# No API key is read here. Status lines go to /opt/jev/hard_status/setup.txt and s3://<bucket>/hard/<BOX>/status/.
set -uo pipefail
BUCKET=${BUCKET:?set BUCKET to your S3 bucket}; B="s3://$BUCKET"; REGION=${AWS_REGION:-us-east-1}
BOX=${BOX:-$(hostname -s)}
BRANCH=${BRANCH:-main}
J=/opt/jev; R=$J/JevAgentHarness/research; S=$J/hard_status
sudo mkdir -p "$J" && sudo chown "$(id -u):$(id -g)" "$J"; mkdir -p "$S"
export PATH="$HOME/.local/bin:$PATH"
st() { echo "$(date -u +%FT%TZ) $*" | tee -a "$S/setup.txt"
       aws s3 sync "$S" "$B/hard/$BOX/status" --only-show-errors --region "$REGION" || true; }
die() { st "SETUP FAILED: $*"; exit 1; }
st "setup start box=$BOX $(nproc) vCPU $(free -g | awk '/Mem:/{print $2}') GB RAM $(df -BG / | awk 'NR==2{print $4}') free"

# 1. system packages + Docker CE
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq && sudo apt-get install -y -qq git jq unzip build-essential ca-certificates curl >/dev/null \
  || die "apt"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh >/dev/null 2>&1 || die "docker install"
fi
sudo usermod -aG docker ubuntu
sudo mkdir -p /etc/docker
cat <<'JSON' | sudo tee /etc/docker/daemon.json >/dev/null
{
  "default-address-pools": [{"base": "10.128.0.0/12", "size": 24}],
  "registry-mirrors": ["https://mirror.gcr.io"],
  "max-concurrent-downloads": 8,
  "log-driver": "json-file",
  "log-opts": {"max-size": "50m", "max-file": "2"},
  "features": {"buildkit": true}
}
JSON
sudo systemctl restart docker && sudo docker info --format '{{.ServerVersion}} {{.DockerRootDir}}' >> "$S/setup.txt" \
  || die "docker restart"
sudo docker compose version >> "$S/setup.txt" 2>&1; sudo docker buildx version >> "$S/setup.txt" 2>&1
st "docker ok"

# 2. uv + Python
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || die "uv"
uv python install 3.11 3.12 >/dev/null 2>&1 || die "uv python"

# 3. repo from the bundle (+ overlay of uncommitted files)
aws s3 cp "$B/code/jev.bundle" "$J/jev.bundle" --only-show-errors --region "$REGION" || die "bundle"
if [ ! -d "$R/.git" ]; then
  git clone -q -b "$BRANCH" "$J/jev.bundle" "$R" || die "clone"
else
  git -C "$R" checkout -q -- . && git -C "$R" pull -q --ff-only "$J/jev.bundle" "$BRANCH" || die "pull"
fi
if aws s3 cp "$B/code/hard_overlay.tgz" "$J/hard_overlay.tgz" --only-show-errors --region "$REGION" 2>/dev/null; then
  tar -xzf "$J/hard_overlay.tgz" -C "$R" && st "overlay applied: $(tar -tzf "$J/hard_overlay.tgz" | tr '\n' ' ')"
fi
st "repo $(git -C "$R" log --oneline -1)"

# 4. venvs
cd "$R" || die "cd repo"
if [ ! -x .venv/bin/python ]; then uv venv -q -p 3.11 .venv || die "venv"; fi
uv pip install -q --python .venv -e ".[all]" "mini-swe-agent==2.4.6" || die "pip repo"
if [ ! -x .venv-eval/bin/python ]; then uv venv -q -p 3.11 .venv-eval || die "venv-eval"; fi
uv pip install -q --python .venv-eval "swebench==4.1.0" || die "pip swebench"
if [ ! -x "$J/harbor-venv/bin/harbor" ]; then uv venv -q -p 3.12 "$J/harbor-venv" || die "harbor venv"; fi
# Harbor at the commit the dev smokes used (pyproject 0.23.0, 2026-09-30): PyPI's harbor 0.23.0 lacks pi's `max_turns`
# option and the pi -> ATIF trajectory.json conversion that run_tbsci.sh and summarize_harbor.py rely on.
HARBOR_REF=${HARBOR_REF:-4f848e367e6e2627c2c0b1826bb3b20ceab1198f}
uv pip install -q --python "$J/harbor-venv" "harbor @ git+https://github.com/laude-institute/harbor@$HARBOR_REF" \
  || die "pip harbor"
st "venvs ok: mini-swe $(.venv/bin/python -c 'import minisweagent; print(minisweagent.__version__)' 2>/dev/null | tail -1)," \
   "swebench $(.venv-eval/bin/python -c 'import swebench; print(swebench.__version__)' 2>/dev/null)," \
   "harbor $("$J/harbor-venv/bin/python" -c 'import importlib.metadata as m; print(m.version("harbor"))') @ ${HARBOR_REF:0:7}"

# 5. Node 22 + pi (host copy; Harbor installs pi inside each task container itself)
if ! node --version 2>/dev/null | grep -q '^v22'; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - >/dev/null 2>&1 && sudo apt-get install -y -qq nodejs >/dev/null \
    || die "node"
fi
npm ls -g @earendil-works/pi-coding-agent 2>/dev/null | grep -q '0.99.2' \
  || sudo npm install -g -s @earendil-works/pi-coding-agent@0.99.2 >/dev/null 2>&1 || die "pi"
st "node $(node --version), pi $(pi --version 2>/dev/null | tail -1)"

# 6. terminal-bench-science at the registered commit (domains / expert estimates for `tbsci.py select`)
if [ ! -d "$J/tbsci/.git" ]; then
  git clone -q https://github.com/harbor-framework/terminal-bench-science "$J/tbsci" || die "tbsci clone"
fi
git -C "$J/tbsci" checkout -q cfff307 || die "tbsci checkout"
st "tbsci $(git -C "$J/tbsci" rev-parse --short HEAD)"

# 7. checks: docker as ubuntu (new group), offline adapter tests, Docker Hub anonymous pull budget
sg docker -c "docker run --rm hello-world" >/dev/null 2>&1 && st "docker run ok" || st "WARNING: docker run as ubuntu failed"
.venv/bin/python -m pytest -q tests/test_mini_swe_adapter.py tests/test_l3_smoke.py > "$S/pytest_adapters.txt" 2>&1
st "pytest adapters: $(tail -1 "$S/pytest_adapters.txt")"
tok=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:ratelimitpreview/test:pull" | jq -r .token)
curl -s --head -H "Authorization: Bearer $tok" https://registry-1.docker.io/v2/ratelimitpreview/test/manifests/latest \
  | grep -i '^ratelimit' > "$S/dockerhub_ratelimit.txt" || true
st "docker hub anonymous limit: $(tr -d '\r' < "$S/dockerhub_ratelimit.txt" | tr '\n' ' ')"
st "setup done"
