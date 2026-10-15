#!/usr/bin/env bash
# Launch one hard-task (H5) CPU box from the cloud session (local side; the box side is hard_setup.sh + hard_swe.sh /
# hard_tb.sh). Guard rails (2026-10-02 brief): 32-vCPU general/compute type, 500 GB gp3 root, shutdown behaviour
# `stop`, user-data `shutdown -h +1800` (30 h), tags Name=jev-hard-<purpose>, Project=jev, no inbound (SSM only).
#   scripts/aws/hard_launch.sh swe                 # m7i.8xlarge in the first AZ with capacity
#   TYPES="c7i.8xlarge m6i.8xlarge" scripts/aws/hard_launch.sh tb
# Prints the instance id. Ubuntu 24.04 (Canonical SSM parameter); user-data also installs the AWS CLI (snap).
set -euo pipefail
PURPOSE=${1:?purpose (e.g. swe, tb)}
REGION=${AWS_REGION:-us-east-1}
TYPES=${TYPES:-"m7i.8xlarge c7i.8xlarge m6i.8xlarge m7a.8xlarge c6i.8xlarge"}
SG=${SG:?set SG to a security group id}                      # jev-train-noingress
PROFILE=${PROFILE:?set PROFILE to an instance profile}
SUBNETS=${SUBNETS:?set SUBNETS (space-separated ids)}
AWS_BIN=$(command -v aws)
aws() { env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY "$AWS_BIN" --region "$REGION" "$@"; }
AMI=${AMI:-$(aws ssm get-parameter --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query Parameter.Value --output text)}
UD=$(mktemp); trap 'rm -f "$UD" "$UD.err"' EXIT
cat > "$UD" <<'EOF'
#!/bin/bash
shutdown -h +1800 "jev-hard: 30 h guard"
snap install aws-cli --classic || (apt-get update -qq && apt-get install -y -qq unzip && cd /tmp \
  && curl -fsSL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o awscliv2.zip && unzip -q awscliv2.zip && ./aws/install)
EOF
for t in $TYPES; do
  for s in $SUBNETS; do
    if id=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$t" --subnet-id "$s" \
          --security-group-ids "$SG" --iam-instance-profile "Name=$PROFILE" \
          --instance-initiated-shutdown-behavior stop --user-data "file://$UD" \
          --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
          --metadata-options HttpTokens=required,HttpEndpoint=enabled \
          --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=jev-hard-$PURPOSE},{Key=Project,Value=jev}]" \
            "ResourceType=volume,Tags=[{Key=Name,Value=jev-hard-$PURPOSE},{Key=Project,Value=jev}]" \
          --query 'Instances[0].InstanceId' --output text 2>"$UD.err"); then
      echo "$id $t $s"; exit 0
    fi
    grep -q -E 'InsufficientInstanceCapacity|Unsupported|not supported' "$UD.err" || { cat "$UD.err" >&2; exit 1; }
    echo "no capacity: $t in $s" >&2
  done
done
echo "no capacity for any of: $TYPES" >&2; exit 1
