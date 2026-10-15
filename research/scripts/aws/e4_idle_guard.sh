#!/usr/bin/env bash
# E4 / E4b idle stop guard (started by scripts/aws/e4_swe.sh): powers the box off after 30 min with no live driver
# (any $S/*.pid) and no running container. The driver also arms a fixed power-off guard (shutdown -h +GUARD_MIN).
S=/opt/jev/hard_status; idle=0
STATUS_S3=${STATUS_S3:-s3://$BUCKET/e4/status}   # e4_swe.sh exports e4r/status for E4r
while sleep 300; do
  alive=0
  for f in "$S"/*.pid; do pid=$(cat "$f" 2>/dev/null); [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1; done
  if [ "$alive" = 1 ] || [ -n "$(docker ps -q 2>/dev/null)" ]; then idle=0; else idle=$((idle + 5)); fi
  if [ "$idle" -ge 30 ]; then
    echo "$(date -u +%FT%TZ) [e4:idle] 30 min without an E4 driver or container: powering off" >> "$S/e4.txt"
    aws s3 sync "$S" "$STATUS_S3" --only-show-errors --region us-east-1 \
      --exclude '*' --include 'e4*' || true
    sudo shutdown -h now
  fi
done
