#!/usr/bin/env bash
# Create a pod from an APPROVED list, cheapest first, retrying until one has capacity.
# Only ever tries what's listed: never falls back to anything the user hasn't approved.
#   usage: scripts/pod_create.sh <pod-id-file> <tries> "COMMUNITY|NVIDIA RTX A5000" "SECURE|NVIDIA RTX A4000" ...
set -uo pipefail
cd "$(dirname "$0")/.."
IDF="$1"; TRIES="$2"; shift 2
KEY=$(grep -E '^\s*RUNPOD_API_KEY\s*=' .env | head -1 | sed -E 's/^[^=]*=\s*//; s/^["'"'"']//; s/["'"'"']\s*$//')
IMG="${RUNPOD_IMAGE:-runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404}"
rm -f "$IDF"
for attempt in $(seq 1 "$TRIES"); do
  for SPEC in "$@"; do
    CLOUD=${SPEC%%|*}; GPU=${SPEC#*|}
    BODY=$(jq -n --arg g "$GPU" --arg c "$CLOUD" --arg img "$IMG" '{name:"dml", imageName:$img, gpuTypeIds:[$g],
      gpuCount:1, cloudType:$c, computeType:"GPU", interruptible:false, containerDiskInGb:40, volumeInGb:0,
      ports:["22/tcp"], dataCenterPriority:"availability"}')
    OUT=$(curl -sS -X POST https://rest.runpod.io/v1/pods -H "Authorization: Bearer $KEY" \
          -H 'Content-Type: application/json' -d "$BODY" || true)
    ID=$(jq -r '.id // empty' <<<"$OUT" 2>/dev/null || true)
    if [ -n "$ID" ]; then
      echo "$ID" > "$IDF"
      jq -r '"CREATED \(.id)  '"$CLOUD $GPU"'  $\(.costPerHr)/hr  attempt '"$attempt"'"' <<<"$OUT"
      exit 0
    fi
  done
  echo "attempt $attempt: no capacity $(date +%H:%M:%S)"
  [ "$attempt" -lt "$TRIES" ] && sleep 60
done
echo "GAVE UP after $TRIES attempts"; exit 1
