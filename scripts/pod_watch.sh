#!/usr/bin/env bash
# Poll a pod whose job is already running; on DONE fetch outputs/ (+ extra dir) and terminate.
# The second half of pod_drive.sh on its own, for re-attaching to a running job.
#   usage: scripts/pod_watch.sh <pod-id> <user@host:port> [extra-fetch-dir] [poll-seconds]
set -uo pipefail
cd "$(dirname "$0")/.."
S=~/.claude/skills/runpod-train/runpod.sh
ID="$1"; SPEC="$2"; EXTRA="${3:-}"; POLL="${4:-60}"
HOST="${SPEC%:*}"; PORT="${SPEC##*:}"; REMOTE=/workspace/project
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10
     -o ServerAliveInterval=30 -i "$HOME/.ssh/id_ed25519" -p "$PORT" "$HOST")
RSYNC_E="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $HOME/.ssh/id_ed25519 -p $PORT"
log() { echo "[$(date +%T)] $*"; }
log "watching $ID ($SPEC), poll ${POLL}s"
fails=0
while true; do
  st=$("${SSH[@]}" "cd $REMOTE; [ -f DONE ] && echo DONE || echo RUNNING; \
       nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader; \
       tr '\r' '\n' < train.log | grep -E '===|env OK|^step |checkpoint|done in|Error|Traceback|__EXIT__' | tail -1" </dev/null 2>/dev/null)
  if [ -z "$st" ]; then
    fails=$((fails + 1)); log "pod unreachable ($fails)"
    [ $fails -ge 8 ] && { log "LOST CONTACT — dead-man deletes the pod 10 min after the job ends"; exit 1; }
    sleep 30; continue
  fi
  fails=0
  log "$(echo "$st" | paste -sd'|' - | sed 's/|/ | gpu /; s/|/ | /2')"
  if [[ "$st" == DONE* ]]; then
    "${SSH[@]}" "grep -E 'Traceback|Error' $REMOTE/train.log | tail -3" </dev/null
    t0=$(date +%s)
    rsync -az -e "$RSYNC_E" "$HOST:$REMOTE/outputs/" outputs/ && rsync -a -e "$RSYNC_E" "$HOST:$REMOTE/train.log" "outputs/pod_train_$ID.log"
    log "fetched outputs/ ($(( $(date +%s) - t0 ))s)"
    if [ -n "$EXTRA" ]; then
      t0=$(date +%s); mkdir -p "$EXTRA"
      rsync -az --partial -e "$RSYNC_E" "$HOST:$REMOTE/$EXTRA/" "$EXTRA/" & RP=$!
      ( sleep 900; kill $RP 2>/dev/null ) & KP=$!
      wait $RP; rc=$?; kill $KP 2>/dev/null
      log "fetched $EXTRA rc=$rc ($(( $(date +%s) - t0 ))s, $(du -sh "$EXTRA" 2>/dev/null | cut -f1))"
    fi
    bash "$S" terminate "$ID"; log "TERMINATED"
    bash "$S" pods | sed 's/^/  /'
    exit 0
  fi
  sleep "$POLL"
done
