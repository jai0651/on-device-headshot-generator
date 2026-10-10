#!/usr/bin/env bash
# Hands-off pod session: wait for the pod id -> ssh -> sync -> verify GPU -> launch -> poll
# every 60 s -> fetch -> terminate. Bash on purpose (zsh doesn't word-split "$SSH" strings).
#
#   usage: scripts/pod_drive.sh <pod-id-file> <create-log> <job-script> [max-minutes] [extra-fetch-dir]
#
# Lessons built in (2026-09-26/28 sessions):
#  - launch with setsid + </dev/null ourselves; runpod.sh `run` hung holding the ssh channel
#  - API key goes to a 0600 file on the pod, never onto a command line (it showed up in `ps`)
#  - upload is a clean copy: no .env, no original photos, no local-only outputs
#  - small outputs/ fetched first; big extra dir fetched after, with a time cap
#  - never pkill by a pattern that appears in our own ssh command
set -uo pipefail
cd "$(dirname "$0")/.."
S=~/.claude/skills/runpod-train/runpod.sh
IDF="$1"; CREATE_LOG="$2"; JOB="$3"; MAX_MIN="${4:-45}"; EXTRA="${5:-}"
KEY=$(grep -E '^\s*RUNPOD_API_KEY\s*=' .env | head -1 | sed -E 's/^[^=]*=\s*//; s/^["'"'"']//; s/["'"'"']\s*$//')
REMOTE=/workspace/project
log() { echo "[$(date +%T)] $*"; }

until [ -s "$IDF" ]; do
  grep -q "GAVE UP" "$CREATE_LOG" 2>/dev/null && { log "NO POD: creation gave up"; exit 1; }
  sleep 5
done
ID="$(cat "$IDF")"
log "pod $ID created"
abort() { log "ABORT: $* — terminating $ID"; bash "$S" terminate "$ID"; bash "$S" pods; exit 1; }

SPEC=""
for _ in $(seq 1 180); do SPEC="$(bash "$S" ssh-of "$ID" 2>/dev/null)" && [ -n "$SPEC" ] && break; sleep 5; done
[ -n "$SPEC" ] || abort "no ssh mapping after 15 min"
HOST="${SPEC%:*}"; PORT="${SPEC##*:}"
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10
     -o ServerAliveInterval=30 -i "$HOME/.ssh/id_ed25519" -p "$PORT" "$HOST")
RSYNC_E="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $HOME/.ssh/id_ed25519 -p $PORT"
for _ in $(seq 1 60); do "${SSH[@]}" true </dev/null 2>/dev/null && break; sleep 10; done
"${SSH[@]}" true </dev/null 2>/dev/null || abort "ssh never came up"
log "ssh ready $SPEC"

STAGE="$(mktemp -d)/p"
# every local-only heavyweight: environments (.venv, .venv-coreml...), data, models, outputs, docs
rsync -a --exclude '.env' --exclude '.git/' --exclude '.venv*/' --exclude 'photos/' --exclude 'runs/' --exclude 'docs/' \
  --exclude '*.mlpackage' --exclude '*.mlmodelc' \
  --exclude 'data/' --exclude 'models/' --exclude 'outputs/' --exclude 'outputs_raw/' --exclude '__pycache__/' \
  --exclude '*.html' --exclude '*.pt' ${POD_EXCLUDE:+--exclude=$POD_EXCLUDE} ./ "$STAGE/"   # POD_EXCLUDE: e.g. lora_data/jai/
t0=$(date +%s)
rsync -az -e "$RSYNC_E" "$STAGE/" "$HOST:$REMOTE/" || abort "sync failed"
log "synced $(du -sh "$STAGE" | cut -f1) in $(( $(date +%s) - t0 ))s"

printf %s "$KEY" | "${SSH[@]}" 'umask 077; cat > /root/.rpkey' || abort "could not place key file"
cat > "$STAGE/_run.sh" <<EOF
#!/usr/bin/env bash
cd $REMOTE
rm -f DONE
timeout ${MAX_MIN}m bash $JOB > train.log 2>&1
echo "__EXIT__=\$?" >> train.log
touch DONE
sleep 600   # dead-man: the laptop normally fetches + terminates well inside this
curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$ID" -H "Authorization: Bearer \$(cat /root/.rpkey)"
EOF
rsync -a -e "$RSYNC_E" "$STAGE/_run.sh" "$HOST:$REMOTE/_run.sh" || abort "could not copy launcher"
# `;` not `&&`: with `cd X && setsid job &` bash backgrounds the whole AND-list in a subshell
# whose stdout is this ssh channel, so ssh never returns until the job (and dead-man) ends.
"${SSH[@]}" "cd $REMOTE; setsid bash _run.sh >/dev/null 2>&1 </dev/null & disown; echo launched" </dev/null \
  | grep -q launched || abort "launch failed"
log "job started: $JOB (hard limit ${MAX_MIN} min, self-delete 10 min after it ends)"

fails=0
while true; do
  sleep 60
  st=$("${SSH[@]}" "cd $REMOTE; [ -f DONE ] && echo DONE || echo RUNNING; \
       nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader; \
       tr '\r' '\n' < train.log | grep -E '===|env OK|^step |checkpoint|done in|Error|Traceback|__EXIT__' | tail -1" </dev/null 2>/dev/null)
  if [ -z "$st" ]; then
    fails=$((fails + 1)); log "pod unreachable ($fails)"
    [ $fails -ge 8 ] && { log "LOST CONTACT — dead-man deletes the pod 10 min after the job ends"; exit 1; }
    continue
  fi
  fails=0
  log "$(echo "$st" | paste -sd'|' - | sed 's/|/ | gpu /; s/|/ | /2')"
  if [[ "$st" == DONE* ]]; then
    "${SSH[@]}" "grep -E 'Traceback|Error' $REMOTE/train.log | tail -3" </dev/null
    t0=$(date +%s)
    rsync -az -e "$RSYNC_E" "$HOST:$REMOTE/outputs/" outputs/ && rsync -a -e "$RSYNC_E" "$HOST:$REMOTE/train.log" "outputs/pod_train_$ID.log"
    log "fetched outputs/ ($(( $(date +%s) - t0 ))s)"
    if [ -n "$EXTRA" ]; then
      t0=$(date +%s)
      # timeout (GNU coreutils) isn't on macOS: cap with a background kill instead
      mkdir -p "$EXTRA"
      rsync -az --partial -e "$RSYNC_E" "$HOST:$REMOTE/$EXTRA/" "$EXTRA/" & RP=$!
      ( sleep 900; kill $RP 2>/dev/null ) & KP=$!
      wait $RP; rc=$?; kill $KP 2>/dev/null
      log "fetched $EXTRA rc=$rc ($(( $(date +%s) - t0 ))s, $(du -sh "$EXTRA" 2>/dev/null | cut -f1))"
    fi
    bash "$S" terminate "$ID"
    log "TERMINATED"
    bash "$S" pods | sed 's/^/  /'
    exit 0
  fi
done
