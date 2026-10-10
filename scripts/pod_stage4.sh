#!/usr/bin/env bash
# Stage 4 GPU work ONLY: the LoRA hyperparameter sweep + a LoRA-scale sweep, run as parallel
# GPU jobs. One training at batch 1 left the GPU ~40% busy (Python-bound on one core), so
# several share the GPU. Parallelism is sized from the pod's real GPU memory and CPU quota.
# Scoring runs on the Mac afterwards (stage3_lora.score_checkpoints).
set -uo pipefail
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
bash scripts/pod_setup.sh --gpu-only || exit 1
mkdir -p outputs/stage4_logs

MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
Q=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo max)
CPUS=$([ "${Q%% *}" = max ] && nproc || python3 -c "q='$Q'.split(); print(int(int(q[0])/int(q[1])))")
BY_MEM=$(( (MEM - 2000) / 7000 ))          # ~6 GB peak per fp32-master LoRA training at 512
BY_CPU=$(( CPUS - 1 ))                      # one busy core per job, one spare
PAR=$(( BY_MEM < BY_CPU ? BY_MEM : BY_CPU )); PAR=$(( PAR < 1 ? 1 : (PAR > 4 ? 4 : PAR) ))
echo "=== GPU ${MEM} MiB, CPU quota ${CPUS} -> ${PAR} parallel jobs ==="

JOBS=(
  "r32_lr1e-4_te-off|python3 -m stage3_lora.train_lora --data lora_data/jai --run r32_lr1e-4_te-off --rank 32"
  "r64_lr1e-4_te-off|python3 -m stage3_lora.train_lora --data lora_data/jai --run r64_lr1e-4_te-off --rank 64"
  "r16_lr2e-4_te-off|python3 -m stage3_lora.train_lora --data lora_data/jai --run r16_lr2e-4_te-off --lr 2e-4"
  "scale_r16_te-off_s1500|python3 -m stage3_lora.scale_sweep --lora lora_data/checkpoints/r16_te-off_s1500 --run scale_r16_te-off_s1500"
)
for j in "${JOBS[@]}"; do
  name=${j%%|*}; cmd=${j#*|}
  # portable scheduling (no `wait -n`): poll the running-job count; each job records its rc
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done
  echo "=== start $name $(date +%T) ==="
  ( bash -c "$cmd" > "outputs/stage4_logs/$name.log" 2>&1; echo $? > "outputs/stage4_logs/$name.rc" ) &
done
wait
fail=0
for j in "${JOBS[@]}"; do
  name=${j%%|*}; rc=$(cat "outputs/stage4_logs/$name.rc" 2>/dev/null || echo missing)
  [ "$rc" = 0 ] || fail=1
  echo "--- $name rc=$rc: $(grep -E 'done in|^-> ' outputs/stage4_logs/$name.log | tail -1)"
  grep -E 'Traceback|Error' "outputs/stage4_logs/$name.log" | tail -2
done
echo "=== all done (failures: $fail) ==="
exit $fail
