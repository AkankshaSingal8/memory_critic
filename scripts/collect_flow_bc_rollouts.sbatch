#!/bin/bash
#SBATCH --job-name=flow_bc_rollouts
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --array=0-29
#SBATCH --output=/home/asingal2/memory_critic/logs/rollouts/%x_%A_%a.out
#SBATCH --error=/home/asingal2/memory_critic/logs/rollouts/%x_%A_%a.err
#SBATCH --open-mode=append

# Rollouts of the 30 frozen flow-BC policies (selected/target_100.pkl), one array task per
# policy: index = dataset*15 + size*3 + seed over (ph, mh) x (10, 20, 50, 100, 200) x (0, 7, 42).
# See docs/collect_flow_bc_rollouts.md. Requeue-safe: finished 8-episode batches are skipped.
#
# Overrides: SAMPLER (sde|ode), EPISODES (400), OUT_DIR. Smoke test:
#   EPISODES=8 sbatch --array=0 scripts/collect_flow_bc_rollouts.sbatch

set -euo pipefail
REPO=/home/asingal2/memory_critic
source "$REPO/run_scripts/_forward_term.sh"
echo "Job started on: $(hostname)  Job ID: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-}  Restart count: ${SLURM_RESTART_COUNT:-0}"

DATASETS=(ph mh); SIZES=(10 20 50 100 200); SEEDS=(0 7 42)
i=${SLURM_ARRAY_TASK_ID:-0}
export TASK=square DATASET_TYPE=${DATASETS[$((i / 15))]} N_DEMOS=${SIZES[$(((i % 15) / 3))]} SEED=${SEEDS[$((i % 3))]}
source "$REPO/scripts/_flow_bc_names.sh"
SAMPLER=${SAMPLER:-sde}
EPISODES=${EPISODES:-400}
OUT_DIR=${OUT_DIR:-/data/user_data/$USER/memory_critic/rollouts/flow_bc_target100}
POLICY_ID="square_${DATASET_TYPE}_${N_DEMOS}demos_seed${SEED}_${SAMPLER}"
echo "Policy: $POLICY_ID  Run dir: $RUN_DIR"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TMPDIR="/data/user_data/$USER/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$OUT_DIR"
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

uv run --locked python scripts/collect_flow_bc_rollouts.py --run_dir "$RUN_DIR" \
    --policy_id "$POLICY_ID" --out_dir "$OUT_DIR" --sampler "$SAMPLER" --episodes "$EPISODES" \
    --n_envs "${N_ENVS:-8}" &
st=0
wait_for_training $! || st=$?
if [ "$st" -eq 3 ]; then
    echo "[requeue] collection incomplete (exit 3); requeueing job $SLURM_JOB_ID"
    if scontrol requeue "$SLURM_JOB_ID"; then exit 0; fi
    exit 3
fi
echo "Job finished (exit $st)."
exit "$st"
