#!/bin/bash
#SBATCH --job-name=mc_critic
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/asingal2/memory_critic/logs/critic/%x_%A_%a.out
#SBATCH --error=/home/asingal2/memory_critic/logs/critic/%x_%A_%a.err
#SBATCH --open-mode=append

# No-memory MC critic baseline (scripts/train_mc_critic.py), one array task per critic seed.
# Overrides: STEPS, POINTS (decision|all), TAG, EXTRA_ARGS. A preempted task restarts from scratch
# (the run is short).

set -euo pipefail
REPO=/home/asingal2/memory_critic
SEEDS=(0 1 2)
SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}
POINTS=${POINTS:-decision}
TAG=${TAG:-mc_${POINTS}}
OUT_DIR=${OUT_DIR:-/data/user_data/$USER/memory_critic/critics/${TAG}_seed${SEED}}
echo "Job started on: $(hostname)  Job ID: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-}  Restart count: ${SLURM_RESTART_COUNT:-0}"
echo "Out dir: $OUT_DIR"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export WANDB_ENTITY=akankshasingal08
export TMPDIR="/data/user_data/$USER/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run --locked python scripts/train_mc_critic.py --out_dir "$OUT_DIR" --seed "$SEED" \
    --points "$POINTS" --steps "${STEPS:-200000}" --run_name "${TAG}_seed${SEED}" ${EXTRA_ARGS:-}
