#!/bin/bash
#SBATCH --job-name=mem_critic
#SBATCH --partition=maxlab
#SBATCH --qos=maxlab_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --array=0-14
#SBATCH --output=/home/asingal2/memory_critic/logs/critic/%x_%A_%a.out
#SBATCH --error=/home/asingal2/memory_critic/logs/critic/%x_%A_%a.err

# Memory-critic arms (scripts/train_memory_critic.py, docs/ogpo_memory_critic_plan.md steps 0-1).
# Task i = (arm, seed) with seed fastest: ARMS x SEEDS. Outputs critics/mem_<arm>_seed<S>.
#   sbatch scripts/train_memory_critic.sbatch
#   ARMS="gru" SEEDS="0" sbatch --array=0 scripts/train_memory_critic.sbatch
# Smoke (CPU, one policy, a few hundred steps per arm):
#   SMOKE=1 sbatch --partition=maxlab-cpu --qos=maxlab_cpu_qos --gres= --array=0 --time=0:30:00 scripts/train_memory_critic.sbatch
set -euo pipefail
REPO=/home/asingal2/memory_critic
read -ra A_ <<< "${ARMS:-m0 clk gru shuf k1}"; read -ra S_ <<< "${SEEDS:-0 1 2}"
i=${SLURM_ARRAY_TASK_ID:-0}
echo "Job started on: $(hostname)  Job ID: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-}"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_ENTITY=akankshasingal08
export TMPDIR="/data/user_data/$USER/tmp/${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
ROOT=/data/user_data/$USER/memory_critic/critics

if [ -n "${SMOKE:-}" ]; then
    export JAX_PLATFORMS=cpu
    uv run --locked pytest -q tests/test_memcritic.py
    for ARM in m0 clk gru shuf k1; do
        uv run --locked python scripts/train_memory_critic.py --arm "$ARM" --policy_glob "square_ph_10demos_seed0_*" \
            --out_dir "$TMPDIR/smoke_$ARM" --steps 300 --eval_every 150 --no_wandb
    done
    echo "SMOKE OK"; exit 0
fi

ARM=${A_[$((i / ${#S_[@]}))]:?task $i out of range}; SEED=${S_[$((i % ${#S_[@]}))]}
OUT_DIR=$ROOT/mem_${ARM}_seed${SEED}
echo "arm $ARM seed $SEED -> $OUT_DIR"
uv run --locked python scripts/train_memory_critic.py --arm "$ARM" --seed "$SEED" --out_dir "$OUT_DIR" \
    --steps "${STEPS:-200000}" --run_name "mem_${ARM}_seed${SEED}" ${EXTRA_ARGS:-}
