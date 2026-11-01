#!/bin/bash
#SBATCH --job-name=bon_eval
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --array=0-29
#SBATCH --output=/home/asingal2/memory_critic/logs/bon/%x_%A_%a.out
#SBATCH --error=/home/asingal2/memory_critic/logs/bon/%x_%A_%a.err
#SBATCH --open-mode=append

# Best-of-N with the no-memory MC critic on the 30 frozen flow-BC policies (same task index as
# collect_flow_bc_rollouts.sbatch). For each N in NS: EPISODES episodes from fresh start seeds
# SEED_BASE + e (shared by every policy and every N, so N=1 is the paired baseline), SDE chunks
# scored by the critic's ensemble mean, argmax executed. Transitions are saved like the rollouts.
# Requeue-safe: finished batches and finished N are skipped.
#
# Overrides: NS ("1 4 8 16 32"), EPISODES (128), CRITIC, SEED_BASE (20000000), OUT_DIR.

set -euo pipefail
REPO=/home/asingal2/memory_critic
source "$REPO/run_scripts/_forward_term.sh"
echo "Job started on: $(hostname)  Job ID: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-}  Restart count: ${SLURM_RESTART_COUNT:-0}"

DATASETS=(ph mh); SIZES=(10 20 50 100 200); SEEDS=(0 7 42)
i=${SLURM_ARRAY_TASK_ID:-0}
export TASK=square DATASET_TYPE=${DATASETS[$((i / 15))]} N_DEMOS=${SIZES[$(((i % 15) / 3))]} SEED=${SEEDS[$((i % 3))]}
source "$REPO/scripts/_flow_bc_names.sh"
NS=${NS:-"1 4 8 16 32"}
EPISODES=${EPISODES:-128}
CRITIC=${CRITIC:-/data/user_data/$USER/memory_critic/critics/mc_decision_seed0/critic_best.pkl}
SEED_BASE=${SEED_BASE:-20000000}
OUT_DIR=${OUT_DIR:-/data/user_data/$USER/memory_critic/bon/mc_decision_seed0}
echo "Run dir: $RUN_DIR  Critic: $CRITIC  N: $NS"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TMPDIR="/data/user_data/$USER/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$OUT_DIR"
export JAX_PLATFORMS=cpu MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for N in $NS; do
    POLICY_ID="square_${DATASET_TYPE}_${N_DEMOS}demos_seed${SEED}_sde_bon${N}"
    [ -f "$OUT_DIR/$POLICY_ID.json" ] && { echo "skip $POLICY_ID (done)"; continue; }
    uv run --locked python scripts/collect_flow_bc_rollouts.py --run_dir "$RUN_DIR" \
        --policy_id "$POLICY_ID" --out_dir "$OUT_DIR" --episodes "$EPISODES" --n_envs 8 \
        --seed_base "$SEED_BASE" --rng_key 2000 --bon_n "$N" --critic "$CRITIC" &
    st=0
    wait_for_training $! || st=$?
    if [ "$st" -eq 3 ]; then
        echo "[requeue] incomplete (exit 3); requeueing job $SLURM_JOB_ID"
        if scontrol requeue "$SLURM_JOB_ID"; then exit 0; fi
        exit 3
    fi
    [ "$st" -eq 0 ] || { echo "N=$N failed (exit $st)"; exit "$st"; }
done
echo "Job finished."
