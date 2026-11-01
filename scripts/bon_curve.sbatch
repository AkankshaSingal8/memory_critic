#!/bin/bash
#SBATCH --job-name=bon_curve
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --array=0-14
#SBATCH --output=/home/asingal2/memory_critic/logs/bon/%x_%A_%a.out
#SBATCH --error=/home/asingal2/memory_critic/logs/bon/%x_%A_%a.err
#SBATCH --open-mode=append

# Best-of-N with the MC critic over BC training, for the pi_base-vs-BoN plot (scripts/plot_bon_curve.py).
# Task index = size*3 + seed over (10, 20, 50, 100, 200) x (0, 7, 42), dataset DATASET_TYPE (ph).
# For each checkpoint actor_<T>.pkl in STEPS and each N in NS: 64 episodes from seeds T..T+63, the
# start states of that checkpoint's SDE eval (eval_bc_run.py), so the eval's eval_sde/success at T
# is the matching pi_base (N = 1) point. Requeue-safe (finished (T, N) are skipped).
#
# Overrides: DATASET_TYPE (ph), STEPS ("50000 ... 500000"), NS ("4 8 16 32"), CRITIC, OUT_DIR.

set -euo pipefail
REPO=/home/asingal2/memory_critic
source "$REPO/run_scripts/_forward_term.sh"
echo "Job started on: $(hostname)  Job ID: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-}  Restart count: ${SLURM_RESTART_COUNT:-0}"

SIZES=(10 20 50 100 200); SEEDS=(0 7 42)
i=${SLURM_ARRAY_TASK_ID:-0}
export TASK=square DATASET_TYPE=${DATASET_TYPE:-ph} N_DEMOS=${SIZES[$((i / 3))]} SEED=${SEEDS[$((i % 3))]}
source "$REPO/scripts/_flow_bc_names.sh"
STEPS=${STEPS:-"50000 100000 150000 200000 250000 300000 350000 400000 450000 500000"}
NS=${NS:-"4 8 16 32"}
CRITIC=${CRITIC:-/data/user_data/$USER/memory_critic/critics/mc_decision_seed0/critic_best.pkl}
OUT_DIR=${OUT_DIR:-/data/user_data/$USER/memory_critic/bon_curve/mc_decision_seed0}
echo "Run dir: $RUN_DIR  Critic: $CRITIC  steps: $STEPS  N: $NS"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TMPDIR="/data/user_data/$USER/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$OUT_DIR"
export JAX_PLATFORMS=cpu MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for T in $STEPS; do
    for N in $NS; do
        POLICY_ID="square_${DATASET_TYPE}_${N_DEMOS}demos_seed${SEED}_step${T}_sde_bon${N}"
        [ -f "$OUT_DIR/$POLICY_ID.json" ] && continue
        uv run --locked python scripts/collect_flow_bc_rollouts.py --run_dir "$RUN_DIR" \
            --checkpoint "actor_${T}.pkl" --policy_id "$POLICY_ID" --out_dir "$OUT_DIR" \
            --episodes 64 --n_envs 8 --seed_base "$T" --rng_key 3000 --bon_n "$N" --critic "$CRITIC" &
        st=0
        wait_for_training $! || st=$?
        if [ "$st" -eq 3 ]; then
            echo "[requeue] incomplete (exit 3); requeueing job $SLURM_JOB_ID"
            if scontrol requeue "$SLURM_JOB_ID"; then exit 0; fi
            exit 3
        fi
        [ "$st" -eq 0 ] || { echo "T=$T N=$N failed (exit $st)"; exit "$st"; }
    done
done
echo "Job finished."
