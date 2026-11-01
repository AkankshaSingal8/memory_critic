#!/bin/bash
#SBATCH --job-name=ogpo_public
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint=VRAM_48GB
#SBATCH --time=5-00:00:00
#SBATCH --output=/home/asingal2/memory_critic/logs/%x_%j.out
#SBATCH --error=/home/asingal2/memory_critic/logs/%x_%j.err
#SBATCH --open-mode=append

# Official public OGPO (/home/asingal2/OGPO_public, github simchowitzlabpublic/OGPO_public
# @ 0b3be41) on robomimic square, run through its own scripts/ogpo/square.sh with every
# hyperparameter at the repo's value. Only these are passed on top: seed, env (square-mh or
# square-ph), W&B project/group/names, save dir, and preemption_backup_interval.
#
# The public code is on branch memcritic-preempt, which only changes resume behaviour:
#   - atomic rolling-checkpoint writes; resume takes the newest complete checkpoint
#   - an online-phase resume restores the ONLINE optimizers (upstream fell back to the
#     BC learning rate after any online resume)
#   - SIGTERM in the online loop saves a checkpoint and exits
#   - square.sh passes preemption_backup_interval through (upstream default -1 = off)
# The BC phase has no checkpoints upstream, so a preemption during BC restarts BC.
#
#   sbatch --export=ALL,SEED=0,DATASET_TYPE=mh --job-name=ogpo_public_square_mh_seed0 scripts/train_ogpo_public.sbatch
# Extra square.sh overrides go after the script name (e.g. --online_steps=3000 for a smoke).
#
# KIND=bc runs only the public BC phase (the same actor, data pipeline, and BC optimiser as
# OGPO's offline phase) as the pure-BC baseline: online_steps=0, clip_bc=false, the full
# bc_pi_steps=500k, eval every 10k steps (ODE + SDE, 64 episodes each), params saved every 50k.
# N_DEMOS picks a nested demo subset from scripts/make_demo_subsets.py (all = full dataset).
#
# Env: SEED (0), DATASET_TYPE (mh), KIND (ogpo | bc), N_DEMOS (all), RUN_TAG (20261022),
# BACKUP_EVERY (50000 online steps).

set -uo pipefail
REPO=/home/asingal2/memory_critic
export OGPO_PUBLIC=/home/asingal2/OGPO_public
export REAL_UV=/home/asingal2/.local/bin/uv

echo "Job started on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"

SEED=${SEED:-0}
DATASET_TYPE=${DATASET_TYPE:-mh}
RUN_TAG=${RUN_TAG:-20261022}
BACKUP_EVERY=${BACKUP_EVERY:-50000}
KIND=${KIND:-ogpo}
N_DEMOS=${N_DEMOS:-all}
case "$KIND" in
    ogpo) KIND_ARGS=() ;;
    bc)   KIND_ARGS=(--online_steps=0 --clip_bc=false --eval_interval_bc=10000 --save_interval_bc=50000) ;;
    *) echo "ERROR: KIND must be ogpo or bc, got '$KIND'" >&2; exit 1 ;;
esac
DEMOS_TAG=$([ "$N_DEMOS" = all ] && echo "" || echo "_${N_DEMOS}demos")
GROUP="square_${DATASET_TYPE}_${KIND}_public${DEMOS_TAG}"
[[ "$RUN_TAG" == smoke* ]] && GROUP="smoke"
NAME="${KIND}_public_square_${DATASET_TYPE}${DEMOS_TAG}_${RUN_TAG}_seed${SEED}"
ROOT=/data/user_data/$USER/memory_critic/ogpo_public

# Per-run checkpoint root: upstream keys rolling checkpoints on seed + env only, so two runs
# sharing a root (e.g. different tags) would resume each other.
export OGPO_CHECKPOINT_ROOT=$ROOT/ckpt/$NAME
export OGPO_EXP_ROOT=$ROOT/exp
if [ "$N_DEMOS" = all ]; then
    export ROBOMIMIC_DATASET_ROOT=/data/user_data/$USER/robomimic
else
    export ROBOMIMIC_DATASET_ROOT=/data/user_data/$USER/robomimic_subsets/square_${DATASET_TYPE}_${N_DEMOS}
fi
export WANDB_ENTITY=akankshasingal08
export PATH="$REPO/scripts/ogpo_public_shim:$PATH"
export PYTHONUNBUFFERED=1
[ -r "$ROBOMIMIC_DATASET_ROOT/square/$DATASET_TYPE/low_dim_v15.hdf5" ] || { echo "ERROR: dataset missing" >&2; exit 1; }
echo "Run: $NAME  group: $GROUP  checkpoints: $OGPO_CHECKPOINT_ROOT"
# A job preempted after square.sh had already exited 0 still gets requeued, and without this
# guard it retrained from scratch (several BC runs finished 2-5 times). Written on success only.
DONE_MARKER=$OGPO_CHECKPOINT_ROOT/DONE
if [ -e "$DONE_MARKER" ]; then
    echo "Already finished ($(cat "$DONE_MARKER")); not rerunning."
    exit 0
fi
echo "OGPO_public: $(git -C $OGPO_PUBLIC rev-parse --abbrev-ref HEAD)@$(git -C $OGPO_PUBLIC rev-parse --short HEAD)$(git -C $OGPO_PUBLIC diff --quiet || echo '+dirty')"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Own process group, so SIGTERM reaches square.sh's python too; then wait for the whole group,
# because bash exits on SIGTERM at once while python is still writing its checkpoint.
setsid bash "$OGPO_PUBLIC/scripts/ogpo/square.sh" \
    --seed="$SEED" \
    --env_name="square-${DATASET_TYPE}-low_dim" \
    --project=memory-critic \
    --run_group="$GROUP" \
    --run_name="$NAME" \
    --wandb_name="$NAME" \
    --SAVE_DIR="$OGPO_EXP_ROOT" \
    --preemption_backup_interval="$BACKUP_EVERY" \
    "${KIND_ARGS[@]}" \
    "$@" &
PGID=$!
PREEMPTED=0
trap 'PREEMPTED=1; echo "[PREEMPTION] forwarding SIGTERM to process group $PGID"; kill -TERM -- -$PGID 2>/dev/null' TERM INT
wait $PGID
STATUS=$?
while kill -0 -- -$PGID 2>/dev/null; do wait; sleep 5; done
echo "Job finished (square.sh status $STATUS)."
# Online-phase SIGTERM makes python save a checkpoint and exit 0, so status 0 alone does not
# mean the run finished: never mark a run done after a preemption signal.
if [ "$STATUS" -eq 0 ] && [ "$PREEMPTED" -eq 0 ]; then
    mkdir -p "$OGPO_CHECKPOINT_ROOT"
    echo "job ${SLURM_JOB_ID:-none} restart ${SLURM_RESTART_COUNT:-0} $(date -Is)" > "$DONE_MARKER"
fi
exit $STATUS
