#!/bin/bash
#SBATCH --job-name=ogpo_mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=80G
#SBATCH --open-mode=append

# One OGPO run of experiment B (docs/ogpo_memory_plan.md): task = (arm, dataset, seed) from config.sh.
# Runs the patched checkout's scripts/ogpo/square.sh with every hyperparameter at square.sh's value; only
# seed, env, W&B names, save dir and the memory_* settings are passed (as train_ogpo_public.sbatch does).
# Submitted by submit.sh, which exports the settings.
set -uo pipefail
: "${REPO:?submit with scripts/ogpo_memory/submit.sh}"
source "$REPO/scripts/ogpo_memory/config.sh"
TASK=${SLURM_ARRAY_TASK_ID:-0}
mapfile -t TASKS < <(ogpo_tasks)
[ -n "${TASKS[$TASK]:-}" ] || { echo "no task $TASK" >&2; exit 2; }
read -r ARM DS SEED <<< "${TASKS[$TASK]}"
echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}_${TASK}  restart ${SLURM_RESTART_COUNT:-0}"
echo "arm $ARM  dataset $DS  seed $SEED  demos $N_DEMOS"
echo "OGPO: $(git -C "$OGPO_DIR" rev-parse --abbrev-ref HEAD)@$(git -C "$OGPO_DIR" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

GROUP="square_${DS}_ogpo_mem_${ARM}_${N_DEMOS}demos"
NAME="ogpo_mem_${ARM}_square_${DS}_${N_DEMOS}demos_seed${SEED}"
export OGPO_CHECKPOINT_ROOT=$OUT_ROOT/ckpt/$NAME
export OGPO_EXP_ROOT=$OUT_ROOT/exp
export ROBOMIMIC_DATASET_ROOT=$DATASET_ROOT/square_${DS}_${N_DEMOS}
export WANDB_ENTITY WANDB_PROJECT WANDB_MODE
export PYTHONUNBUFFERED=1
# the memory hook (imprint_bridge) and IMPRINT's model files are imported from this repo
export PYTHONPATH="$REPO:$REPO/third_party/imprint/src${PYTHONPATH:+:$PYTHONPATH}"
# square.sh calls `uv run python`: run the patched checkout's synced env directly
export OGPO_PUBLIC=$OGPO_DIR REAL_UV=$(command -v uv)
export PATH="$REPO/scripts/ogpo_public_shim:$PATH"
[ -r "$ROBOMIMIC_DATASET_ROOT/square/$DS/low_dim_v15.hdf5" ] || { echo "dataset missing: $ROBOMIMIC_DATASET_ROOT" >&2; exit 1; }
DONE_MARKER=$OGPO_CHECKPOINT_ROOT/DONE
if [ -e "$DONE_MARKER" ]; then echo "already finished ($(cat "$DONE_MARKER"))"; exit 0; fi

PROJ=$MEMORY_PROJECTION
if [ "$ARM" = none ]; then
    MEM_ARGS=(--memory_mode=none)
else
    case $PROJ in
        none) PROJ=null ;;
        pca:*)   # fit once from experiment A's cache; later tasks reuse the file
            K=${PROJ#pca:}
            PROJ=$OUT_ROOT/projection/${MEMORY_FEATURE}_pca${K}.npz
            mkdir -p "$(dirname "$PROJ")"
            (flock 9; [ -f "$PROJ" ] || PYTHONPATH="$REPO" "$OGPO_PY" -m imprint_bridge.fit_projection \
                --cache_dir "$MEMORY_CACHE_DIR" --feature "$MEMORY_FEATURE" --n_components "$K" --out "$PROJ") \
                9>"$PROJ.lock" || exit 1 ;;
    esac
    MEM_ARGS=(--memory_mode="$ARM" --memory_stage1_dir="$MEMORY_STAGE1_DIR" --memory_ckpt="$MEMORY_CKPT"
              --memory_feature="$MEMORY_FEATURE" --memory_projection="$PROJ")
fi
EXTRA=()
[ -n "$BACKUP_EVERY" ] && EXTRA+=(--preemption_backup_interval="$BACKUP_EVERY")
echo "run $NAME  group $GROUP  memory ${MEM_ARGS[*]}"

# own process group, so SIGTERM reaches square.sh's python; wait for the whole group (as train_ogpo_public.sbatch)
setsid bash "$OGPO_DIR/scripts/ogpo/square.sh" --seed="$SEED" --env_name="square-${DS}-low_dim" \
    --project="$WANDB_PROJECT" --run_group="$GROUP" --run_name="$NAME" --wandb_name="$NAME" \
    --SAVE_DIR="$OGPO_EXP_ROOT" "${MEM_ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"} &
PGID=$!
PREEMPTED=0
trap 'PREEMPTED=1; kill -TERM -- -$PGID 2>/dev/null' TERM INT
wait $PGID
STATUS=$?
while kill -0 -- -$PGID 2>/dev/null; do wait; sleep 5; done
echo "square.sh status $STATUS"
if [ "$STATUS" -eq 0 ] && [ "$PREEMPTED" -eq 0 ]; then
    mkdir -p "$OGPO_CHECKPOINT_ROOT"
    echo "job ${SLURM_JOB_ID:-none} $(date -Is)" > "$DONE_MARKER"
fi
exit $STATUS
