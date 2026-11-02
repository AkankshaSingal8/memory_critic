#!/bin/bash
#SBATCH --job-name=memcritic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append

# One stage of the IMPRINT memory critic pipeline (docs/imprint_memory_critic.md), chosen by STAGE:
#   stage1  array over INITS:  train_stage1.py (ft also downloads the released checkpoint)
#   cache   array over INITS:  cache_memory.py on every rollout decision point
#   critic  array over `critic_tasks $CRITIC_KIND` (config.sh; baseline or memory): train_mc_critic.py
# Submitted by submit.sh, which exports the settings from config.sh. Every stage resumes or skips
# finished work, so a failed or preempted task can be resubmitted as is.
set -euo pipefail
: "${REPO:?submit with scripts/memory_critic/submit.sh}" "${STAGE:?}"
source "$REPO/scripts/memory_critic/config.sh"
cd "$REPO"
TASK=${SLURM_ARRAY_TASK_ID:-0}
echo "host $(hostname)  job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}_${TASK}  stage $STAGE  restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi -L || true

export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_DIR="$LOG_DIR"
export TMPDIR="${JOB_TMP_ROOT:-/data/user_data/$USER/tmp}/${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
IMPRINT_PATH="$REPO:$REPO/third_party/imprint/src:$REPO/third_party/imprint/packages/openpi-client/src"
read -ra INIT_LIST <<< "$INITS"
INIT=${INIT_LIST[$TASK]:-}   # stage1 / cache task

case $STAGE in
stage1)
    OUT="$OUT_ROOT/stage1/$INIT"
    EXTRA=()
    if [ "$INIT" = ft ]; then
        CKPT="$OUT_ROOT/released/$RELEASED_FILE"
        if [ ! -f "$CKPT" ]; then
            PYTHONPATH="$IMPRINT_PATH" "$IMPRINT_PY" -c "import sys; from huggingface_hub import hf_hub_download; \
print(hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3]))" \
                "$RELEASED_REPO" "$RELEASED_FILE" "$OUT_ROOT/released"
        fi
        EXTRA=(--init_from "$CKPT")
    fi
    PYTHONPATH="$IMPRINT_PATH" "$IMPRINT_PY" -m imprint_bridge.train_stage1 --data_dir "$DATA_DIR" \
        --out_dir "$OUT" --num_steps "$STAGE1_STEPS" --batch_size "$STAGE1_BATCH" --lr "$STAGE1_LR" \
        --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT" --run_name "imprint_stage1_$INIT" \
        ${EXTRA[@]+"${EXTRA[@]}"}
    ;;
cache)
    PYTHONPATH="$IMPRINT_PATH" "$IMPRINT_PY" -m imprint_bridge.cache_memory --stage1_dir "$OUT_ROOT/stage1/$INIT" \
        --data_dir "$DATA_DIR" --out_dir "$OUT_ROOT/memory/$INIT"
    ;;
critic)
    mapfile -t TASKS < <(critic_tasks "${CRITIC_KIND:?}")
    [ -n "${TASKS[$TASK]:-}" ] || { echo "no critic task $TASK in the $CRITIC_KIND table" >&2; exit 2; }
    read -r TAG SEED ARGS <<< "${TASKS[$TASK]}"
    OUT="$OUT_ROOT/critics/${TAG}_seed${SEED}"
    if [ -f "$OUT/metrics.json" ]; then echo "done already: $OUT"; exit 0; fi
    # shellcheck disable=SC2086  # ARGS is a flag list
    PYTHONPATH="$REPO" "$CRITIC_PY" scripts/train_mc_critic.py --data_dir "$DATA_DIR" --out_dir "$OUT" \
        --seed "$SEED" --steps "$CRITIC_STEPS" --wandb_entity "$WANDB_ENTITY" --wandb_project "$WANDB_PROJECT" \
        --wandb_group critic_mc_imprint --run_name "${TAG}_seed${SEED}" $ARGS
    ;;
*) echo "unknown STAGE $STAGE" >&2; exit 2 ;;
esac
