#!/bin/bash
# Submit the IMPRINT memory critic pipeline (docs/imprint_memory_critic.md). Settings: config.sh.
#
#   scripts/memory_critic/submit.sh            # everything, chained
#   DRY_RUN=1 scripts/memory_critic/submit.sh  # print the jobs and the critic task tables only
#   SKIP_STAGE1=1 ...                          # Stage 1 already done: start at the cache
#
# Jobs (never more than MAX_GPUS GPUs at once):
#   A  stage1          array over INITS                       (len(INITS) GPUs)
#   B  baseline critic array, throttled to MAX_GPUS-len(INITS), alongside A
#   C  cache           array over INITS, after A
#   D  memory critic   array, throttled to MAX_GPUS, after C and after B has ended
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export REPO
source "$REPO/scripts/memory_critic/config.sh"
read -ra INIT_LIST <<< "$INITS"
N_INIT=${#INIT_LIST[@]}
mapfile -t BASE_TASKS < <(critic_tasks baseline)
mapfile -t MEM_TASKS < <(critic_tasks memory)
BASE_THROTTLE=$((MAX_GPUS - N_INIT))
[ "$BASE_THROTTLE" -ge 1 ] || { echo "MAX_GPUS=$MAX_GPUS leaves no GPU for the baselines next to $N_INIT Stage-1 jobs" >&2; exit 1; }

echo "data: $DATA_DIR"
echo "out:  $OUT_ROOT   logs: $LOG_DIR"
echo "W&B:  $WANDB_ENTITY/$WANDB_PROJECT ($WANDB_MODE)"
echo "Slurm: $PARTITION/$QOS $GRES, max $MAX_GPUS GPUs"
echo "Stage 1 inits: $INITS ($STAGE1_STEPS steps, batch $STAGE1_BATCH)"
echo "baseline critics (${#BASE_TASKS[@]}):"; printf '  %s\n' "${BASE_TASKS[@]}"
echo "memory critics (${#MEM_TASKS[@]}):";   printf '  %s\n' "${MEM_TASKS[@]}"
[ "${DRY_RUN:-0}" = 1 ] && exit 0

# preflight: fail here, not hours later on a compute node
[ -x "$IMPRINT_PY" ] || { echo "missing $IMPRINT_PY: build .venv-imprint (docs/imprint.md §0)" >&2; exit 1; }
[ -x "$CRITIC_PY" ]  || { echo "missing $CRITIC_PY: run uv sync --locked --python 3.12 --extra cuda12 --extra robomimic" >&2; exit 1; }
[ -f "$REPO/third_party/imprint/src/mme_vla_suite/__init__.py" ] || { echo "run: git submodule update --init third_party/imprint" >&2; exit 1; }
ls "$DATA_DIR"/square_*.npz > /dev/null || { echo "cannot read rollouts in $DATA_DIR (see config.sh)" >&2; exit 1; }
if [ "$WANDB_MODE" = online ] && ! grep -q "api.wandb.ai" ~/.netrc 2>/dev/null && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "no W&B login (~/.netrc or WANDB_API_KEY): run 'wandb login' with a key for $WANDB_ENTITY" >&2; exit 1
fi

mkdir -p "$LOG_DIR"
submit() {  # submit <name> <time> <array> [sbatch args...] -> job id
    local name=$1 time=$2 array=$3; shift 3
    sbatch --parsable --job-name="$name" --partition="$PARTITION" --qos="$QOS" --gres="$GRES" --time="$time" \
        --array="$array" --export=ALL --output="$LOG_DIR/%x_%A_%a.out" --error="$LOG_DIR/%x_%A_%a.err" "$@" \
        "$REPO/scripts/memory_critic/run.sbatch"
}

DEP_CACHE=()
if [ "${SKIP_STAGE1:-0}" != 1 ]; then
    A=$(STAGE=stage1 submit mc_stage1 "$STAGE1_TIME" "0-$((N_INIT - 1))")
    echo "stage1: $A"; DEP_CACHE=(--dependency="afterok:$A")
fi
B=$(STAGE=critic CRITIC_KIND=baseline submit mc_critic_base "$CRITIC_TIME" "0-$((${#BASE_TASKS[@]} - 1))%$BASE_THROTTLE")
echo "baseline critics: $B"
C=$(STAGE=cache submit mc_cache "$CACHE_TIME" "0-$((N_INIT - 1))" "${DEP_CACHE[@]}")
echo "cache: $C"
D=$(STAGE=critic CRITIC_KIND=memory submit mc_critic_mem "$CRITIC_TIME" "0-$((${#MEM_TASKS[@]} - 1))%$MAX_GPUS" \
    --dependency="afterok:$C,afterany:$B")
echo "memory critics: $D"
echo "watch: squeue --me; logs in $LOG_DIR"
