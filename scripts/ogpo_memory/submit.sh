#!/bin/bash
# Submit experiment B (docs/ogpo_memory_plan.md): one array task per (arm, dataset, seed), at most MAX_GPUS
# at once. Settings: config.sh. The open decisions must be given explicitly, for example
#
#   ARMS="critic actor_critic" MEMORY_STAGE1_DIR=<A>/stage1/ft MEMORY_FEATURE=mean \
#   MEMORY_PROJECTION=pca:64 MEMORY_CACHE_DIR=<A>/memory/ft TIME=2-00:00:00 \
#   DEPEND=afterok:<A cache job id> scripts/ogpo_memory/submit.sh
#
#   DRY_RUN=1 ... prints the task table and stops.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export REPO
set -a
source "$REPO/scripts/ogpo_memory/config.sh"
set +a

missing=()
for v in ARMS TIME; do [ -n "${!v}" ] || missing+=("$v"); done
if [[ " $ARMS " =~ \ (critic|actor_critic)\  ]]; then
    for v in MEMORY_STAGE1_DIR MEMORY_FEATURE MEMORY_PROJECTION; do [ -n "${!v}" ] || missing+=("$v"); done
    [[ "$MEMORY_PROJECTION" == pca:* && -z "$MEMORY_CACHE_DIR" ]] && missing+=(MEMORY_CACHE_DIR)
fi
[ ${#missing[@]} -eq 0 ] || { echo "set these first (open decisions, see config.sh): ${missing[*]}" >&2; exit 1; }
for a in $ARMS; do [[ $a =~ ^(critic|actor_critic|none)$ ]] || { echo "unknown arm $a" >&2; exit 1; }; done

mapfile -t TASKS < <(ogpo_tasks)
echo "OGPO: $OGPO_DIR   out: $OUT_ROOT   logs: $LOG_DIR"
echo "memory: stage1 $MEMORY_STAGE1_DIR  feature $MEMORY_FEATURE  projection $MEMORY_PROJECTION"
echo "Slurm: $PARTITION/$QOS $GRES  time $TIME  max $MAX_GPUS at once   W&B $WANDB_ENTITY/$WANDB_PROJECT"
echo "tasks (${#TASKS[@]}):"; printf '  %s\n' "${TASKS[@]}"
[ "${DRY_RUN:-0}" = 1 ] && exit 0

[ -x "$OGPO_PY" ] || { echo "missing $OGPO_PY: run scripts/ogpo_memory/setup.sh first" >&2; exit 1; }
[ -f "$REPO/third_party/imprint/src/mme_vla_suite/__init__.py" ] || { echo "git submodule update --init third_party/imprint" >&2; exit 1; }
for d in $DATASETS; do
    f=$DATASET_ROOT/square_${d}_${N_DEMOS}/square/$d/low_dim_v15.hdf5
    [ -r "$f" ] || { echo "cannot read $f (scripts/make_demo_subsets.py)" >&2; exit 1; }
done
if [ -z "${DEPEND:-}" ] && [[ " $ARMS " =~ \ (critic|actor_critic)\  ]]; then
    [ -f "$MEMORY_STAGE1_DIR/$MEMORY_CKPT" ] || { echo "no $MEMORY_STAGE1_DIR/$MEMORY_CKPT (or pass DEPEND=afterok:<job>)" >&2; exit 1; }
fi
if [ "$WANDB_MODE" = online ] && ! grep -q "api.wandb.ai" ~/.netrc 2>/dev/null && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "no W&B login: run 'wandb login' with a key for $WANDB_ENTITY" >&2; exit 1
fi

mkdir -p "$LOG_DIR"
DEP=(); [ -n "${DEPEND:-}" ] && DEP=(--dependency="$DEPEND")
J=$(sbatch --parsable --partition="$PARTITION" --qos="$QOS" --gres="$GRES" --time="$TIME" \
    --array="0-$((${#TASKS[@]} - 1))%$MAX_GPUS" --export=ALL \
    --output="$LOG_DIR/%x_%A_%a.out" --error="$LOG_DIR/%x_%A_%a.err" ${DEP[@]+"${DEP[@]}"} \
    "$REPO/scripts/ogpo_memory/run.sbatch")
echo "submitted array $J"
