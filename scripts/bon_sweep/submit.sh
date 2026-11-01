#!/bin/bash
# Submit the best-of-N sweep (docs/bon_sweep.md). Settings: scripts/bon_sweep/config.sh, each
# overridable from the environment.
#
#   scripts/bon_sweep/submit.sh                  # full sweep: 2 datasets x 5 sizes x 3 BC seeds x 2 critic seeds
#   DRY_RUN=1 scripts/bon_sweep/submit.sh        # print the sbatch command and the task table only
#   CHECK=1 scripts/bon_sweep/submit.sh          # sbatch --test-only (start estimate, nothing queued)
#   ARRAY=0-3 scripts/bon_sweep/submit.sh        # a subset of tasks (also how to resubmit failed ones)
#
# Extra arguments are passed to sbatch, e.g. --dependency=afterok:123.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export REPO
source "$REPO/scripts/bon_sweep/config.sh"
ARRAY=${ARRAY:-0-$((N_TASKS - 1))}

export DATASETS SIZES BC_SEEDS CRITIC_SEEDS NS STEPS EPISODES N_ENVS RNG_KEY KEEP_TRANSITIONS \
       INPUT_ROOT DATASET_ROOT OUT_ROOT LOG_DIR SCRATCH_ROOT PARALLEL PYTHON \
       WANDB_ENTITY WANDB_PROJECT WANDB_GROUP WANDB_MODE

echo "tasks: $N_TASKS  (datasets: $DATASETS | sizes: $SIZES | BC seeds: $BC_SEEDS | critic seeds: $CRITIC_SEEDS)"
echo "per task: steps ($STEPS) x N ($NS), $EPISODES episodes each, $PARALLEL at a time"
echo "inputs: $INPUT_ROOT  datasets: $DATASET_ROOT"
echo "outputs: $OUT_ROOT/mc_decision_seed<C>   logs: $LOG_DIR"
echo "W&B: $WANDB_ENTITY/$WANDB_PROJECT group $WANDB_GROUP ($WANDB_MODE)"
if [ "${DRY_RUN:-0}" = 1 ]; then
    read -ra DS_ <<< "$DATASETS"; read -ra SZ_ <<< "$SIZES"; read -ra BS_ <<< "$BC_SEEDS"; read -ra CS_ <<< "$CRITIC_SEEDS"
    i=0
    for d in "${DS_[@]}"; do for s in "${SZ_[@]}"; do for b in "${BS_[@]}"; do for c in "${CS_[@]}"; do
        echo "  task $i: square_${d}_${s}demos_seed${b}, critic seed $c"; i=$((i + 1))
    done; done; done; done
fi

if [ "$WANDB_MODE" = online ] && ! grep -q "api.wandb.ai" ~/.netrc 2>/dev/null && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "warning: no W&B login found (~/.netrc or WANDB_API_KEY); run 'wandb login' first or results won't reach W&B" >&2
fi

mkdir -p "$LOG_DIR/runs"
cmd=(sbatch --partition="$PARTITION" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --array="$ARRAY"
     --export=ALL --output="$LOG_DIR/%x_%A_%a.out" --error="$LOG_DIR/%x_%A_%a.err")
[ -n "$QOS" ] && cmd+=(--qos="$QOS")
[ -n "$GRES" ] && cmd+=(--gres="$GRES")
[ -n "$EXTRA_SBATCH" ] && cmd+=($EXTRA_SBATCH)
cmd+=("$@" "$REPO/scripts/bon_sweep/bon_sweep.sbatch")
[ "${CHECK:-0}" = 1 ] && cmd=("${cmd[0]}" --test-only "${cmd[@]:1}")
echo "+ ${cmd[*]}"
[ "${DRY_RUN:-0}" = 1 ] && exit 0
"${cmd[@]}"
