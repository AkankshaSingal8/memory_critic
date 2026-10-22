#!/bin/bash
# Submit flow-BC training runs, each chained to its CPU eval job (afterok), for every
# DEMOS x SEEDS combination of one TASK/DATASET_TYPE. Mirrors SafeVaDaR
# run_scripts/submit_bc_full_fleet.sh.
#
#   scripts/submit_flow_bc.sh                                   # square ph, 10/20/50/100/200 demos, seed 0
#   DRY_RUN=1 scripts/submit_flow_bc.sh                         # print the sbatch lines only
#   TASK=tool_hang DEMOS="50 200" SEEDS="0 7 42" scripts/submit_flow_bc.sh
#
# Subsets must exist first (scripts/make_demo_subsets.py); DEMOS=all uses the full download.
# Env: TASK (square), DATASET_TYPE (ph), DEMOS ("10 20 50 100 200"), SEEDS ("0"), RUN_TAG,
# HORIZON, OFFLINE_STEPS (passed through when set).
set -euo pipefail
REPO=/home/asingal2/memory_critic
cd "$REPO"
mkdir -p logs
TASK=${TASK:-square}
DATASET_TYPE=${DATASET_TYPE:-ph}
DEMOS=${DEMOS:-"10 20 50 100 200"}
SEEDS=${SEEDS:-"0"}
DRY_RUN=${DRY_RUN:-0}
# --export=ALL would let these silently override _flow_bc_names.sh in both jobs.
for v in ROBOMIMIC_DATASET_ROOT GROUP RUN_DIR REPEAT_ID SAVE_DIR PROJECT; do
    [ -z "${!v:-}" ] || { echo "ERROR: $v is set in this shell (${!v}); unset it first" >&2; exit 1; }
done

n=0
for N in $DEMOS; do
    for S in $SEEDS; do
        EXP="ALL,TASK=$TASK,DATASET_TYPE=$DATASET_TYPE,N_DEMOS=$N,SEED=$S${RUN_TAG:+,RUN_TAG=$RUN_TAG}${HORIZON:+,HORIZON=$HORIZON}${OFFLINE_STEPS:+,OFFLINE_STEPS=$OFFLINE_STEPS}"
        JOB="${TASK}_${DATASET_TYPE}_${N}demos_seed${S}"
        # The dataset the training job will read (same resolution as _flow_bc_names.sh).
        ( TASK=$TASK DATASET_TYPE=$DATASET_TYPE N_DEMOS=$N SEED=$S; source scripts/_flow_bc_names.sh
          echo "$JOB: dataset $ROBOMIMIC_DATASET_ROOT  run dir $RUN_DIR" )
        TRAIN=(sbatch --parsable --export="$EXP" --job-name="flow_bc_$JOB" scripts/train_flow_bc.sbatch)
        if [ "$DRY_RUN" = 1 ]; then
            echo "${TRAIN[*]}"
            echo "sbatch --parsable --dependency=afterok:<train> --export=$EXP --job-name=flow_bc_eval_$JOB scripts/eval_flow_bc.sbatch"
            continue
        fi
        tid=$("${TRAIN[@]}")
        eid=$(sbatch --parsable --dependency=afterok:"$tid" --export="$EXP" \
              --job-name="flow_bc_eval_$JOB" scripts/eval_flow_bc.sbatch)
        echo "$JOB: train=$tid eval=$eid"
        n=$((n + 1))
    done
done
[ "$DRY_RUN" = 1 ] || echo "submitted $n training + $n eval jobs"
