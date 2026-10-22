# Naming shared by train_flow_bc.sbatch, eval_flow_bc.sbatch and submit_flow_bc.sh, so the eval
# job always finds the training run's directory. Source it after setting TASK, DATASET_TYPE,
# N_DEMOS, HORIZON, SEED, RUN_TAG (defaults below).
#
# N_DEMOS=all trains on the full downloaded dataset; a number trains on the nested subset built
# by scripts/make_demo_subsets.py.
TASK=${TASK:-square}
DATASET_TYPE=${DATASET_TYPE:-ph}
N_DEMOS=${N_DEMOS:-all}
HORIZON=${HORIZON:-8}
SEED=${SEED:-0}
# Fixed default (as SafeVaDaR train_bc_full.sh): it is part of the run name and the rolling-
# checkpoint key, so it must not change between a job and its requeue.
RUN_TAG=${RUN_TAG:-20261019}
PROJECT=${PROJECT:-memory-critic}
SAVE_DIR=${SAVE_DIR:-/data/user_data/$USER/memory_critic/flow_bc}

ENV_NAME="${TASK}-${DATASET_TYPE}-low_dim"
if [ "$N_DEMOS" = all ]; then
    ROBOMIMIC_DATASET_ROOT=${ROBOMIMIC_DATASET_ROOT:-/data/user_data/$USER/robomimic}
else
    ROBOMIMIC_DATASET_ROOT=${ROBOMIMIC_DATASET_ROOT:-/data/user_data/$USER/robomimic_subsets/${TASK}_${DATASET_TYPE}_${N_DEMOS}}
fi
GROUP=${GROUP:-${TASK}_${DATASET_TYPE}_flow_bc_${N_DEMOS}demos}
NAME="flow_bc_${TASK}_${DATASET_TYPE}_${N_DEMOS}demos_h${HORIZON}_${RUN_TAG}"
# Rolling checkpoints are keyed on seed+env+algo(+repeat_id+config hash) under the shared
# /data/user_data/$USER/ogpo_checkpoints, which SafeVaDaR also writes to.
REPEAT_ID=${REPEAT_ID:-memcritic_${RUN_TAG}}
# Run dir layout of ogpo/utils/main_helper.setup_experiment_logging:
# <save_dir>/<project>/<group>/<env>/<wandb_name>_qipmseed<seed>_r<repeat_id>
RUN_DIR=${RUN_DIR:-$SAVE_DIR/$PROJECT/$GROUP/$ENV_NAME/${NAME}_qipmseed${SEED}_r${REPEAT_ID}}
