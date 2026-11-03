# Settings for experiment B, OGPO with an IMPRINT memory (docs/ogpo_memory_plan.md). Sourced by submit.sh
# and run.sbatch; every value can be overridden from the environment.
#
# The open decisions from the plan have NO default here: submit.sh refuses to run until they are set.
#   MEMORY_STAGE1_DIR   which Stage-1 encoder (experiment A's  <A OUT_ROOT>/stage1/{scratch,ft})
#   MEMORY_FEATURE      mean | progress
#   MEMORY_PROJECTION   none | pca:<K> (fit from MEMORY_CACHE_DIR at job start) | /path/to/projection.npz
#   ARMS                which of: critic (B1), actor_critic (B2), none (no-memory baseline on this code)
#   TIME                Slurm time limit for one OGPO run (500k BC + 2M online steps)

# --- what to run: one array task per (arm, dataset, seed)
ARMS=${ARMS:-}
DATASETS=${DATASETS:-"ph mh"}
SEEDS=${SEEDS:-"0 7 42"}
N_DEMOS=${N_DEMOS:-100}

# --- memory
MEMORY_STAGE1_DIR=${MEMORY_STAGE1_DIR:-}
MEMORY_FEATURE=${MEMORY_FEATURE:-}
MEMORY_PROJECTION=${MEMORY_PROJECTION:-}
MEMORY_CACHE_DIR=${MEMORY_CACHE_DIR:-}     # needed for pca:<K>: experiment A's <OUT_ROOT>/memory/<init>
MEMORY_CKPT=${MEMORY_CKPT:-stage1_stepbest.npz}

# --- code and envs. OGPO_DIR is the patched OGPO checkout made by setup.sh.
OGPO_DIR=${OGPO_DIR:-$HOME/OGPO_public_memory}
OGPO_PY=${OGPO_PY:-$OGPO_DIR/.venv/bin/python}

# --- data / outputs (the 100-demo subsets from scripts/make_demo_subsets.py)
DATASET_ROOT=${DATASET_ROOT:-/data/user_data/asingal2/robomimic_subsets}
OUT_ROOT=${OUT_ROOT:-/data/user_data/$USER/memory_critic/ogpo_memory}
LOG_DIR=${LOG_DIR:-$REPO/logs/ogpo_memory}

# --- W&B
WANDB_ENTITY=${WANDB_ENTITY:-akankshasingal08}
WANDB_PROJECT=${WANDB_PROJECT:-memory-critic}
WANDB_MODE=${WANDB_MODE:-online}

# --- Slurm: at most MAX_GPUS runs at once
MAX_GPUS=${MAX_GPUS:-4}
PARTITION=${PARTITION:-general}
QOS=${QOS:-normal}
GRES=${GRES:-gpu:L40S:1}
TIME=${TIME:-}
BACKUP_EVERY=${BACKUP_EVERY:-}             # online steps between rolling checkpoints; only if the OGPO base supports it

# --- derived: the task table, one line "<arm> <dataset> <seed>" per array index
ogpo_tasks() {
    local a d s
    for a in $ARMS; do for d in $DATASETS; do for s in $SEEDS; do echo "$a $d $s"; done; done; done
}
