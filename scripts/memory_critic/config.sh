# Settings for the IMPRINT memory critic pipeline (docs/imprint_memory_critic.md). Sourced by submit.sh
# and run.sbatch. Every value can be overridden from the environment, e.g.
#   CRITIC_SEEDS="0" STAGE1_STEPS=20000 scripts/memory_critic/submit.sh

# --- what to run
INITS=${INITS:-"scratch ft"}               # Stage-1 init: scratch, or ft = fine-tune the released RoboCasa recurrent encoder
FEATURES=${FEATURES:-"mean progress"}      # pooled memory fed to the critic (imprint_bridge/cache_memory.py)
CRITIC_SEEDS=${CRITIC_SEEDS:-"0 1 2"}
BASELINES=${BASELINES:-"nomem clock"}      # same critic, no memory / + t/400 (controls, same seeds)

# --- Stage 1 (imprint_bridge/train_stage1.py)
STAGE1_STEPS=${STAGE1_STEPS:-40000}
STAGE1_BATCH=${STAGE1_BATCH:-64}
STAGE1_LR=${STAGE1_LR:-1e-4}
RELEASED_REPO=${RELEASED_REPO:-Amazon-FAR/imprint_models}
RELEASED_FILE=${RELEASED_FILE:-simulation/robocasa/recurrent/stage1/stage1_step160000.npz}

# --- critic (scripts/train_mc_critic.py; everything else at the no-memory critic's defaults)
CRITIC_STEPS=${CRITIC_STEPS:-200000}
MEM_PCA=${MEM_PCA:-64}

# --- inputs / outputs. DATA_DIR must be readable by the submitting account: from another account
# than asingal2, grant it once (setfacl -R -m u:<user>:rX,d:u:<user>:rX on the rollouts folder) or copy it.
DATA_DIR=${DATA_DIR:-/data/user_data/asingal2/memory_critic/rollouts/flow_bc_target100}
OUT_ROOT=${OUT_ROOT:-/data/user_data/$USER/memory_critic/imprint}
LOG_DIR=${LOG_DIR:-$REPO/logs/memory_critic}

# --- envs: .venv-imprint for Stage 1 and the cache, .venv (uv project) for the critic
IMPRINT_PY=${IMPRINT_PY:-$REPO/.venv-imprint/bin/python}
CRITIC_PY=${CRITIC_PY:-$REPO/.venv/bin/python}

# --- W&B: every run goes to this entity; the submitting account must be logged in with a key that can
# write to it (wandb login, or WANDB_API_KEY in the environment).
WANDB_ENTITY=${WANDB_ENTITY:-akankshasingal08}
WANDB_PROJECT=${WANDB_PROJECT:-memory-critic}
WANDB_MODE=${WANDB_MODE:-online}

# --- Slurm. Stages run one after another (afterok), and each array is throttled, so at most
# MAX_GPUS GPUs are in use at any time.
MAX_GPUS=${MAX_GPUS:-4}
PARTITION=${PARTITION:-general}
QOS=${QOS:-normal}
GRES=${GRES:-gpu:L40S:1}
STAGE1_TIME=${STAGE1_TIME:-2-00:00:00}
CACHE_TIME=${CACHE_TIME:-0-06:00:00}
CRITIC_TIME=${CRITIC_TIME:-0-12:00:00}

# --- derived: critic task tables, one line "<tag> <seed> <extra args>" per array index.
# critic_tasks baseline: no-memory and clock controls (need no Stage 1, run alongside it)
# critic_tasks memory:   one critic per (Stage-1 init, memory feature)
critic_tasks() {
    local s a i f
    for s in $CRITIC_SEEDS; do
        if [ "$1" = baseline ]; then
            for a in $BASELINES; do
                case $a in
                    nomem) echo "nomem $s" ;;
                    clock) echo "clock $s --add_timestep" ;;
                    *) echo "unknown baseline $a" >&2; return 1 ;;
                esac
            done
        else
            for i in $INITS; do for f in $FEATURES; do
                echo "mem_${i}_${f} $s --mem_dir $OUT_ROOT/memory/$i --mem_feature $f --mem_pca $MEM_PCA"
            done; done
        fi
    done
}
