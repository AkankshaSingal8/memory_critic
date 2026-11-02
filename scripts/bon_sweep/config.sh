# Settings for the best-of-N sweep over BC training (docs/bon_sweep.md). Sourced by submit.sh,
# bon_sweep.sbatch and stage_inputs.sh. Every value can be overridden from the environment, e.g.
#   CRITIC_SEEDS="1" NS="8" scripts/bon_sweep/submit.sh
#
# Defaults run from the asingal2 account on Babel: its own /data paths, CPU only on maxlab-cpu, W&B
# from its own login. On another account or machine, point INPUT_ROOT and DATASET_ROOT at the staged
# copy (scripts/bon_sweep/stage_inputs.sh) and OUT_ROOT at somewhere that account can write.

# --- what to run: one array task per BC policy (dataset, size, BC seed); it runs every critic seed
DATASETS=${DATASETS:-"ph mh"}
SIZES=${SIZES:-"10 20 50 100 200"}
BC_SEEDS=${BC_SEEDS:-"0 7 42"}
CRITIC_SEEDS=${CRITIC_SEEDS:-"1 2"}       # critic seed 0 was run already (bon_curve.sbatch, N = 4 8 16 32)
NS=${NS:-"8 16 32"}
STEPS=${STEPS:-"50000 100000 150000 200000 250000 300000 350000 400000 450000 500000"}
EPISODES=${EPISODES:-64}                  # from seeds T..T+63, the start states of the SDE eval at step T
N_ENVS=${N_ENVS:-8}
RNG_KEY=${RNG_KEY:-3000}                  # as bon_curve.sbatch, so critic seeds differ only in the critic
KEEP_TRANSITIONS=${KEEP_TRANSITIONS:-0}   # 1 also keeps <policy_id>.npz (about 8 MB per run)

# --- inputs
# INPUT_ROOT holds flow_bc/<project>/<group>/<env>/<run>/{flags.json,actor_<T>.pkl} and
# critics/mc_decision_seed<C>/critic_best.pkl. DATASET_ROOT holds square_<ds>_<n>/square/<ds>/low_dim_v15.hdf5
# (the env reads only its metadata).
INPUT_ROOT=${INPUT_ROOT:-/data/user_data/asingal2/memory_critic}
DATASET_ROOT=${DATASET_ROOT:-/data/user_data/asingal2/robomimic_subsets}

# --- outputs: <OUT_ROOT>/mc_decision_seed<C>/square_<ds>_<n>demos_seed<S>_step<T>_sde_bon<N>.json,
# the layout the plot scripts read (--root <OUT_ROOT>/mc_decision_seed<C>)
OUT_ROOT=${OUT_ROOT:-/data/user_data/$USER/memory_critic/bon_curve}
LOG_DIR=${LOG_DIR:-$REPO/logs/bon_sweep}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/$USER}

# --- W&B: one run per array task, in your entity and project. Log in on the submitting machine first
# (wandb login); the API key in ~/.netrc is what puts the runs in your account.
WANDB_ENTITY=${WANDB_ENTITY:-akankshasingal08}
WANDB_PROJECT=${WANDB_PROJECT:-memory-critic}
WANDB_GROUP=${WANDB_GROUP:-bon_curve_mc_critic}
WANDB_MODE=${WANDB_MODE:-online}          # offline: sync later with `wandb sync $LOG_DIR/wandb/wandb/offline-run-*`

# --- Slurm. The rollouts run fine on CPU (about 1.5 min each), so the default is asingal2's
# maxlab-cpu (no GPU; 10 jobs at once, MaxTime 6 h). For a GPU partition set GRES too, e.g.
#   PARTITION=general QOS=normal GRES=gpu:1
PARTITION=${PARTITION:-maxlab-cpu}
QOS=${QOS-maxlab_cpu_qos}
GRES=${GRES-}                             # empty = CPU only
PARALLEL=${PARALLEL:-3}                   # rollouts at once in a task
CPUS=${CPUS:-27}                          # PARALLEL * (N_ENVS + 1)
MEM=${MEM:-48G}
TIME=${TIME:-6:00:00}                     # a task takes about 30-45 min
EXTRA_SBATCH=${EXTRA_SBATCH:-}            # e.g. "--exclude=babel-x-1 --mail-type=END"

PYTHON=${PYTHON:-$REPO/.venv/bin/python}
[ -z "$GRES" ] && JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}   # no GPU: skip the CUDA probe

n_words() { set -- $1; echo $#; }
N_TASKS=$(( $(n_words "$DATASETS") * $(n_words "$SIZES") * $(n_words "$BC_SEEDS") ))   # 30; QOS limits allow at most 50 queued
