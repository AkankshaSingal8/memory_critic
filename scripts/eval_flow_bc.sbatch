#!/bin/bash
#SBATCH --job-name=flow_bc_eval
#SBATCH --partition=preempt
#SBATCH --qos=preempt_cpu_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=/home/asingal2/memory_critic/logs/%x_%j.out
#SBATCH --error=/home/asingal2/memory_critic/logs/%x_%j.err
#SBATCH --open-mode=append

# CPU evaluation of one finished scripts/train_flow_bc.sbatch run. Ported from SafeVaDaR
# run_scripts/eval_bc_run.sh @ 2c4454c; scripts/safety_pretrain/eval_bc_run.py and
# select_bc_checkpoints.py are verbatim copies. Only the repo path and the run-dir lookup
# (scripts/_flow_bc_names.sh instead of the toolhang CONTROLLER/DATASET mapping) differ.
#
# Every checkpoint (actor_<step>.pkl, else params_<step>.pkl), coarse-to-fine (50k multiples,
# then 25k, then 10k), ODE + SDE, the run's own 64 episodes as 8 envs x 8 (the same start states
# as the in-training 32 x 2 eval), logged into the training run's W&B run against eval/bc_step;
# then the 25/50/75/100% performance-level selection: <run_dir>/selection.json and
# <run_dir>/selected/target_{25,50,75,100}.pkl. target_100.pkl is the BEST checkpoint (highest
# centred 3-point smoothed ODE success; ties to the earliest step), summarised in the W&B run as
# levels/target_100/*. Requeue-safe: confirmed steps are recorded in
# <run_dir>/cpu_eval_state.json and skipped; results are also kept in <run_dir>/cpu_eval/.
#
# Time limit: 7 days is asked for (a long limit only delays scheduling); Slurm sends SIGTERM
# 5 min before it (--signal=B:TERM@300) and on preemption; the evaluator stops at the next
# episode boundary, finishes W&B and exits 3 (incomplete), and this wrapper requeues itself.
#
# Identify the run with RUN_DIR, or with the training job's variables (TASK, DATASET_TYPE,
# N_DEMOS, HORIZON, SEED, RUN_TAG; see scripts/_flow_bc_names.sh). Usually submitted by
# scripts/submit_flow_bc.sh with --dependency=afterok:<training job>.

set -euo pipefail
REPO=/home/asingal2/memory_critic
source "$REPO/run_scripts/_forward_term.sh"
echo "Job started on: $(hostname)  Job ID: $SLURM_JOB_ID  Restart count: ${SLURM_RESTART_COUNT:-0}"

source "$REPO/scripts/_flow_bc_names.sh"
[ -r "$RUN_DIR/flags.json" ] || { echo "ERROR: no flags.json in $RUN_DIR" >&2; exit 1; }
echo "Run dir: $RUN_DIR"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export WANDB_ENTITY=akankshasingal08
export TMPDIR="/data/user_data/$USER/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

${EVAL_PYTHON:-uv run --locked python} scripts/safety_pretrain/eval_bc_run.py --run_dir "$RUN_DIR" \
    --n_envs "${N_ENVS:-8}" ${EXTRA_ARGS:-} &
st=0
wait_for_training $! || st=$?
if [ "$st" -eq 3 ]; then
    echo "[requeue] evaluation incomplete (exit 3); requeueing job $SLURM_JOB_ID"
    if scontrol requeue "$SLURM_JOB_ID"; then exit 0; fi
    echo "[requeue] scontrol requeue failed; exiting 3 (incomplete)" >&2
    exit 3
fi
echo "Job finished (exit $st)."
exit "$st"
