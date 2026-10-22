#!/bin/bash
#SBATCH --job-name=flow_bc
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/asingal2/memory_critic/logs/%x_%j.out
#SBATCH --error=/home/asingal2/memory_critic/logs/%x_%j.err
#SBATCH --open-mode=append

# BC-only flow-policy training (algo=qipm, training.bc_only=true: only the flow actor is
# trained, no critic) on a robomimic low_dim dataset, for any task in the table below.
#
# Ported from SafeVaDaR run_scripts/train_bc_full.sh @ 2c4454c (branch safety-critic-pretrain).
# ogpo/, envs/, pyproject.toml and uv.lock are byte-identical copies of that commit. The
# python flag block below is that script's, verbatim, except:
#   - env.name / per-task agent flags come from TASK (table below) instead of tool_hang only.
#     no_of_subtasks / discount / subtask_* only affect the (untrained) critic and the dense-
#     reward relabeling, not the BC loss; their values are the ones your per-task SafeVaDaR
#     launchers use (tool_hang: train_bc_full.sh, square: scripts/qipm/square.sh).
#   - controller: the dataset's own OSC controller (ROBOSUITE_CONTROLLER_CONFIG unset, like
#     train_bc_full.sh CONTROLLER=osc) and the stock robomimic v1.5 data under
#     /data/user_data/$USER/robomimic, so no mink safety config / side-car / safety_views.
#   - N_DEMOS picks the dataset: all (the full download) or a nested subset built by
#     scripts/make_demo_subsets.py (same construction as SafeVaDaR make_relabel_subsets.py).
#   - names, project, save_dir, and repeat_id (prefixed memcritic_ so a run here never resumes
#     a SafeVaDaR rolling checkpoint in the shared /data/user_data/$USER/ogpo_checkpoints); all
#     of them come from scripts/_flow_bc_names.sh, shared with the eval job.
# Logging is train_bc_full.sh's: no in-training eval (EVAL_EVERY=-1); this job logs
# offline_agent/* and memory/* to W&B; scripts/eval_flow_bc.sbatch (SafeVaDaR eval_bc_run.py,
# verbatim) then evaluates every actor_<step>.pkl on CPU into the SAME W&B run against
# eval/bc_step (eval/*, eval_sde/*), and saves the best checkpoint as selected/target_100.pkl
# (levels/* in the run summary). scripts/submit_flow_bc.sh submits both, chained.
# Everything that shapes the BC actor/optimisation is unchanged: flow actor (4x512 MLP, no LN),
# horizon_length=8, flow_steps=10, denoiser + constant noise std 0.005, SDE->ODE correction,
# actor lr 3e-4 cosine (warmup 2000, decay 50000, end 2e-5), batch 256, actor_tau=0.005,
# clip_bc=false, 500k steps, deterministic XLA + highest matmul precision on an L40S.
#
# Usually submitted (with its eval job) by scripts/submit_flow_bc.sh; directly:
#   sbatch --export=ALL,TASK=square,N_DEMOS=50,SEED=0 --job-name=flow_bc_square_ph_50demos_seed0 scripts/train_flow_bc.sbatch
#
# Smoke test:
#   sbatch --export=ALL,TASK=square,N_DEMOS=10,RUN_TAG=smoke,OFFLINE_STEPS=200,EVAL_EVERY=100,SAVE_EVERY=100 \
#          --time=00:30:00 --job-name=flow_bc_smoke scripts/train_flow_bc.sbatch
#
# Env overrides: TASK (square), DATASET_TYPE (ph), N_DEMOS (all), SEED (0), RUN_TAG (20261019),
# HORIZON (8), OFFLINE_STEPS (500000), EVAL_EVERY (-1 = never; >0 re-enables in-training evals),
# SAVE_EVERY (10000), VIDEO_EVERY (0), PROJECT (memory-critic), GROUP (see _flow_bc_names.sh).

set -euo pipefail
REPO=/home/asingal2/memory_critic
source "$REPO/run_scripts/_forward_term.sh"

echo "Job started on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Restart count: ${SLURM_RESTART_COUNT:-0}"

source "$REPO/scripts/_flow_bc_names.sh"
OFFLINE_STEPS=${OFFLINE_STEPS:-500000}
EVAL_EVERY=${EVAL_EVERY:--1}
SAVE_EVERY=${SAVE_EVERY:-10000}
VIDEO_EVERY=${VIDEO_EVERY:-0}

# Per-task flags. Add a task by adding a case (no_of_subtasks must match the subgoal list
# in envs/robomimic_utils.py:_compute_subtask_status for that env).
case "$TASK" in
    square)
        NO_OF_SUBTASKS=3; DISCOUNT=0.99;  SUBTASK_BASED_DISCOUNT=false; SUBTASK_DISCOUNTS=null ;;
    tool_hang)
        NO_OF_SUBTASKS=4; DISCOUNT=0.999; SUBTASK_BASED_DISCOUNT=true;  SUBTASK_DISCOUNTS="[0.9,0.99,0.995,0.999]" ;;
    *) echo "ERROR: TASK must be square or tool_hang (add a case to support more), got '$TASK'" >&2; exit 1 ;;
esac
# No in-training evals -> no video env (it would be built and never used).
if [ "$EVAL_EVERY" -lt 0 ]; then VIDEO_EVERY=0; fi
# 32 * 2 = 64 episodes per eval, ODE and SDE each (read from flags.json by the CPU eval job).
EVAL_EPISODES=2
echo "Task: $ENV_NAME  Demos: $N_DEMOS  Seed: $SEED  Horizon: $HORIZON  Run name: $NAME  offline_steps=$OFFLINE_STEPS"
echo "Dataset root: $ROBOMIMIC_DATASET_ROOT  Run dir: $RUN_DIR"

export TMPDIR="/data/user_data/$USER/tmp/${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
# One toolchain on every node: JAX's pip CUDA 12 wheels pinned by uv.lock. Only the driver
# varies; it is logged.
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "L40S" || {
    echo "ERROR: expected an L40S (bit-exact runs need one GPU model)" >&2; exit 1; }

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export WANDB_ENTITY=akankshasingal08
# make_env falls back to the controller recorded in the dataset (OSC).
unset ROBOSUITE_CONTROLLER_CONFIG
export ROBOMIMIC_DATASET_ROOT
[ -r "$ROBOMIMIC_DATASET_ROOT/$TASK/$DATASET_TYPE/low_dim_v15.hdf5" ] || { echo "ERROR: dataset missing" >&2; exit 1; }

# Same environment block as SafeVaDaR scripts/safe_qipm/toolhang.sh (deterministic=true
# requires the XLA flag; BLAS thread counts are read at numpy import).
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0

uv run --locked python ogpo/main.py \
    --algo=qipm \
    --experiment.project=$PROJECT \
    --experiment.run_group=$GROUP \
    --experiment.run_name=$NAME \
    --experiment.wandb_name=$NAME \
    --experiment.seed=$SEED \
    --experiment.log=true \
    --experiment.save_dir=$SAVE_DIR \
    --experiment.deterministic=true \
    --experiment.matmul_precision=highest \
    --experiment.repeat_id=$REPEAT_ID \
    --env.name=$ENV_NAME \
    --env.sparse=false \
    --training.offline_steps=$OFFLINE_STEPS \
    --training.calql_steps=0 \
    --training.q_warmup_steps=0 \
    --training.bc_refine_steps=0 \
    --training.online_steps=0 \
    --training.bc_only=true \
    --eval.n_eval_envs=32 \
    --eval.eval_episodes=$EVAL_EPISODES \
    --eval.bc_sde_eval=true \
    --eval.plot_q_vs_mc=false \
    --eval.video_interval=$VIDEO_EVERY \
    --intervals.log_interval=5000 \
    --intervals.eval_interval_bc=$EVAL_EVERY \
    --intervals.save_interval_bc=$SAVE_EVERY \
    --intervals.bc_preemption_backup_interval=5000 \
    --bc.clip_bc=false \
    --bc.save_actor_only=true \
    --agent.horizon_length=$HORIZON \
    --agent.discount=$DISCOUNT \
    --agent.actor_tau=0.005 \
    --agent.flow_steps=10 \
    --agent.bc_coeff=1.0 \
    --agent.use_bc_regularization=true \
    --agent.use_denoiser=true \
    --agent.use_constant_noise=true \
    --agent.constant_noise_std=0.005 \
    --agent.min_noise_std=0.005 \
    --agent.max_noise_std=0.005 \
    --agent.error_correct_sde_to_ode=true \
    --agent.critic_joint_proprio=false \
    --agent.no_of_subtasks=$NO_OF_SUBTASKS \
    --agent.use_dense_rewards=true \
    --agent.normalize_rewards=true \
    --agent.subtask_based_discount=$SUBTASK_BASED_DISCOUNT \
    --agent.subtask_discounts="$SUBTASK_DISCOUNTS" \
    --agent.value_hidden_dims="[512,512,512,512]" \
    --agent.policy_type=flow \
    --agent.use_constant_scheduler_for_bc=false \
    --agent.actor_scheduler=cosine \
    --agent.critic_scheduler=constant \
    --agent.goal_conditioning_mode=null \
    --agent.goal_conditioned_actor=false \
    --agent.goal_conditioned_critic=false \
    --agent.advantage_gating_mode=grandparent_future \
    --agent.td_loss_mode=gated &
wait_for_training $! || exit $?

echo "Job finished."
