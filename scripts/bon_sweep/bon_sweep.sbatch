#!/bin/bash
#SBATCH --job-name=bon_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@300
#SBATCH --open-mode=append
# Partition, QOS, GPU, CPUs, memory, time, array range and log paths come from submit.sh.

# Best-of-N with the no-memory MC critic over BC training (docs/bon_sweep.md). One array task is one
# (dataset, BC dataset size, BC seed, critic seed); it runs every (BC checkpoint T, N) pair:
# 64 episodes from seeds T..T+63 (the start states of eval_sde at T, so BC alone is the N = 1 point),
# pi_base draws N SDE chunks per decision and the critic's ensemble mean keeps one. Up to PARALLEL
# pairs run at once. Finished pairs are skipped, so resubmitting the same array resumes it. At the end
# the task's results go to W&B (scripts/bon_sweep/log_wandb.py).
#
# Task index -> (dataset, size, BC seed, critic seed), critic seed fastest:
#   i = ((d * n_sizes + s) * n_bc_seeds + b) * n_critic_seeds + c

set -uo pipefail
: "${REPO:?submit with scripts/bon_sweep/submit.sh}"
source "$REPO/scripts/bon_sweep/config.sh"
echo "Job started on: $(hostname)  Job ID: ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID:-}  Restart count: ${SLURM_RESTART_COUNT:-0}"

read -ra DS_ <<< "$DATASETS"; read -ra SZ_ <<< "$SIZES"; read -ra BS_ <<< "$BC_SEEDS"; read -ra CS_ <<< "$CRITIC_SEEDS"
i=${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}
c=$(( i % ${#CS_[@]} )); i=$(( i / ${#CS_[@]} ))
b=$(( i % ${#BS_[@]} )); i=$(( i / ${#BS_[@]} ))
s=$(( i % ${#SZ_[@]} )); d=$(( i / ${#SZ_[@]} ))
[ "$d" -lt "${#DS_[@]}" ] || { echo "task index ${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}} out of range (N_TASKS=$N_TASKS)"; exit 2; }
CRITIC_SEED=${CS_[$c]}
export TASK=square DATASET_TYPE=${DS_[$d]} N_DEMOS=${SZ_[$s]} SEED=${BS_[$b]}
SAVE_DIR=$INPUT_ROOT/flow_bc; unset RUN_DIR GROUP NAME REPEAT_ID PROJECT RUN_TAG HORIZON
source "$REPO/scripts/_flow_bc_names.sh"
CRITIC=$INPUT_ROOT/critics/mc_decision_seed$CRITIC_SEED/critic_best.pkl
OUT_DIR=$OUT_ROOT/mc_decision_seed$CRITIC_SEED
DS_ROOT=$DATASET_ROOT/${TASK}_${DATASET_TYPE}_${N_DEMOS}
PREFIX="square_${DATASET_TYPE}_${N_DEMOS}demos_seed${SEED}"
echo "Policy: $PREFIX  critic seed: $CRITIC_SEED  N: $NS  steps: $STEPS"
echo "Run dir: $RUN_DIR"
echo "Critic: $CRITIC  Dataset root: $DS_ROOT  Out: $OUT_DIR"

missing=0
for p in "$RUN_DIR/flags.json" "$CRITIC" "$DS_ROOT/$TASK/$DATASET_TYPE/low_dim_v15.hdf5" "$PYTHON"; do
    [ -e "$p" ] || { echo "missing input: $p"; missing=1; }
done
for T in $STEPS; do [ -e "$RUN_DIR/actor_${T}.pkl" ] || { echo "missing input: $RUN_DIR/actor_${T}.pkl"; missing=1; }; done
[ "$missing" -eq 0 ] || exit 2
[ "${PLAN_ONLY:-0}" = 1 ] && { echo "PLAN_ONLY=1: inputs found, not running"; exit 0; }

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TMPDIR="$SCRATCH_ROOT/bon_sweep_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR" "$OUT_DIR" 2>/dev/null || { export TMPDIR=/tmp/bon_sweep_${SLURM_JOB_ID:-$$}; mkdir -p "$TMPDIR" "$OUT_DIR"; }
export MUJOCO_GL=egl PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Several processes share one GPU: allocate on demand instead of 75% each.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
[ -n "${JAX_PLATFORMS:-}" ] && export JAX_PLATFORMS
DROP=(--drop_transitions); [ "$KEEP_TRANSITIONS" = 1 ] && DROP=()

pids=(); names=(); failed=0
stop() {
    echo "[PREEMPTION] forwarding SIGTERM to ${#pids[@]} rollouts"
    for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null; done
    wait; exit 3
}
trap stop TERM INT

reap() {  # wait for one child; record failures
    local p st j
    wait -n -p p; st=$?
    for j in "${!pids[@]}"; do
        if [ "${pids[$j]}" = "$p" ]; then
            [ "$st" -eq 0 ] || { echo "FAILED (exit $st): ${names[$j]}"; failed=$((failed + 1)); }
            unset 'pids[j]' 'names[j]'; pids=("${pids[@]}"); names=("${names[@]}"); break
        fi
    done
}

for T in $STEPS; do
    for N in $NS; do
        POLICY_ID="${PREFIX}_step${T}_sde_bon${N}"
        [ -f "$OUT_DIR/$POLICY_ID.json" ] && continue
        while [ "${#pids[@]}" -ge "$PARALLEL" ]; do reap; done
        "$PYTHON" scripts/collect_flow_bc_rollouts.py --run_dir "$RUN_DIR" --checkpoint "actor_${T}.pkl" \
            --policy_id "$POLICY_ID" --out_dir "$OUT_DIR" --dataset_root "$DS_ROOT" \
            --episodes "$EPISODES" --n_envs "$N_ENVS" --seed_base "$T" --rng_key "$RNG_KEY" \
            --bon_n "$N" --critic "$CRITIC" "${DROP[@]}" > "$LOG_DIR/runs/$POLICY_ID.c$CRITIC_SEED.log" 2>&1 &
        pids+=($!); names+=("$POLICY_ID")
        echo "started $POLICY_ID (pid $!)"
    done
done
while [ "${#pids[@]}" -gt 0 ]; do reap; done
trap - TERM INT

for T in $STEPS; do for N in $NS; do
    f="$OUT_DIR/${PREFIX}_step${T}_sde_bon${N}.json"
    [ -f "$f" ] && "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"T={sys.argv[2]:>6} N={sys.argv[3]:>2}  success {d['success_rate']:.3f}  mean len {d['mean_length']:.0f}\")" "$f" "$T" "$N"
done; done

"$PYTHON" scripts/bon_sweep/log_wandb.py --out_dir "$OUT_DIR" --prefix "$PREFIX" --steps $STEPS --ns $NS \
    --critic_seed "$CRITIC_SEED" --critic "$CRITIC" --run_dir "$RUN_DIR" --episodes "$EPISODES" \
    || echo "W&B logging failed; rerun scripts/bon_sweep/log_wandb.py by hand (see docs/bon_sweep.md)"
rm -rf "$TMPDIR"
[ "$failed" -eq 0 ] || { echo "$failed rollouts failed; per-run logs in $LOG_DIR/runs/"; exit 1; }
echo "Job finished."
