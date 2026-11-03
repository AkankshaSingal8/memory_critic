#!/bin/bash
#SBATCH --job-name=ogpo_qdiag
#SBATCH --partition=maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --array=0-21
#SBATCH --output=/home/asingal2/memory_critic/logs/ogpo_qdiag/%x_%A_%a.out
#SBATCH --error=/home/asingal2/memory_critic/logs/ogpo_qdiag/%x_%A_%a.err

# OGPO's Q-diagnostics (Q vs MC, value drops, UMAP of action chunks coloured by Q) for the final
# checkpoint of each trained OGPO_public run (scripts/ogpo_q_diagnostics.py). CPU only: rollouts of a
# frozen agent, 64 episodes (8 envs x 8). One task per (run, sampler); sampler fastest.
#   sbatch scripts/ogpo_q_diagnostics.sbatch
#   RUNS="ogpo_public_square_ph_20261022_seed7" SAMPLERS=ode sbatch --array=0 scripts/ogpo_q_diagnostics.sbatch
# Then: python scripts/plot_ogpo_q_umap.py (login node, reads the npz files).
set -euo pipefail
REPO=/home/asingal2/memory_critic
OGPO_PUBLIC=${OGPO_PUBLIC:-/home/asingal2/OGPO_public}
RUNS=${RUNS:-"ogpo_public_square_ph_10demos_20261022_seed0 ogpo_public_square_ph_20demos_20261022_seed0
 ogpo_public_square_ph_50demos_20261022_seed0 ogpo_public_square_ph_100demos_20261022_seed0
 ogpo_public_square_ph_20261022_seed0 ogpo_public_square_mh_10demos_20261022_seed0
 ogpo_public_square_mh_20demos_20261022_seed0 ogpo_public_square_mh_50demos_20261022_seed0
 ogpo_public_square_mh_100demos_20261022_seed0 ogpo_public_square_mh_200demos_20261022_seed0
 ogpo_public_square_mh_20261022_seed0"}
SAMPLERS=${SAMPLERS:-"ode sde"}
OUT=${OUT:-$REPO/plots/ogpo_public/q_diagnostics}

read -ra R_ <<< "$(echo $RUNS)"; read -ra S_ <<< "$SAMPLERS"
i=${SLURM_ARRAY_TASK_ID:-0}
RUN=${R_[$((i / ${#S_[@]}))]:?task $i out of range}; SAMPLER=${S_[$((i % ${#S_[@]}))]}
echo "Job started on: $(hostname)  Job ID: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-}  run: $RUN  sampler: $SAMPLER"
[ -f "$OUT/${RUN}_${SAMPLER}.npz" ] && { echo "already done: $OUT/${RUN}_${SAMPLER}.npz"; exit 0; }

export PYTHONPATH=$OGPO_PUBLIC JAX_PLATFORMS=cpu MUJOCO_GL=egl PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl_$$
cd "$OGPO_PUBLIC"
"$OGPO_PUBLIC/.venv/bin/python" "$REPO/scripts/ogpo_q_diagnostics.py" --run "$RUN" --sampler "$SAMPLER" \
    --n_eval_envs 8 --eval_episodes 8 --out_dir "$OUT"
echo "Job finished."
