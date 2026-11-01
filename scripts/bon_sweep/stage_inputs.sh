#!/bin/bash
# Copy everything the sweep reads into one directory, so it can be submitted from another account or
# machine (docs/bon_sweep.md). Run where the inputs are readable (a Babel compute node: /data is not
# mounted on login nodes):
#
#   srun -p maxlab-cpu -q maxlab_cpu_qos -c 2 --mem 4G -t 1:00:00 scripts/bon_sweep/stage_inputs.sh <DEST>
#
# <DEST> gets the same layout as the originals, so on the other side set INPUT_ROOT=<DEST> and
# DATASET_ROOT=<DEST>/robomimic_subsets:
#   <DEST>/flow_bc/<project>/<group>/<env>/<run>/{flags.json,actor_<T>.pkl}   (STEPS only)
#   <DEST>/critics/mc_decision_seed<C>/{critic_best.pkl,metrics.json}
#   <DEST>/robomimic_subsets/square_<ds>_<n>/square/<ds>/low_dim_v15.hdf5
# Uses DATASETS, SIZES, BC_SEEDS, CRITIC_SEEDS and STEPS from config.sh. Add TAR=1 for <DEST>.tar too.
set -euo pipefail
DEST=${1:?usage: stage_inputs.sh <DEST>}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$REPO/scripts/bon_sweep/config.sh"
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)

copy() {  # copy <src> to the same path relative to <src_root> under <dst_root>
    local src=$1 src_root=$2 dst_root=$3
    local rel=${src#"$src_root"/}
    mkdir -p "$dst_root/$(dirname "$rel")"
    cp -p "$src" "$dst_root/$rel"
}

for C in $CRITIC_SEEDS; do
    for f in critic_best.pkl metrics.json; do
        copy "$INPUT_ROOT/critics/mc_decision_seed$C/$f" "$INPUT_ROOT" "$DEST"
    done
done
for ds in $DATASETS; do for n in $SIZES; do
    copy "$DATASET_ROOT/square_${ds}_${n}/square/${ds}/low_dim_v15.hdf5" "$DATASET_ROOT" "$DEST/robomimic_subsets"
    for S in $BC_SEEDS; do
        (export TASK=square DATASET_TYPE=$ds N_DEMOS=$n SEED=$S SAVE_DIR=$INPUT_ROOT/flow_bc
         unset RUN_DIR GROUP NAME REPEAT_ID PROJECT RUN_TAG HORIZON
         source "$REPO/scripts/_flow_bc_names.sh"
         copy "$RUN_DIR/flags.json" "$INPUT_ROOT" "$DEST"
         for T in $STEPS; do copy "$RUN_DIR/actor_${T}.pkl" "$INPUT_ROOT" "$DEST"; done)
    done
done; done
echo "staged into $DEST:"
du -sh "$DEST"/* "$DEST"
find "$DEST" -type f | wc -l
if [ "${TAR:-0}" = 1 ]; then
    tar -C "$(dirname "$DEST")" -cf "$DEST.tar" "$(basename "$DEST")"
    ls -la "$DEST.tar"
fi
