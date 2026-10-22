# Sourced by the sbatch wrappers. A foreground bash dies on SIGTERM, and
# SLURM then tears the step down before Python can write an emergency
# checkpoint. Background the trainer, forward SIGTERM, and wait.
wait_for_training() {
    local child=$1
    _fwd() {
        echo "[PREEMPTION] launcher forwarding SIGTERM to pid $child"
        kill -TERM "$child" 2>/dev/null || true
    }
    trap _fwd TERM INT
    set +e
    # One `wait` returns as soon as the trap fires, while Python is still
    # writing its emergency checkpoint; loop until the trainer has exited.
    local st
    while :; do
        wait "$child"
        st=$?
        if [ "$st" -le 128 ] || ! kill -0 "$child" 2>/dev/null; then break; fi
    done
    set -e
    trap - TERM INT
    return "$st"
}
