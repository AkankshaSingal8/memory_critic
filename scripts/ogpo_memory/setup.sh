#!/bin/bash
# Make the patched OGPO checkout for experiment B and build its env.
#
#   scripts/ogpo_memory/setup.sh <existing OGPO_public checkout> [target dir, default $HOME/OGPO_public_memory]
#
# Adds a git worktree of the source checkout at its CURRENT commit (so the same command works on public
# 0b3be41 or on memcritic-preempt), on a new branch `memory-<short sha>`, applies ogpo_public.patch
# there, then runs `uv sync --locked --extra cuda12 --extra robomimic` (OGPO's Makefile command).
# The source checkout itself is not modified.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SRC=${1:?usage: setup.sh <OGPO_public checkout> [target dir]}
DST=${2:-$HOME/OGPO_public_memory}
PATCH=$REPO/scripts/ogpo_memory/ogpo_public.patch

[ -d "$SRC/.git" ] || [ -f "$SRC/.git" ] || { echo "$SRC is not a git checkout" >&2; exit 1; }
if [ -e "$DST" ]; then
    echo "$DST exists already: $(git -C "$DST" rev-parse --abbrev-ref HEAD)@$(git -C "$DST" rev-parse --short HEAD)"
    echo "delete it (git -C $SRC worktree remove $DST) to rebuild"; exit 1
fi
git -C "$SRC" diff --quiet || echo "warning: $SRC has uncommitted changes; the worktree uses its last commit only" >&2
BASE=$(git -C "$SRC" rev-parse --short HEAD)
echo "base: $(git -C "$SRC" rev-parse --abbrev-ref HEAD)@$BASE"
git -C "$SRC" worktree add -b "memory-$BASE" "$DST" HEAD
git -C "$DST" apply --check "$PATCH" || { echo "patch does not apply on $BASE; see docs/ogpo_memory_plan.md" >&2; exit 1; }
git -C "$DST" apply "$PATCH"
git -C "$DST" -c user.name="${GIT_AUTHOR_NAME:-memory_critic}" -c user.email="${GIT_AUTHOR_EMAIL:-none@none}" \
    commit -qam "IMPRINT episode memory for the critic / actor (memory_critic scripts/ogpo_memory/ogpo_public.patch)"
echo "patched: $(git -C "$DST" rev-parse --abbrev-ref HEAD)@$(git -C "$DST" rev-parse --short HEAD)"
(cd "$DST" && uv sync --locked --extra cuda12 --extra robomimic)
echo "env: $DST/.venv"
