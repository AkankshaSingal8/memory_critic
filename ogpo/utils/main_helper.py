"""Main helper utilities for training pipelines.

Contains shared functions used across all main_*.py files:
- Environment creation and vectorization
- Dataset processing
- Checkpoint management
- Logging utilities
- Q-value bounds configuration
"""

import glob
import hashlib
import json
import os
import pickle
import random
import signal
import shutil
import tempfile
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from ogpo.agents.modules.flax_utils import save_agent
from ogpo.utils.datasets import Dataset, ReplayBuffer
from ogpo.utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb


# =============================================================================
# Logging Helper
# =============================================================================

class LoggingHelper:
    """Helper class for unified logging to CSV and WandB."""

    def __init__(
        self,
        csv_loggers: Dict[str, CsvLogger],
        wandb_logger: Optional[Any] = None,
        log_enabled: bool = True,
    ):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.log_enabled = log_enabled
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data: Dict, prefix: str, step: int):
        """Log data to CSV and WandB."""
        if not self.log_enabled:
            return
        assert prefix in self.csv_loggers, f"Unknown prefix: {prefix}"
        self.csv_loggers[prefix].log(data, step=step)
        if self.wandb_logger is not None:
            self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

    def close(self):
        """Close all CSV loggers."""
        for csv_logger in self.csv_loggers.values():
            if csv_logger is not None:
                csv_logger.close()


_OGBENCH_PREFIXES = (
    'scene-', 'cube-', 'puzzle-', 'humanoidmaze-', 'antmaze-', 'pointmaze-',
    'antsoccer-', 'powderworld-',
)


def _is_ogbench_env(env_name: str) -> bool:
    return any(env_name.startswith(p) for p in _OGBENCH_PREFIXES)


def _make_ogbench_env(env_name: str, seed: int = 0, **kwargs):
    """Create an OGBench env, applying the same name transform as ogbench_utils."""
    import gymnasium
    import ogbench  # noqa: F401  — registers OGBench envs with gymnasium
    splits = env_name.split('-')
    if 'singletask' in splits:
        pos = splits.index('singletask')
        gym_name = '-'.join(splits[:pos - 1] + splits[pos:])
    else:
        gym_name = env_name
    return gymnasium.make(gym_name)


def get_env_maker(env_name: str) -> Callable:
    """Get the appropriate make_env function for the environment."""
    from envs.env_utils import is_robomimic_env
    from envs.env_utils import GYMNASIUM_ROBOTICS_ADROITHAND_ENVS

    if env_name.startswith('robocasa_'):
        from envs.robocasa_utils import make_env
    elif env_name.startswith('libero_'):
        from envs.libero_utils import make_env
    elif is_robomimic_env(env_name):
        from envs.robomimic_utils import make_env
    elif env_name in GYMNASIUM_ROBOTICS_ADROITHAND_ENVS:
        from envs.gymnasium_robotics_utils import make_env
    elif env_name.startswith('D4RL/kitchen'):
        from envs.franka_kitchen_utils import make_env
    elif env_name.startswith('stacking') or env_name.startswith('gate_insertion'):
        from envs.d3il_utils import make_env
    elif env_name.startswith('pusht'):
        from envs.pusht_utils import make_env
    elif _is_ogbench_env(env_name):
        make_env = _make_ogbench_env
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return make_env


def make_vectorized_env(
    env_name: str,
    num_envs: int,
    seed: Optional[int] = None,
    rew_fn: str = 'sparse',
    dense_reward_wrapper: bool = False,
    use_dense_reward: bool = True,
    no_of_subtasks: int = 3,
    normalize_rewards: bool = False,
    subtask_dag: Optional[dict] = None,
    goal_conditioning_mode: Optional[str] = None,
    anchor_demo: Optional[str] = None,
    obstacle: Optional[dict] = None,
    include_joint_proprio: bool = False,
) -> AsyncVectorEnv:
    """Create vectorized environment."""
    from envs.env_utils import is_robomimic_env
    make_env = get_env_maker(env_name)

    # Goal conditioning applies to robomimic low-dim envs, robocasa, and libero envs.
    gc_supported = is_robomimic_env(env_name) or env_name.startswith('robocasa_') or env_name.startswith('libero_')
    active_gc_mode = goal_conditioning_mode if gc_supported else None
    is_ogbench = _is_ogbench_env(env_name)

    def env_fn(env_idx):
        if seed is None:
            env_seed = np.random.randint(0, 1000000)
        else:
            env_seed = seed + env_idx + 60000

        if env_name.startswith('robocasa_'):
            env = make_env(
                env_name, seed=env_seed, anchor_demo=anchor_demo,
                goal_conditioning_mode=active_gc_mode,
                no_of_subtasks=no_of_subtasks,
            )
        elif env_name.startswith('libero_'):
            env = make_env(
                env_name, seed=env_seed,
                goal_conditioning_mode=active_gc_mode,
                no_of_subtasks=no_of_subtasks,
            )
        elif env_name.startswith("pusht"):
            env = make_env(env_name, seed=env_seed, rew_fn=rew_fn, obstacle_cfg=obstacle)
        elif is_robomimic_env(env_name) and active_gc_mode is not None:
            env = make_env(
                env_name,
                seed=env_seed,
                goal_conditioning_mode=active_gc_mode,
                no_of_subtasks=no_of_subtasks,
                include_joint_proprio=include_joint_proprio,
            )
        elif is_robomimic_env(env_name):
            # Same arg profile as the generic fallthrough, plus the obs-widening
            # flag (eval envs must match the train env/dataset obs layout).
            env = make_env(env_name, seed=env_seed,
                           include_joint_proprio=include_joint_proprio)
        else:
            env = make_env(env_name, seed=env_seed)

        if dense_reward_wrapper:
            if is_ogbench:
                # OGBench envs need to be wrapped with the dense reward wrapper
                from envs.ogbench_dense_wrapper import OGBenchDenseRewardWrapper
                env = OGBenchDenseRewardWrapper(env)
                env.configure_dense_reward(
                    use_dense_reward=use_dense_reward,
                    normalize_rewards=normalize_rewards,
                    no_of_subtasks=no_of_subtasks,
                    goal_conditioning_mode=goal_conditioning_mode,
                )
            elif hasattr(env, 'configure_dense_reward'):
                env.configure_dense_reward(
                    use_dense_reward=True,
                    normalize_rewards=normalize_rewards,
                    no_of_subtasks=no_of_subtasks,
                    subtask_dag=subtask_dag,
                )
        return env

    vec_env = AsyncVectorEnv(
        [lambda idx=i: env_fn(idx) for i in range(num_envs)],
        autoreset_mode=AutoresetMode.DISABLED
    )
    return vec_env


# =============================================================================
# Determinism (Tier A) and run provenance (Tier C)
# =============================================================================

# The XLA flags a bit-exact run needs. Python cannot set them from main(): XLA
# reads XLA_FLAGS at BACKEND INITIALIZATION, which the first jax op triggers,
# and an os.environ write inside main() would silently stop working the moment
# any earlier import touched a device. So the launcher exports them (see the
# active launchers under scripts/) and main() verifies them — see
# resolve_determinism_settings for why BOTH are required. MUST match the export
# line in scripts/*/toolhang.sh.
DETERMINISM_XLA_FLAGS = (
    '--xla_gpu_deterministic_ops=true',  # deterministic reduction/atomic order
    '--xla_gpu_autotune_level=0',        # no compile-time GEMM kernel timing
)
# The exact tail of the launchers' export line, and of the raise message below.
DETERMINISM_XLA_FLAGS_STR = ' '.join(DETERMINISM_XLA_FLAGS)

# Sentinel for a provenance field that could not be captured. It has to be a
# CONSTANT: the provenance block feeds compute_config_hash, so a value that
# varied per attempt (an exception message, a path, a timestamp) would make the
# resume guard nondeterministic and refuse every resume.
PROVENANCE_UNAVAILABLE = 'unavailable'
# Sentinel for a field that does not exist for this env family (the robomimic
# HDF5 identity of an OGBench/PushT run). Deterministic, and distinguishable
# from a capture that failed.
PROVENANCE_NOT_APPLICABLE = 'not_applicable'


def resolve_determinism_settings(config: Dict) -> Optional[str]:
    """Matmul precision to pin for this run, or None to keep the platform default.

    Tier A determinism has two halves that necessarily live in two places:
      * ``jax_default_matmul_precision`` is a Python knob read at TRACE time, so
        main() can set it — this function returns what to set.
      * ``DETERMINISM_XLA_FLAGS`` are read by XLA before main() runs, so they
        have to come from the launcher's environment — this function only checks
        that they did, and raises with the exact export line (naming the missing
        tokens) when they did not. A run that CLAIMS determinism but did not get
        the flags must not start: its results would be non-repeatable while
        flags.json said otherwise.

    BOTH XLA flags are required. ``--xla_gpu_deterministic_ops=true`` alone was
    the 2026-08-24 requirement and it is not sufficient: it fixes reduction and
    atomic ORDER but leaves GEMM autotuning on, and autotuning picks a cuBLAS
    kernel by TIMING candidates at compile time, per process — so the winner
    depends on what else the machine was doing. Round 1 of the GPU acceptance
    run (2026-08-25, twin seed-42 L40S jobs) failed bit-exactness exactly there,
    first diverging by one eval episode at BC step 5000, hence
    ``--xla_gpu_autotune_level=0``.

    ``matmul_precision`` applies on its own when ``deterministic`` is false —
    pinning precision without the XLA flags is a legitimate weaker request, and
    the YAML default (null) keeps today's behavior untouched.
    """
    experiment = config['experiment']
    matmul_precision = experiment['matmul_precision']
    if not experiment['deterministic']:
        return matmul_precision

    xla_flags = os.environ.get('XLA_FLAGS', '')
    # Whole-token match, not a substring: XLA_FLAGS is a whitespace-separated
    # flag list, and a substring test also accepts
    # `--xla_gpu_deterministic_ops=truthy` (or `...=true_but_not_really`),
    # which XLA does NOT parse as the flag this run is claiming to have.
    present = xla_flags.split()
    missing = [flag for flag in DETERMINISM_XLA_FLAGS if flag not in present]
    if missing:
        raise ValueError(
            f"experiment.deterministic=true requires {DETERMINISM_XLA_FLAGS_STR} "
            f"in XLA_FLAGS, but XLA_FLAGS={xla_flags!r} is missing "
            f"{' '.join(missing)}. XLA reads these at backend initialization, "
            f"before main() runs, so they cannot be set from Python — add to the "
            f"launcher, beside MUJOCO_GL:\n"
            f'    export XLA_FLAGS="${{XLA_FLAGS:-}} {DETERMINISM_XLA_FLAGS_STR}"\n'
            f"or set experiment.deterministic=false."
        )
    # 'highest' == F32_F32_F32: no TF32 tensor-core path, the only setting under
    # which two f32 dots on identical inputs agree bit-for-bit.
    return matmul_precision or 'highest'


def _describe_safe_robosuite() -> Dict[str, Any]:
    """Commit and dirty flag of the editable ../safe-robosuite checkout.

    pyproject pins robosuite as `{ path = "../safe-robosuite", editable = true }`
    and uv.lock records only that path — no commit, no hash — while that fork
    supplies the safe controller, the collision margins and the safety-cost
    predicates, i.e. the entire experimental treatment. Located via
    ``robosuite.__file__`` rather than a hardcoded ``../safe-robosuite`` (Q5
    path portability).
    """
    try:
        import subprocess

        import robosuite

        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(robosuite.__file__)))

        # The commit is HASHED, so a transient NFS stall that turns it into
        # the sentinel moves a requeued run to a different checkpoint dir.
        # Retry before giving up.
        def _git(*args):
            for attempt in range(3):
                try:
                    return subprocess.run(
                        ['git', '-C', repo_dir, *args],
                        capture_output=True, text=True, check=True, timeout=60,
                    ).stdout
                except subprocess.TimeoutExpired:
                    if attempt == 2:
                        raise
                    time.sleep(5)

        commit = _git('rev-parse', 'HEAD').strip()
        porcelain = _git('status', '--porcelain')
        return {
            'safe_robosuite_commit': commit,
            'safe_robosuite_dirty': bool(porcelain.strip()),
        }
    except Exception as e:
        # best-effort: provenance telemetry; stable sentinel keeps config hash deterministic
        print(f"[provenance] safe-robosuite git capture unavailable: {e}")
        return {
            'safe_robosuite_commit': PROVENANCE_UNAVAILABLE,
            'safe_robosuite_dirty': PROVENANCE_UNAVAILABLE,
        }


def _describe_safevadar(controller_config: Optional[str]) -> Dict[str, Any]:
    """This repo's commit and dirty flag, and the sha256 of the controller JSON.

    A requeued job runs whatever ogpo/ and configs/ hold at requeue time, so
    these record which code and safety config each attempt actually ran. All
    three are excluded from the config hash (CONFIG_HASH_EXCLUDED_KEYS): they
    are a record, not a resume guard.
    """
    import hashlib
    import subprocess
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out: Dict[str, Any] = {}
    try:
        git = lambda *a: subprocess.run(['git', '-C', repo_dir, *a], capture_output=True,
                                        text=True, check=True, timeout=60).stdout
        out['safevadar_commit'] = git('rev-parse', 'HEAD').strip()
        out['safevadar_dirty'] = bool(git('status', '--porcelain', '--untracked-files=no').strip())
    except Exception as e:
        print(f"[provenance] SafeVaDaR git capture unavailable: {e}")
        out['safevadar_commit'] = PROVENANCE_UNAVAILABLE
        out['safevadar_dirty'] = PROVENANCE_UNAVAILABLE
    try:
        with open(controller_config, 'rb') as f:
            out['robosuite_controller_config_sha256'] = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        out['robosuite_controller_config_sha256'] = (
            None if controller_config is None else PROVENANCE_UNAVAILABLE)
    return out


def collect_run_provenance(env_name: str, is_robomimic: bool) -> Dict[str, Any]:
    """Resolved run inputs that the config alone does not identify.

    flags.json and the wandb config are both ``get_flag_dict(config)``, i.e. the
    resolved YAML+CLI config and nothing else — so the two environment
    variables that DEFINE a SafeVADAR experiment (which composite controller
    publishes the safety signal, which traj-count HDF5 backs
    ``tool_hang-ph-low_dim``) and the commit of the editable ../safe-robosuite
    fork were recorded nowhere. main() injects this block into the TOP-LEVEL
    config (never config['agent'], which is frozen after create()) before
    setup_experiment_logging and compute_config_hash, so one dict covers
    flags.json, wandb, and the resume guard's digest.

    Every value is deterministic for a given run environment; the two "missing"
    cases use the module's constant sentinels rather than error text.

    Args:
        env_name: the resolved env.name.
        is_robomimic: whether the offline data is a robomimic HDF5 — the only
            family with a resolvable dataset identity. Other families record
            PROVENANCE_NOT_APPLICABLE rather than inventing a capture path.
    """
    provenance: Dict[str, Any] = {
        # None, not a sentinel: "unset" is a real and reproducible state —
        # robosuite then keeps the dataset-recorded controller (no safety
        # filter) and the dataset root falls back to /data/user_data/$USER.
        'robosuite_controller_config': os.environ.get('ROBOSUITE_CONTROLLER_CONFIG'),
        'robomimic_dataset_root': os.environ.get('ROBOMIMIC_DATASET_ROOT'),
    }
    if is_robomimic:
        # Lazy: envs.robomimic_utils pulls in robomimic/robosuite, which the
        # OGBench and LIBERO venvs need not have.
        from envs.robomimic_utils import describe_dataset
        provenance.update(describe_dataset(env_name))
    else:
        provenance.update({
            'dataset_path': PROVENANCE_NOT_APPLICABLE,
            'dataset_size_bytes': PROVENANCE_NOT_APPLICABLE,
            'dataset_mtime_ns': PROVENANCE_NOT_APPLICABLE,
            'dataset_transitions': PROVENANCE_NOT_APPLICABLE,
        })
    provenance.update(_describe_safe_robosuite())
    provenance.update(_describe_safevadar(provenance['robosuite_controller_config']))
    return provenance


# =============================================================================
# Rolling checkpoints
# =============================================================================

# Node-local checkpoint roots. Module constants (not inline literals) so tests
# can point the search at fake roots — neither path exists off the cluster.
SCRATCH_ROOT = '/scratch'
COMPUTE_ROOT = '/compute'

# Shared (cross-node) checkpoint root — the PRIMARY write target when it exists.
#
# /scratch is node-local xfs on an NVMe device (a different /dev/nvme*p2 per
# node), and the /compute/<node>/ cross-mount that was meant to reach it turns
# out to be all but empty on this cluster: three probe jobs on three nodes saw
# one entry between them. The consequence, measured over 178 job logs: 2223
# rolling checkpoints written, 0 occurrences of "Found checkpoint on remote
# node", 4 successful loads (all same-node), 31 runs that gave up and restarted
# from step 0. On a PreemptMode=REQUEUE partition — where a requeue almost
# always relocates — the rolling checkpoint was effectively write-only.
#
# /data/user_data is NFS (nas5:/srv/nfs/data/user_data/$USER) and IS visible
# from every node, verified by cross-node marker exchange in
# scripts/repro/fs_probe.sh. Writing the multi-GB buffer there costs NFS
# bandwidth the node-local design was avoiding, but it buys the only thing the
# checkpoint exists for: being readable by the job that resumes.
#
# Auto-detected rather than mandatory so this stays a no-op off-cluster (the
# path does not exist on login nodes or in CI) — tests patch it like the other
# two roots. Override with OGPO_SHARED_CHECKPOINT_ROOT.
SHARED_ROOT = os.environ.get('OGPO_SHARED_CHECKPOINT_ROOT', '/data/user_data')

# On-disk schema of a rolling checkpoint. Bump whenever a field becomes
# REQUIRED. v2 (2026-08-24 reproducibility hardening) added `episode_counter`
# to <step>_meta.json, `dataset_rng` to <step>_rng.pkl, and a sibling
# <step>_config_hash.json. load_rolling_checkpoint REFUSES anything older
# instead of defaulting the missing fields: a defaulted episode_counter
# re-walks the env's initial-state sequence and a rebuilt dataset_rng replays
# the batch stream from step 1 — the exact bugs these fields exist to fix.
# main.py's resume try/except turns the refusal into a clean fresh start.
CHECKPOINT_SCHEMA_VERSION = 2

# Config keys the resume guard's hash ignores, per
# docs/changes/2026-08-24-reproducibility-hardening/BLAST-RADIUS.md §C3. A code
# invariant, not a user knob, so it lives here rather than in YAML. Everything
# not listed here is hashed, so a NEW config section is guarded by default.
#   eval        - no effect on the trained parameters
#   intervals   - bumping a log/checkpoint cadence mid-fleet must not block a resume
#   checkpoint  - restore paths and the guard's own mode
CONFIG_HASH_EXCLUDED_SECTIONS = ('eval', 'intervals', 'checkpoint')
#   experiment naming/logging - renaming or re-projecting a run must not block its resume
#   robometer endpoints       - the server URL moves between jobs
#   training.online_steps     - a run may legitimately be extended
#   provenance.dataset_mtime_ns - changes on a byte-identical copy of the HDF5
#       (a plain `cp` to a new node's scratch, a re-sync); the rest of the
#       provenance block (path, size, transition count) still identifies the
#       file, so the guard keeps its teeth without refusing honest resumes.
#       Everything else in `provenance` IS hashed on purpose: a different
#       controller JSON, a different traj-count dataset or a different
#       safe-robosuite commit is a different experiment, not a resume.
CONFIG_HASH_EXCLUDED_KEYS = (
    'experiment.project',
    'experiment.run_group',
    'experiment.run_name',
    'experiment.wandb_name',
    'experiment.save_dir',
    'experiment.log',
    'robometer.server_url',
    'robometer.url_file',
    'robometer.wait_s',
    'training.online_steps',
    'provenance.dataset_mtime_ns',
    # Maintainer decision 2026-08-25 (verifier FINDING 2): recorded in
    # provenance/flags.json but excluded from the resume digest — the flag
    # covers the WHOLE sibling checkout including untracked files, so any
    # incidental edit there would refuse every in-flight resume. An actual
    # sibling COMMIT (safe_robosuite_commit) still refuses resume.
    'provenance.safe_robosuite_dirty',
    # Recorded per attempt, not guarded (2026-10-11): which SafeVaDaR code and
    # controller JSON a (re)queued attempt ran. Hashing them would refuse every
    # resume across an unrelated commit.
    'provenance.safevadar_commit',
    'provenance.safevadar_dirty',
    'provenance.robosuite_controller_config_sha256',
)

# Keys added to a HASHED section after runs were already in flight. A key is
# elided from the digest while it holds the value below, so a checkpoint
# written before the key existed still resolves to the same hash-keyed
# directory; any other value hashes normally, because it IS a different
# experiment. Without this, adding agent.warmup_behavior_policy would move
# every in-flight run's checkpoint dir on its next requeue.
CONFIG_HASH_ELIDED_DEFAULTS = {
    'agent.warmup_behavior_policy': 'bon',
    # Checkpoint format only; added 2026-10-10 while runs were in flight.
    'bc.save_actor_only': False,
}


def compute_config_hash(config: Dict) -> str:
    """Digest of the training-relevant configuration, for the resume guard.

    The rolling-checkpoint key encodes only 5 identity axes (seed, env and the
    4 axes resolve_checkpoint_algo_name rewrites), so retuning e.g.
    agent.safety_coeff and relaunching at the same seed+env would otherwise
    silently resume the previous arm's networks, optimizer state and replay
    buffer under the new value. This digest is written beside the checkpoint
    and compared on load; the excluded keys above are the ones that legitimately
    differ between a run and its own resume.

    Values are serialized with `default=str`: every config value is a YAML
    scalar/list/dict today, and the fallback only has to be STABLE (an unstable
    repr would make resume nondeterministic), not round-trippable.
    """
    filtered: Dict[str, Any] = {}
    for section, values in config.items():
        if section in CONFIG_HASH_EXCLUDED_SECTIONS:
            continue
        if isinstance(values, dict):
            filtered[section] = {
                key: value for key, value in values.items()
                if f'{section}.{key}' not in CONFIG_HASH_EXCLUDED_KEYS
                and CONFIG_HASH_ELIDED_DEFAULTS.get(f'{section}.{key}', object()) != value
            }
        else:
            filtered[section] = values
    payload = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


# experiment.repeat_id, installed once by main() before the first checkpoint
# path is built. Process-wide rather than a parameter because all seven
# save_rolling_checkpoint call sites live in the four runners, which receive
# only (seed, env_name, algo_name) — a per-call parameter would have to be
# threaded through every runner signature to keep the save and load paths
# agreeing, for a value that is constant for the whole process.
_REPEAT_ID: Optional[str] = None


def set_repeat_id(repeat_id: Any) -> None:
    """Install experiment.repeat_id as this process's checkpoint-path suffix.

    Call before any checkpoint path is built. None (the YAML default) restores
    the un-suffixed layout every existing run uses.
    """
    global _REPEAT_ID
    _REPEAT_ID = None if repeat_id is None else str(repeat_id)


# compute_config_hash(config), installed once by main() before the first
# checkpoint path is built. Process-wide for the same reason as _REPEAT_ID.
#
# It keys the checkpoint DIRECTORY, not just the resume guard. The directory
# used to be keyed on seed + env + the resolver's identity axes only, and the
# fleet kept finding axes the resolver did not know about (safety_coeff, q_agg,
# a new family member): two such runs at one seed shared a directory on /data,
# and each save's cleanup deleted the other's checkpoint. That is how the
# no_proprio l40 seed-7 680k checkpoint was lost on 2026-09-10 — a min_max_relu
# seed-7 run started fresh in the same dir and its first save removed it.
# Keyed on the hash, a run can only ever see and delete its own files.
_CONFIG_HASH: Optional[str] = None

# Characters of the hash used in the directory name. 12 hex = 48 bits: a
# collision among a few hundred runs is ~1e-10, and the full digest is still
# checked on load.
_CONFIG_HASH_DIR_CHARS = 12


def set_checkpoint_config_hash(config_hash: Optional[str]) -> None:
    """Install this run's config hash as its checkpoint-directory suffix.

    Call before any checkpoint path is built. None restores the un-hashed
    layout (what tests and pre-2026-09-11 runs use) and disables the legacy
    fallback search in _checkpoint_candidate_dirs.
    """
    global _CONFIG_HASH
    _CONFIG_HASH = None if config_hash is None else str(config_hash)


def _checkpoint_leaf(seed: int, env_name: str, repeat_id: Optional[str],
                     config_hash: Optional[str]) -> str:
    leaf = f'{seed}_{env_name}' if repeat_id is None else f'{seed}_{env_name}_r{repeat_id}'
    if config_hash is not None:
        leaf = f'{leaf}_c{config_hash[:_CONFIG_HASH_DIR_CHARS]}'
    return leaf


def _get_scratch_subdir(seed: int, env_name: str, algo_name: str = 'ogpo',
                        repeat_id: Optional[str] = None) -> str:
    """Return the relative path under a node's scratch for this experiment.

    `repeat_id` defaults to the process-wide one from set_repeat_id(); pass it
    explicitly to build a path for a different repeat (tests do). It suffixes
    the `<seed>_<env_name>` leaf, NOT the algo component, so the on-disk tree
    still groups every repeat of an arm under one algo dir — and so a deliberate
    same-config repeat neither resumes the original's checkpoint nor is
    resumed by it. Without it a repeat is indistinguishable from a resume.

    The process-wide config hash (set_checkpoint_config_hash) suffixes the leaf
    last, `_c<hash[:12]>`, so runs that differ in ANY hashed config key get
    separate directories.
    """
    user = os.environ.get('USER', 'unknown_user')
    if repeat_id is None:
        repeat_id = _REPEAT_ID
    leaf = _checkpoint_leaf(seed, env_name, repeat_id, _CONFIG_HASH)
    return os.path.join(user, 'ogpo_checkpoints', algo_name, leaf)


def _legacy_checkpoint_dirs(seed: int, env_name: str) -> List[str]:
    """Pre-hash (un-suffixed) directories that may hold THIS run's checkpoint.

    Runs launched before the directory was keyed on the config hash wrote to
    `<algo>/<seed>_<env>[_r<id>]`, and the algo component itself has been
    renamed since (e.g. the aggregation-strategy suffixes), so every algo dir is
    searched rather than reconstructing old names. The loader only accepts one
    of these whose full config hash matches, so another arm's checkpoint in the
    same shared dir is skipped, never loaded. Empty when no hash is installed.
    """
    if _CONFIG_HASH is None:
        return []
    user = os.environ.get('USER', 'unknown_user')
    leaf = _checkpoint_leaf(seed, env_name, _REPEAT_ID, None)
    pattern = os.path.join(user, 'ogpo_checkpoints', '*', leaf)
    roots = []
    if SHARED_ROOT and os.path.isdir(SHARED_ROOT):
        roots.append(SHARED_ROOT)
    roots.append(SCRATCH_ROOT)
    roots += sorted(glob.glob(os.path.join(COMPUTE_ROOT, '*')))
    dirs = []
    for root in roots:
        dirs += sorted(glob.glob(os.path.join(root, pattern)))
    return dirs


def _is_legacy_checkpoint_dir(checkpoint_dir: str) -> bool:
    """True for a pre-hash directory found by _legacy_checkpoint_dirs."""
    return (_CONFIG_HASH is not None
            and not os.path.basename(os.path.normpath(checkpoint_dir)).endswith(
                f'_c{_CONFIG_HASH[:_CONFIG_HASH_DIR_CHARS]}'))


# (legacy_dir, step) this process resumed from, if it was a pre-hash directory.
# The first successful save into the hashed directory deletes that step's files
# so a multi-GB buffer is not left behind on /data forever.
_LEGACY_RESUME_SOURCE: Optional[Tuple[str, int]] = None


def _retire_legacy_resume_source(config_hash: str) -> None:
    """Delete the pre-hash checkpoint this run resumed from, now superseded.

    Only the files of the step that was loaded, and only when that step's
    config hash is this run's: another arm's files in the same legacy dir
    (it may still be running on old code) are never touched.
    """
    global _LEGACY_RESUME_SOURCE
    if _LEGACY_RESUME_SOURCE is None:
        return
    legacy_dir, step = _LEGACY_RESUME_SOURCE
    _LEGACY_RESUME_SOURCE = None
    hash_path = os.path.join(legacy_dir, f'{step}_config_hash.json')
    try:
        with open(hash_path, 'r') as f:
            if json.load(f)['config_hash'] != config_hash:
                return
        for fname in os.listdir(legacy_dir):
            if fname.startswith(f'{step}_'):
                os.unlink(os.path.join(legacy_dir, fname))
        print(f"Removed superseded pre-hash checkpoint {legacy_dir} (step {step})")
        if not os.listdir(legacy_dir):
            os.rmdir(legacy_dir)
    except Exception as e:
        print(f"Could not remove superseded pre-hash checkpoint {legacy_dir}: {e}")


def resolve_checkpoint_algo_name(algo_name: str, agent_config: Any) -> str:
    """Checkpoint identity for the rolling-checkpoint key.

    The CMDP arms (safe_qipm and its SAC-line fork sac_safe_qipm) each share
    ``agent_name`` (the registry key) between their single- and multi-head
    arms, so keying checkpoints on it alone would collide across arms at
    equal seed+env — resuming a subtask_status buffer of the wrong width. The
    multi arms therefore checkpoint under their own names. Obs width is a
    second identity axis: 53- and 67-dim buffers/networks must never
    cross-load at the same seed+env (resume does no shape validation — a
    cross-width load dies cryptically at the first update, and
    validate_resumed_buffer_width checks subtask columns, not obs width).
    qipm with critic_joint_proprio on rewrites to qipm_proprio, and sac_qipm
    to sac_qipm_proprio (still "sac"-prefixed, so qipm_runner's
    startswith('sac') single-eval rule holds); a safe-family
    arm with it OFF (the 2026-08-07 heads-without-proprio ablation) gains a
    _noproprio suffix — composed on top of the multihead rewrite, so the
    multi arms' on/off variants stay distinct too. Every sac_safe_qipm* key
    still starts with "sac", so qipm_runner's startswith('sac') single-eval
    rule holds for every variant. FQL mode is a third identity axis
    (2026-08-11): qipm/safe_qipm runs with use_one_step_policy carry an
    extra one_step TrainState in the agent pytree, and cross-loading against
    a non-FQL checkpoint either aborts the resume (None -> TrainState dies in
    from_state_dict, restarting the run at step 0) or silently leaves a raw
    dict where a TrainState belongs (TrainState -> None) — so FQL runs get a
    trailing _fql suffix, composed last. Scoped to qipm/safe_qipm only:
    ogpo's legacy FQL rolling checkpoints keep their keys, and hjb/sac still
    guard the mode off (or lack the field entirely). The safety-scalarization
    strategy is a fourth axis (2026-08-13): min_max and simple_summation arms
    have IDENTICAL parameter shapes, so a same-seed+env pair would silently
    share a rolling-checkpoint dir and cross-resume each other's networks,
    optimizer state and replay buffer mid-run — a _minmax suffix keeps them
    apart. Scoped to the safe_qipm CMDP line, the only algo whose agent reads
    the key: safe_qipm_hjb inherits it from safe_qipm.yaml but ignores it
    (margins go positive, so the "Q <= 0" premise fails there) and no other
    algo declares it at all, so the bracket read stays behind the algo_name
    check. min_max composes with FQL mode since 2026-08-14
    (docs/changes/2026-08-14-fql-minmax-aggregation/ lifted the create()
    rejection), giving safe_qipm_minmax_fql — _fql still composes last. Every
    other algo has a unique agent_name and passes through unchanged.
    """
    name = algo_name
    if algo_name in ("safe_qipm", "sac_safe_qipm") and agent_config["safety_head_mode"] == "multi":
        name = f"{algo_name}_multihead"
    # Literal copy of ogpo.main.SAFE_QIPM_ALGOS — cannot import it here
    # (circular: main imports this module). A family member added there must
    # be added here too (three entries as of 2026-08-10);
    # test_checkpoint_identity.py parametrizes over the main.py tuple so the
    # mismatch fails loudly instead of silently sharing checkpoint keys
    # across obs widths.
    if algo_name in ("safe_qipm", "safe_qipm_hjb", "sac_safe_qipm") and not agent_config["critic_joint_proprio"]:
        name = f"{name}_noproprio"
    if algo_name == "qipm" and agent_config["critic_joint_proprio"]:
        name = "qipm_proprio"
    if algo_name == "sac_qipm" and agent_config["critic_joint_proprio"]:
        name = "sac_qipm_proprio"
    if algo_name == "safe_qipm":
        # min_max keeps its historical `_minmax` token (existing checkpoints
        # and tests depend on it). Every OTHER non-default strategy used to
        # share the summation arm's directory: same parameter shapes, so a
        # same-seed max_relu / min_relu pair would silently cross-resume.
        # That was merely unlucky on node-local /scratch; on shared storage it
        # is guaranteed, and it would also adopt the other arm's wandb id.
        agg = agent_config["safety_aggregation_strategy"]
        if agg == "min_max":
            name = f"{name}_minmax"
        elif agg not in ("simple_summation", None, ""):
            name = f"{name}_{str(agg).replace('_', '')}"
    if algo_name in ("qipm", "safe_qipm") and agent_config["use_one_step_policy"]:
        name = f"{name}_fql"
    return name


def validate_resumed_buffer_width(resumed_buffer: Any, expected_width: int) -> None:
    """Fail fast if a resumed replay buffer's subtask_status width is wrong.

    load_rolling_checkpoint unpickles the buffer without shape validation, so a
    checkpoint written by a run with a different head count would otherwise
    load silently and corrupt training. Raises ValueError on mismatch; the
    resume try/except in main.py turns that into a fresh start, same as any
    other unloadable checkpoint. None buffers (BC-phase checkpoints) pass.
    """
    if resumed_buffer is None:
        return
    got_width = resumed_buffer['subtask_status'].shape[-1]
    if got_width != expected_width:
        raise ValueError(
            f"resumed replay buffer subtask_status width {got_width} != "
            f"configured no_of_subtasks + no_of_safety_subtasks = {expected_width}; "
            "refusing to resume a checkpoint from a different head configuration"
        )


def shared_checkpoint_dir(seed: int, env_name: str,
                          algo_name: str = 'ogpo') -> Optional[str]:
    """This run's directory under the shared root, or None if there isn't one.

    Returns None when SHARED_ROOT is absent (login node, CI, any cluster
    without it), which is what makes the shared path opt-out-by-environment
    rather than a hard dependency.
    """
    if not SHARED_ROOT or not os.path.isdir(SHARED_ROOT):
        return None
    return os.path.join(SHARED_ROOT, _get_scratch_subdir(seed, env_name, algo_name))


def get_checkpoint_dir(seed: int, env_name: str, algo_name: str = 'ogpo') -> str:
    """Primary write target for this run's rolling checkpoint.

    The shared root when it exists, else the current node's /scratch. Writing
    to shared storage is the whole point on a requeue partition — a checkpoint
    only the original node can read is not a checkpoint. /scratch remains the
    fallback so a node without the shared mount still checkpoints locally
    rather than not at all.
    """
    shared = shared_checkpoint_dir(seed, env_name, algo_name)
    if shared is not None:
        return shared
    return os.path.join(SCRATCH_ROOT, _get_scratch_subdir(seed, env_name, algo_name))


def cleanup_checkpoint(seed: int, env_name: str, algo_name: str = 'ogpo'):
    """Remove rolling checkpoint after successful completion to free disk space.

    Must stay a SUPERSET of _checkpoint_candidate_dirs, which now includes the
    shared root. Missing it would leak a multi-GB replay buffer per finished run
    onto NFS — and worse, leave a stale checkpoint that a later same-seed run
    could be offered before its own.
    """
    # Shared root first: it is the copy that outlives the node.
    for shared_dir in filter(None, [shared_checkpoint_dir(seed, env_name, algo_name)]):
        if os.path.exists(shared_dir):
            shutil.rmtree(shared_dir, ignore_errors=True)
            print(f"Cleaned up shared checkpoint dir: {shared_dir}")

    # Clean up local /scratch. Built from SCRATCH_ROOT directly rather than via
    # get_checkpoint_dir, which now returns the shared path when one exists.
    local_dir = os.path.join(SCRATCH_ROOT,
                             _get_scratch_subdir(seed, env_name, algo_name))
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir, ignore_errors=True)
        print(f"Cleaned up checkpoint dir: {local_dir}")

    # Also clean up any cross-node checkpoint (if job resumed on a different node).
    # Deletes exactly the set _checkpoint_candidate_dirs enumerates (local +
    # every /compute/* candidate), so neither the newest-by-step ordering nor
    # the loader's fall-through to an older candidate can leave a copy behind.
    subdir = _get_scratch_subdir(seed, env_name, algo_name)
    for candidate in glob.glob(os.path.join(COMPUTE_ROOT, '*', subdir)):
        shutil.rmtree(candidate, ignore_errors=True)
        print(f"Cleaned up remote checkpoint dir: {candidate}")

def _job_marker_path(seed: int, env_name: str, algo_name: str) -> Optional[str]:
    """Where this Slurm job records the checkpoint a run last published.

    Keyed on SLURM_JOB_ID, which a requeue keeps, plus the run's identity
    WITHOUT the config hash (algo, seed, env, repeat_id), so the key survives
    exactly the hash drift the marker exists to catch, and two trainings run
    back to back inside one job do not see each other's marker. Stored on the
    shared root so the next attempt reads it from whichever node it lands on.
    None outside Slurm or without the shared mount.
    """
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id or not SHARED_ROOT or not os.path.isdir(SHARED_ROOT):
        return None
    user = os.environ.get('USER', 'unknown_user')
    leaf = _checkpoint_leaf(seed, env_name, _REPEAT_ID, None)
    return os.path.join(SHARED_ROOT, user, 'ogpo_checkpoints', '_jobs', job_id,
                        algo_name, f'{leaf}.json')


def _write_job_marker(seed: int, env_name: str, algo_name: str, checkpoint_dir: str,
                      step: int, wandb_run_id: Optional[str], config_hash: str) -> None:
    path = _job_marker_path(seed, env_name, algo_name)
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump({'checkpoint_dir': checkpoint_dir, 'step': step,
                   'wandb_run_id': wandb_run_id, 'config_hash': config_hash}, f)
    os.replace(tmp, path)


def assert_requeue_resumed(is_resumed: bool, seed: int, env_name: str,
                           algo_name: str) -> None:
    """Refuse to start fresh on a requeue of a job that already checkpointed.

    The checkpoint directory is keyed on the config hash, so anything that
    shifts the hash between attempts (a safe-robosuite commit, the provenance
    git probe timing out on NFS, an edited YAML) makes the requeued attempt see
    an empty directory. It then used to start from step 0 under a NEW wandb run
    and, on its first save, compete with nothing it could see. A config change
    is a legitimate fresh start for a new job, never for a requeue, so this
    compares against what the job itself last published.
    """
    if is_resumed:
        # A resume is a resume wherever the checkpoint came from: /scratch and
        # /compute candidates are legitimate sources when the shared mount is
        # missing on this node.
        return
    restarts = int(os.environ.get('SLURM_RESTART_COUNT', '0') or 0)
    if restarts > 0 and os.environ.get('SLURM_JOB_ID') and not (
            SHARED_ROOT and os.path.isdir(SHARED_ROOT)):
        raise RuntimeError(
            f"requeue #{restarts} of job {os.environ['SLURM_JOB_ID']} resumed "
            f"nothing and cannot see the shared checkpoint root {SHARED_ROOT!r} "
            f"on {os.environ.get('SLURMD_NODENAME', '?')}; refusing to start fresh")
    path = _job_marker_path(seed, env_name, algo_name)
    if path is None or not os.path.exists(path):
        return
    with open(path) as f:
        marker = json.load(f)
    raise RuntimeError(
        f"job {os.environ.get('SLURM_JOB_ID')} already published a rolling "
        f"checkpoint (step {marker.get('step')} in {marker.get('checkpoint_dir')}, "
        f"wandb run {marker.get('wandb_run_id')}, config hash "
        f"{str(marker.get('config_hash'))[:12]}), but this attempt resolved "
        f"config hash {str(_CONFIG_HASH)[:12]} and found nothing to resume. "
        f"Refusing to start fresh under a new wandb run. Restore the config "
        f"(or the safe-robosuite commit) the job started with, or delete "
        f"{path} to deliberately restart from step 0.")


def save_rolling_checkpoint(
        agent: Any,
        replay_buffer: Any,
        step: int,
        online_loop_step: int,
        seed: int,
        env_name: str,
        wandb_run_id: Optional[str] = None,
        algo_name: str = 'ogpo',
        *,
        rng_states: Dict,
        episode_counter: int,
        config_hash: str,
    ):
    """Save rolling checkpoint atomically (keeps only latest).

    Writes to a temp directory first, then renames into place so that a
    kill/preemption mid-write never leaves a corrupt checkpoint.

    The three keyword-only arguments are REQUIRED (no defaults) because
    CHECKPOINT_SCHEMA_VERSION 2 promises the loader that every one of them is
    on disk — a call site that could omit one would produce a checkpoint the
    no-defaulting loader has to reject.

    Args:
        rng_states: dict from _collect_rng_states — 'python_random',
            'numpy_random', 'online_rng' (None for a BC-phase checkpoint),
            'eval_rng', 'dataset_rng'.
        episode_counter: completed training episodes, so the resumed run
            continues the seed + 10000 + episode reset sequence instead of
            re-walking it. 0 for a BC-phase checkpoint (no env episodes yet).
        config_hash: compute_config_hash(config) of the run writing this
            checkpoint; the resume guard compares against it.
    """
    _save_started = time.time()
    checkpoint_dir = get_checkpoint_dir(seed, env_name, algo_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Write to a temp directory first (same filesystem for atomic rename)
    tmp_dir = tempfile.mkdtemp(dir=checkpoint_dir, prefix=f'.tmp_{step}_')
    try:
        agent_path = os.path.join(tmp_dir, f'{step}_agent.pkl')
        buffer_path = os.path.join(tmp_dir, f'{step}_buffer.pkl')
        meta_path = os.path.join(tmp_dir, f'{step}_meta.json')

        with open(agent_path, 'wb') as f:
            pickle.dump(flax.serialization.to_state_dict(agent), f)

        with open(buffer_path, 'wb') as f:
            pickle.dump(replay_buffer, f)

        meta_data = {
            'schema_version': CHECKPOINT_SCHEMA_VERSION,
            'online_loop_step': online_loop_step,
            'episode_counter': episode_counter,
            'node_name': os.environ.get('SLURMD_NODENAME', ''),
        }
        if wandb_run_id is not None:
            meta_data['wandb_run_id'] = wandb_run_id
        with open(meta_path, 'w') as f:
            json.dump(meta_data, f)

        # Save RNG states for exact reproducibility after preemption
        rng_path = os.path.join(tmp_dir, f'{step}_rng.pkl')
        with open(rng_path, 'wb') as f:
            pickle.dump(rng_states, f)

        # Config digest for the resume guard. Written as its own file rather
        # than folded into the checkpoint key so a mismatch is an actionable
        # error naming both hashes, not an unfindable checkpoint.
        hash_path = os.path.join(tmp_dir, f'{step}_config_hash.json')
        with open(hash_path, 'w') as f:
            json.dump({'config_hash': config_hash, 'algo_name': algo_name}, f)

        # Publish by rename. Each os.replace is atomic on its own, but the SET
        # of five is not, so the order matters: `{step}_agent.pkl` goes LAST.
        # _checkpoint_steps enumerates a directory by its *_agent.pkl files, so
        # until that final rename lands step N does not exist as far as any
        # reader is concerned, and N-1 stays the newest candidate. A kill
        # anywhere in the loop therefore leaves a complete checkpoint rather
        # than a headless one — atomic where it counts, without restructuring
        # the layout into per-step subdirectories.
        agent_fname = f'{step}_agent.pkl'
        publish_order = sorted(f for f in os.listdir(tmp_dir) if f != agent_fname)
        if os.path.exists(os.path.join(tmp_dir, agent_fname)):
            publish_order.append(agent_fname)
        for fname in publish_order:
            src = os.path.join(tmp_dir, fname)
            dst = os.path.join(checkpoint_dir, fname)
            os.replace(src, dst)  # atomic on same filesystem

        # Size and duration are logged because the checkpoint now lands on NFS
        # rather than node-local NVMe, and the replay buffer is preallocated at
        # its full configured capacity (2M rows of zeros included), so the cost
        # is the same on step 1 as at convergence. This line is what the
        # preemption_backup_interval should be tuned against.
        nbytes = sum(os.path.getsize(os.path.join(checkpoint_dir, f))
                     for f in os.listdir(checkpoint_dir)
                     if f.startswith(f'{step}_'))
        print(f"Saved rolling checkpoint to {checkpoint_dir} at step {step} "
              f"({nbytes / 1e9:.2f} GB in {time.time() - _save_started:.1f}s)")
    finally:
        # Clean up temp dir (should be empty after moves, but be safe)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Cleanup old checkpoints
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('.tmp_'):
            # Remove any leftover temp dirs from crashed writes
            file_path = os.path.join(checkpoint_dir, filename)
            shutil.rmtree(file_path, ignore_errors=True)
            continue
        # FIXED (2026-08-24): exact-step match. The old `str(step) in filename`
        # substring test kept 1000_agent.pkl when saving step 100, and kept
        # every 1500000_* file when saving step 5 — stale checkpoints that
        # _find_checkpoint_dir's max(steps) would then resume instead.
        if filename.startswith(f'{step}_'):
            continue
        file_path = os.path.join(checkpoint_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    # Step `step` is now fully published in this run's own dir, so the pre-hash
    # copy it was resumed from (if any) is no longer the newest of anything.
    _retire_legacy_resume_source(config_hash)
    # Best-effort: the checkpoint is already published, and the callers exit 0
    # immediately after this on a preemption. A transient shared-FS error here
    # must not turn a successful emergency save into a failed job.
    try:
        _write_job_marker(seed, env_name, algo_name, checkpoint_dir, step,
                          wandb_run_id, config_hash)
    except Exception as exc:
        print(f"[checkpoint] could not write the job marker: {exc}")


def _checkpoint_steps(checkpoint_dir: str) -> List[int]:
    """Every step with an agent pickle in `checkpoint_dir`, newest first.

    Enumerating ALL of them (rather than just the max) is what lets the loader
    fall back WITHIN a directory. save_rolling_checkpoint publishes its five
    files with one os.replace each: every rename is atomic on its own, but the
    set is not, so a kill landing between `{N}_agent.pkl` and `{N}_rng.pkl`
    leaves a complete step N-1 beside a half-published step N. Keying the
    directory on its max step alone made that half-published N the only
    candidate — it fails the loader's completeness guard, and the intact N-1
    sitting right next to it was never even looked at.
    """
    steps = []
    for path in glob.glob(os.path.join(checkpoint_dir, "*_agent.pkl")):
        try:
            steps.append(int(os.path.basename(path).split('_')[0]))
        except ValueError:
            # Not a <step>_agent.pkl file; ignore it exactly as the loader does.
            continue
    return sorted(steps, reverse=True)


def _max_checkpoint_step(checkpoint_dir: str) -> Optional[int]:
    """Highest step with an agent pickle in `checkpoint_dir`, or None if empty."""
    steps = _checkpoint_steps(checkpoint_dir)
    return steps[0] if steps else None


def _checkpoint_candidate_dirs(seed: int, env_name: str,
                               algo_name: str = 'ogpo') -> List[Tuple[str, int]]:
    """Every (directory, step) that could hold this run's checkpoint, newest first.

    On resume after preemption the job usually lands on a different node, so
    the shared root is searched FIRST — it is the only one the new node can
    read. Node-local /scratch and any /compute/<node_name>/ mounts follow, both
    for runs predating the shared root and for nodes that lack it.

    Returns [(directory, step), ...] ordered by DESCENDING step. One entry PER
    STEP, not per directory: a directory can transiently hold two steps when a
    kill interrupts the multi-file publish, and the older one has to remain
    reachable (see _checkpoint_steps). Ties keep the shared copy first, then
    local, then the /compute mounts in sorted order, so ordering is
    deterministic across nodes.

    Ordering, not selection: load_rolling_checkpoint walks this list and takes
    the first candidate that passes every integrity/identity guard. Directories
    with no <step>_agent.pkl at all contribute nothing.
    """
    subdir = _get_scratch_subdir(seed, env_name, algo_name)
    # Shared first: it is the only root a requeued job on a different node can
    # actually read, so on a tie it should win over a node-local copy that may
    # belong to some earlier attempt on this same machine.
    shared_dir = shared_checkpoint_dir(seed, env_name, algo_name)
    local_dir = os.path.join(SCRATCH_ROOT, subdir)
    search_dirs = ([] if shared_dir is None else [shared_dir]) + [local_dir]
    search_dirs += sorted(glob.glob(os.path.join(COMPUTE_ROOT, '*', subdir)))
    # Pre-hash directories last, so on a step tie the run's own copy wins. A
    # legacy dir with MORE progress still sorts first, which is the point: a run
    # requeued onto the hashed layout continues from its old checkpoint.
    search_dirs += [d for d in _legacy_checkpoint_dirs(seed, env_name)
                    if d not in search_dirs]

    found = []
    for rank, candidate in enumerate(search_dirs):
        # Every step, not just the newest — see _checkpoint_steps for why.
        for step in _checkpoint_steps(candidate):
            # rank breaks step ties by search order, which puts shared first.
            found.append((-step, rank, candidate))
    found.sort()
    return [(candidate, -neg_step) for neg_step, _rank, candidate in found]


def _find_checkpoint_dir(seed: int, env_name: str, algo_name: str = 'ogpo') -> Optional[str]:
    """Highest-step checkpoint directory for this run, or None if there is none.

    Thin newest-first selector over _checkpoint_candidate_dirs. It reports only
    which directory *looks* newest — it runs none of the loader's guards, so a
    caller that intends to resume must go through load_rolling_checkpoint,
    which falls through to older candidates when the newest is unusable.
    """
    candidates = _checkpoint_candidate_dirs(seed, env_name, algo_name)
    if not candidates:
        return None
    return candidates[0][0]


def _inspect_checkpoint_candidate(
        checkpoint_dir: str,
        step: int,
        *,
        config_hash: str,
        config_hash_mode: str,
    ) -> Tuple[Optional[Dict], Optional[str]]:
    """Run every integrity/identity guard on one candidate, before any unpickle.

    Returns (meta, None) when the candidate passes all of them, or
    (None, reason) naming the first guard it failed.

    Returning a reason instead of raising is what lets load_rolling_checkpoint
    fall through to the next candidate (2026-08-25 fix): a stale pre-v2 or
    different-arm checkpoint left on some /compute mount at a higher step than
    the run's own must not cost the run its own resumable copy — with a raise
    here that combination restarts the run from step 0 on every preemption,
    forever. A candidate is accepted or skipped WHOLE; nothing inside an
    accepted one is ever defaulted.
    """
    buffer_path = os.path.join(checkpoint_dir, f'{step}_buffer.pkl')
    meta_path = os.path.join(checkpoint_dir, f'{step}_meta.json')
    rng_path = os.path.join(checkpoint_dir, f'{step}_rng.pkl')
    hash_path = os.path.join(checkpoint_dir, f'{step}_config_hash.json')

    # An agent pickle with no buffer beside it is a torn write (the rename loop
    # in save_rolling_checkpoint moves one file at a time), not a resumable
    # checkpoint — a BC-phase checkpoint writes a pickled None, so the file is
    # present there too.
    if not os.path.exists(buffer_path):
        return None, 'no buffer pickle (incomplete checkpoint)'

    if not os.path.exists(meta_path):
        return None, (f'no meta.json; cannot verify schema_version '
                      f'{CHECKPOINT_SCHEMA_VERSION}')
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    # .get: a pre-v2 checkpoint has no such key at all — its absence IS the
    # condition under test, so it must not raise before the message is built.
    schema_version = meta.get('schema_version')
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        return None, (
            f'schema_version {schema_version!r}, expected '
            f'{CHECKPOINT_SCHEMA_VERSION} (episode_counter/dataset_rng/config '
            'hash are not recoverable from an older checkpoint and must not be '
            'defaulted)'
        )

    if config_hash_mode == 'strict':
        if not os.path.exists(hash_path):
            return None, ("no config hash file; refusing under "
                          "checkpoint.config_hash_mode='strict'")
        with open(hash_path, 'r') as f:
            saved_hash = json.load(f)['config_hash']
        if saved_hash != config_hash:
            return None, (
                f'config hash mismatch: checkpoint {saved_hash}, this run '
                f'{config_hash}. It was written by a different configuration at '
                'the same seed+env (see CONFIG_HASH_EXCLUDED_* for what is '
                "ignored); refusing under checkpoint.config_hash_mode='strict'"
            )

    if not os.path.exists(rng_path):
        return None, (f'declares schema_version {CHECKPOINT_SCHEMA_VERSION} but '
                      'has no rng pickle')

    return meta, None


def load_rolling_checkpoint(
        seed: int,
        env_name: str,
        agent_structure: Any,
        algo_name: str = 'ogpo',
        *,
        config_hash: str,
        config_hash_mode: str,
    ) -> Tuple[Any, Any, int, int, Optional[str], Optional[Dict], int]:
    """Load rolling checkpoint.

    Searches local /scratch and cross-node /compute/<node>/ mounts and walks the
    candidates newest step first (see _checkpoint_candidate_dirs), taking the
    first one that passes every guard in _inspect_checkpoint_candidate: a
    present buffer pickle, a schema_version equal to CHECKPOINT_SCHEMA_VERSION,
    a present rng pickle and — when config_hash_mode is 'strict' — a matching
    config digest. A candidate that fails any of them is logged with the
    directory and the reason and SKIPPED, never partially honored.

    Only when every candidate has been refused does this raise, with all the
    reasons; main.py runs it inside its resume try/except, which turns that into
    a logged fresh start (now preceded by the per-candidate refusal lines).
    Finding nothing at all is not an error and returns the all-None tuple.

    Args:
        config_hash: compute_config_hash(config) of the run trying to resume.
        config_hash_mode: 'strict' to enforce the digest, 'off' to skip it
            (checkpoint.config_hash_mode). 'off' waives only the digest — the
            schema and completeness guards still apply.

    Returns:
        (agent, replay_buffer, step, online_loop_step, wandb_run_id, rng_states,
        episode_counter); the first six as before, all None when no checkpoint
        was found, with episode_counter 0.
    """
    # A malformed mode is this run's own configuration error, not a property of
    # any candidate, so it fails fast before the search rather than per-candidate.
    if config_hash_mode not in ('strict', 'off'):
        raise ValueError(
            f"checkpoint.config_hash_mode must be 'strict' or 'off', got "
            f"{config_hash_mode!r}"
        )

    candidates = _checkpoint_candidate_dirs(seed, env_name, algo_name)
    if not candidates:
        return None, None, None, None, None, None, 0

    global _LEGACY_RESUME_SOURCE
    refusals = []
    own_refused = False
    for checkpoint_dir, latest_step in candidates:
        legacy = _is_legacy_checkpoint_dir(checkpoint_dir)
        meta, reason = _inspect_checkpoint_candidate(
            checkpoint_dir, latest_step,
            config_hash=config_hash,
            # A pre-hash dir is shared by every arm at this seed, so the digest
            # is the only thing telling this run's checkpoint from another's.
            config_hash_mode='strict' if legacy else config_hash_mode,
        )
        if reason is None:
            break
        print(f"Refusing rolling checkpoint candidate {checkpoint_dir} "
              f"(step {latest_step}): {reason}")
        refusals.append(f"  - {checkpoint_dir} (step {latest_step}): {reason}")
        own_refused = own_refused or not legacy
    else:
        # Refused candidates that all sit in pre-hash dirs belong to other
        # arms (or are stale). A fresh start writes only to this run's own
        # hashed dir, so it cannot destroy them — start fresh instead of
        # crashing, which on a requeue partition would never be retried.
        if not own_refused:
            print(f"No usable rolling checkpoint for seed={seed} env={env_name} "
                  f"algo={algo_name}: only other runs' pre-hash checkpoints "
                  f"found ({len(refusals)}); starting fresh")
            return None, None, None, None, None, None, 0
        raise ValueError(
            f"refused every rolling checkpoint candidate for seed={seed} "
            f"env={env_name} algo={algo_name} ({len(refusals)} found):\n"
            + "\n".join(refusals)
        )

    if legacy:
        _LEGACY_RESUME_SOURCE = (checkpoint_dir, latest_step)
        print(f"Resuming from pre-hash checkpoint dir: {checkpoint_dir} (step {latest_step})")
    elif checkpoint_dir != get_checkpoint_dir(seed, env_name, algo_name):
        print(f"Found checkpoint outside this node's primary dir: {checkpoint_dir} "
              f"(step {latest_step})")

    agent_path = os.path.join(checkpoint_dir, f'{latest_step}_agent.pkl')
    buffer_path = os.path.join(checkpoint_dir, f'{latest_step}_buffer.pkl')
    rng_path = os.path.join(checkpoint_dir, f'{latest_step}_rng.pkl')

    print(f"Loading rolling checkpoint from step {latest_step}")

    with open(agent_path, 'rb') as f:
        agent_state = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent_structure, agent_state)

    with open(buffer_path, 'rb') as f:
        replay_buffer = pickle.load(f)

    online_loop_step = meta['online_loop_step']
    episode_counter = meta['episode_counter']
    # .get: only written when wandb logging is on, so genuinely optional.
    wandb_run_id = meta.get('wandb_run_id', None)

    with open(rng_path, 'rb') as f:
        rng_states = pickle.load(f)
    print(f"Loaded RNG states from checkpoint")

    return (agent, replay_buffer, latest_step, online_loop_step, wandb_run_id,
            rng_states, episode_counter)


def _collect_rng_states(online_rng, eval_rng, dataset_rng) -> Dict:
    """Collect all RNG states for checkpointing.

    online_rng is None for a BC-phase checkpoint (bc_runner never steps a train
    env, so it has no online key); the resume sites check the VALUE, not the
    key's presence. dataset_rng is the np.random.Generator shared with the
    offline Dataset — it pickles directly.

    WARNING: dataset_rng goes in BY REFERENCE (the other four entries are
    snapshots), so the result must be pickled immediately — call this inline in
    the save_rolling_checkpoint call, never store it for later. During BC the
    Generator is also being drawn from by the prefetcher thread, which runs 1-3
    batches ahead, so bc_runner passes a SNAPSHOT Generator at the position
    right after the last batch a train step consumed (_generator_at); passing
    the live one would make a BC resume skip the prefetched batches.
    """
    return {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'online_rng': online_rng,
        'eval_rng': eval_rng,
        'dataset_rng': dataset_rng,
    }


def _restore_rng_states(rng_states: Dict, dataset_rng: np.random.Generator):
    """Restore the python, numpy-global and dataset RNG states.

    dataset_rng is restored IN PLACE (bit_generator.state assignment) because
    the live Generator is already shared with the Dataset built at startup —
    rebinding the caller's name would leave the dataset drawing from the
    unrestored stream. Every lookup happens before the first mutation so a
    malformed payload cannot leave the global RNGs half-restored.

    Returns (online_rng, eval_rng); online_rng is None for a BC-phase payload.
    """
    python_state = rng_states['python_random']
    numpy_state = rng_states['numpy_random']
    dataset_state = rng_states['dataset_rng'].bit_generator.state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    dataset_rng.bit_generator.state = dataset_state
    return rng_states['online_rng'], rng_states['eval_rng']


# =============================================================================
# SIGTERM handler for graceful preemption
# =============================================================================

# Global flag checked by training loops to trigger an emergency checkpoint
_preemption_requested = False


class PreemptionDuringEval(Exception):
    """Raised to abandon an in-progress evaluation when SIGTERM arrives.

    An evaluation is the longest uninterruptible stretch in the training loop —
    64 episodes of simulation, minutes of wall clock — and it sits between the
    training loop's preemption checks. A SIGTERM landing inside one used to go
    unnoticed until the eval finished, which on a partition with KillWait=120
    means never: the process was SIGKILLed mid-eval with the emergency
    checkpoint unwritten. Across 1860 SIGTERMs only 10 emergency saves ever
    started, and both new preempt tests caught their signal at
    "Q-IPM Parallel Eval: 0/64".

    Raised rather than returned so a half-finished eval cannot be mistaken for
    a real measurement and logged.
    """


def is_preemption_requested() -> bool:
    """Check if a SIGTERM (preemption) signal has been received."""
    return _preemption_requested


def raise_if_preempted() -> None:
    """Abandon an evaluation if a preemption has been signalled.

    Called from inside the eval rollout loops so control returns to the
    training loop — and therefore to the emergency checkpoint — in seconds
    rather than minutes.
    """
    if _preemption_requested:
        raise PreemptionDuringEval(
            "SIGTERM received during evaluation; abandoning it to checkpoint"
        )


def _sigterm_handler(signum, frame):
    """Handle SIGTERM from SLURM preemption. Sets flag so the training loop
    can save a checkpoint at the next safe point."""
    global _preemption_requested
    _preemption_requested = True
    print("\n[PREEMPTION] SIGTERM received — will save checkpoint and exit at next opportunity.")


def install_preemption_handler():
    """Install SIGTERM handler for graceful SLURM preemption.
    Call this once at the start of main()."""
    signal.signal(signal.SIGTERM, _sigterm_handler)


def process_train_dataset(
    ds: Dict,
    env_name: str,
    dataset_proportion: float = 1.0,
    sparse: bool = False,
    numpy_rng: Optional[np.random.Generator] = None,
    compute_mc_returns: bool = False,
    discount: float = 0.99,
) -> Dataset:
    """ Process training dataset """
    from envs.env_utils import is_robomimic_env

    if numpy_rng is not None:
        ds = Dataset.create(numpy_rng=numpy_rng, **ds)
    else:
        ds = Dataset.create(**ds)

    # Handle dataset proportion
    if dataset_proportion < 1.0:
        old_size = len(ds['masks'])
        new_size = int(old_size * dataset_proportion)
        ds_dict = {k: v[:new_size] for k, v in ds.items()}
        if numpy_rng is not None:
            ds = Dataset.create(numpy_rng=numpy_rng, **ds_dict)
        else:
            ds = Dataset.create(**ds_dict)
        print(f"[Dataset] Reduced from {old_size} to {new_size} transitions (proportion={dataset_proportion})")

    # NOTE: Robomimic reward transformation is now done in robomimic_utils.get_dataset()
    # (transforms from [0,1] to [-1,0]). No additional transformation needed here.

    # Sparse reward conversion
    if sparse:
        sparse_rewards = (ds["rewards"] != 0.0) * -1.0
        ds_dict = {k: v for k, v in ds.items()}
        ds_dict["rewards"] = sparse_rewards
        if numpy_rng is not None:
            ds = Dataset.create(numpy_rng=numpy_rng, **ds_dict)
        else:
            ds = Dataset.create(**ds_dict)

    # Compute MC returns if needed
    if compute_mc_returns:
        mc_returns = ds.compute_mc_returns(discount)
        ds_dict = {k: v for k, v in ds.items()}
        ds_dict['mc_returns'] = mc_returns
        if numpy_rng is not None:
            ds = Dataset.create(numpy_rng=numpy_rng, **ds_dict)
        else:
            ds = Dataset.create(**ds_dict)

    return ds


# =============================================================================
# Q-Value Bounds Configuration
# =============================================================================

def get_q_bounds(env_name: str, discount: float) -> Tuple[float, float]:
    """Get Q-value min/max bounds for the environment.

    Args:
        env_name: Environment name.
        discount: Discount factor.

    Returns:
        Tuple of (q_min, q_max).
    """
    from envs.env_utils import is_robomimic_env
    from envs.env_utils import GYMNASIUM_ROBOTICS_ADROITHAND_ENVS

    if env_name.startswith('robocasa_'):
        # Dense rewards in [-no_subtasks, 0], same structure as robomimic
        q_min = -1.0 / (1 - discount)
        q_max = 0.0
    elif env_name.startswith('libero_'):
        q_min = -1.0 / (1 - discount)
        q_max = 0.0
    elif is_robomimic_env(env_name):
        q_min = -1.0 / (1 - discount)
        q_max = 0.0 / (1 - discount)
    elif env_name.startswith('PutBallintoBowl'):
        q_min = 0.0 / (1 - discount)
        q_max = 1.0 / (1 - discount)
    elif env_name.startswith('cube-triple'):
        q_min = -3.0 / (1 - discount)
        q_max = 0.0 / (1 - discount)
    elif env_name.startswith('D4RL/kitchen'):
        q_min = 0 / (1 - discount)
        q_max = 4 / (1 - discount)
    elif env_name in GYMNASIUM_ROBOTICS_ADROITHAND_ENVS:
        q_min = -50
        q_max = 10
    elif env_name.startswith('stacking') or env_name.startswith('gate_insertion'):
        q_min = -100
        q_max = 100
    elif env_name.startswith('pusht'):
        q_min = -1.0 / (1 - discount)
        q_max = 0.0 / (1 - discount)
    elif _is_ogbench_env(env_name):
        # OGBench singletask: sparse reward in {0, 1}; dense wrapper uses [-no_subtasks, 0]
        q_min = -1.0 / (1 - discount)
        q_max = 1.0 / (1 - discount)
    else:
        raise ValueError(f"Unknown environment: {env_name}, please specify min max reward")

    return q_min, q_max

def create_success_buffer_batch(
    replay_buffer: ReplayBuffer,
    batch_size: int,
    sequence_length: int,
    discount: float,
) -> Dict:
    """ Sample a batch from the success buffer """
    success_batch = replay_buffer.sample_sequence(
        batch_size=batch_size,
        sequence_length=sequence_length,
        discount=discount,
        success_only=True,
    )
    return success_batch

def setup_experiment_logging(
    seed: int,
    project: str,
    run_group: str,
    wandb_name: str,
    save_dir: str,
    env_name: str,
    checkpoint_algo_name: str,
    log_enabled: bool = True,
    prefixes: Optional[List[str]] = None,
    checkpoint_dir_fn: Optional[Callable] = None,
    config: Optional[dict] = None,
    repeat_id: Any = None,
    resume_wandb_run_id: Optional[str] = None,
) -> Tuple[str, LoggingHelper, Optional[str]]:
    """ Setup experiment logging infrastructure

    `checkpoint_algo_name` is resolve_checkpoint_algo_name's output, passed in
    from main() rather than re-derived here (it is already computed there,
    before this call). It joins exp_name so the ARTIFACT dir carries the same
    identity axes the rolling-checkpoint key does: two arms differing only in
    e.g. safety_aggregation_strategy or critic_joint_proprio, launched at the
    same seed without a hand-written --wandb_name, used to overwrite each
    other's flags.json and params_<step>.pkl (the 2026-08-14 fql_base seed-0
    repeat destroyed the 2026-08-11 run's params_500000.pkl).

    `repeat_id` (experiment.repeat_id) separates a DELIBERATE same-config
    repeat, which the identity axes alone cannot distinguish from a resume.

    `resume_wandb_run_id` is the wandb id recorded in a rolling checkpoint that
    main() actually LOADED, or None. Pass it and the run continues; pass None
    and a fresh run starts. Caller-supplied rather than read from disk here so
    the id can never outlive the checkpoint it belongs to.
    """
    import wandb

    if prefixes is None:
        prefixes = ["eval", "offline_agent", "online_agent", "env"]

    if wandb_name == "test":
        # Already unique per process (job id + timestamp), so the identity axes
        # would add nothing here.
        exp_name = get_exp_name(seed)
    else:
        exp_name = f"{wandb_name}_{checkpoint_algo_name}seed{seed}"
    if repeat_id is not None:
        exp_name = f"{exp_name}_r{repeat_id}"

    wandb_run_id = None

    if log_enabled:
        # Resume the wandb run IFF the caller loaded the checkpoint that id came
        # from. The hazard the old unconditional-fresh rule guarded against is
        # resuming an id whose checkpoint did NOT load: training restarts at step
        # 0 while wandb still holds previous_step, and every log below that step
        # is silently dropped. Tying the id to a confirmed load removes the
        # hazard without keeping its cost — a requeued job continuing at step N
        # appends to the same curve instead of opening a fragment.
        #
        # The on-disk id file is deliberately NOT consulted: it persists past the
        # checkpoint's deletion, which is exactly the unmatched-id case above.
        # It is written for downstream tooling and read by nobody.
        wandb_id_dir = os.path.join(save_dir, '.wandb_ids')
        wandb_id_file = os.path.join(wandb_id_dir, f'{project}_{exp_name}.json')
        if resume_wandb_run_id is None:
            try:
                if os.path.exists(wandb_id_file):
                    os.remove(wandb_id_file)
                    print(f"[wandb] Removed stale persistent wandb id file {wandb_id_file}")
            except Exception as e:
                print(f"[wandb] Could not remove persistent wandb id: {e}")

        if resume_wandb_run_id is not None:
            print(f"[wandb] Resuming run {resume_wandb_run_id} from the loaded checkpoint")
        setup_wandb(
            project=project,
            group=run_group,
            name=exp_name,
            mode=os.environ.get("WANDB_MODE", "online"),
            config=config,
            id=resume_wandb_run_id,
            # 'allow', not 'must': if the run was deleted server-side, or the id
            # predates a project rename, a new run beats aborting a job that has
            # a perfectly good checkpoint to continue from.
            resume=None if resume_wandb_run_id is None else 'allow',
        )
        wandb_run_id = wandb.run.id

        # Persist the wandb run id (kept for downstream tooling that reads it).
        try:
            os.makedirs(wandb_id_dir, exist_ok=True)
            with open(wandb_id_file, 'w') as f:
                json.dump({'wandb_run_id': wandb_run_id, 'project': project, 'name': exp_name}, f)
        except Exception as e:
            print(f"Warning: could not persist wandb ID: {e}")

        save_dir = os.path.join(save_dir, wandb.run.project, run_group, env_name, exp_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save config
        flag_dict = get_flag_dict(config)
        with open(os.path.join(save_dir, 'flags.json'), 'w') as f:
            json.dump(flag_dict, f)

        # Create CSV loggers
        csv_loggers = {
            prefix: CsvLogger(os.path.join(save_dir, f"{prefix}.csv"))
            for prefix in prefixes
        }

        logger = LoggingHelper(
            csv_loggers=csv_loggers,
            wandb_logger=wandb,
            log_enabled=True,
        )
    else:
        logger = LoggingHelper(
            csv_loggers={prefix: None for prefix in prefixes},
            wandb_logger=None,
            log_enabled=False,
        )

    return save_dir, logger, wandb_run_id

def get_actor_fn_for_bc_eval(agent: Any, config: Dict) -> Callable:
    """ Get the appropriate actor function for BC evaluation """
    return agent.compute_flow_actions

def get_actor_fn_for_rl_eval(agent: Any, config: Dict, use_ode: bool = False) -> Callable:
    """ Get the appropriate actor function for RL evaluation """
    if use_ode:
        return agent.compute_flow_actions
    else:
        # Use stochastic sampling (SDE) if available
        if hasattr(agent, 'sample_actions'):
            return agent.sample_actions
        else:
            return agent.compute_flow_actions


