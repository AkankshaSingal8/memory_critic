from os.path import expanduser
import os
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict
import imageio
import h5py

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic import DATASET_REGISTRY
from ogpo.utils.datasets import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# None  → no goal conditioning
# 'cumulative' → which goals completed so far, e.g. (1, 1, 0, 0, 0)
# 'singular'   → frontier goals (next on each branch), e.g. (0, 0, 1, 0, 0)
GOAL_CONDITIONING_MODES = (None, 'cumulative', 'singular')

# Per-task observation key configuration
# image_size: (H, W) matching --camera_height/--camera_width used in dataset_states_to_obs.py
TASK_CONFIG = {
    'lift': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['agentview_image'],
        'image_size': (96, 96),
    },
    'can': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['agentview_image'],
        'image_size': (96, 96),
    },
    'square': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['agentview_image'],
        'image_size': (96, 96),
    },
    'tool_hang': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'],
        'image_keys': ['sideview_image'],
        'image_size': (240, 240),
    },
    'transport': {
        'low_dim_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                         'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos', 'object'],
        'proprio_keys': ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
                         'robot1_eef_pos', 'robot1_eef_quat', 'robot1_gripper_qpos'],
        'image_keys': ['shouldercamera0_image', 'shouldercamera1_image'],
        'image_size': (96, 96),
    },
}


def _get_task_config(env_name):
    """Get per-task observation key configuration."""
    task = env_name.split("-")[0]
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task}. Supported: {list(TASK_CONFIG.keys())}")
    return TASK_CONFIG[task]


# Joint-space proprioception, appended when agent.critic_joint_proprio is on
# (gate in ogpo/main.py): default-on for the safe_qipm family (opt-out =
# the 2026-08-07 heads-without-proprio ablation), default-off opt-in for
# qipm and sac_qipm
# (the "base + proprio critics" controls): the joint_limits / velocity /
# collision safety signals
# are functions of q and q̇, neither of which the default eef-centric key set
# contains (7-DoF arm: eef pose does not determine q; no velocity information
# in a single frame). Both keys exist in the robomimic low_dim HDF5s' obs/ and
# next_obs/ groups, so env and dataset widen together.
SAFETY_JOINT_PROPRIO_KEYS = ('robot0_joint_pos', 'robot0_joint_vel')


def _resolve_low_dim_keys(task_cfg, include_joint_proprio):
    """Effective low_dim obs keys for env AND dataset construction.

    The single source of truth for both readers: make_env concatenates
    raw_obs[k] and get_dataset concatenates HDF5 obs/{k} in this list's order,
    so any divergence silently permutes features. Proprio keys append AFTER
    'object' — layout with the flag on (tool_hang):
    [eef_pos(3), eef_quat(4), gripper_qpos(2), object(44),
     joint_pos(7), joint_vel(7)] = 67 dims.
    """
    keys = list(task_cfg['low_dim_keys'])
    if include_joint_proprio:
        assert not any(k.startswith('robot1_') for k in keys), (
            "include_joint_proprio is single-robot only: extend "
            "SAFETY_JOINT_PROPRIO_KEYS per robot before using it on a "
            "multi-robot task like transport"
        )
        keys += [k for k in SAFETY_JOINT_PROPRIO_KEYS if k not in keys]
    return keys


def is_robomimic_env(env_name):
    """determine if an env is robomimic"""
    try:
        if ("low_dim" not in env_name) and ("image" not in env_name):
            return False
        task, dataset_type, hdf5_type = env_name.split("-")
        return task in ("lift", "can", "square", "transport", "tool_hang") and dataset_type in ("mh", "ph")
    except ValueError:
        return False

# Default initialization for backward compatibility (overridden per-task in make_env/get_dataset)
low_dim_keys = {"low_dim": ('robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'object')}
ObsUtils.initialize_obs_modality_mapping_from_dict(low_dim_keys)


def _get_max_episode_length(env_name):
    if env_name.startswith("lift"):
        return 300
    elif env_name.startswith("can"):
        return 300
    elif env_name.startswith("square"):
        return 400
    elif env_name.startswith("transport"):
        return 800
    elif env_name.startswith("tool_hang"):
        return 1000
    else:
        raise ValueError(f"Unsupported environment: {env_name}")


# ---------------------------------------------------------------------------
# Goal-vector encoding utilities
# ---------------------------------------------------------------------------

def cumulative_to_singular(cumulative: np.ndarray) -> np.ndarray:
    """Convert a cumulative goal-status array to a singular (frontier) array.

    A goal at index *i* is a **frontier** goal when:
    - it is not yet completed (cumulative[i] == 0), AND
    - it is the first goal (i == 0) OR its predecessor is completed
      (cumulative[i-1] == 1).

    This handles both strictly sequential tasks (produces exactly one 1) and
    tasks with parallel DAG branches (one 1 per active branch).

    Examples:
        cumulative (1, 1, 0, 0, 0) → singular (0, 0, 1, 0, 0)
        cumulative (1, 1, 0, 0, 1, 0, 0) → singular (0, 0, 1, 0, 0, 1, 0)

    Args:
        cumulative: int/bool array of shape (..., goal_dim) with binary done flags.

    Returns:
        int8 array of the same shape, 1 at each frontier goal, 0 elsewhere.
    """
    arr = np.asarray(cumulative)
    incomplete = arr == 0
    # Predecessor done: either no predecessor (first position) or previous is done.
    pred_done = np.concatenate(
        [np.ones((*arr.shape[:-1], 1), dtype=arr.dtype), arr[..., :-1]],
        axis=-1,
    ).astype(bool)
    return np.where(incomplete & pred_done, np.int8(1), np.int8(0))


def _encode_goal_vector(cumulative: np.ndarray, mode: str) -> np.ndarray:
    """Encode a binary cumulative-done array into the requested goal representation.

    Args:
        cumulative: int/bool array (..., goal_dim) of binary done flags (0/1).
        mode: 'cumulative' or 'singular'.

    Returns:
        int8 array of the same shape.
    """
    if mode == 'cumulative':
        return np.asarray(cumulative, dtype=np.int8)
    if mode == 'singular':
        return cumulative_to_singular(cumulative)
    raise ValueError(f'Unknown goal_conditioning_mode: {mode!r}. Choose from {GOAL_CONDITIONING_MODES}.')


# ============================================================================
# ToolHang Subgoal Detection Functions
# ============================================================================

def _check_grasp_frame(base_env, lift_threshold=0.02):
    """Check if the gripper is grasping and lifting the frame.

    Subgoal 1: Pick up the frame from the table

    Returns True if:
    1. Both gripper fingerpads are in contact with frame
    2. Frame is lifted above the table
    """
    gripper = base_env.robots[0].gripper["right"]

    # Check if both fingerpads are grasping the frame
    if not base_env._check_grasp(gripper, base_env.frame.contact_geoms):
        return False

    # Check if frame is lifted above table
    frame_body_id = base_env.sim.model.body_name2id(base_env.frame.root_body)
    frame_z = base_env.sim.data.body_xpos[frame_body_id][2]
    table_z = base_env.model.mujoco_arena.table_offset[2]
    return frame_z > (table_z + lift_threshold)


def _check_grasp_wrench(base_env, lift_threshold=0.02):
    """Check if the gripper is grasping and lifting the wrench (tool).

    Subgoal 3: Pick up the wrench from the table

    Returns True if:
    1. Both gripper fingerpads are in contact with wrench
    2. Wrench is lifted above the table
    """
    gripper = base_env.robots[0].gripper["right"]

    # Check if both fingerpads are grasping the wrench/tool
    if not base_env._check_grasp(gripper, base_env.tool.contact_geoms):
        return False

    # Check if wrench is lifted above table
    tool_body_id = base_env.sim.model.body_name2id(base_env.tool.root_body)
    tool_z = base_env.sim.data.body_xpos[tool_body_id][2]
    table_z = base_env.model.mujoco_arena.table_offset[2]
    return tool_z > (table_z + lift_threshold)


# ============================================================================
# Transport Subgoal Detection Functions
# ============================================================================

def _check_lid_removed(base_env):
    """Check if the lid has been removed from the start bin.
    Detects leftward movement of lid handle (lid slides/drops left when removed).
    """
    lid_pos = base_env.transport.lid_handle_pos
    initial_lid_x = 0.2
    left_threshold = 0.08
    return lid_pos[0] < (initial_lid_x - left_threshold)


def _check_payload_grasped(base_env):
    """Check if Robot 0 is grasping the payload."""
    robot0 = base_env.robots[0]
    if hasattr(robot0, 'gripper'):
        gripper = robot0.gripper
        if isinstance(gripper, dict):
            gripper = list(gripper.values())[0]
        return base_env._check_grasp(gripper, base_env.transport.payload.contact_geoms)
    return False


def _check_payload_lifted(base_env, lift_threshold=0.1):
    """Check if the payload is lifted above the bins.
    Subgoal 3: Lift Payload
    """
    payload_pos = base_env.transport.payload_pos
    table_z = base_env.table_offsets[0, 2]
    return payload_pos[2] > (table_z + lift_threshold)


def _check_trash_grasped(base_env):
    """Check if Robot 1 is grasping the trash."""
    robot1 = base_env.robots[1]
    if hasattr(robot1, 'gripper'):
        gripper = robot1.gripper
        if isinstance(gripper, dict):
            gripper = list(gripper.values())[0]
        return base_env._check_grasp(gripper, base_env.transport.trash.contact_geoms)
    return False


def _check_payload_handed_off(base_env):
    """Check if the payload has been handed off from Robot 0 to Robot 1.

    The handoff is complete when:
    - Robot 1 IS grasping the payload
    - Robot 0 is NOT grasping the payload
    """
    robot0_grasping = robot1_grasping = False
    for i, robot in enumerate(base_env.robots):
        if not hasattr(robot, 'gripper'):
            continue
        grippers = robot.gripper if isinstance(robot.gripper, dict) else {'default': robot.gripper}
        for _, gripper in grippers.items():
            if base_env._check_grasp(gripper, base_env.transport.payload.contact_geoms):
                if i == 0:
                    robot0_grasping = True
                else:
                    robot1_grasping = True
                break
    # Handoff complete when Robot 1 has it and Robot 0 doesn't
    return robot1_grasping and not robot0_grasping


# ---------------------------------------------------------------------------
# Shared subtask status computation
# ---------------------------------------------------------------------------

def _compute_subtask_status(base_env, goal_done, no_of_subtasks):
    """Compute subtask completion status from current sim state.

    Returns (updated_goal_done, reward_info_dict).
    """
    if goal_done is None:
        goal_done = [False] * no_of_subtasks
    else:
        assert len(goal_done) == no_of_subtasks

    if hasattr(base_env, "transport"):
        goals = [
            _check_lid_removed(base_env),                        # 0: LID_REMOVED
            _check_payload_grasped(base_env),                    # 1: PAYLOAD_GRASPED
            _check_payload_lifted(base_env),                     # 2: PAYLOAD_LIFTED
            _check_trash_grasped(base_env),                      # 3: TRASH_GRASPED
            bool(base_env.transport.trash_in_trash_bin),         # 4: TRASH_IN_BIN
            _check_payload_handed_off(base_env),                 # 5: PAYLOAD_HANDED_OFF
            bool(base_env.transport.payload_in_target_bin),      # 6: PAYLOAD_IN_TARGET
        ]
    elif hasattr(base_env, "nuts") and hasattr(base_env, "on_peg"):
        nut_id = getattr(base_env, "nut_id", 0)
        nut_body_id = base_env.sim.model.body_name2id(base_env.nuts[nut_id].root_body)
        nut_pos = base_env.sim.data.body_xpos[nut_body_id]
        peg_body_id = base_env.peg1_body_id if nut_id == 0 else base_env.peg2_body_id
        peg_pos = base_env.sim.data.body_xpos[peg_body_id]
        table_z = base_env.table_offset[2]
        lift_threshold = table_z + 0.0713
        xy_dist = np.linalg.norm(nut_pos[:2] - peg_pos[:2])
        is_grasped = base_env._check_grasp(
            gripper=base_env.robots[0].gripper,
            object_geoms=base_env.nuts[nut_id].contact_geoms,
        )
        goals = [
            is_grasped and (nut_pos[2] < lift_threshold),
            is_grasped and (nut_pos[2] > lift_threshold) and (xy_dist < 0.2),
            bool(base_env._check_success()),
        ]
    elif hasattr(base_env, "tool") and hasattr(base_env, "frame") and hasattr(base_env, "stand"):
        is_frame_grasped = _check_grasp_frame(base_env)
        is_frame_inserted = base_env._check_frame_assembled()
        is_wrench_grasped = _check_grasp_wrench(base_env)
        is_wrench_attached = base_env._check_tool_on_frame()
        if no_of_subtasks == 4:
            goals = [is_frame_grasped, is_frame_inserted, is_frame_inserted and is_wrench_grasped, is_wrench_attached]
        elif no_of_subtasks == 1:
            goals = [is_wrench_attached]
        else:
            raise ValueError("number of subgoals not configured for tool_hang")
    else:
        raise ValueError(f"Unsupported dense reward env: {type(base_env)}")

    updated = list(goal_done)
    for i in range(min(len(goals), no_of_subtasks)):
        if goals[i]:
            updated[i] = True

    return updated, {
        "subtasks_completed": sum(updated),
        "subtask_status": [int(g) for g in updated],
    }


# ---------------------------------------------------------------------------
# Safety cost / margin columns (safe_qipm family)
# ---------------------------------------------------------------------------

SAFETY_HEAD_MODES = ("single", "multi")

# What the trailing subtask_status columns CARRY. Orthogonal to
# SAFETY_HEAD_MODES (which decides how MANY columns): the signal is chosen by
# agent identity in ogpo/main.py, not by config.
#   "cost"   — 0/1 violation flags from info['safety_violations'] (safe_qipm's
#              CMDP cost heads; lower = safer, columns are costs).
#   "margin" — signed distances from info['safety_margins'] (safe_qipm_hjb's
#              HJB reachability heads; HIGHER = safer, 0 = envelope boundary).
# The two are not interchangeable: an HJB head fed cost flags reads "0" as
# "sitting exactly on the boundary", and a cost head fed margins backs up
# positive rewards it can never represent.
SAFETY_SIGNALS = ("cost", "margin")


def _resolve_safety_roster(base_env):
    """Fixed-order safety-constraint names published by the env's controllers.

    Answerable BEFORE the first step: safe-robosuite copies the schema off the
    safety filter at controller construction. Returns [] when no controller has
    a safety filter.

    MUST match `RobotEnv._post_action` in safe-robosuite (the "prefix only when
    more than one robot contributes" rule) — what this returns has to equal
    `info['safety_constraints']` exactly, or the per-constraint columns would be
    attributed to the wrong critic heads. `step()` asserts the two agree.
    """
    per_robot = []
    for idx, robot in enumerate(getattr(base_env, "robots", [])):
        cc = getattr(robot, "composite_controller", None)
        getter = getattr(cc, "get_safety_constraints", None)
        if getter is None:
            continue  # upstream controller without the safety API
        names = list(getter())
        if names:
            per_robot.append((idx, names))

    prefixed = len(per_robot) > 1
    roster = []
    for idx, names in per_robot:
        roster += [f"robot{idx}_{n}" for n in names] if prefixed else names
    return roster


def _safety_cost_columns(info, safety_head_mode, no_of_safety_subtasks, roster):
    """Per-step safety cost column(s) to append to subtask_status (safe_qipm).

    "single" -> one column, the aggregate float(info['unsafe_step']).
    "multi"  -> K columns from info['safety_violations'], the per-constraint 0/1
                decomposition parallel to info['safety_constraints'].

    CONTRACT (safe-robosuite, 2026-08-04): `unsafe_step` is exactly
    `any(safety_violations)`. Both are decided from the REQUEST `q_des` before the
    safety filter runs, so the aggregate and the decomposition come from the same
    quantity and cannot disagree — an all-zero vector means no family flagged the
    step. This used to be one-way only (violations were latched inside a
    `deviation > unsafe_tol` branch on the FILTERED command), which is why an
    "unattributed unsafe" counter existed; that state is no longer representable.

    Because detection is demand-side, the columns describe what the policy ASKED
    for and are identical whether or not `apply_intervention` prevented it — which
    is what makes the four experimental conditions comparable.

    KNOWN GAP: a step whose QP found no solution escapes both. safe-robosuite then
    discards the request (`q_safe = q_curr`, tagged `infeasible`), but "infeasible"
    is not a constraint family, so such a step reports `unsafe_step` False with an
    all-zero vector. Rare, and `velocity` co-fires on essentially every one.
    """
    if safety_head_mode == "single":
        cols = [float(info['unsafe_step'])]
    else:
        names = list(info['safety_constraints'])
        violations = list(info['safety_violations'])
        # Order drift would silently mis-attribute columns to heads, so compare
        # the whole roster rather than just its length.
        assert names == roster, (
            f"env safety roster changed mid-run: configured {roster}, got {names}"
        )
        assert len(violations) == no_of_safety_subtasks, (
            f"env reported {len(violations)} safety violations but "
            f"{no_of_safety_subtasks} safety heads are configured"
        )
        cols = [float(v) for v in violations]

    # Guards the upstream invariant in both modes whenever the key is published
    # (single mode does not consume it, but a break here means the aggregate and
    # the decomposition disagree, which would silently change the cost scale).
    if 'safety_violations' in info and any(info['safety_violations']):
        assert info['unsafe_step'], (
            "safety_violations fired but unsafe_step is False — "
            "upstream invariant any(violations) => unsafe_step is broken"
        )
    return cols


def _safety_margin_columns(info, safety_head_mode, no_of_safety_subtasks, roster):
    """Per-step safety MARGIN column(s) to append to subtask_status (safe_qipm_hjb).

    K columns straight off ``info['safety_margins']``, the signed-distance
    parallel of ``info['safety_violations']`` — same list, same fixed order as
    ``info['safety_constraints']``. Contract of record:
    ``../safe-robosuite/docs/changes/2026-07-26-safety-margin-logging/``.

      - **Sign**: positive = safe, negative = violated, and **zero is the
        CONFIGURED ENVELOPE boundary** (buffered joint band lo+beta..hi-beta,
        the collision pair margin, a workspace box face, v_max) — NOT the
        physical failure surface.
      - **Units are raw and heterogeneous**: rad (joint_limits), rad/s
        (velocity), m (collision, workspace). Nothing is normalized here; the
        buffer and every metric stay in physical units. Cross-head
        commensurability is the agent's problem (safe_qipm_hjb's
        ``safety_margin_scales``, applied inside the HJB backup only).
      - **Min-latch**: upstream records the minimum over the inner sim steps of
        one policy step, every inner step (not gated on unsafe), reset per
        policy step and per episode — so a column is the worst margin actually
        reached during that action, not an end-of-step snapshot.
      - **Measured on the CURRENT state** (post whatever was executed last
        step). Under an intervening safety filter the trajectory is censored at
        the boundary, so margins observed on a shielded (`_p_i`) run describe
        the shielded system, not the unshielded policy.
      - **Multi mode only**: "single" would need a min-over-margins aggregate
        and a roster that single mode does not resolve (see
        configure_safety_status). Rejected fail-fast.

    WARNING — there is NO violation/margin invariant to check here, in either
    direction. Upstream's violation detectors are LOOKAHEADS while margins are
    state-only, so flags LEAD margins (measured ~3 steps upstream):
    ``violation == 1 => margin <= 0`` is FALSE BY DESIGN, and so is its
    converse. ``_safety_cost_columns`` asserts an upstream invariant; this
    function deliberately asserts none.

    Non-finite margins ARE publishable upstream (``inf`` for a velocity family
    with no finite cap or a collision family with every pair pruned; ``nan``
    from the missing-getter fallback) but never occur for the toolhang configs
    we run. They are rejected fail-fast because a single one poisons the whole
    critic: the HJB backup's min/blend propagates it to every timestep of the
    chunk and then, via bootstrapping, to every head.
    """
    assert safety_head_mode == "multi", (
        "safety_signal='margin' supports safety_head_mode='multi' only; "
        "single-mode aggregation (min over margins) is not implemented"
    )
    names = list(info['safety_constraints'])
    margins = list(info['safety_margins'])
    # Order drift would silently mis-attribute columns to heads, so compare the
    # whole roster rather than just its length.
    assert names == roster, (
        f"env safety roster changed mid-run: configured {roster}, got {names}"
    )
    assert len(margins) == no_of_safety_subtasks, (
        f"env reported {len(margins)} safety margins but "
        f"{no_of_safety_subtasks} safety heads are configured"
    )
    cols = [float(m) for m in margins]
    assert all(np.isfinite(c) for c in cols), (
        f"env published a non-finite safety margin: {dict(zip(names, cols))} — "
        "inf/nan propagate through the HJB backup into every head"
    )
    return cols


def _unwrap_lowdim_wrapper(env):
    """Find the RobomimicLowdimWrapper in the wrapper chain."""
    wrapped = env
    for _ in range(20):
        if wrapped is None:
            return None
        if (hasattr(wrapped, "goal_conditioning_mode")
                and hasattr(wrapped, "no_of_subtasks")
                and hasattr(wrapped, "_build_observation")
                and hasattr(wrapped, "env")):
            return wrapped
        if not hasattr(wrapped, "env"):
            return None
        wrapped = wrapped.env
    return None


def _infer_subtask_status_from_states(lowdim_wrapper, states, episode_len, no_of_subtasks):
    """Infer cumulative subtask status by replaying simulator states.

    Returns int8 array of shape (episode_len, no_of_subtasks).
    """
    inferred = np.zeros((episode_len, no_of_subtasks), dtype=np.int8)
    if episode_len <= 0:
        return inferred
    sim_states = np.asarray(states)
    assert sim_states.shape[0] == episode_len, \
        f"State/action length mismatch: states={sim_states.shape[0]}, episode_len={episode_len}"
    goal_done = [False] * no_of_subtasks
    base_env = lowdim_wrapper.env.env
    for t, sim_state in enumerate(sim_states):
        lowdim_wrapper.env.reset_to({"states": sim_state})
        goal_done, reward_info = _compute_subtask_status(base_env, goal_done, no_of_subtasks)
        inferred[t, :] = np.asarray(reward_info["subtask_status"], dtype=np.int8)[:no_of_subtasks]
    return inferred


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(env_name, seed=0, frame_stack=1, goal_conditioning_mode=None, no_of_subtasks=None,
             include_joint_proprio=False, renderable=False):
    """Create a robomimic environment.

    NOTE: call get_dataset() first so that dataset metadata is downloaded.
    include_joint_proprio widens the low_dim obs with SAFETY_JOINT_PROPRIO_KEYS
    (agent.critic_joint_proprio: safe_qipm family default-on, qipm/sac_qipm
    opt-in; see
    _resolve_low_dim_keys). Low-dim envs only.

    renderable=True builds the env with an offscreen renderer (needs MUJOCO_GL=egl)
    so env.render() works, using the task's informative camera (e.g. sideview for
    tool_hang) rather than the wrapper's agentview default. Only the dedicated
    video-capture env needs this; training and eval envs keep it False to avoid
    paying for an extra EGL context per worker. Without it robosuite never builds
    the offscreen context and MjSim.render trips an assert.
    """
    if goal_conditioning_mode not in GOAL_CONDITIONING_MODES:
        raise ValueError(
            f'Invalid goal_conditioning_mode: {goal_conditioning_mode!r}. '
            f'Choose from {GOAL_CONDITIONING_MODES}.'
        )
    if include_joint_proprio and "image" in env_name:
        raise ValueError(
            "include_joint_proprio is supported only for low_dim robomimic envs; "
            "image envs select state via the separate proprio_keys path"
        )

    dataset_path = _check_dataset_exists(env_name)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

    # Override the dataset-recorded controller with a custom composite
    # controller config (e.g. SAFE_WHOLE_BODY_MINK_IK for the safety-heads
    # work). Mirrors OGPO_public; read here so the train env, eval env, and
    # the AsyncVectorEnv subprocess envs (which inherit os.environ) all get
    # the same controller. A bad path raises FileNotFoundError in the loader.
    controller_override = os.environ.get("ROBOSUITE_CONTROLLER_CONFIG")
    if controller_override:
        from robosuite.controllers import load_composite_controller_config
        env_meta["env_kwargs"]["controller_configs"] = load_composite_controller_config(
            controller=controller_override
        )

    max_episode_length = _get_max_episode_length(env_name)
    task_cfg = _get_task_config(env_name)
    # Task-informative camera for video capture (e.g. sideview for tool_hang);
    # the wrapper default is only correct for the tasks whose image_keys are
    # agentview_image, so derive it rather than hardcoding.
    render_camera = task_cfg['image_keys'][0].replace('_image', '') if renderable else "agentview"

    if "image" in env_name:
        if goal_conditioning_mode is not None:
            raise ValueError("goal_conditioning is currently supported only for low_dim robomimic envs")
        image_keys = task_cfg['image_keys']
        proprio_keys = tuple(task_cfg['proprio_keys'])
        img_h, img_w = task_cfg['image_size']

        shape_meta = {
            'obs': {
                'rgb': {'shape': (3, img_h, img_w)},
            },
        }

        obs_keys = {"low_dim": list(proprio_keys), 'rgb': image_keys}
        ObsUtils.initialize_obs_modality_mapping_from_dict(obs_keys)

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=renderable,
            use_image_obs=True,
        )
        env = RobomimicImageWrapper(
            env, shape_meta=shape_meta, image_keys=image_keys,
            proprio_keys=proprio_keys,
            max_episode_length=max_episode_length, num_stack=frame_stack,
            render_camera_name=render_camera,
        )
    else:
        task_low_dim_keys = _resolve_low_dim_keys(task_cfg, include_joint_proprio)
        ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": task_low_dim_keys})

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=renderable,
        )
        env = RobomimicLowdimWrapper(
            env, low_dim_keys=task_low_dim_keys,
            max_episode_length=max_episode_length,
            goal_conditioning_mode=goal_conditioning_mode,
            no_of_subtasks=no_of_subtasks,
            render_camera_name=render_camera,
        )
    env.seed(seed)
    env.env.hard_reset = False

    # Side-car safety info for controllers that publish none (OSC); set by
    # ogpo/main.py from eval.osc_safety_monitor_config. Inherited via os.environ
    # by the AsyncVectorEnv workers, like ROBOSUITE_CONTROLLER_CONFIG above.
    from envs.osc_safety_sidecar import configured_monitor_path, ENV_VAR
    sidecar_json = configured_monitor_path()
    if sidecar_json is not None:
        if controller_override:
            raise ValueError(
                f"{ENV_VAR} (OSC side-car safety monitor) and ROBOSUITE_CONTROLLER_CONFIG are both "
                "set; the side-car is for controllers that publish no safety info -- unset one")
        if "image" in env_name:
            raise ValueError(f"{ENV_VAR} is supported for low_dim robomimic envs only")
        env.enable_osc_safety_monitor(sidecar_json)

    # Executed-motion safety flags + shield-intervention stages (envs/shield_diagnostics.py),
    # opt-in through os.environ like the side-car above (AsyncVectorEnv workers inherit it).
    from envs import shield_diagnostics
    if shield_diagnostics.enabled():
        if "image" in env_name:
            raise ValueError(f"{shield_diagnostics.ENV_VAR} is supported for low_dim robomimic envs only")
        env._shield_diagnostics = shield_diagnostics.ShieldDiagnostics()
    return env

def _check_dataset_exists(env_name):
    # enforce that the dataset exists
    task, dataset_type, hdf5_type = env_name.split("-")
    if hdf5_type == "image":
        file_name = "image_v15.hdf5"
    elif dataset_type == "mg":
        file_name = "low_dim_sparse_v15.hdf5"
    else:
        file_name = f"low_dim_v15.hdf5"
    dataset_root = os.environ.get("ROBOMIMIC_DATASET_ROOT")

    if dataset_root is None:
        username = os.environ.get("USER")
        if username:
            dataset_root = os.path.join("/data/user_data", username, "robomimic")
        else:
            dataset_root = os.path.join(expanduser("~"), ".robomimic")

    dataset_path = os.path.join(
        dataset_root,
        task,
        dataset_type,
        file_name,
    )
    # Restored 2026-08-24 (it had been commented out, leaving a pre-flight `ls`
    # in the launch procedure as the only check). Both callers are protected:
    # without it a typo'd ROBOMIMIC_DATASET_ROOT surfaces as a robomimic
    # metadata error inside make_env, or as an h5py OSError inside get_dataset,
    # minutes into a job that has already claimed a GPU.
    assert os.path.exists(dataset_path), (
        f"robomimic dataset not found: {dataset_path} "
        f"(dataset_root={dataset_root!r}, resolved from "
        f"{'$ROBOMIMIC_DATASET_ROOT' if os.environ.get('ROBOMIMIC_DATASET_ROOT') else 'the built-in fallback'})"
    )

    return dataset_path


def describe_dataset(env_name):
    """Identity of the offline HDF5 `env_name` resolves to, for run provenance.

    `env.name` is `tool_hang-ph-low_dim` for the 20-, 50-, 75- and 100-traj
    datasets alike — only ROBOMIMIC_DATASET_ROOT tells them apart, and the sole
    trace of which file loaded was a stdout print in get_dataset. This dict is
    injected into the top-level config in ogpo/main.py, so it lands in
    flags.json, the wandb config and the resume guard's digest.

    Keyed off _check_dataset_exists — the one path resolver make_env and
    get_dataset both call — so the recorded path is by construction the file
    that was loaded, not a re-derivation that could drift from it.

    Deliberately NOT a content hash: sha256 of a multi-GB HDF5 costs tens of
    seconds at every run start. path + size + mtime + transition count
    distinguishes every realistic mix-up (a different traj-count dataset under
    the same env name changes the transition count, which is read from the
    per-demo action shapes — metadata only, no bulk read).
    """
    dataset_path = _check_dataset_exists(env_name)
    stat = os.stat(dataset_path)
    with h5py.File(dataset_path, "r") as rm_dataset:
        transitions = sum(
            int(rm_dataset[f"data/{ep}/actions"].shape[0])
            for ep in rm_dataset["data"].keys()
        )
    return {
        'dataset_path': dataset_path,
        'dataset_size_bytes': int(stat.st_size),
        'dataset_mtime_ns': int(stat.st_mtime_ns),
        'dataset_transitions': transitions,
    }


def get_dataset(env, env_name, include_joint_proprio=False):
    """Load offline dataset. When goal_conditioning_mode is set, appends goal vectors to observations."""
    dataset_path = _check_dataset_exists(env_name)
    task_cfg = _get_task_config(env_name)

    rm_dataset = h5py.File(dataset_path, "r")
    demos = list(rm_dataset["data"].keys())
    num_demos = len(demos)
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    num_timesteps = 0
    for ep in demos:
        num_timesteps += int(rm_dataset[f"data/{ep}/actions"].shape[0])

    print(f"the size of the dataset is {num_timesteps}")

    # Check if this is an image environment
    is_image_env = "image" in env_name

    # Use per-task keys
    image_keys = task_cfg['image_keys']
    proprio_keys = task_cfg['proprio_keys']
    # Must resolve identically to make_env's list — same helper, same flag
    # (see _resolve_low_dim_keys: env and HDF5 columns concat in this order).
    task_low_dim_keys = _resolve_low_dim_keys(task_cfg, include_joint_proprio)

    # Goal conditioning setup
    lowdim_wrapper = _unwrap_lowdim_wrapper(env)
    gc_mode = getattr(lowdim_wrapper, "goal_conditioning_mode", None) if lowdim_wrapper is not None else None
    goal_conditioned = gc_mode is not None
    status_dim = int(getattr(lowdim_wrapper, "no_of_subtasks", 0)) if lowdim_wrapper is not None else 0
    if goal_conditioned and status_dim <= 0:
        raise AssertionError("Goal-conditioned dataset loading requires a positive no_of_subtasks.")

    # data holders
    observations = []      # always proprioceptive state
    actions = []
    next_observations = []  # always proprioceptive state
    subtask_status = []
    terminals = []
    rewards = []
    masks = []
    all_images = []         # only for image envs
    all_next_images = []    # only for image envs

    # go through and add to the data holder
    for ep in demos:
        ep_group = rm_dataset[f"data/{ep}"]
        a = np.array(ep_group["actions"])
        dones = np.array(ep_group["dones"])
        r = np.array(ep_group["rewards"])
        # Transform rewards from [0, 1] to [-1, 0]
        r = r - 1.0

        if is_image_env:
            # --- Load images ---
            img_obs_parts = []
            img_next_parts = []
            for key in image_keys:
                img_obs_parts.append(np.array(rm_dataset[f"data/{ep}/obs/{key}"]).astype(np.float32))
                img_next_parts.append(np.array(rm_dataset[f"data/{ep}/next_obs/{key}"]).astype(np.float32))
            img_obs = np.concatenate(img_obs_parts, axis=3) if len(img_obs_parts) > 1 else img_obs_parts[0]
            img_next = np.concatenate(img_next_parts, axis=3) if len(img_next_parts) > 1 else img_next_parts[0]

            # Frame stacking for images
            num_stack = 1
            current_env = env
            while hasattr(current_env, 'env'):
                if hasattr(current_env, 'num_stack'):
                    num_stack = current_env.num_stack
                    break
                current_env = current_env.env
            if hasattr(current_env, 'num_stack'):
                num_stack = current_env.num_stack

            if num_stack > 1:
                all_frames = np.concatenate([img_obs, img_next[-1:]], axis=0)
                pad = np.repeat(all_frames[:1], num_stack - 1, axis=0)
                padded_frames = np.concatenate([pad, all_frames], axis=0)
                stacks = [padded_frames[i : i + img_obs.shape[0] + 1] for i in range(num_stack)]
                stacked_frames = np.concatenate(stacks, axis=3)
                img_obs = stacked_frames[:-1]
                img_next = stacked_frames[1:]

            all_images.append(img_obs.astype(np.float32))
            all_next_images.append(img_next.astype(np.float32))

            # --- Load proprioceptive state ---
            state_obs = np.concatenate(
                [np.array(rm_dataset[f"data/{ep}/obs/{k}"]) for k in proprio_keys], axis=-1)
            state_next = np.concatenate(
                [np.array(rm_dataset[f"data/{ep}/next_obs/{k}"]) for k in proprio_keys], axis=-1)
            observations.append(state_obs.astype(np.float32))
            next_observations.append(state_next.astype(np.float32))
            ep_subtask_status = np.zeros((len(a), max(status_dim, 1)), dtype=np.int8)
        else:
            # Load low-dimensional observations
            obs, next_obs = [], []
            for k in task_low_dim_keys:
                obs.append(np.array(ep_group[f"obs/{k}"]))
            for k in task_low_dim_keys:
                next_obs.append(np.array(ep_group[f"next_obs/{k}"]))
            obs = np.concatenate(obs, axis=-1)
            next_obs = np.concatenate(next_obs, axis=-1)

            if goal_conditioned:
                if "states" not in ep_group:
                    raise AssertionError(f"Expected simulator states for {ep}, but `states` is missing.")
                ep_cumulative = _infer_subtask_status_from_states(
                    lowdim_wrapper=lowdim_wrapper,
                    states=ep_group["states"],
                    episode_len=len(a),
                    no_of_subtasks=status_dim,
                )
                # obs[t] reflects status at t-1; episodes start from zero status.
                obs_cumulative = np.zeros_like(ep_cumulative)
                obs_cumulative[1:] = ep_cumulative[:-1]
                obs_goal = _encode_goal_vector(obs_cumulative, gc_mode).astype(obs.dtype, copy=False)
                next_goal = _encode_goal_vector(ep_cumulative, gc_mode).astype(next_obs.dtype, copy=False)
                obs = np.concatenate([obs, obs_goal], axis=-1)
                next_obs = np.concatenate([next_obs, next_goal], axis=-1)
                ep_subtask_status = ep_cumulative
            else:
                ep_subtask_status = np.zeros((len(a), max(status_dim, 1)), dtype=np.int8)

            observations.append(obs.astype(np.float32))
            next_observations.append(next_obs.astype(np.float32))

        actions.append(a.astype(np.float32))
        rewards.append(r.astype(np.float32))
        terminals.append(dones.astype(np.float32))
        masks.append(1.0 - dones.astype(np.float32))
        subtask_status.append(ep_subtask_status)

    concat_next_obs = np.concatenate(next_observations, axis=0)
    dataset_fields = dict(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=concat_next_obs,
        subtask_status=np.concatenate(subtask_status, axis=0),
    )
    if is_image_env:
        dataset_fields['images'] = np.concatenate(all_images, axis=0)
        dataset_fields['next_images'] = np.concatenate(all_next_images, axis=0)

    return Dataset.create(**dataset_fields)


class RobomimicLowdimWrapper(gym.Env):
    """
    Environment wrapper for Robomimic environments with state observations.
    Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py
    """
    def __init__(
        self,
        env,
        normalization_path=None,
        low_dim_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=None,
        goal_conditioning_mode=None,
        no_of_subtasks=None,
    ):
        if goal_conditioning_mode not in GOAL_CONDITIONING_MODES:
            raise ValueError(
                f'Invalid goal_conditioning_mode: {goal_conditioning_mode!r}. '
                f'Choose from {GOAL_CONDITIONING_MODES}.'
            )

        self.env = env
        self.obs_keys = low_dim_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.env_step = 0
        self.n_episodes = 0
        self.t = 0
        self.episode_return = 0
        self.episode_length = 0
        self.goal_conditioning_mode = goal_conditioning_mode
        self.no_of_subtasks = int(no_of_subtasks) if no_of_subtasks is not None else 0
        if self.goal_conditioning_mode is not None and self.no_of_subtasks <= 0:
            raise ValueError("goal_conditioning_mode != None requires no_of_subtasks to be a positive integer.")
        self.goal_done = None
        self._last_reward_info = {}

        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces - use [-1, 1]
        low = np.full(env.action_dimension, fill_value=-1.)
        high = np.full(env.action_dimension, fill_value=1.)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1, dtype=np.float32)
        high = np.full_like(obs_example, fill_value=1, dtype=np.float32)
        if self.goal_conditioning_mode is not None:
            low[-self.no_of_subtasks:] = 0.0
            high[-self.no_of_subtasks:] = 1.0
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def _current_goal_vector(self):
        """Return encoded goal vector for the current step."""
        if self.no_of_subtasks <= 0 or self.goal_conditioning_mode is None:
            return np.zeros((0,), dtype=np.int8)
        if self.goal_done is None:
            self.goal_done = [False] * self.no_of_subtasks
        cumulative = np.asarray(self.goal_done, dtype=np.int8)
        return _encode_goal_vector(cumulative, self.goal_conditioning_mode)

    def _build_observation(self, raw_obs):
        """Build observation from raw dict, optionally appending goal vector."""
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        if self.normalize:
            obs = self.normalize_obs(obs)
        if self.goal_conditioning_mode is not None:
            goal_vec = self._current_goal_vector().astype(obs.dtype, copy=False)
            obs = np.concatenate([obs, goal_vec], axis=0)
        return obs

    def get_observation(self):
        raw_obs = self.env.get_observation()
        return self._build_observation(raw_obs)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, seed=None, options=None, **kwargs):
        """Reset environment with optional seed for reproducible diverse resets."""
        self.t = 0
        self.episode_return, self.episode_length = 0, 0
        self.goal_done = None
        self._last_reward_info = {}
        self.n_episodes += 1
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        if options is None:
            options = {}

        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Prefer explicit seed kwarg (gymnasium standard), fall back to options
        new_seed = seed if seed is not None else options.get("seed", None)
        if self.init_state is not None:
            self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            self.env.reset()
        else:
            self.env.reset()

        # OSC side-car safety monitor (envs/osc_safety_sidecar.py): rebind to the
        # live sim after every reset.
        if getattr(self, '_safety_sidecar', None) is not None:
            self._safety_sidecar.after_reset()

        return self.get_observation(), {}

    def enable_osc_safety_monitor(self, controller_json_path):
        """Publish side-car safety info keys on every step (see envs/osc_safety_sidecar.py)."""
        from envs.osc_safety_sidecar import SideCarSafetyInfo
        self._safety_sidecar = SideCarSafetyInfo(self, controller_json_path)

    def configure_dense_reward(self, use_dense_reward=True, normalize_rewards=False, no_of_subtasks=3, subtask_dag=None):
        """Configure persistent dense reward settings for Q-IPM."""
        self._dense_reward_enabled = use_dense_reward
        self._dense_normalize_rewards = normalize_rewards
        self._dense_no_of_subtasks = no_of_subtasks

    def configure_safety_status(self, no_of_safety_subtasks, safety_head_mode='single',
                                safety_signal='cost'):
        """Enable appending per-step safety columns to subtask_status (safe_qipm family).

        When enabled, step() appends instantaneous safety columns to
        reward_info['subtask_status'], after the ratcheting task columns — one
        per always-active safety Q-head. Off by default: every other algo sees
        the task-only vector unchanged.

        Two orthogonal switches. safety_head_mode decides HOW MANY columns:
          - "single": one column, the aggregate "was this step unsafe" bit.
            Requires no_of_safety_subtasks == 1.
          - "multi": K columns, the per-constraint decomposition parallel to
            info['safety_constraints'].

        safety_signal decides WHAT THEY CARRY (see SAFETY_SIGNALS):
          - "cost": 0/1 violation flags (info['unsafe_step'] /
            info['safety_violations']) — safe_qipm's CMDP cost heads.
          - "margin": signed distances (info['safety_margins']) — safe_qipm_hjb's
            HJB reachability heads. Multi mode only.

        The multi roster is resolved HERE, at configure time, and asserted to
        have exactly no_of_safety_subtasks entries. That is deliberate: the
        buffer width and the critic head count are both fixed before the first
        ONLINE step, which only happens after the offline BC phase — validating
        against the first info dict instead would surface a misconfiguration
        hours into a run, after the buffer and critic were already sized.
        """
        assert safety_head_mode in SAFETY_HEAD_MODES, (
            f"safety_head_mode must be one of {SAFETY_HEAD_MODES}, got {safety_head_mode!r}"
        )
        assert safety_signal in SAFETY_SIGNALS, (
            f"safety_signal must be one of {SAFETY_SIGNALS}, got {safety_signal!r}"
        )
        assert not (safety_signal == 'margin' and safety_head_mode == 'single'), (
            "safety_signal='margin' requires safety_head_mode='multi': the "
            "single-mode aggregate has no margin analogue (it would need a "
            "min over the per-constraint margins, and single mode resolves no "
            "roster to take that min over)"
        )
        no_of_safety_subtasks = int(no_of_safety_subtasks)
        if safety_head_mode == 'single':
            assert no_of_safety_subtasks == 1, (
                "safety_head_mode='single' bridges the one aggregate unsafe_step "
                "flag; use safety_head_mode='multi' for per-constraint columns"
            )
            # Same probe the multi branch asserts on, for the opposite reason:
            # single mode never USES the names, but info['unsafe_step'] is
            # published unconditionally by safe-robosuite and defaults to False
            # when no controller carries a safety filter. Without this the
            # always-active safety head trains on a constant-zero cost column
            # for the whole run, and flags.json is byte-identical to a correct
            # run — the failure mode that motivated the 2026-08-24 provenance
            # work. Cheap: the roster is resolved from the controllers, before
            # the first step.
            assert _resolve_safety_roster(self.env.env), (
                "safety_head_mode='single' but no controller publishes safety "
                "constraints — the composite controller has no safety filter, so "
                "info['unsafe_step'] would be False on every step and the safety "
                "head would train on an all-zero cost column "
                "(check ROBOSUITE_CONTROLLER_CONFIG)"
            )
            roster = []  # names are not needed; the single column is unnamed
        else:
            roster = _resolve_safety_roster(self.env.env)
            assert roster, (
                "safety_head_mode='multi' but no controller publishes safety "
                "constraints — the composite controller has no safety filter "
                "(check ROBOSUITE_CONTROLLER_CONFIG)"
            )
            assert len(roster) == no_of_safety_subtasks, (
                f"no_of_safety_subtasks={no_of_safety_subtasks} but the env "
                f"publishes {len(roster)} safety constraints: {roster}"
            )
        self._safety_status_dims = no_of_safety_subtasks
        self._safety_head_mode = safety_head_mode
        self._safety_signal = safety_signal
        self._safety_constraint_names = roster

    def step(self, action, use_dense_reward=False, normalize_rewards=False, no_of_subtasks=3):
        # Use persistent config if set, otherwise use kwargs
        use_dense_reward = getattr(self, '_dense_reward_enabled', False) or use_dense_reward
        normalize_rewards = getattr(self, '_dense_normalize_rewards', False) or normalize_rewards
        no_of_subtasks = getattr(self, '_dense_no_of_subtasks', None) or self.no_of_subtasks or no_of_subtasks
        assert self.no_of_subtasks == no_of_subtasks or self.no_of_subtasks == 0

        if self.normalize:
            action = self.unnormalize_action(action)
        _sidecar = getattr(self, '_safety_sidecar', None)
        if _sidecar is not None:
            _sidecar.before_step()
        _shield_diag = getattr(self, '_shield_diagnostics', None)
        if _shield_diag is not None:
            _shield_diag.before_step(self.env.env)
        raw_obs, reward, done, info = self.env.step(action)
        if _sidecar is not None:
            _sidecar.after_step(info)
        if _shield_diag is not None:
            _shield_diag.after_step(info)
        # Transform reward from [0, 1] to [-1, 0]
        reward = reward - 1.0

        # Compute dense reward / update goal state for GC
        is_goal_conditioned = self.goal_conditioning_mode is not None
        if use_dense_reward or is_goal_conditioned:
            previous_goal_dones = self.goal_done
            dense_reward = self.__dense_reward(normalize_rewards, no_of_subtasks)
            if use_dense_reward:
                reward = dense_reward
            elif previous_goal_dones is not None and previous_goal_dones != self.goal_done: # Provide singular reward bump upon subtask completion.
                assert all(self.goal_done[i] for i in range(no_of_subtasks) if previous_goal_dones[i]), \
                    f"Subtask completion is not monotonic: previous={previous_goal_dones}, current={self.goal_done}"
                reward += 5.0
            info['reward_info'] = self._last_reward_info
            info['reward_info']['normalized_reward'] = reward if normalize_rewards else reward / no_of_subtasks
            info['reward_info']['raw_reward'] = reward

        if getattr(self, '_safety_status_dims', 0) > 0:
            # safe_qipm family: append the per-step safety signal as trailing
            # column(s) (instantaneous, NOT ratcheting). reward_info must exist
            # (dense reward configured) and the env must publish the configured
            # source key — missing either while safety is enabled is a
            # misconfiguration and raises.
            # getattr default: mirrors the _safety_head_mode pattern above —
            # 'cost' is the pre-2026-07-26 behaviour, so a wrapper configured by
            # older code (or unpickled from before the kwarg existed) keeps it.
            _signal = getattr(self, '_safety_signal', 'cost')
            _columns_fn = (
                _safety_margin_columns if _signal == 'margin' else _safety_cost_columns
            )
            reward_info = info['reward_info']
            reward_info['subtask_status'] = (
                list(reward_info['subtask_status'])
                + _columns_fn(
                    info,
                    getattr(self, '_safety_head_mode', 'single'),
                    self._safety_status_dims,
                    getattr(self, '_safety_constraint_names', []),
                )
            )

        obs = self._build_observation(raw_obs)

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        # Check for success
        check_success_result = self.env.env._check_success()
        info["success"] = int(check_success_result)

        if info["success"]:
            done = True

        # Always include return and length when episode ends
        terminated = done
        truncated = self.t >= self.max_episode_length
        if terminated or truncated:
            info['return'] = self.episode_return
            info['length'] = self.episode_length

        if terminated:
            return obs, reward, True, False, info
        if truncated:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def __dense_reward(self, normalize_rewards=False, no_of_subtasks=5):
        """Compute dense per-subtask reward using shared subtask detection."""
        self.goal_done, self._last_reward_info = _compute_subtask_status(
            base_env=self.env.env,
            goal_done=self.goal_done,
            no_of_subtasks=no_of_subtasks,
        )
        reward = -sum(1 for i in range(no_of_subtasks) if not self.goal_done[i])
        if normalize_rewards:
            reward /= no_of_subtasks
        return reward

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )

    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}
    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}



class RobomimicImageWrapper(gym.Env):
    """Robomimic image environment wrapper.

    Returns dict observations: {'state': proprio_vec, 'image': image_array}
    so that the pipeline can independently route state and image to actor/critic.
    """
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        image_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        proprio_keys=(
            'robot0_eef_pos',
            'robot0_eef_quat',
            'robot0_gripper_qpos',
        ),
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=None,
        num_stack=1,
    ):
        self.env = env
        self.init_state = init_state
        self.has_reset_before = False
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.env_step = 0
        self.n_episodes = 0
        self.t = 0
        self.num_stack = num_stack
        self.frame_stack = deque(maxlen=num_stack)
        self.proprio_keys = proprio_keys

        # set up normalization for actions only
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]

        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        self.image_keys = image_keys

        # Image observation space
        rgb_shape = shape_meta["obs"]["rgb"]["shape"]  # (3, 96, 96) as (C, H, W)
        channels, height, width = rgb_shape
        num_cameras = len(image_keys)
        full_image_shape = (height, width, channels * num_cameras * num_stack)

        # Proprioception dimension: get from a dummy obs
        raw_obs = self.env.get_observation()
        proprio_dim = sum(raw_obs[k].shape[0] for k in self.proprio_keys if k in raw_obs)

        self.observation_space = Dict({
            'state': Box(low=-np.inf, high=np.inf, shape=(proprio_dim,), dtype=np.float32),
            'image': Box(low=0, high=255, shape=full_image_shape, dtype=np.float32),
        })
        self.goal_done = None
        self._last_reward_info = {}

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def get_single_observation(self, raw_obs):
        """Extract image and proprioception from raw robomimic observation."""
        img_parts = []
        for key in self.image_keys:
            if key in raw_obs:
                img_parts.append(raw_obs[key].astype(np.float32))  # (H, W, C)
        image = np.concatenate(img_parts, axis=2) if len(img_parts) > 1 else img_parts[0]
        state_parts = [raw_obs[k].astype(np.float32) for k in self.proprio_keys if k in raw_obs]
        state = np.concatenate(state_parts, axis=0)
        return {'state': state, 'image': image}

    def _get_stacked_observation(self):
        assert len(self.frame_stack) == self.num_stack
        frames = list(self.frame_stack)
        stacked_image = np.concatenate([f['image'] for f in frames], axis=-1)
        current_state = frames[-1]['state']  # latest proprioception
        return {'state': current_state, 'image': stacked_image}

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, seed=None, options=None, **kwargs):
        """Reset environment with optional seed for reproducible diverse resets."""
        if options is None:
            options = {}

        self.t = 0
        self.episode_return, self.episode_length = 0, 0
        self.goal_done = None
        self._last_reward_info = {}
        self.n_episodes += 1

        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Prefer explicit seed kwarg (gymnasium standard), fall back to options
        new_seed = seed if seed is not None else options.get("seed", None)
        if self.init_state is not None:
            if not self.has_reset_before:
                self.env.reset()
                self.has_reset_before = True
            raw_obs = self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs = self.env.reset()
        else:
            raw_obs = self.env.reset()

        # Initialize frame stack
        current_obs = self.get_single_observation(raw_obs)
        for _ in range(self.num_stack):
            self.frame_stack.append(current_obs)

        return self._get_stacked_observation(), {}

    def configure_dense_reward(self, use_dense_reward=True, normalize_rewards=False, no_of_subtasks=3, subtask_dag=None):
        """Configure persistent dense reward settings for Q-IPM."""
        self._dense_reward_enabled = use_dense_reward
        self._dense_normalize_rewards = normalize_rewards
        self._dense_no_of_subtasks = no_of_subtasks

    def configure_safety_status(self, no_of_safety_subtasks, safety_head_mode='single',
                                safety_signal='cost'):
        """Enable appending per-step safety columns to subtask_status (safe_qipm family).

        When enabled, step() appends instantaneous safety columns to
        reward_info['subtask_status'], after the ratcheting task columns — one
        per always-active safety Q-head. Off by default: every other algo sees
        the task-only vector unchanged.

        Two orthogonal switches. safety_head_mode decides HOW MANY columns:
          - "single": one column, the aggregate "was this step unsafe" bit.
            Requires no_of_safety_subtasks == 1.
          - "multi": K columns, the per-constraint decomposition parallel to
            info['safety_constraints'].

        safety_signal decides WHAT THEY CARRY (see SAFETY_SIGNALS):
          - "cost": 0/1 violation flags (info['unsafe_step'] /
            info['safety_violations']) — safe_qipm's CMDP cost heads.
          - "margin": signed distances (info['safety_margins']) — safe_qipm_hjb's
            HJB reachability heads. Multi mode only.

        The multi roster is resolved HERE, at configure time, and asserted to
        have exactly no_of_safety_subtasks entries. That is deliberate: the
        buffer width and the critic head count are both fixed before the first
        ONLINE step, which only happens after the offline BC phase — validating
        against the first info dict instead would surface a misconfiguration
        hours into a run, after the buffer and critic were already sized.
        """
        assert safety_head_mode in SAFETY_HEAD_MODES, (
            f"safety_head_mode must be one of {SAFETY_HEAD_MODES}, got {safety_head_mode!r}"
        )
        assert safety_signal in SAFETY_SIGNALS, (
            f"safety_signal must be one of {SAFETY_SIGNALS}, got {safety_signal!r}"
        )
        assert not (safety_signal == 'margin' and safety_head_mode == 'single'), (
            "safety_signal='margin' requires safety_head_mode='multi': the "
            "single-mode aggregate has no margin analogue (it would need a "
            "min over the per-constraint margins, and single mode resolves no "
            "roster to take that min over)"
        )
        no_of_safety_subtasks = int(no_of_safety_subtasks)
        if safety_head_mode == 'single':
            assert no_of_safety_subtasks == 1, (
                "safety_head_mode='single' bridges the one aggregate unsafe_step "
                "flag; use safety_head_mode='multi' for per-constraint columns"
            )
            # Same probe the multi branch asserts on, for the opposite reason:
            # single mode never USES the names, but info['unsafe_step'] is
            # published unconditionally by safe-robosuite and defaults to False
            # when no controller carries a safety filter. Without this the
            # always-active safety head trains on a constant-zero cost column
            # for the whole run, and flags.json is byte-identical to a correct
            # run — the failure mode that motivated the 2026-08-24 provenance
            # work. Cheap: the roster is resolved from the controllers, before
            # the first step.
            assert _resolve_safety_roster(self.env.env), (
                "safety_head_mode='single' but no controller publishes safety "
                "constraints — the composite controller has no safety filter, so "
                "info['unsafe_step'] would be False on every step and the safety "
                "head would train on an all-zero cost column "
                "(check ROBOSUITE_CONTROLLER_CONFIG)"
            )
            roster = []  # names are not needed; the single column is unnamed
        else:
            roster = _resolve_safety_roster(self.env.env)
            assert roster, (
                "safety_head_mode='multi' but no controller publishes safety "
                "constraints — the composite controller has no safety filter "
                "(check ROBOSUITE_CONTROLLER_CONFIG)"
            )
            assert len(roster) == no_of_safety_subtasks, (
                f"no_of_safety_subtasks={no_of_safety_subtasks} but the env "
                f"publishes {len(roster)} safety constraints: {roster}"
            )
        self._safety_status_dims = no_of_safety_subtasks
        self._safety_head_mode = safety_head_mode
        self._safety_signal = safety_signal
        self._safety_constraint_names = roster

    def step(self, action, use_dense_reward=False, normalize_rewards=False, no_of_subtasks=3):
        # Use persistent config if set, otherwise use kwargs
        use_dense_reward = getattr(self, '_dense_reward_enabled', False) or use_dense_reward
        normalize_rewards = getattr(self, '_dense_normalize_rewards', False) or normalize_rewards
        no_of_subtasks = getattr(self, '_dense_no_of_subtasks', None) or no_of_subtasks

        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(action)
        # Transform reward from [0, 1] to [-1, 0]
        reward = reward - 1.0

        if use_dense_reward:
            reward = self.__dense_reward(normalize_rewards, no_of_subtasks)
            info['reward_info'] = self._last_reward_info
            info['reward_info']['normalized_reward'] = reward if normalize_rewards else reward / no_of_subtasks
            info['reward_info']['raw_reward'] = reward

        if getattr(self, '_safety_status_dims', 0) > 0:
            # safe_qipm family: append the per-step safety signal as trailing
            # column(s) (instantaneous, NOT ratcheting). reward_info must exist
            # (dense reward configured) and the env must publish the configured
            # source key — missing either while safety is enabled is a
            # misconfiguration and raises.
            # getattr default: mirrors the _safety_head_mode pattern above —
            # 'cost' is the pre-2026-07-26 behaviour, so a wrapper configured by
            # older code (or unpickled from before the kwarg existed) keeps it.
            _signal = getattr(self, '_safety_signal', 'cost')
            _columns_fn = (
                _safety_margin_columns if _signal == 'margin' else _safety_cost_columns
            )
            reward_info = info['reward_info']
            reward_info['subtask_status'] = (
                list(reward_info['subtask_status'])
                + _columns_fn(
                    info,
                    getattr(self, '_safety_head_mode', 'single'),
                    self._safety_status_dims,
                    getattr(self, '_safety_constraint_names', []),
                )
            )

        # Update stack
        current_obs = self.get_single_observation(raw_obs)
        self.frame_stack.append(current_obs)
        obs = self._get_stacked_observation()

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        # Check for success
        check_success_result = self.env.env._check_success()
        info["success"] = int(check_success_result)

        if info["success"]:
            done = True

        # Always include return and length when episode ends
        terminated = done
        truncated = self.t >= self.max_episode_length
        if terminated or truncated:
            info['return'] = self.episode_return
            info['length'] = self.episode_length

        if terminated:
            return obs, reward, True, False, info
        if truncated:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def __dense_reward(self, normalize_rewards=False, no_of_subtasks=5):
        """Compute dense per-subtask reward using shared subtask detection."""
        self.goal_done, self._last_reward_info = _compute_subtask_status(
            base_env=self.env.env,
            goal_done=self.goal_done,
            no_of_subtasks=no_of_subtasks,
        )
        reward = -sum(1 for i in range(no_of_subtasks) if not self.goal_done[i])
        if normalize_rewards:
            reward /= no_of_subtasks
        return reward

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode,
            height=h,
            width=w,
            camera_name=self.render_camera_name,
        )

    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}

    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}

if __name__ == "__main__":
    # for testing
    import sys
    env_name = sys.argv[1] if len(sys.argv) > 1 else "square-mh-image"
    print(f"Testing: {env_name}")
    task_cfg = _get_task_config(env_name)
    print(f"Task config: {task_cfg}")
    env = make_env(env_name)
    dataset = get_dataset(env, env_name)
    print(f"Observations shape: {dataset['observations'].shape}")
    print(f"Actions shape: {dataset['actions'].shape}")
    if 'images' in dataset._dict:
        print(f"Images shape: {dataset['images'].shape}")
    print("OK")
