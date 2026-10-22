"""
RoboCasa environment utilities for OGPO training.

Supports multiple RoboCasa tasks:
  - LoadDishwasher (Layout 5 / Style 5)
  - PreSoakPan (Layout 2 / Style 2)
  - ArrangeTea   (Layout 5, kettle+mug+tray+HingeCabinet)

Requires robocasa >= 1.0 (robocasa365) to be installed.

Anchoring: locks env resets to a specific demo's layout/style via HDF5 state
restoration, ensuring reproducible fixture placement across all resets.
"""

import os
import json
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import h5py
import imageio
import robosuite
import robocasa  # registers robocasa envs with robosuite

from ogpo.utils.datasets import Dataset
from envs.ogbench_dense_wrapper import GOAL_CONDITIONING_MODES, _encode_goal_vector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_OBS_KEY = 'observation_state'
DATASET_OBS_DIM = 31  # default for LoadDishwasher / PreSoakPan; override per task below

# Per-task obs dim (wrapper output dim after concatenation of raw keys + extra_obs_fn).
# Falls back to DATASET_OBS_DIM if not listed.
_TASK_OBS_DIMS = {
    'LoadDishwasher': 31,
    'PreSoakPan': 31,
    'ArrangeTea': 39,
}

MAX_EPISODE_LENGTH = 800  # default for LoadDishwasher / PreSoakPan

# Per-task episode length cap. Falls back to MAX_EPISODE_LENGTH if not listed.
_TASK_MAX_EP_LEN = {
    'LoadDishwasher': 800,
    'PreSoakPan': 800,
    'ArrangeTea': 1200,
}


# ---------------------------------------------------------------------------
# Env identification helpers
# ---------------------------------------------------------------------------

def is_robocasa_env(env_name):
    """Check if env_name refers to a RoboCasa environment."""
    return env_name.startswith('robocasa_')


def _get_task_name(env_name):
    """Extract task name from 'robocasa_TaskName' format."""
    return env_name.replace('robocasa_', '', 1)


_TASK_DATASET_DEFAULTS = {
    'LoadDishwasher': ('LoadDishWasher', 'LoadDishwasher_noisy_replay_ready_31dim_v2.hdf5'),
    'PreSoakPan': ('PreSoakPan', 'PreSoakPan_noisy_replay_ready_31dim_v2.hdf5'),
    'ArrangeTea': ('ArrangeTea', 'ArrangeTea_layout5_human_39dim.hdf5'),
}


def _get_dataset_path(env_name):
    """Resolve the HDF5 dataset path for a RoboCasa env."""
    path = os.environ.get('ROBOCASA_DATASET_PATH', None)
    if path and os.path.isfile(path):
        return path

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_name = _get_task_name(env_name)
    if task_name in _TASK_DATASET_DEFAULTS:
        subdir, default_file = _TASK_DATASET_DEFAULTS[task_name]
        default_path = os.path.join(project_root, 'data', 'robocasa', subdir, default_file)
        if os.path.isfile(default_path):
            return default_path

    raise FileNotFoundError(
        f"Cannot find RoboCasa dataset for {task_name}. "
        f"Set ROBOCASA_DATASET_PATH or place dataset in "
        f"data/robocasa/<task_dir>/",
        f"Provided: {path}"
    )


def _get_env_meta(dataset_path):
    """Read env metadata directly from HDF5 (no robomimic dependency)."""
    with h5py.File(dataset_path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def _detect_obs_keys(env, target_dim=DATASET_OBS_DIM, extra_obs_dim=0):
    """
    Auto-detect which obs keys from the raw env observation dict produce a
    concatenated vector of dimension (target_dim - extra_obs_dim). The remaining
    `extra_obs_dim` dims are provided by the wrapper's extra_obs_fn.

    Returns the list of keys to concatenate.
    """
    raw_obs = env._get_observations()
    raw_target = target_dim - extra_obs_dim
    print(f"[RoboCasa] Available obs keys and shapes:")
    for k in sorted(raw_obs.keys()):
        v = raw_obs[k]
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape}")

    # Strategy 1: if observation_state is directly available, use it
    if DATASET_OBS_KEY in raw_obs:
        dim = raw_obs[DATASET_OBS_KEY].shape[0] if raw_obs[DATASET_OBS_KEY].ndim > 0 else 1
        if dim == target_dim:
            print(f"[RoboCasa] Using '{DATASET_OBS_KEY}' directly ({dim}D)")
            return (DATASET_OBS_KEY,)

    # Strategy 2: known key combinations (raw robosuite obs keys)
    candidate_sets = [
        # LoadDishwasher: proprio(16) + dish0(7) + dish1(7) = 30
        # (31st dim = dishwasher door comes from extra_obs_fn)
        ('robot0_base_pos', 'robot0_base_quat', 'robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_gripper_qpos',
         'dish0_pos', 'dish0_quat', 'dish1_pos', 'dish1_quat'),
        # PreSoakPan: proprio(16) + obj1(7) + obj2(7) = 30
        # (31st dim = water_on comes from extra_obs_fn)
        ('robot0_base_pos', 'robot0_base_quat', 'robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_gripper_qpos',
         'obj1_pos', 'obj1_quat', 'obj2_pos', 'obj2_quat'),
        # ArrangeTea: proprio(16) + mug(7) + kettle(7) + tray(7) = 37
        # (last 2 dims = cab hinge L/R come from extra_obs_fn)
        ('robot0_base_pos', 'robot0_base_quat', 'robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_gripper_qpos',
         'obj_pos', 'obj_quat', 'obj2_pos', 'obj2_quat', 'container_pos', 'container_quat'),
        # Robot proprioception only: 3+4+3+4+2 = 16
        ('robot0_base_pos', 'robot0_base_quat', 'robot0_base_to_eef_pos', 'robot0_base_to_eef_quat', 'robot0_gripper_qpos'),
    ]

    for keys in candidate_sets:
        if all(k in raw_obs for k in keys):
            dim = sum(raw_obs[k].shape[0] if raw_obs[k].ndim > 0 else 1 for k in keys)
            if dim == raw_target:
                print(f"[RoboCasa] Matched obs keys: {keys} -> {dim}D (+{extra_obs_dim} extra = {target_dim})")
                return keys

    # Strategy 3: brute-force search over all 1D-array keys
    scalar_keys = []
    for k in sorted(raw_obs.keys()):
        v = raw_obs[k]
        if hasattr(v, 'shape') and v.ndim <= 1 and v.size < 20:
            scalar_keys.append(k)

    robot_keys = [k for k in scalar_keys if k.startswith('robot0')]
    obj_keys = [k for k in scalar_keys if not k.startswith('robot0') and not k.startswith('world')]

    robot_dim = sum(raw_obs[k].shape[0] if raw_obs[k].ndim > 0 else 1 for k in robot_keys)
    needed_obj_dim = raw_target - robot_dim

    if needed_obj_dim > 0:
        selected = list(robot_keys)
        current_dim = robot_dim
        for k in obj_keys:
            d = raw_obs[k].shape[0] if raw_obs[k].ndim > 0 else 1
            if current_dim + d <= raw_target:
                selected.append(k)
                current_dim += d
                if current_dim == raw_target:
                    print(f"[RoboCasa] Auto-detected obs keys: {selected} -> {current_dim}D (+{extra_obs_dim} extra = {target_dim})")
                    return tuple(selected)

    raise ValueError(
        f"Could not find obs keys that produce {raw_target}D observation "
        f"(target_dim={target_dim}, extra_obs_dim={extra_obs_dim}). "
        f"Available keys: {list(raw_obs.keys())}. "
        f"Set ROBOCASA_OBS_KEYS env var to comma-separated key list."
    )


# ---------------------------------------------------------------------------
# LoadDishwasher subtask detection
# ---------------------------------------------------------------------------

def check_cup_grasped(base_env):
    """Check if the gripper is grasping dish0 (the cup)."""
    import robocasa.utils.object_utils as OU
    return OU.check_obj_grasped(base_env, "dish0")


def check_cup_in_dw(base_env):
    """Check if the cup is in contact with the dishwasher rack."""
    return base_env.dishwasher.check_rack_contact(base_env, "dish0")


def check_bowl_grasped(base_env):
    """Check if the gripper is grasping dish1 (the bowl)."""
    import robocasa.utils.object_utils as OU
    return OU.check_obj_grasped(base_env, "dish1")


def check_bowl_in_dw(base_env):
    """Check if the bowl is in contact with the dishwasher rack."""
    return base_env.dishwasher.check_rack_contact(base_env, "dish1")


def check_dw_closed(base_env):
    """Check if the dishwasher door is fully closed."""
    return (check_cup_in_dw(base_env)
            and check_bowl_in_dw(base_env)
            and base_env.dishwasher.is_closed(base_env, th=0.05))


SUBGOAL_NAMES = [
    "cup_grasped",
    "cup_in_dw",
    "bowl_grasped",
    "bowl_in_dw",
    "dw_closed",
]

SUBGOAL_CHECKS = [
    check_cup_grasped,
    check_cup_in_dw,
    check_bowl_grasped,
    check_bowl_in_dw,
    check_dw_closed,
]


# ---------------------------------------------------------------------------
# PreSoakPan subtask detection
# ---------------------------------------------------------------------------

def check_pan_grasped_lifted(base_env):
    """Check if the gripper is grasping the pan (obj1) and has lifted it."""
    import robocasa.utils.object_utils as OU
    if not OU.check_obj_grasped(base_env, "obj1"):
        return False
    obj_pos = base_env.sim.data.body_xpos[base_env.obj_body_id["obj1"]]
    return obj_pos[2] > 0.88


def check_pan_in_sink(base_env):
    """Check if the pan is inside the sink basin."""
    import robocasa.utils.object_utils as OU
    return OU.obj_inside_of(base_env, "obj1", base_env.sink, partial_check=False)


def check_sponge_grasped_lifted(base_env):
    """Check if the gripper is grasping the sponge (obj2) and has lifted it."""
    import robocasa.utils.object_utils as OU
    if not OU.check_obj_grasped(base_env, "obj2"):
        return False
    obj_pos = base_env.sim.data.body_xpos[base_env.obj_body_id["obj2"]]
    return obj_pos[2] > 0.88


def check_sponge_in_pan_and_sink(base_env):
    """Check if the sponge is inside the sink and on/in the pan."""
    import robocasa.utils.object_utils as OU
    if not OU.obj_inside_of(base_env, "obj2", base_env.sink, partial_check=False):
        return False
    pan_pos = base_env.sim.data.body_xpos[base_env.obj_body_id["obj1"]]
    sponge_pos = base_env.sim.data.body_xpos[base_env.obj_body_id["obj2"]]
    xy_dist = np.linalg.norm(sponge_pos[:2] - pan_pos[:2])
    z_diff = sponge_pos[2] - pan_pos[2]
    return xy_dist < 0.25 and z_diff > -0.05 and z_diff < 0.20


def check_water_on(base_env):
    """Check if water is on AND all prerequisites complete AND gripper far from objects.

    Matches env's _check_success() exactly so subtask_4 == env success.
    """
    import robocasa.utils.object_utils as OU
    handle_state = base_env.sink.get_handle_state(base_env)
    water_on = handle_state["water_on"]
    pan_in_sink = OU.obj_inside_of(base_env, "obj1", base_env.sink, partial_check=False)
    sponge_in_sink = OU.obj_inside_of(base_env, "obj2", base_env.sink, partial_check=False)
    gripper_far_pan = OU.gripper_obj_far(base_env, "obj1")
    gripper_far_sponge = OU.gripper_obj_far(base_env, "obj2")
    return (water_on and pan_in_sink and sponge_in_sink
            and gripper_far_pan and gripper_far_sponge)


PRESOAKPAN_SUBGOAL_NAMES = [
    "pan_grasped_lifted",
    "pan_in_sink",
    "sponge_grasped_lifted",
    "sponge_in_pan_and_sink",
    "water_on",
]

PRESOAKPAN_SUBGOAL_CHECKS = [
    check_pan_grasped_lifted,
    check_pan_in_sink,
    check_sponge_grasped_lifted,
    check_sponge_in_pan_and_sink,
    check_water_on,
]


# ---------------------------------------------------------------------------
# ArrangeTea subtask detection
#
# Naming convention from the ArrangeTea env:
#   obj       -> mug
#   obj2      -> kettle
#   container -> tray
#   env.cab   -> target HingeCabinet (pre-opened at scene setup)
#
# Strict linear chain (NOT a DAG):
#   kettle_grasped -> kettle_on_tray -> mug_grasped -> mug_on_tray -> cab_closed
# ---------------------------------------------------------------------------

def check_kettle_grasped(base_env):
    import robocasa.utils.object_utils as OU
    return OU.check_obj_grasped(base_env, "obj2")


def check_kettle_on_tray(base_env):
    import robocasa.utils.object_utils as OU
    return OU.check_obj_in_receptacle(base_env, "obj2", "container")


def check_mug_grasped(base_env):
    import robocasa.utils.object_utils as OU
    return OU.check_obj_grasped(base_env, "obj")


def check_mug_on_tray(base_env):
    import robocasa.utils.object_utils as OU
    return OU.check_obj_in_receptacle(base_env, "obj", "container")


def check_right_door_closed(base_env):
    cab = base_env.cab
    return cab.is_closed(env=base_env,
                         joint_names=[f"{cab.name}_rightdoorhinge"])


def check_cab_closed(base_env):
    return base_env.cab.is_closed(env=base_env)


# Compound subgoal checks: each instantaneous, requires prior placements to hold.
def check_kettle_on_tray_and_mug_grasped(base_env):
    return check_kettle_on_tray(base_env) and check_mug_grasped(base_env)


def check_kettle_on_tray_and_mug_on_tray(base_env):
    return check_kettle_on_tray(base_env) and check_mug_on_tray(base_env)


def check_both_on_tray_and_right_door_closed(base_env):
    return (check_kettle_on_tray(base_env)
            and check_mug_on_tray(base_env)
            and check_right_door_closed(base_env))


def check_both_on_tray_and_cab_closed(base_env):
    return (check_kettle_on_tray(base_env)
            and check_mug_on_tray(base_env)
            and check_cab_closed(base_env))


ARRANGETEA_SUBGOAL_NAMES = [
    "kettle_grasped",
    "kettle_on_tray",
    "kettle_on_tray+mug_grasped",
    "kettle_on_tray+mug_on_tray",
    "both_on_tray+right_door_closed",
    "both_on_tray+cab_closed",
]

ARRANGETEA_SUBGOAL_CHECKS = [
    check_kettle_grasped,
    check_kettle_on_tray,
    check_kettle_on_tray_and_mug_grasped,
    check_kettle_on_tray_and_mug_on_tray,
    check_both_on_tray_and_right_door_closed,
    check_both_on_tray_and_cab_closed,
]


# ---------------------------------------------------------------------------
# Task subtask registry
# ---------------------------------------------------------------------------

TASK_SUBTASK_CONFIGS = {
    'LoadDishwasher': {
        'names': SUBGOAL_NAMES,
        'checks': SUBGOAL_CHECKS,
    },
    'PreSoakPan': {
        'names': PRESOAKPAN_SUBGOAL_NAMES,
        'checks': PRESOAKPAN_SUBGOAL_CHECKS,
    },
    'ArrangeTea': {
        'names': ARRANGETEA_SUBGOAL_NAMES,
        'checks': ARRANGETEA_SUBGOAL_CHECKS,
    },
}


def get_subtask_config(env_name):
    """Get subtask config for the given env_name."""
    task_name = _get_task_name(env_name)
    if task_name not in TASK_SUBTASK_CONFIGS:
        raise ValueError(
            f"Unknown RoboCasa task: {task_name}. "
            f"Supported: {list(TASK_SUBTASK_CONFIGS.keys())}"
        )
    return TASK_SUBTASK_CONFIGS[task_name]


# ---------------------------------------------------------------------------
# Extra observation extractors (task-specific)
# ---------------------------------------------------------------------------

def _get_dishwasher_door_obs(env):
    """Extract dishwasher door state as a 1-D array."""
    return np.array([env.dishwasher.get_state(env)['door']], dtype=np.float32)


def _get_sink_water_obs(env):
    """Extract sink water-on state as a 1-D array."""
    return np.array([float(env.sink.get_handle_state(env)['water_on'])], dtype=np.float32)


def _get_arrangetea_hinges_obs(env):
    """Extract [left_hinge_qpos, right_hinge_qpos] for the target cabinet.

    Mirrors the convention used by the HDF5's 39-dim obs (dims [37], [38]).
    Falls back to a cached cab_name if resolution from ep_meta fails.
    """
    cab = getattr(env, 'cab', None)
    cab_name = None
    if cab is not None:
        cab_name = getattr(cab, 'naming_prefix', None)
        if cab_name is not None:
            cab_name = cab_name.rstrip('_')
    if cab_name is None:
        fixture_refs = getattr(env, 'fixture_refs', None) or {}
        cab_name = fixture_refs.get('cab', None)
    if cab_name is None:
        raise RuntimeError("ArrangeTea: cannot resolve cabinet name for hinge obs")

    l = env.sim.data.joint(f"{cab_name}_leftdoorhinge").qpos[0]
    r = env.sim.data.joint(f"{cab_name}_rightdoorhinge").qpos[0]
    return np.array([l, r], dtype=np.float32)


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def make_env(env_name, seed=0, enable_render=False, anchor_demo=None,
             goal_conditioning_mode=None, no_of_subtasks=None):
    """
    Create a RoboCasa environment from the HDF5 metadata.
    Uses robosuite.make() directly (no robomimic dependency).

    Args:
        env_name: RoboCasa env name (e.g. 'robocasa_LoadDishwasher').
        seed: Random seed.
        enable_render: If True, initialize offscreen renderer (for video).
        anchor_demo: If set (e.g. "demo_0"), lock env resets to this demo's
                     initial state (layout/style anchoring).
    """
    dataset_path = _get_dataset_path(env_name)
    env_meta = _get_env_meta(dataset_path)

    env_kwargs = dict(env_meta.get("env_kwargs", {}))
    env_kwargs.pop("env_name", None)
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = enable_render
    env_kwargs["use_camera_obs"] = False

    # Opt-in: when collecting BC demos via the GR00T inference server, render
    # the 3 cameras the VLA needs. Zero effect on training (flag is off by
    # default).
    if os.environ.get("ROBOCASA_VLA_CAMERAS") == "1":
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = True
        env_kwargs["camera_names"] = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]
        env_kwargs["camera_heights"] = 256
        env_kwargs["camera_widths"] = 256

    raw_env = robocasa.make(env_meta["env_name"], **env_kwargs)

    # If anchoring, set anchor's ep_meta BEFORE the first reset so _load_model
    # uses the anchor's exact object_cfgs (specific mug/kettle/tray meshes).
    # Otherwise the initial sim is built with randomly-sampled objects and the
    # saved state vector won't align with the sim's DOF layout.
    _preload_ep_meta = None
    if anchor_demo is not None:
        with h5py.File(dataset_path, "r") as f:
            em = f[f"data/{anchor_demo}"].attrs.get("ep_meta", None)
            if em is not None:
                _preload_ep_meta = json.loads(em) if isinstance(em, (str, bytes)) else em
        if _preload_ep_meta is not None and hasattr(raw_env, "set_ep_meta"):
            raw_env.set_ep_meta(_preload_ep_meta)

    raw_env.reset()

    task_name = _get_task_name(env_name)
    target_obs_dim = _TASK_OBS_DIMS.get(task_name, DATASET_OBS_DIM)
    max_ep_len = _TASK_MAX_EP_LEN.get(task_name, MAX_EPISODE_LENGTH)

    # Task-specific extra-obs extractors (synthesized dims appended after raw keys)
    extra_obs_fn = None
    extra_obs_dim = 0
    if not os.environ.get('ROBOCASA_SKIP_EXTRA_OBS'):
        if task_name == 'LoadDishwasher':
            extra_obs_fn = _get_dishwasher_door_obs
            extra_obs_dim = 1
        elif task_name == 'PreSoakPan':
            extra_obs_fn = _get_sink_water_obs
            extra_obs_dim = 1
        elif task_name == 'ArrangeTea':
            extra_obs_fn = _get_arrangetea_hinges_obs
            extra_obs_dim = 2

    # Auto-detect obs keys or use override
    obs_keys_override = os.environ.get('ROBOCASA_OBS_KEYS', None)
    if obs_keys_override:
        detected_keys = tuple(obs_keys_override.split(','))
        print(f"[RoboCasa] Using ROBOCASA_OBS_KEYS override: {detected_keys}")
    else:
        detected_keys = _detect_obs_keys(raw_env, target_dim=target_obs_dim, extra_obs_dim=extra_obs_dim)

    # Load anchor state from HDF5 if requested
    anchor_state_dict = None
    if anchor_demo is not None:
        with h5py.File(dataset_path, "r") as f:
            grp = f[f"data/{anchor_demo}"]
            anchor_state_dict = {
                "states": np.array(grp["states"][0]),
                "model": grp.attrs["model_file"],
                "ep_meta": grp.attrs.get("ep_meta", None),
            }
        print(f"[RoboCasa] Anchor demo loaded: {anchor_demo} from {dataset_path}")

    # Get task-specific subtask checks
    subtask_config = get_subtask_config(env_name)

    env = RoboCasaLowdimWrapper(
        raw_env,
        low_dim_keys=detected_keys,
        max_episode_length=max_ep_len,
        subgoal_checks=subtask_config['checks'],
        anchor_state_dict=anchor_state_dict,
        extra_obs_fn=extra_obs_fn,
        goal_conditioning_mode=goal_conditioning_mode,
        no_of_subtasks=no_of_subtasks,
    )
    env.seed(seed)
    env.env.hard_reset = False
    return env


# ---------------------------------------------------------------------------
# Cumulative subtask-status helpers (for goal-conditioned BC data relabel).
# Mirrors robomimic_utils._infer_subtask_status_from_states.
# ---------------------------------------------------------------------------

def _cumulate_goal_done(goal_done, subgoal_checks, base_env, no_of_subtasks,
                        subtask_dag=None):
    """Sticky/cumulative ratcheting of subgoal done-flags at the current env state.
    Sequential by default; optional DAG gating via subtask_dag."""
    goals = [fn(base_env) for fn in subgoal_checks[:no_of_subtasks]]
    for i in range(min(len(goals), no_of_subtasks)):
        if subtask_dag is not None:
            prereqs = subtask_dag.get(i, [])
            if goals[i] and all(goal_done[p] for p in prereqs):
                goal_done[i] = True
        else:
            if goals[i] and all(goal_done[j] for j in range(i)):
                goal_done[i] = True
    return goal_done


def _infer_subtask_status_from_states(lowdim_wrapper, states, episode_len,
                                      no_of_subtasks, subgoal_checks,
                                      subtask_dag=None):
    """Replay each stored sim state and compute cumulative subtask_status per timestep.

    Returns int8 array of shape (episode_len, no_of_subtasks).
    """
    inferred = np.zeros((episode_len, no_of_subtasks), dtype=np.int8)
    if episode_len <= 0:
        return inferred
    sim_states = np.asarray(states)
    assert sim_states.shape[0] >= episode_len, \
        f"states too short: {sim_states.shape[0]} < episode_len={episode_len}"
    base_env = lowdim_wrapper.env
    goal_done = [False] * no_of_subtasks
    for t in range(episode_len):
        base_env.sim.set_state_from_flattened(sim_states[t])
        base_env.sim.forward()
        if hasattr(base_env, "update_sites"):
            base_env.update_sites()
        if hasattr(base_env, "update_state"):
            base_env.update_state()
        goal_done = _cumulate_goal_done(
            goal_done, subgoal_checks, base_env, no_of_subtasks, subtask_dag
        )
        inferred[t, :] = np.asarray([int(g) for g in goal_done], dtype=np.int8)
    return inferred


def _unwrap_lowdim_wrapper(env):
    """Walk .env chain to find the underlying RoboCasaLowdimWrapper."""
    cur = env
    while cur is not None:
        if isinstance(cur, RoboCasaLowdimWrapper):
            return cur
        cur = getattr(cur, "env", None)
    return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def get_dataset(env, env_name):
    """Load the offline BC dataset from the HDF5 file.

    If the wrapper has a `goal_conditioning_mode` set (e.g. 'cumulative'), replay
    each stored sim state to compute per-timestep cumulative subtask_status and
    append the encoded goal vector to obs/next_obs (matching the online env).
    """
    dataset_path = _get_dataset_path(env_name)
    subtask_config = get_subtask_config(env_name)
    subgoal_checks = subtask_config['checks']
    no_of_subtasks_cfg = len(subtask_config['names'])

    lowdim_wrapper = _unwrap_lowdim_wrapper(env)
    gc_mode = getattr(lowdim_wrapper, "goal_conditioning_mode", None) if lowdim_wrapper is not None else None
    status_dim = int(getattr(lowdim_wrapper, "no_of_subtasks", 0) or 0) if lowdim_wrapper is not None else 0
    if status_dim <= 0:
        status_dim = no_of_subtasks_cfg
    goal_conditioned = gc_mode is not None
    if goal_conditioned and status_dim <= 0:
        raise AssertionError("Goal-conditioned dataset loading requires a positive no_of_subtasks.")
    subtask_dag = getattr(lowdim_wrapper, "_subtask_dag", None) if lowdim_wrapper is not None else None

    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        inds = np.argsort([int(d.split('_')[1]) for d in demos])
        demos = [demos[i] for i in inds]

        num_timesteps = sum(f[f"data/{d}/actions"].shape[0] for d in demos)
        task_name = _get_task_name(env_name)
        print(f"RoboCasa [{task_name}] dataset: {len(demos)} demos, {num_timesteps} timesteps "
              f"(gc_mode={gc_mode}, status_dim={status_dim})")

        observations = []
        actions = []
        next_observations = []
        terminals = []
        rewards = []
        masks = []
        subtask_status = []

        for d in demos:
            obs = np.array(f[f"data/{d}/obs/{DATASET_OBS_KEY}"]).astype(np.float32)
            next_obs = np.array(f[f"data/{d}/next_obs/{DATASET_OBS_KEY}"]).astype(np.float32)
            a = np.array(f[f"data/{d}/actions"]).astype(np.float32)
            r = np.array(f[f"data/{d}/rewards"]).astype(np.float32)
            dones = np.array(f[f"data/{d}/dones"]).astype(np.float32)

            if goal_conditioned:
                if "states" not in f[f"data/{d}"]:
                    raise AssertionError(f"demo {d}: GC relabel requires `states` group in HDF5.")
                states = np.array(f[f"data/{d}/states"])
                ep_cumulative = _infer_subtask_status_from_states(
                    lowdim_wrapper=lowdim_wrapper,
                    states=states, episode_len=len(a),
                    no_of_subtasks=status_dim,
                    subgoal_checks=subgoal_checks,
                    subtask_dag=subtask_dag,
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

            observations.append(obs)
            actions.append(a)
            next_observations.append(next_obs)
            rewards.append(r)
            terminals.append(dones)
            masks.append(1.0 - dones)
            subtask_status.append(ep_subtask_status)

    concat_next_obs = np.concatenate(next_observations, axis=0)
    return Dataset.create(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=concat_next_obs,
        phase=-1 * np.ones(concat_next_obs.shape[0]),
        subtask_steps=np.zeros((concat_next_obs.shape[0], status_dim)),
        subtask_status=np.concatenate(subtask_status, axis=0).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Gymnasium wrapper
# ---------------------------------------------------------------------------

class RoboCasaLowdimWrapper(gym.Env):
    """
    Wraps a raw robosuite env as a gymnasium env with dense reward support.
    Mirrors the interface of RobomimicLowdimWrapper.
    """

    def __init__(
        self,
        env,
        low_dim_keys=('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object'),
        max_episode_length=MAX_EPISODE_LENGTH,
        render_hw=(256, 256),
        render_camera_name="robot0_agentview_center",
        subgoal_checks=None,
        anchor_state_dict=None,
        extra_obs_fn=None,
        goal_conditioning_mode=None,
        no_of_subtasks=None,
    ):
        if goal_conditioning_mode is not None and goal_conditioning_mode not in GOAL_CONDITIONING_MODES:
            raise ValueError(
                f"goal_conditioning_mode={goal_conditioning_mode!r} must be one of {GOAL_CONDITIONING_MODES}")
        self.env = env
        self.obs_keys = low_dim_keys
        self.extra_obs_fn = extra_obs_fn
        self.max_episode_length = max_episode_length
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.subgoal_checks = subgoal_checks if subgoal_checks is not None else SUBGOAL_CHECKS
        self.anchor_state_dict = anchor_state_dict
        self._anchor_model_loaded = False

        # Goal conditioning
        self.goal_conditioning_mode = goal_conditioning_mode
        if no_of_subtasks is None:
            no_of_subtasks = len(self.subgoal_checks) if self.subgoal_checks else 0
        self.no_of_subtasks = int(no_of_subtasks)

        self.env_step = 0
        self.n_episodes = 0
        self.t = 0
        self.episode_return = 0
        self.episode_length = 0
        self.goal_done = None

        # Action space
        low = np.full(env.action_dim, fill_value=-1.0)
        high = np.full(env.action_dim, fill_value=1.0)
        self.action_space = Box(low=low, high=high, shape=low.shape, dtype=low.dtype)

        # Observation space
        obs_example = self.get_observation()
        print(f"[RoboCasa] Obs dim: {obs_example.shape[0]} (keys: {list(self.obs_keys)})")
        low_obs = np.full_like(obs_example, fill_value=-1)
        high_obs = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(low=low_obs, high=high_obs, shape=low_obs.shape, dtype=low_obs.dtype)

    def _current_goal_vector(self):
        if self.goal_conditioning_mode is None:
            return np.zeros((0,), dtype=np.float32)
        if self.goal_done is None or len(self.goal_done) != self.no_of_subtasks:
            cumulative = np.zeros(self.no_of_subtasks, dtype=np.int8)
        else:
            cumulative = np.asarray(self.goal_done, dtype=np.int8)
        return _encode_goal_vector(cumulative, self.goal_conditioning_mode).astype(np.float32)

    def get_observation(self):
        raw_obs = self.env._get_observations()
        parts = [raw_obs[key] for key in self.obs_keys]
        if self.extra_obs_fn is not None:
            parts.append(self.extra_obs_fn(self.env))
        if self.goal_conditioning_mode is not None:
            parts.append(self._current_goal_vector())
        return np.concatenate(parts, axis=0)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options=None, **kwargs):
        self.t = 0
        self.episode_return = 0
        self.episode_length = 0
        self.n_episodes += 1
        self.goal_done = None

        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        if options is None:
            options = {}

        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        if self.anchor_state_dict is not None:
            self._reset_to_anchor()
        else:
            new_seed = options.get("seed", None)
            if new_seed is not None:
                self.seed(seed=new_seed)
                self.env.reset()
            else:
                self.env.reset()

        return self.get_observation(), {}

    def _reset_to_anchor(self):
        """Reset the environment to the anchored demo state.

        On the first call, loads the model XML (layout/fixtures) and caches it.
        On every call (including the first), restores the sim state.
        """
        state = self.anchor_state_dict
        env = self.env

        if not self._anchor_model_loaded:
            ep_meta = {}
            if state.get("ep_meta", None) is not None:
                ep_meta = json.loads(state["ep_meta"]) if isinstance(state["ep_meta"], str) else state["ep_meta"]

            if hasattr(env, "set_ep_meta"):
                env.set_ep_meta(ep_meta)
            elif hasattr(env, "set_attrs_from_ep_meta"):
                env.set_attrs_from_ep_meta(ep_meta)

            env.reset()
            xml = env.edit_model_xml(state["model"])
            env.reset_from_xml_string(xml)
            env.sim.reset()

            fref = ep_meta.get("fixture_refs") if isinstance(ep_meta, dict) else None
            if fref and hasattr(env, "fixtures"):
                for attr_name, tgt_name in fref.items():
                    if not isinstance(tgt_name, str) or not hasattr(env, attr_name):
                        continue
                    for fx in env.fixtures.values():
                        if getattr(fx, "naming_prefix", "").rstrip("_") == tgt_name:
                            setattr(env, attr_name, fx)
                            break

            self._anchor_model_loaded = True

        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()

        if hasattr(env, "update_sites"):
            env.update_sites()
        if hasattr(env, "update_state"):
            env.update_state()

        # Reset controller internal state to match post-restore sim:
        # update_state() refreshes origin/base-pose references; reset() resets
        # per-part goals to current ref_pos/ref_ori. Same sequence robot.reset
        # uses (see robosuite robots/robot.py ~line 298-300).
        for robot in getattr(env, "robots", []):
            cc = getattr(robot, "composite_controller", None)
            if cc is None:
                continue
            if hasattr(cc, "update_state"):
                cc.update_state()
            if hasattr(cc, "reset"):
                cc.reset()

    def configure_dense_reward(self, use_dense_reward=True, normalize_rewards=False, no_of_subtasks=5, subtask_dag=None):
        """Configure persistent dense reward settings for Q-IPM."""
        self._dense_reward_enabled = use_dense_reward
        self._dense_normalize_rewards = normalize_rewards
        self._dense_no_of_subtasks = no_of_subtasks
        self._subtask_dag = subtask_dag

    def step(self, action, use_dense_reward=False, normalize_rewards=False, no_of_subtasks=5):
        # Use persistent config if set, otherwise use kwargs
        use_dense_reward = getattr(self, '_dense_reward_enabled', False) or use_dense_reward
        normalize_rewards = getattr(self, '_dense_normalize_rewards', False) or normalize_rewards
        no_of_subtasks = getattr(self, '_dense_no_of_subtasks', None) or self.no_of_subtasks or no_of_subtasks

        raw_obs, reward, done, info = self.env.step(action)

        # Ratchet subtask status if needed for dense reward or GC observations.
        if use_dense_reward or self.goal_conditioning_mode is not None:
            self._update_goal_done(no_of_subtasks)
            info['reward_info'] = self._last_reward_info

        if use_dense_reward:
            reward = -sum(1 for i in range(no_of_subtasks) if not self.goal_done[i])
            if normalize_rewards:
                reward /= no_of_subtasks

        parts = [raw_obs[key] for key in self.obs_keys]
        if self.extra_obs_fn is not None:
            parts.append(self.extra_obs_fn(self.env))
        if self.goal_conditioning_mode is not None:
            parts.append(self._current_goal_vector())
        obs = np.concatenate(parts, axis=0)

        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1

        check_success_result = self.env._check_success()
        info["success"] = int(check_success_result)

        if info["success"]:
            done = True
            info['return'] = self.episode_return
            info['length'] = self.episode_length

        if done:
            return obs, reward, True, False, info
        if self.t >= self.max_episode_length:
            return obs, reward, False, True, info
        return obs, reward, False, False, info

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        if mode == "rgb_array" and hasattr(self.env, 'sim'):
            return self.env.sim.render(
                camera_name=self.render_camera_name,
                height=h, width=w,
            )[::-1]
        return np.zeros((h, w, 3), dtype=np.uint8)

    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}

    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}

    # ------------------------------------------------------------------
    # Dense reward from subtask detection
    # ------------------------------------------------------------------

    def _update_goal_done(self, no_of_subtasks):
        """Sticky/cumulative ratchet of self.goal_done from current env state."""
        if self.goal_done is None or len(self.goal_done) != no_of_subtasks:
            self.goal_done = [False] * no_of_subtasks
        self.goal_done = _cumulate_goal_done(
            self.goal_done, self.subgoal_checks, self.env, no_of_subtasks,
            subtask_dag=getattr(self, '_subtask_dag', None),
        )
        self._last_reward_info = {
            "subtasks_completed": sum(self.goal_done),
            "subtask_status": [int(g) for g in self.goal_done],
        }

    def _dense_reward(self, normalize_rewards=False, no_of_subtasks=5):
        self._update_goal_done(no_of_subtasks)
        reward = -sum(1 for i in range(no_of_subtasks) if not self.goal_done[i])
        if normalize_rewards:
            reward /= no_of_subtasks
        return reward
