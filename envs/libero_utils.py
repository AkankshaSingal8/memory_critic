"""
LIBERO environment utilities for OGPO training.

Multi-task: dispatches per-task obs keys, subgoal predicates, and dataset paths
via TASK_CONFIGS. To add a new LIBERO task, append a new entry there.

Currently supported:
  - libero_KITCHEN_SCENE8: "put both moka pots on the stove" (4 sequential subgoals)
  - libero_KITCHEN_SCENE3: "turn on the stove and put the moka pot on it" (3 sequential subgoals)

Mirrors envs/robocasa_utils.py contract:
  - LiberoLowdimWrapper exposes goal_conditioning_mode + no_of_subtasks.
  - get_observation() concatenates the cumulative goal vector when GC enabled.
  - step() ratchets self.goal_done and emits info['reward_info']['subtask_status'].
"""

import os
import json
import pickle
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import h5py
import imageio

from ogpo.utils.datasets import Dataset
from envs.ogbench_dense_wrapper import GOAL_CONDITIONING_MODES, _encode_goal_vector
from envs.libero_subtasks import (
    SUBGOAL_NAMES_KITCHEN_SCENE8, SUBGOAL_CHECKS_KITCHEN_SCENE8,
    SUBGOAL_NAMES_KITCHEN_SCENE3, SUBGOAL_CHECKS_KITCHEN_SCENE3,
    SUBGOAL_NAMES_KITCHEN_SCENE4, SUBGOAL_CHECKS_KITCHEN_SCENE4,
    SUBGOAL_NAMES_KITCHEN_SCENE6, SUBGOAL_CHECKS_KITCHEN_SCENE6,
)


# ---------------------------------------------------------------------------
# Per-task configuration
# ---------------------------------------------------------------------------

DATASET_OBS_KEY = 'observation_state'

_ROBOT_OBS_KEYS = [
    'robot0_eef_pos', 'robot0_eef_quat',
    'robot0_gripper_qpos', 'robot0_joint_pos',
]


def _stove_on_obs(env):
    """Return [1.0] if flat_stove_1 is on, else [0.0]. 1-D float32 array."""
    return np.array(
        [float(env._eval_predicate(("turnon", "flat_stove_1")))],
        dtype=np.float32,
    )


def _drawer_open_obs_ks4(env):
    """Return [1.0] if white_cabinet_1 bottom region is open, else [0.0]."""
    return np.array(
        [float(env._eval_predicate(("open", "white_cabinet_1_bottom_region")))],
        dtype=np.float32,
    )


def _microwave_open_obs_ks6(env):
    """Return [1.0] if microwave_1 is open, else [0.0]."""
    return np.array(
        [float(env._eval_predicate(("open", "microwave_1")))],
        dtype=np.float32,
    )


# Each entry: obs_keys (raw robosuite obs dict keys to flatten + concat) +
# extra_obs_fn (callable env -> 1-D np.array of additional obs) + obs_dim +
# subgoals + filenames.
TASK_CONFIGS = {
    'KITCHEN_SCENE8': {
        'obs_keys': _ROBOT_OBS_KEYS + [
            'moka_pot_1_pos', 'moka_pot_1_quat',
            'moka_pot_2_pos', 'moka_pot_2_quat',
        ],
        'extra_obs_fn': None,
        'extra_obs_dim': 0,
        'obs_dim': 30,  # 16 robot + 14 object
        'max_episode_length': 600,
        'subgoal_names': SUBGOAL_NAMES_KITCHEN_SCENE8,
        'subgoal_checks': SUBGOAL_CHECKS_KITCHEN_SCENE8,
        'bddl_file': 'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl',
        'init_file': 'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.pruned_init',
        'dataset_file': 'KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo_processed.hdf5',
    },
    'KITCHEN_SCENE3': {
        'obs_keys': _ROBOT_OBS_KEYS + [
            'moka_pot_1_pos', 'moka_pot_1_quat',
        ],
        'extra_obs_fn': _stove_on_obs,
        'extra_obs_dim': 1,
        'obs_dim': 24,  # 16 robot + 7 moka_pot + 1 stove_on
        'max_episode_length': 500,
        'subgoal_names': SUBGOAL_NAMES_KITCHEN_SCENE3,
        'subgoal_checks': SUBGOAL_CHECKS_KITCHEN_SCENE3,
        'bddl_file': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl',
        'init_file': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.pruned_init',
        'dataset_file': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo_processed.hdf5',
    },
    'KITCHEN_SCENE4': {
        'obs_keys': _ROBOT_OBS_KEYS + [
            'akita_black_bowl_1_pos', 'akita_black_bowl_1_quat',
        ],
        'extra_obs_fn': _drawer_open_obs_ks4,
        'extra_obs_dim': 1,
        'obs_dim': 24,  # 16 robot + 7 bowl + 1 drawer_open
        'max_episode_length': 500,
        'subgoal_names': SUBGOAL_NAMES_KITCHEN_SCENE4,
        'subgoal_checks': SUBGOAL_CHECKS_KITCHEN_SCENE4,
        'bddl_file': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl',
        'init_file': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.pruned_init',
        'dataset_file': 'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo_processed.hdf5',
    },
    'KITCHEN_SCENE6': {
        'obs_keys': _ROBOT_OBS_KEYS + [
            'white_yellow_mug_1_pos', 'white_yellow_mug_1_quat',
        ],
        'extra_obs_fn': _microwave_open_obs_ks6,
        'extra_obs_dim': 1,
        'obs_dim': 24,  # 16 robot + 7 mug + 1 microwave_open
        'max_episode_length': 500,
        'subgoal_names': SUBGOAL_NAMES_KITCHEN_SCENE6,
        'subgoal_checks': SUBGOAL_CHECKS_KITCHEN_SCENE6,
        'bddl_file': 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl',
        'init_file': 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.pruned_init',
        'dataset_file': 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo_processed.hdf5',
    },
}


# ---------------------------------------------------------------------------
# Env identification helpers
# ---------------------------------------------------------------------------

def is_libero_env(env_name):
    return env_name.startswith('libero_')


def _get_task_name(env_name):
    return env_name.replace('libero_', '', 1)


def _get_task_config(env_name):
    task_name = _get_task_name(env_name)
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown LIBERO task: {task_name}. "
            f"Supported: {list(TASK_CONFIGS.keys())}"
        )
    return TASK_CONFIGS[task_name]


def get_subtask_config(env_name):
    """Compatibility shim — returns {names, checks} like robocasa_utils."""
    cfg = _get_task_config(env_name)
    return {'names': cfg['subgoal_names'], 'checks': cfg['subgoal_checks']}


def _get_dataset_path(env_name):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = _get_task_config(env_name)
    return os.path.join(project_root, 'data', 'libero', cfg['dataset_file'])


def _get_env_meta(dataset_path):
    with h5py.File(dataset_path, "r") as f:
        return json.loads(f["data"].attrs["env_args"])


def _get_bddl_path(env_name):
    envs_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = _get_task_config(env_name)
    p = os.path.join(envs_dir, 'libero_bddl', cfg['bddl_file'])
    if not os.path.isfile(p):
        raise FileNotFoundError(f"BDDL file not found: {p}")
    return p


def _get_init_states(env_name):
    envs_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = _get_task_config(env_name)
    p = os.path.join(envs_dir, 'libero_data', cfg['init_file'])
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Init states file not found: {p}")
    npz = np.load(p, allow_pickle=True)
    return pickle.loads(npz['archive/data.pkl'])


def _extract_obs_from_raw(raw_obs, obs_keys, extra_obs_fn=None, env=None):
    """Flatten raw obs dict for the configured keys and append extra_obs (if any)."""
    parts = [np.asarray(raw_obs[k]).flatten() for k in obs_keys]
    if extra_obs_fn is not None:
        parts.append(np.asarray(extra_obs_fn(env)).flatten())
    return np.concatenate(parts, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Cumulative subtask-status helpers (duplicated from robocasa_utils to avoid
# importing robocasa, which is not installed in this venv).
# ---------------------------------------------------------------------------

def _cumulate_goal_done(goal_done, subgoal_checks, base_env, no_of_subtasks,
                        subtask_dag=None):
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


def _unwrap_lowdim_wrapper(env):
    cur = env
    while cur is not None:
        if isinstance(cur, LiberoLowdimWrapper):
            return cur
        cur = getattr(cur, "env", None)
    return None


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def make_env(env_name, seed=0, enable_render=False, enable_render_images=False,
             goal_conditioning_mode=None, no_of_subtasks=None):
    """Create a LIBERO environment wrapped for OGPO training."""
    cfg = _get_task_config(env_name)
    dataset_path = _get_dataset_path(env_name)
    env_meta = _get_env_meta(dataset_path)
    bddl_path = _get_bddl_path(env_name)

    env_kwargs = dict(env_meta.get("env_kwargs", {}))
    env_kwargs.pop("env_name", None)
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = enable_render or enable_render_images
    env_kwargs["use_camera_obs"] = False
    env_kwargs["bddl_file_name"] = bddl_path

    import robosuite  # noqa: F401
    from libero.libero.envs import TASK_MAPPING

    problem_name = env_meta.get("problem_name", "libero_kitchen_tabletop_manipulation")
    task_cls = TASK_MAPPING.get(problem_name)
    if task_cls is None:
        raise ValueError(
            f"LIBERO task '{problem_name}' not found in TASK_MAPPING. "
            f"Available: {list(TASK_MAPPING.keys())}"
        )

    raw_env = task_cls(
        robots=env_kwargs.pop("robots", ["Panda"]),
        controller_configs=env_kwargs.pop("controller_configs", None),
        **{k: v for k, v in env_kwargs.items()
           if k not in ("camera_names", "camera_heights", "camera_widths",
                        "camera_depths", "camera_segmentations", "reward_shaping")}
    )
    raw_env.reset()

    init_states = _get_init_states(env_name)

    env = LiberoLowdimWrapper(
        raw_env,
        obs_keys=cfg['obs_keys'],
        extra_obs_fn=cfg['extra_obs_fn'],
        max_episode_length=cfg['max_episode_length'],
        subgoal_checks=cfg['subgoal_checks'],
        init_states=init_states,
        enable_render=enable_render,
        enable_render_images=enable_render_images,
        goal_conditioning_mode=goal_conditioning_mode,
        no_of_subtasks=no_of_subtasks,
    )
    env.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def get_dataset(env, env_name):
    """Load the offline BC dataset from the processed HDF5.

    Per-timestep cumulative `subtask_status` is read from the HDF5 (precomputed
    during extraction). When the env wrapper has `goal_conditioning_mode` set,
    the encoded goal vector is concatenated to obs/next_obs so dataset shape
    matches what the live wrapper produces during eval/online training.
    """
    dataset_path = _get_dataset_path(env_name)
    cfg = _get_task_config(env_name)
    no_of_subtasks_cfg = len(cfg['subgoal_names'])

    lowdim_wrapper = _unwrap_lowdim_wrapper(env)
    status_dim = int(getattr(lowdim_wrapper, "no_of_subtasks", 0) or 0) if lowdim_wrapper is not None else 0
    if status_dim <= 0:
        status_dim = no_of_subtasks_cfg
    gc_mode = getattr(lowdim_wrapper, "goal_conditioning_mode", None) if lowdim_wrapper is not None else None
    goal_conditioned = gc_mode is not None
    if goal_conditioned and status_dim <= 0:
        raise AssertionError("Goal-conditioned dataset loading requires a positive no_of_subtasks.")

    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        inds = np.argsort([int(d.split('_')[1]) for d in demos])
        demos = [demos[i] for i in inds]

        num_timesteps = sum(f[f"data/{d}/actions"].shape[0] for d in demos)
        task_name = _get_task_name(env_name)
        print(f"LIBERO [{task_name}] dataset: {len(demos)} demos, {num_timesteps} timesteps "
              f"(gc_mode={gc_mode}, status_dim={status_dim})")

        observations, actions, next_observations = [], [], []
        terminals, rewards, masks, subtask_status = [], [], [], []

        for d in demos:
            obs = np.array(f[f"data/{d}/obs/{DATASET_OBS_KEY}"]).astype(np.float32)
            next_obs = np.array(f[f"data/{d}/next_obs/{DATASET_OBS_KEY}"]).astype(np.float32)
            a = np.array(f[f"data/{d}/actions"]).astype(np.float32)
            r = np.array(f[f"data/{d}/rewards"]).astype(np.float32)
            dones = np.array(f[f"data/{d}/dones"]).astype(np.float32)

            if "subtask_status" in f[f"data/{d}"]:
                ep_cumulative = np.array(f[f"data/{d}/subtask_status"]).astype(np.int8)
                # Truncate/pad to match action length, in case of off-by-one.
                if ep_cumulative.shape[0] != len(a):
                    ep_cumulative = ep_cumulative[:len(a)]
            else:
                ep_cumulative = np.zeros((len(a), max(status_dim, 1)), dtype=np.int8)
            # Use only the first status_dim columns in case the stored array is wider.
            ep_cumulative = ep_cumulative[:, :status_dim]

            if goal_conditioned:
                # obs[t]'s goal reflects status at the *previous* timestep
                # (matches robocasa/robomimic convention; obs[0] starts at zero).
                obs_cumulative = np.zeros_like(ep_cumulative)
                obs_cumulative[1:] = ep_cumulative[:-1]
                obs_goal = _encode_goal_vector(obs_cumulative, gc_mode).astype(obs.dtype, copy=False)
                next_goal = _encode_goal_vector(ep_cumulative, gc_mode).astype(next_obs.dtype, copy=False)
                obs = np.concatenate([obs, obs_goal], axis=-1)
                next_obs = np.concatenate([next_obs, next_goal], axis=-1)

            observations.append(obs)
            actions.append(a)
            next_observations.append(next_obs)
            rewards.append(r)
            terminals.append(dones)
            masks.append(1.0 - dones)
            subtask_status.append(ep_cumulative)

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

class LiberoLowdimWrapper(gym.Env):
    """Wraps a raw LIBERO/robosuite env as a gymnasium env with dense reward and
    optional goal-conditioning support. Mirrors RoboCasaLowdimWrapper interface."""

    def __init__(
        self,
        env,
        obs_keys,
        extra_obs_fn=None,
        max_episode_length=600,
        render_hw=(256, 256),
        render_camera_name="agentview",
        subgoal_checks=None,
        init_states=None,
        enable_render=False,
        enable_render_images=False,
        goal_conditioning_mode=None,
        no_of_subtasks=None,
    ):
        if goal_conditioning_mode is not None and goal_conditioning_mode not in GOAL_CONDITIONING_MODES:
            raise ValueError(
                f"goal_conditioning_mode={goal_conditioning_mode!r} must be one of {GOAL_CONDITIONING_MODES}")
        self.env = env
        self.obs_keys = list(obs_keys)
        self.extra_obs_fn = extra_obs_fn
        self.max_episode_length = max_episode_length
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.subgoal_checks = subgoal_checks if subgoal_checks is not None else []
        self.init_states = init_states
        self.enable_render = enable_render
        self.enable_render_images = enable_render_images

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
        self._last_reward_info = None

        # Action space
        low = np.full(env.action_dim, fill_value=-1.0)
        high = np.full(env.action_dim, fill_value=1.0)
        self.action_space = Box(low=low, high=high, shape=low.shape, dtype=np.float32)

        # Observation space
        state_example = self._get_state_obs()
        print(f"[LIBERO] Obs dim: {state_example.shape[0]} "
              f"(gc_mode={self.goal_conditioning_mode}, no_of_subtasks={self.no_of_subtasks}, "
              f"render_images={self.enable_render_images})")
        low_obs = np.full_like(state_example, fill_value=-np.inf)
        high_obs = np.full_like(state_example, fill_value=np.inf)
        state_space = Box(low=low_obs, high=high_obs, shape=low_obs.shape, dtype=np.float32)
        if self.enable_render_images:
            image_space = Box(
                low=0, high=255,
                shape=(self.render_hw[0], self.render_hw[1], 3),
                dtype=np.uint8,
            )
            self.observation_space = gym.spaces.Dict({
                "state": state_space,
                "image": image_space,
            })
        else:
            self.observation_space = state_space

    def _current_goal_vector(self):
        if self.goal_conditioning_mode is None:
            return np.zeros((0,), dtype=np.float32)
        if self.goal_done is None or len(self.goal_done) != self.no_of_subtasks:
            cumulative = np.zeros(self.no_of_subtasks, dtype=np.int8)
        else:
            cumulative = np.asarray(self.goal_done, dtype=np.int8)
        return _encode_goal_vector(cumulative, self.goal_conditioning_mode).astype(np.float32)

    def _get_state_obs(self):
        raw_obs = self.env._get_observations(force_update=True)
        parts = [np.asarray(raw_obs[k]).flatten() for k in self.obs_keys]
        if self.extra_obs_fn is not None:
            parts.append(np.asarray(self.extra_obs_fn(self.env)).flatten())
        if self.goal_conditioning_mode is not None:
            parts.append(self._current_goal_vector())
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _get_image_obs(self):
        h, w = self.render_hw
        if hasattr(self.env, 'sim'):
            return self.env.sim.render(
                camera_name=self.render_camera_name,
                height=h, width=w,
            )[::-1].astype(np.uint8)
        return np.zeros((h, w, 3), dtype=np.uint8)

    def get_observation(self):
        state = self._get_state_obs()
        if self.enable_render_images:
            return {"state": state, "image": self._get_image_obs()}
        return state

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

        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)

        self.env.reset()
        if self.init_states is not None and len(self.init_states) > 0:
            idx = np.random.randint(0, len(self.init_states))
            self.env.sim.set_state_from_flattened(self.init_states[idx])
            self.env.sim.forward()

        return self.get_observation(), {}

    def configure_dense_reward(self, use_dense_reward=True, normalize_rewards=False,
                                no_of_subtasks=None, subtask_dag=None):
        self._dense_reward_enabled = use_dense_reward
        self._dense_normalize_rewards = normalize_rewards
        self._dense_no_of_subtasks = no_of_subtasks if no_of_subtasks is not None else self.no_of_subtasks
        self._subtask_dag = subtask_dag

    def step(self, action, use_dense_reward=False, normalize_rewards=False, no_of_subtasks=None):
        use_dense_reward = getattr(self, '_dense_reward_enabled', False) or use_dense_reward
        normalize_rewards = getattr(self, '_dense_normalize_rewards', False) or normalize_rewards
        no_of_subtasks = (
            getattr(self, '_dense_no_of_subtasks', None)
            or no_of_subtasks
            or self.no_of_subtasks
        )

        raw_obs, reward, done, info = self.env.step(action)

        if use_dense_reward or self.goal_conditioning_mode is not None:
            self._update_goal_done(no_of_subtasks)
            info['reward_info'] = self._last_reward_info

        if use_dense_reward:
            reward = -sum(1 for i in range(no_of_subtasks) if not self.goal_done[i])
            if normalize_rewards:
                reward /= no_of_subtasks

        parts = [np.asarray(raw_obs[k]).flatten() for k in self.obs_keys]
        if self.extra_obs_fn is not None:
            parts.append(np.asarray(self.extra_obs_fn(self.env)).flatten())
        if self.goal_conditioning_mode is not None:
            parts.append(self._current_goal_vector())
        state_obs = np.concatenate(parts, axis=0).astype(np.float32)
        if self.enable_render_images:
            obs = {"state": state_obs, "image": self._get_image_obs()}
        else:
            obs = state_obs

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

    def _update_goal_done(self, no_of_subtasks):
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

    def _dense_reward(self, normalize_rewards=False, no_of_subtasks=None):
        no_of_subtasks = no_of_subtasks or self.no_of_subtasks
        self._update_goal_done(no_of_subtasks)
        reward = -sum(1 for i in range(no_of_subtasks) if not self.goal_done[i])
        if normalize_rewards:
            reward /= no_of_subtasks
        return reward
