"""Dense reward wrapper for OGBench scene environments.

Defines sequential subtasks with ratchet-style rewards for Q-IPM,
following the same pattern as robomimic_utils.py.
Supports optional goal conditioning (cumulative / singular).
"""
import gymnasium
import numpy as np
from gymnasium.spaces import Box


# Task 4: put_in_drawer
# Sequential subtasks:
#   0. Press button 0 (unlock drawer)
#   1. Open drawer (drawer_pos < -0.12)
#   2. Place block in drawer
#   3. Close drawer (|drawer_pos| < 0.04)
#
# The block target is inside the closed drawer at (0.33, -0.356, 0.066).
# To place it there the agent must open the drawer first, then close it.

SCENE_TASK4_SUBTASKS = [
    "press_button_0",
    "open_drawer",
    "block_in_drawer",
    "close_drawer",
]

GOAL_CONDITIONING_MODES = (None, 'cumulative', 'singular')


def cumulative_to_singular(cumulative: np.ndarray) -> np.ndarray:
    """Convert a cumulative goal-status array to a singular (frontier) array.

    A goal at index *i* is a **frontier** goal when:
    - it is not yet completed (cumulative[i] == 0), AND
    - it is the first goal (i == 0) OR its predecessor is completed
      (cumulative[i-1] == 1).

    Examples:
        cumulative (1, 1, 0, 0, 0) -> singular (0, 0, 1, 0, 0)
        cumulative (1, 1, 0, 0, 1, 0, 0) -> singular (0, 0, 1, 0, 0, 1, 0)

    Args:
        cumulative: int/bool array of shape (..., goal_dim) with binary done flags.

    Returns:
        int8 array of the same shape, 1 at each frontier goal, 0 elsewhere.
    """
    arr = np.asarray(cumulative)
    incomplete = arr == 0
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


def check_subtask_conditions(base_env, no_of_subtasks=4):
    """Check raw subtask conditions from env state (no ratcheting).

    Returns:
        List of bools, one per subtask.
    """
    # Subtask 0: button 0 pressed
    button_0_pressed = bool(base_env._cur_button_states[0] == 1)

    # Subtask 1: drawer open (drawer_pos < -0.12, range is [-0.16, 0])
    drawer_pos = float(base_env._data.joint('drawer_slide').qpos[0])
    drawer_open = drawer_pos < -0.12

    # Subtask 2: block in drawer
    block_pos = base_env._data.joint('object_joint_0').qpos[:3]
    block_in_drawer = base_env._is_in_drawer(block_pos)

    # Subtask 3: drawer closed (back to ~0)
    drawer_closed = abs(drawer_pos) < 0.04

    raw_goals = [button_0_pressed, drawer_open, block_in_drawer, drawer_closed]
    return raw_goals[:no_of_subtasks]


class OGBenchDenseRewardWrapper(gymnasium.Wrapper):
    """Wraps an OGBench scene env to provide dense sequential subtask rewards."""

    def __init__(self, env):
        super().__init__(env)
        self.goal_done = None
        self._dense_reward_enabled = False
        self._dense_normalize_rewards = False
        self._dense_no_of_subtasks = 4
        self._last_reward_info = {}
        self._goal_conditioning_mode = None

    @property
    def observation_space(self):
        base_space = self.env.observation_space
        if self._goal_conditioning_mode is not None:
            low = np.concatenate([base_space.low, np.zeros(self._dense_no_of_subtasks, dtype=np.float32)])
            high = np.concatenate([base_space.high, np.ones(self._dense_no_of_subtasks, dtype=np.float32)])
            return Box(low=low, high=high, dtype=np.float32)
        return base_space

    def configure_dense_reward(self, use_dense_reward=True, normalize_rewards=False,
                               no_of_subtasks=4, goal_conditioning_mode=None):
        """Configure persistent dense reward settings for Q-IPM."""
        self._dense_reward_enabled = use_dense_reward
        self._dense_normalize_rewards = normalize_rewards
        self._dense_no_of_subtasks = no_of_subtasks
        self._goal_conditioning_mode = goal_conditioning_mode

    def _current_goal_vector(self):
        """Return encoded goal vector for the current step."""
        if self._goal_conditioning_mode is None:
            return np.zeros((0,), dtype=np.float32)
        cumulative = np.asarray(self.goal_done, dtype=np.int8)
        return _encode_goal_vector(cumulative, self._goal_conditioning_mode).astype(np.float32)

    def _build_observation(self, obs):
        """Append goal vector to observation if goal conditioning is enabled."""
        if self._goal_conditioning_mode is not None:
            goal_vec = self._current_goal_vector()
            obs = np.concatenate([obs, goal_vec], axis=-1)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.goal_done = [False] * self._dense_no_of_subtasks
        obs = self._build_observation(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update goal_done from environment state
        self._update_goal_done()

        if self._dense_reward_enabled:
            reward = self._compute_dense_reward()
            info['reward_info'] = self._last_reward_info

        # Terminate on success — all subtasks done
        if all(self.goal_done):
            terminated = True
            info['success'] = 1.0

        obs = self._build_observation(obs)
        return obs, reward, terminated, truncated, info

    def _update_goal_done(self):
        """Update self.goal_done from current env state with ratcheting."""
        no_of_subtasks = self._dense_no_of_subtasks
        if self.goal_done is None or len(self.goal_done) != no_of_subtasks:
            self.goal_done = [False] * no_of_subtasks

        raw_goals = check_subtask_conditions(self.env.unwrapped, no_of_subtasks)

        for i in range(min(len(raw_goals), no_of_subtasks)):
            if i == 0:
                condition = raw_goals[i]
            else:
                condition = raw_goals[i] and self.goal_done[i - 1]
            if condition:
                self.goal_done[i] = True

    def _compute_dense_reward(self):
        """Compute dense per-subtask reward from current goal_done state."""
        no_of_subtasks = self._dense_no_of_subtasks
        normalize_rewards = self._dense_normalize_rewards

        reward = -sum(1 for i in range(no_of_subtasks) if not self.goal_done[i])

        self._last_reward_info = {
            'subtasks_completed': sum(self.goal_done),
            'subtask_status': [int(g) for g in self.goal_done],
            'normalized_reward': reward / no_of_subtasks if normalize_rewards else reward,
            'raw_reward': reward,
        }

        if normalize_rewards:
            reward /= no_of_subtasks

        return reward


def relabel_subtask_status(env, dataset, no_of_subtasks, goal_conditioning_mode=None):
    """Relabel offline dataset with subtask_status (and optionally goal vectors).

    Replays each timestep via env.unwrapped.set_state() to determine subtask
    completion status. When goal_conditioning_mode is set, appends goal vectors
    to observations and next_observations.

    Args:
        env: The OGBench gymnasium environment (will use env.unwrapped).
        dataset: Dict with 'observations', 'next_observations', 'terminals',
                 'qpos', 'qvel', 'button_states'.
        no_of_subtasks: Number of subtasks.
        goal_conditioning_mode: None, 'cumulative', or 'singular'.
    """
    N = len(dataset['observations'])
    subtask_status = np.zeros((N, no_of_subtasks), dtype=np.float32)

    base_env = env.unwrapped
    qpos_all = dataset['qpos']
    qvel_all = dataset['qvel']
    button_states_all = dataset['button_states']
    terminals = dataset['terminals']

    # Track cumulative goal_done per trajectory with ratcheting
    goal_done = [False] * no_of_subtasks

    for t in range(N):
        # Reset goal_done at trajectory boundaries
        # If t == 0 or the previous timestep was a terminal, start fresh
        if t == 0 or terminals[t - 1] > 0.5:
            goal_done = [False] * no_of_subtasks

        # Set environment state to this timestep
        qpos = qpos_all[t]
        qvel = qvel_all[t]
        button_states = button_states_all[t]
        base_env.set_state(qpos, qvel, button_states)

        # Check subtask conditions and apply ratcheting
        raw_goals = check_subtask_conditions(base_env, no_of_subtasks)
        for i in range(min(len(raw_goals), no_of_subtasks)):
            if i == 0:
                condition = raw_goals[i]
            else:
                condition = raw_goals[i] and goal_done[i - 1]
            if condition:
                goal_done[i] = True

        subtask_status[t] = np.array([int(g) for g in goal_done], dtype=np.float32)

    dataset['subtask_status'] = subtask_status
    print(f"[relabel_subtask_status] Relabeled {N} timesteps with subtask_status "
          f"(final sum: {subtask_status.sum(axis=0)})")

    # Append goal vectors to observations if goal conditioning is enabled
    if goal_conditioning_mode is not None:
        obs = dataset['observations']
        next_obs = dataset['next_observations']

        # Goal vector at timestep t uses subtask_status from t-1
        # (agent sees pre-action status), matching robomimic convention
        prev_status = np.zeros_like(subtask_status)
        for t in range(N):
            if t == 0 or terminals[t - 1] > 0.5:
                prev_status[t] = np.zeros(no_of_subtasks, dtype=np.float32)
            else:
                prev_status[t] = subtask_status[t - 1]

        goal_vectors = np.array([
            _encode_goal_vector(prev_status[t], goal_conditioning_mode)
            for t in range(N)
        ], dtype=np.float32)

        # next_observations goal uses current subtask_status
        next_goal_vectors = np.array([
            _encode_goal_vector(subtask_status[t], goal_conditioning_mode)
            for t in range(N)
        ], dtype=np.float32)

        dataset['observations'] = np.concatenate([obs, goal_vectors], axis=-1).astype(obs.dtype)
        dataset['next_observations'] = np.concatenate([next_obs, next_goal_vectors], axis=-1).astype(next_obs.dtype)

        print(f"[relabel_subtask_status] Appended goal vectors (mode={goal_conditioning_mode}). "
              f"obs shape: {obs.shape} -> {dataset['observations'].shape}")
