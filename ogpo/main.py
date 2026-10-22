import glob
import os
import random
from functools import partial

import jax
import numpy as np

if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets

from ogpo.utils.config_loader import load_config, parse_cli_args, config_to_agent_dict
from ogpo.agents.modules.flax_utils import save_agent
from ogpo.utils.main_helper import (
    make_vectorized_env,
    get_checkpoint_dir,
    _find_checkpoint_dir,
    collect_run_provenance,
    compute_config_hash,
    load_rolling_checkpoint,
    cleanup_checkpoint,
    resolve_checkpoint_algo_name,
    resolve_determinism_settings,
    set_checkpoint_config_hash,
    set_repeat_id,
    validate_resumed_buffer_width,
    process_train_dataset,
    get_q_bounds,
    setup_experiment_logging,
    install_preemption_handler,
    assert_requeue_resumed,
    PreemptionDuringEval,
)
from ogpo.runners.bc_runner import (
    run_bc_training,
    run_calql_training,
    _build_eval_flags,
    _run_evaluation,
)
from ogpo.runners.online_rl_runner import run_online_rl
from ogpo.runners.dsrl_runner import run_dsrl_online_rl, DSRL_ALGORITHMS
from ogpo.runners.qipm_runner import run_qipm_online_rl, _run_qipm_evaluation
from ogpo.agents.ogpo import OGPOAgent
from ogpo.agents.qc import ACFQLAgent
from ogpo.agents.bptt import OGPOBPTTAgent
from ogpo.agents.dsrl import DSRLBestOfNAgent
from ogpo.agents.expo import EXPOAgent
from ogpo.agents.dsrl_plus_expo import DSRLEXPOAgent
from ogpo.agents.qipm import OGPOQIPMAgent
from ogpo.agents.safe_qipm import OGPOSafeQIPMAgentSharedBase
from ogpo.agents.safe_qipm_hjb import OGPOSafeQIPMHJBAgentSharedBase
from ogpo.agents.learned_lambda_safe_qipm import OGPOLearnedLambdaSafeQIPMAgentSharedBase
from ogpo.agents.temporal_bias_safe_qipm import OGPOTemporalBiasSafeQIPMAgent
from ogpo.agents.sac import SACAgent
from ogpo.agents.sac_qipm import SACQIPMAgent
from ogpo.agents.sac_qipm_avoid_obstacle import SACQIPMAvoidObstacleAgent
from ogpo.agents.sac_qipm_floq import SACQIPMFloqAgent
from ogpo.agents.sac_safe_qipm import SACSafeQIPMAgent

AGENTS = {
    "ogpo": OGPOAgent,
    "qc": ACFQLAgent,
    "acfql": ACFQLAgent,
    "bptt": OGPOBPTTAgent,
    "ogpo_bptt": OGPOBPTTAgent,
    "dsrl": DSRLBestOfNAgent,
    "dsrl_bon": DSRLBestOfNAgent,
    "expo": EXPOAgent,
    "dsrl_plus_expo": DSRLEXPOAgent,
    "qipm": OGPOQIPMAgent,
    "safe_qipm": OGPOSafeQIPMAgentSharedBase,
    "safe_qipm_hjb": OGPOSafeQIPMHJBAgentSharedBase,
    "learned_lambda_safe_qipm": OGPOLearnedLambdaSafeQIPMAgentSharedBase,
    "temporal_bias_safe_qipm": OGPOTemporalBiasSafeQIPMAgent,
    "sac": SACAgent,
    "sac_qipm": SACQIPMAgent,
    "sac_qipm_avoid_obstacle": SACQIPMAvoidObstacleAgent,
    "sac_qipm_floq": SACQIPMFloqAgent,
    "sac_safe_qipm": SACSafeQIPMAgent,
}

# Which per-step safety signal each safe_qipm-family agent's TRAIN env should
# append to subtask_status. Selected by AGENT IDENTITY, not by config: the
# backup math and the column semantics are two halves of one contract (a CMDP
# cost head fed margins, or an HJB head fed 0/1 flags, trains silently on
# nonsense), so they must not be independently settable.
#   safe_qipm     -> "cost"   (0/1 violation flags; lower = safer)
#   safe_qipm_hjb -> "margin" (signed distances; higher = safer)
#   sac_safe_qipm -> "cost"   (SAC-line CMDP fork; same columns as safe_qipm)
#   learned_lambda_safe_qipm -> "cost"  (copy-fork of safe_qipm; identical head
#                               semantics, only the task/safety scalarization
#                               differs, so it takes the same columns)
SAFE_QIPM_SAFETY_SIGNALS = {
    "safe_qipm": "cost",
    "safe_qipm_hjb": "margin",
    "sac_safe_qipm": "cost",
    "learned_lambda_safe_qipm": "cost",
    # Subclass of safe_qipm (same cost heads); only the safety fusion differs.
    "temporal_bias_safe_qipm": "cost",
}

# THE family test. Every `algo_name in (...)` safe_qipm branch below unpacks
# this tuple, so adding a family member is one edit here instead of ~17 easily
# missed literals — the two that bite hardest are the _use_dense gate (a miss
# means configure_safety_status never runs and the safety columns are silently
# all-zero) and the runner dispatch (a miss falls through to run_online_rl,
# which writes no subtask_status labels at all). Audit: 19 lines in this file
# reference the family tuples (2 definitions, 12 full-tuple branches, 5
# flow-only-subset branches); the only bare "safe_qipm" string literal left in
# this file is the AGENTS registry key.
SAFE_QIPM_ALGOS = tuple(SAFE_QIPM_SAFETY_SIGNALS)

# Flow-policy (OGPO-line) family members only. Four branches are specific to
# the OGPO agent surface and must not see a SAC-line member: the 6-arg
# create() call (SAC create() takes 4), the CalQL phase (no calql_update on
# SAC agents), the final BC SDE eval (flow-only sampling), and mc_returns
# computation (CalQL fuel — wasted work for SAC). The prefix registration
# also splits on this: SAC-line members register step_data but not eval_sde
# (never logged for a `sac*` checkpoint name — see qipm_runner's
# startswith('sac') single-eval rule).
OGPO_SAFE_QIPM_ALGOS = tuple(a for a in SAFE_QIPM_ALGOS if not a.startswith("sac"))


def main():
    install_preemption_handler()
    algo_name, cli_overrides = parse_cli_args()
    config = load_config(algo_name, cli_overrides)

    if "agent.agent_name" in cli_overrides:
        algo_name = cli_overrides["agent.agent_name"]
    else:
        algo_name = config["agent"]["agent_name"]
    print(f"Running algorithm: {algo_name}")

    exp_config = config["experiment"]
    env_config = config["env"]
    train_config = config["training"]
    buffer_config = config["buffer"]
    eval_config = config["eval"]
    interval_config = config["intervals"]
    bc_config = config["bc"]
    success_buffer_config = config["success_buffer"]
    dataset_config = config["dataset"]
    checkpoint_config = config["checkpoint"]
    agent_config = config_to_agent_dict(config)

    # OSC side-car safety monitor (envs/osc_safety_sidecar.py): exported before
    # any env is built so make_env attaches it to the eval env and, through the
    # inherited environment, to every AsyncVectorEnv worker. A pure observer, so
    # it lives in the (unhashed) eval section.
    _sidecar_json = eval_config.get("osc_safety_monitor_config")
    if _sidecar_json:
        from envs.osc_safety_sidecar import ENV_VAR as _SIDECAR_ENV_VAR
        _sidecar_json = os.path.abspath(str(_sidecar_json))
        if not os.path.isfile(_sidecar_json):
            raise FileNotFoundError(f"eval.osc_safety_monitor_config={_sidecar_json} is not a file")
        os.environ[_SIDECAR_ENV_VAR] = _sidecar_json
        print(f"[safety] OSC side-car safety monitor enabled: {_sidecar_json}")

    # Tier A determinism, resolved here rather than beside the PRNGKey below:
    # jax_default_matmul_precision is read at TRACE time, so anywhere before the
    # first traced op is early enough, and doing it first means a run that
    # claims determinism without the launcher's XLA_FLAGS dies in its first
    # second instead of after building 32 eval envs. None = leave the platform
    # default (today's behavior).
    matmul_precision = resolve_determinism_settings(config)
    if matmul_precision is not None:
        jax.config.update("jax_default_matmul_precision", matmul_precision)
        print(f"[determinism] jax_default_matmul_precision={matmul_precision} "
              f"(experiment.deterministic={exp_config['deterministic']})")

    # experiment.repeat_id: installed before the first rolling-checkpoint path
    # is built (_find_checkpoint_dir below) so the save, load and cleanup paths
    # all agree on the suffixed scratch dir.
    set_repeat_id(exp_config["repeat_id"])

    if algo_name not in AGENTS:
        raise ValueError(
            f"Unknown agent: {algo_name}. Available: {list(AGENTS.keys())}"
        )
    agent_class = AGENTS[algo_name]

    # Rolling-checkpoint identity — NOT the registry name. Rewritten today by
    # the safe_qipm multi arm (-> safe_qipm_multihead) and by qipm/sac_qipm
    # with critic_joint_proprio (-> qipm_proprio / sac_qipm_proprio); these
    # algos dispatch to
    # qipm_runner/bc_runner, whose algo_name kwarg feeds only
    # save_rolling_checkpoint and a startswith('sac') check. Rewriting an algo
    # routed to online_rl_runner/dsrl_runner (which do set-membership checks on
    # their algo_name kwarg) would need a separate kwarg instead.
    checkpoint_algo_name = resolve_checkpoint_algo_name(algo_name, config["agent"])

    # Joint-proprio obs widening (robot0_joint_pos + robot0_joint_vel), gated
    # per family by agent.critic_joint_proprio: the joint_limits / velocity /
    # collision safety signals are functions of q/q̇, which the default
    # eef-centric keys do not contain. The safe_qipm family defaults ON
    # (safe_qipm.yaml — preserves the 67-dim contract of every existing run;
    # single/multi/hjb arms stay directly comparable); turning it off is the
    # 2026-08-07 heads-without-proprio ablation (safety critics predict from
    # state + action chunk only) and rekeys checkpoints to *_noproprio. qipm
    # defaults OFF and can opt in (2026-08-07: the "base + proprio critics"
    # control) — its critics then see the widened obs while the actor is
    # sliced back to actor_obs_dim. sac_qipm has the same opt-in
    # (2026-08-10, the SAC line's base+proprio cell). Every other algo keeps
    # the base obs.
    include_joint_proprio = (
        algo_name in SAFE_QIPM_ALGOS and config["agent"]["critic_joint_proprio"]
    ) or (algo_name == "qipm" and config["agent"]["critic_joint_proprio"]) or (
        algo_name == "sac_qipm" and config["agent"]["critic_joint_proprio"]
    )
    if include_joint_proprio:
        from envs.robomimic_utils import SAFETY_JOINT_PROPRIO_KEYS
        print(f"[{algo_name}] widening low_dim obs with joint proprioception: "
              f"{list(SAFETY_JOINT_PROPRIO_KEYS)}")

    env_name = env_config["name"]
    seed = exp_config["seed"]

    if algo_name in ("qipm", "sac_qipm") and config["agent"]["critic_joint_proprio"]:
        from envs.env_utils import is_robomimic_env as _is_robomimic
        assert _is_robomimic(env_name), (
            f"agent.critic_joint_proprio=true is robomimic-only (joint "
            f"proprio keys come from the robomimic low_dim HDF5s); got "
            f"env {env_name!r}"
        )
        assert config["agent"]["actor_obs_dim"] is not None, (
            "agent.critic_joint_proprio=true requires agent.actor_obs_dim: "
            "an unsliced actor conditioned on OSC-recorded joint dims "
            "degrades during BC under the mink eval controller "
            "(2026-07-27 post-mortem; toolhang: actor_obs_dim=53)"
        )
    ogbench_dataset_dir = dataset_config["ogbench_dir"]

    # Optional OGBench TimeLimit override. Patches the gymnasium registry once,
    # so every subsequent `gymnasium.make(env_name)` (training + 32 vec eval envs)
    # picks up the new max_episode_steps without per-call plumbing.
    _max_ep = env_config.get("max_episode_steps", None)
    if _max_ep is not None and 'singletask' in env_name:
        import gymnasium
        # Map 'scene-play-singletask-task4-v0' -> 'scene-singletask-task4-v0' (same as ogbench_utils).
        _splits = env_name.split('-')
        _pos = _splits.index('singletask')
        _gym_id = '-'.join(_splits[: _pos - 1] + _splits[_pos:])
        if _gym_id in gymnasium.envs.registry:
            print(f"[env] Overriding {_gym_id} max_episode_steps: "
                  f"{gymnasium.envs.registry[_gym_id].max_episode_steps} -> {_max_ep}")
            gymnasium.envs.registry[_gym_id].max_episode_steps = int(_max_ep)
        else:
            print(f"[env] WARNING: {_gym_id} not in gymnasium registry; max_episode_steps override skipped")

    if ogbench_dataset_dir is not None:
        assert not include_joint_proprio, (
            "joint-proprio obs widening (agent.critic_joint_proprio: "
            "default-on for the safe_qipm family, opt-in for qipm/sac_qipm) "
            "is robomimic-only; the OGBench path would silently skip it"
        )
        assert dataset_config["replace_interval"] != 0
        assert dataset_config["proportion"] == 1.0
        dataset_paths = [
            f
            for f in sorted(glob.glob(f"{ogbench_dataset_dir}/*.npz"))
            if "-val.npz" not in f
        ]
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            env_name, dataset_path=dataset_paths[0], compact_dataset=False,
            goal_conditioning_mode=agent_config.get("goal_conditioning_mode", None),
            no_of_subtasks=agent_config.get("no_of_subtasks", 4),
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
            env_name, rew_fn=env_config["rew_fn"],
            goal_conditioning_mode=agent_config.get("goal_conditioning_mode", None),
            no_of_subtasks=agent_config.get("no_of_subtasks", None),
            anchor_demo=env_config.get("anchor_demo", None),
            obstacle=env_config.get("obstacle", None),
            live_render_images=env_config.get("live_render_images", False),
            include_joint_proprio=include_joint_proprio,
        )

    del val_dataset  # Not used; free immediately to save RAM

    # safe_qipm multi-head: names of the per-constraint safety heads, in the
    # column order the env publishes them. Stays empty for every other algo and
    # for safety_head_mode="single" (whose one column is unnamed). Consumed only
    # to build logger keys — never reaches the TD math.
    safety_constraint_names = []

    # Set up subtask DAG for non-sequential task structures (e.g., Transport)
    # NOTE: subtask_dag and subtask_discounts are inert for the SAC-line
    # family member — its local gate helper has no DAG support and its critic
    # uses the scalar discount only. Neither fires on toolhang.
    if algo_name in ("qipm", *SAFE_QIPM_ALGOS) and 'transport' in env_name.lower() and agent_config.get('use_subtask_dag', True):
        agent_config['subtask_dag'] = {
            0: [], 1: [0], 2: [1], 3: [], 4: [3], 5: [2, 4], 6: [5],
        }
        if agent_config.get('subtask_discounts') is None:
            agent_config['subtask_discounts'] = [0.9, 0.99, 0.99, 0.9, 0.99, 0.99, 0.999]

    # Set up subtask DAG for LoadDishwasher (two parallel pipelines: cup and bowl)
    # Goal 0: cup_grasped, 1: cup_in_dw, 2: bowl_grasped, 3: bowl_in_dw, 4: dw_closed
    if algo_name in ("qipm", *SAFE_QIPM_ALGOS) and 'loaddishwasher' in env_name.lower() and agent_config.get('use_subtask_dag', True):
        agent_config['subtask_dag'] = {
            0: [], 1: [0], 2: [], 3: [2], 4: [1, 3],
        }
        if agent_config.get('subtask_discounts') is None:
            agent_config['subtask_discounts'] = [0.99, 0.99, 0.99, 0.99, 0.999]

    # Default subtask discounts for RoboCasa (5 sequential subtasks)
    if algo_name in ("qipm", *SAFE_QIPM_ALGOS) and env_name.startswith('robocasa_'):
        if agent_config.get('subtask_discounts') is None:
            agent_config['subtask_discounts'] = [0.9, 0.99, 0.99, 0.995, 0.999]

    # Configure dense reward on training and eval envs for QIPM / SAC-QIPM
    _use_dense = algo_name in ("qipm", *SAFE_QIPM_ALGOS, "sac_qipm", "sac_qipm_floq") and agent_config.get("use_dense_rewards", False)
    _gc_mode = agent_config.get("goal_conditioning_mode", None)
    _is_ogbench = ogbench_dataset_dir is not None
    if _use_dense or (_gc_mode is not None and _is_ogbench):
        _no_sub = agent_config.get("no_of_subtasks", 3)
        _norm_r = agent_config.get("normalize_rewards", False)
        _dag = agent_config.get("subtask_dag", None)

        # OGBench envs need to be wrapped explicitly with the dense reward wrapper
        if _is_ogbench:
            from envs.ogbench_dense_wrapper import OGBenchDenseRewardWrapper
            if not isinstance(env, OGBenchDenseRewardWrapper):
                env = OGBenchDenseRewardWrapper(env)
                eval_env = OGBenchDenseRewardWrapper(eval_env)

        for _e in [env, eval_env]:
            _inner = _e if hasattr(_e, 'configure_dense_reward') else (
                _e.unwrapped if hasattr(_e, 'unwrapped') else _e
            )
            if hasattr(_inner, 'configure_dense_reward'):
                if _is_ogbench:
                    _inner.configure_dense_reward(
                        use_dense_reward=_use_dense,
                        normalize_rewards=_norm_r,
                        no_of_subtasks=_no_sub,
                        goal_conditioning_mode=_gc_mode,
                    )
                else:
                    _inner.configure_dense_reward(
                        use_dense_reward=True,
                        normalize_rewards=_norm_r,
                        no_of_subtasks=_no_sub,
                        subtask_dag=_dag,
                    )
            else:
                raise RuntimeError(f"{type(_inner)} does not support configure_dense_reward")

        # safe_qipm family: the TRAIN env additionally appends the per-step
        # safety signal as trailing subtask_status columns — costs for
        # safe_qipm, margins for safe_qipm_hjb (eval envs stay task-only in v1
        # — eval consumes task heads only).
        if algo_name in SAFE_QIPM_ALGOS:
            _train_inner = env if hasattr(env, 'configure_safety_status') else (
                env.unwrapped if hasattr(env, 'unwrapped') else env
            )
            _safety_signal = SAFE_QIPM_SAFETY_SIGNALS[algo_name]
            _train_inner.configure_safety_status(
                agent_config["no_of_safety_subtasks"],
                safety_head_mode=agent_config["safety_head_mode"],
                safety_signal=_safety_signal,
            )
            safety_constraint_names = list(
                getattr(_train_inner, '_safety_constraint_names', [])
            )
            # Record which head is which in the run log — the mapping is
            # otherwise only recoverable from the controller JSON.
            print(f"[{algo_name}] safety_head_mode={agent_config['safety_head_mode']} "
                  f"signal={_safety_signal} "
                  f"K={agent_config['no_of_safety_subtasks']} "
                  f"constraints={safety_constraint_names or '<aggregate unsafe_step>'}")

    n_eval_envs = eval_config["n_eval_envs"]
    eval_envs = None
    if n_eval_envs > 1:
        # SAC-QIPM uses dense rewards via OGBench wrapper; classic QIPM uses robomimic dense path
        use_dense = algo_name in ("qipm", *SAFE_QIPM_ALGOS, "sac_qipm", "sac_qipm_floq") and agent_config.get("use_dense_rewards", False)
        # Vectorized eval envs need wrapping when dense reward is on, OR when OGBench GC is active
        need_wrapper = use_dense or (_gc_mode is not None and _is_ogbench)
        eval_envs = make_vectorized_env(
            env_name,
            num_envs=n_eval_envs,
            seed=seed + 1000,
            rew_fn=env_config["rew_fn"],
            dense_reward_wrapper=need_wrapper,
            use_dense_reward=use_dense,
            no_of_subtasks=agent_config.get("no_of_subtasks", 3),
            normalize_rewards=agent_config.get("normalize_rewards", False),
            subtask_dag=agent_config.get("subtask_dag", None),
            goal_conditioning_mode=agent_config.get("goal_conditioning_mode", None),
            anchor_demo=env_config.get("anchor_demo", None),
            obstacle=env_config.get("obstacle", None),
            include_joint_proprio=include_joint_proprio,
        )

    # A failure anywhere below (the resume guard, a shape mismatch, a bad
    # config) used to leave the AsyncVectorEnv workers running: the interpreter
    # then blocked at shutdown and the job sat on its GPU until the wall clock
    # ran out, which is how job 10504857 burned 4 hours after refusing to start.
    # atexit runs before that join, so closing here bounds a failed start to
    # seconds.
    if eval_envs is not None:
        import atexit

        def _close_eval_envs():
            try:
                eval_envs.close()
            except Exception as exc:  # teardown must not mask the real error
                print(f"[cleanup] closing eval envs failed: {exc}")

        atexit.register(_close_eval_envs)

    # Dedicated renderable env for success/failure video capture. The eval_envs
    # workers are built without an offscreen renderer (one EGL context per worker
    # would be paid on every eval, for every algo), so they cannot render.
    # include_joint_proprio must match the eval envs or the obs layout won't match
    # the actor.
    video_interval = eval_config["video_interval"]
    video_max_attempts = eval_config["video_max_attempts"]
    video_env = None
    if video_interval > 0:
        from envs.env_utils import is_robomimic_env as _is_robomimic_env
        if _is_robomimic_env(env_name):
            from envs.robomimic_utils import make_env as make_robomimic_env
            video_env = make_robomimic_env(
                env_name, seed=seed + 5000,
                goal_conditioning_mode=agent_config.get("goal_conditioning_mode", None),
                no_of_subtasks=agent_config.get("no_of_subtasks", 3),
                include_joint_proprio=include_joint_proprio,
                renderable=True,
            )
            print(f"Video capture enabled: every {video_interval} steps "
                  f"(camera={video_env.render_camera_name}, max_attempts={video_max_attempts})")
        else:
            print(f"eval.video_interval={video_interval} set but '{env_name}' is not "
                  f"a robomimic env; video capture disabled")

    random.seed(seed)
    np.random.seed(seed)
    dataset_rng = np.random.default_rng(seed + 20000)
    online_rng, rng = jax.random.split(jax.random.PRNGKey(seed), 2)

    horizon_length = agent_config["horizon_length"]
    discount = agent_config["discount"]
    best_of_n = agent_config["best_of_n"]

    q_min, q_max = get_q_bounds(env_name, discount)
    agent_config["q_min"] = q_min
    agent_config["q_max"] = q_max
    print(f"Q bounds: [{q_min}, {q_max}]")

    needs_mc_returns = algo_name in [
        "ogpo",
        "reinflow_calql",
        "bptt",
        "ogpo_bptt",
        "expo",
        "qipm",
        *OGPO_SAFE_QIPM_ALGOS,
    ]
    train_dataset = process_train_dataset(
        train_dataset,
        env_name,
        dataset_proportion=dataset_config["proportion"],
        sparse=env_config["sparse"],
        numpy_rng=dataset_rng,
        compute_mc_returns=needs_mc_returns,
        discount=discount,
    )
    train_dataset.p_aug = dataset_config["p_aug"]

    # Optional: replace train_dataset with a BC dataset loaded from npz.
    # Used for BC-pretraining sac_qipm_avoid_obstacle from a vanilla SAC checkpoint.
    bc_dataset_path = dataset_config.get("bc_dataset_path")
    if bc_dataset_path:
        print(f"[BC dataset] loading custom BC dataset from {bc_dataset_path}")
        from ogpo.utils.datasets import Dataset
        bc_npz = np.load(bc_dataset_path)
        bc_dict = {k: bc_npz[k] for k in bc_npz.files}
        train_dataset = Dataset.create(**bc_dict, numpy_rng=dataset_rng)
        train_dataset.p_aug = dataset_config["p_aug"]
        print(
            f"[BC dataset] loaded {len(bc_dict['observations'])} transitions, "
            f"keys = {list(bc_dict.keys())}"
        )

    example_batch = train_dataset.sample(())
    action_dim = example_batch["actions"].shape[-1]
    ex_images = example_batch.get("images")

    if algo_name in ["ogpo", "reinflow_calql", "qipm", *OGPO_SAFE_QIPM_ALGOS]:
        agent = agent_class.create(
            seed,
            example_batch["observations"],
            example_batch["actions"],
            agent_config,
            agent_config["adv_clip_min"],
            ex_images=ex_images,
        )
    else:
        agent = agent_class.create(
            seed, example_batch["observations"], example_batch["actions"], agent_config
        )

    prefixes = ["eval", "env", "offline_agent", "memory"]
    online_steps = train_config["online_steps"]
    bc_only = train_config["bc_only"]
    if online_steps > 0 and not bc_only:
        prefixes.append("online_agent")
    if algo_name in ["ogpo", "reinflow_calql", "qipm", *OGPO_SAFE_QIPM_ALGOS]:
        prefixes.extend(["eval_sde", "step_data"])
    elif algo_name in SAFE_QIPM_ALGOS:
        # SAC-line family members: per-step safety telemetry only. Without
        # this the runner's step_data logging is silently swallowed
        # (qipm_runner wraps logger.log in a best-effort try/except and
        # LoggingHelper asserts the prefix is registered). eval_sde is never
        # logged for a `sac*` checkpoint name, so it is deliberately not
        # registered here.
        prefixes.append("step_data")
    # The periodic BC-phase SDE eval (eval.bc_sde_eval) logs under eval_sde for
    # any algo; LoggingHelper asserts every prefix is registered.
    if eval_config.get("bc_sde_eval", False) and "eval_sde" not in prefixes:
        prefixes.append("eval_sde")

    # Run provenance (Tier C). Injected into the TOP-LEVEL config — not
    # config['agent'], which is frozen after create() — and injected HERE, before
    # setup_experiment_logging writes flags.json / the wandb config and before
    # compute_config_hash below, so one block covers all three. Without it
    # neither flags.json nor wandb records which composite controller supplied
    # the safety signal or which traj-count HDF5 backed `tool_hang-ph-low_dim`.
    from envs.env_utils import is_robomimic_env as _is_robomimic_for_provenance
    config["provenance"] = collect_run_provenance(
        env_name,
        # OGBench runs load .npz from dataset.ogbench_dir, so even a
        # robomimic-looking name has no HDF5 identity on that path.
        is_robomimic=(ogbench_dataset_dir is None
                      and _is_robomimic_for_provenance(env_name)),
    )
    print(f"[provenance] {config['provenance']}")

    run_name = exp_config["run_name"]

    # Resume guard: one digest of the training-relevant config, written beside
    # every rolling checkpoint below and compared against the checkpoint's on
    # load. Computed once here so both sides of a resume hash the same dict
    # (config itself is never mutated by main(); config_to_agent_dict copies).
    config_hash = compute_config_hash(config)
    # Keys the rolling-checkpoint DIRECTORY too (see _CONFIG_HASH), so it must
    # be installed before load_rolling_checkpoint builds the first path.
    set_checkpoint_config_hash(config_hash)

    # Checkpoint loading
    is_resumed = False
    resumed_step = 0
    resumed_online_step = 0
    resumed_episode_counter = 0
    resumed_buffer = None
    resumed_rng_states = None
    resumed_wandb_id = None
    resumed_eval_rng = None
    fresh_agent = agent
    try:
        (
            resumed_agent,
            resumed_buffer,
            resumed_step,
            resumed_online_step,
            resumed_wandb_id,
            resumed_rng_states,
            resumed_episode_counter,
        ) = load_rolling_checkpoint(
            seed, env_name, agent, checkpoint_algo_name,
            config_hash=config_hash,
            config_hash_mode=checkpoint_config["config_hash_mode"],
        )
        if algo_name in SAFE_QIPM_ALGOS:
            validate_resumed_buffer_width(
                resumed_buffer,
                agent_config["no_of_subtasks"] + agent_config["no_of_safety_subtasks"],
            )
        if resumed_agent is not None:
            agent = resumed_agent
            is_resumed = True
            print(f"Resumed from step {resumed_step}")
            # Restore global RNG states if available
            if resumed_rng_states is not None:
                from ogpo.utils.main_helper import _restore_rng_states
                # dataset_rng is restored IN PLACE — train_dataset already holds
                # this exact Generator (handed to process_train_dataset above),
                # and the checkpoint load necessarily happens after that, so
                # rebinding the name here would restore nothing the dataset
                # actually samples from.
                # The eval key is part of the restored state: without it a BC
                # resume restarts the eval key sequence, so the resumed run
                # evaluates on different episodes than an uninterrupted one.
                _, resumed_eval_rng = _restore_rng_states(resumed_rng_states, dataset_rng)
                print("Restored RNG states from checkpoint")
        else:
            print(f"No rolling checkpoint found for seed={seed} env={env_name} "
                  f"algo={checkpoint_algo_name}; starting fresh")
    except Exception as e:
        # Checkpoints EXISTED for this key but every one was refused (config
        # hash mismatch, schema, incomplete). Starting fresh here is not a
        # recovery: the fresh run's first rolling save deletes the refused
        # checkpoint, and its params_<step>.pkl overwrite the original run's
        # under the same name. That is how the no_proprio l40 seed-7 680k
        # checkpoint was lost (2026-09-10). Fail instead, so a key collision or
        # config drift is visible before anything is destroyed.
        #
        # The same holds for ANY failure here: finding no checkpoint is not an
        # exception (load_rolling_checkpoint returns all-None), so an exception
        # means a checkpoint existed and could not be used — an unpickle or
        # pytree mismatch, the buffer-width guard, the RNG restore. Starting
        # fresh would delete that checkpoint on the first save and open a
        # second wandb run for the same experiment. Only a deliberate opt-out
        # (checkpoint.fresh_on_load_error=true) keeps the old fallback.
        if ("refused every rolling checkpoint candidate" in str(e)
                or not checkpoint_config.get("fresh_on_load_error", False)):
            raise
        print(f"Could not load checkpoint: {e}")
        # Restore the full pre-try state: raises can occur after the unpack
        # (width guard) or after agent adoption (RNG restore), so no partial
        # checkpoint state may leak into the nominally-fresh run.
        agent = fresh_agent
        is_resumed = False
        resumed_step = 0
        resumed_online_step = 0
        resumed_episode_counter = 0
        resumed_buffer = None
        resumed_rng_states = None
        resumed_wandb_id = None
        resumed_eval_rng = None

    # Outside the try above on purpose: this refusal must not be turned into
    # a fresh start.
    assert_requeue_resumed(is_resumed, seed, env_name, checkpoint_algo_name)

    # wandb init runs HERE, after the checkpoint load, so that a resumed run can
    # continue its ORIGINAL wandb run rather than opening a new one. It used to
    # run before the load, which left no way to tell a genuine resume from a
    # fresh start — so every attempt started a new run, and on a requeue
    # partition one experiment arrived as a dozen disjoint fragments sharing a
    # group. Passing the id only when is_resumed is true preserves the property
    # that made the unconditional-fresh rule right in the first place: an id is
    # never reused unless the step series actually continues from where that id
    # left off. resumed_wandb_id comes from the loaded checkpoint's own meta, so
    # it always names the run whose step the training loop is about to resume at.
    save_dir, logger, wandb_run_id = setup_experiment_logging(
        seed=seed,
        project=exp_config["project"],
        run_group=exp_config["run_group"],
        wandb_name=exp_config["wandb_name"],
        save_dir=exp_config["save_dir"],
        env_name=env_name,
        checkpoint_algo_name=checkpoint_algo_name,
        log_enabled=exp_config["log"],
        prefixes=prefixes,
        checkpoint_dir_fn=partial(_find_checkpoint_dir, algo_name=checkpoint_algo_name),
        config=config,
        repeat_id=exp_config["repeat_id"],
        resume_wandb_run_id=resumed_wandb_id if is_resumed else None,
    )

    # Register experiment in registry
    try:
        from scripts.experiment_registry import register_experiment
        register_experiment(config, wandb_run_id)
    except Exception as e:
        print(f"[Registry] Could not register experiment: {e}")

    # Restore pre-trained weights
    if not is_resumed:
        restore_actor_path = checkpoint_config["restore_actor_path"]
        restore_critic_path = checkpoint_config["restore_critic_path"]
        if restore_actor_path and restore_actor_path != "None":
            agent.restore_actor_params(restore_actor_path)
            print(f"Restored actor from {restore_actor_path}")
        if restore_critic_path and restore_critic_path != "None":
            agent.restore_critic_params(restore_critic_path)
            print(f"Restored critic from {restore_critic_path}")
        restore_safety_path = checkpoint_config.get("restore_safety_critic_path")
        if restore_safety_path and restore_safety_path != "None":
            from ogpo.agents.modules.maintenance import restore_safety_critic_params
            restore_safety_critic_params(agent.critic_network, restore_safety_path)

    # BC Training
    ep_resume = checkpoint_config["ep_resume"]
    log_step = ep_resume if not is_resumed else resumed_step
    offline_steps = train_config["offline_steps"]

    # Determine if we need to run (or resume) BC
    bc_resumed = is_resumed and resumed_online_step == 0
    if bc_resumed:
        bc_steps_done = resumed_step - ep_resume
        bc_remaining = offline_steps - bc_steps_done
    else:
        bc_remaining = offline_steps

    if bc_remaining > 0 and (not is_resumed or bc_resumed):
        if bc_resumed:
            print(f"\n{'=' * 60}\nResuming BC Training ({bc_steps_done} done, {bc_remaining} remaining)\n{'=' * 60}")
        else:
            print("\n" + "=" * 60 + "\nBC Training Phase\n" + "=" * 60)

        # Parsed here (JSON string or mapping) so a malformed eval.safety_views
        # fails before training rather than at the first eval.
        from ogpo.utils.eval_safety import parse_safety_views
        _bc_safety_views = parse_safety_views(eval_config.get("safety_views"))

        # For QIPM, use QIPM eval to get per-subtask success rates
        qipm_bc_eval_fn = None
        if algo_name in ("qipm", *SAFE_QIPM_ALGOS, "sac_qipm", "sac_qipm_avoid_obstacle", "sac_qipm_floq"):
            # safe_qipm: eval receives the TASK count only — per-subtask eval
            # metrics cover task heads; the safety head is excluded in v1.
            _no_of_subtasks = agent_config.get("no_of_subtasks", 3)
            def qipm_bc_eval_fn(agent, eval_env, eval_envs, eval_flags,
                                action_dim, actor_fn, log_step, eval_episodes, run_name,
                                eval_rng=None):
                return _run_qipm_evaluation(
                    agent, eval_env, eval_envs, eval_flags, action_dim,
                    actor_fn, log_step, eval_episodes, run_name,
                    no_of_subtasks=_no_of_subtasks, discount=discount, plot=False,
                    eval_rng=eval_rng, safety_views=_bc_safety_views,
                )

        agent, log_step, eval_info = run_bc_training(
            agent=agent,
            resumed_eval_rng=resumed_eval_rng if bc_resumed else None,
            train_dataset=train_dataset,
            eval_env=eval_env,
            eval_envs=eval_envs,
            config=agent_config,
            logger=logger,
            save_dir=save_dir,
            offline_steps=bc_remaining,
            horizon_length=horizon_length,
            discount=discount,
            batch_size=agent_config["batch_size"],
            dataset_rng=dataset_rng,
            config_hash=config_hash,
            n_eval_envs=n_eval_envs,
            eval_episodes=eval_config["eval_episodes"],
            eval_interval_bc=interval_config["eval_interval_bc"],
            video_episodes=eval_config["video_episodes"],
            video_frame_skip=eval_config["video_frame_skip"],
            video_env=video_env,
            video_interval=video_interval,
            video_max_attempts=video_max_attempts,
            plot_q_vs_mc=eval_config["plot_q_vs_mc"],
            run_name=run_name,
            log_interval=interval_config["log_interval"],
            save_interval_bc=interval_config["save_interval_bc"],
            # BC-specific cadence when set; the shared one otherwise. A BC
            # checkpoint is agent-only (no replay buffer), so it is cheap enough
            # to run far more often than the online one — which matters on a
            # preemptible partition, where a job that never survives long enough
            # to reach the interval restarts from step 0 every time.
            preemption_backup_interval=(
                interval_config.get("bc_preemption_backup_interval")
                or interval_config["preemption_backup_interval"]
            ),
            step_data_log_interval=interval_config["step_data_log_interval"],
            clip_bc=bc_config["clip_bc"],
            clip_bc_threshold=bc_config["clip_bc_threshold"],
            env_name=env_name,
            start_step=log_step,
            custom_eval_fn=qipm_bc_eval_fn,
            seed=seed,
            algo_name=checkpoint_algo_name,
            wandb_run_id=wandb_run_id,
            bc_sde_eval=bool(eval_config.get("bc_sde_eval", False)),
            save_actor_only=bool(bc_config.get("save_actor_only", False)),
            safety_views=_bc_safety_views,
        )
        print(
            f"BC Training Complete. Final success rate: {eval_info.get('success', 'N/A')}"
        )

    # CalQL Phase
    calql_steps = train_config["calql_steps"]
    if (not is_resumed or bc_resumed) and calql_steps > 0 and algo_name in ["ogpo", "reinflow_calql", "qipm", *OGPO_SAFE_QIPM_ALGOS]:
        print("\n" + "=" * 60 + "\nCalQL Training Phase\n" + "=" * 60)
        agent, log_step, eval_info = run_calql_training(
            agent=agent,
            train_dataset=train_dataset,
            eval_env=eval_env,
            eval_envs=eval_envs,
            config=agent_config,
            logger=logger,
            save_dir=save_dir,
            calql_steps=calql_steps,
            horizon_length=horizon_length,
            discount=discount,
            batch_size=agent_config["batch_size"],
            n_eval_envs=n_eval_envs,
            eval_episodes=eval_config["eval_episodes"],
            eval_interval_bc=interval_config["eval_interval_bc"],
            video_episodes=eval_config["video_episodes"],
            video_frame_skip=eval_config["video_frame_skip"],
            video_env=video_env,
            video_interval=video_interval,
            video_max_attempts=video_max_attempts,
            plot_q_vs_mc=eval_config["plot_q_vs_mc"],
            run_name=run_name,
            log_interval=interval_config["log_interval"],
            step_data_log_interval=interval_config["step_data_log_interval"],
            env_name=env_name,
            start_step=log_step,
            seed=seed,
        )

    # Final BC evaluation (SDE) for OGPO. Same gate as the CalQL block: it
    # belongs to the BC->online transition, so an online resume skips it (it
    # used to re-evaluate the ONLINE agent on every requeue and log that as a
    # BC eval at a stale step).
    if (algo_name in ["ogpo", "reinflow_calql", "qipm", *OGPO_SAFE_QIPM_ALGOS] and not bc_only
            and (not is_resumed or bc_resumed)):
        print("\nEvaluating with Stochastic Flow (SDE)...")
        eval_flags = _build_eval_flags(
            n_eval_envs, eval_config["eval_episodes"], discount, run_name, env_name
        )
        try:
            eval_info = _run_evaluation(
                agent,
                eval_env,
                eval_envs,
                eval_flags,
                action_dim,
                None,
                log_step,
                eval_config["eval_episodes"],
                eval_config["video_episodes"],
                eval_config["video_frame_skip"],
                run_name,
                plot=False,
            )
            print(f"SDE Success Rate: {eval_info.get('success', 'N/A')}")
            try:
                logger.log(eval_info, "eval_sde", step=log_step)
            except Exception:
                pass
        except PreemptionDuringEval as exc:
            # Exit here rather than falling through: between this point and the
            # online loop's first preemption poll the run builds the replay
            # buffer, resets the eval envs and JIT-compiles the first update -
            # minutes, against KillWait=120s, so the emergency checkpoint would
            # never be written. Nothing has been trained since the BC phase's
            # own checkpoint, so exiting 0 loses no work and the requeue
            # resumes from it.
            print(f"[PREEMPTION] {exc}")
            print("[PREEMPTION] Abandoning the final BC eval and exiting for requeue.")
            logger.close()
            import sys
            sys.exit(0)

    if bc_only or online_steps <= 0:
        print("\nBC training complete. Skipping online RL phase.")
        save_agent(agent, save_dir, log_step)
        logger.close()
        return

    print("\n" + "=" * 60 + "\nOnline RL Phase\n" + "=" * 60)

    # Common kwargs shared by both runners
    runner_kwargs = dict(
        agent=agent,
        env=env,
        eval_env=eval_env,
        eval_envs=eval_envs,
        train_dataset=train_dataset,
        config=agent_config,
        logger=logger,
        save_dir=save_dir,
        algo_name=checkpoint_algo_name,
        online_steps=online_steps,
        start_training=buffer_config["start_training"],
        batch_size=agent_config["batch_size"],
        horizon_length=horizon_length,
        discount=discount,
        dataset_rng=dataset_rng,
        config_hash=config_hash,
        utd_warmup=buffer_config["utd_warmup"],
        utd_online=buffer_config["utd_online"],
        utd_q=buffer_config.get("utd_q", None),
        utd_pi=buffer_config.get("utd_pi", None),
        offline_ratio=buffer_config["offline_ratio"],
        warmup_offline_ratio=buffer_config.get("warmup_offline_ratio", 0.0),
        bc_refine_steps=train_config["bc_refine_steps"],
        q_warmup_steps=train_config["q_warmup_steps"],
        use_success_buffer=success_buffer_config["use_success_buffer"],
        success_buffer_batch_size=success_buffer_config["batch_size"],
        best_of_n=best_of_n,
        eval_interval=interval_config["eval_interval"],
        n_eval_envs=n_eval_envs,
        eval_episodes=eval_config["eval_episodes"],
        video_episodes=eval_config["video_episodes"],
        video_frame_skip=eval_config["video_frame_skip"],
        plot_q_vs_mc=eval_config["plot_q_vs_mc"],
        run_name=run_name,
        env_name=env_name,
        log_interval=interval_config["log_interval"],
        step_data_log_interval=interval_config["step_data_log_interval"],
        save_interval=interval_config["save_interval"],
        preemption_backup_interval=interval_config["preemption_backup_interval"],
        log_enabled=exp_config["log"],
        sparse=env_config["sparse"],
        p_aug=dataset_config["p_aug"],
        buffer_size=buffer_config["size"],
        seed=seed,
        is_resumed=is_resumed,
        resumed_online_step=resumed_online_step,
        resumed_episode_counter=resumed_episode_counter if is_resumed else 0,
        resumed_replay_buffer=resumed_buffer if is_resumed else None,
        start_step=log_step,
        wandb_run_id=wandb_run_id,
        resumed_rng_states=resumed_rng_states,
        progress_rewards=bool(config.get("robometer", {}).get("progress_rewards", False)),
        robometer_server_url=config.get("robometer", {}).get("server_url", None),
        robometer_url_file=config.get("robometer", {}).get("url_file", None),
        robometer_wait_s=float(config.get("robometer", {}).get("wait_s", 1800.0)),
        robometer_max_frames=int(config.get("robometer", {}).get("max_frames", 8)),
        robometer_use_pbrs=bool(config.get("robometer", {}).get("use_pbrs", False)),
    )

    if algo_name in DSRL_ALGORITHMS:
        agent, log_step, final_eval_info, replay_buffer = run_dsrl_online_rl(
            **runner_kwargs
        )
    elif algo_name in ("qipm", *SAFE_QIPM_ALGOS, "sac_qipm", "sac_qipm_avoid_obstacle", "sac_qipm_floq"):
        if algo_name in SAFE_QIPM_ALGOS:
            # Offline Datasets carry task-only subtask_status (no safety
            # columns); mixing them with the widened online buffer is
            # unsupported until offline safety relabeling lands.
            assert float(buffer_config.get("offline_ratio", 0.0) or 0.0) == 0.0, \
                f"{algo_name} does not support offline_ratio > 0"
            assert float(buffer_config.get("warmup_offline_ratio", 0.0) or 0.0) == 0.0, \
                f"{algo_name} does not support warmup_offline_ratio > 0"
        qipm_kwargs = dict(
            # Video params ride on qipm_kwargs, not runner_kwargs: runner_kwargs is
            # shared with run_dsrl_online_rl and run_online_rl, which do not accept
            # them.
            video_env=video_env,
            video_interval=video_interval,
            video_max_attempts=video_max_attempts,
            use_dense_rewards=agent_config.get("use_dense_rewards", False),
            normalize_rewards=agent_config.get("normalize_rewards", False),
            no_of_subtasks=agent_config.get("no_of_subtasks", 3),
            no_of_safety_subtasks=agent_config.get("no_of_safety_subtasks", 0),
            safety_constraint_names=safety_constraint_names,
            # .get default: this branch is shared with qipm/sac_qipm*, which
            # have no safety columns at all (no_of_safety_subtasks=0 gates every
            # consumer), so "cost" is an inert placeholder for them.
            safety_signal=SAFE_QIPM_SAFETY_SIGNALS.get(algo_name, "cost"),
        )
        # Periodic obstacle visualization for sac_qipm_avoid_obstacle:
        # render one greedy episode per eval cycle and save an obstacle-overlaid GIF.
        if (
            algo_name == "sac_qipm_avoid_obstacle"
            and (env_config.get("obstacle") or {}).get("enabled", False)
        ):
            from envs.pusht_utils import make_pusht_obstacle_video
            _viz_root = "/home/ssaxena2/OGPO/paper_experiments/pushT/videos"
            _viz_run_subdir = f"{exp_config.get('run_name', 'run')}_seed{exp_config.get('seed', 0)}"
            _viz_run_dir = os.path.join(_viz_root, _viz_run_subdir)
            _viz_seed = int((env_config.get("obstacle") or {}).get("viz_seed", 12345))

            def _viz_callback(_agent, _eval_env, _step, _save_dir):
                path = os.path.join(_viz_run_dir, f"eval_step_{_step}.gif")
                try:
                    make_pusht_obstacle_video(_agent, _eval_env, path, seed=_viz_seed)
                    print(f"[viz] saved {path}")
                except Exception as _e:
                    print(f"[viz] failed at step {_step}: {_e}")

            qipm_kwargs["viz_callback"] = _viz_callback

        agent, log_step, final_eval_info, replay_buffer = run_qipm_online_rl(
            **runner_kwargs, **qipm_kwargs
        )
    else:
        agent, log_step, final_eval_info, replay_buffer = run_online_rl(**runner_kwargs)

    print(
        f"\nOnline RL complete. Final success rate: {final_eval_info.get('success', 'N/A')}"
    )

    # Update experiment registry with final results
    try:
        from scripts.experiment_registry import update_experiment_results
        update_experiment_results(wandb_run_id, final_eval_info, final_step=log_step)
    except Exception as e:
        print(f"[Registry] Could not update results: {e}")

    # Clean up rolling checkpoint from /scratch after successful completion
    cleanup_checkpoint(seed, env_name, checkpoint_algo_name)

    logger.close()


if __name__ == '__main__':
    main()
