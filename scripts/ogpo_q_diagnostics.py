"""OGPO's own Q-diagnostics figure (Q vs MC, value-drop consistency, UMAP of action chunks coloured by Q)
for a trained OGPO_public run, from its final checkpoint, plus the raw data for scripts/plot_ogpo_q_umap.py.

It is what OGPO_public plots during training when eval.plot_q_vs_mc is on: ogpo.utils.evaluation.
evaluate_parallel(plot=True) rolls out the policy, scores every decision's action chunk with the target
critic (q_agg over the ensemble), and calls visualize_q_diagnostics. Here the agent is rebuilt exactly as
ogpo/main.py builds it, from the run's own flags.json, and the final checkpoint's weights are loaded
(the replay buffer is not). visualize_q_diagnostics is wrapped to also save its inputs to
<out_dir>/<run>_<sampler>.npz, with each episode's success from the env info.

Run with OGPO_public's venv and PYTHONPATH (scripts/ogpo_q_diagnostics.sbatch does this):
    PYTHONPATH=$OGPO_PUBLIC $OGPO_PUBLIC/.venv/bin/python scripts/ogpo_q_diagnostics.py \
        --run ogpo_public_square_mh_20261022_seed0 --out_dir plots/ogpo_public/q_diagnostics
Samplers: ode (default) is the plain flow policy, the eval OGPO plots in training; sde is the deployed
best-of-N policy (agent.sample_actions_BON, critic-ranked).
"""
import argparse
import glob
import json
import os
import pickle
import re
from types import SimpleNamespace

import numpy as np

ROOT = "/data/user_data/asingal2/memory_critic/ogpo_public"


def dataset_root(run):
    m = re.match(r"ogpo_public_square_(ph|mh)(?:_(\d+)demos)?_", run)
    assert m, f"can't parse dataset from run name {run}"
    ds, n = m.groups()
    if n is None:
        return "/data/user_data/asingal2/robomimic"
    return f"/data/user_data/asingal2/robomimic_subsets/square_{ds}_{n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, i.e. the directory under <root>/ckpt")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--sampler", choices=("ode", "sde"), default="ode")
    ap.add_argument("--n_eval_envs", type=int, default=8)
    ap.add_argument("--eval_episodes", type=int, default=8, help="rollouts per env (total = n_eval_envs x this)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault("ROBOMIMIC_DATASET_ROOT", dataset_root(args.run))
    os.environ["OGPO_CHECKPOINT_ROOT"] = os.path.join(args.root, "ckpt", args.run)

    import flax
    import jax
    from envs.env_utils import make_env_and_datasets
    from ogpo.agents.ogpo import OGPOAgent
    from ogpo.utils import evaluation
    from ogpo.utils.config_loader import config_to_agent_dict, load_config
    from ogpo.utils.main_helper import (get_checkpoint_dir, get_q_bounds, make_vectorized_env,
                                        process_train_dataset)

    # --- the run's config, through the same loader main.py uses
    flags = glob.glob(os.path.join(args.root, "exp", "**", f"{args.run}seed*", "flags.json"), recursive=True)
    assert len(flags) == 1, f"expected one flags.json for {args.run}, found {flags}"
    with open(flags[0]) as f:
        saved = json.load(f)
    overrides = {f"{sec}.{k}": v for sec, d in saved.items() if isinstance(d, dict) for k, v in d.items()}
    config = load_config("ogpo", overrides)
    assert config["agent"]["agent_name"] == "ogpo", config["agent"]["agent_name"]
    env_config, dataset_config = config["env"], config["dataset"]
    env_name, seed = env_config["name"], config["experiment"]["seed"]
    agent_config = config_to_agent_dict(config)
    discount = agent_config["discount"]
    agent_config["q_min"], agent_config["q_max"] = get_q_bounds(env_name, discount)
    agent_config["_original_actor_obs"] = agent_config.get("actor_obs", "state")
    agent_config["_original_critic_obs"] = agent_config.get("critic_obs", "state")
    print(f"run {args.run}: env {env_name} seed {seed}; flags {flags[0]}")
    print(f"dataset root {os.environ['ROBOMIMIC_DATASET_ROOT']}")

    # --- the agent template, as ogpo/main.py builds it
    _, _, train_dataset, val_dataset = make_env_and_datasets(
        env_name, rew_fn=env_config["rew_fn"], post_success_steps=env_config.get("post_success_steps", 0))
    del val_dataset
    train_dataset = process_train_dataset(
        train_dataset, env_name, dataset_proportion=dataset_config["proportion"],
        num_offline_trajs=dataset_config.get("num_offline_trajs", -1), sparse=env_config["sparse"],
        numpy_rng=np.random.default_rng(seed + 20000), compute_mc_returns=True, discount=discount)
    example = train_dataset.sample(())
    action_dim = example["actions"].shape[-1]
    agent = OGPOAgent.create(seed, example["observations"], example["actions"], agent_config,
                             agent_config["adv_clip_min"], ex_images=example.get("images"),
                             ex_full_states=example.get("full_states"))
    del train_dataset

    # --- final checkpoint's weights (as load_rolling_checkpoint, without the replay buffer)
    ckpt_dir = get_checkpoint_dir(seed, env_name, "ogpo")
    steps = [int(os.path.basename(p).split("_")[0]) for p in glob.glob(os.path.join(ckpt_dir, "*_meta.json"))]
    steps = [s for s in steps if os.path.exists(os.path.join(ckpt_dir, f"{s}_agent.pkl"))]
    assert steps, f"no complete checkpoint in {ckpt_dir}"
    step = max(steps)
    with open(os.path.join(ckpt_dir, f"{step}_meta.json")) as f:
        online_loop_step = json.load(f).get("online_loop_step", 0)
    if online_loop_step > 0 and hasattr(agent, "reset_optimizers_with_lr"):
        agent = agent.reset_optimizers_with_lr()
    with open(os.path.join(ckpt_dir, f"{step}_agent.pkl"), "rb") as f:
        agent = flax.serialization.from_state_dict(agent, pickle.load(f))
    print(f"loaded {ckpt_dir}/{step}_agent.pkl (online loop step {online_loop_step})")

    # --- capture what visualize_q_diagnostics gets, plus each episode's success, in episode order
    tag = f"{args.run}_{args.sampler}"
    successes = []
    add_to = evaluation.add_to

    def add_to_recording(stats, info):
        successes.append(float(info.get("success", np.nan)))
        return add_to(stats, info)

    original_plot = evaluation.visualize_q_diagnostics

    def plot_and_save(scatter_data, episode_q_mc_sequences, actions_data, global_step, suffix, dir_suffix,
                      save_dir="./plots"):
        q = np.array([d[0] for d in scatter_data], dtype=np.float32)
        mc = np.array([d[1] for d in scatter_data], dtype=np.float32)
        lens = [len(s[0]) for s in episode_q_mc_sequences]
        episode = np.repeat(np.arange(len(lens)), lens)
        decision = np.concatenate([np.arange(k) for k in lens]) if lens else np.zeros(0, int)
        np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), q=q, mc=mc, actions=np.array(actions_data, np.float32),
                            episode=episode, decision=decision, episode_success=np.array(successes, np.float32),
                            run=args.run, sampler=args.sampler, ckpt_step=step, horizon=agent_config["horizon_length"],
                            action_dim=action_dim, discount=discount, q_agg=agent_config.get("q_agg", "mean"))
        original_plot(scatter_data, episode_q_mc_sequences, actions_data, global_step, args.sampler.upper(),
                      dir_suffix=args.run, save_dir=out_dir)

    evaluation.add_to = add_to_recording
    evaluation.visualize_q_diagnostics = plot_and_save

    eval_envs = make_vectorized_env(env_name, num_envs=args.n_eval_envs, seed=seed + 1000,
                                    rew_fn=env_config["rew_fn"],
                                    post_success_steps=env_config.get("post_success_steps", 0))
    eval_flags = SimpleNamespace(n_eval_envs=args.n_eval_envs, eval_episodes=args.eval_episodes, discount=discount,
                                 run_name=args.run, env_name=env_name)
    if args.sampler == "ode":
        actor_fn = getattr(agent, "compute_flow_actions_corrected", agent.compute_flow_actions)
    else:
        actor_fn = agent.sample_actions_BON if agent_config["best_of_n"] > 1 else agent.sample_actions
    info = evaluation.evaluate_parallel(eval_envs, agent, eval_flags, action_dim, step, actor_fn=actor_fn, plot=True)
    eval_envs.close()
    summary = {"run": args.run, "sampler": args.sampler, "ckpt_step": step, "episodes": len(successes),
               "success": float(np.nanmean(successes)) if successes else None,
               "eval_info": {k: float(v) for k, v in info.items() if np.ndim(v) == 0}}
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("run", "sampler", "ckpt_step", "episodes", "success")}))


if __name__ == "__main__":
    main()
