"""CPU evaluation of a finished BC-only training run, logged into the SAME W&B run.

run_scripts/train_bc_full.sh trains with no evals (intervals.eval_interval_bc=-1) and saves
an actor-only checkpoint actor_<step>.pkl every 10000 steps (bc.save_actor_only: the online
actor the ODE eval reads and the EMA target actor/noise nets the SDE eval reads), plus a full
params_<final>.pkl. This script evaluates every checkpoint afterwards (actor_<step>.pkl when
present, else params_<step>.pkl) and logs it into the training run's W&B run (resumed by id,
resume="must"), against a custom x-axis eval/bc_step:

  eval/*      ODE eval (agent.compute_flow_actions_corrected; the BC eval's default
              eval_rng, PRNGKey(0) inside evaluate_qipm_parallel)
  eval_sde/*  SDE eval (agent.sample_actions with PRNGKey(1), as bc_runner's SDE eval);
              on by default, --no_sde to skip

both through ogpo.runners.qipm_runner._run_qipm_evaluation -- the function the in-training
BC eval calls -- with global_step = the checkpoint step. The episode count is the run's own
(eval.n_eval_envs * eval.eval_episodes from flags.json, 64), split as 8 envs x 8 episodes:
episode b*8+i resets with seed step + b*8 + i, so seeds step..step+63, the same start states
as the in-training 32 x 2 eval. (Per-episode policy noise draws differ from a 32-env batch,
and CPU numerics differ from GPU, so values are statistically, not bitwise, the in-training
eval.) The protocol is recorded in the W&B config under cpu_eval. safety_views, the
controller and the dataset root come from the run's flags.json, so the env is exactly the
training run's:
  provenance.robosuite_controller_config  -> ROBOSUITE_CONTROLLER_CONFIG (mink; unset = OSC)
  provenance.robomimic_dataset_root       -> ROBOMIMIC_DATASET_ROOT
  eval.osc_safety_monitor_config          -> OGPO_OSC_SAFETY_MONITOR_CONFIG (OSC side-car)

Order is coarse-to-fine: multiples of 50000 (ascending), then of 25000, then of 10000, then
anything else; only checkpoints that exist.

Requeue-safe: each step's eval dicts go to <run_dir>/cpu_eval/step_<N>.json BEFORE logging.
<run_dir>/cpu_eval_state.json (atomic) keeps "pending" (logged this session, not yet
confirmed) and "logged" (confirmed by a successful run.finish()). On restart logged steps
are skipped; pending and computed-but-unlogged steps are re-logged from their json without
re-evaluating (a rare duplicate point is possible, a silent gap is not). SIGTERM
(preemption, the time limit, scontrol requeue) aborts the current step at the next episode
boundary, finishes W&B and exits with EXIT_INCOMPLETE so the sbatch wrapper requeues.

Once every checkpoint is evaluated, performance levels are selected
(select_bc_checkpoints.select_levels: centred 3-point smoothed ODE success over consecutive
evaluated steps, closest to each target, ties to the earliest step; target 1.0 = best):
<run_dir>/selection.json, <run_dir>/selected/target_{25,50,75,100}.pkl (copies of the
selected checkpoints), and in the same W&B run summary levels/target_XX/* plus the
wandb.Table levels/table.

Executed-motion safety and shield-intervention keys (safety/executed_*, safety/requested_any_*,
safety/shield_*; see ogpo/utils/eval_safety.py and envs/shield_diagnostics.py) are added to
every eval via OGPO_SHIELD_DIAGNOSTICS=1 (--no_shield_diagnostics to skip; they cost one extra
margin evaluation per inner sim step). No pre-existing key changes.

Side re-evaluations (--results_dir): json only into that directory, never W&B, never
cpu_eval_state.json, never the selection -- the run dir's own eval record is untouched. Steps
whose json already exists there are skipped (requeue-safe). Combine with --max_step /
--step_multiple to pick a grid, --controller_config to swap the controller (same roster) and
--episodes_per_env for a short smoke.

    python scripts/safety_pretrain/eval_bc_run.py --run_dir <training run dir>
    python scripts/safety_pretrain/eval_bc_run.py --run_dir <dir> --results_dir <out> \
        --step_multiple 50000 --max_step 500000
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

STATE_FILE = "cpu_eval_state.json"
RESULTS_DIR = "cpu_eval"
SELECTION_FILE = "selection.json"
STEP_KEY = "eval/bc_step"
SDE_EVAL_KEY = 1  # bc_runner._SDE_EVAL_KEY
TARGETS = (0.25, 0.5, 0.75, 1.0)
EXIT_INCOMPLETE = 3  # the sbatch wrapper requeues on this
# Consecutive non-preemption W&B finish() failures before giving up instead of requeueing.
MAX_FINISH_FAILURES = 3
_DERIVED_KEYS = ("full_act_dim", "proprio_dim", "ob_dims", "action_dim")
LEVEL_KEYS = ("safety/floor080_step_rate", "safety/floor080_episode_rate",
              "safety/floor080_safe_success", "safety/floor081_step_rate",
              "safety/floor081_episode_rate", "safety/floor081_safe_success")


# --------------------------------------------------------------------------- pure helpers

def checkpoint_files(run_dir):
    """{step: path} -- actor_<step>.pkl preferred over params_<step>.pkl."""
    out = {}
    for prefix in ("params", "actor"):  # actor second: it wins
        for p in glob.glob(os.path.join(run_dir, f"{prefix}_*.pkl")):
            m = re.fullmatch(rf"{prefix}_(\d+)\.pkl", os.path.basename(p))
            if m:
                out[int(m.group(1))] = p
    return out


def checkpoint_steps(run_dir):
    return sorted(checkpoint_files(run_dir))


def filter_steps(steps, max_step=None, step_multiple=None):
    """Steps <= max_step (when given) that are multiples of step_multiple (when given)."""
    return [s for s in steps
            if (max_step is None or s <= max_step) and (not step_multiple or s % step_multiple == 0)]


def ordered_steps(steps, grains=(50000, 25000, 10000)):
    """Coarse-to-fine evaluation order: multiples of each grain ascending, then the rest."""
    remaining = sorted(set(steps))
    out = []
    for g in grains:
        out += [s for s in remaining if s % g == 0]
        remaining = [s for s in remaining if s % g != 0]
    return out + remaining


def episodes_per_env(flags, n_envs):
    ev = flags.get("eval", {}) or {}
    total = int(ev.get("n_eval_envs", 32)) * int(ev.get("eval_episodes", 2))
    if total % n_envs:
        raise ValueError(f"run's eval size {total} is not divisible by n_envs={n_envs}")
    return total // n_envs, total


def _atomic_write_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_state(run_dir):
    path = os.path.join(run_dir, STATE_FILE)
    if not os.path.exists(path):
        return {"logged": [], "pending": [], "finish_failures": 0}
    with open(path) as f:
        d = json.load(f)
    # pre-pending format: {"logged_steps": [...]}
    return {"logged": sorted(int(s) for s in d.get("logged", d.get("logged_steps", []))),
            "pending": sorted(int(s) for s in d.get("pending", [])),
            "finish_failures": int(d.get("finish_failures", 0))}


def _write_state(run_dir, state, wandb_run_id=None):
    _atomic_write_json(os.path.join(run_dir, STATE_FILE),
                       {"logged": sorted(set(state["logged"])),
                        "pending": sorted(set(state["pending"]) - set(state["logged"])),
                        "finish_failures": int(state.get("finish_failures", 0)),
                        "wandb_run_id": wandb_run_id})


def load_logged_steps(run_dir):
    """Steps whose W&B logging was confirmed (pending ones are NOT included)."""
    return set(load_state(run_dir)["logged"])


def mark_pending(run_dir, step, wandb_run_id=None):
    st = load_state(run_dir)
    st["pending"] = sorted(set(st["pending"]) | {int(step)})
    _write_state(run_dir, st, wandb_run_id)


def promote_pending(run_dir, wandb_run_id=None):
    """After a successful run.finish(): every pending step is confirmed logged."""
    st = load_state(run_dir)
    st["logged"] = sorted(set(st["logged"]) | set(st["pending"]))
    st["pending"] = []
    st["finish_failures"] = 0
    _write_state(run_dir, st, wandb_run_id)


def count_finish_failure(run_dir, wandb_run_id=None):
    """Record a non-preemption W&B finish failure; returns the consecutive count."""
    st = load_state(run_dir)
    st["finish_failures"] = int(st.get("finish_failures", 0)) + 1
    _write_state(run_dir, st, wandb_run_id)
    return st["finish_failures"]


def _results_root(run_dir, results_dir=None):
    return results_dir if results_dir else os.path.join(run_dir, RESULTS_DIR)


def step_result_path(run_dir, step, results_dir=None):
    return os.path.join(_results_root(run_dir, results_dir), f"step_{int(step)}.json")


def save_step_result(run_dir, step, ode, sde, protocol=None, results_dir=None):
    os.makedirs(_results_root(run_dir, results_dir), exist_ok=True)
    _atomic_write_json(step_result_path(run_dir, step, results_dir),
                       {"step": int(step), "ode": ode, "sde": sde, "protocol": protocol})


def load_step_result(run_dir, step, results_dir=None):
    path = step_result_path(run_dir, step, results_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d["ode"], d["sde"]


def numeric_only(info):
    """Scalar float view of an eval dict (drops non-numeric values; NaN kept)."""
    import numpy as np
    out = {}
    for k, v in info.items():
        if isinstance(v, (bool, int, float, np.bool_, np.number)):
            out[str(k)] = float(v)
    return out


def build_log_row(step, ode, sde):
    """One W&B row: eval/<k> (ODE), eval_sde/<k> (SDE) and the x-axis eval/bc_step."""
    row = {f"eval/{k}": v for k, v in ode.items()}
    row.update({f"eval_sde/{k}": v for k, v in sde.items()})
    assert STEP_KEY not in row, f"eval dict carries a key colliding with {STEP_KEY}"
    row[STEP_KEY] = int(step)
    return row


def eval_line(tag, step, info):
    subs = " | ".join(f"sub{s}={info[f'subtask_{s}_success_rate']}"
                      for s in range(10) if f"subtask_{s}_success_rate" in info)
    safety = " ".join(f"{k[len('safety/'):]}={v:.3f}" for k, v in info.items()
                      if k.startswith("safety/") and k.endswith("_episode_rate"))
    return (f"[EVAL {tag}] step={step:7d} | success={info.get('success', 0.0):.4f} | "
            f"return={info.get('return', 0.0):.4f}" + (f" | {subs}" if subs else "")
            + (f" | safety: {safety}" if safety else ""))


def level_rows(selection, results):
    """Per selected level: the selection plus ODE/SDE success and floor safety at that step."""
    rows = []
    for p in selection:
        ode, sde = results[p["step"]]
        row = {"target": p["target"], "step": p["step"],
               "smoothed_success": p["smoothed_success"], "ode_success": ode.get("success"),
               "sde_success": sde.get("success") if sde else None,
               "gap_warning": abs(p["smoothed_success"] - p["target"]) > 0.10}
        for k in LEVEL_KEYS:
            row[f"ode_{k}"] = ode.get(k)
            row[f"sde_{k}"] = sde.get(k) if sde else None
        rows.append(row)
    return rows


def level_summary(rows):
    """run.summary entries levels/target_XX/<field> for every level row."""
    out = {}
    for r in rows:
        tag = f"levels/target_{int(round(r['target'] * 100))}"
        for k, v in r.items():
            if k != "target" and v is not None:
                out[f"{tag}/{k}"] = v
    return out


def select_and_save(run_dir, files, results, targets=TARGETS):
    """selection.json + selected/target_XX.pkl copies; returns the level rows."""
    from select_bc_checkpoints import WARN_GAP, select_levels
    raw = {s: ode["success"] for s, (ode, _) in results.items()}
    selection = select_levels(raw, [s for s in raw if s in files], targets)
    os.makedirs(os.path.join(run_dir, "selected"), exist_ok=True)
    for p in selection:
        src = files[p["step"]]
        dst = os.path.join(run_dir, "selected", f"target_{int(round(p['target'] * 100))}.pkl")
        tmp = dst + ".tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        p["checkpoint"] = src
        p["copy"] = dst
        if abs(p["smoothed_success"] - p["target"]) > WARN_GAP:
            print(f"[select] WARNING: target {p['target']}: closest checkpoint (step {p['step']}) "
                  f"has smoothed success {p['smoothed_success']:.4f}, more than {WARN_GAP} away",
                  flush=True)
    rows = level_rows(selection, results)
    _atomic_write_json(os.path.join(run_dir, SELECTION_FILE),
                       {"evaluated_steps": sorted(results), "selection": selection, "levels": rows})
    return rows


# --------------------------------------------------------------------------- run setup

def apply_run_environment(flags):
    """Controller / dataset / side-car environment of the training run, from flags.json."""
    prov = flags.get("provenance", {}) or {}
    ctrl = prov.get("robosuite_controller_config")
    if ctrl:
        os.environ["ROBOSUITE_CONTROLLER_CONFIG"] = ctrl
    else:
        os.environ.pop("ROBOSUITE_CONTROLLER_CONFIG", None)
    root = prov.get("robomimic_dataset_root")
    if root:
        os.environ["ROBOMIMIC_DATASET_ROOT"] = root
    else:
        os.environ.pop("ROBOMIMIC_DATASET_ROOT", None)
    from envs.osc_safety_sidecar import ENV_VAR
    sidecar = (flags.get("eval", {}) or {}).get("osc_safety_monitor_config")
    if sidecar:
        os.environ[ENV_VAR] = os.path.abspath(str(sidecar))
    else:
        os.environ.pop(ENV_VAR, None)
    return ctrl, root, sidecar


def find_wandb_id(run_dir, flags, explicit=None):
    if explicit:
        return explicit
    exp = flags["experiment"]
    exp_name = os.path.basename(os.path.normpath(run_dir))
    id_file = os.path.join(exp["save_dir"], ".wandb_ids", f"{exp['project']}_{exp_name}.json")
    if os.path.exists(id_file):
        with open(id_file) as f:
            d = json.load(f)
        if d.get("name") == exp_name and d.get("wandb_run_id"):
            return d["wandb_run_id"]
    import wandb
    runs = list(wandb.Api().runs(f"{os.environ.get('WANDB_ENTITY', 'akankshasingal08')}/{exp['project']}",
                                 filters={"group": exp["run_group"], "display_name": exp_name}))
    if len(runs) != 1:
        raise RuntimeError(f"cannot identify the W&B run for {exp_name}: {len(runs)} candidates")
    return runs[0].id


def build_eval_envs(flags, n_envs):
    from ogpo.utils.config_loader import config_to_agent_dict
    from ogpo.utils.main_helper import make_vectorized_env
    algo = flags["agent"]["agent_name"]
    assert algo == "qipm", f"only qipm BC runs are supported, got {algo}"
    agent_config = config_to_agent_dict(flags)
    env_config = flags["env"]
    include_joint_proprio = bool(flags["agent"].get("critic_joint_proprio"))
    use_dense = bool(agent_config.get("use_dense_rewards", False))
    # Same call as ogpo/main.py's eval envs (qipm, robomimic: no OGBench GC wrapper).
    return make_vectorized_env(
        env_config["name"], num_envs=n_envs, seed=flags["experiment"]["seed"] + 1000,
        rew_fn=env_config["rew_fn"], dense_reward_wrapper=use_dense, use_dense_reward=use_dense,
        no_of_subtasks=agent_config.get("no_of_subtasks", 3),
        normalize_rewards=agent_config.get("normalize_rewards", False),
        subtask_dag=agent_config.get("subtask_dag", None),
        goal_conditioning_mode=agent_config.get("goal_conditioning_mode", None),
        anchor_demo=env_config.get("anchor_demo", None),
        obstacle=env_config.get("obstacle", None),
        include_joint_proprio=include_joint_proprio,
    )


def build_agent_template(flags, obs_dim, action_dim):
    """An OGPOQIPMAgent built as ogpo/main.py builds it; checkpoints restore its full state."""
    import jax.numpy as jnp
    from ogpo.agents.qipm import OGPOQIPMAgent
    from ogpo.utils.config_loader import config_to_agent_dict
    from ogpo.utils.main_helper import get_q_bounds
    cfg = dict(flags)
    cfg["agent"] = {k: v for k, v in flags["agent"].items() if k not in _DERIVED_KEYS}
    agent_config = config_to_agent_dict(cfg)
    agent_config["q_min"], agent_config["q_max"] = get_q_bounds(flags["env"]["name"],
                                                                agent_config["discount"])
    return OGPOQIPMAgent.create(flags["experiment"]["seed"], jnp.zeros((obs_dim,), jnp.float32),
                                jnp.zeros((action_dim,), jnp.float32), agent_config,
                                agent_config.get("adv_clip_min"))


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--wandb_id", default=None, help="default: <save_dir>/.wandb_ids, then a W&B lookup")
    ap.add_argument("--n_envs", type=int, default=8)
    ap.add_argument("--no_sde", action="store_true", help="ODE only (default: ODE + SDE)")
    ap.add_argument("--no_wandb", action="store_true", help="evaluate + save json only")
    ap.add_argument("--no_ode", action="store_true", help="SDE only")
    ap.add_argument("--results_dir", default=None,
                    help="side re-evaluation: json-only into this dir (relative = under run_dir); "
                         "implies --no_wandb, ignores cpu_eval_state.json, no selection")
    ap.add_argument("--max_step", type=int, default=None, help="only checkpoints with step <= this")
    ap.add_argument("--step_multiple", type=int, default=None,
                    help="only checkpoints whose step is a multiple of this")
    ap.add_argument("--episodes_per_env", type=int, default=None,
                    help="override the run's episode count (smoke tests; requires --results_dir)")
    ap.add_argument("--controller_config", default=None,
                    help="override ROBOSUITE_CONTROLLER_CONFIG (requires --results_dir)")
    ap.add_argument("--no_shield_diagnostics", action="store_true",
                    help="skip the safety/executed_* and safety/shield_* keys")
    args = ap.parse_args(argv)
    assert args.n_envs > 1, "--n_envs must be > 1 (the vectorized, global_step-seeded eval path)"
    side = args.results_dir is not None
    if (args.episodes_per_env is not None or args.controller_config) and not side:
        ap.error("--episodes_per_env / --controller_config change the protocol: use --results_dir")
    if args.no_ode and not side:
        ap.error("--no_ode leaves the selection without ODE success: use --results_dir")
    if side:
        args.no_wandb = True

    run_dir = os.path.abspath(args.run_dir)
    with open(os.path.join(run_dir, "flags.json")) as f:
        flags = json.load(f)
    ctrl, root, sidecar = apply_run_environment(flags)
    if args.controller_config:
        ctrl = os.path.abspath(args.controller_config)
        assert os.path.isfile(ctrl), ctrl
        os.environ["ROBOSUITE_CONTROLLER_CONFIG"] = ctrl
    from envs.shield_diagnostics import ENV_VAR as SHIELD_ENV_VAR
    shield_diag = not args.no_shield_diagnostics
    if shield_diag:
        os.environ[SHIELD_ENV_VAR] = "1"
    else:
        os.environ.pop(SHIELD_ENV_VAR, None)
    results_dir = None
    if side:
        results_dir = os.path.join(run_dir, args.results_dir) if not os.path.isabs(
            args.results_dir) else args.results_dir
        results_dir = os.path.abspath(results_dir)
    mp = flags["experiment"].get("matmul_precision")
    if mp:
        import jax
        jax.config.update("jax_default_matmul_precision", mp)
    n_ep, total = episodes_per_env(flags, args.n_envs)
    if args.episodes_per_env is not None:
        n_ep, total = int(args.episodes_per_env), int(args.episodes_per_env) * args.n_envs
    sde_on = not args.no_sde
    ode_on = not args.no_ode
    assert sde_on or ode_on, "--no_ode and --no_sde together evaluate nothing"
    protocol = {"n_envs": args.n_envs, "episodes_per_env": n_ep, "episodes": total,
                "sde": sde_on, "device": "cpu", "ode_eval_rng": "PRNGKey(0)",
                "sde_eval_rng": f"PRNGKey({SDE_EVAL_KEY})"}
    if side:  # extra fields only off the default path (its W&B config stays as it was)
        protocol.update({"ode": ode_on, "controller": ctrl, "shield_diagnostics": shield_diag,
                         "results_dir": results_dir})
    print(f"[cpu-eval] run_dir={run_dir}\n[cpu-eval] controller={ctrl or '<dataset OSC>'} "
          f"dataset_root={root} sidecar={sidecar}\n[cpu-eval] protocol={protocol}", flush=True)

    files = checkpoint_files(run_dir)
    if args.max_step is not None or args.step_multiple:
        files = {s: files[s] for s in filter_steps(sorted(files), args.max_step, args.step_multiple)}
    steps = sorted(files)
    if not steps:
        print("[cpu-eval] no checkpoints (actor_<step>.pkl / params_<step>.pkl) in the run dir")
        return 1
    if steps:
        print(f"[cpu-eval] checkpoint size: {os.path.basename(files[steps[0]])} "
              f"{os.path.getsize(files[steps[0]]) / 1e6:.1f} MB", flush=True)
    # A "logged" step whose result json is gone is evaluated again (and logged
    # again, a duplicate point) rather than blocking the selection forever.
    if side:  # done = json present in the side results dir; the run's own state is ignored
        logged = {s for s in steps if load_step_result(run_dir, s, results_dir) is not None}
    else:
        logged = {s for s in load_logged_steps(run_dir)
                  if load_step_result(run_dir, s) is not None}
    todo = [s for s in ordered_steps(steps) if s not in logged]
    print(f"[cpu-eval] {len(steps)} checkpoints, {len(logged)} already logged, order: {todo}", flush=True)
    sel_path = os.path.join(run_dir, SELECTION_FILE)
    selection_done = False
    if os.path.exists(sel_path):
        with open(sel_path) as f:
            selection_done = json.load(f).get("evaluated_steps") == steps
    if not todo and (selection_done or args.no_wandb or side):
        print("[cpu-eval] nothing to do")
        return 0

    from ogpo.utils.main_helper import (
        install_preemption_handler, is_preemption_requested, PreemptionDuringEval)
    install_preemption_handler()

    run = None
    wandb_id = None
    if not args.no_wandb:
        import wandb
        wandb_id = find_wandb_id(run_dir, flags, args.wandb_id)
        run = wandb.init(project=flags["experiment"]["project"], id=wandb_id, resume="must")
        run.define_metric(STEP_KEY)
        run.define_metric("eval/*", step_metric=STEP_KEY)
        run.define_metric("eval_sde/*", step_metric=STEP_KEY)
        run.config.update({"cpu_eval": protocol}, allow_val_change=True)
        print(f"[cpu-eval] resumed W&B run {wandb_id} ({run.name})", flush=True)

    def log_step(step, ode, sde):
        if run is not None:
            run.log(build_log_row(step, ode, sde))  # no step=: x-axis is eval/bc_step
            mark_pending(run_dir, step, wandb_id)

    def finish(exit_code=0, preempting=False):
        """True iff W&B confirmed the run's data; pending steps are promoted only then.

        On preemption Slurm signals every process in the job, the wandb service
        included, so finish() can fail; the pending steps then stay pending and
        are re-logged from their json on the next start.
        """
        if run is None:
            return True
        try:
            if preempting:
                run.mark_preempting()
            run.finish(exit_code=exit_code)
        except Exception as exc:  # noqa: BLE001
            print(f"[cpu-eval] W&B finish failed ({type(exc).__name__}: {exc}); "
                  "pending steps stay pending and are re-logged on restart", flush=True)
            return False
        promote_pending(run_dir, wandb_id)
        return True

    # Computed earlier but not confirmed logged: re-log from json, no re-evaluation.
    for s in list(todo):
        res = load_step_result(run_dir, s, results_dir)
        if res is not None:
            if run is not None:
                log_step(s, *res)
                print(f"[cpu-eval] (re)logged saved results for step {s}", flush=True)
            todo.remove(s)

    envs = None
    try:
        import jax
        from ogpo.agents.modules.flax_utils import restore_actor_checkpoint, restore_agent_with_file
        from ogpo.runners.qipm_runner import _build_eval_flags, _run_qipm_evaluation
        from ogpo.utils.eval_safety import parse_safety_views

        if todo:
            envs = build_eval_envs(flags, args.n_envs)
            obs_dim = envs.single_observation_space.shape[-1]
            action_dim = envs.single_action_space.shape[-1]
            template = build_agent_template(flags, obs_dim, action_dim)
            safety_views = parse_safety_views(flags["eval"].get("safety_views"))
            agent_cfg = flags["agent"]
            n_sub = agent_cfg.get("no_of_subtasks", 3)
            discount = agent_cfg["discount"]
            run_name = flags["experiment"]["run_name"]
            eval_flags = _build_eval_flags(args.n_envs, n_ep, discount, run_name,
                                           flags["env"]["name"], safety_views=safety_views)

        for step in todo:
            if is_preemption_requested():
                raise PreemptionDuringEval("preemption requested between checkpoints")
            t0 = time.time()
            path = files[step]
            if os.path.basename(path).startswith("actor_"):
                agent = restore_actor_checkpoint(template, path)
            else:
                agent = restore_agent_with_file(template, path)
            common = dict(eval_env=None, eval_envs=envs, eval_flags=eval_flags,
                          action_dim=action_dim, log_step=step, eval_episodes=n_ep,
                          run_name=run_name, no_of_subtasks=n_sub, discount=discount,
                          plot=False, safety_views=safety_views)
            ode_actor = getattr(agent, "compute_flow_actions_corrected", agent.compute_flow_actions)
            ode = {}
            if ode_on:
                ode = numeric_only(_run_qipm_evaluation(agent=agent, actor_fn=ode_actor, **common))
            t1 = time.time()
            sde = {}
            if sde_on:
                sde = numeric_only(_run_qipm_evaluation(
                    agent=agent, actor_fn=agent.sample_actions,
                    eval_rng=jax.random.PRNGKey(SDE_EVAL_KEY), **common))
            t2 = time.time()
            save_step_result(run_dir, step, ode, sde, protocol, results_dir)
            if ode_on:
                print(eval_line("ODE", step, ode), flush=True)
            if sde_on:
                print(eval_line("SDE", step, sde), flush=True)
            print(f"[cpu-eval] step={step} ode={t1 - t0:.0f}s sde={t2 - t1:.0f}s", flush=True)
            log_step(step, ode, sde)
    except PreemptionDuringEval as exc:
        print(f"[PREEMPTION] {exc}; finishing W&B and exiting for requeue", flush=True)
        finish(exit_code=0, preempting=True)
        return EXIT_INCOMPLETE
    except Exception:
        # A preemption SIGTERM also reaches the env worker processes, which can
        # surface as a broken pipe here rather than PreemptionDuringEval.
        if not is_preemption_requested():
            raise
        import traceback
        traceback.print_exc()
        print("[PREEMPTION] eval aborted by the preemption signal; exiting for requeue", flush=True)
        finish(exit_code=0, preempting=True)
        return EXIT_INCOMPLETE
    finally:
        if envs is not None:
            try:
                envs.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[cleanup] closing eval envs failed: {exc}")

    if side:
        print(f"[cpu-eval] side re-evaluation done: {results_dir}", flush=True)
        return 0
    # All checkpoints evaluated: performance-level selection into the same run.
    results = {s: load_step_result(run_dir, s) for s in steps}
    missing = [s for s, r in results.items() if r is None]
    if missing:
        # Every step without a result was in todo, so a requeue would not fix this.
        print(f"[cpu-eval] ERROR: steps without results after evaluating all: {missing}", flush=True)
        finish(exit_code=1)
        return 1
    rows = select_and_save(run_dir, files, results)
    for r in rows:
        print(f"[select] target {r['target']:.2f}: step {r['step']} smoothed "
              f"{r['smoothed_success']:.4f} ode {r['ode_success']:.4f} "
              f"sde {r['sde_success'] if r['sde_success'] is None else round(r['sde_success'], 4)}",
              flush=True)
    if run is not None:
        import wandb
        run.summary.update(level_summary(rows))
        cols = list(rows[0])
        run.log({"levels/table": wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})
    if not finish(preempting=is_preemption_requested()):
        # Requeue to re-log the unconfirmed steps and the levels, but not forever:
        # a finish() that fails every time (outside a preemption) is not transient.
        # A preemption kills the wandb service, so that failure does not count.
        if is_preemption_requested():
            return EXIT_INCOMPLETE
        n = count_finish_failure(run_dir, wandb_id)
        if n >= MAX_FINISH_FAILURES:
            print(f"[cpu-eval] ERROR: W&B finish failed on {n} consecutive attempts; "
                  "not requeueing. Results are in cpu_eval/step_*.json.", flush=True)
            return 1
        return EXIT_INCOMPLETE
    print("[cpu-eval] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
