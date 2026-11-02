"""IMPRINT Stage 1 (memory bootstrap) on square low-dim rollouts. Runs in .venv-imprint.

Model, loss, optimizer and checkpoint format are IMPRINT's own (mme_vla_suite): the recurrent
gated-delta encoder plus the Perceiver-IO reconstruction decoder, trained with
composite_alignment_loss (align 1.0, action 0.5, proprio 0.5, progress 0.5; paper weights).
Only the dataloader is ours (imprint_bridge/data.py): IMPRINT's loader reads a WAN/DINOv3 feature
store that our image-free rollouts don't have.

Why train on our data at all: the released encoders consume 3-view WAN2.2 latents + DINOv3
features, 12-d RoboCasa actions and 16-d proprio. Our inputs are 14-d object state, 7-d OSC
actions and 9-d proprio, so no released encoder runs on them as is; the input projections,
tokenizers, event-slot embedding and decoder heads must be (re)trained. `--init_from` fine-tunes:
it copies every shape-compatible tensor (the delta-net core, read-out, recent fusion, 256 queries,
decoder trunk) from the released RoboCasa recurrent Stage-1 checkpoint and trains everything.
Without it, Stage 1 starts from scratch.

Episodes are split as in train_mc_critic.py: episode_id < val_start trains, [val_start, test_start)
is validation; test episodes are never seen.

    python -m imprint_bridge.train_stage1 --data_dir <rollouts> --out_dir <dir> [--init_from stage1_step160000.npz]
"""

import argparse
import functools
import json
import logging
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO, "imprint_bridge", "square_lowdim_gated_delta.yaml")
logger = logging.getLogger("imprint_stage1")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="/data/user_data/asingal2/memory_critic/rollouts/flow_bc_target100")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--init_from", default=None, help="released IMPRINT stage1_step*.npz (fine-tune)")
    p.add_argument("--num_steps", type=int, default=40_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--w_align", type=float, default=1.0)
    p.add_argument("--w_action", type=float, default=0.5)
    p.add_argument("--w_proprio", type=float, default=0.5)
    p.add_argument("--w_reward", type=float, default=0.5)
    p.add_argument("--compute_dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--val_start", type=int, default=288)
    p.add_argument("--test_start", type=int, default=320)
    p.add_argument("--val_batches", type=int, default=16)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=400, help="episode length cap (padded history)")
    p.add_argument("--max_policies", type=int, default=None, help="smoke tests only")
    p.add_argument("--config_override", action="append", default=[],
                   help="OmegaConf dotlist entry, e.g. memory_token_dim=64 (smoke tests only)")
    p.add_argument("--wandb_entity", default=None, help="default: $WANDB_ENTITY (run.sbatch sets akankshasingal08)")
    p.add_argument("--wandb_project", default="memory-critic")
    p.add_argument("--wandb_group", default="imprint_stage1")
    p.add_argument("--run_name", default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args(argv)


def load_config(path, compute_dtype, overrides=()):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    cfg.cosmo_bottleneck_memory.memory_compute_dtype = compute_dtype
    cfg.cosmo_bottleneck_memory.delta_wy_matmul_dtype = compute_dtype
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def init_from_released(modules, path, min_fraction=0.5):
    """Copy every tensor whose key and shape match from a released Stage-1 npz; report the rest.

    IMPRINT's own loader (_load_stage1_checkpoint_into_modules) requires every shape to match, which
    ours can't (different input/output widths), so this is its shape-tolerant variant. Keys come from
    IMPRINT's _module_flat_params, the format _save_checkpoint writes."""
    import flax.traverse_util as trav
    import numpy as np
    from flax import nnx
    from mme_vla_suite.scripts.pretrain_bottleneck import _module_flat_params

    src = np.load(path, allow_pickle=True)
    want = str(modules["bottleneck"].architecture_name)
    got = str(np.asarray(src["metadata/memory_architecture"]).item()) if "metadata/memory_architecture" in src.files else None
    if got != want:
        raise ValueError(f"--init_from {path}: architecture {got!r}, expected {want!r}")
    report = {}
    for prefix in ("bottleneck", "decoder"):
        graphdef, st = nnx.split(modules[prefix])
        flat = trav.flatten_dict(st.to_pure_dict())
        names = list(_module_flat_params(modules[prefix], prefix))   # same order as flat
        loaded, shape_skip, missing = [], [], []
        n_cur = n_loaded = 0
        for (key, cur), name in zip(flat.items(), names, strict=True):
            n_cur += int(np.size(cur))
            if name not in src.files:
                missing.append(name)
                continue
            arr = np.asarray(src[name])
            if arr.shape != np.shape(cur):
                shape_skip.append(f"{name} {arr.shape}->{np.shape(cur)}")
                continue
            flat[key] = arr.astype(np.asarray(cur).dtype)
            loaded.append(name)
            n_loaded += arr.size
        frac = n_loaded / max(n_cur, 1)
        if frac < min_fraction:
            raise ValueError(f"--init_from {path} filled only {frac:.1%} of {prefix} params "
                             f"(key or architecture mismatch?); refusing to train a de facto scratch run")
        st.replace_by_pure_dict(trav.unflatten_dict(flat))
        modules[prefix] = nnx.merge(graphdef, st)
        report[prefix] = {"tensors": len(flat), "loaded": len(loaded), "shape_mismatch": shape_skip,
                          "missing_in_ckpt": missing, "param_fraction_loaded": frac}
        logger.info("init_from %s: %d/%d tensors (%.1f%% of params); shape-mismatched (re-initialised): %s",
                    prefix, len(loaded), len(flat), 100 * frac, shape_skip)
    return modules, report


def persistent_run_id(out_dir):
    """One W&B run per out_dir, so a requeued job appends to the same run."""
    path = os.path.join(out_dir, "wandb_run_id.txt")
    if not os.path.exists(path):
        import uuid
        with open(path, "w") as f:
            f.write(uuid.uuid4().hex[:12])
    return open(path).read().strip()


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    os.makedirs(args.out_dir, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import nnx
    from mme_vla_suite.scripts.pretrain_bottleneck import (
        _restore_training_state, _save_checkpoint, _save_training_state, build_optimizer)
    from mme_vla_suite.training.pretrain_loss import composite_alignment_loss, make_stage1_modules

    from imprint_bridge import data as D

    t0 = time.time()
    episodes = D.load_episodes(args.data_dir, lambda e: e < args.test_start)
    if args.max_policies:
        keep = sorted({e.policy for e in episodes})[:args.max_policies]
        episodes = [e for e in episodes if e.policy in keep]
    train = [e for e in episodes if e.episode_id < args.val_start]
    val = [e for e in episodes if e.episode_id >= args.val_start]
    longest = max(len(e.obj) for e in episodes)
    assert longest <= args.max_steps, f"episode of length {longest} > --max_steps {args.max_steps}"
    norm = D.Norm.fit(train)
    train = [norm.apply(e) for e in train]
    val = [norm.apply(e) for e in val]
    has_progress = all(e.progress is not None for e in train)
    if not has_progress and args.w_reward > 0:
        logger.warning("rollouts have no subtask_status: progress loss disabled")
        args.w_reward = 0.0
    sh = D.Shapes(max_steps=args.max_steps)
    np.savez(os.path.join(args.out_dir, "norm.npz"), **norm.to_dict())
    logger.info("%d train / %d val episodes from %d policies, longest %d steps (%.0fs)", len(train), len(val),
                len({e.policy for e in episodes}), longest, time.time() - t0)

    cfg = load_config(args.config, args.compute_dtype, args.config_override)
    assert int(cfg.cosmo_bottleneck_memory.temporal_factor) == sh.temporal_factor
    dtype = jnp.bfloat16 if args.compute_dtype == "bfloat16" else jnp.float32
    modules = make_stage1_modules(cfg, seed=args.seed, dtype=dtype)
    init_report = None
    if args.init_from:
        modules, init_report = init_from_released(modules, args.init_from)
    graphdef, state = nnx.split((modules["bottleneck"], modules["decoder"]))
    tx = build_optimizer(args.lr, args.weight_decay, args.warmup_steps, args.num_steps)
    opt_state = tx.init(state)

    step = 0
    full_dir = os.path.join(args.out_dir, "full_state")
    done = sorted(int(d) for d in os.listdir(full_dir) if d.isdigit()) if os.path.isdir(full_dir) else []
    if done:  # preemption resume
        repl = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        state, opt_state, step = _restore_training_state(os.path.join(full_dir, str(done[-1])), state,
                                                         opt_state, repl)
        logger.info("resumed from step %d", step)

    weights = {"align": args.w_align, "action": args.w_action, "proprio": args.w_proprio,
               "reward": args.w_reward}

    def loss_fn(st, batch):
        bmk, dec = nnx.merge(graphdef, st)
        return composite_alignment_loss({"bottleneck": bmk, "decoder": dec}, batch, weights)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def train_step(st, os_, batch):
        (total, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(st, batch)
        updates, os_ = tx.update(grads, os_, st)
        return optax.apply_updates(st, updates), os_, total, parts

    eval_step = jax.jit(loss_fn)

    val_it = D.stage1_batches(val, sh, args.batch_size, np.random.default_rng(12345))
    val_set = [next(val_it) for _ in range(args.val_batches)]

    def target_var(key):
        m = np.concatenate([b["query_mask"].reshape(-1) for b in val_set])
        x = np.concatenate([b[key].reshape(b["query_mask"].size, -1) for b in val_set])[m]
        return float(x.var(0).mean())

    var = {"action": target_var("action_target"), "proprio": target_var("proprio_target")}
    if has_progress:
        r = np.concatenate([b["reward_target"] for b in val_set])
        var["reward"] = float(r.var())

    def evaluate(st):
        sums = {}
        for b in val_set:
            _, parts = eval_step(st, b)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v) / len(val_set)
        out = {f"val/{k}": v for k, v in sums.items()}
        for k, v in var.items():
            if k in sums and v > 0:
                out[f"val/r2_{k}"] = 1.0 - sums[k] / v
        return out

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group,
                         name=args.run_name or os.path.basename(os.path.normpath(args.out_dir)),
                         id=persistent_run_id(args.out_dir), resume="allow",
                         config={**vars(args), "n_train_episodes": len(train), "n_val_episodes": len(val),
                                 "init_report": init_report})

    train_it = D.stage1_batches(train, sh, args.batch_size, np.random.default_rng(args.seed + 1000 * step))
    history = []
    best = {"val_loss": float("inf"), "step": None}
    hist_path = os.path.join(args.out_dir, "history.json")
    if done and os.path.exists(hist_path):
        prev = json.load(open(hist_path))
        history = [h for h in prev["history"] if h["step"] <= step]
    best_path = os.path.join(args.out_dir, "best.json")
    if done and os.path.exists(best_path):  # stage1_stepbest.npz may be newer than the resumed state
        best = json.load(open(best_path))
    tlog = time.time()
    while step < args.num_steps:
        state, opt_state, total, parts = train_step(state, opt_state, next(train_it))
        step += 1
        if step % args.log_every == 0:
            total = float(total)
            if not np.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}: {parts}")
            rec = {"train/loss": total, **{f"train/{k}": float(v) for k, v in parts.items()},
                   "sec_per_step": (time.time() - tlog) / args.log_every}
            tlog = time.time()
            logger.info("step %d %s", step, {k: round(v, 4) for k, v in rec.items()})
            if run is not None:
                run.log(rec, step=step)
        if step % args.eval_every == 0 or step == args.num_steps:
            ev = evaluate(state)
            val_loss = sum(weights[k] * ev[f"val/{k}"] for k in weights if f"val/{k}" in ev)
            ev["val/loss"] = val_loss
            history.append({"step": step, **ev})
            logger.info("eval step %d %s", step, {k: round(v, 4) for k, v in ev.items()})
            if run is not None:
                run.log(ev, step=step)
            if val_loss < best["val_loss"]:
                best = {"val_loss": val_loss, "step": step}
                bmk, dec = nnx.merge(graphdef, state)
                _save_checkpoint(args.out_dir, "best", {"bottleneck": bmk, "decoder": dec})
                with open(best_path, "w") as f:
                    json.dump(best, f)
        if step % args.ckpt_every == 0 or step == args.num_steps:
            _save_training_state(args.out_dir, step, state, opt_state)
            for old in (d for d in os.listdir(full_dir) if d.isdigit() and int(d) < step):
                shutil.rmtree(os.path.join(full_dir, old), ignore_errors=True)
            with open(hist_path, "w") as f:
                json.dump({"history": history, "best": best}, f, indent=1)

    bmk, dec = nnx.merge(graphdef, state)
    _save_checkpoint(args.out_dir, step, {"bottleneck": bmk, "decoder": dec})
    summary = {"best": best, "final": history[-1] if history else None, "target_var": var,
               "init_report": init_report, "config": vars(args)}
    with open(os.path.join(args.out_dir, "stage1_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if run is not None:
        run.summary.update({"best_step": best["step"], "best_val_loss": best["val_loss"]})
        run.finish()
    logger.info("done: best val loss %.4f at step %s -> %s", best["val_loss"], best["step"], args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
