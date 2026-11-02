"""Freeze a Stage-1 IMPRINT encoder and cache pooled memory at every policy decision point. Runs in .venv-imprint.

For each rollout file `<policy>.npz`, writes `<out_dir>/<policy>.npz` with one row per decision point
(chunk_pos == 0, in file order, i.e. the rows train_mc_critic.py keeps with --points decision):

  episode_id, timestep         alignment keys (checked by train_mc_critic.py)
  feat_mean      (n, 1024) f16 mean of the 256 memory tokens of history 0..t
  feat_progress  (n, 1024) f16 the Stage-1 progress query's read-out of the memory (a trained pool),
                               taken before its scalar head
  progress_pred  (n,)      f32 the progress head's output (diagnostic)

Memory at step t sees frames 0..t, executed actions 0..t-1 and proprio 0..t (IMPRINT's causal
contract). Candidate action chunks never enter the encoder.

    python -m imprint_bridge.cache_memory --stage1_dir <dir> --data_dir <rollouts> --out_dir <dir>
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1_dir", required=True, help="train_stage1.py out_dir (norm.npz, stage1_stepbest.npz)")
    ap.add_argument("--ckpt", default="stage1_stepbest.npz")
    ap.add_argument("--data_dir", default="/data/user_data/asingal2/memory_critic/rollouts/flow_bc_target100")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_policies", type=int, default=None, help="smoke tests only")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from mme_vla_suite.scripts.pretrain_bottleneck import _load_stage1_checkpoint_into_modules
    from mme_vla_suite.training.pretrain_loss import make_stage1_modules

    from imprint_bridge import data as D
    from imprint_bridge.train_stage1 import load_config

    s1 = json.load(open(os.path.join(args.stage1_dir, "stage1_summary.json")))["config"]
    cfg = load_config(s1["config"], s1["compute_dtype"], s1.get("config_override", []))
    dtype = jnp.bfloat16 if s1["compute_dtype"] == "bfloat16" else jnp.float32
    modules = make_stage1_modules(cfg, seed=0, dtype=dtype)
    modules = _load_stage1_checkpoint_into_modules(modules, os.path.join(args.stage1_dir, args.ckpt))
    graphdef, state = nnx.split((modules["bottleneck"], modules["decoder"]))
    norm = D.Norm.from_dict(np.load(os.path.join(args.stage1_dir, "norm.npz")))
    sh = D.Shapes(max_steps=s1["max_steps"])

    @jax.jit
    def encode(st, b):
        bmk, dec = nnx.merge(graphdef, st)
        mem, _, _ = bmk(b["cosmo_latent_emb"], b["cosmo_recent_emb"], b["cosmo_action_emb"],
                        b["cosmo_proprio_emb"], b["cosmo_mem_mask"], lang_emb=None,
                        recent_mask=b["cosmo_recent_mask"], action_mask=b["cosmo_action_mask"],
                        proprio_mask=b["cosmo_proprio_mask"])
        # the decoder's progress branch (cosmo_decoder.py, CosmoMemoryDecoder.__call__), stopped before its head
        rq = jnp.broadcast_to(dec.reward_query.value.astype(dec.dtype)[None, None], (mem.shape[0], 1, dec.dim))
        rq = rq + dec.reward_cross(dec.reward_norm_q(rq), dec.reward_norm_kv(mem))
        prog = jax.nn.sigmoid(dec.head_reward(rq)[:, 0, 0])
        return (mem.mean(1).astype(jnp.float16), rq[:, 0].astype(jnp.float16), prog.astype(jnp.float32))

    files = D.rollout_files(args.data_dir)[:args.max_policies]
    t0 = time.time()
    for path in files:
        policy = os.path.basename(path)[:-4]
        out_path = os.path.join(args.out_dir, f"{policy}.npz")
        if os.path.exists(out_path):
            print(f"[cache] skip {policy} (exists)", flush=True)
            continue
        z = np.load(path)
        eps = {e.episode_id: norm.apply(e) for e in D.split_episodes(z, policy)}
        rows = np.flatnonzero(z["chunk_pos"] == 0)
        ep_ids, ts = z["episode_id"][rows].astype(np.int64), z["timestep"][rows].astype(np.int64)
        outs = []
        for i in range(0, len(rows), args.batch_size):
            pairs = list(zip(ep_ids[i:i + args.batch_size], ts[i:i + args.batch_size]))
            n = len(pairs)
            pairs += [pairs[-1]] * (args.batch_size - n)   # constant batch shape -> one compile
            b = D.collate([D.history_inputs(eps[int(e)], int(t), sh) for e, t in pairs])
            outs.append([np.asarray(x)[:n] for x in encode(state, b)])
        fm, fp, pr = (np.concatenate(c) for c in zip(*outs))
        tmp = out_path + ".tmp.npz"
        np.savez(tmp, episode_id=ep_ids, timestep=ts, feat_mean=fm, feat_progress=fp, progress_pred=pr)
        os.replace(tmp, out_path)
        print(f"[cache] {policy}: {len(rows)} decision points ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
