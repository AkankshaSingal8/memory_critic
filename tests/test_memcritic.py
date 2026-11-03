"""Tests for memcritic/history.py, memcritic/critics.py and load_dataset(events=True), on synthetic rollouts.

    JAX_PLATFORMS=cpu PYTHONPATH=. uv run --locked pytest -q tests/test_memcritic.py
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]

from memcritic import history  # noqa: E402
from train_mc_critic import load_dataset  # noqa: E402

H, OBS, ACT = 8, 3, 2


def write_policy(path, lengths, seed):
    """Episodes of the given lengths; obs[:, 0] = timestep + 1, action[:, 0] = timestep + 1."""
    rng = np.random.default_rng(seed)
    ep, t = np.concatenate([np.full(n, e) for e, n in enumerate(lengths)]), np.concatenate(
        [np.arange(n) for n in lengths])
    obs = rng.normal(size=(len(t), OBS)).astype(np.float32); obs[:, 0] = t + 1
    act = rng.normal(size=(len(t), ACT)).astype(np.float32); act[:, 0] = t + 1
    np.savez(path, obs=obs, action=act, reward=(t == np.array(lengths)[ep] - 1).astype(np.float32),
             episode_id=ep.astype(np.int32), timestep=t.astype(np.int16), chunk_pos=(t % H).astype(np.int8),
             ep_success=(np.array(lengths) < 400).astype(np.int8))


def split_event(ev):
    """(..., H * OBS + H * ACT) -> obs (..., H, OBS), act (..., H, ACT)."""
    return ev[..., :H * OBS].reshape(*ev.shape[:-1], H, OBS), ev[..., H * OBS:].reshape(*ev.shape[:-1], H, ACT)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("rollouts")
    rng = np.random.default_rng(0)
    for i, name in enumerate(("square_ph_10demos_seed0_sde", "square_mh_10demos_seed0_sde")):
        lengths = rng.integers(5, 120, size=12).tolist() + [400]
        write_policy(d / f"{name}.npz", lengths, i)
    out, _ = load_dataset(str(d), 0.99, H, "decision", events=True)
    return out


def test_events_cover_the_chunk_and_not_the_next_decision(data):
    ev = data["event"]
    assert ev.shape[1] == H * (OBS + ACT)
    ep_start, k = history.episode_index(data["policy"], data["episode"], data["timestep"], H)
    for i in np.flatnonzero(k > 0):
        start, length = history.prefix(ep_start[i:i + 1], k[i:i + 1], history.MAX_EVENTS)
        mem = history.gather(ev, start, length, history.MAX_EVENTS)[0, : length[0]]
        mo, ma = split_event(mem)
        steps = mo[..., 0][mo[..., 0] > 0] - 1                   # obs timesteps in memory
        assert steps.max() == data["timestep"][i] - 1            # up to t - 1 ...
        assert data["timestep"][i] not in steps                  # ... never the decision obs itself
        assert np.array_equal(np.sort(steps), np.arange(data["timestep"][i]))
        assert np.array_equal(ma[..., 0], mo[..., 0])            # actions aligned with obs


def test_short_last_chunk_is_zero_padded(data):
    mo, ma = split_event(data["event"])
    _, k = history.episode_index(data["policy"], data["episode"], data["timestep"], H)
    last = np.flatnonzero(np.r_[k[1:] == 0, True])
    for i in last:
        n_valid = int((mo[i, :, 0] > 0).sum())
        assert n_valid >= 1 and np.all(mo[i, n_valid:] == 0) and np.all(ma[i, n_valid:] == 0)


def test_first_decision_has_empty_memory_and_window(data):
    ep_start, k = history.episode_index(data["policy"], data["episode"], data["timestep"], H)
    s, n = history.prefix(ep_start, k, history.MAX_EVENTS)
    assert np.all(n[k == 0] == 0) and n.max() == 49
    s1, n1 = history.prefix(ep_start, k, 1)
    assert np.array_equal(n1, np.minimum(k, 1)) and np.array_equal(s1[k > 0], (ep_start + k - 1)[k > 0])


def test_shuffle_donors(data):
    ep_start, k = history.episode_index(data["policy"], data["episode"], data["timestep"], H)
    split = (data["episode"] >= 8).astype(int)
    donor, alone = history.shuffle_donors(ep_start, k, data["policy"], split, seed=0)
    again, _ = history.shuffle_donors(ep_start, k, data["policy"], split, seed=0)
    assert np.array_equal(donor, again)
    starts = {s: i for i, s in enumerate(ep_start)}
    for i in range(len(k)):
        j = starts[donor[i]]                                      # donor episode's first decision
        assert data["policy"][j] == data["policy"][i] and split[j] == split[i]
        reached = np.flatnonzero((ep_start == donor[i]) & (k == k[i]))
        assert len(reached) == 1                                  # donor reached decision k
    assert (donor != ep_start).sum() == len(k) - alone


def test_memory_value_all_modes():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from memcritic.critics import MemoryValue
    B, W, E = 5, 7, H * (OBS + ACT)
    obs, act = jnp.ones((B, OBS)), jnp.ones((B, H * ACT))
    hist = jax.random.normal(jax.random.PRNGKey(0), (B, W, E))
    hlen, t = jnp.array([0, 1, 3, 7, 7]), jnp.array([0, 8, 24, 56, 56])
    for mode in ("none", "clk", "gru"):
        m = MemoryValue(mode, hidden_dims=(16, 16), num_qs=3, mem_dim=4, gru_dim=8)
        p = m.init(jax.random.PRNGKey(1), obs, act, hist, hlen, t)
        assert m.apply(p, obs, act, hist, hlen, t).shape == (3, B)
    # padding past the length doesn't matter; empty memory is the same for any hist
    q = m.apply(p, obs, act, hist, hlen, t)
    q2 = m.apply(p, obs, act, hist.at[:, 3:].set(9.0).at[0].set(5.0), hlen, t)
    np.testing.assert_allclose(q[:, :3], q2[:, :3], atol=1e-5)
    assert not np.allclose(q[:, 3], q2[:, 3])
    # numpy and jax gather agree
    ev = np.random.default_rng(0).normal(size=(30, E)).astype(np.float32)
    s, n = np.array([0, 5, 28]), np.array([0, 3, 7])
    np.testing.assert_allclose(history.gather(ev, s, n, W), history.gather(jnp.asarray(ev), jnp.asarray(s),
                                                                           jnp.asarray(n), W, xp=jnp))
