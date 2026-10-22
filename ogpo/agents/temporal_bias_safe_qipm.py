"""Temporal-bias min/max-ReLU safe_qipm (subclass; safe_qipm.py is untouched).

The safe_qipm ``max_relu`` arm fuses the K safety heads as

    safety = min(0, max_k Q_k)            Q_k = min over the M ensemble members

so a candidate is penalised only when EVERY head is below 0. This arm moves
each head's threshold from 0 down to a per-state tolerance taken from the
trajectory the sample came from:

    q_k(j)  = min_m Q_k^m(s_{t+j}, a_{t+j : t+j+H})     target critic, j = 0..n-1
    m_k     = mean_j q_k(j)   over steps still in the episode   (stop-gradient)
    tol_k   = clip(-m_k, 0, cap)
    safety  = min(0, max_k (Q_k + tol_k))

A candidate is penalised only when every head is worse than what that head has
been reporting along the trajectory. The cap keeps an absolute anchor: a
trajectory that has been persistently unsafe (m_k << -cap) cannot tolerate
itself, so a stable violation is still penalised by at least |Q_k| - cap.

Each Q value is a full H-step action chunk, so n chunk starts need a window of
n + H - 1 steps; the replay sampler returns them as hist_* keys
(Dataset.sample_sequence(history_length=n)).

Inference has no replay history. sample_actions_BON and get_q_values use
tol_ema, an EMA (tau = safety_margin_tau) of the batch-mean tolerance per head,
stored in the inherited ``safety_margin`` field as a (K,) vector. It starts at 0,
so action selection begins as plain min_max_relu.

Only the configuration this arm was designed for is supported (ensemble Q,
q_agg=min, vanilla advantages, simple_summation); create() rejects the rest
instead of silently falling back.
"""
from functools import partial

import flax
import jax
import jax.numpy as jnp

from ogpo.agents.safe_qipm import (
    OGPOSafeQIPMAgentSharedBase,
    _zero_dual,
    compute_gate_weights,
)

SAFETY_AGG = 'temporal_capped_max_relu'


def _temporal_capped_max_relu(q_safety_per_head, tol):
    """min(0, max_k (Q_k + tol_k)). Heads on axis 0; tol broadcasts against q."""
    return -jax.nn.relu(-jnp.max(q_safety_per_head + tol, axis=0))


def _validate_config(config):
    def need(cond, msg):
        if not cond:
            raise ValueError(f"temporal_bias_safe_qipm: {msg}")
    need(config.get('safety_agg') == SAFETY_AGG,
         f"safety_agg must be '{SAFETY_AGG}', got {config.get('safety_agg')!r}")
    need(config['safety_aggregation_strategy'] == 'simple_summation',
         "safety_aggregation_strategy must be 'simple_summation'")
    need(config['q_agg'] == 'min', f"q_agg must be 'min', got {config['q_agg']!r}")
    need(not config.get('use_mip_q', False), "use_mip_q is not supported")
    need(config.get('adv_strategy', 'vanilla') == 'vanilla', "adv_strategy must be 'vanilla'")
    need(bool(config['action_chunking']), "action_chunking must be true (Q takes an H-step chunk)")
    need(not config.get('useSimBa', False), "useSimBa is not supported")
    need(not config.get('encoder'), "image encoders are not supported (history has no images)")
    # actor_loss subsamples obs when ppo_batch_size < batch_size, which would
    # misalign the per-sample tolerance computed from the full batch.
    need(int(config['ppo_batch_size']) >= int(config['batch_size']),
         "ppo_batch_size must be >= batch_size")
    need(int(config.get('safety_bias_window', 0) or 0) >= 1, "safety_bias_window must be >= 1")
    need(float(config['safety_margin_cap']) > 0, "safety_margin_cap must be > 0")
    need(int(config['no_of_safety_subtasks']) >= 1, "needs at least one safety head")


class OGPOTemporalBiasSafeQIPMAgent(OGPOSafeQIPMAgentSharedBase):
    """safe_qipm with a trajectory-relative, capped per-head safety threshold."""

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, adv_clip_min=None, ex_images=None):
        _validate_config(config)
        # The base create() asserts safety_agg is one of its own modes and reads
        # it for nothing else. Validate under 'max_relu' (the arm this one
        # generalises) and restore the real mode on the same config object,
        # which create() stores and every method reads.
        config['safety_agg'] = 'max_relu'
        agent = super().create(seed, ex_observations, ex_actions, config,
                               adv_clip_min=adv_clip_min, ex_images=ex_images)
        config['safety_agg'] = SAFETY_AGG
        assert agent.config['safety_agg'] == SAFETY_AGG
        K = int(agent.config['no_of_safety_subtasks'])
        return agent.replace(safety_margin=jnp.zeros((K,), jnp.float32))

    # --- the bias ------------------------------------------------------------

    def _history_tolerance(self, batch):
        """Per-sample tolerance from the replay history.

        Returns (tol, m, q_hist, valid): tol and m are (K, B); q_hist is the
        member-min safety Q at every chunk start, (K, B, n); valid is (B, n).
        """
        for key in ('hist_observations', 'hist_actions', 'hist_valid'):
            if key not in batch:
                raise KeyError(
                    f"temporal_bias_safe_qipm needs '{key}' in the batch; sample with "
                    f"history_length=safety_bias_window")
        obs = batch['hist_observations']
        acts = batch['hist_actions']
        valid = batch['hist_valid'].astype(jnp.float32)
        B, n = valid.shape
        S_task = self.config['no_of_subtasks']
        K = self.config['no_of_safety_subtasks']

        enc = self._encode_for_critic(obs.reshape((B * n,) + obs.shape[2:]), use_target=True)
        q = self.critic_network.select('target_critic')(
            enc, acts.reshape(B * n, -1), is_encoded=bool(self.config.encoder))   # (S, M, B*n)
        q_hist = q[S_task:].min(axis=1).reshape(K, B, n)
        # valid[:, 0] is always 1, so the denominator is never 0.
        m = (q_hist * valid[None]).sum(-1) / valid.sum(-1)[None]
        tol = jnp.clip(-m, 0.0, float(self.config['safety_margin_cap']))
        sg = jax.lax.stop_gradient
        return sg(tol), sg(m), sg(q_hist), valid

    # --- training ------------------------------------------------------------

    def actor_loss(self, batch, batch_success, grad_params, success_flag, rng):
        # The advantage computation reads the tolerance from `safety_margin`.
        # Swap in the per-sample (K, B) tolerance for this call only; the
        # agent's stored (K,) EMA is untouched.
        tol, _, _, _ = self._history_tolerance(batch)
        agent = self.replace(safety_margin=tol)
        return OGPOSafeQIPMAgentSharedBase.actor_loss(
            agent, batch, batch_success, grad_params, success_flag, rng)

    def _compute_advantages(self, obs, actions, rng, subtask_status=None, obs_encoded=None, images=None):
        """GRPO advantages with the temporal-bias safety term (ensemble-Q path only)."""
        tol = self.safety_margin
        if jnp.ndim(tol) != 2:
            raise ValueError(
                "_compute_advantages expects the per-sample (K, B) tolerance set by "
                f"actor_loss; got shape {jnp.shape(tol)}")
        assert subtask_status is not None, "_compute_advantages requires subtask_status"
        S_task = self.config['no_of_subtasks']
        S_safety = self.config['no_of_safety_subtasks']
        safety_coeff = self.config['safety_coeff']
        critic_obs_enc = self._encode_for_critic(obs, images=images, use_target=True)
        gate_weights = compute_gate_weights(
            subtask_status_last=subtask_status[:, :S_task],
            gating_mode=self.config.get('advantage_gating_mode', 'grandparent_future'),
            dag=self.config.get('subtask_dag'),
        )

        def compute_q_for_group(actions_g):
            q_vals = self.critic_network.select('target_critic')(
                critic_obs_enc, actions_g, is_encoded=bool(self.config.encoder))   # (S, M, B)
            q_min = q_vals.min(axis=1)                                             # (S, B)
            q_task = (q_min[:S_task] * gate_weights).sum(axis=0)                   # (B,)
            saf = safety_coeff * _temporal_capped_max_relu(q_min[S_task:], tol)   # (B,)
            return q_task + saf, saf

        q_out, saf_per_group = jax.vmap(compute_q_for_group)(actions)             # (G, B)
        adv = q_out - q_out.mean(axis=0, keepdims=True)
        if self.config.adv_clip_min is not None:
            adv = jnp.clip(adv, self.config.adv_clip_min)

        task_per_group = q_out - saf_per_group
        sd_saf = saf_per_group.std(axis=0)
        sd_task = task_per_group.std(axis=0)
        adv_diag = {
            'adv_safety_std_frac': (sd_saf / jnp.maximum(sd_task + sd_saf, 1e-8)).mean(),
            'adv_rank_flip_rate': (q_out.argmax(axis=0) != task_per_group.argmax(axis=0)).mean(),
            'adv_safety_std': sd_saf.mean(),
            'adv_task_std': sd_task.mean(),
            # Fraction of candidate actions the safety term penalises at all.
            'tbias_candidate_activation_rate': (saf_per_group < 0).mean(),
        }
        return adv, q_out, _zero_dual(S_task, S_safety), adv_diag

    def critic_loss(self, batch, grad_params, rng):
        loss, info = OGPOSafeQIPMAgentSharedBase.critic_loss(self, batch, grad_params, rng)
        tol, m, q_hist, valid = self._history_tolerance(batch)
        cap = float(self.config['safety_margin_cap'])
        q0 = q_hist[..., 0]                         # (K, B): Q at the stored (s_t, a_t)
        contrib = _temporal_capped_max_relu(q0, tol)
        stats = {
            'tbias_activation_rate': (contrib < 0).mean(),
            'tbias_contrib_mean': contrib.mean(),
            'tbias_hist_valid_frac': valid.mean(),
        }
        for k in range(q0.shape[0]):
            stats[f'tbias_tol_{k}'] = tol[k].mean()          # read by _next_safety_margin
            stats[f'tbias_m_{k}'] = m[k].mean()
            stats[f'tbias_frac_below_bias_{k}'] = (q0[k] < m[k]).mean()
            stats[f'tbias_frac_cap_binds_{k}'] = (-m[k] > cap).mean()
            stats[f'tbias_tol_ema_{k}'] = self.safety_margin[k]
        return loss, {**info, **stats}

    @staticmethod
    def _next_safety_margin(agent, critic_info):
        """tol_ema <- clip(EMA(tol_ema, batch-mean tol), 0, cap), per head."""
        K = int(agent.config['no_of_safety_subtasks'])
        batch_tol = jnp.stack([critic_info[f'critic/tbias_tol_{k}'] for k in range(K)])
        tau = float(agent.config['safety_margin_tau'])
        cap = float(agent.config['safety_margin_cap'])
        return jnp.clip((1.0 - tau) * agent.safety_margin + tau * batch_tol, 0.0, cap)

    # --- inference -------------------------------------------------------------

    @jax.jit
    def sample_actions_BON(self, observations, rng=None, images=None):
        """Best-of-N with the temporal-bias safety term, using tol_ema (no history at inference)."""
        is_single_obs = observations.ndim == 1
        if is_single_obs:
            observations = observations[None]
            if images is not None:
                images = images[None]
        batch_size = observations.shape[0]
        num_samples = self.config.best_of_n

        rng, action_key = jax.random.split(rng)
        obs_expanded = jnp.repeat(observations, num_samples, axis=0)
        images_expanded = jnp.repeat(images, num_samples, axis=0) if images is not None else None
        sampled_actions, _, _, _ = self.sample_actions_with_noise(
            obs_expanded, rng=action_key, use_target=True, is_encoded=False, images=images_expanded)

        critic_obs = self._encode_for_critic(obs_expanded, images=images_expanded, use_target=True)
        S_task = self.config['no_of_subtasks']
        S = S_task + self.config['no_of_safety_subtasks']
        q_ensemble = self.critic_network.select('target_critic')(
            critic_obs, actions=sampled_actions, is_encoded=bool(self.config.encoder))
        q_ensemble = q_ensemble.reshape(S, self.config.num_qs, batch_size, num_samples)
        # Same member set as the base class: task and safety read the same members.
        if self.config.subsample_bon:
            rng, sub_key = jax.random.split(rng)
            idxs = jax.random.randint(sub_key, (2,), 0, self.config.num_qs)
            q_ensemble = q_ensemble[:, idxs, :, :]
        min_qs = q_ensemble.min(axis=1)                                       # (S, B, N)

        q_safety = _temporal_capped_max_relu(min_qs[S_task:], self.safety_margin[:, None, None])
        q_total = min_qs[:S_task].sum(axis=0) + self.config['safety_coeff'] * q_safety

        best_idx = jnp.argmax(q_total, axis=1)
        actions_reshaped = sampled_actions.reshape(batch_size, num_samples, -1)
        return jax.vmap(lambda a, i: a[i])(actions_reshaped, best_idx)

    @partial(jax.jit, static_argnames=('is_encoded',))
    def get_q_values(self, observations, actions, is_encoded=False, images=None):
        """Logged scalarised objective, with the same tol_ema as action selection."""
        if not is_encoded:
            observations = self._encode_for_critic(observations, images=images, use_target=True)
        actions = jnp.reshape(actions, (actions.shape[0], -1))
        qs = self._get_subtask_q_ensemble(observations, actions)
        S_task = self.config['no_of_subtasks']
        q_min = qs.min(axis=1)                                                 # (S, B)
        q_safety = _temporal_capped_max_relu(q_min[S_task:], self.safety_margin[:, None])
        return q_min[:S_task].sum(axis=0) + self.config['safety_coeff'] * q_safety
