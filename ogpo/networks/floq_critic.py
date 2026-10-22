"""FloQ flow-matching critic for OGPO, with QIPM per-subtask compatibility.

Ports the FloQ flow-matching critic from acorn (acorn/nn/jax/floq_critic.py)
into OGPO conventions. The critic combines:

  - CriticVectorField: ensembled MLP that learns a velocity field which
    transports a noise distribution to TD targets via Euler integration.
    HL-Gauss embedding for the return scalar input, Fourier embedding for
    flow time. Per-subtask velocity output for QIPM compatibility.

  - DistilledCritic: a fast standard ensembled MLP critic that is distilled
    from the velocity field. Used at inference time so the actor and Best-of-N
    don't pay flow-integration cost on every call. Implemented by reusing
    OGPO's existing Value(critic_loss_type='mse') so the output shape matches
    the (S, M, B) / (M, B) convention exactly.

  - FloQCritic: the wrapper module. __call__ returns the distilled critic's
    output for drop-in compatibility with code that calls
    `network.select('critic')(obs, actions=...)`. velocity() and
    integrate_flow() expose the velocity field for the training step.

Reference: FloQ (github.com/CMU-AIRe/floq) and acorn/acorn/nn/jax/floq_critic.py
Note: We use the bug-fixed Fourier time embedding (cos(freqs * t), not the
broken cos(t) variant from the upstream FloQ implementation).
"""
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from ogpo.networks.critics import Value, ensemblize
from ogpo.networks.modules.mlp import MLP


# =============================================================================
# HL-Gauss embedding helpers (ported from acorn/acorn/nn/jax/floq_velocity_field.py)
#
# Used internally by CriticVectorField to embed the per-subtask z_t scalar
# inputs as soft categorical distributions over a fixed return support. NOT
# the same as OGPO's HL-Gauss critic-output path in Value (which uses fixed
# linspace + softmax). These are kept private to floq_critic.py.
# =============================================================================

def compute_support(q_min: float, q_max: float, num_bins: int) -> jnp.ndarray:
    """Evenly-spaced bin edges for the HL-Gauss return embedding."""
    return q_min + jnp.arange(num_bins) * (q_max - q_min) / (num_bins - 1)


def to_probs(target: jnp.ndarray, support: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """Soft-bin a tensor of scalar return values into categorical probs.

    Uses the Gaussian CDF over a fixed support, then takes finite differences
    between adjacent bin edges to produce probabilities.

    Args:
        target: scalar return values, any shape (..,).
        support: bin edges, shape (num_bins,).
        sigma: kernel width (already pre-multiplied by bin_width by the caller).

    Returns:
        bin probabilities, shape (..., num_bins - 1).
    """
    cdf_evals = jax.scipy.special.erf(
        (support - target[..., None]) / (jnp.sqrt(2.0) * sigma)
    )
    z = cdf_evals[..., -1:] - cdf_evals[..., :1]
    bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    return bin_probs / z


# =============================================================================
# CriticVectorField — the slow flow-matching head
# =============================================================================

class CriticVectorField(nn.Module):
    """Per-subtask velocity field v(obs, action, z_t, t) for flow-matching Q.

    The MLP trunk processes all S subtasks jointly with output dim = num_subtasks.
    Inputs and outputs:
      observations: (B, obs_dim)
      actions:      (B, action_dim)
      returns z_t:  (E, B, S)   per-subtask scalar at flow time t
      times:        (B, 1)      flow time t in [0, 1]
      output:       (E, B, S)   per-subtask velocity

    The ensemble vmap is applied via OGPO's `ensemblize` helper with `in_axes=0`,
    so we tile (obs ⊕ action) to the leading ensemble dim before concatenating
    the per-ensemble z_t. This is acorn's pattern, adapted to OGPO's MLP class
    (whose output dim is the last element of `hidden_dims`, not a separate kwarg).
    """
    hidden_dims: Sequence[int] = (512, 512, 512, 512)
    num_ensembles: int = 2
    num_subtasks: int = 1
    layer_norm: bool = True
    embed_time: bool = True
    time_embed_dim: int = 64
    use_prob_embed: bool = True   # HL-Gauss for z_t input embedding
    return_embed_bins: int = 51
    sigma: float = 16.0
    q_min: float = 0.0
    q_max: float = 0.0

    def setup(self):
        # OGPO's MLP doesn't have an output_dim kwarg — the last element of
        # hidden_dims IS the output dim. Append num_subtasks to hidden_dims.
        full_hidden_dims = tuple(self.hidden_dims) + (self.num_subtasks,)
        EnsembleMLP = ensemblize(MLP, self.num_ensembles, in_axes=0)
        self.mlp = EnsembleMLP(
            hidden_dims=full_hidden_dims,
            layer_norm=self.layer_norm,
            activate_final=False,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        returns: jnp.ndarray,
        times: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict per-subtask velocity. Returns (E, B, S)."""
        E = self.num_ensembles
        S = self.num_subtasks

        # Defensive: handle the no-batch-dim init case where observations is
        # 1D (obs_dim,) — happens when main.py passes example_batch single-row
        # samples to network.init. Mirrors OGPO's Value at critics.py:127.
        if observations.ndim == 1:
            observations = observations[None]   # (obs_dim,) -> (1, obs_dim)
        if actions.ndim == 1:
            actions = actions[None]             # (action_dim,) -> (1, action_dim)
        B = observations.shape[0]

        # Tile (obs ⊕ action) to ensemble dim
        obs_action = jnp.concatenate([observations, actions], axis=-1)  # (B, obs+act)
        obs_action = jnp.tile(obs_action[None], [E, 1, 1])              # (E, B, obs+act)

        # Embed returns (per-subtask HL-Gauss)
        if self.use_prob_embed:
            support = compute_support(self.q_min, self.q_max, self.return_embed_bins)
            bin_width = support[1] - support[0]
            # returns shape: (E, B, S). Soft-bin each scalar -> (E, B, S, num_bins-1)
            returns_embed = to_probs(returns, support, self.sigma * bin_width)
            # Flatten the (S, num_bins-1) trailing dims into a single feature dim
            returns_embed = returns_embed.reshape(E, B, S * (self.return_embed_bins - 1))
        else:
            returns_embed = (returns - self.q_min) / (self.q_max - self.q_min + 1e-8)
            returns_embed = returns_embed.reshape(E, B, S)

        # Embed times (Fourier cos features). Acorn line 138-142 fixes a bug
        # in the upstream FloQ where `cos(times)` was used instead of
        # `cos(freqs * times)`. We use the corrected version.
        if self.embed_time:
            freqs = jnp.arange(1, self.time_embed_dim + 1, dtype=jnp.float32) * jnp.pi
            times_embed = jnp.cos(freqs * times)                        # (B, time_embed_dim)
            times_embed = jnp.tile(times_embed[None], [E, 1, 1])        # (E, B, time_embed_dim)
        else:
            times_embed = jnp.tile(times[None], [E, 1, 1])              # (E, B, 1)

        inputs = jnp.concatenate([obs_action, returns_embed, times_embed], axis=-1)
        return self.mlp(inputs)  # (E, B, num_subtasks)


# =============================================================================
# FloQCritic — the wrapper combining velocity field + distilled head
# =============================================================================

class FloQCritic(nn.Module):
    """Drop-in replacement for OGPO's `Value` (MSE path) with flow-matching training.

    External interface (matches Value(critic_loss_type='mse')):
      __call__(obs, action, ...) -> (M, B) when num_subtasks=1, (S, M, B) otherwise.

    Internal training interface (used by floq_helpers.py):
      velocity(obs, action, z, t)              -> (E, B, S)
      integrate_flow(obs, action, noise_ratios) -> (E, B, S)

    The distilled head is OGPO's existing Value(critic_loss_type='mse')
    instantiated with matching num_subtasks and num_ensembles=distill_num_ensembles.
    This guarantees output-shape compatibility with sac_qipm's actor_loss.
    """
    # Velocity field config
    hidden_dims: Sequence[int] = (512, 512, 512, 512)
    flow_num_ensembles: int = 2
    layer_norm: bool = True
    embed_time: bool = True
    time_embed_dim: int = 64
    use_prob_embed: bool = True
    return_embed_bins: int = 51
    sigma: float = 16.0
    q_min: float = 0.0
    q_max: float = 0.0
    # Distilled head config
    distill_hidden_dims: Sequence[int] = (512, 512, 512, 512)
    distill_num_ensembles: int = 10
    # Flow integration
    critic_flow_steps: int = 4
    noise_min: float = 0.0
    noise_max: float = 0.0
    # Per-subtask
    num_subtasks: int = 1

    def setup(self):
        self.velocity_field = CriticVectorField(
            hidden_dims=self.hidden_dims,
            num_ensembles=self.flow_num_ensembles,
            num_subtasks=self.num_subtasks,
            layer_norm=self.layer_norm,
            embed_time=self.embed_time,
            time_embed_dim=self.time_embed_dim,
            use_prob_embed=self.use_prob_embed,
            return_embed_bins=self.return_embed_bins,
            sigma=self.sigma,
            q_min=self.q_min,
            q_max=self.q_max,
            name="velocity_field",
        )
        # Reuse OGPO's existing Value as the distilled head — guaranteed to
        # match the (S, M, B) / (M, B) shape convention sac_qipm expects.
        self.distilled = Value(
            hidden_dims=self.distill_hidden_dims,
            layer_norm=self.layer_norm,
            num_ensembles=self.distill_num_ensembles,
            critic_loss_type='mse',
            num_subtasks=self.num_subtasks,
            name="distilled",
        )

    def __call__(self, observations, actions, **kwargs):
        """Fast inference via the distilled critic. Drop-in for Value(...)."""
        # Force velocity_field param creation at init time so both submodules'
        # params get registered under one FloQCritic param tree. The distilled
        # head is called below regardless; without this dummy call, the
        # velocity_field would not be initialized.
        if self.is_initializing():
            E = self.flow_num_ensembles
            B = observations.shape[0] if observations.ndim > 1 else 1
            S = self.num_subtasks
            dummy_z = jnp.zeros((E, B, S), dtype=jnp.float32)
            dummy_t = jnp.zeros((B, 1), dtype=jnp.float32)
            self.velocity_field(observations, actions, dummy_z, dummy_t)
        return self.distilled(observations, actions=actions, **kwargs)

    def velocity(self, observations, actions, z, t):
        """Velocity field forward pass. Returns (E, B, S)."""
        return self.velocity_field(observations, actions, z, t)

    def integrate_flow(self, observations, actions, noise_ratios):
        """Euler integration of the velocity field.

        Args:
            observations: (B, obs_dim)
            actions:      (B, action_dim)
            noise_ratios: (B, S) — independent per subtask, in [0, 1]

        Returns:
            (E, B, S) — per-subtask flow Q values per ensemble member.
        """
        E = self.flow_num_ensembles
        K = self.critic_flow_steps
        B = observations.shape[0]

        # Initial noise (B, S) -> shared across ensemble members (acorn pattern)
        z = self.noise_min * (1 - noise_ratios) + self.noise_max * noise_ratios  # (B, S)
        z = jnp.tile(z[None], [E, 1, 1])  # (E, B, S)

        for i in range(K):
            t = jnp.full((B, 1), i / K, dtype=jnp.float32)
            v = self.velocity_field(observations, actions, z, t)
            z = z + v / K
        return z
