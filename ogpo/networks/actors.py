"""Actor network implementations for reinforcement learning.

This module contains various actor (policy) network architectures including:
- Actor: Standard Gaussian policy with MLP backbone
- ActorVectorField: Flow matching policy with optional FiLM conditioning
- ActorVectorFieldTF: Transformer-based flow matching policy with AdaLN
- ActorVectorFieldSimBa: SimBa architecture-based flow matching policy
- EditPolicy: Simple Gaussian MLP policy for action refinement
- EditActor: Edit actor for DSRL+EXPO that refines flow-refined actions

All actors support optional encoders for observation processing.
"""

from typing import Any, Optional, Sequence, Tuple
from ogpo.networks.modules.time_embedding import SinusoidalTimeEmbedding

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from ogpo.networks.modules import (
    MLP,
    MLPCond,
    MLPWithFiLM,
    SimBaMLP,
    TanhMultivariateNormalDiag,
    AdaLNLayer,
    AdaLNFinalLayer,
)


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


class Actor(nn.Module):
    """Standard Gaussian policy with MLP backbone.

    Attributes:
        hidden_dims: Hidden layer dimensions for the MLP.
        action_dim: Dimension of the action space.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value for log standard deviation.
        log_std_max: Maximum value for log standard deviation.
        tanh_squash: Whether to apply tanh squashing to actions.
        state_dependent_std: Whether standard deviation depends on state.
        const_std: Whether to use constant (zero) standard deviation.
        final_fc_init_scale: Initialization scale for final fully connected layers.
        encoder: Optional encoder module for observations.
        low: Lower bound for action space (used with tanh_squash).
        high: Upper bound for action space (used with tanh_squash).
    """
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = True
    const_std: bool = False
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, observations, temperature=1.0, is_encoded=False):
        if self.encoder is not None and not is_encoded:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TanhMultivariateNormalDiag(distribution, low=self.low, high=self.high)

        return distribution


class ActorVectorField(nn.Module):
    """Flow matching policy with optional FiLM conditioning.

    This actor learns a vector field for flow matching policies.
    Supports both standard MLP and FiLM-conditioned architectures.

    Attributes:
        hidden_dims: Hidden layer dimensions for the MLP.
        action_dim: Dimension of the action space.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module for observations.
        use_film: Whether to use FiLM conditioning.
        use_denoiser: Whether to include a denoiser output.
    """
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None
    use_film: bool = False
    use_denoiser: bool = False
    time_embedding: Optional[nn.Module] = None

    def setup(self) -> None:
        if self.use_film:
            self.mlp = MLPWithFiLM(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm
            )
        else:
            self.mlp = MLP(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm
            )

        if self.use_denoiser:
            if self.use_film:
                self.denoiser_mlp = MLPWithFiLM(
                    (*self.hidden_dims, self.action_dim),
                    activate_final=False,
                    layer_norm=self.layer_norm
                )
            else:
                self.denoiser_mlp = MLP(
                    (*self.hidden_dims, self.action_dim),
                    activate_final=False,
                    layer_norm=self.layer_norm
                )

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    @nn.compact
    def __call__(self, observations, actions, times=None, dt=None, is_encoded=False, return_denoiser=False, images=None):
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        # Embed timestep (sinusoidal or raw scalar)
        if times is not None and self.time_embedding is not None:
            times = self.time_embedding(times)

        if self.use_film:
            cond = observations
            if times is not None:
                cond = jnp.concatenate([cond, times], axis=-1)
            if dt is not None:
                cond = jnp.concatenate([cond, dt], axis=-1)
            # Velocity output
            v = self.mlp(actions, cond)
        else:
            if times is None:
                inputs = jnp.concatenate([observations, actions], axis=-1)
            else:
                inputs = jnp.concatenate([observations, actions, times], axis=-1)
            if dt is not None:
                inputs = jnp.concatenate([inputs, dt], axis=-1)
            # Velocity output
            v = self.mlp(inputs)

        if return_denoiser:
            if not self.use_denoiser:
                raise ValueError("Actor was initialized with use_denoiser=False")

            if self.use_film:
                z, _ = self.denoiser_mlp(actions, cond)  # MLPWithFiLM returns tuple
            else:
                z = self.denoiser_mlp(inputs)
            return v, z

        return v


class ActorVectorFieldTF(nn.Module):
    """Decoder-only GPT flow matching policy with AdaLN conditioning.

    Uses a causal transformer with adaptive layer normalization (AdaLN)
    for conditioning on observations and time steps. Actions are the
    input/output tokens; states and times condition via AdaLN.

    Attributes:
        hidden_dim: Hidden dimension size for transformer.
        action_dim: Dimension of the action space.
        action_chunk_size: Number of actions in a chunk (temporal dimension).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module for observations.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout_rate: Dropout rate for transformer.
        use_denoiser: Whether to include a denoiser output head.
    """
    hidden_dim: int
    action_dim: int
    action_chunk_size: int = 4  # T
    layer_norm: bool = False
    encoder: nn.Module = None
    num_layers: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.0
    use_denoiser: bool = False

    def setup(self) -> None:
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"

        self.action_proj = nn.Dense(self.hidden_dim, kernel_init=default_init())
        self.cond_proj = nn.Dense(self.hidden_dim, kernel_init=default_init())
        self.time_proj = nn.Dense(self.hidden_dim, kernel_init=default_init())
        self.dt_proj = nn.Dense(self.hidden_dim, kernel_init=default_init())

        # Learned positional embeddings for action tokens
        self.pos_embed = self.param(
            'pos_embed',
            nn.initializers.normal(0.02),
            (1, self.action_chunk_size, self.hidden_dim),
        )

        self.transformer_layers = [
            AdaLNLayer(self.hidden_dim, self.num_heads, self.hidden_dim * 4,
                       self.dropout_rate)
            for _ in range(self.num_layers)
        ]

        self.final_layer = AdaLNFinalLayer(self.hidden_dim, self.action_dim)

        if self.use_denoiser:
            self.denoiser_final_layer = AdaLNFinalLayer(self.hidden_dim, self.action_dim)

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    @nn.compact
    def __call__(self, observations, actions, times=None, dt=None, is_encoded=False,
                 return_denoiser=False, deterministic=True, images=None):
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        # Reshape flat actions (B, T*a_dim) -> (B, T, a_dim)
        actions = actions.reshape(-1, self.action_chunk_size, self.action_dim)

        # Project action tokens and add positional embeddings
        x = self.action_proj(actions) + self.pos_embed  # (B, T, hidden_dim)

        # Build conditioning: state + time (broadcast over T via AdaLN)
        cond = self.cond_proj(observations)  # (B, hidden_dim)
        if times is not None:
            cond = cond + self.time_proj(times)
        if dt is not None:
            cond = cond + self.dt_proj(dt)

        # Causal mask: token t only attends to positions <= t
        T = self.action_chunk_size
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[None, None, :, :]

        for layer in self.transformer_layers:
            x = layer(x, cond, mask=causal_mask, deterministic=deterministic)

        v = self.final_layer(x, cond)  # (B, T, action_dim)
        v = v.reshape(-1, self.action_chunk_size * self.action_dim)  # (B, T*a_dim)

        if self.use_denoiser:
            z = self.denoiser_final_layer(x, cond)  # (B, T, action_dim)
            z = z.reshape(-1, self.action_chunk_size * self.action_dim)
            if return_denoiser:
                return v, z

        return v


class ActorVectorFieldSimBa(nn.Module):
    """Actor using SimBa architecture for vector field prediction.

    SimBa (Simple Baseline) architecture uses residual blocks with
    running statistics normalization for improved training stability.

    Attributes:
        hidden_dims: Hidden layer dimensions for SimBa MLP.
        action_dim: Dimension of the action space.
        encoder: Optional encoder module for observations.
        rs_norm_momentum: Momentum for running statistics normalization.
        rs_norm_epsilon: Epsilon for numerical stability in normalization.
    """
    hidden_dims: Sequence[int]
    action_dim: int
    encoder: Optional[nn.Module] = None
    rs_norm_momentum: float = 0.99
    rs_norm_epsilon: float = 1e-8
    time_embedding: Optional[nn.Module] = None

    def setup(self):
        self.simba_mlp = SimBaMLP(
            hidden_dims=(*self.hidden_dims, self.action_dim),
            rs_norm_momentum=self.rs_norm_momentum,
            rs_norm_epsilon=self.rs_norm_epsilon
        )

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions, times=None, dt=None, is_encoded=False, images=None):
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        # Embed timestep (sinusoidal or raw scalar)
        if times is not None and self.time_embedding is not None:
            times = self.time_embedding(times)

        # Concatenate all inputs
        inputs = [observations, actions]
        if times is not None:
            inputs.append(times)
        if dt is not None:
            inputs.append(dt)

        x = jnp.concatenate(inputs, axis=-1)

        # Pass through SimBa MLP
        return self.simba_mlp(x)


class EditPolicy(nn.Module):
    """Simple Gaussian MLP policy that outputs an action edit (delta).

    Conditioned on the observation and the base policy's action.
    Implements pi_edit(a_hat | s, a_base) for action refinement.

    Attributes:
        hidden_dims: Hidden layer dimensions for the MLP.
        action_dim: Dimension of the action space.
        encoder: Optional encoder module for observations.
        log_std_min: Minimum value for log standard deviation.
        log_std_max: Maximum value for log standard deviation.
        edit_bound: Maximum magnitude of the edit (applied via tanh).
        layer_norm: Whether to apply layer normalization.
    """
    hidden_dims: Tuple[int, ...]
    action_dim: int
    encoder: nn.Module = None
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    edit_bound: float = 0.1
    layer_norm: bool = True

    def setup(self):
        if self.encoder:
            self.encoder_module = self.encoder

        mlp_dims = self.hidden_dims

        self.net = nn.Sequential([
            nn.Dense(mlp_dims[0]),
            nn.LayerNorm() if self.layer_norm else lambda x: x,
            nn.relu,
            *[
                nn.Sequential([
                    nn.Dense(dim),
                    nn.LayerNorm() if self.layer_norm else lambda x: x,
                    nn.relu,
                ]) for dim in mlp_dims[1:]
            ]
        ])

        self.mean_layer = nn.Dense(self.action_dim)
        self.log_std_layer = nn.Dense(self.action_dim)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, base_actions: jnp.ndarray, is_encoded: bool = False) -> distrax.Distribution:
        if self.encoder and not is_encoded:
            observations = self.encoder_module(observations)

        if observations.ndim > 2:
            observations = observations.reshape(observations.shape[0], -1)

        x = jnp.concatenate([observations, base_actions], axis=-1)
        x = self.net(x)
        mean = self.mean_layer(x)
        mean = nn.tanh(mean) * self.edit_bound
        log_std = self.log_std_layer(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        std = jnp.exp(log_std)
        return distrax.Normal(mean, std)


class EditActor(nn.Module):
    """Edit actor for DSRL+EXPO that refines flow-refined actions.

    Conditioned on observations and base actions (output of flow refinement).
    Outputs bounded residuals that are added to base actions.
    Follows the EXPO paper's edit actor design adapted for DSRL's architecture.

    Attributes:
        hidden_dims: Hidden layer dimensions for the MLP.
        action_dim: Dimension of the action space.
        encoder: Optional encoder module for observations.
        log_std_min: Minimum value for log standard deviation.
        log_std_max: Maximum value for log standard deviation.
        edit_action_scale: Maximum magnitude of edit residuals.
        layer_norm: Whether to apply layer normalization.
        state_dependent_std: Whether standard deviation depends on state.
        final_fc_init_scale: Initialization scale for final fully connected layers.
    """
    hidden_dims: Sequence[int]
    action_dim: int
    encoder: nn.Module = None
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    edit_action_scale: float = 0.5  # Maximum edit magnitude
    layer_norm: bool = False
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1e-2

    def setup(self):
        # MLP backbone
        self.edit_net = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm
        )

        # Output layers for mean and std
        self.mean_net = nn.Dense(
            self.action_dim,
            kernel_init=default_init(self.final_fc_init_scale)
        )

        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim,
                kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            self.log_stds = self.param(
                'log_stds',
                nn.initializers.zeros,
                (self.action_dim,)
            )

    def __call__(self, observations, base_actions, is_encoded=False, temperature=1.0):
        """Compute distribution over edit residuals.

        Args:
            observations: Current observations.
            base_actions: Actions from flow refinement (noise → flow → base_actions).
            is_encoded: Whether observations are already encoded.
            temperature: Temperature for exploration.

        Returns:
            Distribution over edit residuals (will be scaled and added to base_actions).
        """
        if self.encoder is not None and not is_encoded:
            observations = self.encoder(observations)

        # Concatenate observations and base actions
        inputs = jnp.concatenate([observations, base_actions], axis=-1)
        outputs = self.edit_net(inputs)

        # Compute mean and std
        means = self.mean_net(outputs)

        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        # Create distribution - note we use TanhMultivariateNormalDiag to bound edits
        distribution = distrax.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds) * temperature
        )

        # Apply tanh squashing to bound edits to [-edit_action_scale, +edit_action_scale]
        distribution = TanhMultivariateNormalDiag(
            distribution,
            low=-self.edit_action_scale,
            high=self.edit_action_scale
        )

        return distribution


__all__ = [
    'Actor',
    'ActorVectorField',
    'ActorVectorFieldTF',
    'ActorVectorFieldSimBa',
    'EditPolicy',
    'EditActor',
]
