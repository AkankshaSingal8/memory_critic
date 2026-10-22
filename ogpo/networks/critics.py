"""Critic/Value network architectures for OGPO.

This module contains various critic (Q-function/value function) network implementations:
- Value: Standard MLP-based critic with optional FiLM conditioning
- ValueTF: Transformer-based critic for sequential action chunks
- ValueSimBa: SimBa architecture-based critic

All critics support:
- Ensemble learning (multiple Q-functions)
- Distributional RL (HL-Gauss loss)
- Optional encoder integration
"""

from typing import Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from ogpo.networks.modules import (
    MLP,
    MLPCond,
    MLPWithFiLM,
    SimBaMLP,
    AdaLNLayer,
)


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True, 'dropout': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Value(nn.Module):
    """Standard MLP-based critic network.

    Supports:
    - Standard MLP or FiLM-conditioned architecture
    - Ensemble of Q-functions
    - MSE or HL-Gauss distributional critic
    - Optional encoder for observations

    Attributes:
        hidden_dims: Hidden layer dimensions
        layer_norm: Whether to use layer normalization
        num_ensembles: Number of ensemble members (default: 2)
        encoder: Optional encoder module for observations
        critic_loss_type: 'mse' or 'hlgauss' for distributional critic
        num_bins: Number of bins for HL-Gauss critic
        q_min: Minimum Q-value for HL-Gauss
        q_max: Maximum Q-value for HL-Gauss
        action_repeat: Whether to use MLPCond for action repetition
        use_film: Whether to use FiLM conditioning
    """
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: float = None
    q_max: float = None
    action_repeat: bool = False
    use_film: bool = False
    num_subtasks: int = 1

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1
        final_output_dim = num_output * self.num_subtasks if self.num_subtasks > 1 else num_output

        if self.use_film:
            mlp_class = MLPWithFiLM
        else:
            mlp_class = MLPCond if self.action_repeat else MLP

        if self.num_ensembles > 1:
            if self.use_film:
                # For FiLM, ensemblize with two inputs (x, cond)
                mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=(None, None), out_axes=0)
            else:
                mlp_class = ensemblize(mlp_class, self.num_ensembles)

        value_net = mlp_class(
            (*self.hidden_dims, final_output_dim),
            activate_final=False,
            layer_norm=self.layer_norm
        )
        self.value_net = value_net

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions=None, return_logits=False, is_encoded=False, images=None):
        """Forward pass through the critic.

        Args:
            observations: State observations
            actions: Actions (required for FiLM or action_repeat mode)
            return_logits: Whether to return logits for HL-Gauss critic
            is_encoded: Whether observations are already encoded
            images: Optional image observations for VisionProprioEncoder

        Returns:
            Q-values of shape (M, B) when num_subtasks=1, or (S, M, B) when num_subtasks>1.
            Optionally with logits for distributional critic.
        """
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)
        batch_size = observations.shape[0] if observations.ndim > 1 else 1

        if self.use_film:
            # FiLM: observations condition the network, actions are the input
            assert actions is not None, "FiLM critic requires actions"
            if self.critic_loss_type == 'hlgauss':
                q_logits_flat = self.value_net(actions, observations)
                if self.num_subtasks > 1:
                    q_logits = q_logits_flat.reshape(self.num_ensembles, batch_size, self.num_subtasks, self.num_bins)
                    q_logits = jnp.transpose(q_logits, (2, 0, 1, 3))  # (S, M, B, num_bins)
                else:
                    q_logits = q_logits_flat
                v = jnp.sum(
                    jax.nn.softmax(q_logits, axis=-1)
                    * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1,
                )
                if return_logits:
                    return v, q_logits
            else:
                v_flat = self.value_net(actions, observations)
                if self.num_subtasks > 1:
                    v = v_flat.reshape(self.num_ensembles, batch_size, self.num_subtasks)
                    v = jnp.transpose(v, (2, 0, 1))  # (S, M, B)
                else:
                    v = v_flat.squeeze(-1)
        else:
            # Standard MLP path
            inputs = [observations]
            if actions is not None and not self.action_repeat:
                inputs.append(actions)
            inputs = jnp.concatenate(inputs, axis=-1)

            if self.critic_loss_type == 'hlgauss':
                if self.action_repeat:
                    q_logits_flat = self.value_net(inputs, actions)
                else:
                    q_logits_flat = self.value_net(inputs)
                if self.num_subtasks > 1:
                    q_logits = q_logits_flat.reshape(self.num_ensembles, batch_size, self.num_subtasks, self.num_bins)
                    q_logits = jnp.transpose(q_logits, (2, 0, 1, 3))  # (S, M, B, num_bins)
                else:
                    q_logits = q_logits_flat
                v = jnp.sum(
                    jax.nn.softmax(q_logits, axis=-1)
                    * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1,
                )
                if return_logits:
                    return v, q_logits
            elif self.critic_loss_type == 'mse':
                if self.action_repeat:
                    v_flat = self.value_net(inputs, actions)
                else:
                    v_flat = self.value_net(inputs)
                if self.num_subtasks > 1:
                    v = v_flat.reshape(self.num_ensembles, batch_size, self.num_subtasks)
                    v = jnp.transpose(v, (2, 0, 1))  # (S, M, B)
                else:
                    v = v_flat.squeeze(-1)
            else:
                raise ValueError(f"Unknown critic loss type: {self.critic_loss_type}")

        if self.num_subtasks == 1:
            return v
        return v


class CompositeValue(nn.Module):
    """Task critic and a separate safety critic, concatenated on the head axis.

    The task Value reads observations[..., :task_obs_dim]; the safety Value
    reads observations[..., safety_obs_idx] (robot state incl. joints, no
    objects) -- the network scripts/safety_pretrain/safety_value.py pretrains,
    so its params restore into the 'safety' subtree unchanged. Output is
    (num_task_heads + num_safety_heads, M, B), task heads first: the same shape
    and order as the shared-trunk Value(num_subtasks=S+K), so every caller in
    the agent is unaffected. MSE only, no encoder.
    """
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    num_task_heads: int = 4
    num_safety_heads: int = 4
    task_obs_dim: int = 53
    safety_obs_idx: Sequence[int] = ()

    def setup(self):
        assert self.num_task_heads > 1 and self.num_safety_heads > 1, (
            "Value drops the head axis for a single head; composite needs >= 2 of each")
        assert len(self.safety_obs_idx) > 0, "safety_obs_idx must name the safety critic's dims"
        self.task = Value(hidden_dims=self.hidden_dims, layer_norm=self.layer_norm,
                          num_ensembles=self.num_ensembles, num_subtasks=self.num_task_heads)
        self.safety = Value(hidden_dims=self.hidden_dims, layer_norm=self.layer_norm,
                            num_ensembles=self.num_ensembles, num_subtasks=self.num_safety_heads)

    def __call__(self, observations, actions=None, return_logits=False, is_encoded=False, images=None):
        assert not return_logits, "CompositeValue is MSE-only"
        idx = jnp.asarray(self.safety_obs_idx)
        q_task = self.task(observations[..., :self.task_obs_dim], actions)
        q_safety = self.safety(jnp.take(observations, idx, axis=-1), actions)
        return jnp.concatenate([q_task, q_safety], axis=0)


class ValuePerSubtask(nn.Module):
    """Per-subtask critic: separate parameters for each subtask via nn.vmap.

    Output shape matches `Value(num_subtasks=S)` exactly: (S, M, B) for MSE,
    plus (S, M, B, num_bins) logits when HL-Gauss + return_logits=True.

    The encoder is held outside the vmap (shared across subtasks). Only the
    MLP trunk and final layer are duplicated S times.
    """
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: float = None
    q_max: float = None
    action_repeat: bool = False
    use_film: bool = False
    num_subtasks: int = 1

    def setup(self):
        inner_cls = nn.vmap(
            Value,
            variable_axes={'params': 0, 'intermediates': 0},
            split_rngs={'params': True, 'dropout': True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_subtasks,
        )
        self.value_net = inner_cls(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            num_ensembles=self.num_ensembles,
            encoder=None,
            critic_loss_type=self.critic_loss_type,
            num_bins=self.num_bins,
            q_min=self.q_min,
            q_max=self.q_max,
            action_repeat=self.action_repeat,
            use_film=self.use_film,
            num_subtasks=1,
        )

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions=None, return_logits=False, is_encoded=False, images=None):
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)
        return self.value_net(
            observations,
            actions=actions,
            return_logits=return_logits,
            is_encoded=True,
        )


class ScalarGoalConditionedCritic(nn.Module):
    """Single shared scalar critic queried at S one-hot subtask goals, stacked to (S, M, B).

    Drop-in replacement for `Value(num_subtasks=S)` that ablates per-subtask
    output heads down to one shared scalar Q-function differentiated only by a
    learned goal-embedding projection of a one-hot subtask index. The one-hot
    is generated INTERNALLY — no goal suffix needed in observations.

    Forward pass: encode obs once, then for each subtask i in 0..S-1, project
    one_hot(i) through Dense(goal_embed_dim)+gelu, concatenate to obs, call the
    inner Value (num_subtasks=1, num_ensembles=M) → (M, B). Stack S calls along
    a new leading axis → (S, M, B). Output shape matches Value(num_subtasks=S).
    """
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: float = None
    q_max: float = None
    action_repeat: bool = False
    use_film: bool = False
    num_subtasks: int = 1
    goal_embed_dim: int = 32

    def setup(self):
        self.value_net = Value(
            hidden_dims=self.hidden_dims,
            layer_norm=self.layer_norm,
            num_ensembles=self.num_ensembles,
            encoder=None,
            critic_loss_type=self.critic_loss_type,
            num_bins=self.num_bins,
            q_min=self.q_min,
            q_max=self.q_max,
            action_repeat=self.action_repeat,
            use_film=self.use_film,
            num_subtasks=1,
        )
        self.goal_proj = nn.Dense(self.goal_embed_dim)

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions=None, return_logits=False, is_encoded=False, images=None):
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        S = self.num_subtasks
        goals = jnp.eye(S, dtype=jnp.float32)             # (S, S)
        goal_embeds = nn.gelu(self.goal_proj(goals))      # (S, goal_embed_dim)

        qs = []
        logits_list = []
        for i in range(S):
            ge_i = jnp.broadcast_to(
                goal_embeds[i],
                (*observations.shape[:-1], self.goal_embed_dim),
            )
            obs_i = jnp.concatenate([observations, ge_i], axis=-1)
            out_i = self.value_net(
                obs_i,
                actions=actions,
                return_logits=return_logits,
                is_encoded=True,
            )
            if return_logits and self.critic_loss_type == 'hlgauss':
                v_i, logit_i = out_i
                qs.append(v_i)
                logits_list.append(logit_i)
            else:
                qs.append(out_i)

        v = jnp.stack(qs, axis=0)                         # (S, M, B)
        if return_logits and self.critic_loss_type == 'hlgauss':
            logits = jnp.stack(logits_list, axis=0)       # (S, M, B, num_bins)
            return v, logits
        return v


class ValueTF(nn.Module):
    """Causal transformer critic for sequential action chunks.

    Uses AdaLN (Adaptive Layer Normalization) conditioned on observations.
    Actions are input tokens with causal masking and positional embeddings.
    The sequence is pooled (mean over T) then projected to a scalar Q-value.

    Attributes:
        hidden_dim: Hidden dimension (must be divisible by num_heads)
        action_dim: Action dimension
        action_chunk_size: Number of action steps in sequence
        layer_norm: Whether to use layer normalization
        num_ensembles: Number of ensemble members
        encoder: Optional encoder for observations
        critic_loss_type: 'mse' or 'hlgauss'
        num_bins: Number of bins for HL-Gauss
        q_min: Minimum Q-value for HL-Gauss
        q_max: Maximum Q-value for HL-Gauss
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout_rate: Dropout rate
    """
    hidden_dim: int
    action_dim: int
    action_chunk_size: int = 1
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: float = None
    q_max: float = None
    num_layers: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.0

    def setup(self):
        assert self.hidden_dim % self.num_heads == 0

        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        class TransformerCritic(nn.Module):
            hidden_dim: int
            action_chunk_size: int
            num_layers: int
            num_heads: int
            dropout_rate: float
            num_output: int

            @nn.compact
            def __call__(self, observations, actions, deterministic=True):
                B = actions.shape[0]
                T = self.action_chunk_size

                x = nn.Dense(self.hidden_dim, kernel_init=default_init())(actions)

                # Learned positional embeddings
                pos_embed = self.param(
                    'pos_embed',
                    nn.initializers.normal(0.02),
                    (1, T, self.hidden_dim),
                )
                x = x + pos_embed

                cond = nn.Dense(self.hidden_dim, kernel_init=default_init())(observations)

                # Causal mask
                causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[None, None, :, :]

                for _ in range(self.num_layers):
                    x = AdaLNLayer(
                        self.hidden_dim, self.num_heads, self.hidden_dim * 4,
                        self.dropout_rate
                    )(x, cond, mask=causal_mask, deterministic=deterministic)

                # Pool over sequence: (B, T, hidden_dim) -> (B, hidden_dim)
                x = jnp.mean(x, axis=1)
                return nn.Dense(self.num_output, kernel_init=default_init())(x)

        self.value_net = ensemblize(TransformerCritic, self.num_ensembles,
            in_axes=(None, None),
            out_axes=0,
        )(
            hidden_dim=self.hidden_dim, action_chunk_size=self.action_chunk_size,
            num_layers=self.num_layers, num_heads=self.num_heads,
            dropout_rate=self.dropout_rate, num_output=num_output,
        )

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions=None, return_logits=False, is_encoded=False, deterministic=True, images=None):
        """Forward pass through the transformer critic.

        Args:
            observations: State observations
            actions: Action sequence (will be reshaped to [B, T, action_dim])
            return_logits: Whether to return logits for HL-Gauss critic
            is_encoded: Whether observations are already encoded
            deterministic: Whether to use deterministic mode (no dropout)
            images: Optional image observations for VisionProprioEncoder

        Returns:
            Q-values, optionally with logits for distributional critic
        """
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        # Reshape flat actions (B, T*a_dim) -> (B, T, a_dim)
        actions = actions.reshape(-1, self.action_chunk_size, self.action_dim)

        output = self.value_net(observations, actions, deterministic=deterministic)

        if self.critic_loss_type == 'hlgauss':
            q_logits = output
            v = jnp.sum(jax.nn.softmax(q_logits, axis=-1) * jnp.linspace(self.q_min, self.q_max, self.num_bins), axis=-1)
            if return_logits:
                return v, q_logits
        else:
            v = output.squeeze(-1)
        return v


class ValueSimBa(nn.Module):
    """Critic using SimBa (Simple Baseline) architecture.

    SimBa uses running statistics normalization for improved stability.

    Attributes:
        hidden_dims: Hidden layer dimensions
        num_ensembles: Number of ensemble members
        encoder: Optional encoder for observations
        critic_loss_type: 'mse' or 'hlgauss'
        num_bins: Number of bins for HL-Gauss
        q_min: Minimum Q-value for HL-Gauss
        q_max: Maximum Q-value for HL-Gauss
        rs_norm_momentum: Momentum for running statistics normalization
        rs_norm_epsilon: Epsilon for running statistics normalization
    """
    hidden_dims: Sequence[int]
    num_ensembles: int = 2
    encoder: Optional[nn.Module] = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: Optional[float] = None
    q_max: Optional[float] = None
    rs_norm_momentum: float = 0.99
    rs_norm_epsilon: float = 1e-8

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        # Create single or ensemble of SimBa MLPs
        mlp_class = SimBaMLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)

        # Create the value network(s)
        self.value_net = mlp_class(
            hidden_dims=(*self.hidden_dims, num_output),
            rs_norm_momentum=self.rs_norm_momentum,
            rs_norm_epsilon=self.rs_norm_epsilon
        )

    def encode(self, observations, images=None):
        """Encode observations using the encoder if available."""
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(self, observations, actions=None, return_logits=False,
                 is_encoded=False, images=None):
        """Forward pass through the SimBa critic.

        Args:
            observations: State observations
            actions: Actions (optional, will be concatenated with observations)
            return_logits: Whether to return logits for HL-Gauss critic
            is_encoded: Whether observations are already encoded
            images: Optional image observations for VisionProprioEncoder

        Returns:
            Q-values, optionally with logits for distributional critic
        """
        # Encode observations if needed
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        # Prepare inputs
        inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        x = jnp.concatenate(inputs, axis=-1)

        q_values = self.value_net(x)

        # Handle distributional critic if needed
        if self.critic_loss_type == 'hlgauss':
            if self.num_ensembles > 1:
                # Process each ensemble member
                v_list = []
                for q_logits in q_values:
                    v = jnp.sum(
                        jax.nn.softmax(q_logits, axis=-1) *
                        jnp.linspace(self.q_min, self.q_max, self.num_bins),
                        axis=-1
                    )
                    v_list.append(v)
                v = jnp.stack(v_list, axis=0)
            else:
                v = jnp.sum(
                    jax.nn.softmax(q_values, axis=-1) *
                    jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1
                )

            if return_logits:
                return v, q_values
        else:
            v = q_values.squeeze(-1)

        return v


class ValueMIP(nn.Module):
    """MIP-parameterized Q-function with noise-based implicit ensembles.

    Instead of explicit ensembles (num_qs separate networks), uses a single
    network conditioned on random scalar noise z ~ Uniform[-u, u] to create diversity.

    Architecture follows MIP actor pattern:
    - Step 1 (t=0): Initial Q estimate from [obs, action, noise, t]
    - Step 2 (t=t*): Refined Q estimate from [obs, action, q_0_features, t*]

    When num_subtasks > 1, scalar_input and output become S-dimensional vectors,
    and the output is reshaped to (S, B) to match the Value class convention.
    """
    hidden_dims: Sequence[int]
    mip_q_noise_scale: float = 1.0
    mip_t_star: float = 0.9
    layer_norm: bool = True
    encoder: Optional[nn.Module] = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: Optional[float] = None
    q_max: Optional[float] = None
    num_subtasks: int = 1

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1
        final_output_dim = num_output * self.num_subtasks if self.num_subtasks > 1 else num_output
        self.mlp = MLP(
            (*self.hidden_dims, final_output_dim),
            activate_final=False,
            layer_norm=self.layer_norm
        )

    @property
    def noise_dim(self):
        """Dimension of scalar_input: S for subtasks, 1 for scalar."""
        return self.num_subtasks if self.num_subtasks > 1 else 1

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(
        self,
        observations,
        actions=None,
        scalar_input=None,
        time=None,
        return_logits=False,
        is_encoded=False,
        images=None,
        rng=None,
    ):
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        obs_shape = observations.shape
        batch_size = obs_shape[0]

        # Sample noise if not provided: [B, noise_dim]
        if scalar_input is None:
            if rng is None:
                raise ValueError("Must provide either scalar_input or rng")
            scalar_input = jax.random.uniform(
                rng, (batch_size, self.noise_dim),
                minval=-self.mip_q_noise_scale,
                maxval=self.mip_q_noise_scale
            )

        if time is None:
            if len(obs_shape) == 3:
                chunk_size = obs_shape[1]
                time = jnp.zeros((batch_size, chunk_size, 1))
            else:
                time = jnp.zeros((batch_size, 1))

        # Broadcast scalar_input for action chunking
        if len(obs_shape) == 3:
            chunk_size = obs_shape[1]
            if scalar_input.ndim == 2:
                scalar_input = jnp.tile(scalar_input[:, None, :], (1, chunk_size, 1))

        mlp_input = jnp.concatenate([observations, actions, scalar_input, time], axis=-1)
        q_output = self.mlp(mlp_input)

        if self.critic_loss_type == 'hlgauss':
            if self.num_subtasks > 1:
                # [B, S*num_bins] -> [B, S, num_bins]
                q_logits = q_output.reshape(batch_size, self.num_subtasks, self.num_bins)
                q_values = jnp.sum(
                    jax.nn.softmax(q_logits, axis=-1)
                    * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1,
                )  # [B, S]
                q_values = q_values.T  # [S, B]
                q_logits = jnp.transpose(q_logits, (1, 0, 2))  # [S, B, num_bins]
            else:
                q_logits = q_output
                q_values = jnp.sum(
                    jax.nn.softmax(q_logits, axis=-1)
                    * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1,
                )
            if return_logits:
                return q_values, q_logits
        else:
            if self.num_subtasks > 1:
                # [B, S] -> [S, B]
                q_values = q_output.reshape(batch_size, self.num_subtasks).T
            else:
                q_values = q_output.squeeze(-1)  # [B]

        return q_values


class ValueMIPEnsemble(nn.Module):
    """Explicit ensemble of MIP-Q networks.

    Uses multiple networks + single shared noise (explicit ensemble).
    Each ensemble member is a separate MIP-Q network with independent parameters.

    When num_subtasks > 1, scalar_input is S-dimensional and output is (S, M, B).
    """
    hidden_dims: Sequence[int]
    num_ensembles: int = 2
    mip_q_noise_scale: float = 1.0
    mip_t_star: float = 0.9
    layer_norm: bool = True
    encoder: Optional[nn.Module] = None
    critic_loss_type: str = 'mse'
    num_bins: int = 256
    q_min: Optional[float] = None
    q_max: Optional[float] = None
    num_subtasks: int = 1

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1
        final_output_dim = num_output * self.num_subtasks if self.num_subtasks > 1 else num_output
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=0, out_axes=0)
        self.mlp = mlp_class(
            (*self.hidden_dims, final_output_dim),
            activate_final=False,
            layer_norm=self.layer_norm
        )

    @property
    def noise_dim(self):
        """Dimension of scalar_input: S for subtasks, 1 for scalar."""
        return self.num_subtasks if self.num_subtasks > 1 else 1

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(
        self,
        observations,
        actions=None,
        scalar_input=None,
        time=None,
        return_logits=False,
        is_encoded=False,
        images=None,
        rng=None,
    ):
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        obs_shape = observations.shape
        batch_size = obs_shape[0]

        if scalar_input is None:
            if rng is None:
                raise ValueError("Must provide either scalar_input or rng")
            scalar_input = jax.random.uniform(
                rng, (batch_size, self.noise_dim),
                minval=-self.mip_q_noise_scale,
                maxval=self.mip_q_noise_scale
            )

        if time is None:
            if len(obs_shape) == 3:
                chunk_size = obs_shape[1]
                time = jnp.zeros((batch_size, chunk_size, 1))
            else:
                time = jnp.zeros((batch_size, 1))

        # Broadcast scalar_input for action chunking
        if len(obs_shape) == 3:
            chunk_size = obs_shape[1]
            if scalar_input.ndim == 2:
                scalar_input = jnp.tile(scalar_input[:, None, :], (1, chunk_size, 1))

        # Handle scalar_input dimensions for ensemble routing
        # 2D [B, noise_dim]: step 1 (shared) -> tile to [M, B, noise_dim]
        # 3D [B, M, noise_dim]: step 2 (per-member) -> transpose to [M, B, noise_dim]
        if scalar_input.ndim == 3:
            if len(obs_shape) == 3:
                # Action chunking: [B, chunk, noise_dim] -> [M, B, chunk, noise_dim]
                scalar_input = jnp.tile(scalar_input[None, :, :, :], (self.num_ensembles, 1, 1, 1))
            elif scalar_input.shape[1] == self.num_ensembles:
                # Step 2: [B, M, noise_dim] -> [M, B, noise_dim]
                scalar_input = jnp.transpose(scalar_input, (1, 0, 2))
            else:
                raise ValueError(f"Unexpected scalar_input shape: {scalar_input.shape}")
        elif scalar_input.ndim == 2:
            # Step 1: [B, noise_dim] -> [M, B, noise_dim]
            scalar_input = jnp.tile(scalar_input[None, :, :], (self.num_ensembles, 1, 1))
        else:
            raise ValueError(f"scalar_input must be 2D or 3D, got shape: {scalar_input.shape}")

        # Broadcast other inputs to [M, B, ...]
        if observations.ndim == 2:
            observations = jnp.tile(observations[None, :, :], (self.num_ensembles, 1, 1))
            actions = jnp.tile(actions[None, :, :], (self.num_ensembles, 1, 1))
            time = jnp.tile(time[None, :, :], (self.num_ensembles, 1, 1))
        else:
            observations = jnp.tile(observations[None, :, :, :], (self.num_ensembles, 1, 1, 1))
            actions = jnp.tile(actions[None, :, :, :], (self.num_ensembles, 1, 1, 1))
            time = jnp.tile(time[None, :, :, :], (self.num_ensembles, 1, 1, 1))

        mlp_input = jnp.concatenate([observations, actions, scalar_input, time], axis=-1)
        q_output = self.mlp(mlp_input)  # [M, B, final_output_dim]

        if self.critic_loss_type == 'hlgauss':
            if self.num_subtasks > 1:
                # [M, B, S*num_bins] -> [M, B, S, num_bins]
                q_logits = q_output.reshape(self.num_ensembles, batch_size, self.num_subtasks, self.num_bins)
                q_values = jnp.sum(
                    jax.nn.softmax(q_logits, axis=-1)
                    * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1,
                )  # [M, B, S]
                q_values = jnp.transpose(q_values, (2, 0, 1))  # [S, M, B]
                q_logits = jnp.transpose(q_logits, (2, 0, 1, 3))  # [S, M, B, num_bins]
            else:
                q_logits = q_output  # [M, B, num_bins]
                q_values = jnp.sum(
                    jax.nn.softmax(q_logits, axis=-1)
                    * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                    axis=-1,
                )  # [M, B]
            if return_logits:
                return q_values, q_logits
        else:
            if self.num_subtasks > 1:
                # [M, B, S] -> [S, M, B]
                q_values = jnp.transpose(
                    q_output.reshape(self.num_ensembles, batch_size, self.num_subtasks),
                    (2, 0, 1)
                )
            else:
                q_values = q_output.squeeze(-1)  # [M, B]

        return q_values


# Export list
__all__ = [
    'default_init',
    'ensemblize',
    'Value',
    'ValueTF',
    'ValueSimBa',
    'ValueMIP',
    'ValueMIPEnsemble',
]
