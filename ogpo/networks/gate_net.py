"""GateNet: per-subtask done predictor for Q-IPM.

GateNet predicts subtask completion probabilities using sigmoid activation.
Used to gate Q-values by (1 - done_prob) so completed subtasks don't contribute.
"""

from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from ogpo.networks.modules.mlp import MLP, default_init


class GateNet(nn.Module):
    """Gate weights per subtask: g_i(s,a) in (0,1).

    Output shape: (B, num_subtasks) where [:, i] corresponds to subtask i.

    Attributes:
        hidden_dims: Hidden layer dimensions for the trunk MLP.
        num_subtasks: Number of subtasks to predict.
        layer_norm: Whether to use layer normalization.
        encoder: Optional encoder module for observations.
        final_fc_init_scale: Initialization scale for the output head.
    """
    hidden_dims: Sequence[int]
    num_subtasks: int
    layer_norm: bool = True
    encoder: nn.Module = None
    final_fc_init_scale: float = 1e-2

    def setup(self):
        self.trunk = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.head = nn.Dense(self.num_subtasks, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, is_encoded: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if (not is_encoded) and (self.encoder is not None):
            observations = self.encoder(observations)

        if observations.ndim > 2:
            observations = observations.reshape(observations.shape[0], -1)

        if actions.ndim > 2:
            actions = actions.reshape(actions.shape[0], -1)

        x = jnp.concatenate([observations, actions], axis=-1)
        h = self.trunk(x)
        logits = self.head(h)
        gate = jax.nn.sigmoid(logits)
        return gate, logits


class GateNetIdentity(nn.Module):
    """Identity gate network that always returns gate values of 1.0.

    Used for ablation studies to disable gate learning.
    Output shape: (B, num_subtasks) where all values are 1.0.
    """
    hidden_dims: Sequence[int]  # Kept for interface compatibility
    num_subtasks: int
    layer_norm: bool = True
    encoder: nn.Module = None
    final_fc_init_scale: float = 1e-2

    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, is_encoded: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_size = observations.shape[0]
        gate = jnp.ones((batch_size, self.num_subtasks), dtype=jnp.float32)
        logits = jnp.zeros((batch_size, self.num_subtasks), dtype=jnp.float32)
        return gate, logits
