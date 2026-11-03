"""MemoryValue: the qipm critic (ogpo.networks.critics.Value) with a 64-d memory vector appended to obs.

memory = "none": Value(obs, chunk), the no-memory baseline (scripts/train_mc_critic.py).
memory = "clk":  m = Dense(t / 400).
memory = "gru":  m = Dense(GRU over the memory events), the GRU, SHUF and K1 arms (they differ only in
                 which events are fed: memcritic/history.py).
The candidate chunk enters only the Value MLP, so memory and action interact there.
"""
import flax.linen as nn
import jax.numpy as jnp

from ogpo.networks.critics import Value

TIME_LIMIT = 400.0


class MemoryValue(nn.Module):
    memory: str
    hidden_dims: tuple = (512, 512, 512, 512)
    num_qs: int = 10
    mem_dim: int = 64
    gru_dim: int = 256

    @nn.compact
    def __call__(self, obs, act, hist=None, hist_len=None, t=None):
        value = Value(hidden_dims=self.hidden_dims, layer_norm=True, num_ensembles=self.num_qs,
                      critic_loss_type="mse", num_subtasks=1, name="value")
        if self.memory == "none":
            return value(obs, act)
        if self.memory == "clk":
            f = t[:, None].astype(jnp.float32) / TIME_LIMIT
        elif self.memory == "gru":
            e = nn.relu(nn.LayerNorm()(nn.Dense(self.gru_dim)(hist)))
            f, _ = nn.RNN(nn.GRUCell(self.gru_dim), return_carry=True, name="gru")(e, seq_lengths=hist_len)
        else:
            raise ValueError(self.memory)
        m = nn.Dense(self.mem_dim, name="mem_proj")(f)
        return value(jnp.concatenate([obs, m], -1), act)
