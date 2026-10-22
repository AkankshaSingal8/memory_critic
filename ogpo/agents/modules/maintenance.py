"""LR scheduling, optimizer creation, checkpoint restore, param logging."""

import pickle
from typing import Any, Dict, Union

import jax
import jax.numpy as jnp
import optax


def create_lr_schedule(
    scheduler_type: str,
    base_lr: float,
    warmup_steps: int = 5000,
    decay_steps: int = 10000,
    end_value: float = 0.0
) -> Union[float, callable]:
    """Create learning rate schedule based on scheduler type."""
    if scheduler_type == 'constant':
        return base_lr
    elif scheduler_type == 'cosine':
        base_schedule = optax.schedules.warmup_cosine_decay_schedule(
            init_value=end_value,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_value,
        )
        period = decay_steps
        def repeating_schedule(count):
            return base_schedule(count % period)
        return repeating_schedule
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Use 'constant' or 'cosine'.")


def compute_param_norm(params: Any) -> jnp.ndarray:
    """Compute L2 norm of parameters."""
    flat_params, _ = jax.tree_util.tree_flatten(params)
    return jnp.sqrt(sum(jnp.sum(p**2) for p in flat_params))


def compute_param_diff_norm(params1: Any, params2: Any) -> jnp.ndarray:
    """Compute L2 norm of parameter difference."""
    diff_tree = jax.tree_util.tree_map(lambda p1, p2: p1 - p2, params1, params2)
    flat_diff, _ = jax.tree_util.tree_flatten(diff_tree)
    return jnp.sqrt(sum(jnp.sum(p**2) for p in flat_diff))


def get_lr_info(
    actor_state: Any,
    critic_state: Any,
    actor_key: str = 'modules_actor',
    target_actor_key: str = 'modules_target_actor',
    critic_key: str = 'modules_critic',
    target_critic_key: str = 'modules_target_critic',
) -> Dict[str, float]:
    """Get learning rate information and parameter norms for logging."""
    actor_norm = compute_param_norm(actor_state.params[actor_key])
    target_actor_norm = compute_param_norm(actor_state.params[target_actor_key])
    critic_norm = compute_param_norm(critic_state.params[critic_key])
    target_critic_norm = compute_param_norm(critic_state.params[target_critic_key])

    actor_diff_norm = compute_param_diff_norm(
        actor_state.params[actor_key],
        actor_state.params[target_actor_key]
    )
    critic_diff_norm = compute_param_diff_norm(
        critic_state.params[critic_key],
        critic_state.params[target_critic_key]
    )

    return {
        'actor_norm': float(actor_norm),
        'target_actor_norm': float(target_actor_norm),
        'critic_norm': float(critic_norm),
        'target_critic_norm': float(target_critic_norm),
        'actor_diff_norm': float(actor_diff_norm),
        'critic_diff_norm': float(critic_diff_norm),
    }


def _assert_matching_param_shapes(current: Any, loaded: Any, label: str, checkpoint_path: str) -> None:
    """Fail fast when a checkpoint's param tree does not match the live network.

    A blind copy of mismatched params "succeeds" at load time and only crashes
    at the first forward pass with a cryptic shape error. The likeliest cause
    of a shape mismatch here is a critic head-count change
    (no_of_subtasks / no_of_safety_subtasks); checkpoint surgery is not
    supported — use a matching checkpoint or train from scratch.
    """
    cur_struct = jax.tree_util.tree_structure(current)
    load_struct = jax.tree_util.tree_structure(loaded)
    if cur_struct != load_struct:
        raise ValueError(
            f"{label}: checkpoint {checkpoint_path} has a different param tree "
            f"structure than the live network.\ncheckpoint: {load_struct}\n"
            f"network:    {cur_struct}"
        )
    cur_leaves = jax.tree_util.tree_leaves_with_path(current)
    load_leaves = jax.tree_util.tree_leaves(loaded)
    for (key_path, cur_leaf), load_leaf in zip(cur_leaves, load_leaves):
        cur_shape = jnp.shape(cur_leaf)
        load_shape = jnp.shape(load_leaf)
        if cur_shape != load_shape:
            raise ValueError(
                f"{label}: param shape mismatch at "
                f"{jax.tree_util.keystr(key_path)}: checkpoint {load_shape} vs "
                f"network {cur_shape} (from {checkpoint_path}). Likely a "
                "head-count (no_of_subtasks / no_of_safety_subtasks) mismatch; "
                "checkpoint surgery is not supported — use a matching "
                "checkpoint or train from scratch."
            )


def restore_critic_params(
    critic_network: Any,
    critic_path: str,
    agent_key: str = "agent",
    network_key: str = "critic_network",
) -> None:
    """Restore critic parameters from checkpoint."""
    with open(critic_path, 'rb') as f:
        load_dict = pickle.load(f)
    src_params = load_dict[agent_key][network_key]["params"]
    _assert_matching_param_shapes(
        critic_network.params['modules_critic'], src_params["modules_critic"],
        "restore_critic_params[modules_critic]", critic_path)
    _assert_matching_param_shapes(
        critic_network.params['modules_target_critic'], src_params["modules_target_critic"],
        "restore_critic_params[modules_target_critic]", critic_path)
    critic_network.params['modules_critic'] = src_params["modules_critic"]
    critic_network.params['modules_target_critic'] = src_params["modules_target_critic"]
    print(f"Restored critic from {critic_path}")


def restore_safety_critic_params(critic_network: Any, path: str) -> None:
    """Load a pretrained safety critic into the 'safety' subtree of a
    CompositeValue critic: online params into modules_critic, the pretrained
    target into modules_target_critic. Task heads are left as they are.

    `path` is a scripts/safety_pretrain/safety_value.save_state pickle:
    {"state": state_dict of SafetyCriticState, "meta": {...}}.
    """
    with open(path, 'rb') as f:
        blob = pickle.load(f)
    state = blob["state"]
    for module, key in (('modules_critic', 'params'), ('modules_target_critic', 'target_params')):
        dst = critic_network.params[module]
        assert 'safety' in dst, (
            f"{module} has no 'safety' subtree: set agent.composite_safety_critic=true")
        _assert_matching_param_shapes(dst['safety'], state[key],
                                      f"restore_safety_critic_params[{module}]", path)
        # A new dict, never an in-place write: create() aliases the target
        # critic's params to the online critic's dict.
        critic_network.params[module] = {**dst, 'safety': state[key]}
    print(f"Restored safety critic from {path} (meta: {blob.get('meta')})")


def restore_actor_params(
    actor_network: Any,
    actor_path: str,
    has_encoder: bool = False,
    agent_key: str = "agent",
) -> None:
    """Restore actor parameters from checkpoint.

    Handles multiple checkpoint formats:
    - actor_network format (from OGPO/BPTT training)
    - network format with modules_actor_bc_flow (from BC pretraining)
    """
    with open(actor_path, 'rb') as f:
        load_dict = pickle.load(f)

    agent_dict = load_dict[agent_key]

    if "actor_network" in agent_dict:
        src_params = agent_dict["actor_network"]["params"]
        _assert_matching_param_shapes(
            actor_network.params['modules_actor'], src_params["modules_actor"],
            "restore_actor_params[modules_actor]", actor_path)
        actor_network.params['modules_actor'] = src_params["modules_actor"]
        actor_network.params['modules_target_actor'] = src_params["modules_target_actor"]
        print(f"Restored actor from {actor_path}")

    elif "network" in agent_dict and "modules_actor_bc_flow" in agent_dict["network"]["params"]:
        src_params = agent_dict["network"]["params"]

        if has_encoder:
            _assert_matching_param_shapes(
                actor_network.params['modules_actor']['encoder'],
                src_params["modules_actor_bc_flow_encoder"],
                "restore_actor_params[encoder]", actor_path)
            actor_network.params['modules_actor']['encoder'] = src_params["modules_actor_bc_flow_encoder"]
            actor_network.params['modules_target_actor']['encoder'] = src_params["modules_actor_bc_flow_encoder"]

            _assert_matching_param_shapes(
                actor_network.params['modules_actor']['mlp'],
                src_params["modules_actor_bc_flow"]["mlp"],
                "restore_actor_params[mlp]", actor_path)
            actor_network.params['modules_actor']['mlp'] = src_params["modules_actor_bc_flow"]["mlp"]
            actor_network.params['modules_target_actor']['mlp'] = src_params["modules_actor_bc_flow"]["mlp"]
        else:
            _assert_matching_param_shapes(
                actor_network.params['modules_actor'],
                src_params["modules_actor_bc_flow"],
                "restore_actor_params[modules_actor]", actor_path)
            actor_network.params['modules_actor'] = src_params["modules_actor_bc_flow"]
            actor_network.params['modules_target_actor'] = src_params.get(
                "modules_target_actor_bc_flow",
                src_params["modules_actor_bc_flow"]
            )
        print(f"Restored actor from {actor_path}")
    else:
        raise ValueError(f"Unknown checkpoint format in {actor_path}")


def create_optimizer(
    lr_schedule: Union[float, callable],
    use_muon: bool = False,
    clip_grad_norm: float = 1.0,
    weight_decay: float = 0.0,
    muon_beta: float = 0.95,
    muon_ns_steps: int = 5,
    muon_nesterov: bool = True,
) -> optax.GradientTransformation:
    """Create optimizer with optional Muon or AdamW."""
    if use_muon:
        return optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.contrib.muon(
                learning_rate=lr_schedule,
                beta=muon_beta,
                ns_steps=muon_ns_steps,
                weight_decay=weight_decay,
                nesterov=muon_nesterov,
            )
        )
    else:
        @optax.inject_hyperparams
        def adam_optimizer(learning_rate, weight_decay):
            return optax.adamw(learning_rate, weight_decay=weight_decay)
        return optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            adam_optimizer(
                learning_rate=lr_schedule,
                weight_decay=weight_decay,
            )
        )


def rebind_online_optimizers(agent: Any) -> Any:
    """Point a resumed agent's optimizers at the ONLINE schedules, keeping state.

    `TrainState.tx` is a non-pytree field, so a checkpoint restores `opt_state`
    and `step` but not the optimizer: `from_state_dict` keeps the `tx` of the
    freshly created agent, which is built on the BC learning rates
    (actor_lr/critic_lr). An online resume therefore used to keep training on
    the BC schedule instead of the one `reset_optimizers_with_lr` installed at
    the start of the online phase. Calling `reset_optimizers_with_lr` again
    would fix `tx` but also zero the Adam moments and restart the schedule, so
    this takes only `tx` from it and keeps the restored `opt_state` and `step`.

    Generic over agents: every TrainState field whose `tx` the agent's own
    reset replaces gets rebound, so it covers whichever networks that agent's
    reset covers.
    """
    from ogpo.agents.modules.flax_utils import TrainState

    fresh = agent.reset_optimizers_with_lr()
    updates = {}
    for name in agent.__dataclass_fields__:
        old, new = getattr(agent, name), getattr(fresh, name)
        if not isinstance(old, TrainState) or new.tx is old.tx:
            continue
        old_def = jax.tree_util.tree_structure(old.opt_state)
        new_def = jax.tree_util.tree_structure(new.opt_state)
        if old_def != new_def:
            raise ValueError(
                f"rebind_online_optimizers: restored {name}.opt_state does not "
                f"match the online optimizer's state structure; the checkpoint "
                f"cannot continue on the online schedule.\n"
                f"restored: {old_def}\nonline:   {new_def}"
            )
        updates[name] = old.replace(tx=new.tx)
    print(f"Rebound online optimizers on resume (opt_state and step kept): "
          f"{sorted(updates)}")
    return agent.replace(**updates)
