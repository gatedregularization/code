from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy.special import rel_entr

if TYPE_CHECKING:
    from flax import nnx

    from scs.cartpole.agent_a2c_online import A2Cc
    from scs.lander.agent_a2c_online import A2Cll


def gate_original(p: jax.Array) -> jax.Array:
    """Gate function based on the relative entropy.

    Returns a value between
    - 0 if p is uniform
    - 1 if p is deterministic

    The absolute value is used to ensure that tiny value close to zero are not
    negative due to floating point precision errors.
    """
    ic = jnp.nan_to_num(p * jnp.log(p))
    shannon_entropy = -jnp.sum(ic, axis=-1, keepdims=True)
    # Capture numerical instabilities ensuring positive values
    return jnp.abs(1 - shannon_entropy / jnp.log(p.shape[-1]))


def scaling_gate_original(p: jax.Array) -> jax.Array:
    """Gate function based on the relative entropy.

    Returns a value between
    - 0 if p is uniform
    - -> infinity as p approaches deterministic

    The absolute value is used to ensure that tiny value close to zero are not
    negative due to floating point precision errors.

    The idea is to enforce to mimic the behavioral data if the policy deterministic
    since no other action is present in the distribution of the behavioral data.
    """
    ic = jnp.nan_to_num(p * jnp.log(p))
    shannon_entropy = -jnp.sum(ic, axis=-1, keepdims=True)
    # Capture numerical instabilities ensuring positive values
    return jnp.abs(jnp.log(p.shape[-1]) / shannon_entropy - 1)


def scaled_gate_original(p: jax.Array) -> jax.Array:
    """Gate function based on the relative entropy.

    Returns a value between
    - 0 if p is uniform
    - 4 if p is deterministic

    Similar to the `scaling_gate_original` function but with the maximum value
    set to 4 instead of infinity.
    """
    denominator = (1 - gate_original(p)) ** 0.5 - 2
    return 4 / denominator + 4


def gate_tvd(p: jax.Array) -> jax.Array:
    """Gate function based on the total variation distance.

    Returns a value between
    - 0 if p is uniform
    - 1 if p is deterministic
    """
    n_actions = p.shape[-1]
    tvd = 0.5 * jnp.sum(jnp.abs(p - 1 / n_actions), axis=-1, keepdims=True)
    return tvd * n_actions / (n_actions - 1)


def scaling_gate_tvd(p: jax.Array) -> jax.Array:
    """Gate function based on the total variation distance.

    Returns a value between
    - 0 if p is uniform
    - -> infinity as p approaches deterministic

    The idea is to enforce to mimic the behavioral data if the policy deterministic
    since no other action is present in the distribution of the behavioral data.
    """
    n_actions = p.shape[-1]
    tvd = 0.5 * jnp.sum(jnp.abs(p - 1 / n_actions), axis=-1, keepdims=True)
    return tvd / ((n_actions - 1) / n_actions - tvd)


def gate_tvd_concave(p: jax.Array) -> jax.Array:
    """Gate function based on the total variation distance.

    Returns a value between
    - 0 if p is uniform
    - 1 if p is deterministic

    Compared to the other gates this one is concave, instead of convex, to keep
    a higher level of regularization while the policy is close to deterministic.
    """
    denominator = (1 - gate_tvd(p)) ** 2 - 2
    return 2 / denominator + 2


# Set the default gate function to the relative entropy gate
gate = gate_original


def gated_regularization(
    q_values: jax.Array,
    behavioral_policy: jax.Array,
    beta: float,
) -> tuple[jax.Array, jax.Array]:
    """Calculates the maximizing policy under gated regularization as well as the
    resulting regularization penalty.

    Utilizes "Nestrov (2005) 'Smooth minimization of non-smooth functions', p.148"
    suggestion to improve numerical stability
    """
    max_q = jnp.max(q_values, axis=-1, keepdims=True)
    gate_value = gate(behavioral_policy)
    # Avoid zero in denominator by scaling a gate or beta of zero to a tiny value
    exp_scaled_q_values = jnp.exp((q_values - max_q) / (beta * gate_value + 1e-8))
    weighted_q_values = exp_scaled_q_values * behavioral_policy
    policy = weighted_q_values / jnp.sum(weighted_q_values, axis=-1, keepdims=True)
    regularization_value = jnp.sum(rel_entr(policy, behavioral_policy), axis=-1)
    return policy, beta * gate_value[..., 0] * regularization_value


def _sample_cartpole_action(
    key: jax.Array,
    policy: jax.Array,
) -> jax.Array:
    return jax.random.choice(key, jnp.arange(policy.shape[0]), p=policy)


def eval_action_gated(
    states: jax.Array,
    q_values: jax.Array,
    rngs: nnx.Rngs,
    b_model: A2Cc | A2Cll,
    beta: float,
) -> tuple[jax.Array, nnx.Rngs]:
    """Selects actions using the greedy gated policy derived from q-values and a
    behavioral policy.
    """
    b_logits, _values = b_model(states)
    b_policy = jax.nn.softmax(b_logits, axis=-1)
    policies, _regularization = gated_regularization(q_values, b_policy, beta)
    keys = jax.random.split(rngs.sample(), num=policies.shape[0])
    actions = jax.vmap(_sample_cartpole_action)(keys, policies)
    return actions, rngs
