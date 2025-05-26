from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy.special import rel_entr

if TYPE_CHECKING:
    from flax import nnx

    from scs.cartpole.agent_a2c_online import A2Cc
    from scs.lander.agent_a2c_online import A2Cll


def regularization(
    q_values: jax.Array,
    behavioral_policy: jax.Array,
    beta: float,
) -> tuple[jax.Array, jax.Array]:
    """Calculates the maximizing policy under regularization as well as the
    resulting regularization penalty.

    Utilizes "Nestrov (2005) 'Smooth minimization of non-smooth functions', p.148"
    suggestion to improve numerical stability
    """
    max_q = jnp.max(q_values, axis=-1, keepdims=True)
    # Avoid zero in denominator by scaling a beta of zero to a tiny value
    exp_scaled_q_values = jnp.exp((q_values - max_q) / (beta + 1e-8))
    weighted_q_values = exp_scaled_q_values * behavioral_policy
    policy = weighted_q_values / jnp.sum(weighted_q_values, axis=-1, keepdims=True)
    regularization_value = jnp.sum(rel_entr(policy, behavioral_policy), axis=-1)
    return policy, beta * regularization_value


def _sample_cartpole_action(
    key: jax.Array,
    policy: jax.Array,
) -> jax.Array:
    return jax.random.choice(key, jnp.arange(policy.shape[0]), p=policy)


def eval_action_regularized(
    states: jax.Array,
    q_values: jax.Array,
    rngs: nnx.Rngs,
    b_model: A2Cc | A2Cll,
    beta: float,
) -> tuple[jax.Array, nnx.Rngs]:
    """Selects actions using the greedy regularized policy derived from q-values
    and a behavioral policy.
    """
    b_logits, _values = b_model(states)
    b_policy = jax.nn.softmax(b_logits, axis=-1)
    policies, _regularization = regularization(q_values, b_policy, beta)
    keys = jax.random.split(rngs.sample(), num=policies.shape[0])
    actions = jax.vmap(_sample_cartpole_action)(keys, policies)
    return actions, rngs
