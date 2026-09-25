from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    import jax


def unregularized_greedy(
    q_values: jax.Array,
    behavioral_policy: jax.Array,
    beta: float,
) -> tuple[jax.Array, jax.Array]:
    """Returns the greedy policy based on the Q-values as well as a regularization
    penaly of zero.

    Matches the function signature of the `regularization` and `gated_regularization`
    functions to be used in the the existing update functions.
    """
    policy = jnp.zeros(q_values.shape, dtype=jnp.float32)
    if len(q_values.shape) > 1:
        max_actions = jnp.argmax(q_values, axis=-1)
        indices = jnp.arange(q_values.shape[0])
        policy = policy.at[indices, max_actions].set(1.0)
        regularization_value = jnp.zeros(q_values.shape[0], dtype=jnp.float32)
    else:
        policy = policy.at[jnp.argmax(q_values)].set(1.0)
        regularization_value = jnp.zeros(1, dtype=jnp.float32)
    return policy, regularization_value
