from __future__ import annotations

from dataclasses import asdict
from functools import partial
import pickle
from typing import NamedTuple

from flax import (
    nnx,
    struct,
)
import jax
import jax.numpy as jnp


class AgentConfig(NamedTuple):
    """Configuration for the agent.

    Default values set to a neutral value such that only required parameters
    need to be set.
    """

    agents: jax.Array = jnp.array([])
    alpha: float = 0.0
    alpha_scaling: float = 0.0
    gamma: float = 0.0
    beta: float = 0.0
    tau: float = 0.0
    episodes: int = 0
    max_steps: int = 0
    min_episodes_criterions: int = 0
    reward_threshold: int = 0
    behavioral_policy: jax.Array = jnp.array([])
    epsilon: float = 0.0
    epsilon_decay: float = 0.0
    batchsize: int = 0


@struct.dataclass
class BehavioralData:
    """JAX-friendly PyTree data container.

    Fields that are scanned over axis 0:
      - states:           [T, ...]
      - actions:          [T, ...]
      - rewards:          [T]
      - next_states:      [T, ...]
      - terminals:        [T]
      - next_state_logits:[T, ...]

    Static metadata:
        - n_steps:         int
        - agents:          [N]
    """

    states: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_states: jax.Array
    terminals: jax.Array
    next_state_logits: jax.Array

    n_steps: int = struct.field(pytree_node=False)
    agents: jax.Array = struct.field(pytree_node=False)

    @classmethod
    def load(cls, path: str) -> BehavioralData:
        """Loads BehavioralData from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)

    def save(self, path: str) -> None:
        """Saves BehavioralData to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(asdict(self), f)

    def get_batch_data(self, batch_indices: jax.Array) -> BehavioralData:
        """Returns a batch of data based on the provided indices."""
        return BehavioralData(
            states=self.states[batch_indices],
            actions=self.actions[batch_indices],
            rewards=self.rewards[batch_indices],
            next_states=self.next_states[batch_indices],
            terminals=self.terminals[batch_indices],
            next_state_logits=self.next_state_logits[batch_indices],
            n_steps=batch_indices.shape[0],
            agents=self.agents,
        )


def get_train_batch_indices(
    samples: int,
    batch_size: int,
    max_index: int,
    rngs: nnx.Rngs,
    replace_for_rows: bool = False,
) -> jax.Array:
    """Generates random indices for training batches.

    Allows different methods to train on the same sampled data. The replace option
    allows to select if the same data point should be allowed to occur multiple
    times in the same batch.
    """
    indices = jnp.arange(max_index)
    if replace_for_rows:
        return jax.random.choice(
            rngs.sample(), indices, (samples, batch_size), replace=True
        )
    else:
        keys = jax.random.split(rngs.sample(), samples)
        return jax.lax.map(
            partial(jax.random.choice, a=indices, shape=(batch_size,), replace=False),
            keys,
        )


def convex_combination_uniform(
    x: jax.Array,
    weight: float,
) -> jax.Array:
    """Computes a convex combination of the input vector and a corresponding
    uniform distribution.
    """
    assert 0 <= weight <= 1, f"Weight must be between 0 and 1; received {weight}"
    uniform = jnp.ones(x.shape) / x.shape[-1]
    return (1 - weight) * x + weight * uniform


def masked_standardize(
    x: jax.Array, mask: jax.Array, epsilon: float = 1e-5
) -> jax.Array:
    """Standardizes the input array `x` using the provided mask.

    Since the `scan` provided by JAX does prefer condional free operations all
    episodes are padded to the same length. The mask is used to ignore all steps
    that occur after the end of an episode.
    """
    sum_elements = jnp.sum(mask, axis=0, keepdims=True)
    mean = jnp.sum(x * mask, axis=0, keepdims=True) / sum_elements
    variance = jnp.sum(((x - mean) * mask) ** 2 * mask) / sum_elements
    std = jnp.sqrt(variance) + epsilon
    return ((x - mean) / std) * mask


@partial(jax.jit, static_argnums=(2, 3))
def get_expected_return(
    rewards: jax.Array, mask: jax.Array, gamma: float, standardize: bool = True
) -> jax.Array:
    """Computes discounted returns for a sequence of rewards.

    NOTE: Potential numerical instability with large number of steps, could
    be fixed with `jax.lax.scan`.
    """
    discounts = (gamma ** jnp.arange(rewards.shape[0])).reshape(
        (rewards.shape[0],) + (1,) * (rewards.ndim - 1)
    )  # Match dimension of discount to rewards which can be (n,) or (n, m)
    discounted_rewards = rewards * mask * discounts
    returns = jnp.cumsum(discounted_rewards[::-1], axis=0)[::-1] / discounts
    if standardize:
        returns = masked_standardize(returns, mask)
    return returns


@partial(jax.jit, static_argnums=(1, 2))
def gae_from_td_residuals(
    td_residuals: jax.Array,
    gamma: float,
    lmbda: float,
) -> jax.Array:
    """Computes the Generalized Advantage Estimation (GAE) from the TD residuals.

    NOTE: Potential numerical instability with large number of steps, could
    be fixed with `jax.lax.scan`.
    """
    weighted_discounts = (gamma * lmbda) ** jnp.arange(td_residuals.shape[0])
    discounted_td_residuals = td_residuals * weighted_discounts
    gae = jnp.cumsum(discounted_td_residuals[::-1])[::-1] / weighted_discounts
    return gae
