from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
)

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from scs.gated import gated_regularization
from scs.qlearning import unregularized_greedy
from scs.regularized import regularization

if TYPE_CHECKING:
    from scs.utils import (
        AgentConfig,
        BehavioralData,
    )
    from scs.utils_nn import NNTrainingState


class QNNc(nnx.Module):
    def __init__(self, rngs: nnx.Rngs) -> None:
        self.linear_1: nnx.Linear = nnx.Linear(
            in_features=4,
            out_features=256,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layernorm_1: nnx.LayerNorm = nnx.LayerNorm(
            num_features=256,
            rngs=rngs,
        )
        self.linear_2: nnx.Linear = nnx.Linear(
            in_features=256,
            out_features=256,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )
        self.layernorm_2: nnx.LayerNorm = nnx.LayerNorm(
            num_features=256,
            rngs=rngs,
        )
        self.q_values: nnx.Linear = nnx.Linear(
            in_features=256,
            out_features=2,
            kernel_init=nnx.initializers.glorot_uniform(),
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.relu(self.layernorm_1(self.linear_1(x)))
        x = jax.nn.relu(self.layernorm_2(self.linear_2(x)))
        return self.q_values(x)


def compute_loss(
    model: QNNc,
    target_model: QNNc,
    states: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_states: jax.Array,
    terminals: jax.Array,
    next_state_behavioral_logits: jax.Array,
    config: AgentConfig,
    policy_and_regularization: Callable,
) -> jax.Array:
    """Calculates the Huber loss between predicted Q-values and target Q-values
    derived from the Bellman equation with a maximizing policy and the corresponding
    regularization penalty.
    """
    values = model(states)
    next_values = target_model(next_states)
    max_policies, regularization = policy_and_regularization(
        next_values, jax.nn.softmax(next_state_behavioral_logits), config.beta
    )
    next_state_values = (
        config.gamma * jnp.einsum("ij,ij->i", max_policies, next_values)
        - regularization
    ) * (1 - terminals)
    td_targets = rewards + config.gamma * next_state_values
    targets = values.at[jnp.arange(states.shape[0]), actions].set(td_targets)
    return jnp.sum(optax.huber_loss(values, jax.lax.stop_gradient(targets)))


def train_step(
    train_state: NNTrainingState,
    data: BehavioralData,
    config: AgentConfig,
    policy_and_regularization: str,
) -> tuple[NNTrainingState, jax.Array]:
    """Perform a single training step on a batch of data that can be used in
    JAX scan.

    Args:
        train_state: Current training state containing model and optimizer.
        data: Batch of behavioral data for training.
        config: Agent configuration parameters.
        policy_and_regularization: Policy type to use for maximization
            ('regularized', 'gated', or 'greedy').

    Returns:
        A tuple containing the updated training state and the training loss.
    """
    if policy_and_regularization == "regularized":
        get_policy_and_regularization = regularization
    elif policy_and_regularization == "gated":
        get_policy_and_regularization = gated_regularization
    elif policy_and_regularization == "greedy":
        get_policy_and_regularization = unregularized_greedy
    else:
        raise ValueError(
            f"Unknown maximizing policy: {policy_and_regularization}; "
            "Expected either 'regularized', 'gated', or 'greedy'."
        )
    grad_fn = nnx.value_and_grad(
        partial(compute_loss, policy_and_regularization=get_policy_and_regularization)
    )
    loss, grads = grad_fn(
        nnx.merge(train_state.model_def, train_state.model_state),
        nnx.merge(train_state.model_def, train_state.target_model_state),
        data.states,
        data.actions,
        data.rewards,
        data.next_states,
        data.terminals,
        data.next_state_logits,
        config,
    )
    return train_state.apply_gradients(grads), loss


@nnx.jit(static_argnums=(3,))
def train_on_data(
    train_state: NNTrainingState,
    data: BehavioralData,
    config: AgentConfig,
    policy_and_regularization: str,
) -> tuple[NNTrainingState, jax.Array]:
    """Train the Q-network on 'data' of behavioral data."""
    train_state, losses = jax.lax.scan(
        partial(
            train_step,
            config=config,
            policy_and_regularization=policy_and_regularization,
        ),
        train_state,
        data,
    )
    return train_state, losses


def train_on_data_greedy(
    train_state: NNTrainingState,
    data: BehavioralData,
    config: AgentConfig,
) -> tuple[NNTrainingState, jax.Array]:
    return train_on_data(
        train_state,
        data,
        config,
        policy_and_regularization="greedy",
    )


def train_on_data_regularized(
    train_state: NNTrainingState,
    data: BehavioralData,
    config: AgentConfig,
) -> tuple[NNTrainingState, jax.Array]:
    return train_on_data(
        train_state,
        data,
        config,
        policy_and_regularization="regularized",
    )


def train_on_data_gated(
    train_state: NNTrainingState,
    data: BehavioralData,
    config: AgentConfig,
) -> tuple[NNTrainingState, jax.Array]:
    return train_on_data(
        train_state,
        data,
        config,
        policy_and_regularization="gated",
    )
