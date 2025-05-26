from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from scs.utils import AgentConfig


def update_actor(
    state: jax.Array,
    action: jax.Array,
    policy_logits: jax.Array,
    policy: jax.Array,
    advantage: jax.Array,
    config: AgentConfig,
) -> jax.Array:
    """Update the actor policy logits via policy gradient using the advantage
    function.
    """
    n_actions = policy.shape[-1]
    action_one_hot = jax.nn.one_hot(action, num_classes=n_actions)
    policy_gradient = (action_one_hot - policy[state]) * advantage
    policy_logits = policy_logits.at[state].add(
        config.alpha_scaling * config.alpha * policy_gradient
    )
    return policy_logits


def update_critic(
    state: jax.Array,
    action: jax.Array,
    next_state: jax.Array,
    terminal: jax.Array,
    reward: jax.Array,
    policy: jax.Array,
    q_values: jax.Array,
    config: AgentConfig,
) -> tuple[jax.Array, jax.Array]:
    """Update the critic q-values using the TD error."""
    next_state_value = jnp.dot(policy[next_state], q_values[next_state])
    next_state_value = jax.lax.select(terminal, 0.0, next_state_value)
    td_target = reward + config.gamma * next_state_value
    td_error = td_target - q_values[state, action]
    q_values = q_values.at[state, action].add(config.alpha * td_error)
    return q_values, td_error


def update_step(
    carry: tuple[jax.Array, jax.Array],
    timestep: jax.Array,
    config: AgentConfig,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Scan wrapped update step.

    Updates the critic first, then the actor policy logits using the advantage
    function based on the updated q-values.
    """
    policy_logits, q_values = carry
    state, action, next_state, terminal = timestep[:4].astype(jnp.uint32)
    reward = timestep[4]
    policy = jax.nn.softmax(policy_logits)
    q_values, td_error = update_critic(
        state, action, next_state, terminal, reward, policy, q_values, config
    )
    state_value = jnp.dot(policy[state], q_values[state])
    advantage = q_values[state, action] - state_value
    policy_logits = update_actor(
        state, action, policy_logits, policy, advantage, config
    )
    return (policy_logits, q_values), td_error


@jax.jit
def train(
    policy_logits: jax.Array,
    q_values: jax.Array,
    timesteps: jax.Array,
    config: AgentConfig,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Trains the policy and q-values on 'timesteps' data."""
    (policy_logits, q_values), td_errors = jax.lax.scan(
        partial(update_step, config=config),
        (policy_logits, q_values),
        timesteps,
    )
    return (policy_logits, q_values), td_errors


def update_actor_parallel(
    states: jax.Array,
    policy_logits: jax.Array,
    policies: jax.Array,
    advantage: jax.Array,
    config: AgentConfig,
) -> jax.Array:
    """Update the actor policy logits via policy gradient using the advantage
    function in parallel for `config.agents` agents.
    """
    policy_gradient = policies[config.agents, states] * advantage[
        config.agents, states
    ] - jnp.sum(policies[config.agents, states] * advantage[config.agents, states])
    policy_logits = policy_logits.at[config.agents, states].add(
        config.alpha * policy_gradient
    )
    return policy_logits


def update_critic_parallel(
    states: jax.Array,
    actions: jax.Array,
    next_states: jax.Array,
    terminals: jax.Array,
    rewards: jax.Array,
    policies: jax.Array,
    q_values: jax.Array,
    config: AgentConfig,
) -> tuple[jax.Array, jax.Array]:
    """Update the critic q-values using the TD error in parallel for `config.agents`
    agents.
    """
    next_state_values = jnp.einsum(
        "ij,ij->i",
        policies[config.agents, next_states],
        q_values[config.agents, next_states],
    )
    next_state_values = jax.lax.select(terminals, 0.0, next_state_values)
    td_targets = rewards + config.gamma * next_state_values
    td_errors = td_targets - q_values[config.agents, states, actions]
    q_values = q_values.at[config.agents, states, actions].add(config.alpha * td_errors)
    return q_values, td_errors


def update_step_parallel(
    carry: tuple[jax.Array, jax.Array],
    timestep: jax.Array,
    config: AgentConfig,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Scan wrapped update step for parallel updating of `config.agents` agents.

    Updates the critics first, then the actors policy logits using the advantage
    functions based on the updated q-values.
    """
    policy_logits, q_values = carry
    states, actions, next_states, terminals = timestep[:4].astype(jnp.uint32)
    rewards = timestep[4]
    policies = jax.nn.softmax(policy_logits)
    q_values, td_errors = update_critic_parallel(
        states, actions, next_states, terminals, rewards, policies, q_values, config
    )
    state_values = jnp.einsum(
        "ij,ij->i", policies[config.agents, states], q_values[config.agents, states]
    )
    advantages = q_values[config.agents, states, actions] - state_values
    policy_logits = update_actor_parallel(
        states, policy_logits, policies, advantages, config
    )
    return (policy_logits, q_values), td_errors


@jax.jit
def train_parallel(
    policy_logits: jax.Array,
    q_values: jax.Array,
    timesteps: jax.Array,
    config: AgentConfig,
) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    """Train the policies and q-values of `config.agents` agents on 'timesteps'
    data, which contains one trajectory of timesteps for each agent.
    """
    (policy_logits, q_values), td_errors = jax.lax.scan(
        partial(update_step_parallel, config=config),
        (policy_logits, q_values),
        timesteps,
    )
    return (policy_logits, q_values), td_errors
