from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
)

from jax import random
import jax.numpy as jnp

from scs.tabular import env_gym

if TYPE_CHECKING:
    import jax
    from utils import AgentConfig


def random_action(state: jax.Array, n_actions: jax.Array, key: jax.Array) -> jax.Array:
    """Selects a random action uniformly from the available actions.

    The 'state' argument is only included to maintain the required function signature.
    """
    return random.randint(key, (), 0, n_actions)


def random_action_parallel(
    states: jax.Array, n_actions: jax.Array, key: jax.Array
) -> jax.Array:
    """Selects random actions for multiple states in parallel."""
    return random.randint(key, (states.shape[0],), 0, n_actions)


def deterministic_action(
    state: jax.Array, n_actions: jax.Array, key: jax.Array, policy: jax.Array
) -> jax.Array:
    """Selects an action deterministically according to a policy lookup table.

    The 'n_actions' and 'key' arguments are included to maintain the required
    function signature.
    """
    return policy[state]


def policies_deterministic_action_parallel(
    states: jax.Array, n_actions: jax.Array, key: jax.Array, policies: jax.Array
) -> jax.Array:
    """Selects deterministic actions for multiple states in parallel.

    The 'n_actions' and 'key' arguments are included to maintain the required
    function signature.
    """
    return policies[jnp.arange(policies.shape[0]), states]


def policy_action(
    state: jax.Array, n_actions: jax.Array, key: jax.Array, policy: jax.Array
) -> jax.Array:
    """Samples an action from a stochastic policy for a single state.

    The 'n_actions' argument is included to maintain the required function
    signature.
    """
    return random.choice(key, policy.shape[1], p=policy[state])


def policy_action_parallel(
    states: jax.Array,
    n_actions: jax.Array,
    key: jax.Array,
    policy: jax.Array,
) -> jax.Array:
    """Samples actions from a stochastic policy for multiple states in parallel.

    Uses the cumulative probabilities to efficiently sample multiple actions
    simultaneously.

    The 'n_actions' argument is included to maintain the required function
    signature.
    """
    action_probs = policy[states]
    random_actions = random.uniform(key, (states.shape[0], 1))
    cumulative_probs = jnp.cumsum(action_probs, axis=1)
    return jnp.argmax(cumulative_probs > random_actions, axis=1)


def policies_action_parallel(
    states: jax.Array,
    n_actions: jax.Array,
    key: jax.Array,
    policies: jax.Array,
) -> jax.Array:
    """Samples actions from multiple stochastic policies for multiple states, and
    corresponding policies, in parallel.

    Similar to `policy_action_parallel` but handles multiple policies such that
    each action is sampled according to its respective state and policy.

    The 'n_actions' argument is included to maintain the required function
    signature.
    """
    action_probs = policies[jnp.arange(policies.shape[0]), states]
    random_actions = random.uniform(key, (policies.shape[0], 1))
    cumulative_probs = jnp.cumsum(action_probs, axis=1)
    return jnp.argmax(cumulative_probs > random_actions, axis=1)


def train_value_agents(
    learning_methods: dict[str, Callable],
    agent_params: AgentConfig,
    data: jax.Array,
    q_shape: tuple[int, int],
) -> dict[str, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Trains multiple value-based agents as defined in `learning_methods`.

    Args:
        learning_methods: Dictionary mapping method names to their training
            functions. Each function should accept (q_values, data, agent_params)
            and return (q_values, policy, td_errors).
        agent_params: Configuration object containing agent parameter.s
        data: Array of training data containing state-action-reward trajectories.
        q_shape: Tuple of (n_states, n_actions) specifying the shape of a single
            Q-value tables.

    Returns:
        Dictionary mapping method names to tuples containing:
        - Q-values array
        - Learned policy array
        - Greedy policy derived from Q-values
        - TD errors during training
    """
    training_results = {}
    for method_name, learning_method in learning_methods.items():
        print(f"Training with '{method_name}'-method")
        if agent_params.agents.shape[0] > 1:
            q_values = jnp.zeros((agent_params.agents.shape[0], *q_shape))
        else:
            q_values = jnp.zeros(q_shape)
        q_values, policy, td = learning_method(q_values, data, agent_params)
        greedy_policy = jnp.argmax(q_values, axis=-1)
        training_results[method_name] = (q_values, policy, greedy_policy, td)
    return training_results


def evaluate_value_agents(
    method_parameter: dict[str, tuple[jax.Array, jax.Array, jax.Array, jax.Array]],
    env: env_gym.TabularGymParameters,
    key: jax.Array,
    n_episodes: int,
) -> tuple[dict[str, dict[str, dict[str, jax.Array]]], jax.Array]:
    """Evaluates trained value-based agents in an environment.

    For each training method's results, evaluates both the learned stochastic
    policy and the greedy deterministic policy derived from Q-values.

    Args:
        method_parameter: Dictionary mapping method names to tuples containing:
            [Q-values array, policy array with shape (n_agents, n_states, n_actions),
            policy array with shape (n_agents, n_states), TD errors during training]
        env: Environment parameters object;
        key: JAX PRNG key for random number generation.
        n_episodes: Number of episodes to run for each evaluation.

    Returns:
        Nested dictionary containing evaluation metrics for each method and
        policy type:
            method -> policy_type -> metrics
        where metrics include:
        - sum_rewards: Raw episode rewards
        - reward_means: Mean rewards per agent
        - reward_stds: Standard deviation of rewards per agent
        - reward_mean: Overall mean reward
        - reward_std: Overall standard deviation
        - reward_percentile: Overall reward distribution percentiles
    """
    evaluation_results: dict[str, dict[str, dict[str, jax.Array]]] = {
        method_name: {} for method_name in method_parameter.keys()
    }
    for method_name, training_results in method_parameter.items():
        _q_values, policy, greedy_policy, _td = training_results
        # Select respective action-selection function based on number of agents
        n_agents = policy.shape[0] if len(policy.shape) > 2 else 1
        if n_agents > 1:
            select_action_deterministic = partial(
                policies_deterministic_action_parallel, policies=greedy_policy
            )
            select_action_policy = partial(policies_action_parallel, policies=policy)
        else:
            select_action_deterministic = partial(
                deterministic_action, policy=greedy_policy
            )
            select_action_policy = partial(policy_action_parallel, policy=policy)
        for policy_type, eval_poliy in zip(
            ["policy", "greedy"], [select_action_policy, select_action_deterministic]
        ):
            print(f"Evaluating '{method_name}' with '{policy_type}'")
            data, key = env_gym.evaluate_policy_parallel(
                n_episodes=n_episodes,
                n_agents=n_agents,
                env=env,
                key=key,
                policy=eval_poliy,
            )
            sum_rewards = jnp.sum(data, axis=1)
            evaluation_results[method_name][policy_type] = {
                "sum_rewards": sum_rewards,
                "reward_means": jnp.mean(sum_rewards, axis=0),
                "reward_stds": jnp.std(sum_rewards, axis=0),
                "reward_mean": jnp.mean(sum_rewards),
                "reward_std": jnp.std(sum_rewards),
                "reward_percentile": jnp.percentile(
                    sum_rewards, q=jnp.array([5, 10, 20, 25, 50, 75, 80, 90, 95])
                ),
            }
            if (policy == 1.0).sum() == (policy.shape[0] * policy.shape[1]):
                # Policy is deterministic and no greedy evaluation is needed
                break
    return evaluation_results, key
