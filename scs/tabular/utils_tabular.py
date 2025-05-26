from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Callable,
)

from jax import random
import jax.numpy as jnp

from scs.tabular import env_gym
from scs.utils import convex_combination_uniform

if TYPE_CHECKING:
    import jax
    from utils import AgentConfig

    from scs.tabular.env_gym import TabularGymParameters
    from scs.tabular.env_mdp import MDPparameters


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


def initialize_logits(
    key: jax.Array, env: MDPparameters | TabularGymParameters, agents: int = 1
) -> tuple[jax.Array, jax.Array]:
    """Initializes policy logits with uniform random values.

    Args:
        key: JAX PRNG key for random initialization.
        env: Environment parameter object containing state and action space information.
        agents: Number of agents/policies to initialize.

    Returns:
        - The initialized logits array with shape (agents, n_states, n_actions)
          if agents > 1, otherwise (n_states, n_actions)
        - The updated PRNG key
    """
    init_key, key = random.split(key)
    if agents == 1:
        logits = random.uniform(init_key, (env.states.shape[0], int(env.n_actions)))
    else:
        logits = random.uniform(
            init_key, (agents, env.states.shape[0], int(env.n_actions))
        )
    return logits, key


def process_n_trajectories(
    trajectories: jax.Array,
    not_terminated_mask: jax.Array,
    n_trajectories: int,
    n_samples: int,
) -> jax.Array:
    """
    Processes a specified number of trajectories and samples from the given data
    by slicing the data that is not terminal as well as cutting out the desired
    number of samples. The data is then reshaped from how
    'env_gym.generate_trajectory_parallel' generates it (n_episodes, n_steps,
    features, n_trajectories) to (n_samples, features, n_trajectories).
    """
    # Ensure enough timesteps are available to slice out the n_samples
    assert jnp.any(jnp.sum(not_terminated_mask, axis=(0, 1)) >= n_samples)
    behavioral_data = jnp.array(
        [
            trajectories[..., i][not_terminated_mask[..., i]][:n_samples]
            for i in range(n_trajectories)
        ]
    )
    print(
        f"Environment Steps generated: {jnp.sum(not_terminated_mask, axis=(0, 1))}\n"
        f"Samples Shape: {behavioral_data.shape[:-1]}"
    )
    return jnp.moveaxis(behavioral_data, source=(0, 1, 2), destination=(2, 0, 1))


def get_epmirical_policy(
    timesteps: jax.Array, agents: jax.Array | None = None, states: int = 500, actions=6
) -> jax.Array:
    """Computes an empirical policy from observed state-action pairs.

    Creates a probability distribution over actions for each state by counting
    the frequency of action selections in the provided timesteps data.

    Args:
        timesteps: Array where each row contains
            [(optional: agent index), step, state_index, action_index, ...].
        agents: Optional array of agent indices corresponding to each timestep.
            If provided, computes separate policies for each agent.
        states: Number of states in the environment.
            Defaults to 500 (for taxi environment).
        actions: Number of actions in the environment.
            Defaults to 6 (for taxi environment).

    Returns:
        A policy array of shape [states, actions] if agents is None,
        otherwise [n_agents, states, actions]. Each row sums to 1.0 and represents
        a probability distribution over actions for that state. States with
        no visits are assigned a uniform distribution.
    """
    if agents is None:
        visitation_count = jnp.zeros((states, actions))
        visitation_count = visitation_count.at[timesteps[:, 0], timesteps[:, 1]].add(1)
    else:
        visitation_count = jnp.zeros((agents.shape[0], states, actions))
        visitation_count = visitation_count.at[
            agents, timesteps[:, 0], timesteps[:, 1]
        ].add(1)
    policy = visitation_count / jnp.sum(visitation_count, axis=-1, keepdims=True)
    # Replace NaNs from division of zero visit states with uniform policy
    return jnp.nan_to_num(policy, nan=1 / actions)


def policy_modification(
    key: jax.Array,
    policy: jax.Array,
    modification_type: str = "uniform",
    indices: jax.Array | None = None,
    modify_fraction: float = 1.0,
    convex_weight: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Applies a specified modification to a subset of a tabular policy array.

    A subset of state indices is either provided or sampled according to
    `modify_fraction`. The selected policy vectors are then perturbed by
    mixing with either a uniform distribution or Dirichlet noise.

    Uniform Modification:
        A uniform distribution over actions is generated and convexly
        combined with the original policy using `convex_weight`. This
        results in a smoothed policy that interpolates between the original
        policy and complete randomness.

    Dirichlet Modification:
        Symmetric Dirichlet noise is sampled for each selected policy vector.
        A scaling parameter is randomly drawn to control the noise
        concentration: small values yield spiky, near one-hot distributions
        (high exploration), while large values yield distributions close to
        uniform (low exploration). The sampled noise is then convexly combined
        with the original policy using `convex_weight`.

    Args:
        key: A JAX PRNGKey used for random operations.
        policy: A 2D array of shape (n_states, n_actions) representing the
            tabular policy to modify.
        modification_type: Specifies the modification mode. Must be one of
            "uniform" or "dirichlet".
        indices: Optional 1D array of state indices to modify. If None, indices
            are sampled uniformly from all states according to
            `modify_fraction`.
        modify_fraction: Fraction of states to modify when `indices` is None.
            A value in (0, 1] that defaults to 1.0, indicating all states.
        convex_weight: Mixing weight in [0, 1] for convex combination between
            the original policy and the uniform or Dirichlet noise.

    Returns:
        tuple[jax.Array, jax.Array]: A tuple containing the updated PRNGKey and
            the policy array with modifications applied at `indices`.

    Raises:
        ValueError: If `modification_type` is not one of "uniform" or
            "dirichlet".
    """
    if indices is None:
        n_states = policy.shape[0]
        if modify_fraction == 1.0:
            indices = jnp.arange(n_states)
        else:
            modify_states = int(n_states * modify_fraction)
            key, index_key = random.split(key)
            indices = random.choice(
                index_key, n_states, shape=(modify_states,), replace=False
            )
    modify_policies = policy[indices]
    if modification_type == "uniform":
        return key, policy.at[indices, ...].set(
            convex_combination_uniform(modify_policies, convex_weight)
        )
    elif modification_type == "dirichlet":
        key, scaling_key, noise_key = random.split(key, 3)
        alphas = jnp.ones(modify_policies.shape)
        scaling = random.choice(
            scaling_key,
            10 ** (jnp.arange(7, dtype=jnp.float32) - 2),
            shape=(alphas.shape[0], 1),
        )
        noise = random.dirichlet(noise_key, alphas * scaling)
        return key, policy.at[indices, ...].set(
            (1 - convex_weight) * modify_policies + convex_weight * noise
        )
    else:
        raise ValueError(
            f"Modification type {modification_type} not recognized; "
            f"Use 'uniform' or 'dirichlet'."
        )


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


def print_evaluation_results(
    evaluation_results: dict[str, dict[str, dict[str, jax.Array]]],
    beta: float = 0.0,
    convex_weight: float = 0.0,
    train_samples: int = 0,
    output_path: str | None = None,
) -> None:
    lines_to_print = []
    lines_to_print.append("#" * 84)
    lines_to_print.append("#" * 84)
    lines_to_print.append("#" * 84)
    lines_to_print.append(
        f"Beta: {beta} - Convex Weight: {convex_weight} "
        f"- Train Samples: {train_samples}"
    )
    lines_to_print.append(f"Percentiles: {[5, 10, 20, 25, 50, 75, 80, 90, 95]}")
    lines_to_print.append("#" * 84)
    lines_to_print.append("#" * 84)
    for method_name in evaluation_results.keys():
        lines_to_print.append(f"Method: {method_name}")
        for policy_type in evaluation_results[method_name].keys():
            results = evaluation_results[method_name][policy_type]
            lines_to_print.append(
                f"Mean Reward ({policy_type}): "
                f"{results['reward_mean']} ± {results['reward_std']}"
            )
            lines_to_print.append(
                f"Percentiles ({policy_type}): {results['reward_percentile']}"
            )
        lines_to_print.append("#" * 84)

    if output_path:
        with open(output_path, "a+") as f:
            for line in lines_to_print:
                f.write(line + "\n")
    else:
        for line in lines_to_print:
            print(line)
