from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

from jax import random
import jax.numpy as jnp

from scs.utils import convex_combination_uniform

if TYPE_CHECKING:
    import jax

    from scs.tabular.env_gym import TabularGymParameters
    from scs.tabular.env_mdp import MDPparameters


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
        (low exploration), while large values yield distributions close to
        uniform (high exploration). The sampled noise is then convexly combined
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
