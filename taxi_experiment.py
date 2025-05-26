from __future__ import annotations

from functools import partial
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from jax import random
import jax.numpy as jnp

from scs.tabular import (
    agents_value_offline,
    env_gym,
)
from scs.tabular.utils_tabular import (
    evaluate_value_agents,
    get_epmirical_policy,
    policy_action_parallel,
    policy_modification,
    print_evaluation_results,
    process_n_trajectories,
    train_value_agents,
)
from scs.utils import AgentConfig

############################################################################
# Hyperparameters
############################################################################
mean_over: int = 10
max_steps: int = 100
behavioral_samples: int = 500000
beta: float = 2
convex_weight: float = 0.5
seed: int = 0
use_d_policy: bool = True  # Use the or the empirical or actual behavioral policy
############################################################################

# Setup Experiment
key = random.PRNGKey(seed)
env = env_gym.create_Taxi(max_steps)
n_states = int(env.states.shape[0])
n_actions = int(env.n_actions)

# Get optimal q-values
b_values, b_errors = env_gym.solve_taxi_greedy(env, 0.99, 1000)
b_policy = (
    jnp.zeros((n_states, n_actions))
    .at[jnp.arange(n_states), jnp.argmax(b_values, axis=-1)]
    .set(1.0)
)

# Make the behavioral policy suboptimal and inhomogeneous

# The OpenAI Taxi environment encodes states as integers between 0 and 499.
# Each state is calculated as:
#
#  ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
#
# Where:
# - `taxi_row`: The row where the taxi is located (0-4).
# - `taxi_col`: The column where the taxi is located (0-4).
# - `passenger_location`: The passenger's location (0-4). Values:
#     - 0: Red
#     - 1: Green
#     - 2: Yellow
#     - 3: Blue
#     - 4: In the taxi
# - `destination`: The destination location (0-3). Values:
#     - 0: Red
#     - 1: Green
#     - 2: Yellow
#     - 3: Blue

indices_left_half = jnp.array(
    [
        ((row * 5 + col) * 5 + p_loc) * 4 + dest
        for row in range(5)
        for col in range(3)
        for p_loc in range(5)
        for dest in range(4)
    ]
)
key, b_policy = policy_modification(
    key,
    b_policy,
    modification_type="dirichlet",
    indices=indices_left_half,
    convex_weight=convex_weight,
)

# Generate behavioral data
(
    behavioral_trajectories,
    not_terminated,
    key,
) = env_gym.generate_trajectory_parallel(
    n_episodes=behavioral_samples // 10,
    n_agents=mean_over,
    env=env,
    key=key,
    policy=partial(policy_action_parallel, policy=b_policy),
)

# Slice out the terminated steps and the number of required samples
behavioral_data = process_n_trajectories(
    behavioral_trajectories, not_terminated, mean_over, behavioral_samples
)

# Get empirical policy
d_policy = get_epmirical_policy(
    behavioral_data, jnp.arange(mean_over), n_states, n_actions
)

# Use true behavioral policy or empirical policy as regularization target
if use_d_policy:
    bd_policy = d_policy
else:
    bd_policy = jnp.repeat(b_policy[jnp.newaxis, ...], mean_over, axis=0)

# Setup agent parameters and training methods
agent_params = AgentConfig(
    agents=jnp.arange(mean_over),
    alpha=0.2,
    gamma=0.99,
    beta=beta,
    behavioral_policy=bd_policy,
)
learning_methods = {
    "q": agents_value_offline.train_parallel_greedy,
    "r": agents_value_offline.train_parallel_regularized,
    "g": agents_value_offline.train_parallel_gated,
}

# Run the experiment
training_results = train_value_agents(
    learning_methods,
    agent_params,
    behavioral_data,
    q_shape=(n_states, n_actions),
)
training_results["b"] = (
    b_values,
    bd_policy,
    jnp.argmax(bd_policy, axis=-1),
    b_errors,
)
evaluation_results, key = evaluate_value_agents(
    training_results, env, key, n_episodes=100000
)

print_evaluation_results(
    evaluation_results,
    beta=beta,
    convex_weight=convex_weight,
    train_samples=behavioral_samples,
)
