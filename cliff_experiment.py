from __future__ import annotations

from functools import partial
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
from jax import random
import jax.numpy as jnp

from scs.tabular import (
    agents_value_offline,
    env_gym,
)
from scs.tabular.core_tabular import (
    evaluate_value_agents,
    policy_action_parallel,
    train_value_agents,
)
from scs.tabular.utils_results import (
    TabularExperimentHyperparameters,
    save_tabular_experiment_results,
)
from scs.tabular.utils_tabular import (
    get_epmirical_policy,
    policy_modification,
    process_n_trajectories,
)
from scs.utils import AgentConfig

############################################################################
# Hyperparameters
############################################################################
mean_over: int = 10
max_steps: int = 100
behavioral_samples: int = 2500000
behavioral_episodes_per_agent: int = behavioral_samples // mean_over
evaluation_episodes_per_agent: int = 100000
value_iteration_steps: int = 1000
alpha: float = 0.2
gamma: float = 0.99
beta: float = 2
gate_floor: float = 0.0
convex_weight: float = 0.5
seed: int = 0
perturbation: str = "dirichlet"  # "constructed", "uniform", or "dirichlet"
use_d_policy: bool = False  # Use the or the empirical or actual behavioral policy
store_outputs: bool = True
############################################################################

# Setup Experiment
key = random.PRNGKey(seed)
env = env_gym.create_CliffWalker(max_steps)
n_states = int(env.states.shape[0])
n_actions = int(env.n_actions)

# Get optimal q-values
b_values, b_errors = env_gym.solve_taxi_greedy(env, gamma, value_iteration_steps)
b_policy = (
    jnp.zeros((n_states, n_actions))
    .at[jnp.arange(n_states), jnp.argmax(b_values, axis=-1)]
    .set(1.0)
)

# Make the behavioral policy suboptimal and inhomogeneous

# The CliffWalker environment represents a grid world with 4 rows and 12
# columns.
# Each state is encoded as an integer: state = row * 12 + col
#
# The layout is as follows:
# - The agent starts at the bottom-left corner
# - The goal is at the bottom-right corner
# - The bottom row (row 3) contains a cliff between the start and goal
# - Falling off the cliff results in a large negative reward and a reset to the
#   start
# - There are 4 possible actions: up, right, down, left (0, 1, 2, 3)
# - Each step incurs a small negative reward, encouraging the agent to reach the
#   goal quickly
# - The episode terminates when the agent reaches the goal or after max_steps

perturb_indices: jax.Array | None
if perturbation == "constructed":
    top_half = jnp.array([row * 12 + col for row in range(2) for col in range(12)])
    perturb_indices = top_half
    modification_type = "dirichlet"
else:
    perturb_indices = None
    modification_type = perturbation

key, b_policy = policy_modification(
    key,
    b_policy,
    modification_type=modification_type,
    indices=perturb_indices,
    convex_weight=convex_weight,
)

# Ensure the behavioral agent does not simply follow the deterministic policy
# along the cliff but does reach the non-optimal rows 1 and 2.
if perturbation == "constructed":
    b_policy = b_policy.at[24].set(jnp.array([0.5, 0.5, 0.0, 0.0]))

# Ensure the first state policy covers the action of falling off the cliff
# for q-agent to not choose the unencoutered action with value 0.
if perturbation == "constructed":
    b_policy = b_policy.at[36].set(jnp.array([0.99, 0.01, 0.0, 0.0]))

# Generate behavioral data
(
    behavioral_trajectories,
    not_terminated,
    key,
) = env_gym.generate_trajectory_parallel(
    n_episodes=behavioral_episodes_per_agent,
    n_agents=mean_over,
    env=env,
    key=key,
    policy=partial(policy_action_parallel, policy=b_policy),
)

# Slice out the terminated steps and the number of required samples
behavioral_data = process_n_trajectories(
    behavioral_trajectories, not_terminated, mean_over, behavioral_samples
)

# Get empirical policyprint results to the console
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
    alpha=alpha,
    gamma=gamma,
    beta=beta,
    gate_floor=gate_floor,
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
    training_results, env, key, n_episodes=evaluation_episodes_per_agent
)

if store_outputs:
    run_path = save_tabular_experiment_results(
        TabularExperimentHyperparameters(
            environment="cliffwalker",
            seed=seed,
            beta=beta,
            gate_floor=gate_floor,
            convex_weight=convex_weight,
            perturbation=perturbation,
            modification_type=modification_type,
            behavioral_policy_source="empirical" if use_d_policy else "true",
            behavioral_samples=behavioral_samples,
            behavioral_episodes_per_agent=behavioral_episodes_per_agent,
            n_agents=mean_over,
            evaluation_episodes_per_agent=evaluation_episodes_per_agent,
            max_steps=max_steps,
            value_iteration_steps=value_iteration_steps,
            alpha=alpha,
            gamma=gamma,
        ),
        evaluation_results,
        training_results,
    )
    print(f"Results stored in {run_path}")
