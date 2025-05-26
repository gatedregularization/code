from __future__ import annotations

import copy
from typing import (
    TYPE_CHECKING,
    Callable,
)

from flax import (
    nnx,
    struct,
)
import jax
import optax

from scs import (
    ACNN,
    QNN,
)
from scs.cartpole.agent_a2c_online import A2Cc
from scs.cartpole.agents_value_offline import QNNc
from scs.lander.agent_a2c_online import A2Cll
from scs.lander.agents_value_offline import QNNll
from scs.qlearning import eval_action_greedy

if TYPE_CHECKING:
    import gymnasium as gym

    from scs.utils import (
        AgentConfig,
        BehavioralData,
    )


class NNTrainingState(struct.PyTreeNode):
    """Training state container for Neural Network that can be passed through
    JAX transformations.

    Attributes:
        model_def: Network graph definition.
        model_state: Current model parameters.
        target_model_state: Target network parameters for stable training.
        tau: Weight for target network update (soft update parameter).
        optimizer: Gradient transformation for optimization.
        optimizer_state: Current state of the optimizer.
    """

    model_def: nnx.GraphDef = struct.field(pytree_node=False)
    model_state: nnx.State
    target_model_state: nnx.State
    tau: float = struct.field(pytree_node=False)
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    optimizer_state: optax.OptState

    def apply_gradients(self, grads: nnx.State) -> NNTrainingState:
        """Apply gradients to model parameters and update target network.

        Performs a gradient step and updates the target network using soft updates
        with the tau parameter.

        Args:
            grads: Gradients for the model parameters.

        Returns:
            Updated `NNTrainingState` instance.
        """
        updates, optimizer_state = self.optimizer.update(grads, self.optimizer_state)
        model_state = optax.apply_updates(self.model_state, updates)
        target_model_state = jax.tree.map(
            lambda tp, p: self.tau * p + (1 - self.tau) * tp,
            self.target_model_state,
            model_state,
        )
        return self.replace(
            model_state=model_state,
            target_model_state=target_model_state,
            optimizer_state=optimizer_state,
        )

    @classmethod
    def create(
        cls,
        model_def: nnx.GraphDef,
        model_state: nnx.State,
        target_model_state: nnx.State,
        tau: float,
        optimizer: optax.GradientTransformation,
    ) -> NNTrainingState:
        """Create a new training state instance.

        Args:
            model_def: Network graph definition.
            model_state: Initial model parameters.
            target_model_state: Initial target network parameters.
            tau: Target network update rate (soft update parameter).
            optimizer: Gradient transformation for optimization.

        Returns:
            A new NNTrainingState instance.
        """
        optimizer_state = optimizer.init(model_state)
        return cls(
            model_def=model_def,
            model_state=model_state,
            target_model_state=target_model_state,
            tau=tau,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
        )


@nnx.jit(static_argnums=(2,))
def soft_update_target_model(
    model: nnx.Module,
    model_target: nnx.Module,
    tau: float,
) -> nnx.Module:
    """Soft updates the target model..

    Updates the parameters of the target model using a convex combination, with
    weight `tau`, of the current model parameters and the target model parameters.
    """
    model_params = nnx.state(model)
    graph_def, target_params, batch_stats = nnx.split(  # type: ignore[misc]
        model_target, nnx.Param, nnx.BatchStat
    )
    updated_params = jax.tree.map(
        lambda tp, p: tau * p + (1 - tau) * tp,
        target_params,
        model_params,
    )
    return nnx.merge(graph_def, updated_params, batch_stats)


def train_value_agents(
    learning_methods: dict[str, Callable],
    base_train_state: NNTrainingState,
    agent_params: AgentConfig,
    offline_train_data: BehavioralData,
) -> dict[str, tuple[QNNc | QNNll | A2Cc | A2Cll, jax.Array]]:
    """Trains multiple neural network value-based agents as defined in
    `learning_methods`.

    Args:
        learning_methods: Dictionary mapping method names to their training
            functions. Each function should accept (train_state, offline_train_data,
            agent_params) and return (updated_train_state, losses).
        base_train_state: Initial training state containing model definition,
            parameters, target network, optimizer, and other training components.
        agent_params: Configuration object containing agent parameters.
        offline_train_data: Behavioral data containing states, actions, rewards,
            next_states, terminals, and the behavioral logits of the next states
            that can be used to obtain the behavioral policy for the value updates.

    Returns:
        Dictionary mapping method names to tuples containing:
        - Trained neural network model (QNNc, QNNll, A2Cc, or A2Cll)
        - Training loss array recorded during the training process
    """
    training_results = {}
    for method_name, learning_method in learning_methods.items():
        print(
            f"Training {method_name} on "
            f"{offline_train_data.states.shape[0]} samples."
        )
        train_state = copy.deepcopy(base_train_state)
        train_state, losses = learning_method(
            train_state,
            offline_train_data,
            agent_params,
        )
        training_results[method_name] = (
            nnx.merge(train_state.model_def, train_state.model_state),
            losses,
        )
    return training_results


def evaluate_agents(
    models: dict[str, tuple[QNNc | QNNll | A2Cc | A2Cll, jax.Array]],
    agent_params: AgentConfig,
    envs: gym.vector.SyncVectorEnv,
    n_episodes: int,
    rngs: nnx.Rngs,
) -> tuple[dict[str, list[float]], nnx.Rngs]:
    """Evaluates trained neural network agents on `n_episodes` episodes.

    Args:
        models: Dictionary mapping method names to tuples containing:
            [Trained neural network model, training losses array]
        agent_params: Configuration object containing agent parameters.
        envs: Vectorized gymnasium environment for parallel episode execution.
        n_episodes: Number of episodes to run for each model evaluation.
        rngs: Random number generators for stochastic operations during evaluation.

    Returns:
        Tuple containing:
        - Dictionary mapping method names to lists of episode rewards
        - Updated random number generators after evaluation
    """
    evaluation_results: dict[str, list[float]] = {}
    for method_name, (model, _) in models.items():
        print(f"Evaluating {method_name} model on {n_episodes} episodes.")
        if isinstance(model, (QNNc, QNNll)):
            eval_results, rngs = QNN.evaluate_QNN_model(
                model,
                eval_action_greedy,
                envs,
                agent_params,
                rngs,
                episodes=n_episodes,
            )
        elif isinstance(model, (A2Cc, A2Cll)):
            eval_results, rngs = ACNN.evaluate_ACNN_model(
                model,
                envs,
                agent_params,
                rngs,
                episodes=n_episodes,
            )
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        evaluation_results[method_name] = eval_results
    return evaluation_results, rngs
