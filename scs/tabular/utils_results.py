from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

import jax
import numpy as np
import polars as pl

SCHEMA_VERSION = 1
PERCENTILES = (5, 10, 20, 25, 50, 75, 80, 90, 95)


@dataclass(frozen=True)
class TabularExperimentHyperparameters:
    """Effective configuration needed to identify and reproduce a tabular run."""

    environment: str
    seed: int
    beta: float
    convex_weight: float
    perturbation: str
    modification_type: str
    behavioral_policy_source: str
    behavioral_samples: int
    behavioral_episodes_per_agent: int
    n_agents: int
    evaluation_episodes_per_agent: int
    max_steps: int
    value_iteration_steps: int
    alpha: float
    gamma: float
    gate_floor: float = 0.0


EvaluationResults = dict[str, dict[str, dict[str, jax.Array]]]
TrainingResults = dict[str, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]


def _percentile_columns(values: np.ndarray) -> dict[str, float]:
    """Computes the named reward percentiles used in result tables.

    Args:
        values: Reward samples. All dimensions are pooled into one distribution.

    Returns:
        Mapping from stable percentile column names to scalar values.
    """
    percentiles = np.percentile(values, PERCENTILES)
    return {
        f"reward_p{percentile:02d}": float(value)
        for percentile, value in zip(PERCENTILES, percentiles, strict=True)
    }


def _reward_frames(
    evaluation_results: EvaluationResults,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Builds pooled and per-agent reward tables from evaluation returns.

    Args:
        evaluation_results: Results grouped by method and policy type. Each result
            must contain ``sum_rewards`` with shape ``(episodes, agents)`` or
            ``(episodes,)`` for a single agent.

    Returns:
        Data frames containing one row per evaluated policy and one row per agent,
        respectively.

    Raises:
        ValueError: If an episode-return array does not have a supported shape.
    """
    summary_rows: list[dict[str, str | int | float]] = []
    agent_rows: list[dict[str, str | int | float]] = []

    for method, policy_results in evaluation_results.items():
        for policy_type, metrics in policy_results.items():
            rewards = np.asarray(metrics["sum_rewards"])
            if rewards.ndim == 1:
                rewards = rewards[:, np.newaxis]
            if rewards.ndim != 2:
                raise ValueError(
                    "sum_rewards must have shape (episodes, agents), "
                    f"got {rewards.shape} for {method}/{policy_type}"
                )

            episodes, n_agents = rewards.shape
            summary_rows.append(
                {
                    "method": method,
                    "policy_type": policy_type,
                    "n_agents": n_agents,
                    "episodes_per_agent": episodes,
                    "reward_mean": float(np.mean(rewards)),
                    "reward_std": float(np.std(rewards)),
                    **_percentile_columns(rewards),
                }
            )
            for agent, agent_rewards in enumerate(rewards.T):
                agent_rows.append(
                    {
                        "method": method,
                        "policy_type": policy_type,
                        "agent": agent,
                        "episodes": episodes,
                        "reward_mean": float(np.mean(agent_rewards)),
                        "reward_std": float(np.std(agent_rewards)),
                        **_percentile_columns(agent_rewards),
                    }
                )

    return pl.DataFrame(summary_rows), pl.DataFrame(agent_rows)


def _state_value_comparison_frame(
    training_results: TrainingResults,
) -> pl.DataFrame:
    """Builds per-agent checks of gated values against regularized values.

    Args:
        training_results: Trained arrays keyed by method. The ``r`` and ``g``
            entries must contain Q-values and policies with matching agent and state
            dimensions.

    Returns:
        A data frame containing the number and fraction of states where
        ``V_gated < V_regularized`` for each agent.

    Raises:
        ValueError: If regularized and gated state-value arrays have different
            shapes.
    """
    regularized_q_values, regularized_policy, _, _ = training_results["r"]
    gated_q_values, gated_policy, _, _ = training_results["g"]
    regularized_values = np.asarray(
        (regularized_policy * regularized_q_values).sum(axis=-1)
    )
    gated_values = np.asarray((gated_policy * gated_q_values).sum(axis=-1))

    if regularized_values.ndim == 1:
        regularized_values = regularized_values[np.newaxis, :]
    if gated_values.ndim == 1:
        gated_values = gated_values[np.newaxis, :]
    if regularized_values.shape != gated_values.shape:
        raise ValueError(
            "regularized and gated state values must have matching shapes, "
            f"got {regularized_values.shape} and {gated_values.shape}"
        )

    rows: list[dict[str, int | float | bool]] = []
    for agent, (regularized, gated) in enumerate(
        zip(regularized_values, gated_values, strict=True)
    ):
        violations = gated < regularized
        n_states = int(violations.size)
        violating_states = int(np.sum(violations))
        rows.append(
            {
                "agent": agent,
                "n_states": n_states,
                "all_states_hold": violating_states == 0,
                "violating_states": violating_states,
                "violating_fraction": violating_states / n_states,
            }
        )
    return pl.DataFrame(rows)


def save_tabular_experiment_results(
    hyperparameters: TabularExperimentHyperparameters,
    evaluation_results: EvaluationResults,
    training_results: TrainingResults,
    output_root: str | Path = "data/results",
) -> Path:
    """Stores one completed experiment run in a folder.

    Files are first written to a temporary sibling directory. Renaming that
    directory publishes the run atomically, preventing incomplete runs from being
    mistaken for valid results.

    Args:
        hyperparameters: Effective configuration used by the experiment.
        evaluation_results: Episode returns grouped by method and policy type.
        training_results: Trained method arrays used for state-value comparisons.
        output_root: Parent directory under which environments and runs are stored.

    Returns:
        Path to the newly published run directory.
    """
    created_at = datetime.now(UTC)
    run_id = f"{created_at:%Y%m%dT%H%M%S.%fZ}_{uuid4().hex[:8]}"
    environment_dir = Path(output_root) / hyperparameters.environment
    environment_dir.mkdir(parents=True, exist_ok=True)
    final_path = environment_dir / run_id
    summary, agents = _reward_frames(evaluation_results)
    state_comparison = _state_value_comparison_frame(training_results)
    temporary_path = Path(tempfile.mkdtemp(prefix=f".{run_id}_", dir=environment_dir))

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        **asdict(hyperparameters),
    }

    try:
        with (temporary_path / "hyperparameters.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
            file.write("\n")
        summary.write_csv(temporary_path / "evaluation_summary.csv")
        agents.write_csv(temporary_path / "evaluation_agents.csv")
        state_comparison.write_csv(temporary_path / "state_value_comparison.csv")
        temporary_path.replace(final_path)
    except BaseException:
        shutil.rmtree(temporary_path, ignore_errors=True)
        raise

    return final_path
