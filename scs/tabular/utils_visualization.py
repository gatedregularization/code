from __future__ import annotations

from collections import defaultdict
import os
from typing import TYPE_CHECKING

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

if TYPE_CHECKING:
    import jax


def animated_taxi(episodes: int, q_values: jax.Array | np.ndarray) -> None:
    """Runs rendered episodes of the OpenAI Taxi environment using the greedy
    policy based in the passed q_values and prints the total reward for each
    episode.
    """
    q_values = np.asarray(q_values)
    env = gym.make("Taxi-v3", render_mode="human")
    for e in range(episodes):
        reward_sum = 0.0
        state = env.reset()[0]
        for _t in range(1000):
            action = np.argmax(q_values[state])
            state, reward, terminal = env.step(int(action))[:3]
            reward_sum += float(reward)
            if terminal:
                break
        print(f"Episode {e} reward: {reward_sum}")
    env.close()


def parse_csv_data(
    folder: str | os.PathLike,
) -> dict[float, dict[int, pl.DataFrame]]:
    """Parses CSV files from the specified folder and organizes them into a
    nested dictionary structure.

    The structure is as follows:
    {
        convex_weight: {
            n_samples: DataFrame
        }
    }

    Each CSV file should be named in the format:
    <environment>_<n_samples>_<convex_weight>.csv
    """
    data: dict[float, dict[int, pl.DataFrame]] = defaultdict(
        lambda: defaultdict(dict)  # type: ignore[arg-type]
    )
    if not os.path.isdir(folder):
        raise ValueError(f"{folder} is not a directory")
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        if not os.path.isfile(path):
            raise ValueError(f"{path} is not a file")
        if not file.endswith(".csv"):
            raise TypeError(f"{path} is not a .csv file")
        split_filename = file.split("_")
        n_samples = int(split_filename[1])
        weight = float(split_filename[2])
        data[weight][n_samples] = pl.read_csv(path)
    return data


def plot_taxi_beta_rewards(
    data: dict[float, dict[int, pl.DataFrame]],
    convex_weight: float,
    n_samples: int,
    percentiles: tuple[int, int] = (5, 95),
    show_baseline: bool = True,
    max_beta: float | None = None,
    show_plot: bool = True,
) -> None:
    """Plots the mean rewards and confidence intervals over different beta values.

    Visualization Details:
    ------------------------
    This function creates a plot showing how rewards change as the regularization
    parameter beta varies. For each method, it shows:
    - The mean reward line
    - A confidence interval band defined by the specified percentiles


    Args:
        data: Nested dictionary containing the parsed CSV data, as returned by
            `parse_csv_data`, organized by convex_weight, and n_samples.
        convex_weight: The convex weight value to plot data for.
        n_samples: The number of samples to plot data for.
        percentiles: Tuple of (lower, upper) percentiles to use for confidence
            intervals.
        show_baseline: Whether to include baseline methods 'b' and 'q' in the plot.
            'b' stands for behavioral policy and 'q' for the Q-learning method.
        max_beta: Optional maximum beta value for x-axis limit.
        show_plot: Whether to display the plot immediately.
    """
    df = data[convex_weight][n_samples]
    fig, ax = plt.subplots(figsize=(30, 20))
    methods = ["r", "g"]
    colors = ["r", "g"]
    if show_baseline:
        methods = ["b", "q"] + methods
        colors = ["k", "b"] + colors
    for method, color in zip(methods, colors):
        ax.plot(
            df["beta"],
            df[f"{method}_mean"],
            color=color,
            label=f"{method} mean",
            linestyle="--" if method == "b" else "-",
        )
        ax.fill_between(
            df["beta"],
            df[f"{method}_q{percentiles[0]}"],
            df[f"{method}_q{percentiles[1]}"],
            color=color,
            alpha=0.2,
        )
    ax.set_xlabel("Beta")
    ax.set_ylabel("Mean reward")
    ax.set_title(f"Mean reward for c-weight {convex_weight} with {n_samples} samples")
    if max_beta is not None:
        ax.set_xlim(0, max_beta)
    ax.legend()
    plt.tight_layout()
    if show_plot:
        plt.show()


def plot_taxi_samples_rewards(
    data: dict[float, dict[int, pl.DataFrame]],
    convex_weight: float,
    beta: float | None = None,
    percentiles: tuple[int, int] = (5, 95),
    show_baseline: bool = True,
    show_plot: bool = True,
) -> None:
    """Plots the mean rewards and confidence intervals over different sample sizes.

    Visualization Details:
    ------------------------
    This function creates a plot showing how rewards change as the number of samples
    varies. For each method, it shows:
    - The mean reward line
    - A confidence interval band defined by the specified percentiles

    The most important argument is the 'beta' parameter. If it is set to a specific
    value it behaves as a fixed parameter and plots the mean reward values across
    the sample sizes for each method. If it is set to None, the function will, for
    each method and sample size, select the beta value that maximizes the mean
    reward. This allows to see the best possible performance of each method
    across different sample sizes.

    Args:
        data: Nested dictionary containing the parsed CSV data, as returned by
            `parse_csv_data`, organized by convex_weight, and n_samples.
        convex_weight: The convex weight value to plot data for.
        beta: Optional specific beta value to use for comparison. If None, the
            optimal beta (that maximizes mean reward) is selected for each method
            and sample size.
        percentiles: Tuple of (lower, upper) percentiles to use for confidence
            intervals.
        show_baseline: Whether to include baseline methods 'b' and 'q' in the plot.
            'b' stands for behavioral policy and 'q' for the Q-learning method.
        show_plot: Whether to display the plot immediately.
    """
    fig, ax = plt.subplots(figsize=(30, 20))
    methods = ["r", "g"]
    colors = ["r", "g"]
    if show_baseline:
        methods = ["b", "q"] + methods
        colors = ["k", "b"] + colors
    plot_data = defaultdict(list)
    samples = list(data[convex_weight].keys())
    samples.sort()
    for n_samples in samples:
        df = data[convex_weight][n_samples]
        for method in methods:
            if beta is not None:
                filtered_df = df.filter(df["beta"] == beta)
                plot_data[f"{method}_mean"].append(filtered_df[f"{method}_mean"][0])
                plot_data[f"{method}_std"].append(filtered_df[f"{method}_std"][0])
                plot_data[f"{method}_p{percentiles[0]}"].append(
                    filtered_df[f"{method}_p{percentiles[0]}"][0]
                )
                plot_data[f"{method}_p{percentiles[1]}"].append(
                    filtered_df[f"{method}_p{percentiles[1]}"][0]
                )
            else:
                # Filter df to get row of optimal beta for 'method'
                filtered_df = df[df[f"{method}_mean"].arg_max()]  # type: ignore[index]

                plot_data[f"{method}_mean"].append(filtered_df[f"{method}_mean"][0])
                plot_data[f"{method}_std"].append(filtered_df[f"{method}_std"][0])
                plot_data[f"{method}_p{percentiles[0]}"].append(
                    filtered_df[f"{method}_p{percentiles[0]}"][0]
                )
                plot_data[f"{method}_p{percentiles[1]}"].append(
                    filtered_df[f"{method}_p{percentiles[1]}"][0]
                )
                plot_data[f"{method}_beta"].append(filtered_df["beta"][0])
        plot_data["n_samples"].append(n_samples)
    plot_data_df = pl.DataFrame(plot_data)
    for method, color in zip(methods, colors):
        ax.plot(
            plot_data_df["n_samples"],
            plot_data_df[f"{method}_mean"],
            color=color,
            label=f"{method} mean",
            linestyle="--" if method == "b" else "-",
        )
        ax.fill_between(
            plot_data_df["n_samples"],
            plot_data_df[f"{method}_p{percentiles[0]}"],
            plot_data_df[f"{method}_p{percentiles[1]}"],
            color=color,
            alpha=0.2,
        )
    ax.set_xlabel("n samples")
    ax.set_ylabel("Mean reward")
    ax.set_ylim(-150, 15)
    if beta is None:
        ax.set_title(
            f"Mean reward for c-weight {convex_weight} with individually tuned betas"
        )
    else:
        ax.set_title(f"Mean reward for c-weight {convex_weight} with beta {beta}")
    ax.legend()
    plt.tight_layout()
    if show_plot:
        plt.show()


def visualize_taxi_values(q_table: jax.Array | np.ndarray) -> None:
    """Visualizes the Q-table for the OpenAI Taxi environment.

    Visualization Details:
    ------------------------
    The function creates a 4x4 grid to visualize all scenarios:
    - Each subplot corresponds to a combination of passenger starting location
      and drop-off location.
    - If the passenger's starting location matches the drop-off location, the
      subplot represents the scenario where the passenger is already in the taxi
      and ready to be dropped off.

    The Q-values are reduced by taking the maximum action value for each state,
    highlighting the optimal policy. Also the label of the action corresponding
    to the maximum Q-value is displayed in each cell.

    Explanation of State Encoding:
    --------------------------------
    The OpenAI Taxi environment encodes states as integers between 0 and 499.
    Each state is calculated as:

        ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination

    Where:
    - `taxi_row`: The row where the taxi is located (0-4).
    - `taxi_col`: The column where the taxi is located (0-4).
    - `passenger_location`: The passenger's location (0-4). Values:
        - 0: Red
        - 1: Green
        - 2: Yellow
        - 3: Blue
        - 4: In the taxi
    - `destination`: The destination location (0-3). Values:
        - 0: Red
        - 1: Green
        - 2: Yellow
        - 3: Blue

    State decoding:
    To visualize the Q-values, the state encoding is reversed:
    1. Determine the destination: `state % 4`.
    2. Determine the passenger location: `(state // 4) % 5`.
    3. Determine the taxi's column: `(state // 20) % 5`.
    4. Determine the taxi's row: `(state // 100) % 5`.
    """
    q_table = np.asarray(q_table)
    action_labels = ["S", "N", "E", "W", "P", "D"]

    # Compute the optimal action for each state
    optimal_actions = np.argmax(q_table, axis=1)
    max_q_values = np.max(q_table, axis=1)

    # Determine global min and max for consistent scaling
    global_min = max_q_values.min()
    global_max = max_q_values.max()

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i in range(4):  # Passenger starting location
        for j in range(4):  # Drop-off location
            if i == j:  # If drop-off = pickup; show passenger in car instead
                indices = [
                    state
                    for state in range(500)
                    if (state % 20 // 4 == 4) and (state % 4 == j)
                ]
            else:
                indices = [
                    state
                    for state in range(500)
                    if (state % 20 // 4 == i) and (state % 4 == j)
                ]
            q_subset = max_q_values[indices]
            action_subset = optimal_actions[indices]
            matrix = q_subset.reshape(5, 5)  # Reshape to 5x5 grid for taxi positions
            action_matrix = action_subset.reshape(5, 5)

            ax = axes[i, j]
            ax.imshow(
                matrix, cmap="viridis", aspect="auto", vmin=global_min, vmax=global_max
            )
            if i == j:
                ax.set_title(f"In Car, Drop-off: {j}", fontsize=10)
            else:
                ax.set_title(f"Start: {i}, Drop-off: {j}", fontsize=10)
            ax.set_xlabel("pos Col")
            ax.set_ylabel("pos Row")

            # Add action labels to each cell
            for row in range(5):
                for col in range(5):
                    label = action_labels[action_matrix[row, col]]
                    ax.text(
                        col,
                        row,
                        label,
                        ha="center",
                        va="center",
                        color="red",
                        fontsize=10,
                    )
    plt.show()
