from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
import numpy as np

# The experiment is normally run from an IDE or a terminal, where a GUI plotting
# backend is not required.  Agg writes image files without opening any windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from point_maze_medium_config import PointMazeMediumConfig
from point_maze_open_config import PointMazeOpenConfig
from experiment import _episode, _new_agent, _summary
from environment import flatten_observation, make_env
from planner import planner_for_env


PointMazeConfig = Union[PointMazeOpenConfig, PointMazeMediumConfig]


def _epsilon_decay_horizon(config: PointMazeConfig, training_episodes: int) -> int:
    """Return the per-agent episode at which epsilon reaches its minimum."""
    if training_episodes == 0:
        return 0
    return max(1, int(math.ceil(training_episodes * config.epsilon_decay_fraction)))


def _training_epsilon(
    config: PointMazeConfig,
    episode_index: int,
    training_episodes: int,
) -> float:
    """Linearly anneal epsilon over the configured fraction of this budget."""
    decay_horizon = _epsilon_decay_horizon(config, training_episodes)
    # episode_index is zero-based: first normal-training episode uses epsilon_start,
    # and the final episode of the decay horizon uses epsilon_final.
    fraction = min(1.0, episode_index / max(1, decay_horizon - 1))
    return config.epsilon_start + fraction * (config.epsilon_final - config.epsilon_start)


def _normal_training(
    env,
    agent,
    config: PointMazeConfig,
    episodes: int,
    reset_seed_offset: int,
) -> List[Dict[str, object]]:
    return [
        _episode(
            env,
            agent,
            config.seed + reset_seed_offset + index,
            planner=None,
            learn=True,
            config=config,
            epsilon=_training_epsilon(config, index, episodes),
        )
        for index in range(episodes)
    ]


def _evaluation(
    env,
    agent,
    config: PointMazeConfig,
    episodes: int,
    reset_seed_offset: int,
) -> List[Dict[str, object]]:
    return [
        _episode(
            env,
            agent,
            config.seed + reset_seed_offset + index,
            planner=None,
            learn=False,
            config=config,
        )
        for index in range(episodes)
    ]


def _rewards(episodes: List[Dict[str, object]]) -> List[float]:
    """Extract one total reward per episode from the saved episode records."""
    return [float(episode["reward"]) for episode in episodes]


def _plot_phase(
    axes,
    rewards: List[float],
    start_episode: int,
    label: str,
    color: str,
    smoothing_window: int = 25,
) -> None:
    """Plot raw rewards lightly and a trailing moving-average reward curve."""
    if not rewards:
        return

    x_values = np.arange(start_episode + 1, start_episode + len(rewards) + 1)
    reward_values = np.asarray(rewards, dtype=float)
    axes.plot(x_values, reward_values, color=color, alpha=0.18, linewidth=0.8)

    window = min(smoothing_window, len(rewards))
    # Use the rewards available so far at the beginning of a phase.  This keeps
    # the visible learning curve aligned with episode 1 instead of beginning at
    # episode 25 solely because the smoothing window has length 25.
    cumulative_rewards = np.cumsum(reward_values)
    smoothed_rewards = np.empty(len(rewards), dtype=float)
    for index in range(len(rewards)):
        first_index = max(0, index - window + 1)
        reward_sum = cumulative_rewards[index]
        if first_index:
            reward_sum -= cumulative_rewards[first_index - 1]
        smoothed_rewards[index] = reward_sum / (index - first_index + 1)
    axes.plot(
        x_values,
        smoothed_rewards,
        color=color,
        linewidth=2.8,
        marker="o",
        markersize=3.5,
        markevery=max(1, len(rewards) // 30),
        label="{} reward ({}-episode moving average)".format(label, window),
    )


def _save_figure(figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_epsilon(
    reward_axes,
    config: PointMazeConfig,
    training_episodes: int,
    start_episode: int,
) -> object:
    """Add the epsilon schedule on a secondary axis for a DQN training phase."""
    epsilon_axes = reward_axes.twinx()
    if training_episodes:
        x_values = np.arange(start_episode + 1, start_episode + training_episodes + 1)
        epsilon_values = [
            _training_epsilon(config, episode_index, training_episodes)
            for episode_index in range(training_episodes)
        ]
        epsilon_axes.plot(
            x_values,
            epsilon_values,
            color="tab:purple",
            linewidth=2.0,
            linestyle=":",
            label="Epsilon",
            zorder=5,
        )
    epsilon_axes.set_ylabel("Exploration epsilon", color="tab:purple")
    epsilon_axes.tick_params(axis="y", colors="tab:purple")
    epsilon_axes.set_ylim(0.0, max(1.05, config.epsilon_start * 1.05))
    return epsilon_axes


def _add_combined_legend(reward_axes, epsilon_axes) -> None:
    """Show reward and epsilon labels together despite the two y-axes."""
    reward_handles, reward_labels = reward_axes.get_legend_handles_labels()
    epsilon_handles, epsilon_labels = epsilon_axes.get_legend_handles_labels()
    if reward_handles or epsilon_handles:
        reward_axes.legend(
            reward_handles + epsilon_handles,
            reward_labels + epsilon_labels,
            loc="best",
        )


def save_planned_vs_plain_training_plots(
    config: PointMazeConfig,
    result: Dict[str, object],
) -> Dict[str, str]:
    """Save the planned and plain DQN reward curves produced by one comparison.

    The planned plot places planner-only warm-up and normal DQN training on one
    episode axis.  The red line marks the first normal-training episode.
    """
    plots_dir = config.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        "warmup_{}_planned_train_{}_plain_train_{}_seed_{}".format(
            result["planner_warmup_episodes"],
            result["planned_train_episodes"],
            result["plain_train_episodes"],
            config.seed,
        )
    )

    planned = result["planned"]
    planned_warmup = _rewards(planned["warmup"])
    planned_training = _rewards(planned["training"])
    planned_path = plots_dir / "{}_planned_dqn_training_{}.png".format(
        config.environment_slug(), suffix
    )

    figure, axes = plt.subplots(figsize=(10, 5))
    _plot_phase(axes, planned_warmup, 0, "Planner warm-up", "tab:blue")
    _plot_phase(
        axes,
        planned_training,
        len(planned_warmup),
        "Normal DQN training",
        "tab:orange",
    )
    if planned_warmup:
        boundary_x = len(planned_warmup) + 0.5
        axes.axvline(
            x=boundary_x,
            color="red",
            linewidth=2.5,
            linestyle="--",
            label="Warm-up ends / normal training starts",
            zorder=10,
        )
        axes.annotate(
            "Warm-up ends",
            xy=(boundary_x, 0.98),
            xycoords=("data", "axes fraction"),
            xytext=(5, -5),
            textcoords="offset points",
            color="red",
            ha="left",
            va="top",
            fontsize=9,
            zorder=11,
        )
    if not planned_warmup and not planned_training:
        axes.text(0.5, 0.5, "No planned training episodes", ha="center", va="center", transform=axes.transAxes)
    axes.set_title("{} planned DQN: planner warm-up and normal training".format(config.env_id))
    axes.set_xlabel("Episode")
    axes.set_ylabel("Total episode reward")
    axes.grid(alpha=0.25)
    epsilon_axes = _plot_epsilon(
        axes,
        config,
        training_episodes=len(planned_training),
        start_episode=len(planned_warmup),
    )
    _add_combined_legend(axes, epsilon_axes)
    _save_figure(figure, planned_path)

    plain = result["plain"]
    plain_training = _rewards(plain["training"])
    plain_path = plots_dir / "{}_plain_dqn_training_{}.png".format(
        config.environment_slug(), suffix
    )

    figure, axes = plt.subplots(figsize=(10, 5))
    _plot_phase(axes, plain_training, 0, "Normal DQN training", "tab:green")
    if not plain_training:
        axes.text(0.5, 0.5, "No plain-DQN training episodes", ha="center", va="center", transform=axes.transAxes)
    axes.set_title("{} plain DQN: normal training".format(config.env_id))
    axes.set_xlabel("Episode")
    axes.set_ylabel("Total episode reward")
    axes.grid(alpha=0.25)
    epsilon_axes = _plot_epsilon(
        axes,
        config,
        training_episodes=len(plain_training),
        start_episode=0,
    )
    _add_combined_legend(axes, epsilon_axes)
    _save_figure(figure, plain_path)

    return {
        "planned_training_curve": str(planned_path),
        "plain_training_curve": str(plain_path),
    }


def run_planned_vs_plain_dqn(
    config: Optional[PointMazeConfig] = None,
    planner_warmup_episodes: int = 0,
    planned_train_episodes: int = 0,
    plain_train_episodes: int = 0,
    test_episodes: int = 500,
) -> Dict[str, object]:
    """Compare DQN with planner-only warm-up against an otherwise identical DQN.

    Planned DQN: planner-only warm-up, a greedy post-warm-up test when normal
    training is requested, then epsilon-greedy normal DQN training and a final
    greedy test.
    Plain DQN: no planner warm-up, then its independently sized DQN training.
    Both agents use the same deterministic evaluation tasks with epsilon fixed at zero.
    """
    if min(planner_warmup_episodes, planned_train_episodes, plain_train_episodes, test_episodes) < 0:
        raise ValueError("Phase episode counts cannot be negative")
    if test_episodes == 0:
        raise ValueError("test_episodes must be positive")

    config = config or PointMazeOpenConfig()
    config.validate()
    env = make_env(config.env_id, config.max_episode_steps)
    observation, _ = env.reset(seed=config.seed)
    state_size, action_size = flatten_observation(observation).size, env.action_space.n

    planned_agent = _new_agent(config, state_size, action_size)
    planner = planner_for_env(config.env_id)
    planned_warmup = [
        _episode(env, planned_agent, config.seed + index, planner, True, config)
        for index in range(planner_warmup_episodes)
    ]
    planned_post_warmup_test = None
    if planned_train_episodes > 0:
        planned_post_warmup_test = _evaluation(
            env,
            planned_agent,
            config,
            test_episodes,
            reset_seed_offset=300_000,
        )
    planned_training = _normal_training(env, planned_agent, config, planned_train_episodes, 200_000)
    planned_final_test = _evaluation(
        env,
        planned_agent,
        config,
        test_episodes,
        reset_seed_offset=400_000,
    )

    plain_agent = _new_agent(config, state_size, action_size)
    plain_training = _normal_training(env, plain_agent, config, plain_train_episodes, 200_000)
    # This seed set matches the planned agent's final test, not its diagnostic
    # post-warm-up test.
    plain_test = _evaluation(
        env,
        plain_agent,
        config,
        test_episodes,
        reset_seed_offset=400_000,
    )
    env.close()

    planned_steps = _summary(planned_warmup + planned_training)["environment_steps"]
    plain_steps = _summary(plain_training)["environment_steps"]
    return {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "planner_warmup_episodes": planner_warmup_episodes,
        "planned_train_episodes": planned_train_episodes,
        "plain_train_episodes": plain_train_episodes,
        "epsilon_decay_fraction": config.epsilon_decay_fraction,
        "test_episodes": test_episodes,
        "planned": {
            "warmup": planned_warmup,
            "warmup_summary": _summary(planned_warmup),
            "post_warmup_test": planned_post_warmup_test,
            "post_warmup_test_summary": (
                _summary(planned_post_warmup_test)
                if planned_post_warmup_test is not None
                else None
            ),
            "training": planned_training,
            "training_summary": _summary(planned_training),
            "post_training_test": planned_final_test if planned_train_episodes > 0 else None,
            "post_training_test_summary": (
                _summary(planned_final_test) if planned_train_episodes > 0 else None
            ),
            # Kept as the final planned test for existing callers and for the
            # direct planned-vs-plain comparison.
            "test": planned_final_test,
            "test_summary": _summary(planned_final_test),
            "pre_test_environment_steps": planned_steps,
            "epsilon_decay_horizon_episodes": _epsilon_decay_horizon(
                config, planned_train_episodes
            ),
            "optimizer_updates": planned_agent.updates,
        },
        "plain": {
            "training": plain_training,
            "training_summary": _summary(plain_training),
            "test": plain_test,
            "test_summary": _summary(plain_test),
            "pre_test_environment_steps": plain_steps,
            "epsilon_decay_horizon_episodes": _epsilon_decay_horizon(
                config, plain_train_episodes
            ),
            "optimizer_updates": plain_agent.updates,
        },
    }


def save_planned_vs_plain_result(
    config: PointMazeConfig,
    result: Dict[str, object],
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    result["artifacts"] = save_planned_vs_plain_training_plots(config, result)
    path = config.output_dir / (
        f"{config.environment_slug()}_dqn_planned_vs_plain_warmup_{result['planner_warmup_episodes']}"
        f"_planned_train_{result['planned_train_episodes']}"
        f"_plain_train_{result['plain_train_episodes']}_seed_{config.seed}.json"
    )
    path.write_text(json.dumps(result, indent=2))
    return path
