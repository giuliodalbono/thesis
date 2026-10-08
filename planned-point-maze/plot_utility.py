"""
Plotting utilities for PointMaze training curves and test distributions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


PLOT_SHOW = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "resources" / "planned-point-maze" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_WINDOW_DIVISOR = 50
TRAIN_WINDOW_MIN = 50
TRAIN_WINDOW_MAX = 1000
TEST_HIST_BINS = 20


def _safe_filename(title, fallback):
    base = title.strip() if title else fallback
    base = re.sub(r"[^\w\s\-().]", "_", base)
    base = re.sub(r"\s+", " ", base).strip()
    return f"{base}.png"


def _resolve_output_dir(output_subdir=None):
    path = OUTPUT_DIR if not output_subdir else OUTPUT_DIR / output_subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig, title, fallback, output_subdir=None):
    filename = _safe_filename(title, fallback)
    path = _resolve_output_dir(output_subdir) / filename
    fig.savefig(path, dpi=150)


def _format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.2f} s"
    if seconds < 3600:
        minutes, secs = divmod(seconds, 60)
        return f"{int(minutes)} min {secs:.1f} s"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours)} h {int(minutes)} min {secs:.0f} s"


def _stats_text(stats: Dict[str, float]):
    lines = [
        f"Episodes: {stats['episodes']}",
        f"Training time: {_format_duration(stats['training_time_s'])}",
    ]
    return "\n".join(lines)


def _smoothed_rewards(rewards, window=None):
    rewards = np.asarray(rewards, dtype=float)
    if window is None:
        window = int(
            np.clip(len(rewards) // TRAIN_WINDOW_DIVISOR, TRAIN_WINDOW_MIN, TRAIN_WINDOW_MAX)
        )
    return np.array(
        [np.mean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))]
    )


def plot_training_curve(
    rewards,
    title: str = "",
    epsilon=None,
    stats: Optional[Dict[str, float]] = None,
    output_subdir=None,
):
    rewards = np.asarray(rewards, dtype=float)
    fig, ax_reward = plt.subplots(figsize=(12, 5))
    episodes = np.arange(len(rewards))
    smoothed = _smoothed_rewards(rewards)

    if len(rewards) > 0:
        ax_reward.plot(episodes, rewards, color="C0", alpha=0.2, linewidth=1, label="Reward (raw)")
    ax_reward.plot(episodes, smoothed, color="C0", linewidth=2, label="Reward (smoothed)")
    ax_reward.set_title(title)
    ax_reward.set_xlabel("Episodes")
    ax_reward.set_ylabel("Reward")

    legend_handles = []
    legend_labels = []
    if len(rewards) > 0:
        legend_handles.append(ax_reward.lines[0])
        legend_labels.append("Reward (raw)")
    legend_handles.append(ax_reward.lines[-1])
    legend_labels.append("Reward (smoothed)")

    if epsilon is not None:
        eps = np.asarray(epsilon, dtype=float)
        if len(eps) == len(rewards):
            ax_eps = ax_reward.twinx()
            ax_eps.set_ylabel("Epsilon")
            ax_eps.set_ylim(0.0, 1.05)
            (line_eps,) = ax_eps.plot(
                episodes,
                eps,
                color="orange",
                linewidth=1.5,
                alpha=0.9,
                label="Epsilon",
            )
            legend_handles.append(line_eps)
            legend_labels.append("Epsilon")

    ax_reward.legend(legend_handles, legend_labels, loc="upper left")

    if stats is not None:
        ax_reward.text(
            0.99,
            0.98,
            _stats_text(stats),
            transform=ax_reward.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="C0"),
        )

    fig.tight_layout()
    _save_figure(fig, title, "PointMaze Training Curve", output_subdir=output_subdir)
    if PLOT_SHOW:
        plt.show()
    plt.close(fig)


def plot_combined_training_curve(
    rewards_with_planner,
    rewards_without_planner,
    title: str = "",
    epsilon_with_planner=None,
    epsilon_without_planner=None,
    stats_with_planner: Optional[Dict[str, float]] = None,
    stats_without_planner: Optional[Dict[str, float]] = None,
    x_offset_without_planner: int = 0,
    reward_label_with_planner: str = "Reward (planned DQN)",
    reward_label_without_planner: str = "Reward (plain DQN)",
    epsilon_label_with_planner: str = "Epsilon (planned DQN)",
    epsilon_label_without_planner: str = "Epsilon (plain DQN)",
    stats_title_with_planner: str = "Planned DQN",
    stats_title_without_planner: str = "Plain DQN",
    output_subdir=None,
):
    rewards_with_planner = np.asarray(rewards_with_planner, dtype=float)
    rewards_without_planner = np.asarray(rewards_without_planner, dtype=float)

    fig, ax_reward = plt.subplots(figsize=(12, 5))
    legend_handles = []
    legend_labels = []

    episodes_wp = np.arange(len(rewards_with_planner))
    episodes_np = np.arange(x_offset_without_planner, x_offset_without_planner + len(rewards_without_planner))

    if len(rewards_with_planner) > 0:
        smoothed_wp = _smoothed_rewards(rewards_with_planner)
        (line_wp,) = ax_reward.plot(
            episodes_wp, smoothed_wp, color="C0", linewidth=2, label=reward_label_with_planner
        )
        legend_handles.append(line_wp)
        legend_labels.append(reward_label_with_planner)

    if len(rewards_without_planner) > 0:
        smoothed_np = _smoothed_rewards(rewards_without_planner)
        (line_np,) = ax_reward.plot(
            episodes_np, smoothed_np, color="C3", linewidth=2, label=reward_label_without_planner
        )
        legend_handles.append(line_np)
        legend_labels.append(reward_label_without_planner)

    ax_reward.set_title(title)
    ax_reward.set_xlabel("Episodes")
    ax_reward.set_ylabel("Reward")

    if epsilon_with_planner is not None or epsilon_without_planner is not None:
        ax_eps = ax_reward.twinx()
        ax_eps.set_ylabel("Epsilon")
        ax_eps.set_ylim(0.0, 1.05)

        if epsilon_with_planner is not None:
            eps_wp = np.asarray(epsilon_with_planner, dtype=float)
            if len(eps_wp) == len(rewards_with_planner):
                (line_eps_wp,) = ax_eps.plot(
                    episodes_wp,
                    eps_wp,
                    color="C1",
                    linewidth=1.3,
                    alpha=0.9,
                    label=epsilon_label_with_planner,
                )
                legend_handles.append(line_eps_wp)
                legend_labels.append(epsilon_label_with_planner)

        if epsilon_without_planner is not None:
            eps_np = np.asarray(epsilon_without_planner, dtype=float)
            if len(eps_np) == len(rewards_without_planner):
                (line_eps_np,) = ax_eps.plot(
                    episodes_np,
                    eps_np,
                    color="gold",
                    linewidth=1.3,
                    alpha=0.9,
                    label=epsilon_label_without_planner,
                )
                legend_handles.append(line_eps_np)
                legend_labels.append(epsilon_label_without_planner)

    if legend_handles:
        ax_reward.legend(legend_handles, legend_labels, loc="upper left")

    if stats_with_planner is not None:
        ax_reward.text(
            0.99,
            0.98,
            f"{stats_title_with_planner}\n" + _stats_text(stats_with_planner),
            transform=ax_reward.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="C0"),
        )

    if stats_without_planner is not None:
        ax_reward.text(
            0.99,
            0.80,
            f"{stats_title_without_planner}\n" + _stats_text(stats_without_planner),
            transform=ax_reward.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="C3"),
        )

    fig.tight_layout()
    _save_figure(fig, title, "PointMaze Combined Curve", output_subdir=output_subdir)
    if PLOT_SHOW:
        plt.show()
    plt.close(fig)


def plot_test_distribution(results, title="", output_subdir=None):
    results = np.asarray(results, dtype=float)
    mean = results.mean()
    std = results.std()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(results, bins=TEST_HIST_BINS, edgecolor="black", alpha=0.8)
    axes[0].axvline(mean, color="C1", linewidth=1)
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("Reward")
    axes[0].set_ylabel("Count")

    axes[1].boxplot(results, vert=False)
    axes[1].set_title("Box Plot")
    axes[1].set_xlabel("Reward")
    axes[1].text(
        0.98,
        0.95,
        f"mean={mean:.2f}\nstd={std:.2f}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle(title)
    fig.tight_layout()
    _save_figure(fig, title, "PointMaze Test Distribution", output_subdir=output_subdir)
    if PLOT_SHOW:
        plt.show()
    plt.close(fig)

