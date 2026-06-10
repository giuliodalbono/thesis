"""
Plotting utilities for training curves, test reward distributions, and multi-run summary.
Figures saved in resources/graphs/; window size for smoothing scales with data length.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_SHOW = True
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "resources" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_WINDOW_DIVISOR = 50
TRAIN_WINDOW_MIN = 50
TRAIN_WINDOW_MAX = 1000
TEST_HIST_BINS = 20


def _safe_filename(title, fallback):
    """Sanitize title for use as filename (no special chars, single spaces)."""
    base = title.strip() if title else fallback
    base = re.sub(r"[^\w\s\-().]", "_", base)
    base = re.sub(r"\s+", " ", base).strip()
    return f"{base}.png"


def _save_current_plot(title, fallback):
    filename = _safe_filename(title, fallback)
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=150)


def _save_figure(fig, title, fallback):
    filename = _safe_filename(title, fallback)
    path = OUTPUT_DIR / filename
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


def _training_stats_text(stats, include_planner_timing=False):
    lines = [
        f"Episodi: {stats['episodes']}",
        f"Tempo training: {_format_duration(stats['training_time_s'])}",
    ]
    if include_planner_timing:
        lines.append(
            f"Tempo pianificazione: {_format_duration(stats['plan_time_s'])}"
        )
        lines.append(f"% su training: {stats['plan_time_pct']:.2f}%")
    return "\n".join(lines)


def _smoothed_rewards(rewards):
    rewards = np.asarray(rewards)
    window = int(
        np.clip(len(rewards) // TRAIN_WINDOW_DIVISOR, TRAIN_WINDOW_MIN, TRAIN_WINDOW_MAX)
    )
    return np.array(
        [np.mean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))]
    )


def plot_combined_training_curve(
    rewards_with_planner,
    rewards_without_planner,
    title="",
    epsilon_with_planner=None,
    epsilon_2_with_planner=None,
    epsilon_without_planner=None,
    stats_with_planner=None,
    stats_without_planner=None,
):
    """
    Single figure: smoothed rewards and epsilon traces for planner vs no-planner runs.
    Blue / orange / orange dashed = with planner; red / yellow = without planner.
    """
    rewards_with_planner = np.asarray(rewards_with_planner)
    rewards_without_planner = np.asarray(rewards_without_planner)

    fig, ax_reward = plt.subplots(figsize=(12, 5))
    legend_handles = []
    legend_labels = []

    episodes_wp = np.arange(len(rewards_with_planner))
    smoothed_wp = _smoothed_rewards(rewards_with_planner)
    (ln_wp,) = ax_reward.plot(
        episodes_wp,
        smoothed_wp,
        color="blue",
        linewidth=2,
        label="Reward (with planner)",
    )
    legend_handles.append(ln_wp)
    legend_labels.append("Reward (with planner)")

    episodes_np = np.arange(len(rewards_without_planner))
    smoothed_np = _smoothed_rewards(rewards_without_planner)
    (ln_np,) = ax_reward.plot(
        episodes_np,
        smoothed_np,
        color="red",
        linewidth=2,
        label="Reward (no planner)",
    )
    legend_handles.append(ln_np)
    legend_labels.append("Reward (no planner)")

    ax_reward.set_title(title)
    ax_reward.set_xlabel("Episodes")
    ax_reward.set_ylabel("Reward")

    ax_eps = ax_reward.twinx()
    ax_eps.set_ylabel("Epsilon")
    ax_eps.set_ylim(0.0, 1.05)

    if epsilon_with_planner is not None:
        eps = np.asarray(epsilon_with_planner, dtype=float)
        if len(eps) == len(rewards_with_planner):
            (ln_eps,) = ax_eps.plot(
                episodes_wp,
                eps,
                color="orange",
                linewidth=1.5,
                alpha=0.9,
                label="ε (with planner)",
            )
            legend_handles.append(ln_eps)
            legend_labels.append("ε (with planner)")

    if epsilon_2_with_planner is not None:
        eps2 = np.asarray(epsilon_2_with_planner, dtype=float)
        if len(eps2) == len(rewards_with_planner):
            (ln_eps2,) = ax_eps.plot(
                episodes_wp,
                eps2,
                color="orange",
                linewidth=1.5,
                alpha=0.9,
                linestyle="--",
                label="ε₂ (with planner)",
            )
            legend_handles.append(ln_eps2)
            legend_labels.append("ε₂ (with planner)")

    if epsilon_without_planner is not None:
        eps_np = np.asarray(epsilon_without_planner, dtype=float)
        if len(eps_np) == len(rewards_without_planner):
            (ln_eps_np,) = ax_eps.plot(
                episodes_np,
                eps_np,
                color="yellow",
                linewidth=1.5,
                alpha=0.9,
                label="ε (no planner)",
            )
            legend_handles.append(ln_eps_np)
            legend_labels.append("ε (no planner)")

    ax_reward.legend(legend_handles, legend_labels, loc="upper left")

    if stats_with_planner is not None:
        ax_reward.text(
            0.99,
            0.98,
            "DQN + planner\n" + _training_stats_text(stats_with_planner, include_planner_timing=True),
            transform=ax_reward.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="blue"),
        )

    if stats_without_planner is not None:
        ax_reward.text(
            0.99,
            0.52,
            "DQN base\n" + _training_stats_text(stats_without_planner, include_planner_timing=False),
            transform=ax_reward.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="red"),
        )

    fig.tight_layout()

    _save_figure(fig, title, "Training Curve Combined")
    if PLOT_SHOW:
        plt.show()

    plt.close(fig)


def plot_test_distribution(results, title=""):
    """Histogram + box plot of test episode returns; mean and std in box."""
    results = np.asarray(results)
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

    stats_text = f"mean={mean:.2f}\nstd={std:.2f}"
    axes[1].text(
        0.98,
        0.95,
        stats_text,
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle(title)
    fig.tight_layout()

    _save_figure(fig, title, "Test Distribution")
    if PLOT_SHOW:
        plt.show()

    plt.close(fig)
