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


def plot_training_curve(
    rewards,
    title="",
    epsilon_floor_episode=None,
    epsilon_per_episode=None,
    epsilon_2_per_episode=None,
):
    """
    Raw episode rewards + trailing mean (adaptive window ~2% of length).
    Optional vertical line at episode where epsilon hits minimum (exploration floor).
    Optional epsilon (and epsilon_2 when using planner exploration) on a right-hand axis.
    """
    rewards = np.asarray(rewards)
    window = int(
        np.clip(len(rewards) // TRAIN_WINDOW_DIVISOR, TRAIN_WINDOW_MIN, TRAIN_WINDOW_MAX)
    )
    fig, ax_reward = plt.subplots(figsize=(10, 5))
    episodes = np.arange(len(rewards))

    ax_reward.plot(episodes, rewards, alpha=0.3, label="Raw", color="C0")

    smoothed = np.array(
        [np.mean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))]
    )
    ax_reward.plot(episodes, smoothed, label=f"Mean (w={window})", linewidth=2, color="C0")

    ax_reward.set_title(title)
    ax_reward.set_xlabel("Episodes")
    ax_reward.set_ylabel("Reward")

    legend_handles = []
    legend_labels = []

    if epsilon_per_episode is not None:
        eps = np.asarray(epsilon_per_episode, dtype=float)
        if len(eps) == len(rewards):
            ax_eps = ax_reward.twinx()
            (ln_eps,) = ax_eps.plot(
                episodes, eps, color="tab:orange", linewidth=1.5, alpha=0.9, label="ε"
            )
            ax_eps.set_ylabel("Epsilon")
            ax_eps.set_ylim(0.0, 1.05)
            ax_eps.tick_params(axis="y")
            legend_handles.append(ln_eps)
            legend_labels.append("ε")

            if epsilon_2_per_episode is not None:
                eps2 = np.asarray(epsilon_2_per_episode, dtype=float)
                if len(eps2) == len(rewards):
                    (ln_eps2,) = ax_eps.plot(
                        episodes,
                        eps2,
                        color="tab:green",
                        linewidth=1.5,
                        alpha=0.85,
                        linestyle="--",
                        label="ε₂ (planner mix)",
                    )
                    legend_handles.append(ln_eps2)
                    legend_labels.append("ε₂ (planner mix)")

    if epsilon_floor_episode is not None:
        ln_floor = ax_reward.axvline(
            epsilon_floor_episode,
            color="purple",
            linestyle=":",
            linewidth=1,
            label="Epsilon floor",
        )
        legend_handles.append(ln_floor)
        legend_labels.append("Epsilon floor")

    h_r, lab_r = ax_reward.get_legend_handles_labels()
    if legend_handles:
        ax_reward.legend(h_r + legend_handles, lab_r + legend_labels, loc="upper left")
    else:
        ax_reward.legend(h_r, lab_r, loc="best")

    fig.tight_layout()

    _save_figure(fig, title, "Training Curve")
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
