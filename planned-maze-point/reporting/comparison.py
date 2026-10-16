"""Comparison figure: 200-episode reward averages, epsilon, and test summaries.

The plain curve is shifted right by the planner warm-up count. This aligns the
start of normal training, but is NOT a common environment-interaction budget.
Only plotting coordinates are shifted; actual episode/step counts stay in JSON.
"""
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from agents.exploration import training_epsilon
from configurations import config_from_dict
from .artifacts import new_run_directory, write_json


def trailing_mean(values, window=200):
    """Use all available samples for the first 199 episodes; no initial gap."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return values
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return (cumulative[end] - cumulative[start]) / (end - start)


def build_comparison_figure(result):
    if result["config"].get("agent_name") == "sac_her":
        from .sac import build_sac_comparison_figure
        return build_sac_comparison_figure(result)
    # Always reconstruct from the recorded configuration, not today's preset.
    config = config_from_dict(result["config"])
    planned, plain = result["planned"], result["plain"]
    warmup = planned["warmup"]
    offset = len(warmup)
    figure = Figure(figsize=(13, 7))
    FigureCanvasAgg(figure)
    axes = figure.add_axes([0.08, 0.17, 0.63, 0.70])
    epsilon_axes = axes.twinx()

    def reward_curve(history, start, color, label):
        if history:
            axes.plot(np.arange(start + 1, start + len(history) + 1),
                      trailing_mean([row["reward"] for row in history]),
                      color=color, linewidth=1.8, label=label,
                      marker="o" if len(history) == 1 else None)

    reward_curve(warmup, 0, "#777777", "Planner warm-up reward")
    # Smooth each phase separately: successful planner rewards must not leak
    # into the first 199 episodes of the network's normal-training curve.
    for data, color, label in ((planned, "#0072B2", "Planned DQN"),
                                (plain, "#D55E00", "Plain DQN")):
        history = data["training"]
        reward_curve(history, offset, color, label + " reward")
        if history:
            epsilon_axes.plot(
                np.arange(offset + 1, offset + len(history) + 1),
                [training_epsilon(config, i, len(history)) for i in range(len(history))],
                color=color, linestyle="--", linewidth=1.5,
                marker="o" if len(history) == 1 else None,
                label=label + " epsilon")
    if offset:
        axes.axvline(offset + 0.5, color="#C62828", linestyle=":",
                     linewidth=1.8, label="Warm-up ends")
        axes.axvspan(0.5, offset + 0.5, color="#777777", alpha=0.07)

    def percent(summary):
        return "{:.2%}".format(summary["success_rate"]) if summary else "N/A (phase not run)"

    post_warmup = planned.get("post_warmup_test_summary")
    # Historical zero-training runs store their only test under 'test_summary'.
    if not planned["training"]:
        post_warmup = planned["test_summary"]
    lines = [
        "GREEDY TEST SUCCESS",
        "",
        "Planned: after warm-up",
        percent(post_warmup),
        "",
        "Planned: after training",
        percent(planned["test_summary"] if planned["training"] else None),
        "",
        "Plain: final test",
        percent(plain["test_summary"]),
        "",
        "{} test episodes / phase".format(result["test_episodes"]),
        "Recovery per saved config",
    ]
    figure.text(0.79, 0.85, "\n".join(lines), ha="left", va="top",
                fontsize=10, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.8", facecolor="#F5F5F5", edgecolor="#CCCCCC"))
    if not warmup and not planned["training"] and not plain["training"]:
        axes.text(0.5, 0.5, "No warm-up or training episodes",
                  transform=axes.transAxes, ha="center")
    axes.set_xlabel("Episode (plain training shifted right by {} warm-up episodes)".format(offset))
    axes.set_ylabel("Episode reward — trailing mean (up to 200 episodes)")
    epsilon_axes.set_ylabel("Epsilon")
    epsilon_axes.set_ylim(-0.02, max(1.05, config.epsilon_start * 1.05))
    end = max(offset + len(planned["training"]), offset + len(plain["training"]), 1)
    axes.set_xlim(0.5, end + 0.5)
    axes.grid(alpha=0.2)
    handles, labels = axes.get_legend_handles_labels()
    extra_handles, extra_labels = epsilon_axes.get_legend_handles_labels()
    figure.legend(handles + extra_handles, labels + extra_labels,
                  loc="lower center", ncol=3, fontsize=9, frameon=False)
    figure.suptitle("{} | planned vs plain DQN | seed {}".format(config.env_id, config.seed))
    return figure


def save_planned_vs_plain_result(config, result):
    """Save detailed JSON and one combined PNG in a unique execution folder."""
    saved_config = config_from_dict(result["config"])
    # config supplies the destination only; labels/schedules use the recorded run.
    from dataclasses import replace
    saved_config = replace(saved_config, output_dir=config.output_dir)
    directory = new_run_directory(saved_config, "comparison")
    plot_path = directory / "comparison.png"
    figure = build_comparison_figure(result)
    figure.savefig(plot_path, dpi=160)
    result["artifacts"] = {"comparison_curve": str(plot_path)}
    if saved_config.agent_name == "sac_her":
        from .sac import build_sac_interaction_figure
        interaction_path = directory / "success_by_interactions.png"
        build_sac_interaction_figure(result).savefig(interaction_path, dpi=160)
        result["artifacts"]["success_by_interactions"] = str(interaction_path)
    result["plot_description"] = {
        "moving_average_episodes": 200,
        "plain_episode_offset": result["planner_warmup_episodes"],
        "alignment": "normal-training starts; not equal interaction budgets",
    }
    return write_json(directory / "result.json", result)


def replot_saved_comparison(path, output_dir=None):
    """Regenerate the new output from historical JSON, without rerunning training."""
    import json
    result = json.loads(Path(path).read_text())
    config = config_from_dict(result["config"])
    if output_dir is not None:
        from dataclasses import replace
        config = replace(config, output_dir=Path(output_dir))
    return save_planned_vs_plain_result(config, result)
