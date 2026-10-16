"""SAC reports distinguish aligned episodes from actual interaction budgets."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def build_sac_comparison_figure(result):
    from .comparison import trailing_mean
    figure = Figure(figsize=(13, 9))
    FigureCanvasAgg(figure)
    rewards = figure.add_axes([0.08, 0.43, 0.62, 0.46])
    alpha = figure.add_axes([0.08, 0.12, 0.62, 0.21], sharex=rewards)
    warmup = result["planned"]["warmup"]
    offset = len(warmup)
    guidance_mode = result["protocol"].get("warmup_guidance_mode", "planner_only")
    warmup_label = (
        "Planner warm-up reward" if guidance_mode == "planner_only"
        else ("Annealed planner-guidance reward"
              if guidance_mode == "linear_warmup_anneal"
              else ("Planner-directed noisy warm-up reward"
                    if guidance_mode == "planner_action_noise"
                    else "Planner warm-up + behaviour cloning reward"))
    )
    if warmup:
        rewards.plot(np.arange(1, offset + 1), trailing_mean([r["reward"] for r in warmup]),
                     color="#777777", label=warmup_label)
    for name, color in (("planned", "#0072B2"), ("plain", "#D55E00")):
        data = result[name]
        history = data["training"]
        label = name.capitalize() + " SAC+HER"
        x = np.arange(offset + 1, offset + len(history) + 1)
        if history:
            rewards.plot(x, trailing_mean([r["reward"] for r in history]),
                         color=color, label=label, marker="o" if len(history) == 1 else None)
        # Alpha updates during demonstrations too. It is not epsilon and need
        # not decrease monotonically or remain in [0, 1].
        alpha_history = warmup + history if name == "planned" else history
        alpha_x = np.arange(1, len(alpha_history) + 1) if name == "planned" else x
        if alpha_history:
            alpha.plot(alpha_x, [r["entropy_coefficient"] for r in alpha_history],
                       color=color, label=label + " alpha")
    for axes in (rewards, alpha):
        if offset:
            axes.axvline(offset + .5, color="#C62828", linestyle=":", label="Warm-up ends")
            axes.axvspan(.5, offset + .5, color="#777777", alpha=.07)
        axes.grid(alpha=.2)
        axes.set_xlim(.5, max(offset + len(result["planned"]["training"]),
                             offset + len(result["plain"]["training"]), 1) + .5)
    def rate(summary):
        return "{:.2%}".format(summary["success_rate"]) if summary else "N/A (phase not run)"
    box = ["DETERMINISTIC TEST SUCCESS", "",
           "Planned: after warm-up", rate(result["planned"]["post_warmup_test_summary"]),
           "Planned: after training", rate(result["planned"]["post_training_test_summary"]),
           "Plain: final test", rate(result["plain"]["test_summary"]), "",
           "{} test episodes / phase".format(result["test_episodes"]),
           "HER on for both; recovery off", "No learning during tests", "",
           "WARM-UP GUIDANCE", guidance_mode,
           "Planner actions: {:,} ({:.1%})".format(
               result["planned"]["warmup_summary"].get("planner_actions", 0),
               result["planned"]["warmup_summary"].get("planner_action_fraction", 0.0),
           ),
           "Initial noise std: {:.3f}".format(
               result["protocol"].get("planner_action_noise_initial_std", 0.0)
           ),
           "BC weight: {:.3f} → {:.3f}".format(
               result["protocol"].get("behavior_cloning_initial_weight", 0.0),
               result["protocol"].get("behavior_cloning_final_weight", 0.0),
           ),
           "",
           "REAL TRAINING STEPS", "Planned: {:,}".format(result["planned"]["pre_test_environment_steps"]),
           "Plain: {:,}".format(result["plain"]["pre_test_environment_steps"])]
    figure.text(.75, .87, "\n".join(box), va="top", fontsize=9, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=.8", facecolor="#F5F5F5", edgecolor="#CCCCCC"))
    rewards.set_ylabel("Episode reward — trailing mean (up to 200 episodes)")
    alpha.set_ylabel("Entropy coefficient alpha")
    alpha.set_xlabel("Episode (plain shifted by {} warm-up episodes; not equal step budgets)".format(offset))
    for axes in (rewards, alpha):
        if axes.get_legend_handles_labels()[0]:
            axes.legend(loc="best", fontsize=9)
    figure.suptitle("{} | planned vs plain SAC+HER | seed {}".format(
        result["config"]["env_id"], result["config"]["seed"]))
    return figure


def build_sac_interaction_figure(result):
    """Policy validation success at real interaction counts; no x-axis shift.

    Final held-out tests are separate crosses, not connected to the validation
    curve as if they used the same evaluation tasks.
    """
    figure = Figure(figsize=(10, 6))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    for name, color in (("planned", "#0072B2"), ("plain", "#D55E00")):
        data = result[name]
        points = data["evaluation_checkpoints"]
        axes.plot([p["environment_steps"] for p in points],
                  [p["summary"]["success_rate"] for p in points], "o-",
                  color=color, label=name.capitalize() + " validation")
        axes.scatter([data["pre_test_environment_steps"]], [data["test_summary"]["success_rate"]],
                     marker="x", s=90, color=color, label=name.capitalize() + " held-out final test")
    axes.set(xlabel="Real training interactions (planner warm-up included)",
             ylabel="Deterministic policy success rate", ylim=(-.03, 1.03),
             title="SAC+HER: actual interaction budgets; evaluation steps excluded")
    axes.grid(alpha=.2)
    axes.legend()
    figure.tight_layout()
    return figure
