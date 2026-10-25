"""Repeated-seed SAC+HER ablation for planner behaviour cloning.

Each seed produces a BC comparison and optionally a planner-only control.
The default retains both conditions for existing Medium callers. In each comparison the
plain arm is stopped at the planned arm's real environment-interaction budget.
The aggregate JSON contains only concise, seed-level outcome statistics.
"""
from dataclasses import replace
from math import sqrt, isfinite
from pathlib import Path

from reporting.artifacts import new_run_directory, write_json
from reporting.comparison import save_planned_vs_plain_result
from .sac_comparison import run_planned_vs_plain_sac


def _summary(values):
    """Return JSON-safe descriptive statistics without implying significance."""
    mean = sum(values) / len(values)
    sample_std = (
        sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        if len(values) > 1 else 0.0
    )
    return {"n": len(values), "mean": mean, "sample_std": sample_std,
            "min": min(values), "max": max(values)}


def _outcome(result):
    """Select the held-out tests and actual pre-test training interactions."""
    return {
        "planned_post_warmup_success": result["planned"]["post_warmup_test_summary"]["success_rate"],
        "planned_final_success": result["planned"]["test_summary"]["success_rate"],
        "plain_final_success": result["plain"]["test_summary"]["success_rate"],
        "planned_pre_test_steps": result["planned"]["pre_test_environment_steps"],
        "plain_pre_test_steps": result["plain"]["pre_test_environment_steps"],
        "planned_training_episodes": result["planned"]["training_summary"]["episodes"],
        "plain_training_episodes": result["plain"]["training_summary"]["episodes"],
    }


def run_sac_bc_multiseed(config, seeds=(42, 43, 44, 45, 46),
                                 planner_warmup_episodes=400,
                                 planned_train_episodes=50,
                                 test_episodes=200,
                                 behavior_cloning_final_weight=0.4, *,
                                 behavior_cloning_initial_weight=1.0,
                                 include_planner_only_control=True):
    """Evaluate a BC candidate and optional planner-only control across seeds.

    Both conditions use planner actions for warm-up.  Only the BC condition
    adds planner-demonstration loss to the SAC actor.  The plain control uses
    no planner and receives an interaction-matched environment-step budget.
    """
    config.validate()
    if config.agent_name != "sac_her":
        raise ValueError("Use a SAC+HER configuration")
    if not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("seeds must be a non-empty sequence of integers")
    if not isinstance(include_planner_only_control, bool):
        raise ValueError("include_planner_only_control must be a boolean")
    if not isfinite(behavior_cloning_initial_weight) or behavior_cloning_initial_weight <= 0.0:
        raise ValueError("Initial BC weight must be finite and positive")
    if not isfinite(behavior_cloning_final_weight) or behavior_cloning_final_weight < 0.0:
        raise ValueError("Final BC weight must be finite and nonnegative")

    conditions = [
        ("planner_behavior_cloning", {
            "planner_guidance_mode": "planner_behavior_cloning",
            "behavior_cloning_initial_weight": behavior_cloning_initial_weight,
            "behavior_cloning_final_weight": behavior_cloning_final_weight,
        }),
    ]
    if include_planner_only_control:
        conditions.append(("planner_only", {
            "planner_guidance_mode": "planner_only",
            "behavior_cloning_initial_weight": 0.0,
            "behavior_cloning_final_weight": 0.0,
        }))
    records = []
    for seed in seeds:
        seeded_config = replace(config, seed=seed)
        for condition_name, guidance in conditions:
            print("\n=== seed {} | {} ===".format(seed, condition_name), flush=True)
            result = run_planned_vs_plain_sac(
                seeded_config,
                planner_warmup_episodes=planner_warmup_episodes,
                planned_train_episodes=planned_train_episodes,
                plain_train_episodes=0,
                test_episodes=test_episodes,
                evaluation_interval_episodes=0,
                match_plain_interaction_budget=True,
                **guidance,
            )
            report_path = save_planned_vs_plain_result(seeded_config, result)
            record = {"seed": seed, "condition": condition_name,
                      "report": str(report_path), **_outcome(result)}
            records.append(record)
            print("{} | planned {:.1%}; plain {:.1%}; steps {}/{}".format(
                condition_name, record["planned_final_success"],
                record["plain_final_success"], record["planned_pre_test_steps"],
                record["plain_pre_test_steps"]), flush=True)

    aggregate = {
        "schema_version": 1,
        "config": config.to_dict(),
        "protocol": {
            "environment": config.env_id,
            "agent": "SAC+HER",
            "seeds": list(seeds),
            "planner_warmup_episodes": planner_warmup_episodes,
            "planned_train_episodes": planned_train_episodes,
            "test_episodes": test_episodes,
            "behavior_cloning_weight": {"initial": behavior_cloning_initial_weight,
                                         "final": behavior_cloning_final_weight},
            "include_planner_only_control": include_planner_only_control,
            "bc_implementation_version": 2,
            "actor_gradient_reset": "per_update",
            "plain_budget": "matched to each planned condition's pre-test environment steps",
            "deterministic_evaluation": True,
            "evaluation_learns": False,
        },
        "runs": records,
        "conditions": {},
    }
    for condition_name, _ in conditions:
        values = [record for record in records if record["condition"] == condition_name]
        planned = [record["planned_final_success"] for record in values]
        plain = [record["plain_final_success"] for record in values]
        aggregate["conditions"][condition_name] = {
            "planned_post_warmup_success": _summary(
                [record["planned_post_warmup_success"] for record in values]),
            "planned_final_success": _summary(planned),
            "plain_final_success": _summary(plain),
            "paired_planned_minus_plain": _summary(
                [planned_value - plain_value for planned_value, plain_value in zip(planned, plain)]
            ),
        }

    directory = new_run_directory(config, "bc_multiseed")
    aggregate_path = write_json(directory / "aggregate.json", aggregate)
    print("\nSaved multi-seed aggregate: {}".format(aggregate_path), flush=True)
    return aggregate, Path(aggregate_path)


def run_medium_sac_bc_multiseed(config, *args, **kwargs):
    """Compatibility wrapper for existing Medium callers."""
    if config.env_id != "PointMaze_Medium-v3":
        raise ValueError("Use a Medium SAC configuration")
    return run_sac_bc_multiseed(config, *args, **kwargs)
