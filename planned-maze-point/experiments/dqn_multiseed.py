"""Repeated-seed planned-versus-plain DQN reports for the PointMaze suite."""
from dataclasses import replace
from math import sqrt
from pathlib import Path

from reporting.artifacts import new_run_directory, write_json
from reporting.comparison import save_planned_vs_plain_result
from .comparison import run_planned_vs_plain_dqn


def _summary(values):
    """Descriptive statistics only; inferential testing belongs in analysis."""
    mean = sum(values) / len(values)
    sample_std = (
        sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        if len(values) > 1 else 0.0
    )
    return {"n": len(values), "mean": mean, "sample_std": sample_std,
            "min": min(values), "max": max(values)}


def _outcome(result):
    """Keep each seed-level aggregate row compact while individual JSON stays complete."""
    planned, plain = result["planned"], result["plain"]
    return {
        "planner_warmup_success": planned["warmup_summary"]["success_rate"],
        "planned_post_warmup_success": (
            planned["post_warmup_test_summary"]["success_rate"]
            if planned["post_warmup_test_summary"] else None
        ),
        "planned_final_success": planned["test_summary"]["success_rate"],
        "plain_final_success": plain["test_summary"]["success_rate"],
        "planned_pre_test_steps": planned["pre_test_environment_steps"],
        "plain_pre_test_steps": plain["pre_test_environment_steps"],
        "planned_optimizer_updates": planned["optimizer_updates"],
        "plain_optimizer_updates": plain["optimizer_updates"],
    }


def run_dqn_multiseed(config, seeds=(42, 43, 44, 45, 46),
                       planner_warmup_episodes=0,
                       planned_train_episodes=0,
                       plain_train_episodes=0,
                       test_episodes=5_000,
                       study_name="planned_vs_plain_multiseed"):
    """Repeat one unchanged DQN protocol across seeds and save an aggregate.

    Every seed has a full, independently plotted comparison report.  The
    aggregate links those reports and records final deterministic-test success,
    the post-warm-up diagnostic, real interactions, and optimiser updates.
    Episode budgets are deliberately retained as specified by the DQN protocol;
    actual interaction counts are reported rather than assumed equal.
    """
    if config.agent_name != "dqn":
        raise ValueError("run_dqn_multiseed requires a DQN configuration")
    if (not seeds or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)):
        raise ValueError("seeds must be a non-empty sequence of integers")
    counts = (planner_warmup_episodes, planned_train_episodes,
              plain_train_episodes, test_episodes)
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts):
        raise ValueError("Episode counts must be nonnegative integers")
    if test_episodes < 1:
        raise ValueError("test_episodes must be positive")

    records = []
    for seed in seeds:
        seeded_config = replace(config, seed=seed)
        print("\n=== {} DQN | seed {} ===".format(seeded_config.env_id, seed), flush=True)
        result = run_planned_vs_plain_dqn(
            seeded_config,
            planner_warmup_episodes=planner_warmup_episodes,
            planned_train_episodes=planned_train_episodes,
            plain_train_episodes=plain_train_episodes,
            test_episodes=test_episodes,
        )
        report_path = save_planned_vs_plain_result(seeded_config, result)
        record = {"seed": seed, "report": str(report_path), **_outcome(result)}
        records.append(record)
        print("final success | planned {:.2%}; plain {:.2%} | steps {}/{}".format(
            record["planned_final_success"], record["plain_final_success"],
            record["planned_pre_test_steps"], record["plain_pre_test_steps"]), flush=True)

    planned = [record["planned_final_success"] for record in records]
    plain = [record["plain_final_success"] for record in records]
    aggregate = {
        "schema_version": 1,
        "study_name": study_name,
        "config": config.to_dict(),
        "protocol": {
            "environment": config.env_id,
            "agent": "DQN",
            "seeds": list(seeds),
            "planner_warmup_episodes": planner_warmup_episodes,
            "planned_train_episodes": planned_train_episodes,
            "plain_train_episodes": plain_train_episodes,
            "test_episodes": test_episodes,
            "budget_unit": "episodes",
            "equal_interaction_budget": False,
            "deterministic_evaluation": True,
            "evaluation_learns": False,
        },
        "runs": records,
        "summary": {
            "planned_final_success": _summary(planned),
            "plain_final_success": _summary(plain),
            "paired_planned_minus_plain": _summary(
                [planned_value - plain_value for planned_value, plain_value in zip(planned, plain)]
            ),
        },
    }
    directory = new_run_directory(config, study_name)
    path = write_json(directory / "aggregate.json", aggregate)
    print("Saved DQN multi-seed aggregate: {}".format(path), flush=True)
    return aggregate, Path(path)
