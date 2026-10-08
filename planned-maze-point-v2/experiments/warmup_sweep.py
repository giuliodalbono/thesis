"""Cumulative planner-only training with frozen-policy tests at chosen checkpoints.

One agent keeps its replay, optimizer, target network and update counters across
checkpoints. Tests never enter replay or perform updates. Each test uses the same
reset seeds; the agent RNG is restored before planner learning resumes.
Only compact configuration and phase summaries are written to disk.
"""
from __future__ import annotations
import time
from typing import Dict, List
from agents import create_agent
from agents.evaluation import preserve_training_state
from configurations import PointMazeConfig, PointMazeOpenConfig, PointMazeMediumConfig
from environments.common import flatten_observation
from planners import create_planner
from reporting.artifacts import new_run_directory, write_json
from .runner import seed_everything, run_episode, summarize

def _evaluate_without_learning(
    env,
    agent,
    config: PointMazeConfig,
) -> List[Dict[str, object]]:
    """Evaluate greedily without changing replay, network, updates, or RNG state."""
    replay_steps_before = agent.stored_environment_steps
    updates_before = agent.updates
    with preserve_training_state(agent):
        history = [
            run_episode(
                env,
                agent,
                config.seed + 100_000 + episode_index,
                planner=None,
                learn=False,
                config=config,
                epsilon=0.0,
            )
            for episode_index in range(config.evaluation_episodes)
        ]
    if agent.stored_environment_steps != replay_steps_before or agent.updates != updates_before:
        raise RuntimeError("Greedy evaluation unexpectedly changed DQN learning state")
    return history



def run_planner_only_warmup_sweep(config=None):
    """Return (aggregate_report, path); candidate counts are cumulative totals."""
    config = config or PointMazeOpenConfig()
    config.validate()
    directory = new_run_directory(config, "warmup_sweep")
    path = directory / "summary.json"
    report = {
        "configuration": config.to_dict(),
        "protocol": {
            "cumulative_single_agent": True,
            "evaluation_episodes_per_checkpoint": config.evaluation_episodes,
            "evaluation_epsilon": 0.0,
            "evaluation_learns": False,
            "recovery_can_activate_during_evaluation":
                config.stuck_detection_steps < config.max_episode_steps,
            "note": "Recovery follows the configured detection threshold.",
        },
        "checkpoints": [],
    }
    seed_everything(config.seed)
    env = config.environment.create(config.max_episode_steps)
    history, elapsed, previous = [], 0.0, 0
    try:
        observation, _ = env.reset(seed=config.seed)
        agent = create_agent(config, flatten_observation(observation).size, env.action_space.n)
        planner = create_planner(config)
        for target in config.warmup_episode_counts:
            print("\n=== {}: {} total warm-up episodes (+{}) ===".format(
                config.env_id, target, target - previous))
            started = time.perf_counter()
            for index in range(previous, target):
                history.append(run_episode(env, agent, config.seed + index,
                                           planner, True, config))
            elapsed += time.perf_counter() - started
            evaluation = _evaluate_without_learning(env, agent, config)
            report["checkpoints"].append({
                "warmup_episodes": target,
                "additional_warmup_episodes": target - previous,
                "warmup_summary": dict(summarize(history),
                                       elapsed_seconds=elapsed,
                                       optimizer_updates=agent.updates),
                "evaluation_summary": summarize(evaluation),
            })
            # Save after every completed test; interrupted runs retain completed summaries.
            best = max(report["checkpoints"], key=lambda row: (
                row["evaluation_summary"]["success_rate"], -row["warmup_episodes"]))
            report["best_observed_warmup_episodes"] = best["warmup_episodes"]
            write_json(path, report)
            print("Planner success: {:.3f} | evaluation success: {:.3f}".format(
                report["checkpoints"][-1]["warmup_summary"]["success_rate"],
                report["checkpoints"][-1]["evaluation_summary"]["success_rate"]))
            previous = target
    finally:
        env.close()
    print("Saved sweep summary: {}".format(path))
    return report, path


def run_medium_planner_only_warmup_sweep(config=None):
    return run_planner_only_warmup_sweep(config or PointMazeMediumConfig())
