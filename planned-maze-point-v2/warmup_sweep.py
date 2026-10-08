from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from environment import flatten_observation, make_env
from experiment import _episode, _new_agent, _seed, _summary, save_trial
from planner import planner_for_env
from point_maze_medium_config import PointMazeMediumConfig
from point_maze_open_config import PointMazeOpenConfig


PointMazeConfig = Union[PointMazeOpenConfig, PointMazeMediumConfig]


def _evaluate_without_learning(
    env,
    agent,
    config: PointMazeConfig,
) -> List[Dict[str, object]]:
    """Evaluate greedily without changing replay, network, updates, or RNG state."""
    replay_steps_before = agent.stored_environment_steps
    updates_before = agent.updates
    agent_rng_state = agent.rng.getstate()
    try:
        history = [
            _episode(
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
    finally:
        # Evaluation recovery actions can consume the agent's private RNG. Put
        # it back so the subsequent training trajectory is unaffected.
        agent.rng.setstate(agent_rng_state)
    if agent.stored_environment_steps != replay_steps_before or agent.updates != updates_before:
        raise RuntimeError("Greedy evaluation unexpectedly changed DQN learning state")
    return history


def _save_sweep_report(config: PointMazeConfig, trials: List[Dict[str, object]]) -> Path:
    ranked = sorted(
        (
            {
                "warmup_episodes": trial["warmup_episodes"],
                "new_warmup_episodes": trial["new_warmup_episodes"],
                "evaluation_success_rate": trial["evaluation_summary"]["success_rate"],
                "evaluation_mean_reward": trial["evaluation_summary"]["mean_reward"],
                "evaluation_environment_steps": trial["evaluation_summary"]["environment_steps"],
                "cumulative_warmup_environment_steps": trial["warmup_summary"]["environment_steps"],
                "cumulative_warmup_time_seconds": trial["warmup_time_seconds"],
                "dqn_optimizer_updates": trial["dqn_optimizer_updates"],
            }
            for trial in trials
        ),
        key=lambda row: (-row["evaluation_success_rate"], row["warmup_episodes"]),
    )
    report = {
        "purpose": (
            "Cumulative planner-only warm-up sweep. One DQN receives planner "
            "transitions continuously; greedy evaluations pause but never alter learning."
        ),
        "environment": config.env_id,
        "seed": config.seed,
        "evaluation_episodes_per_checkpoint": config.evaluation_episodes,
        "protocol": {
            "single_network": True,
            "warmup_candidates_are_cumulative": True,
            "evaluation_epsilon": 0.0,
            "evaluation_updates_network": False,
            "evaluation_enters_replay": False,
            "evaluation_preserves_agent_rng": True,
        },
        "candidates_ranked_by_evaluation_success": ranked,
        "best_observed_warmup_episodes": ranked[0]["warmup_episodes"],
        "note": (
            "This is a one-seed cumulative learning curve, not independent trials. "
            "Confirm the selected warm-up count on separate seeds."
        ),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / (
        f"{config.environment_slug()}_dqn_planner_warmup_sweep_seed_{config.seed}.json"
    )
    path.write_text(json.dumps(report, indent=2))
    return path


def run_planner_only_warmup_sweep(
    config: Optional[PointMazeConfig] = None,
) -> Tuple[Dict[str, object], Path]:
    """Measure cumulative planner warm-up checkpoints using one persistent DQN.

    For candidates ``(250, 500, 1000)``, the agent receives 250 planner
    episodes, is greedily evaluated, receives 250 further planner episodes,
    is evaluated again, and finally receives 500 additional planner episodes.
    """
    config = config or PointMazeOpenConfig()
    config.validate()
    _seed(config.seed)
    env = make_env(config.env_id, config.max_episode_steps)
    observation, _ = env.reset(seed=config.seed)
    agent = _new_agent(config, flatten_observation(observation).size, env.action_space.n)
    planner = planner_for_env(config.env_id)

    trials: List[Dict[str, object]] = []
    cumulative_warmup: List[Dict[str, object]] = []
    cumulative_warmup_time = 0.0
    previous_target = 0
    try:
        for target_warmup_episodes in config.warmup_episode_counts:
            new_warmup_episodes = target_warmup_episodes - previous_target
            print(
                "\n=== Planner-only warm-up checkpoint: {} total episodes "
                "(+{}) ===".format(target_warmup_episodes, new_warmup_episodes)
            )
            started = time.perf_counter()
            segment = [
                _episode(
                    env,
                    agent,
                    config.seed + episode_index,
                    planner,
                    True,
                    config,
                )
                for episode_index in range(previous_target, target_warmup_episodes)
            ]
            cumulative_warmup_time += time.perf_counter() - started
            cumulative_warmup.extend(segment)

            evaluation = _evaluate_without_learning(env, agent, config)
            trial = {
                "environment": config.env_id,
                "warmup_episodes": target_warmup_episodes,
                "new_warmup_episodes": new_warmup_episodes,
                "warmup_segment": segment,
                "warmup_segment_summary": _summary(segment),
                "warmup_summary": _summary(cumulative_warmup),
                "warmup_time_seconds": cumulative_warmup_time,
                "evaluation": evaluation,
                "evaluation_summary": _summary(evaluation),
                "dqn_optimizer_updates": agent.updates,
                "evaluation_protocol": {
                    "epsilon": 0.0,
                    "learn": False,
                    "enters_replay": False,
                    "preserves_agent_rng": True,
                },
            }
            trials.append(trial)
            print(
                "Cumulative warm-up success: {:.3f} | Greedy DQN test success: {:.3f}".format(
                    trial["warmup_summary"]["success_rate"],
                    trial["evaluation_summary"]["success_rate"],
                )
            )
            print("Saved checkpoint: {}".format(save_trial(config, trial)))
            previous_target = target_warmup_episodes
    finally:
        env.close()

    report_path = _save_sweep_report(config, trials)
    best = max(
        trials,
        key=lambda trial: (
            trial["evaluation_summary"]["success_rate"],
            -trial["warmup_episodes"],
        ),
    )
    print(
        "\nBest observed warm-up: {} episodes (test success {:.3f})".format(
            best["warmup_episodes"], best["evaluation_summary"]["success_rate"]
        )
    )
    print("Saved sweep report: {}".format(report_path))
    return {"trials": trials, "best_trial": best}, report_path


def run_medium_planner_only_warmup_sweep(
    config: Optional[PointMazeMediumConfig] = None,
) -> Tuple[Dict[str, object], Path]:
    """Run the cumulative planner-only sweep with Medium-maze defaults."""
    return run_planner_only_warmup_sweep(config or PointMazeMediumConfig())
