"""Standalone fresh warm-up trials and planner execution diagnostics."""
from __future__ import annotations
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict
from configurations import PointMazeConfig
from agents import create_agent
from environments.common import flatten_observation
from planners import planner_for_env
from .runner import seed_everything, run_episode, summarize
from environments.common import ACTION_TABLE, transition_success
from reporting.artifacts import new_run_directory, write_json

def run_warmup_trial(config: PointMazeConfig, warmup_episodes: int) -> Dict[str, object]:
    """Train a fresh DQN only from planner episodes, then test immediately."""
    config.validate()
    if warmup_episodes < 1:
        raise ValueError("warmup_episodes must be positive")
    seed_everything(config.seed)
    env = config.environment.create(config.max_episode_steps)
    observation, _ = env.reset(seed=config.seed)
    agent = create_agent(config, flatten_observation(observation).size, env.action_space.n)
    planner = planner_for_env(config.env_id)

    started = time.perf_counter()
    warmup_history = [
        run_episode(env, agent, config.seed + index, planner, True, config)
        for index in range(warmup_episodes)
    ]
    warmup_time = time.perf_counter() - started
    evaluation_history = [
        run_episode(env, agent, config.seed + 100_000 + index, None, False, config)
        for index in range(config.evaluation_episodes)
    ]
    env.close()

    return {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "warmup_episodes": warmup_episodes,
        "epsilon_schedule": {
            "type": "bounded_linear_per_training_budget",
            "decay_fraction": config.epsilon_decay_fraction,
            "applicable": False,
            "reason": (
                "Planner-only warm-up selects every action, and the subsequent "
                "evaluation is greedy with epsilon fixed at zero."
            ),
        },
        "warmup": warmup_history,
        "warmup_summary": summarize(warmup_history),
        "warmup_time_seconds": warmup_time,
        "evaluation": evaluation_history,
        "evaluation_summary": summarize(evaluation_history),
        "dqn_optimizer_updates": agent.updates,
    }


def run_planner_validation(
    config: PointMazeConfig,
    episodes: int = 100,
) -> Dict[str, object]:
    """Execute planner-only episodes without learning to validate a maze abstraction."""
    config.validate()
    if episodes < 1:
        raise ValueError("episodes must be positive")
    seed_everything(config.seed)
    # Validate the physical controller without constructing a neural agent.
    # Both action interfaces execute the same force vectors.
    continuous = config.agent_name == "sac_her"
    env = config.environment.create(config.max_episode_steps, continuous=continuous)
    planner = planner_for_env(config.env_id)
    history = []
    started = time.perf_counter()
    try:
        for index in range(episodes):
            observation, _ = env.reset(seed=config.seed + index)
            reward_sum, steps, success = 0.0, 0, False
            terminated = truncated = False
            while not (terminated or truncated):
                action_index = planner.action(observation)
                action = ACTION_TABLE[action_index].copy() if continuous else action_index
                observation, reward, terminated, truncated, info = env.step(action)
                steps += 1
                reward_sum += float(reward)
                success = success or transition_success(float(reward), info)
            history.append(dict(reward=reward_sum, steps=steps, success=success,
                                recovery_events=0, recovery_actions=0,
                                reset_seed=config.seed + index))
    finally:
        env.close()
    return {
        "config": config.to_dict(),
        "environment": config.env_id,
        "planner": type(planner).__name__,
        "episodes": episodes,
        "history": history,
        "summary": summarize(history),
        "elapsed_seconds": time.perf_counter() - started,
        "learning": False,
    }


def save_planner_validation(config: PointMazeConfig, validation: Dict[str, object]) -> Path:
    """Save a reproducible physical-environment planner validation report."""
    directory = new_run_directory(config, "planner_validation")
    return write_json(directory / "result.json", validation)


def save_trial(config: PointMazeConfig, trial: Dict[str, object]) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    warmup_episodes = trial["warmup_episodes"]
    path = config.output_dir / (
        f"{config.environment_slug()}_dqn_planner_warmup_{warmup_episodes}_seed_{config.seed}.json"
    )
    path.write_text(json.dumps(trial, indent=2))
    return path
