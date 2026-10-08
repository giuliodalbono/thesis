from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from point_maze_medium_config import PointMazeMediumConfig
from point_maze_open_config import PointMazeOpenConfig
from dqn import DQNAgent, DQNHyperparameters, RecoveryConfig
from environment import ACTION_TABLE, flatten_observation, make_env, transition_success
from planner import planner_for_env


Episode = Dict[str, Union[float, bool, int]]
PointMazeConfig = Union[PointMazeOpenConfig, PointMazeMediumConfig]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _new_agent(config: PointMazeConfig, state_size: int, action_size: int) -> DQNAgent:
    if action_size != len(ACTION_TABLE):
        raise ValueError("DQN action count does not match the PointMaze action table.")
    return DQNAgent(
        state_size,
        np.asarray(ACTION_TABLE),
        DQNHyperparameters(
            config.gamma, config.learning_rate, config.batch_size,
            config.replay_capacity, config.learning_starts_transitions,
            config.target_sync_updates,
        ),
        RecoveryConfig(
            config.stuck_detection_steps, config.opposite_recovery_steps,
            config.random_recovery_steps, config.minimum_directional_progress,
        ),
        config.seed,
    )


def _episode(
    env,
    agent: DQNAgent,
    reset_seed: int,
    planner: Optional[object],
    learn: bool,
    config: PointMazeConfig,
    epsilon: float = 0.0,
) -> Episode:
    observation, _ = env.reset(seed=reset_seed)
    agent.reset_episode()
    state = flatten_observation(observation)
    reward_sum, steps, success = 0.0, 0, False
    recovery_events_start, recovery_actions_start = agent.recovery_events, agent.recovery_actions
    terminated = truncated = False
    while not (terminated or truncated):
        action = planner.action(observation) if planner is not None else agent.act(state, epsilon=epsilon)
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = flatten_observation(next_observation)
        done = bool(terminated or truncated)
        if learn:
            agent.store((state, action, float(reward), next_state, done))
            agent.train_if_due(config.train_every_steps, config.updates_per_training_event)
        reward_sum += float(reward)
        steps += 1
        success = success or transition_success(float(reward), info)
        observation, state = next_observation, next_state
    return {
        "reward": reward_sum,
        "success": success,
        "steps": steps,
        "recovery_events": agent.recovery_events - recovery_events_start,
        "recovery_actions": agent.recovery_actions - recovery_actions_start,
    }


def _summary(history: List[Episode]) -> Dict[str, Union[float, int]]:
    return {
        "episodes": len(history),
        "environment_steps": sum(int(x["steps"]) for x in history),
        "success_rate": float(np.mean([x["success"] for x in history])) if history else 0.0,
        "mean_reward": float(np.mean([x["reward"] for x in history])) if history else 0.0,
        "recovery_events": sum(int(x["recovery_events"]) for x in history),
        "recovery_actions": sum(int(x["recovery_actions"]) for x in history),
    }


def run_warmup_trial(config: PointMazeConfig, warmup_episodes: int) -> Dict[str, object]:
    """Train a fresh DQN only from planner episodes, then test immediately."""
    config.validate()
    if warmup_episodes < 1:
        raise ValueError("warmup_episodes must be positive")
    _seed(config.seed)
    env = make_env(config.env_id, config.max_episode_steps)
    observation, _ = env.reset(seed=config.seed)
    agent = _new_agent(config, flatten_observation(observation).size, env.action_space.n)
    planner = planner_for_env(config.env_id)

    started = time.perf_counter()
    warmup_history = [
        _episode(env, agent, config.seed + index, planner, True, config)
        for index in range(warmup_episodes)
    ]
    warmup_time = time.perf_counter() - started
    evaluation_history = [
        _episode(env, agent, config.seed + 100_000 + index, None, False, config)
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
        "warmup_summary": _summary(warmup_history),
        "warmup_time_seconds": warmup_time,
        "evaluation": evaluation_history,
        "evaluation_summary": _summary(evaluation_history),
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
    _seed(config.seed)
    env = make_env(config.env_id, config.max_episode_steps)
    observation, _ = env.reset(seed=config.seed)
    agent = _new_agent(config, flatten_observation(observation).size, env.action_space.n)
    planner = planner_for_env(config.env_id)
    history = [
        _episode(env, agent, config.seed + index, planner, False, config)
        for index in range(episodes)
    ]
    env.close()
    return {
        "environment": config.env_id,
        "planner": type(planner).__name__,
        "episodes": episodes,
        "history": history,
        "summary": _summary(history),
    }


def save_planner_validation(config: PointMazeConfig, validation: Dict[str, object]) -> Path:
    """Save a reproducible physical-environment planner validation report."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / (
        f"{config.environment_slug()}_planner_validation_"
        f"{validation['episodes']}_episodes_seed_{config.seed}.json"
    )
    path.write_text(json.dumps(validation, indent=2))
    return path


def save_trial(config: PointMazeConfig, trial: Dict[str, object]) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    warmup_episodes = trial["warmup_episodes"]
    path = config.output_dir / (
        f"{config.environment_slug()}_dqn_planner_warmup_{warmup_episodes}_seed_{config.seed}.json"
    )
    path.write_text(json.dumps(trial, indent=2))
    return path
