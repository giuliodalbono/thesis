"""Shared episode execution and metrics used by both experiment services.

learn=True stores transitions and updates DQN; learn=False does neither.
Planner warm-up bypasses epsilon and uses only planner actions. Greedy testing
sets epsilon=0 but the configured recovery controller may still choose actions.
Time-limit handling is retained from the original implementation.
"""
from __future__ import annotations
import random
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from agents.dqn import DQNAgent
from configurations import PointMazeConfig
from environments.common import flatten_observation, transition_success
Episode = Dict[str, Union[float, bool, int]]

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(
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


def summarize(history: List[Episode]) -> Dict[str, Union[float, int]]:
    return {
        "episodes": len(history),
        "environment_steps": sum(int(x["steps"]) for x in history),
        "success_rate": float(np.mean([x["success"] for x in history])) if history else 0.0,
        "mean_reward": float(np.mean([x["reward"] for x in history])) if history else 0.0,
        "recovery_events": sum(int(x["recovery_events"]) for x in history),
        "recovery_actions": sum(int(x["recovery_actions"]) for x in history),
    }
