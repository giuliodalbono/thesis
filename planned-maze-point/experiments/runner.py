"""Algorithm-neutral episode collection shared by comparisons and sweeps.

learn=False never inserts experience or updates weights. The adapter owns
observation/action encoding and terminal semantics. DQN retains its existing
epsilon/recovery rules; SAC tests its deterministic actor without recovery.
Planner warm-up bypasses BOTH epsilon and SAC's random-data collection phase.
"""
from __future__ import annotations
import random
import time
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from agents.interface import ExperimentAgent
from configurations import PointMazeConfig
from environments.common import transition_success
Episode = Dict[str, Union[float, bool, int]]

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_episode(
    env,
    agent: ExperimentAgent,
    reset_seed: int,
    planner: Optional[object],
    learn: bool,
    config: PointMazeConfig,
    epsilon: float = 0.0,
    planner_probability: float = 1.0,
    planner_action_noise_std: float = 0.0,
) -> Episode:
    if not 0.0 <= planner_probability <= 1.0:
        raise ValueError("planner_probability must be in [0, 1]")
    if planner_action_noise_std < 0.0:
        raise ValueError("planner_action_noise_std must be nonnegative")
    started = time.perf_counter()
    planner_seconds = 0.0
    observation, _ = env.reset(seed=reset_seed)
    agent.reset_episode()
    reward_sum, steps, success = 0.0, 0, False
    planner_actions = agent_actions = 0
    recovery_events_start, recovery_actions_start = agent.recovery_events, agent.recovery_actions
    terminated = truncated = False
    while not (terminated or truncated):
        use_planner = (
            planner is not None and
            (planner_probability >= 1.0 or
             (planner_probability > 0.0 and agent.use_planner(planner_probability)))
        )
        if use_planner:
            planner_started = time.perf_counter()
            action = agent.planner_action(
                planner.action(observation), noise_std=planner_action_noise_std
            )
            planner_seconds += time.perf_counter() - planner_started
            planner_actions += 1
        else:
            action = agent.select_action(observation, training=learn, epsilon=epsilon)
            agent_actions += 1
        next_observation, reward, terminated, truncated, info = env.step(action)
        if learn:
            agent.observe(observation, action, reward, next_observation,
                          terminated, truncated, info, config,
                          is_planner_action=use_planner)
        reward_sum += float(reward)
        steps += 1
        success = success or transition_success(float(reward), info)
        observation = next_observation
    result = {
        "reward": reward_sum,
        "success": success,
        "steps": steps,
        "recovery_events": agent.recovery_events - recovery_events_start,
        "recovery_actions": agent.recovery_actions - recovery_actions_start,
    }
    if config.agent_name == "sac_her":
        result.update(agent.diagnostics(), elapsed_seconds=time.perf_counter() - started,
                      planner_seconds=planner_seconds, planner_actions=planner_actions,
                      agent_actions=agent_actions,
                      planner_probability=planner_probability if planner is not None else 0.0,
                      planner_action_noise_std=(planner_action_noise_std
                                                if planner is not None else 0.0))
    return result


def summarize(history: List[Episode]) -> Dict[str, Union[float, int]]:
    return {
        "episodes": len(history),
        "environment_steps": sum(int(x["steps"]) for x in history),
        "success_rate": float(np.mean([x["success"] for x in history])) if history else 0.0,
        "mean_reward": float(np.mean([x["reward"] for x in history])) if history else 0.0,
        "recovery_events": sum(int(x["recovery_events"]) for x in history),
        "recovery_actions": sum(int(x["recovery_actions"]) for x in history),
    }
