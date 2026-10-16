"""Continuous PointMaze adapter shared by Open and Medium.

DQN discretizes the action space; SAC keeps native continuous actions. Both
observe continuous position, velocity and goals; only the planner uses a grid.
"""
from __future__ import annotations

from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

try:
    import gymnasium_robotics
except ImportError as exc:
    raise ImportError("Install gymnasium-robotics to run this experiment.") from exc


gym.register_envs(gymnasium_robotics)

# Four cardinal plus four normalized diagonal force vectors. There is no no-op:
# an agent must choose a movement direction at every environment step.
ACTION_TABLE: Tuple[np.ndarray, ...] = (
    np.array([1.0, 0.0], dtype=np.float32),
    np.array([-1.0, 0.0], dtype=np.float32),
    np.array([0.0, 1.0], dtype=np.float32),
    np.array([0.0, -1.0], dtype=np.float32),
    np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2),
    np.array([1.0, -1.0], dtype=np.float32) / np.sqrt(2),
    np.array([-1.0, 1.0], dtype=np.float32) / np.sqrt(2),
    np.array([-1.0, -1.0], dtype=np.float32) / np.sqrt(2),
)


def flatten_observation(observation: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(observation["observation"], dtype=np.float32).reshape(-1),
            np.asarray(observation["achieved_goal"], dtype=np.float32).reshape(-1),
            np.asarray(observation["desired_goal"], dtype=np.float32).reshape(-1),
        ]
    )


class DiscretePointMaze(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(ACTION_TABLE))

    def action(self, action: int) -> np.ndarray:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid discrete action: {action}")
        return ACTION_TABLE[int(action)].copy()


def make_env(env_id: str, max_episode_steps: int, continuous: bool = False):
    # Rendering is never enabled in training or evaluation timing.
    env = gym.make(env_id, continuing_task=False, max_episode_steps=max_episode_steps)
    return env if continuous else DiscretePointMaze(env)


def transition_success(reward: float, info: Dict[str, object]) -> bool:
    return bool(info.get("success", False) or reward > 0.0)
