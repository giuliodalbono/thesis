from __future__ import annotations

from typing import Iterable, List, Sequence

import gymnasium as gym
import numpy as np


def build_action_table(mode: str = "nine") -> List[np.ndarray]:
    """
    Build a fixed set of 2D force vectors for PointMaze.

    Modes:
    - "five": 4 cardinal directions.
    - "nine": "five" + 4 normalized diagonals.
    - "thirteen": "nine" + 4 half-strength cardinals.
    """
    mode = str(mode).strip().lower()

    action_table: List[np.ndarray] = [
        np.array([1.0, 0.0], dtype=np.float32),   # right
        np.array([-1.0, 0.0], dtype=np.float32),  # left
        np.array([0.0, 1.0], dtype=np.float32),   # up
        np.array([0.0, -1.0], dtype=np.float32),  # down
    ]

    if mode in {"nine", "thirteen"}:
        diag = float(1.0 / np.sqrt(2.0))
        action_table.extend(
            [
                np.array([diag, diag], dtype=np.float32),
                np.array([diag, -diag], dtype=np.float32),
                np.array([-diag, diag], dtype=np.float32),
                np.array([-diag, -diag], dtype=np.float32),
            ]
        )

    if mode == "thirteen":
        action_table.extend(
            [
                np.array([0.5, 0.0], dtype=np.float32),
                np.array([-0.5, 0.0], dtype=np.float32),
                np.array([0.0, 0.5], dtype=np.float32),
                np.array([0.0, -0.5], dtype=np.float32),
            ]
        )

    if mode not in {"five", "nine", "thirteen"}:
        raise ValueError(f"Unknown action-table mode {mode!r}. Expected one of: five, nine, thirteen.")

    return action_table


class GoalObservationFlattenWrapper(gym.ObservationWrapper):
    """
    Flatten PointMaze goal-conditioned dict observations into a single vector.

    Output order:
    [observation, achieved_goal, desired_goal]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError(
                "GoalObservationFlattenWrapper requires Dict observation space."
            )

        keys: Sequence[str] = ("observation", "achieved_goal", "desired_goal")
        missing = [key for key in keys if key not in env.observation_space.spaces]
        if missing:
            raise ValueError(f"Missing expected observation keys: {missing}")

        self._keys = keys
        total_dim = int(
            sum(np.prod(env.observation_space.spaces[key].shape) for key in keys)
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

    def observation(self, observation):
        pieces: Iterable[np.ndarray] = (
            np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in self._keys
        )
        return np.concatenate(list(pieces), axis=0)


class ContinuousToDiscreteActionWrapper(gym.ActionWrapper):
    """Expose a discrete action space over fixed continuous force vectors."""

    def __init__(self, env: gym.Env, action_table: Sequence[Sequence[float]]):
        super().__init__(env)

        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("ContinuousToDiscreteActionWrapper requires Box action space.")
        if env.action_space.shape != (2,):
            raise ValueError(
                f"Expected PointMaze action space shape (2,), got {env.action_space.shape!r}"
            )

        if not action_table:
            raise ValueError("action_table cannot be empty.")

        self.action_table = [
            np.asarray(vec, dtype=np.float32).reshape(2) for vec in action_table
        ]
        self.action_space = gym.spaces.Discrete(len(self.action_table))

    def action(self, action: int) -> np.ndarray:
        if isinstance(action, np.ndarray):
            action = int(action.item())
        if not 0 <= int(action) < len(self.action_table):
            raise ValueError(
                f"Discrete action index {action!r} out of range [0, {len(self.action_table) - 1}]."
            )

        force = self.action_table[int(action)].copy()
        force = np.clip(force, -1.0, 1.0)
        return force.astype(np.float32)

