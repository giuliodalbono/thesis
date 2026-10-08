from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

GridCoord = Tuple[int, int]


def _clip_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _safe_observation_array(observation: Sequence[float]) -> np.ndarray:
    arr = np.asarray(observation, dtype=np.float32).reshape(-1)
    if arr.size < 6:
        raise ValueError(
            "PointMaze flattened observation is unexpectedly short. "
            f"Expected >= 6 values, got {arr.size}."
        )
    return arr


def _extract_position_and_goal_xy(observation: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Observation comes from GoalObservationFlattenWrapper as:
    [observation..., achieved_goal..., desired_goal...]

    For PointMaze, we use first 2 dims for current xy and last 2 dims for desired xy.
    """
    arr = _safe_observation_array(observation)
    current_xy = arr[:2].astype(np.float32)
    desired_xy = arr[-2:].astype(np.float32)
    return current_xy, desired_xy


@dataclass(frozen=True)
class PlannerConfig:
    grid_size: int = 21
    world_min: float = -6.0
    world_max: float = 6.0

    def __post_init__(self) -> None:
        if self.grid_size < 3:
            raise ValueError(f"grid_size must be >= 3, got {self.grid_size}")
        if self.world_min >= self.world_max:
            raise ValueError(
                f"world_min must be < world_max, got {self.world_min} >= {self.world_max}"
            )


class DeterministicPointMazePlanner:
    """
    Determinized planner over a fixed discrete XY grid.

    - State space: grid_size x grid_size discrete XY cells.
    - Transitions: deterministic, induced by action-table vector signs.
    - Objective: shortest path to discretized goal cell.
    """

    def __init__(self, action_table: Sequence[Sequence[float]], config: PlannerConfig):
        if len(action_table) == 0:
            raise ValueError("action_table cannot be empty for planner.")
        self.config = config
        self._action_deltas = self._build_action_deltas(action_table)
        self._cached_policies: Dict[GridCoord, Dict[GridCoord, int]] = {}
        self.calls = 0
        self.cache_hits = 0
        self.misses = 0

    @property
    def grid_size(self) -> int:
        return self.config.grid_size

    @staticmethod
    def _build_action_deltas(action_table: Sequence[Sequence[float]]) -> List[GridCoord]:
        deltas: List[GridCoord] = []
        for vec in action_table:
            arr = np.asarray(vec, dtype=np.float32).reshape(2)
            dx = int(np.sign(float(arr[0])))
            dy = int(np.sign(float(arr[1])))
            deltas.append((dx, dy))
        return deltas

    def _continuous_to_grid(self, xy: np.ndarray) -> GridCoord:
        x = float(xy[0])
        y = float(xy[1])
        span = self.config.world_max - self.config.world_min
        eps = 1e-8
        x_norm = (x - self.config.world_min) / (span + eps)
        y_norm = (y - self.config.world_min) / (span + eps)
        x_idx = int(round(x_norm * (self.grid_size - 1)))
        y_idx = int(round(y_norm * (self.grid_size - 1)))
        return (
            _clip_int(x_idx, 0, self.grid_size - 1),
            _clip_int(y_idx, 0, self.grid_size - 1),
        )

    def _next_cell(self, cell: GridCoord, action_idx: int) -> GridCoord:
        x, y = cell
        dx, dy = self._action_deltas[action_idx]
        return (
            _clip_int(x + dx, 0, self.grid_size - 1),
            _clip_int(y + dy, 0, self.grid_size - 1),
        )

    def _build_policy_to_goal(self, goal_cell: GridCoord) -> Dict[GridCoord, int]:
        if goal_cell in self._cached_policies:
            return self._cached_policies[goal_cell]

        queue: deque[GridCoord] = deque([goal_cell])
        visited = {goal_cell}
        reverse_policy: Dict[GridCoord, int] = {}

        while queue:
            cell = queue.popleft()
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    prev = (x, y)
                    if prev in visited:
                        continue
                    for action_idx in range(len(self._action_deltas)):
                        if self._next_cell(prev, action_idx) == cell:
                            visited.add(prev)
                            reverse_policy[prev] = action_idx
                            queue.append(prev)
                            break

        self._cached_policies[goal_cell] = reverse_policy
        return reverse_policy

    def first_action(self, observation: Sequence[float]) -> Optional[int]:
        self.calls += 1
        current_xy, goal_xy = _extract_position_and_goal_xy(observation)
        start = self._continuous_to_grid(current_xy)
        goal = self._continuous_to_grid(goal_xy)
        if goal in self._cached_policies:
            self.cache_hits += 1

        if start == goal:
            return 0

        policy = self._build_policy_to_goal(goal)
        action = policy.get(start)
        if action is None:
            self.misses += 1
        return action

    def clear_cache(self) -> None:
        self._cached_policies.clear()


def action_from_planner(
    planner: DeterministicPointMazePlanner, observation: Sequence[float]
) -> Optional[int]:
    return planner.first_action(observation)
