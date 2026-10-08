from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from environment import ACTION_TABLE


Cell = Tuple[int, int]


@dataclass(frozen=True)
class PointMazeOpenGrid:
    """Reachable cell centres in Gymnasium Robotics PointMaze_Open-v3."""

    x_centres: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    y_centres: Tuple[float, ...] = (-1.0, 0.0, 1.0)

    def nearest_cell(self, xy: np.ndarray) -> Cell:
        return (
            int(np.argmin(np.abs(np.asarray(self.x_centres) - float(xy[0])))),
            int(np.argmin(np.abs(np.asarray(self.y_centres) - float(xy[1])))),
        )

    def cell_centre(self, cell: Cell) -> np.ndarray:
        return np.array([self.x_centres[cell[0]], self.y_centres[cell[1]]], dtype=np.float32)

    def neighbours(self, cell: Cell) -> Iterable[Cell]:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            candidate = (cell[0] + dx, cell[1] + dy)
            if 0 <= candidate[0] < len(self.x_centres) and 0 <= candidate[1] < len(self.y_centres):
                yield candidate


class PointMazeOpenPlanner:
    """Original shortest-path planner for the obstacle-free Open maze."""

    def __init__(self, grid: Optional[PointMazeOpenGrid] = None):
        self.grid = grid or PointMazeOpenGrid()
        self._policies: Dict[Cell, Dict[Cell, Cell]] = {}

    def _policy_to(self, goal: Cell) -> Dict[Cell, Cell]:
        if goal in self._policies:
            return self._policies[goal]
        policy: Dict[Cell, Cell] = {}
        seen, queue = {goal}, deque([goal])
        while queue:
            cell = queue.popleft()
            for neighbour in self.grid.neighbours(cell):
                if neighbour not in seen:
                    seen.add(neighbour)
                    policy[neighbour] = cell
                    queue.append(neighbour)
        self._policies[goal] = policy
        return policy

    @staticmethod
    def _action_for(vector: np.ndarray) -> int:
        direction = np.sign(vector).astype(int)
        for index, action in enumerate(ACTION_TABLE):
            if np.array_equal(np.sign(action).astype(int), direction):
                return index
        return 0

    def action(self, observation: Dict[str, np.ndarray]) -> int:
        position = np.asarray(observation["achieved_goal"], dtype=np.float32)
        goal_xy = np.asarray(observation["desired_goal"], dtype=np.float32)
        start, goal = self.grid.nearest_cell(position), self.grid.nearest_cell(goal_xy)
        next_cell: Optional[Cell] = self._policy_to(goal).get(start)
        target = goal_xy if next_cell is None else self.grid.cell_centre(next_cell)
        error = target - position
        if float(np.linalg.norm(error)) < 0.05:
            velocity = np.asarray(observation["observation"], dtype=np.float32)[2:4]
            return self._action_for(-velocity)
        return self._action_for(error)
