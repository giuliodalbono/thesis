from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from environment import ACTION_TABLE


Cell = Tuple[int, int]


@dataclass(frozen=True)
class PointMazeMediumGrid:
    """Obstacle-aware abstraction of Gymnasium Robotics PointMaze_Medium-v3."""

    maze_map: Tuple[Tuple[int, ...], ...] = (
        (1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 0, 0, 1, 0, 0, 0, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (1, 0, 0, 1, 0, 0, 0, 1),
        (1, 0, 1, 0, 0, 1, 0, 1),
        (1, 0, 0, 0, 1, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1),
    )
    maze_size_scaling: float = 1.0

    @property
    def map_length(self) -> int:
        return len(self.maze_map)

    @property
    def map_width(self) -> int:
        return len(self.maze_map[0])

    @property
    def x_map_center(self) -> float:
        return self.map_width / 2.0 * self.maze_size_scaling

    @property
    def y_map_center(self) -> float:
        return self.map_length / 2.0 * self.maze_size_scaling

    def is_free(self, cell: Cell) -> bool:
        row, column = cell
        return 0 <= row < self.map_length and 0 <= column < self.map_width and self.maze_map[row][column] == 0

    def cell_centre(self, cell: Cell) -> np.ndarray:
        row, column = cell
        return np.array(
            [
                (column + 0.5) * self.maze_size_scaling - self.x_map_center,
                self.y_map_center - (row + 0.5) * self.maze_size_scaling,
            ],
            dtype=np.float32,
        )

    def nearest_cell(self, xy: np.ndarray) -> Cell:
        row = int(np.floor((self.y_map_center - float(xy[1])) / self.maze_size_scaling))
        column = int(np.floor((float(xy[0]) + self.x_map_center) / self.maze_size_scaling))
        candidate = (row, column)
        if self.is_free(candidate):
            return candidate
        free_cells = [
            (map_row, map_column)
            for map_row in range(self.map_length)
            for map_column in range(self.map_width)
            if self.is_free((map_row, map_column))
        ]
        return min(free_cells, key=lambda cell: float(np.linalg.norm(self.cell_centre(cell) - xy)))

    def neighbours(self, cell: Cell) -> Iterable[Cell]:
        for row_delta, column_delta in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            candidate = (cell[0] + row_delta, cell[1] + column_delta)
            if not self.is_free(candidate):
                continue
            if row_delta and column_delta:
                if not self.is_free((cell[0] + row_delta, cell[1])):
                    continue
                if not self.is_free((cell[0], cell[1] + column_delta)):
                    continue
            yield candidate


class PointMazeMediumPlanner:
    """Shortest-path planner that avoids all Medium-maze wall cells."""

    def __init__(self, grid: Optional[PointMazeMediumGrid] = None):
        self.grid = grid or PointMazeMediumGrid()
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
