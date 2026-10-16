"""Deterministic Open planner: BFS on the environment's grid, then steering.

The existing neighbour ordering, waypoint choice and braking rule are retained
so this refactor does not change demonstrated actions.
"""
from __future__ import annotations
from collections import deque
from typing import Dict, Optional, Tuple
import numpy as np
from environments.common import ACTION_TABLE
from environments.point_maze_open import PointMazeOpenGrid
Cell = Tuple[int, int]


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
