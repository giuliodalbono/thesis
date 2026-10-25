"""Open planner wrapper: select geometry, reuse BFS and steering."""
from environments.point_maze_open import PointMazeOpenGrid
from .grid_waypoint import GridWaypointPlanner


class PointMazeOpenPlanner(GridWaypointPlanner):
    def __init__(self, grid=None):
        super().__init__(grid if grid is not None else PointMazeOpenGrid())
