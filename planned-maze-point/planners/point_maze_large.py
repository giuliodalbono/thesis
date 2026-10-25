"""Large planner wrapper: select geometry, reuse BFS and steering."""
from environments.point_maze_large import PointMazeLargeGrid
from .grid_waypoint import GridWaypointPlanner


class PointMazeLargePlanner(GridWaypointPlanner):
    def __init__(self, grid=None):
        super().__init__(grid if grid is not None else PointMazeLargeGrid())
