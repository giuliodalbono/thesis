"""Medium planner wrapper: select geometry, reuse BFS and steering."""
from environments.point_maze_medium import PointMazeMediumGrid
from .grid_waypoint import GridWaypointPlanner


class PointMazeMediumPlanner(GridWaypointPlanner):
    def __init__(self, grid=None):
        super().__init__(grid if grid is not None else PointMazeMediumGrid())
