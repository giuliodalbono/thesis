"""Create a fresh planner from the geometry owned by the environment layer."""
from .point_maze_open import PointMazeOpenPlanner
from .point_maze_medium import PointMazeMediumPlanner
from .point_maze_large import PointMazeLargePlanner
from environments import environment_for


_PLANNERS = {
    "point_maze_open": PointMazeOpenPlanner,
    "point_maze_medium": PointMazeMediumPlanner,
    "point_maze_large": PointMazeLargePlanner,
}


def create_planner(config):
    return planner_for_env(config.env_id)


def planner_for_env(env_id):
    # Also used by the standalone planner-validation service.
    spec = environment_for(env_id)
    cls = _PLANNERS[spec.planner_name]
    return cls(spec.create_grid())
