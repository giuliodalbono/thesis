"""Create a fresh planner from the geometry owned by the environment layer."""
from .point_maze_open import PointMazeOpenPlanner
from .point_maze_medium import PointMazeMediumPlanner
from environments import environment_for


def create_planner(config):
    spec = config.environment
    planners = {
        "point_maze_open": PointMazeOpenPlanner,
        "point_maze_medium": PointMazeMediumPlanner,
    }
    return planners[spec.planner_name](spec.create_grid())


def planner_for_env(env_id):
    # Also used by the standalone planner-validation service.
    spec = environment_for(env_id)
    cls = PointMazeOpenPlanner if spec.planner_name == "point_maze_open" else PointMazeMediumPlanner
    return cls(spec.create_grid())
