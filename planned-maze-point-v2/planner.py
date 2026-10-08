"""Select the dedicated planner implementation required by an environment."""

from point_maze_medium_planner import PointMazeMediumPlanner
from point_maze_open_planner import PointMazeOpenPlanner


def planner_for_env(env_id: str):
    if env_id == "PointMaze_Open-v3":
        return PointMazeOpenPlanner()
    if env_id == "PointMaze_Medium-v3":
        return PointMazeMediumPlanner()
    raise ValueError("No abstraction planner is configured for environment: {}".format(env_id))
