"""Bind each supported Gymnasium id to its coordinate/grid abstraction."""
from dataclasses import dataclass
from .common import make_env
from .point_maze_open import PointMazeOpenGrid
from .point_maze_medium import PointMazeMediumGrid
from .point_maze_large import PointMazeLargeGrid


@dataclass(frozen=True)
class PointMazeEnvironment:
    env_id: str
    grid_type: type
    planner_name: str

    def create(self, max_episode_steps, continuous=False):
        return make_env(self.env_id, max_episode_steps, continuous=continuous)

    def create_grid(self):
        return self.grid_type()


_ENVIRONMENTS = {
    "PointMaze_Open-v3": PointMazeEnvironment(
        "PointMaze_Open-v3", PointMazeOpenGrid, "point_maze_open"),
    "PointMaze_Medium-v3": PointMazeEnvironment(
        "PointMaze_Medium-v3", PointMazeMediumGrid, "point_maze_medium"),
    "PointMaze_Large-v3": PointMazeEnvironment(
        "PointMaze_Large-v3", PointMazeLargeGrid, "point_maze_large"),
}


def environment_for(env_id):
    try:
        return _ENVIRONMENTS[env_id]
    except KeyError:
        raise ValueError("Unsupported PointMaze environment: {}".format(env_id))
