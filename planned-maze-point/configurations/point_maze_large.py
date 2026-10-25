"""Initial Large DQN pilot; inherits Medium learning/recovery settings.

Learning/recovery settings are unchanged from Medium. Large has its own episode
limit and exploratory sweep checkpoints; these are not tuned performance claims.
"""
from dataclasses import dataclass
from .point_maze_medium import PointMazeMediumConfig


@dataclass(frozen=True)
class PointMazeLargeConfig(PointMazeMediumConfig):
    env_id: str = "PointMaze_Large-v3"
    max_episode_steps: int = 2_000
    warmup_episode_counts: tuple = (25, 50, 75, 100, 125, 150, 200)
    evaluation_episodes: int = 500

    def validate(self):
        if self.env_id != "PointMaze_Large-v3":
            raise ValueError("Use the configuration class matching the environment")
        self._validate_common()

    def environment_slug(self):
        return "pointmaze_large"
