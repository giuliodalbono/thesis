"""Ready-to-use agent/environment presets; customize with dataclasses.replace."""
from typing import Union
from .point_maze_open import PointMazeOpenConfig
from .point_maze_medium import PointMazeMediumConfig
from .point_maze_large import PointMazeLargeConfig
from .sac import PointMazeOpenSACConfig, PointMazeMediumSACConfig, PointMazeLargeSACConfig

PointMazeConfig = Union[PointMazeOpenConfig, PointMazeMediumConfig,
                       PointMazeLargeConfig, PointMazeOpenSACConfig,
                       PointMazeMediumSACConfig, PointMazeLargeSACConfig]


def config_from_dict(values):
    """Reconstruct a saved preset for replotting without using today's defaults."""
    from dataclasses import fields
    from pathlib import Path
    classes = {"PointMaze_Open-v3": PointMazeOpenConfig,
               "PointMaze_Medium-v3": PointMazeMediumConfig,
               "PointMaze_Large-v3": PointMazeLargeConfig}
    if values.get("agent_name", "dqn") == "sac_her":
        classes = {"PointMaze_Open-v3": PointMazeOpenSACConfig,
                   "PointMaze_Medium-v3": PointMazeMediumSACConfig,
                   "PointMaze_Large-v3": PointMazeLargeSACConfig}
    elif values.get("agent_name", "dqn") != "dqn":
        raise ValueError("Unknown saved agent")
    cls = classes[values["env_id"]]
    allowed = {field.name for field in fields(cls)}
    kwargs = {key: value for key, value in values.items() if key in allowed}
    if "output_dir" in kwargs:
        kwargs["output_dir"] = Path(kwargs["output_dir"])
    if "warmup_episode_counts" in kwargs:
        kwargs["warmup_episode_counts"] = tuple(kwargs["warmup_episode_counts"])
    if "hidden_sizes" in kwargs:
        kwargs["hidden_sizes"] = tuple(kwargs["hidden_sizes"])
    return cls(**kwargs)
