"""Large-v3 wall layout, checked against the installed Gymnasium environment."""
from dataclasses import dataclass
from .wall_grid import WallMazeGrid


@dataclass(frozen=True)
class PointMazeLargeGrid(WallMazeGrid):
    maze_map: tuple = (
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1),
        (1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1),
        (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1),
        (1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1),
        (1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1),
        (1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1),
        (1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    )
