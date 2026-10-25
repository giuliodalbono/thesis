"""Medium geometry; routing and coordinate operations are shared."""
from dataclasses import dataclass
from typing import Tuple
from .wall_grid import WallMazeGrid


@dataclass(frozen=True)
class PointMazeMediumGrid(WallMazeGrid):
    maze_map: Tuple[Tuple[int, ...], ...] = (
        (1, 1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 1, 1, 0, 0, 1),
        (1, 0, 0, 1, 0, 0, 0, 1),
        (1, 1, 0, 0, 0, 1, 1, 1),
        (1, 0, 0, 1, 0, 0, 0, 1),
        (1, 0, 1, 0, 0, 1, 0, 1),
        (1, 0, 0, 0, 1, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1, 1),
    )
    maze_size_scaling: float = 1.0
