from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np



Cell = Tuple[int, int]


@dataclass(frozen=True)
class PointMazeMediumGrid:
    """Medium wall map using Gymnasium's (row, column) convention.

    Rows increase downwards but MuJoCo y increases upwards. The conversion
    below therefore reverses the row direction and centers coordinates on the
    maze. Cells with value 1 are walls; only zero-valued cells are reachable.
    """

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

    @property
    def map_length(self) -> int:
        return len(self.maze_map)

    @property
    def map_width(self) -> int:
        return len(self.maze_map[0])

    @property
    def x_map_center(self) -> float:
        return self.map_width / 2.0 * self.maze_size_scaling

    @property
    def y_map_center(self) -> float:
        return self.map_length / 2.0 * self.maze_size_scaling

    def is_free(self, cell: Cell) -> bool:
        row, column = cell
        return 0 <= row < self.map_length and 0 <= column < self.map_width and self.maze_map[row][column] == 0

    def cell_centre(self, cell: Cell) -> np.ndarray:
        row, column = cell
        return np.array(
            [
                (column + 0.5) * self.maze_size_scaling - self.x_map_center,
                self.y_map_center - (row + 0.5) * self.maze_size_scaling,
            ],
            dtype=np.float32,
        )

    def nearest_cell(self, xy: np.ndarray) -> Cell:
        row = int(np.floor((self.y_map_center - float(xy[1])) / self.maze_size_scaling))
        column = int(np.floor((float(xy[0]) + self.x_map_center) / self.maze_size_scaling))
        candidate = (row, column)
        if self.is_free(candidate):
            return candidate
        free_cells = [
            (map_row, map_column)
            for map_row in range(self.map_length)
            for map_column in range(self.map_width)
            if self.is_free((map_row, map_column))
        ]
        return min(free_cells, key=lambda cell: float(np.linalg.norm(self.cell_centre(cell) - xy)))

    def neighbours(self, cell: Cell) -> Iterable[Cell]:
        for row_delta, column_delta in (
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ):
            candidate = (cell[0] + row_delta, cell[1] + column_delta)
            if not self.is_free(candidate):
                continue
            if row_delta and column_delta:
                # A clear destination alone is insufficient for a diagonal:
                # both side cells must be free to avoid clipping a wall corner.
                if not self.is_free((cell[0] + row_delta, cell[1])):
                    continue
                if not self.is_free((cell[0], cell[1] + column_delta)):
                    continue
            yield candidate
