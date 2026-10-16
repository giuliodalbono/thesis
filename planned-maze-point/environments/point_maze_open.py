from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np



Cell = Tuple[int, int]


@dataclass(frozen=True)
class PointMazeOpenGrid:
    """Open abstraction using the original (x-index, y-index) cell convention.

    Keep axis ordering and neighbour ordering stable: changing tie-breaking in
    BFS would change planner demonstrations even when all paths remain valid.
    """

    x_centres: Tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    y_centres: Tuple[float, ...] = (-1.0, 0.0, 1.0)

    def nearest_cell(self, xy: np.ndarray) -> Cell:
        return (
            int(np.argmin(np.abs(np.asarray(self.x_centres) - float(xy[0])))),
            int(np.argmin(np.abs(np.asarray(self.y_centres) - float(xy[1])))),
        )

    def cell_centre(self, cell: Cell) -> np.ndarray:
        return np.array([self.x_centres[cell[0]], self.y_centres[cell[1]]], dtype=np.float32)

    def neighbours(self, cell: Cell) -> Iterable[Cell]:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            candidate = (cell[0] + dx, cell[1] + dy)
            if 0 <= candidate[0] < len(self.x_centres) and 0 <= candidate[1] < len(self.y_centres):
                yield candidate
