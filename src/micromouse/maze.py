"""Maze representation for a micromouse contest maze.

Conventions follow the classic micromouse standard:

* Cell ``(0, 0)`` is the south-west corner; ``x`` grows east, ``y`` grows north.
* Walls are a per-cell bitmask ``N=1, E=2, S=4, W=8`` (the ``.maz`` layout).
* Internal walls are stored twice, once from each side, and are kept in sync.
* The goal is the 2x2 block at the centre of an even-sized maze.
"""

from __future__ import annotations

from collections import deque
from enum import Enum
from typing import Iterable, Iterator

import numpy as np


class Direction(Enum):
    """Compass direction, ordered clockwise so turns are modular arithmetic."""

    N = 0
    E = 1
    S = 2
    W = 3

    @property
    def bit(self) -> int:
        """Wall bit in the ``.maz`` encoding: N=1, E=2, S=4, W=8."""
        return 1 << self.value

    @property
    def delta(self) -> tuple[int, int]:
        return ((0, 1), (1, 0), (0, -1), (-1, 0))[self.value]

    @property
    def opposite(self) -> "Direction":
        return Direction((self.value + 2) % 4)

    def right(self) -> "Direction":
        return Direction((self.value + 1) % 4)

    def left(self) -> "Direction":
        return Direction((self.value - 1) % 4)

    def turns_to(self, other: "Direction") -> int:
        """Number of 90-degree turns to face ``other`` (0, 1 or 2)."""
        diff = (other.value - self.value) % 4
        return min(diff, 4 - diff)


Cell = tuple[int, int]


class Maze:
    """A grid of cells with walls between them."""

    def __init__(self, size: int = 16) -> None:
        if size < 2:
            raise ValueError(f"maze size must be at least 2, got {size}")
        self.size = size
        self._walls = np.zeros((size, size), dtype=np.uint8)
        self._seal_border()

    # ---------------------------------------------------------------- geometry

    def _seal_border(self) -> None:
        self._walls[0, :] |= Direction.W.bit
        self._walls[-1, :] |= Direction.E.bit
        self._walls[:, 0] |= Direction.S.bit
        self._walls[:, -1] |= Direction.N.bit

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def _is_border(self, x: int, y: int, d: Direction) -> bool:
        dx, dy = d.delta
        return not self.in_bounds(x + dx, y + dy)

    @property
    def start(self) -> Cell:
        return (0, 0)

    @property
    def goal_cells(self) -> set[Cell]:
        """The centre block: 2x2 for an even maze, a single cell for an odd one."""
        c = (self.size - 1) // 2
        if self.size % 2:
            return {(c, c)}
        return {(c, c), (c + 1, c), (c, c + 1), (c + 1, c + 1)}

    def cells(self) -> Iterator[Cell]:
        for y in range(self.size):
            for x in range(self.size):
                yield (x, y)

    # ------------------------------------------------------------------- walls

    def has_wall(self, x: int, y: int, d: Direction) -> bool:
        if not self.in_bounds(x, y):
            return True  # outside the maze is solid
        return bool(self._walls[x, y] & d.bit)

    def walls_at(self, x: int, y: int) -> int:
        """Raw wall bitmask for a cell."""
        return int(self._walls[x, y])

    def set_wall(self, x: int, y: int, d: Direction) -> None:
        """Raise a wall, from both sides."""
        self._walls[x, y] |= d.bit
        dx, dy = d.delta
        nx, ny = x + dx, y + dy
        if self.in_bounds(nx, ny):
            self._walls[nx, ny] |= d.opposite.bit

    def clear_wall(self, x: int, y: int, d: Direction) -> None:
        """Remove a wall from both sides. The outer border is never removed."""
        if self._is_border(x, y, d):
            return
        self._walls[x, y] &= 0xFF ^ d.bit
        dx, dy = d.delta
        nx, ny = x + dx, y + dy
        if self.in_bounds(nx, ny):
            self._walls[nx, ny] &= 0xFF ^ d.opposite.bit

    def open_neighbours(self, x: int, y: int) -> set[Cell]:
        """Cells reachable from ``(x, y)`` in one step."""
        out = set()
        for d in Direction:
            if self.has_wall(x, y, d):
                continue
            dx, dy = d.delta
            if self.in_bounds(x + dx, y + dy):
                out.add((x + dx, y + dy))
        return out

    # --------------------------------------------------------------- distances

    def distance_field(self, sources: Iterable[Cell]) -> dict[Cell, float]:
        """BFS step-distance from any of ``sources`` to every cell."""
        dist: dict[Cell, float] = {c: float("inf") for c in self.cells()}
        queue: deque[Cell] = deque()
        for cell in sources:
            dist[cell] = 0
            queue.append(cell)
        while queue:
            x, y = queue.popleft()
            for nxt in self.open_neighbours(x, y):
                if dist[nxt] == float("inf"):
                    dist[nxt] = dist[(x, y)] + 1
                    queue.append(nxt)
        return dist

    def shortest_path(self, origin: Cell, targets: Iterable[Cell]) -> list[Cell] | None:
        """A fewest-cells path from ``origin`` to the nearest target, or None."""
        targets = set(targets)
        field = self.distance_field(targets)
        if field[origin] == float("inf"):
            return None
        path = [origin]
        current = origin
        while current not in targets:
            # Sort the tie-break key: set iteration order is not stable across
            # processes, and an unstable path would break reproducibility.
            current = min(sorted(self.open_neighbours(*current)), key=lambda c: field[c])
            path.append(current)
        return path

    def is_solvable(self) -> bool:
        return self.distance_field(self.goal_cells)[self.start] != float("inf")

    # ----------------------------------------------------------- serialisation

    def to_maz_bytes(self) -> bytes:
        """256-byte ``.maz`` blob, indexed ``y * size + x`` (north adds ``size``)."""
        return bytes(self._walls[x, y] for y in range(self.size) for x in range(self.size))

    @classmethod
    def from_maz_bytes(cls, blob: bytes) -> "Maze":
        size = int(round(len(blob) ** 0.5))
        if size * size != len(blob):
            raise ValueError(f"maze blob of {len(blob)} bytes is not square")
        maze = cls(size=size)
        for i, value in enumerate(blob):
            maze._walls[i % size, i // size] = value
        maze._seal_border()
        return maze

    def to_text(self) -> str:
        """Classic text rendering: ``o`` posts, ``---`` and ``|`` walls."""
        lines = []
        for y in range(self.size - 1, -1, -1):
            top = "".join(
                "o" + ("---" if self.has_wall(x, y, Direction.N) else "   ")
                for x in range(self.size)
            )
            lines.append(top + "o")
            mid = "".join(
                ("|" if self.has_wall(x, y, Direction.W) else " ") + "   "
                for x in range(self.size)
            )
            lines.append(mid + "|")
        bottom = "".join(
            "o" + ("---" if self.has_wall(x, 0, Direction.S) else "   ")
            for x in range(self.size)
        )
        lines.append(bottom + "o")
        return "\n".join(lines)

    @classmethod
    def from_text(cls, text: str) -> "Maze":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        size = len(lines) // 2
        maze = cls(size=size)
        for y in range(size):
            top_row = lines[2 * (size - 1 - y)]
            mid_row = lines[2 * (size - 1 - y) + 1]
            for x in range(size):
                if top_row[4 * x + 1 : 4 * x + 4] == "---":
                    maze.set_wall(x, y, Direction.N)
                if mid_row[4 * x] == "|":
                    maze.set_wall(x, y, Direction.W)
        bottom = lines[2 * size]
        for x in range(size):
            if bottom[4 * x + 1 : 4 * x + 4] == "---":
                maze.set_wall(x, 0, Direction.S)
        return maze

    # ------------------------------------------------------------------- dunder

    def copy(self) -> "Maze":
        clone = Maze(size=self.size)
        clone._walls = self._walls.copy()
        return clone

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Maze):
            return NotImplemented
        return self.size == other.size and np.array_equal(self._walls, other._walls)

    def __hash__(self) -> int:
        return hash((self.size, self._walls.tobytes()))

    def __repr__(self) -> str:
        return f"Maze(size={self.size})"
