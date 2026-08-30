"""The map a mouse builds for itself from partial observations.

A :class:`BeliefMap` is the bot-side counterpart of :class:`~micromouse.maze.Maze`.
It starts optimistic — every unseen wall is assumed absent, the standard
flood-fill assumption — and fills in as the mouse senses cells. Planning on an
optimistic map is what makes a mouse explore: unknown territory always looks
like the fastest way to the goal until it is proven otherwise.
"""

from __future__ import annotations

import heapq
from typing import Iterable

from .maze import Cell, Direction, Maze
from .sim import Observation

__all__ = ["BeliefMap"]


class BeliefMap:
    def __init__(self, size: int, goal_cells: Iterable[Cell]) -> None:
        self.size = size
        self.goal_cells = set(goal_cells)
        self._maze = Maze(size=size)  # unknown interior walls stay open
        self._known: set[Cell] = set()

    @classmethod
    def from_maze(cls, maze: Maze) -> "BeliefMap":
        """A fully-informed belief map. For planning tests and speed-run oracles."""
        bm = cls(size=maze.size, goal_cells=maze.goal_cells)
        bm._maze = maze.copy()
        bm._known = set(maze.cells())
        return bm

    # --------------------------------------------------------------- learning

    def observe(self, obs: Observation) -> None:
        """Record the walls sensed at the mouse's current cell."""
        x, y = obs.position
        for d, present in obs.walls.items():
            if present:
                self._maze.set_wall(x, y, d)
            else:
                self._maze.clear_wall(x, y, d)
        self._known.add(obs.position)

    def is_known(self, cell: Cell) -> bool:
        return cell in self._known

    @property
    def known_cells(self) -> int:
        return len(self._known)

    def as_maze(self) -> Maze:
        return self._maze.copy()

    # ----------------------------------------------------------------- walls

    def has_wall(self, x: int, y: int, d: Direction) -> bool:
        return self._maze.has_wall(x, y, d)

    def set_wall(self, x: int, y: int, d: Direction) -> None:
        self._maze.set_wall(x, y, d)

    def open_neighbours(self, x: int, y: int) -> set[Cell]:
        return self._maze.open_neighbours(x, y)

    # --------------------------------------------------------------- planning

    def distance_to_targets(
        self, targets: Iterable[Cell] | None = None
    ) -> dict[Cell, float]:
        """Flood-fill distances to ``targets`` (the goal by default)."""
        return self._maze.distance_field(self.goal_cells if targets is None else targets)

    def next_step(
        self,
        cell: Cell,
        heading: Direction,
        targets: Iterable[Cell],
    ) -> Direction | None:
        """The single best direction from ``cell`` towards ``targets``.

        Ties on distance are broken in favour of the cheapest turn, so the mouse
        carries straight on rather than pirouetting between equal options.
        """
        field = self.distance_to_targets(targets)
        x, y = cell
        best: tuple[float, int, int] | None = None
        choice: Direction | None = None
        for d in Direction:
            if self.has_wall(x, y, d):
                continue
            nxt = (x + d.delta[0], y + d.delta[1])
            if not self._maze.in_bounds(*nxt):
                continue
            key = (field[nxt], heading.turns_to(d), d.value)
            if best is None or key < best:
                best, choice = key, d
        return choice

    def fastest_route(
        self,
        origin: Cell,
        heading: Direction,
        targets: Iterable[Cell],
        cell_time: float = 0.12,
        turn_time: float = 0.18,
        known_only: bool = False,
    ) -> list[Direction] | None:
        """Time-optimal route as a list of directions, or None if unreachable.

        Dijkstra over ``(cell, heading)`` states, pricing turns as well as
        distance — the fewest-cells route is not always the fastest one.

        With ``known_only`` the route may only cross cells the mouse has
        actually sensed, which is what makes a speed run safe: no surprises.
        """
        targets = set(targets)
        if origin in targets:
            return []

        start = (origin, heading)
        dist: dict[tuple[Cell, Direction], float] = {start: 0.0}
        prev: dict[tuple[Cell, Direction], tuple[tuple[Cell, Direction], Direction]] = {}
        queue: list[tuple[float, int, int, int]] = [(0.0, origin[0], origin[1], heading.value)]
        goal_state = None

        while queue:
            cost, x, y, hval = heapq.heappop(queue)
            state = ((x, y), Direction(hval))
            if cost > dist.get(state, float("inf")):
                continue
            if (x, y) in targets:
                goal_state = state
                break
            for d in Direction:
                if self.has_wall(x, y, d):
                    continue
                nxt = (x + d.delta[0], y + d.delta[1])
                if not self._maze.in_bounds(*nxt):
                    continue
                if known_only and nxt not in self._known and nxt not in targets:
                    continue
                step_cost = cell_time + Direction(hval).turns_to(d) * turn_time
                nstate = (nxt, d)
                if cost + step_cost < dist.get(nstate, float("inf")):
                    dist[nstate] = cost + step_cost
                    prev[nstate] = (state, d)
                    heapq.heappush(queue, (cost + step_cost, nxt[0], nxt[1], d.value))

        if goal_state is None:
            return None

        route: list[Direction] = []
        state = goal_state
        while state in prev:
            state, d = prev[state]
            route.append(d)
        return list(reversed(route))

    # -------------------------------------------------------------- frontier

    def frontier(self) -> set[Cell]:
        """Reachable cells the mouse has not yet sensed."""
        reachable = {
            c
            for c, d in self._maze.distance_field(self._known or {(0, 0)}).items()
            if d != float("inf")
        }
        return reachable - self._known
