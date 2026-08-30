"""Procedural generation of contest-style micromouse mazes.

Real contest mazes are not perfect mazes: they contain loops, so there are
several routes to the centre and a mouse that merely *finds* the goal has not
necessarily found the *fast* route. Generation is three stages:

1. Randomised depth-first search carves a spanning tree (every cell reachable).
2. A fraction of the remaining walls is knocked out to create loops.
3. The contest fixtures are stamped on: a start cell walled on three sides and
   a goal block that is open inside but has exactly one gateway.
"""

from __future__ import annotations

import random

from .maze import Cell, Direction, Maze

__all__ = ["generate_maze"]


def generate_maze(
    seed: int | None = None,
    size: int = 16,
    loop_ratio: float = 0.1,
) -> Maze:
    """Build a solvable contest maze.

    Args:
        seed: Seed for reproducible mazes.
        size: Grid edge length.
        loop_ratio: Fraction of interior walls to remove after carving the
            spanning tree. ``0.0`` gives a perfect maze; higher means more
            alternative routes.
    """
    if not 0.0 <= loop_ratio <= 1.0:
        raise ValueError(f"loop_ratio must be in [0, 1], got {loop_ratio}")

    rng = random.Random(seed)
    maze = Maze(size=size)
    _fill_all_walls(maze)
    _carve_spanning_tree(maze, rng)
    _add_loops(maze, rng, loop_ratio)
    _stamp_goal(maze, rng)
    _stamp_start(maze)
    _repair_connectivity(maze, rng)
    return maze


def _fill_all_walls(maze: Maze) -> None:
    for x, y in maze.cells():
        for d in Direction:
            maze.set_wall(x, y, d)


def _carve_spanning_tree(maze: Maze, rng: random.Random) -> None:
    """Randomised DFS ("recursive backtracker") — long, winding corridors."""
    start: Cell = (rng.randrange(maze.size), rng.randrange(maze.size))
    visited = {start}
    stack = [start]
    while stack:
        x, y = stack[-1]
        candidates = []
        for d in Direction:
            dx, dy = d.delta
            nxt = (x + dx, y + dy)
            if maze.in_bounds(*nxt) and nxt not in visited:
                candidates.append((d, nxt))
        if not candidates:
            stack.pop()
            continue
        d, nxt = rng.choice(candidates)
        maze.clear_wall(x, y, d)
        visited.add(nxt)
        stack.append(nxt)


def _add_loops(maze: Maze, rng: random.Random, loop_ratio: float) -> None:
    """Knock out interior walls to create alternative routes."""
    standing = [
        (x, y, d)
        for x, y in maze.cells()
        for d in (Direction.N, Direction.E)
        if maze.has_wall(x, y, d)
        and maze.in_bounds(x + d.delta[0], y + d.delta[1])
    ]
    rng.shuffle(standing)
    for x, y, d in standing[: int(len(standing) * loop_ratio)]:
        maze.clear_wall(x, y, d)


def _stamp_goal(maze: Maze, rng: random.Random) -> None:
    """Open the goal block internally, then seal it down to one gateway."""
    goal = maze.goal_cells

    for x, y in goal:
        for d in Direction:
            neighbour = (x + d.delta[0], y + d.delta[1])
            if neighbour in goal:
                maze.clear_wall(x, y, d)  # open inside the block
            else:
                maze.set_wall(x, y, d)  # seal the perimeter

    # Reopen exactly one perimeter wall, preferring one that keeps the maze
    # solvable without carving through the outer border.
    perimeter = [
        (x, y, d)
        for x, y in sorted(goal)
        for d in Direction
        if (x + d.delta[0], y + d.delta[1]) not in goal
        and maze.in_bounds(x + d.delta[0], y + d.delta[1])
    ]
    rng.shuffle(perimeter)
    x, y, d = perimeter[0]
    maze.clear_wall(x, y, d)


def _repair_connectivity(maze: Maze, rng: random.Random) -> None:
    """Reconnect anything the goal and start seals cut off.

    Stamping the fixtures can strand a region whose only route ran through the
    goal block. Repair joins each stranded region back to the main body, never
    touching the goal perimeter or the start dead end so those rules survive.
    """
    goal = maze.goal_cells
    total = maze.size * maze.size

    def protected(cell: Cell, neighbour: Cell) -> bool:
        return (
            cell in goal
            or neighbour in goal
            or cell == maze.start
            or neighbour == maze.start
        )

    for _ in range(total):  # each pass connects at least one region
        field = maze.distance_field({maze.start})
        reachable = {c for c, d in field.items() if d != float("inf")}
        if len(reachable) == total:
            return
        bridges = [
            (x, y, d)
            for (x, y) in reachable
            for d in Direction
            if maze.has_wall(x, y, d)
            and maze.in_bounds(x + d.delta[0], y + d.delta[1])
            and (x + d.delta[0], y + d.delta[1]) not in reachable
            and not protected((x, y), (x + d.delta[0], y + d.delta[1]))
        ]
        if not bridges:
            return
        maze.clear_wall(*rng.choice(sorted(bridges, key=lambda b: (b[0], b[1], b[2].value))))


def _stamp_start(maze: Maze) -> None:
    """Contest rule: the start cell is a dead end, open on exactly one side.

    The opening faces north, matching the classic south-west start.
    """
    for d in Direction:
        maze.set_wall(0, 0, d)
    maze.clear_wall(0, 0, Direction.N)
