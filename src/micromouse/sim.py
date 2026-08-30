"""Simulation of a micromouse driving a maze.

The mouse is *partially observing*: at each step it is told only the walls of
the cell it currently occupies, plus the static contest geometry it would know
in advance (maze size and where the goal is). Building a map is the bot's job.

Actions are absolute compass directions. The simulator charges the turn needed
to face that direction plus the forward move, so straight corridors are cheaper
than zigzags — the same trade-off a real mouse faces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .maze import Cell, Direction, Maze

__all__ = ["Motion", "Observation", "Pose", "MouseSim"]


@dataclass(frozen=True)
class Motion:
    """Timing model, in seconds."""

    cell_time: float = 0.12
    """Time to drive forward one cell."""

    turn_time: float = 0.18
    """Time for one 90-degree turn in place."""

    collision_time: float = 0.50
    """Penalty for driving into a wall (recovery and re-alignment)."""


@dataclass(frozen=True)
class Observation:
    """What the mouse can perceive. Deliberately does not contain the maze."""

    position: Cell
    heading: Direction
    walls: dict[Direction, bool]
    time_s: float
    step: int
    size: int
    goal_cells: frozenset[Cell]


@dataclass(frozen=True)
class Pose:
    """One entry in the recorded trajectory."""

    position: Cell
    heading: Direction
    time_s: float
    collided: bool = False


class MouseSim:
    """Drives a mouse around a hidden maze."""

    def __init__(self, maze: Maze, motion: Motion | None = None) -> None:
        self.maze = maze
        self.motion = motion or Motion()
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> Observation:
        self.position: Cell = self.maze.start
        self.heading: Direction = Direction.N
        self.elapsed: float = 0.0
        self.steps: int = 0
        self.collisions: int = 0
        self.visited: set[Cell] = {self.maze.start}
        self.trace: list[Pose] = [Pose(self.position, self.heading, 0.0)]
        return self.observe()

    def observe(self) -> Observation:
        x, y = self.position
        return Observation(
            position=self.position,
            heading=self.heading,
            walls={d: self.maze.has_wall(x, y, d) for d in Direction},
            time_s=self.elapsed,
            step=self.steps,
            size=self.maze.size,
            goal_cells=frozenset(self.maze.goal_cells),
        )

    @property
    def at_goal(self) -> bool:
        return self.position in self.maze.goal_cells

    @property
    def at_start(self) -> bool:
        return self.position == self.maze.start

    # ------------------------------------------------------------------- step

    def step(self, direction: Direction) -> Observation:
        """Turn to face ``direction`` and drive one cell if the way is clear."""
        turns = self.heading.turns_to(direction)
        self.elapsed += turns * self.motion.turn_time
        self.heading = direction
        self.steps += 1

        x, y = self.position
        collided = self.maze.has_wall(x, y, direction)
        if collided:
            self.collisions += 1
            self.elapsed += self.motion.collision_time
        else:
            dx, dy = direction.delta
            self.position = (x + dx, y + dy)
            self.visited.add(self.position)
            self.elapsed += self.motion.cell_time

        self.trace.append(Pose(self.position, self.heading, self.elapsed, collided))
        return self.observe()
