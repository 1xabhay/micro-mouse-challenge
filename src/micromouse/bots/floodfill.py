"""Flood-fill: the classical micromouse algorithm, and the bot to beat.

No learning, no randomness. The mouse keeps an optimistic map, repeatedly
flood-fills distances to whatever it currently wants to reach, and always steps
down the gradient. Because unseen walls are assumed absent, unexplored routes
always look attractive, which makes the same rule serve as both an explorer and
a solver.

The attempt runs in three phases, exactly as a contest mouse does it:

1. ``SEARCH``  — drive to the goal, learning walls on the way.
2. ``RETURN``  — drive home, still learning.
3. ``SPEED``   — race the time-optimal route across cells already proven safe.

Then it retires, because idling in the maze only adds search penalty.
"""

from __future__ import annotations

from enum import Enum, auto

from ..mapping import BeliefMap
from ..maze import Direction
from ..sim import Motion, Observation
from .base import Bot
from .registry import register_bot

__all__ = ["FloodFillBot"]


class Phase(Enum):
    SEARCH = auto()
    RETURN = auto()
    SPEED = auto()
    DONE = auto()


@register_bot
class FloodFillBot(Bot):
    name = "floodfill"
    style = "classical"
    description = (
        "Contest-standard flood fill: optimistic mapping, gradient descent to "
        "the goal, then a turn-aware speed run over proven cells."
    )

    def __init__(self, name: str | None = None, speed_runs: int = 1) -> None:
        if name:
            self.name = name
        self.speed_runs = speed_runs
        self.motion = Motion()
        self.map: BeliefMap | None = None

    def reset(self, obs: Observation) -> None:
        self.map = BeliefMap(size=obs.size, goal_cells=obs.goal_cells)
        self.phase = Phase.SEARCH
        self.start = obs.position
        self.runs_done = 0

    def act(self, obs: Observation) -> Direction | None:
        self.map.observe(obs)
        self._advance_phase(obs)

        if self.phase is Phase.DONE:
            return None
        if self.phase is Phase.SPEED:
            return self._race(obs)
        targets = self.map.goal_cells if self.phase is Phase.SEARCH else {self.start}
        return self.map.next_step(obs.position, obs.heading, targets)

    # ---------------------------------------------------------------- phases

    def _advance_phase(self, obs: Observation) -> None:
        at_goal = obs.position in self.map.goal_cells
        at_start = obs.position == self.start

        if self.phase is Phase.SEARCH and at_goal:
            self.phase = Phase.RETURN
        elif self.phase is Phase.RETURN and at_start:
            self.phase = Phase.SPEED
        elif self.phase is Phase.SPEED and at_goal:
            self.runs_done += 1
            self.phase = Phase.DONE if self.runs_done >= self.speed_runs else Phase.RETURN

    def _race(self, obs: Observation) -> Direction | None:
        """Follow the time-optimal known route, re-planned every step."""
        route = self.map.fastest_route(
            obs.position,
            obs.heading,
            self.map.goal_cells,
            cell_time=self.motion.cell_time,
            turn_time=self.motion.turn_time,
            known_only=True,
        )
        if route:
            return route[0]
        # No proven route (shouldn't happen after a search) — fall back to gradient.
        return self.map.next_step(obs.position, obs.heading, self.map.goal_cells)
