"""The bot plugin contract.

A bot is anything that can look at an :class:`~micromouse.sim.Observation` and
name the compass direction to drive next. Everything else — mapping, planning,
learning — is up to the bot. Learning bots can additionally use the
:meth:`Bot.on_step` hook to see the outcome of their own actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..maze import Direction
from ..sim import Observation

__all__ = ["Bot"]


class Bot(ABC):
    """Base class for every mouse driver."""

    name: str = "unnamed"
    style: str = "unknown"
    description: str = ""

    @abstractmethod
    def act(self, obs: Observation) -> Direction | None:
        """Choose the compass direction to drive next.

        Return ``None`` to retire from the attempt. Retiring stops the clock,
        which matters: every extra second in the maze is charged back as a
        search penalty, so a bot that has nothing left to learn should stop.
        """

    def reset(self, obs: Observation) -> None:
        """Start a fresh attempt on a new maze. Default: nothing to do."""

    def on_step(
        self,
        obs: Observation,
        action: Direction,
        next_obs: Observation,
    ) -> None:
        """Observe the outcome of an action. Default: ignore it."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, style={self.style!r})"
