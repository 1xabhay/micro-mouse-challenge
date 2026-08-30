"""The deep-RL competitor: a trained recurrent policy driving the mouse.

Unlike the other two bots this one carries no map and does no planning. It is a
reactive policy with an LSTM for memory: it sees the walls of the cell it is in,
remembers what it has seen, and picks a direction. All of its competence has to
have been baked in during training on other mazes, which makes it the only bot
here that is tested on *generalisation* rather than on solving the maze in front
of it.

Illegal directions are masked out, so it physically cannot drive into a wall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..env import OBS_DIM, MicromouseEnv
from ..maze import Cell, Direction
from ..sim import Observation
from .base import Bot
from .registry import register_bot

__all__ = ["PPOBot", "DEFAULT_CHECKPOINT"]

DEFAULT_CHECKPOINT = Path("checkpoints/ppo.pt")
DIRECTIONS = MicromouseEnv.DIRECTIONS


@register_bot
class PPOBot(Bot):
    name = "ppo"
    style = "deep-rl"
    description = (
        "Recurrent PPO policy with an LSTM memory and wall masking. Carries no "
        "map: it must generalise from mazes it trained on to the one it is in."
    )

    def __init__(
        self,
        name: str | None = None,
        checkpoint: str | Path | None = DEFAULT_CHECKPOINT,
        seed: int | None = None,
        greedy: bool = False,
        hidden: int = 128,
        stop_at_goal: bool = True,
    ) -> None:
        from ..rl.ppo import ActorCriticLSTM

        if name:
            self.name = name
        self.greedy = greedy
        self.stop_at_goal = stop_at_goal
        self.generator = torch.Generator().manual_seed(seed if seed is not None else 0)

        path = Path(checkpoint) if checkpoint else None
        if path and path.exists():
            self.net = ActorCriticLSTM.load(path)
            self.trained = True
        else:
            # Untrained weights still produce a legal (if aimless) mouse, so the
            # bot is always runnable — useful before the first training run.
            torch.manual_seed(seed if seed is not None else 0)
            self.net = ActorCriticLSTM(obs_dim=OBS_DIM, n_actions=4, hidden=hidden)
            self.net.eval()
            self.trained = False

    def reset(self, obs: Observation) -> None:
        self.state = self.net.initial_state(batch=1)
        self.visited: set[Cell] = {obs.position}
        self.size = obs.size

    def act(self, obs: Observation) -> Direction | None:
        self.visited.add(obs.position)
        if self.stop_at_goal and obs.position in obs.goal_cells:
            return None  # a reactive policy has nothing more to do

        mask = np.array([not obs.walls[d] for d in DIRECTIONS], dtype=bool)
        if not mask.any():
            return None

        with torch.no_grad():
            logits, _, self.state = self.net(
                torch.as_tensor(self._features(obs)).view(1, 1, -1),
                self.state,
                torch.as_tensor(mask).view(1, 1, -1),
            )
            if self.greedy:
                action = int(torch.argmax(logits[0, 0]))
            else:
                action = _sample(logits[0, 0], self.generator)
        return DIRECTIONS[action]

    def _features(self, obs: Observation) -> np.ndarray:
        """Must match :meth:`MicromouseEnv._observe` exactly."""
        x, y = obs.position
        span = self.size - 1
        gx = sum(c[0] for c in obs.goal_cells) / len(obs.goal_cells)
        gy = sum(c[1] for c in obs.goal_cells) / len(obs.goal_cells)

        walls = [float(obs.walls[d]) for d in DIRECTIONS]
        heading = [float(obs.heading is d) for d in DIRECTIONS]
        visited = [
            float((x + d.delta[0], y + d.delta[1]) in self.visited) for d in DIRECTIONS
        ]
        vec = walls + heading + visited + [(gx - x) / span, (gy - y) / span] + [x / span, y / span]
        return np.clip(np.asarray(vec, dtype=np.float32), -1.0, 1.0)


def _sample(logits: torch.Tensor, generator: torch.Generator) -> int:
    """Sample an action reproducibly.

    ``torch.distributions.Categorical`` draws from the global RNG with no
    generator argument, so we sample explicitly to keep attempts repeatable.
    """
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
