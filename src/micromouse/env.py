"""Gymnasium environment for training a mouse.

The observation is deliberately *egocentric and partial* — walls of the current
cell, heading, a bearing to the goal, and which neighbours have already been
visited. That is everything a real mouse can sense plus the memory it is
allowed to keep about where it has been. It never contains a wall the mouse has
not driven up to, so a policy trained here is legal in a contest.

Reward is potential-based: the shaping term uses the true distance-to-goal,
which is privileged information available only during *training*. Because it is
a potential difference it does not change the optimal policy, it only makes the
credit assignment tractable.
"""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .generator import generate_maze
from .maze import Direction, Maze
from .sim import Motion, MouseSim

__all__ = ["MicromouseEnv", "OBS_DIM"]

OBS_DIM = 16


class MicromouseEnv(gym.Env):
    """One episode is one search run: start cell to goal."""

    metadata = {"render_modes": ["ansi"]}

    DIRECTIONS: tuple[Direction, ...] = (
        Direction.N,
        Direction.E,
        Direction.S,
        Direction.W,
    )

    def __init__(
        self,
        size: int = 16,
        maze_seeds: Sequence[int] | None = None,
        max_steps: int = 2_000,
        loop_ratio: float = 0.1,
        shaping: float = 0.5,
        collision_penalty: float = 0.5,
        step_penalty: float = 0.01,
        goal_reward: float = 5.0,
        motion: Motion | None = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.maze_seeds = list(maze_seeds) if maze_seeds is not None else list(range(200))
        self.max_steps = max_steps
        self.loop_ratio = loop_ratio
        self.shaping = shaping
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.motion = motion or Motion()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self._episode = 0
        self.maze: Maze | None = None
        self.maze_seed: int | None = None

    # ----------------------------------------------------------------- gym api

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Gym contract: reset(seed=s) must be reproducible, so an explicit seed
        # selects the maze. Seedless resets walk the pool, which is what the
        # training loop wants.
        forced = (options or {}).get("maze_seed")
        if forced is not None:
            self.maze_seed = int(forced)
        elif seed is not None:
            self.maze_seed = self.maze_seeds[seed % len(self.maze_seeds)]
        else:
            self.maze_seed = self.maze_seeds[self._episode % len(self.maze_seeds)]
            self._episode += 1

        self.maze = generate_maze(
            seed=self.maze_seed, size=self.size, loop_ratio=self.loop_ratio
        )
        self.sim = MouseSim(self.maze, motion=self.motion)
        self.sim.reset()
        self._true_dist = self.maze.distance_field(self.maze.goal_cells)
        self._prev_potential = self._potential()
        return self._observe(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        direction = self.DIRECTIONS[int(action)]
        before_cells = len(self.sim.visited)
        collisions_before = self.sim.collisions

        self.sim.step(direction)

        reward = -self.step_penalty
        if self.sim.collisions > collisions_before:
            reward -= self.collision_penalty

        potential = self._potential()
        reward += self.shaping * (potential - self._prev_potential)
        self._prev_potential = potential

        if len(self.sim.visited) > before_cells:
            reward += 0.01  # a nudge towards seeing new ground

        terminated = self.sim.at_goal
        if terminated:
            reward += self.goal_reward
        truncated = self.sim.steps >= self.max_steps and not terminated

        return self._observe(), float(reward), bool(terminated), bool(truncated), self._info()

    def render(self) -> str:
        return self.maze.to_text()

    # -------------------------------------------------------------- internals

    def _potential(self) -> float:
        """Normalised closeness to the goal; 1.0 at the centre."""
        d = self._true_dist[self.sim.position]
        if d == float("inf"):
            return 0.0
        return 1.0 - d / (2 * self.size)

    def _observe(self) -> np.ndarray:
        obs = self.sim.observe()
        x, y = obs.position
        span = self.size - 1
        gx = sum(c[0] for c in obs.goal_cells) / len(obs.goal_cells)
        gy = sum(c[1] for c in obs.goal_cells) / len(obs.goal_cells)

        walls = [float(obs.walls[d]) for d in self.DIRECTIONS]
        heading = [float(obs.heading is d) for d in self.DIRECTIONS]
        visited = [
            float((x + d.delta[0], y + d.delta[1]) in self.sim.visited)
            for d in self.DIRECTIONS
        ]
        bearing = [(gx - x) / span, (gy - y) / span]
        position = [x / span, y / span]

        vec = walls + heading + visited + bearing + position
        return np.clip(np.asarray(vec, dtype=np.float32), -1.0, 1.0)

    def _info(self) -> dict:
        return {
            "cells_visited": len(self.sim.visited),
            "elapsed": self.sim.elapsed,
            "collisions": self.sim.collisions,
            "maze_seed": self.maze_seed,
        }

    # ------------------------------------------------------------------ oracle

    def expert_action(self) -> int:
        """Privileged shortest-path action. For tests and imitation warm-starts."""
        x, y = self.sim.position
        best, choice = float("inf"), 0
        for i, d in enumerate(self.DIRECTIONS):
            if self.maze.has_wall(x, y, d):
                continue
            nxt = (x + d.delta[0], y + d.delta[1])
            if self._true_dist[nxt] < best:
                best, choice = self._true_dist[nxt], i
        return choice
