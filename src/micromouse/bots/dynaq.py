"""Dyna-Q: tabular model-based reinforcement learning, learned on the spot.

This mouse arrives knowing nothing about the maze and learns it by trial and
error *during the attempt* — no pretraining, no privileged information.

It is Dyna-Q in the Sutton & Barto sense. Every real step does three things:

1. **Learn the model.** The observed transition ``(cell, direction) -> cell'``
   and its reward are stored. The maze is deterministic, so one observation of
   a transition is enough to know it forever.
2. **Q-learning update** on the real transition.
3. **Planning.** ``planning_steps`` extra Q-updates are replayed from the
   learned model, which is what makes this sample-efficient: value propagates
   back from the goal across the whole known maze without the mouse having to
   physically drive it again.

Exploration is epsilon-greedy with an optimistic initial value, so unvisited
actions are tried before the mouse settles. Between episodes it drives home on
its learned model and races again, and it retires once its runs stop improving.
"""

from __future__ import annotations

import heapq
import random
from collections import defaultdict

from ..mapping import BeliefMap
from ..maze import Cell, Direction
from ..sim import Observation
from .base import Bot
from .registry import register_bot

__all__ = ["DynaQBot"]

State = Cell
Action = Direction


@register_bot
class DynaQBot(Bot):
    name = "dynaq"
    style = "tabular-rl"
    description = (
        "Model-based tabular RL. Learns the maze online with Q-learning plus "
        "Dyna planning sweeps, then exploits the value function to race."
    )

    def __init__(
        self,
        name: str | None = None,
        seed: int | None = None,
        alpha: float = 0.5,
        gamma: float = 0.97,
        epsilon: float = 0.08,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.0,
        planning_steps: int = 30,
        optimistic_init: float = 0.5,
        step_cost: float = 0.05,
        goal_reward: float = 1.0,
        max_episodes: int = 6,
        patience: int = 2,
        theta: float = 1e-4,
    ) -> None:
        if name:
            self.name = name
        self.rng = random.Random(seed)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.planning_steps = planning_steps
        self.optimistic_init = optimistic_init
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.max_episodes = max_episodes
        self.patience = patience
        self.theta = theta

    # ------------------------------------------------------------------ setup

    def reset(self, obs: Observation) -> None:
        self.map = BeliefMap(size=obs.size, goal_cells=obs.goal_cells)
        self.start = obs.position
        self.q: dict[tuple[State, Action], float] = {}
        self.model: dict[tuple[State, Action], tuple[State, float, bool]] = {}
        # An insertion-ordered dict, not a set: sets of Direction-keyed tuples
        # iterate in PYTHONHASHSEED-dependent order, which would make whole
        # attempts irreproducible between processes.
        self.predecessors: dict[State, dict[tuple[State, Action], None]] = defaultdict(dict)
        self._queue: list[tuple[float, int, State, Action]] = []
        self._tie = 0
        self.visits: dict[State, int] = defaultdict(int)
        self.visits[obs.position] += 1
        self.total_updates = 0
        self.episode = 0
        self.homing = False
        self.best_episode_cost = float("inf")
        self.stale_episodes = 0
        self._episode_cost = 0.0
        self._pending: tuple[State, Action] | None = None

    @property
    def q_size(self) -> int:
        return len(self.q)

    # ------------------------------------------------------------------ policy

    def act(self, obs: Observation) -> Direction | None:
        self.map.observe(obs)

        if obs.position in self.map.goal_cells and not self.homing:
            self._end_episode()
            if self.episode >= self.max_episodes or self.stale_episodes >= self.patience:
                return None
            self.homing = True

        if self.homing:
            if obs.position == self.start:
                self.homing = False
                self._episode_cost = 0.0
            else:
                # Drive home on the learned map. Homing is navigation, not
                # learning, so it does not disturb the value function.
                return self.map.next_step(obs.position, obs.heading, {self.start})

        return self._epsilon_greedy(obs)

    def _legal(self, obs: Observation) -> list[Direction]:
        x, y = obs.position
        return [
            d
            for d in Direction
            if not obs.walls[d] and self.map._maze.in_bounds(x + d.delta[0], y + d.delta[1])
        ]

    def _q(self, state: State, action: Action) -> float:
        return self.q.get((state, action), self.optimistic_init)

    def _epsilon_greedy(self, obs: Observation) -> Direction | None:
        legal = self._legal(obs)
        if not legal:
            return None
        if self.rng.random() < self.epsilon:
            choice = self.rng.choice(legal)
        else:
            best = max(self._q(obs.position, d) for d in legal)
            choice = self.rng.choice(
                [d for d in legal if self._q(obs.position, d) == best]
            )
        self._pending = (obs.position, choice)
        return choice

    # ---------------------------------------------------------------- learning

    def on_step(
        self, obs: Observation, action: Direction, next_obs: Observation
    ) -> None:
        if self.homing or self._pending is None:
            return
        state, act = self._pending
        self._pending = None

        nxt = next_obs.position
        self.visits[nxt] += 1
        at_goal = nxt in self.map.goal_cells
        reward = self.goal_reward if at_goal else -self.step_cost
        self._episode_cost += self.step_cost

        self.model[(state, act)] = (nxt, reward, at_goal)
        self.predecessors[nxt][(state, act)] = None
        self._update(state, act, nxt, reward, at_goal)
        self._enqueue(state, act)
        self._plan()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _update(
        self, state: State, action: Action, nxt: State, reward: float, done: bool
    ) -> None:
        future = 0.0 if done else max(
            (self._q(nxt, d) for d in self._known_actions(nxt)), default=0.0
        )
        target = reward + self.gamma * future
        self.q[(state, action)] = self._q(state, action) + self.alpha * (
            target - self._q(state, action)
        )
        self.total_updates += 1

    def _known_actions(self, state: State) -> list[Direction]:
        x, y = state
        return [
            d
            for d in Direction
            if not self.map.has_wall(x, y, d)
            and self.map._maze.in_bounds(x + d.delta[0], y + d.delta[1])
        ]

    def _priority(self, state: State, action: Action) -> float:
        """How far this pair's stored value is from its backed-up target."""
        if (state, action) not in self.model:
            return 0.0
        nxt, reward, done = self.model[(state, action)]
        future = 0.0 if done else max(
            (self._q(nxt, d) for d in self._known_actions(nxt)), default=0.0
        )
        return abs(reward + self.gamma * future - self._q(state, action))

    def _enqueue(self, state: State, action: Action) -> None:
        priority = self._priority(state, action)
        if priority > self.theta:
            self._tie += 1
            heapq.heappush(self._queue, (-priority, self._tie, state, action))

    def _plan(self) -> None:
        """Prioritised sweeping: replay the transitions whose value moved most.

        Plain Dyna samples the model uniformly, so news of the goal takes many
        sweeps to reach the far side of the maze. Sweeping in priority order
        propagates it backwards along predecessors immediately, which is what
        makes this agent competitive within a single attempt.
        """
        for _ in range(self.planning_steps):
            if not self._queue:
                return
            _, _, state, action = heapq.heappop(self._queue)
            nxt, reward, done = self.model[(state, action)]
            self._update(state, action, nxt, reward, done)
            for prev_state, prev_action in self.predecessors[state]:
                self._enqueue(prev_state, prev_action)

    # ---------------------------------------------------------------- episodes

    def _end_episode(self) -> None:
        self.episode += 1
        if self._episode_cost < self.best_episode_cost - 1e-9:
            self.best_episode_cost = self._episode_cost
            self.stale_episodes = 0
        else:
            self.stale_episodes += 1
        self._episode_cost = 0.0
