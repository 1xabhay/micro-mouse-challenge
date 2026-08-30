"""Race several bots over the same pool of mazes and rank them.

Fairness rule: every bot faces exactly the same mazes, generated from the same
seeds, with the same budget. Ranking is by mean handicapped score — the contest
metric — with unsolved mazes counted as an infinite score so a bot cannot climb
the board by quietly failing the hard ones.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .bots import bot_info, get_bot
from .generator import generate_maze
from .runner import DEFAULT_BUDGET_S, DEFAULT_MAX_STEPS, RunResult, run_attempt

__all__ = ["Tournament", "TournamentResult", "run_tournament"]


@dataclass
class TournamentResult:
    attempts: list[RunResult]
    seeds: list[int]
    bots: list[str]

    def leaderboard(self) -> list[dict]:
        """One row per bot, best first."""
        styles = {info["name"]: info["style"] for info in bot_info()}
        wins = self._wins()
        rows = []
        for name in self.bots:
            runs = [a for a in self.attempts if a.bot == name]
            solved = [a for a in runs if a.solved]
            rows.append(
                {
                    "bot": name,
                    "style": styles.get(name, "unknown"),
                    "attempts": len(runs),
                    "solved": len(solved),
                    "solve_rate": len(solved) / len(runs) if runs else 0.0,
                    "wins": wins.get(name, 0),
                    # fmean propagates the inf of an unsolved maze, which is
                    # exactly the ranking behaviour we want.
                    "mean_score": statistics.fmean(a.score for a in runs)
                    if runs
                    else float("inf"),
                    "best_run": min((a.best_run_time for a in solved), default=None),
                    "mean_run": statistics.fmean(a.best_run_time for a in solved)
                    if solved
                    else None,
                    "mean_search": statistics.fmean(a.search_time for a in solved)
                    if solved
                    else None,
                    "mean_maze_time": statistics.fmean(a.maze_time for a in runs)
                    if runs
                    else None,
                    "mean_coverage": statistics.fmean(a.coverage for a in runs)
                    if runs
                    else None,
                    "collisions": sum(a.collisions for a in runs),
                }
            )
        return sorted(rows, key=lambda r: (r["mean_score"], -r["solve_rate"]))

    def _wins(self) -> dict[str, int]:
        """A win is the best score on one maze."""
        wins: dict[str, int] = {name: 0 for name in self.bots}
        for seed in self.seeds:
            heat = [a for a in self.attempts if a.seed == seed]
            if not heat:
                continue
            # Ties and all-failed heats fall to the first bot by score order,
            # which keeps the win totals summing to the number of mazes.
            wins[min(heat, key=lambda a: a.score).bot] += 1
        return wins

    def to_dict(self) -> dict:
        return {
            "seeds": self.seeds,
            "bots": self.bots,
            "leaderboard": self.leaderboard(),
            "attempts": [a.to_dict() for a in self.attempts],
        }


@dataclass
class Tournament:
    bots: Sequence[str]
    seeds: Sequence[int]
    size: int = 16
    loop_ratio: float = 0.1
    budget_s: float = DEFAULT_BUDGET_S
    max_steps: int = DEFAULT_MAX_STEPS
    bot_kwargs: dict[str, dict] = field(default_factory=dict)

    def run(self, on_attempt: Callable[[RunResult], None] | None = None) -> TournamentResult:
        attempts: list[RunResult] = []
        for seed in self.seeds:
            maze = generate_maze(seed=seed, size=self.size, loop_ratio=self.loop_ratio)
            for name in self.bots:
                bot = get_bot(name, **self.bot_kwargs.get(name, {}))
                result = run_attempt(
                    bot,
                    maze,
                    seed=seed,
                    budget_s=self.budget_s,
                    max_steps=self.max_steps,
                )
                attempts.append(result)
                if on_attempt:
                    on_attempt(result)
        return TournamentResult(
            attempts=attempts, seeds=list(self.seeds), bots=list(self.bots)
        )


def run_tournament(
    bots: Iterable[str],
    seeds: Iterable[int],
    on_attempt: Callable[[RunResult], None] | None = None,
    **kwargs,
) -> TournamentResult:
    """Convenience wrapper around :class:`Tournament`."""
    return Tournament(bots=list(bots), seeds=list(seeds), **kwargs).run(on_attempt)
