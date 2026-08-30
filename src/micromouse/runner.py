"""Run a bot through a contest attempt and score it.

A contest attempt is a single continuous session in the maze. The runner does
not tell the bot when to search and when to race: it simply watches the
trajectory and, exactly as a judge would, times every start-to-goal journey.
The fastest of those is the scoring run; total time in the maze becomes the
search penalty. So a bot is free to explore, return to the start and race,
however many times it likes within the budget.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .bots.base import Bot
from .maze import Maze
from .scoring import handicapped_score
from .sim import Motion, MouseSim

__all__ = ["RunResult", "run_attempt"]

DEFAULT_BUDGET_S = 600.0  # the classic 10-minute allowance
DEFAULT_MAX_STEPS = 20_000


@dataclass
class RunResult:
    """The outcome of one bot's attempt at one maze."""

    bot: str
    seed: int | None
    solved: bool
    runs: list[float]
    best_run_time: float | None
    search_time: float | None
    maze_time: float
    score: float
    steps: int
    collisions: int
    coverage: float
    optimal_cells: int | None
    trace: list[dict] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return asdict(self)


def run_attempt(
    bot: Bot,
    maze: Maze,
    seed: int | None = None,
    budget_s: float = DEFAULT_BUDGET_S,
    max_steps: int = DEFAULT_MAX_STEPS,
    motion: Motion | None = None,
) -> RunResult:
    """Drive ``bot`` around ``maze`` until it runs out of budget."""
    sim = MouseSim(maze, motion=motion)
    obs = sim.reset()
    bot.reset(obs)

    runs: list[float] = []
    search_time: float | None = None
    run_start_t: float | None = None
    # A run is only armed once the mouse has left the start cell, and rearms
    # only after it comes home — so idling in the goal cannot score twice.
    armed = False

    while sim.steps < max_steps and sim.elapsed < budget_s:
        was_at_start = sim.at_start
        action = bot.act(obs)
        if action is None:
            break  # the bot has retired; the clock stops here
        next_obs = sim.step(action)
        bot.on_step(obs, action, next_obs)
        obs = next_obs

        if was_at_start and not sim.at_start and not armed:
            armed = True
            run_start_t = sim.trace[-2].time_s  # time at the moment it left

        if armed and sim.at_goal:
            runs.append(sim.elapsed - run_start_t)
            armed = False
            if search_time is None:
                search_time = sim.elapsed

        if sim.at_start:
            armed = False

    best = min(runs) if runs else None
    total_cells = maze.size * maze.size
    optimal = maze.shortest_path(maze.start, maze.goal_cells)

    return RunResult(
        bot=bot.name,
        seed=seed,
        solved=bool(runs),
        runs=runs,
        best_run_time=best,
        search_time=search_time,
        maze_time=sim.elapsed,
        score=handicapped_score(best, sim.elapsed),
        steps=sim.steps,
        collisions=sim.collisions,
        coverage=len(sim.visited) / total_cells,
        optimal_cells=len(optimal) if optimal else None,
        trace=[
            {
                "x": p.position[0],
                "y": p.position[1],
                "heading": p.heading.name,
                "t": round(p.time_s, 4),
                "collided": p.collided,
            }
            for p in sim.trace
        ],
    )
