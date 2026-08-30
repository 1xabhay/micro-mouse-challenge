"""The contest runner: run detection, budgets, results."""
import pytest

from micromouse.bots import Bot
from micromouse.generator import generate_maze
from micromouse.maze import Direction, Maze
from micromouse.runner import RunResult, run_attempt


class ScriptedBot(Bot):
    """Replays a fixed list of directions, then idles facing north."""

    name = "scripted"

    def __init__(self, moves):
        self.moves = list(moves)
        self.i = 0

    def act(self, obs):
        if self.i < len(self.moves):
            move = self.moves[self.i]
            self.i += 1
            return move
        return Direction.N


class OracleBot(Bot):
    """Cheats: walks a precomputed path. Used to exercise the runner only."""

    name = "oracle"

    def __init__(self, maze: Maze, laps: int = 1):
        path = maze.shortest_path(maze.start, maze.goal_cells)
        moves = []
        for _ in range(laps):
            legs = _to_dirs(path)
            moves += legs + [d.opposite for d in reversed(legs)]
        self.inner = ScriptedBot(moves)

    def act(self, obs):
        return self.inner.act(obs)


def _to_dirs(path):
    out = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        out.append(next(d for d in Direction if d.delta == (x1 - x0, y1 - y0)))
    return out


class TestRunDetection:
    def test_reaching_the_goal_records_a_run(self):
        maze = generate_maze(seed=1)
        result = run_attempt(OracleBot(maze), maze)
        assert result.solved
        assert len(result.runs) == 1

    def test_run_time_excludes_time_before_leaving_the_start(self):
        maze = Maze(size=4)  # open maze: goal (1,1)/(2,1)/(1,2)/(2,2)
        # spin on the spot, then drive to the goal
        bot = ScriptedBot([Direction.S, Direction.S, Direction.N, Direction.E])
        result = run_attempt(bot, maze, max_steps=8)
        assert result.solved
        # the two wall-bashes happened before the mouse left the start cell
        assert result.runs[0] < result.maze_time

    def test_returning_to_start_allows_a_second_run(self):
        maze = generate_maze(seed=1)
        result = run_attempt(OracleBot(maze, laps=2), maze)
        assert len(result.runs) == 2
        assert result.best_run_time == pytest.approx(min(result.runs))

    def test_lingering_in_the_goal_does_not_count_extra_runs(self):
        maze = Maze(size=4)
        bot = ScriptedBot([Direction.N, Direction.E, Direction.N, Direction.S])
        result = run_attempt(bot, maze, max_steps=10)
        assert len(result.runs) == 1


class TestBudgets:
    def test_stops_at_the_step_budget(self):
        maze = generate_maze(seed=2)
        result = run_attempt(ScriptedBot([]), maze, max_steps=25)
        assert result.steps == 25
        assert not result.solved

    def test_stops_at_the_time_budget(self):
        maze = generate_maze(seed=2)
        result = run_attempt(ScriptedBot([]), maze, budget_s=5.0, max_steps=100_000)
        assert result.maze_time <= 5.0 + 1.0
        assert not result.solved

    def test_unsolved_attempt_scores_infinity(self):
        maze = generate_maze(seed=2)
        result = run_attempt(ScriptedBot([]), maze, max_steps=20)
        assert result.score == float("inf")
        assert result.best_run_time is None


class TestResult:
    def test_result_reports_the_headline_metrics(self):
        maze = generate_maze(seed=1)
        result = run_attempt(OracleBot(maze), maze)
        assert isinstance(result, RunResult)
        assert result.bot == "oracle"
        assert result.solved
        assert result.score < float("inf")
        assert result.score == pytest.approx(
            result.best_run_time + result.maze_time / 30
        )
        assert 0 < result.coverage <= 1.0
        assert result.collisions >= 0

    def test_search_time_is_the_time_to_first_reach_the_goal(self):
        maze = generate_maze(seed=1)
        result = run_attempt(OracleBot(maze), maze)
        assert 0 < result.search_time <= result.maze_time

    def test_trace_is_serialisable_for_the_dashboard(self):
        maze = generate_maze(seed=1)
        result = run_attempt(OracleBot(maze), maze)
        payload = result.to_dict()
        assert payload["bot"] == "oracle"
        assert len(payload["trace"]) == result.steps + 1
        assert set(payload["trace"][0]) == {"x", "y", "heading", "t", "collided"}
        import json

        json.dumps(payload)  # must not raise

    def test_bot_reset_is_called_once_at_the_start(self):
        maze = generate_maze(seed=1)

        class Counter(ScriptedBot):
            name = "counter"
            resets = 0

            def reset(self, obs):
                type(self).resets += 1

        run_attempt(Counter([]), maze, max_steps=5)
        assert Counter.resets == 1

    def test_on_step_hook_sees_every_transition(self):
        maze = generate_maze(seed=1)
        seen = []

        class Hooked(ScriptedBot):
            name = "hooked"

            def on_step(self, obs, action, next_obs):
                seen.append((obs.position, action, next_obs.position))

        run_attempt(Hooked([]), maze, max_steps=7)
        assert len(seen) == 7


class TestRetirement:
    def test_returning_none_retires_and_stops_the_clock(self):
        maze = generate_maze(seed=1)

        class Quitter(Bot):
            name = "quitter"

            def __init__(self):
                self.n = 0

            def act(self, obs):
                self.n += 1
                return None if self.n > 3 else Direction.N

        result = run_attempt(Quitter(), maze, max_steps=5_000)
        assert result.steps == 3
        assert result.maze_time < 5.0
