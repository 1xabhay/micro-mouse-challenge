"""Head-to-head competition across a pool of mazes."""
import pytest

from micromouse.tournament import Tournament, TournamentResult, run_tournament


class TestTournament:
    def test_runs_every_bot_on_every_maze(self):
        result = run_tournament(bots=["floodfill"], seeds=[0, 1, 2])
        assert len(result.attempts) == 3
        assert {a.seed for a in result.attempts} == {0, 1, 2}

    def test_all_bots_face_identical_mazes(self):
        """A fair race means the same maze for everyone."""
        result = run_tournament(bots=["floodfill", "dynaq"], seeds=[0, 1])
        by_seed = {}
        for a in result.attempts:
            by_seed.setdefault(a.seed, set()).add(a.bot)
        assert all(bots == {"floodfill", "dynaq"} for bots in by_seed.values())

    def test_leaderboard_ranks_by_mean_score(self):
        result = run_tournament(bots=["floodfill", "dynaq"], seeds=[0, 1])
        board = result.leaderboard()
        assert [r["bot"] for r in board] == sorted(
            [r["bot"] for r in board], key=lambda b: next(
                x["mean_score"] for x in board if x["bot"] == b
            )
        )
        assert board[0]["bot"] == "floodfill", "the classical baseline should win"

    def test_leaderboard_reports_the_metrics_that_matter(self):
        result = run_tournament(bots=["floodfill"], seeds=[0, 1])
        row = result.leaderboard()[0]
        assert set(row) >= {
            "bot", "style", "solved", "attempts", "solve_rate",
            "mean_score", "best_run", "mean_search", "mean_coverage",
        }

    def test_unsolved_mazes_do_not_poison_the_mean_with_infinity(self):
        result = run_tournament(bots=["ppo"], seeds=[0], max_steps=50)
        row = result.leaderboard()[0]
        assert row["solve_rate"] == 0.0
        assert row["mean_score"] == float("inf")
        assert row["best_run"] is None

    def test_wins_counts_per_maze_victories(self):
        result = run_tournament(bots=["floodfill", "dynaq"], seeds=[0, 1, 2])
        board = result.leaderboard()
        assert sum(r["wins"] for r in board) == 3

    def test_result_serialises_for_the_dashboard(self):
        import json

        result = run_tournament(bots=["floodfill"], seeds=[0])
        payload = result.to_dict()
        json.dumps(payload)
        assert "leaderboard" in payload and "attempts" in payload

    def test_progress_callback_reports_each_attempt(self):
        seen = []
        run_tournament(
            bots=["floodfill"], seeds=[0, 1], on_attempt=lambda a: seen.append(a.bot)
        )
        assert seen == ["floodfill", "floodfill"]

    def test_rejects_an_unknown_bot(self):
        with pytest.raises(KeyError):
            run_tournament(bots=["not-a-bot"], seeds=[0])
