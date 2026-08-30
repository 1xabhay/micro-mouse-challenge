"""Command line interface."""
import json

import pytest

from micromouse.cli import build_parser, main


def run(argv, capsys):
    code = main(argv)
    return code, capsys.readouterr().out


class TestParser:
    def test_exposes_the_documented_commands(self):
        parser = build_parser()
        actions = [a for a in parser._actions if a.dest == "command"]
        assert set(actions[0].choices) >= {"list", "run", "race", "train", "dash", "maze"}

    def test_no_command_prints_help_and_fails(self, capsys):
        assert main([]) == 1


class TestList:
    def test_lists_the_three_competition_styles(self, capsys):
        code, out = run(["list"], capsys)
        assert code == 0
        assert {"floodfill", "dynaq", "ppo"} <= set(out.split())
        assert "classical" in out and "tabular-rl" in out and "deep-rl" in out


class TestMaze:
    def test_prints_a_maze(self, capsys):
        code, out = run(["maze", "--seed", "1"], capsys)
        assert code == 0
        assert out.count("o---") > 10

    def test_is_reproducible(self, capsys):
        _, a = run(["maze", "--seed", "1"], capsys)
        _, b = run(["maze", "--seed", "1"], capsys)
        assert a == b


class TestRun:
    def test_reports_the_headline_result(self, capsys):
        code, out = run(["run", "--bot", "floodfill", "--seed", "1"], capsys)
        assert code == 0
        assert "floodfill" in out and "score" in out.lower()

    def test_json_output_is_machine_readable(self, capsys):
        code, out = run(["run", "--bot", "floodfill", "--seed", "1", "--json"], capsys)
        payload = json.loads(out)
        assert payload["bot"] == "floodfill" and payload["solved"] is True

    def test_unknown_bot_fails_cleanly(self, capsys):
        assert main(["run", "--bot", "nope"]) == 2


class TestRace:
    def test_prints_a_leaderboard(self, capsys):
        code, out = run(
            ["race", "--mazes", "2", "--bots", "floodfill", "--quiet"], capsys
        )
        assert code == 0
        assert "floodfill" in out and "score" in out.lower()

    def test_json_race_output(self, capsys):
        code, out = run(
            ["race", "--mazes", "2", "--bots", "floodfill", "--json", "--quiet"], capsys
        )
        payload = json.loads(out)
        assert len(payload["leaderboard"]) == 1
        assert payload["leaderboard"][0]["attempts"] == 2

    def test_writes_results_to_a_file(self, capsys, tmp_path):
        out_file = tmp_path / "race.json"
        run(
            ["race", "--mazes", "1", "--bots", "floodfill", "--out", str(out_file),
             "--quiet"],
            capsys,
        )
        assert json.loads(out_file.read_text())["leaderboard"]


class TestLeaderboardFormatting:
    def test_columns_line_up_with_the_header(self):
        from micromouse.cli import _HEADER, _leaderboard_row

        row = _leaderboard_row(
            {
                "bot": "floodfill", "style": "classical", "mean_score": 6.61,
                "solved": 12, "attempts": 12, "wins": 12, "mean_run": 5.62,
                "mean_search": 15.64, "mean_coverage": 0.28,
            }
        )
        assert len(row) == len(_HEADER)

    def test_an_unsolved_bot_still_lines_up(self):
        from micromouse.cli import _HEADER, _leaderboard_row

        row = _leaderboard_row(
            {
                "bot": "ppo", "style": "deep-rl", "mean_score": float("inf"),
                "solved": 9, "attempts": 12, "wins": 0, "mean_run": 77.97,
                "mean_search": 139.79, "mean_coverage": 0.49,
            }
        )
        assert len(row) == len(_HEADER)
        assert "inf" in row
