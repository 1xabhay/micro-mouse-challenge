"""The classical flood-fill mouse — the benchmark every RL bot must beat."""
import pytest

from micromouse.bots import get_bot
from micromouse.generator import generate_maze
from micromouse.maze import Direction
from micromouse.runner import run_attempt


@pytest.fixture
def bot():
    return get_bot("floodfill")


class TestMetadata:
    def test_is_registered_as_a_classical_bot(self, bot):
        assert bot.name == "floodfill"
        assert bot.style == "classical"
        assert bot.description


class TestSolving:
    @pytest.mark.parametrize("seed", range(8))
    def test_solves_every_generated_maze(self, seed, bot):
        maze = generate_maze(seed=seed)
        result = run_attempt(bot, maze, seed=seed)
        assert result.solved, f"floodfill failed maze {seed}"

    def test_never_drives_into_a_wall(self, bot):
        maze = generate_maze(seed=3)
        result = run_attempt(bot, maze)
        assert result.collisions == 0, "it senses walls before choosing a direction"

    def test_finishes_well_inside_the_ten_minute_budget(self, bot):
        maze = generate_maze(seed=3)
        result = run_attempt(bot, maze)
        assert result.maze_time < 600.0


class TestTwoPhaseBehaviour:
    def test_returns_to_the_start_and_races_again(self, bot):
        maze = generate_maze(seed=3)
        result = run_attempt(bot, maze)
        assert len(result.runs) >= 2, "search run, then at least one speed run"

    def test_the_speed_run_beats_the_search_run(self, bot):
        maze = generate_maze(seed=3)
        result = run_attempt(bot, maze)
        assert result.runs[-1] < result.runs[0]

    def test_the_speed_run_is_close_to_the_true_optimum(self, bot):
        maze = generate_maze(seed=3)
        result = run_attempt(bot, maze)
        # cells travelled on the best run vs the optimal cell count
        assert result.best_run_time <= (result.optimal_cells - 1) * 0.12 * 2.5

    def test_it_explores_but_does_not_map_the_whole_maze(self, bot):
        """A good mouse stops searching once it can prove its route is fastest."""
        maze = generate_maze(seed=3)
        result = run_attempt(bot, maze)
        assert 0.05 < result.coverage < 1.0


class TestPolicy:
    def test_first_move_leaves_the_walled_in_start_cell(self, bot):
        from micromouse.sim import MouseSim

        sim = MouseSim(generate_maze(seed=1))
        obs = sim.reset()
        bot.reset(obs)
        assert bot.act(obs) is Direction.N  # the only opening

    def test_is_deterministic(self):
        maze = generate_maze(seed=5)
        a = run_attempt(get_bot("floodfill"), maze)
        b = run_attempt(get_bot("floodfill"), maze)
        assert a.score == b.score
