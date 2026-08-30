"""Dyna-Q: tabular model-based RL that learns the maze during the attempt."""
import pytest

from micromouse.bots import get_bot
from micromouse.generator import generate_maze
from micromouse.maze import Direction
from micromouse.runner import run_attempt
from micromouse.sim import MouseSim


@pytest.fixture
def bot():
    return get_bot("dynaq", seed=0)


class TestMetadata:
    def test_is_registered_as_a_tabular_rl_bot(self, bot):
        assert bot.name == "dynaq"
        assert bot.style == "tabular-rl"
        assert bot.description


class TestLearning:
    def test_q_table_starts_empty(self, bot):
        maze = generate_maze(seed=1)
        bot.reset(MouseSim(maze).reset())
        assert bot.q_size == 0

    def test_experience_populates_the_q_table(self, bot):
        maze = generate_maze(seed=1)
        run_attempt(bot, maze, max_steps=200)
        assert bot.q_size > 0

    def test_model_records_observed_transitions(self, bot):
        maze = generate_maze(seed=1)
        run_attempt(bot, maze, max_steps=200)
        assert len(bot.model) > 0

    def test_planning_sweeps_update_more_than_the_visited_states(self):
        """Dyna's whole point: replaying the learned model spreads value faster."""
        maze = generate_maze(seed=1)
        planner = get_bot("dynaq", seed=0, planning_steps=40)
        no_planner = get_bot("dynaq", seed=0, planning_steps=0)
        run_attempt(planner, maze, max_steps=400)
        run_attempt(no_planner, maze, max_steps=400)
        assert planner.total_updates > no_planner.total_updates

    def test_values_propagate_back_from_the_goal(self, bot):
        maze = generate_maze(seed=1)
        run_attempt(bot, maze)
        goal_adjacent = max(bot.q.values())
        assert goal_adjacent > 0, "reaching the goal must create positive value"


class TestSolving:
    @pytest.mark.parametrize("seed", range(6))
    def test_solves_the_maze_within_the_budget(self, seed):
        maze = generate_maze(seed=seed)
        result = run_attempt(get_bot("dynaq", seed=0), maze, seed=seed)
        assert result.solved

    def test_improves_over_repeated_runs(self):
        """An exploring learner is noisy run to run, but it must end up faster."""
        maze = generate_maze(seed=3)
        result = run_attempt(get_bot("dynaq", seed=0), maze)
        assert len(result.runs) >= 2
        assert result.best_run_time < result.runs[0] / 2

    def test_retires_rather_than_burning_search_penalty(self):
        maze = generate_maze(seed=3)
        result = run_attempt(get_bot("dynaq", seed=0), maze, budget_s=600.0)
        assert result.maze_time < 600.0

    def test_never_collides_because_it_senses_before_moving(self):
        maze = generate_maze(seed=3)
        result = run_attempt(get_bot("dynaq", seed=0), maze)
        assert result.collisions == 0


class TestReproducibility:
    def test_same_seed_gives_the_same_attempt(self):
        maze = generate_maze(seed=4)
        a = run_attempt(get_bot("dynaq", seed=7), maze)
        b = run_attempt(get_bot("dynaq", seed=7), maze)
        assert a.score == b.score

    def test_different_seeds_explore_differently(self):
        maze = generate_maze(seed=4)
        a = run_attempt(get_bot("dynaq", seed=1), maze)
        b = run_attempt(get_bot("dynaq", seed=2), maze)
        assert a.trace != b.trace


class TestExploration:
    def test_epsilon_decays_with_experience(self, bot):
        maze = generate_maze(seed=1)
        start_eps = bot.epsilon
        run_attempt(bot, maze, max_steps=500)
        assert bot.epsilon < start_eps
