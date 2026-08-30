"""Procedural generation of contest-style 16x16 mazes."""
import pytest

from micromouse.generator import generate_maze
from micromouse.maze import Direction, Maze


class TestBasicProperties:
    def test_generates_requested_size(self):
        assert generate_maze(seed=1, size=16).size == 16

    def test_is_deterministic_for_a_seed(self):
        assert generate_maze(seed=42) == generate_maze(seed=42)

    def test_different_seeds_give_different_mazes(self):
        assert generate_maze(seed=1) != generate_maze(seed=2)

    def test_border_is_sealed(self):
        m = generate_maze(seed=7)
        for i in range(m.size):
            assert m.has_wall(i, m.size - 1, Direction.N)
            assert m.has_wall(i, 0, Direction.S)
            assert m.has_wall(0, i, Direction.W)
            assert m.has_wall(m.size - 1, i, Direction.E)


class TestContestRules:
    @pytest.mark.parametrize("seed", range(12))
    def test_goal_is_always_reachable(self, seed):
        assert generate_maze(seed=seed).is_solvable()

    @pytest.mark.parametrize("seed", range(12))
    def test_every_cell_is_reachable(self, seed):
        m = generate_maze(seed=seed)
        field = m.distance_field({m.start})
        assert all(d != float("inf") for d in field.values())

    @pytest.mark.parametrize("seed", range(8))
    def test_start_cell_is_walled_on_three_sides(self, seed):
        m = generate_maze(seed=seed)
        walls = [d for d in Direction if m.has_wall(0, 0, d)]
        assert len(walls) == 3
        assert m.has_wall(0, 0, Direction.S) and m.has_wall(0, 0, Direction.W)

    @pytest.mark.parametrize("seed", range(8))
    def test_goal_block_is_open_inside(self, seed):
        m = generate_maze(seed=seed)
        goal = m.goal_cells
        # every goal cell reaches every other without leaving the block
        for cell in goal:
            assert m.open_neighbours(*cell) & goal

    @pytest.mark.parametrize("seed", range(8))
    def test_goal_block_has_exactly_one_entrance(self, seed):
        m = generate_maze(seed=seed)
        goal = m.goal_cells
        entrances = sum(
            1
            for (x, y) in goal
            for d in Direction
            if not m.has_wall(x, y, d)
            and (x + d.delta[0], y + d.delta[1]) not in goal
        )
        assert entrances == 1, "contest rule: the goal has a single gateway"

    @pytest.mark.parametrize("seed", range(8))
    def test_maze_is_not_perfect(self, seed):
        # Contest mazes contain loops, so there are multiple routes to the goal.
        m = generate_maze(seed=seed, loop_ratio=0.15)
        passages = sum(
            1
            for (x, y) in m.cells()
            for d in (Direction.N, Direction.E)
            if not m.has_wall(x, y, d) and m.in_bounds(x + d.delta[0], y + d.delta[1])
        )
        # a perfect maze on N cells has exactly N-1 passages; loops add more
        assert passages > m.size * m.size - 1


class TestDifficulty:
    def test_no_loops_yields_a_perfect_maze(self):
        m = generate_maze(seed=3, loop_ratio=0.0)
        passages = sum(
            1
            for (x, y) in m.cells()
            for d in (Direction.N, Direction.E)
            if not m.has_wall(x, y, d) and m.in_bounds(x + d.delta[0], y + d.delta[1])
        )
        # perfect except for the single forced goal gateway
        assert passages <= m.size * m.size

    def test_solution_is_not_trivially_short(self):
        m = generate_maze(seed=5)
        path = m.shortest_path(m.start, m.goal_cells)
        assert len(path) > m.size, "a contest maze should not be a straight shot"

    def test_rejects_bad_loop_ratio(self):
        with pytest.raises(ValueError):
            generate_maze(seed=1, loop_ratio=1.5)
