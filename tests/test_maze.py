"""Maze data structure: wall encoding, symmetry, geometry, serialisation."""
import pytest

from micromouse.maze import Direction, Maze


class TestDirection:
    def test_bit_encoding_matches_maz_standard(self):
        # Standard micromouse .maz encoding: N=1, E=2, S=4, W=8
        assert Direction.N.bit == 1
        assert Direction.E.bit == 2
        assert Direction.S.bit == 4
        assert Direction.W.bit == 8

    def test_deltas(self):
        # x grows east, y grows north; (0,0) is the south-west corner
        assert Direction.N.delta == (0, 1)
        assert Direction.E.delta == (1, 0)
        assert Direction.S.delta == (0, -1)
        assert Direction.W.delta == (-1, 0)

    def test_opposite(self):
        assert Direction.N.opposite is Direction.S
        assert Direction.W.opposite is Direction.E

    def test_turn_arithmetic(self):
        assert Direction.N.right() is Direction.E
        assert Direction.N.left() is Direction.W
        assert Direction.W.right() is Direction.N

    def test_relative_turn_cost_in_quarter_turns(self):
        assert Direction.N.turns_to(Direction.N) == 0
        assert Direction.N.turns_to(Direction.E) == 1
        assert Direction.N.turns_to(Direction.W) == 1
        assert Direction.N.turns_to(Direction.S) == 2


class TestMazeConstruction:
    def test_default_is_16x16(self):
        assert Maze().size == 16

    def test_empty_maze_has_only_border_walls(self):
        m = Maze(size=4)
        assert m.has_wall(0, 0, Direction.S)  # south border
        assert m.has_wall(0, 0, Direction.W)  # west border
        assert not m.has_wall(0, 0, Direction.N)  # interior
        assert not m.has_wall(0, 0, Direction.E)
        assert m.has_wall(3, 3, Direction.N)
        assert m.has_wall(3, 3, Direction.E)

    def test_rejects_bad_size(self):
        with pytest.raises(ValueError):
            Maze(size=1)

    def test_out_of_bounds_is_walled(self):
        m = Maze(size=4)
        assert not m.in_bounds(-1, 0)
        assert not m.in_bounds(4, 0)
        assert m.in_bounds(3, 3)


class TestWallSymmetry:
    def test_setting_a_wall_sets_it_from_both_sides(self):
        m = Maze(size=4)
        m.set_wall(1, 1, Direction.N)
        assert m.has_wall(1, 1, Direction.N)
        assert m.has_wall(1, 2, Direction.S), "internal walls are shared by both cells"

    def test_clearing_a_wall_clears_it_from_both_sides(self):
        m = Maze(size=4)
        m.set_wall(1, 1, Direction.E)
        m.clear_wall(2, 1, Direction.W)
        assert not m.has_wall(1, 1, Direction.E)
        assert not m.has_wall(2, 1, Direction.W)

    def test_border_walls_cannot_be_cleared(self):
        m = Maze(size=4)
        m.clear_wall(0, 0, Direction.S)
        assert m.has_wall(0, 0, Direction.S), "outer boundary must stay sealed"


class TestGeometry:
    def test_start_cell_is_south_west_corner(self):
        assert Maze(size=16).start == (0, 0)

    def test_goal_is_the_four_centre_cells(self):
        assert Maze(size=16).goal_cells == {(7, 7), (8, 7), (7, 8), (8, 8)}

    def test_odd_size_goal_is_single_centre_cell(self):
        assert Maze(size=5).goal_cells == {(2, 2)}

    def test_open_neighbours_respect_walls(self):
        m = Maze(size=4)
        assert m.open_neighbours(0, 0) == {(1, 0), (0, 1)}
        m.set_wall(0, 0, Direction.N)
        assert m.open_neighbours(0, 0) == {(1, 0)}


class TestDistances:
    def test_distance_field_on_open_maze(self):
        m = Maze(size=4)
        d = m.distance_field({(0, 0)})
        assert d[(0, 0)] == 0
        assert d[(3, 3)] == 6  # manhattan on a fully open grid

    def test_unreachable_cells_are_infinite(self):
        m = Maze(size=4)
        for d in (Direction.N, Direction.E):
            m.set_wall(0, 0, d)
        field = m.distance_field({(0, 0)})
        assert field[(3, 3)] == float("inf")

    def test_shortest_path_follows_the_distance_field(self):
        m = Maze(size=4)
        path = m.shortest_path((0, 0), {(3, 3)})
        assert path[0] == (0, 0)
        assert path[-1] == (3, 3)
        assert len(path) == 7

    def test_no_path_returns_none(self):
        m = Maze(size=4)
        for d in (Direction.N, Direction.E):
            m.set_wall(0, 0, d)
        assert m.shortest_path((0, 0), {(3, 3)}) is None

    def test_is_solvable(self):
        m = Maze(size=16)
        assert m.is_solvable()
        for d in Direction:
            m.set_wall(0, 0, d)
        assert not m.is_solvable()


class TestSerialisation:
    def test_maz_bytes_roundtrip(self):
        m = Maze(size=16)
        m.set_wall(3, 4, Direction.N)
        m.set_wall(9, 2, Direction.W)
        blob = m.to_maz_bytes()
        assert len(blob) == 256
        assert Maze.from_maz_bytes(blob) == m

    def test_maz_start_cell_value_is_14(self):
        # Classic: start cell walled N/A -> open north, walls E,S,W = 2+4+8 = 14
        m = Maze(size=16)
        m.set_wall(0, 0, Direction.E)
        assert m.to_maz_bytes()[0] == 14

    def test_text_roundtrip(self):
        m = Maze(size=8)
        m.set_wall(2, 3, Direction.E)
        m.set_wall(5, 5, Direction.S)
        assert Maze.from_text(m.to_text()) == m

    def test_text_render_uses_posts_and_dashes(self):
        text = Maze(size=4).to_text()
        lines = text.splitlines()
        assert len(lines) == 9  # 2*size + 1
        assert lines[0] == "o---o---o---o---o"
        assert lines[-1] == "o---o---o---o---o"
        assert lines[1].startswith("|") and lines[1].endswith("|")

    def test_equality_ignores_identity(self):
        assert Maze(size=4) == Maze(size=4)
        assert Maze(size=4) != Maze(size=8)
