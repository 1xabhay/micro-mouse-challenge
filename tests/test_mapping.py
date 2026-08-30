"""The belief map a bot builds from partial observations."""
import pytest

from micromouse.generator import generate_maze
from micromouse.mapping import BeliefMap
from micromouse.maze import Direction, Maze
from micromouse.sim import MouseSim


@pytest.fixture
def sim():
    return MouseSim(generate_maze(seed=4))


class TestLearning:
    def test_starts_optimistic_with_no_interior_walls_known(self):
        bm = BeliefMap(size=16, goal_cells={(7, 7)})
        assert not bm.is_known((5, 5))
        assert bm.known_cells == 0

    def test_observing_records_the_walls_of_the_current_cell(self, sim):
        obs = sim.reset()
        bm = BeliefMap(size=obs.size, goal_cells=obs.goal_cells)
        bm.observe(obs)
        assert bm.is_known((0, 0))
        assert bm.known_cells == 1
        for d in Direction:
            assert bm.has_wall(0, 0, d) == obs.walls[d]

    def test_learned_walls_are_shared_with_the_neighbouring_cell(self, sim):
        obs = sim.reset()
        bm = BeliefMap(size=obs.size, goal_cells=obs.goal_cells)
        bm.observe(obs)
        # the start cell's north side is open, so (0,1) must agree
        assert bm.has_wall(0, 1, Direction.S) is False

    def test_after_full_exploration_the_belief_matches_the_real_maze(self):
        maze = generate_maze(seed=6, size=6)
        bm = BeliefMap(size=6, goal_cells=maze.goal_cells)
        sim = MouseSim(maze)
        obs = sim.reset()
        for cell in maze.cells():  # teleport-observe every cell
            sim.position = cell
            bm.observe(sim.observe())
        assert bm.as_maze() == maze


class TestPlanning:
    def test_unknown_cells_are_assumed_open(self):
        bm = BeliefMap(size=4, goal_cells={(3, 3)})
        assert bm.distance_to_targets()[(0, 0)] == 6

    def test_next_step_moves_down_the_distance_gradient(self):
        bm = BeliefMap(size=4, goal_cells={(3, 3)})
        d = bm.next_step((0, 0), Direction.N, bm.goal_cells)
        assert d in (Direction.N, Direction.E)

    def test_next_step_prefers_carrying_straight_on(self):
        bm = BeliefMap(size=4, goal_cells={(3, 3)})
        # both N and E make equal progress; heading north should stay north
        assert bm.next_step((0, 0), Direction.N, bm.goal_cells) is Direction.N
        assert bm.next_step((0, 0), Direction.E, bm.goal_cells) is Direction.E

    def test_next_step_returns_none_when_boxed_in(self):
        bm = BeliefMap(size=4, goal_cells={(3, 3)})
        for d in Direction:
            bm.set_wall(0, 0, d)
        assert bm.next_step((0, 0), Direction.N, bm.goal_cells) is None


class TestFastestRoute:
    def test_route_reaches_the_target(self):
        maze = generate_maze(seed=8)
        bm = BeliefMap.from_maze(maze)
        route = bm.fastest_route(maze.start, Direction.N, maze.goal_cells)
        x, y = maze.start
        for d in route:
            assert not maze.has_wall(x, y, d)
            x, y = x + d.delta[0], y + d.delta[1]
        assert (x, y) in maze.goal_cells

    def test_turn_aware_route_beats_a_plain_cell_count_route(self):
        """With turns priced in, the fastest route is not always the shortest."""
        maze = Maze(size=5)  # fully open
        bm = BeliefMap.from_maze(maze)
        route = bm.fastest_route((0, 0), Direction.N, {(4, 4)}, turn_time=5.0)
        turns = sum(a.turns_to(b) for a, b in zip(route, route[1:]))
        assert turns == 1, "an open grid should be crossed in a single L, not a stair"

    def test_route_is_empty_when_already_at_the_target(self):
        maze = generate_maze(seed=8)
        bm = BeliefMap.from_maze(maze)
        assert bm.fastest_route((7, 7), Direction.N, {(7, 7)}) == []

    def test_no_route_returns_none(self):
        maze = Maze(size=4)
        bm = BeliefMap.from_maze(maze)
        for d in Direction:
            bm.set_wall(0, 0, d)
        assert bm.fastest_route((0, 0), Direction.N, {(3, 3)}) is None


class TestFrontier:
    def test_unvisited_reachable_cells_are_frontier(self, sim):
        obs = sim.reset()
        bm = BeliefMap(size=obs.size, goal_cells=obs.goal_cells)
        bm.observe(obs)
        assert (0, 1) in bm.frontier()
        assert (0, 0) not in bm.frontier()
