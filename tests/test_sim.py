"""Mouse simulation: partial observability, movement, collisions, timing."""
import pytest

from micromouse.maze import Direction, Maze
from micromouse.sim import Motion, MouseSim


@pytest.fixture
def open_maze():
    return Maze(size=4)


class TestReset:
    def test_starts_at_the_start_cell_facing_north(self, open_maze):
        sim = MouseSim(open_maze)
        obs = sim.reset()
        assert obs.position == (0, 0)
        assert obs.heading is Direction.N
        assert obs.time_s == 0.0
        assert sim.steps == 0

    def test_reset_clears_previous_state(self, open_maze):
        sim = MouseSim(open_maze)
        sim.reset()
        sim.step(Direction.N)
        sim.reset()
        assert sim.steps == 0
        assert sim.elapsed == 0.0


class TestPartialObservability:
    def test_observation_only_exposes_the_current_cell_walls(self, open_maze):
        obs = MouseSim(open_maze).reset()
        assert obs.walls[Direction.S] is True   # south border
        assert obs.walls[Direction.W] is True   # west border
        assert obs.walls[Direction.N] is False
        assert obs.walls[Direction.E] is False

    def test_observation_does_not_leak_the_maze(self, open_maze):
        obs = MouseSim(open_maze).reset()
        assert not hasattr(obs, "maze")
        assert set(vars(obs)) == {
            "position", "heading", "walls", "time_s", "step", "size", "goal_cells",
        }

    def test_bot_knows_the_static_contest_geometry(self, open_maze):
        obs = MouseSim(open_maze).reset()
        assert obs.size == 4
        assert obs.goal_cells == open_maze.goal_cells


class TestMovement:
    def test_moving_into_open_space_updates_position_and_heading(self, open_maze):
        sim = MouseSim(open_maze)
        sim.reset()
        obs = sim.step(Direction.E)
        assert obs.position == (1, 0)
        assert obs.heading is Direction.E

    def test_moving_into_a_wall_is_a_collision_and_does_not_move(self, open_maze):
        sim = MouseSim(open_maze)
        sim.reset()
        obs = sim.step(Direction.S)
        assert obs.position == (0, 0)
        assert sim.collisions == 1

    def test_collision_still_costs_time(self, open_maze):
        sim = MouseSim(open_maze)
        sim.reset()
        sim.step(Direction.S)
        assert sim.elapsed > 0

    def test_steps_increment(self, open_maze):
        sim = MouseSim(open_maze)
        sim.reset()
        sim.step(Direction.N)
        sim.step(Direction.N)
        assert sim.steps == 2


class TestTiming:
    def test_straight_move_costs_one_cell_time(self, open_maze):
        motion = Motion(cell_time=0.1, turn_time=0.2, collision_time=0.5)
        sim = MouseSim(open_maze, motion=motion)
        sim.reset()  # facing north
        sim.step(Direction.N)
        assert sim.elapsed == pytest.approx(0.1)

    def test_quarter_turn_adds_one_turn_time(self, open_maze):
        motion = Motion(cell_time=0.1, turn_time=0.2, collision_time=0.5)
        sim = MouseSim(open_maze, motion=motion)
        sim.reset()  # facing north
        sim.step(Direction.E)  # one 90-degree turn, then forward
        assert sim.elapsed == pytest.approx(0.3)

    def test_reversal_costs_two_turns(self, open_maze):
        motion = Motion(cell_time=0.1, turn_time=0.2, collision_time=0.5)
        sim = MouseSim(open_maze, motion=motion)
        sim.reset()
        sim.step(Direction.N)
        sim.step(Direction.S)
        assert sim.elapsed == pytest.approx(0.1 + 0.4 + 0.1)

    def test_collision_charges_collision_time(self, open_maze):
        motion = Motion(cell_time=0.1, turn_time=0.2, collision_time=0.5)
        sim = MouseSim(open_maze, motion=motion)
        sim.reset()
        sim.step(Direction.S)  # 2 turns to face south, then hit the wall
        assert sim.elapsed == pytest.approx(0.4 + 0.5)

    def test_straight_corridors_are_cheaper_than_zigzags(self, open_maze):
        motion = Motion(cell_time=0.1, turn_time=0.2)
        straight = MouseSim(open_maze, motion=motion)
        straight.reset()
        for _ in range(3):
            straight.step(Direction.N)

        zigzag = MouseSim(open_maze, motion=motion)
        zigzag.reset()
        for d in (Direction.N, Direction.E, Direction.N, Direction.E):
            zigzag.step(d)

        assert straight.elapsed < zigzag.elapsed


class TestGoalTracking:
    def test_at_goal_detects_the_centre(self):
        maze = Maze(size=4)
        sim = MouseSim(maze)
        sim.reset()
        assert not sim.at_goal
        for d in (Direction.N, Direction.E):
            sim.step(d)
        assert sim.position == (1, 1)
        assert sim.at_goal

    def test_trace_records_every_pose(self, open_maze):
        sim = MouseSim(open_maze)
        sim.reset()
        sim.step(Direction.N)
        sim.step(Direction.E)
        assert [p.position for p in sim.trace] == [(0, 0), (0, 1), (1, 1)]
        assert sim.trace[-1].time_s > 0


class TestKnownWalls:
    def test_sim_reports_the_cells_the_mouse_has_visited(self, open_maze):
        sim = MouseSim(open_maze)
        sim.reset()
        sim.step(Direction.N)
        assert sim.visited == {(0, 0), (0, 1)}
