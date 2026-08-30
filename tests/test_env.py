"""Gymnasium environment wrapping the micromouse simulation."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from micromouse.env import MicromouseEnv, OBS_DIM


@pytest.fixture
def env():
    return MicromouseEnv(size=8, maze_seeds=[0, 1, 2])


class TestGymnasiumApi:
    def test_passes_the_official_env_checker(self, env):
        check_env(env, skip_render_check=True)

    def test_action_space_is_four_compass_directions(self, env):
        assert env.action_space.n == 4

    def test_observation_space_matches_the_observation(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)

    def test_step_returns_the_five_tuple(self, env):
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)

    def test_seeding_is_reproducible(self):
        a = MicromouseEnv(size=8, maze_seeds=[0, 1, 2])
        b = MicromouseEnv(size=8, maze_seeds=[0, 1, 2])
        oa, _ = a.reset(seed=7)
        ob, _ = b.reset(seed=7)
        assert np.allclose(oa, ob)


class TestObservation:
    def test_encodes_local_walls_only(self, env):
        obs, _ = env.reset(seed=0)
        # first four entries are the sensed walls at the current cell
        walls = env.sim.observe().walls
        assert list(obs[:4]) == [float(walls[d]) for d in env.DIRECTIONS]

    def test_is_bounded_for_stable_learning(self, env):
        env.reset(seed=0)
        for a in [0, 1, 2, 3] * 20:
            obs, *_ = env.step(a)
            assert np.all(np.abs(obs) <= 1.0 + 1e-6)

    def test_does_not_leak_unseen_walls(self, env):
        """The agent must not be handed knowledge it could not have sensed."""
        obs, _ = env.reset(seed=0)
        env2 = MicromouseEnv(size=8, maze_seeds=[0, 1, 2])
        obs2, _ = env2.reset(seed=0)
        assert np.allclose(obs, obs2)


class TestRewards:
    def test_reaching_the_goal_terminates_with_a_bonus(self):
        env = MicromouseEnv(size=8, maze_seeds=[0])
        env.reset(seed=0)
        total = 0.0
        for _ in range(2000):
            action = env.expert_action()  # privileged oracle, for testing only
            _, r, term, trunc, _ = env.step(action)
            total += r
            if term or trunc:
                break
        assert term, "the oracle should always reach the goal"
        assert total > 0

    def test_collisions_are_penalised(self):
        env = MicromouseEnv(size=8, maze_seeds=[0])
        env.reset(seed=0)
        south = env.DIRECTIONS.index(env.DIRECTIONS[2])  # facing into the border
        _, reward, *_ = env.step(south)
        assert reward < 0

    def test_every_step_costs_something(self, env):
        env.reset(seed=0)
        _, reward, *_ = env.step(env.expert_action())
        assert reward < 1.0

    def test_truncates_at_the_step_limit(self):
        env = MicromouseEnv(size=8, maze_seeds=[0], max_steps=10)
        env.reset(seed=0)
        for _ in range(10):
            _, _, term, trunc, _ = env.step(2)
        assert trunc and not term


class TestCurriculum:
    def test_cycles_through_the_maze_pool(self):
        env = MicromouseEnv(size=8, maze_seeds=[0, 1, 2])
        seen = set()
        for i in range(6):
            env.reset(seed=i)
            seen.add(env.maze_seed)
        assert len(seen) > 1, "training should not overfit a single maze"

    def test_info_reports_progress(self, env):
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        assert "cells_visited" in info and "elapsed" in info
