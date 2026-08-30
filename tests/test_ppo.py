"""Recurrent PPO: deep RL with memory, the third competition style."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from micromouse.env import OBS_DIM, MicromouseEnv
from micromouse.rl.ppo import ActorCriticLSTM, PPOConfig, compute_gae, train_ppo


class TestNetwork:
    def test_forward_shapes(self):
        net = ActorCriticLSTM(obs_dim=OBS_DIM, n_actions=4, hidden=32)
        obs = torch.zeros(5, 2, OBS_DIM)  # (time, batch, features)
        logits, values, state = net(obs, net.initial_state(batch=2))
        assert logits.shape == (5, 2, 4)
        assert values.shape == (5, 2)
        assert state[0].shape[1] == 2

    def test_memory_changes_the_output_for_identical_observations(self):
        """Without working memory the bot cannot tell two identical corridors apart."""
        torch.manual_seed(0)
        net = ActorCriticLSTM(obs_dim=OBS_DIM, n_actions=4, hidden=32)
        obs = torch.randn(1, 1, OBS_DIM)
        first, _, state = net(obs, net.initial_state(batch=1))
        second, _, _ = net(obs, state)
        assert not torch.allclose(first, second)

    def test_action_masking_never_selects_a_walled_direction(self):
        torch.manual_seed(0)
        net = ActorCriticLSTM(obs_dim=OBS_DIM, n_actions=4, hidden=32)
        obs = torch.randn(1, 1, OBS_DIM)
        mask = torch.tensor([[[True, False, False, False]]])  # only north is legal
        with torch.no_grad():
            logits, _, _ = net(obs, net.initial_state(batch=1), action_mask=mask)
        probs = torch.softmax(logits, dim=-1)
        assert probs[0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert probs[0, 0, 1:].sum().item() == pytest.approx(0.0, abs=1e-6)


class TestGAE:
    def test_zero_rewards_give_zero_advantage(self):
        adv, ret = compute_gae(
            rewards=np.zeros(4, dtype=np.float32),
            values=np.zeros(5, dtype=np.float32),
            dones=np.zeros(4, dtype=np.float32),
            gamma=0.99,
            lam=0.95,
        )
        assert np.allclose(adv, 0.0)

    def test_a_terminal_reward_propagates_backwards_with_discount(self):
        adv, ret = compute_gae(
            rewards=np.array([0, 0, 1], dtype=np.float32),
            values=np.zeros(4, dtype=np.float32),
            dones=np.array([0, 0, 1], dtype=np.float32),
            gamma=0.9,
            lam=1.0,
        )
        assert adv[2] == pytest.approx(1.0)
        assert adv[1] == pytest.approx(0.9)
        assert adv[0] == pytest.approx(0.81)

    def test_done_flags_cut_the_bootstrap(self):
        adv, _ = compute_gae(
            rewards=np.array([0, 0], dtype=np.float32),
            values=np.array([0, 0, 10], dtype=np.float32),
            dones=np.array([1, 0], dtype=np.float32),
            gamma=0.9,
            lam=1.0,
        )
        assert adv[0] == pytest.approx(0.0), "a finished episode must not bootstrap"

    def test_returns_equal_advantages_plus_values(self):
        rewards = np.array([1, 2, 3], dtype=np.float32)
        values = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        dones = np.zeros(3, dtype=np.float32)
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
        assert np.allclose(ret, adv + values[:-1])


class TestTraining:
    def test_short_training_run_produces_a_policy_and_stats(self, tmp_path):
        cfg = PPOConfig(total_steps=512, rollout_steps=128, epochs=1, hidden=32, size=6)
        net, stats = train_ppo(cfg, maze_seeds=[0, 1], progress=False)
        assert isinstance(net, ActorCriticLSTM)
        assert stats["updates"] >= 1
        assert "policy_loss" in stats and np.isfinite(stats["policy_loss"])

    def test_checkpoint_roundtrip(self, tmp_path):
        cfg = PPOConfig(total_steps=256, rollout_steps=128, epochs=1, hidden=32, size=6)
        net, _ = train_ppo(cfg, maze_seeds=[0], progress=False)
        path = tmp_path / "ppo.pt"
        net.save(path, config=cfg)
        loaded = ActorCriticLSTM.load(path)
        obs = torch.randn(1, 1, OBS_DIM)
        a, _, _ = net(obs, net.initial_state(1))
        b, _, _ = loaded(obs, loaded.initial_state(1))
        assert torch.allclose(a, b)


class TestPPOBot:
    def test_runs_legally_without_a_trained_checkpoint(self):
        from micromouse.bots import get_bot
        from micromouse.generator import generate_maze
        from micromouse.runner import run_attempt

        bot = get_bot("ppo", checkpoint=None, seed=0)
        assert bot.style == "deep-rl"
        result = run_attempt(bot, generate_maze(seed=1), max_steps=300)
        assert result.collisions == 0, "wall-masking makes collisions impossible"
        assert result.steps > 0

    def test_is_deterministic_when_greedy(self):
        from micromouse.bots import get_bot
        from micromouse.generator import generate_maze
        from micromouse.runner import run_attempt

        maze = generate_maze(seed=1)
        a = run_attempt(get_bot("ppo", checkpoint=None, seed=3, greedy=True), maze, max_steps=200)
        b = run_attempt(get_bot("ppo", checkpoint=None, seed=3, greedy=True), maze, max_steps=200)
        assert a.trace == b.trace
