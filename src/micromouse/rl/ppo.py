"""Recurrent PPO — the deep-RL competitor.

Two choices here follow directly from what the literature says about navigation
under partial observability:

* **Recurrence.** The policy is an LSTM. A mouse that can only see the walls of
  its current cell cannot distinguish two identical-looking corridors; memory is
  what breaks that ambiguity, and recurrent agents consistently beat feedforward
  ones on maze POMDPs.
* **Action masking.** Walls are sensed, so driving into one is never a decision
  worth learning. Masking illegal directions removes that entire failure mode
  from the search space and lets the policy spend its capacity on routing.

Trained across a pool of mazes so the policy has to generalise rather than
memorise one layout.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..env import OBS_DIM, MicromouseEnv

__all__ = ["ActorCriticLSTM", "PPOConfig", "compute_gae", "train_ppo"]


@dataclass
class PPOConfig:
    total_steps: int = 400_000
    rollout_steps: int = 512
    epochs: int = 4
    minibatches: int = 4
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    size: int = 16
    max_episode_steps: int = 1_200
    seed: int = 0


class ActorCriticLSTM(nn.Module):
    """Shared encoder, LSTM memory, then a policy head and a value head."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = 4, hidden: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden = hidden

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(hidden, hidden)
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

        for layer, gain in ((self.policy, 0.01), (self.value, 1.0)):
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)

    def initial_state(self, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(1, batch, self.hidden)
        return (zeros, zeros.clone())

    def forward(
        self,
        obs: torch.Tensor,                      # (time, batch, obs_dim)
        state: tuple[torch.Tensor, torch.Tensor],
        action_mask: torch.Tensor | None = None,  # (time, batch, n_actions) bool
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        features = self.encoder(obs)
        out, state = self.lstm(features, state)
        logits = self.policy(out)
        if action_mask is not None:
            # -inf on illegal actions: softmax then assigns them exactly zero.
            logits = logits.masked_fill(~action_mask, float("-inf"))
        return logits, self.value(out).squeeze(-1), state

    # ----------------------------------------------------------- persistence

    def save(self, path: str | Path, config: PPOConfig | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "hidden": self.hidden,
                "config": asdict(config) if config else None,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ActorCriticLSTM":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net = cls(
            obs_dim=blob["obs_dim"], n_actions=blob["n_actions"], hidden=blob["hidden"]
        )
        net.load_state_dict(blob["state_dict"])
        net.eval()
        return net


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,   # length len(rewards) + 1, the last is the bootstrap
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in reversed(range(n)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        running = delta + gamma * lam * not_done * running
        advantages[t] = running
    return advantages, advantages + values[:n]


def _action_mask(env: MicromouseEnv) -> np.ndarray:
    obs = env.sim.observe()
    return np.array([not obs.walls[d] for d in env.DIRECTIONS], dtype=bool)


def train_ppo(
    cfg: PPOConfig,
    maze_seeds: Sequence[int] | None = None,
    progress: bool = True,
    checkpoint: str | Path | None = None,
) -> tuple[ActorCriticLSTM, dict]:
    """Train a recurrent policy and return it with summary statistics."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = MicromouseEnv(
        size=cfg.size,
        maze_seeds=list(maze_seeds) if maze_seeds is not None else list(range(64)),
        max_steps=cfg.max_episode_steps,
    )
    net = ActorCriticLSTM(obs_dim=OBS_DIM, n_actions=4, hidden=cfg.hidden)
    optimiser = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
    state = net.initial_state(batch=1)
    episode_return, episode_len = 0.0, 0
    returns_log: list[float] = []
    success_log: list[float] = []
    stats: dict = {"updates": 0}
    started = time.time()

    steps_done = 0
    while steps_done < cfg.total_steps:
        obs_buf = np.zeros((cfg.rollout_steps, OBS_DIM), dtype=np.float32)
        mask_buf = np.zeros((cfg.rollout_steps, 4), dtype=bool)
        act_buf = np.zeros(cfg.rollout_steps, dtype=np.int64)
        logp_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        rew_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        done_buf = np.zeros(cfg.rollout_steps, dtype=np.float32)
        val_buf = np.zeros(cfg.rollout_steps + 1, dtype=np.float32)
        # The LSTM state at the start of each step, so the update can replay
        # the rollout with the same memory the actor had.
        start_state = (state[0].detach().clone(), state[1].detach().clone())

        for t in range(cfg.rollout_steps):
            mask = _action_mask(env)
            obs_buf[t], mask_buf[t] = obs, mask

            with torch.no_grad():
                logits, value, state = net(
                    torch.as_tensor(obs).view(1, 1, -1),
                    state,
                    torch.as_tensor(mask).view(1, 1, -1),
                )
                dist = Categorical(logits=logits[0, 0])
                action = dist.sample()

            val_buf[t] = value.item()
            act_buf[t] = int(action)
            logp_buf[t] = dist.log_prob(action).item()

            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            rew_buf[t], done_buf[t] = reward, float(done)
            episode_return += reward
            episode_len += 1

            if done:
                returns_log.append(episode_return)
                success_log.append(1.0 if terminated else 0.0)
                episode_return, episode_len = 0.0, 0
                obs, _ = env.reset(seed=int(rng.integers(1 << 30)))
                state = net.initial_state(batch=1)  # memory does not cross episodes

        with torch.no_grad():
            _, bootstrap, _ = net(
                torch.as_tensor(obs).view(1, 1, -1),
                state,
                torch.as_tensor(_action_mask(env)).view(1, 1, -1),
            )
        val_buf[-1] = bootstrap.item()
        steps_done += cfg.rollout_steps

        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, cfg.gamma, cfg.lam
        )
        stats.update(
            _ppo_update(
                net, optimiser, cfg, obs_buf, mask_buf, act_buf, logp_buf,
                advantages, returns, done_buf, start_state,
            )
        )
        stats["updates"] += 1

        if progress and stats["updates"] % 10 == 0:
            recent = returns_log[-50:]
            print(
                f"  step {steps_done:>7}/{cfg.total_steps}  "
                f"return {np.mean(recent) if recent else float('nan'):7.2f}  "
                f"success {np.mean(success_log[-50:]) if success_log else 0:5.1%}  "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

    stats["mean_return"] = float(np.mean(returns_log[-100:])) if returns_log else float("nan")
    stats["success_rate"] = float(np.mean(success_log[-100:])) if success_log else 0.0
    stats["episodes"] = len(returns_log)
    stats["seconds"] = time.time() - started
    if checkpoint:
        net.save(checkpoint, config=cfg)
    return net, stats


def _ppo_update(
    net: ActorCriticLSTM,
    optimiser: torch.optim.Optimizer,
    cfg: PPOConfig,
    obs_buf, mask_buf, act_buf, logp_buf, advantages, returns, done_buf, start_state,
) -> dict:
    """Clipped-surrogate update over contiguous chunks of the recurrent rollout."""
    obs_t = torch.as_tensor(obs_buf).unsqueeze(1)          # (T, 1, obs)
    mask_t = torch.as_tensor(mask_buf).unsqueeze(1)
    act_t = torch.as_tensor(act_buf).unsqueeze(1)
    old_logp = torch.as_tensor(logp_buf).unsqueeze(1)
    adv_t = torch.as_tensor(advantages).unsqueeze(1)
    ret_t = torch.as_tensor(returns).unsqueeze(1)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    chunk = max(1, cfg.rollout_steps // cfg.minibatches)
    last = {}
    for _ in range(cfg.epochs):
        state = (start_state[0].clone(), start_state[1].clone())
        for begin in range(0, cfg.rollout_steps, chunk):
            end = min(begin + chunk, cfg.rollout_steps)
            logits, values, state = net(obs_t[begin:end], state, mask_t[begin:end])
            state = (state[0].detach(), state[1].detach())  # truncated BPTT

            dist = Categorical(logits=logits)
            logp = dist.log_prob(act_t[begin:end])
            ratio = torch.exp(logp - old_logp[begin:end])

            slice_adv = adv_t[begin:end]
            unclipped = ratio * slice_adv
            clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * slice_adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = ((values - ret_t[begin:end]) ** 2).mean()
            entropy = dist.entropy().mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy
            )
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimiser.step()

            last = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "entropy": entropy.item(),
            }
    return last
