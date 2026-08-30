"""Train the recurrent PPO policy and write checkpoints/ppo.pt."""
import argparse

from micromouse.rl.ppo import PPOConfig, train_ppo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--mazes", type=int, default=64)
    ap.add_argument("--out", default="checkpoints/ppo.pt")
    args = ap.parse_args()

    cfg = PPOConfig(total_steps=args.steps, size=args.size, hidden=args.hidden)
    print(f"training on {args.mazes} mazes of {args.size}x{args.size} for {args.steps} steps")
    net, stats = train_ppo(
        cfg, maze_seeds=list(range(args.mazes)), progress=True, checkpoint=args.out
    )
    print("done:", {k: round(v, 4) if isinstance(v, float) else v for k, v in stats.items()})


if __name__ == "__main__":
    main()
