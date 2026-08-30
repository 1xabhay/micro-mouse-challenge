"""Command line entry point: ``micromouse <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bots import bot_info, get_bot
from .generator import generate_maze
from .runner import run_attempt
from .tournament import run_tournament

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="micromouse",
        description="Micromouse 16x16 maze challenge: simulate, race and train mice.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="list the available bots")

    maze = sub.add_parser("maze", help="print a generated maze")
    maze.add_argument("--seed", type=int, default=0)
    maze.add_argument("--size", type=int, default=16)

    run_cmd = sub.add_parser("run", help="run one bot on one maze")
    run_cmd.add_argument("--bot", default="floodfill")
    run_cmd.add_argument("--seed", type=int, default=0)
    run_cmd.add_argument("--size", type=int, default=16)
    run_cmd.add_argument("--json", action="store_true")

    race = sub.add_parser("race", help="race every bot over a pool of mazes")
    race.add_argument("--mazes", type=int, default=20)
    race.add_argument("--bots", nargs="*", default=None)
    race.add_argument("--size", type=int, default=16)
    race.add_argument("--json", action="store_true")
    race.add_argument("--out", type=Path, default=None)
    race.add_argument("--quiet", action="store_true")

    train = sub.add_parser("train", help="train a learning bot")
    train.add_argument("--bot", default="ppo")
    train.add_argument("--steps", type=int, default=300_000)
    train.add_argument("--size", type=int, default=16)
    train.add_argument("--mazes", type=int, default=64)
    train.add_argument("--out", default="checkpoints/ppo.pt")

    dash = sub.add_parser("dash", help="serve the dashboard")
    dash.add_argument("--host", default="127.0.0.1")
    dash.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    return {
        "list": _cmd_list,
        "maze": _cmd_maze,
        "run": _cmd_run,
        "race": _cmd_race,
        "train": _cmd_train,
        "dash": _cmd_dash,
    }[args.command](args)


# --------------------------------------------------------------------- commands


def _cmd_list(args) -> int:
    for info in bot_info():
        print(f"{info['name']:<12} {info['style']:<12} {info['description']}")
    return 0


def _cmd_maze(args) -> int:
    print(generate_maze(seed=args.seed, size=args.size).to_text())
    return 0


def _cmd_run(args) -> int:
    try:
        bot = get_bot(args.bot)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2

    maze = generate_maze(seed=args.seed, size=args.size)
    result = run_attempt(bot, maze, seed=args.seed)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"{result.bot} on maze {args.seed}")
    print(f"  solved      {result.solved}")
    print(f"  runs        {[round(r, 2) for r in result.runs]}")
    print(f"  best run    {_fmt(result.best_run_time)}")
    print(f"  search      {_fmt(result.search_time)}")
    print(f"  maze time   {result.maze_time:.2f}s")
    print(f"  score       {result.score:.2f}")
    print(f"  coverage    {result.coverage:.0%}   collisions {result.collisions}")
    return 0


def _cmd_race(args) -> int:
    bots = args.bots or [info["name"] for info in bot_info()]
    seeds = list(range(args.mazes))

    def report(result):
        if not args.quiet:
            print(
                f"  {result.bot:<10} maze {result.seed:<3} "
                f"score {result.score:8.2f}  "
                f"{'solved' if result.solved else 'FAILED'}",
                flush=True,
            )

    if not args.quiet:
        print(f"racing {', '.join(bots)} over {len(seeds)} mazes")
    result = run_tournament(bots=bots, seeds=seeds, size=args.size, on_attempt=report)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.to_dict(), indent=2))

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print()
    print(_HEADER)
    print("-" * len(_HEADER))
    for row in result.leaderboard():
        print(_leaderboard_row(row))
    return 0


_HEADER = (
    f"{'bot':<12}{'style':<12}{'score':>9}{'solved':>9}"
    f"{'wins':>6}{'best run':>10}{'search':>9}{'cover':>7}"
)


def _leaderboard_row(row: dict) -> str:
    solved = f"{row['solved']}/{row['attempts']}"
    return (
        f"{row['bot']:<12}{row['style']:<12}{_fmt(row['mean_score']):>9}{solved:>9}"
        f"{row['wins']:>6}{_fmt(row['mean_run']):>10}"
        f"{_fmt(row['mean_search']):>9}{row['mean_coverage']:>7.0%}"
    )


def _cmd_train(args) -> int:
    if args.bot != "ppo":
        print(
            f"{args.bot!r} learns online during its attempt and needs no "
            "pretraining; only 'ppo' can be trained.",
            file=sys.stderr,
        )
        return 2

    from .rl.ppo import PPOConfig, train_ppo

    cfg = PPOConfig(total_steps=args.steps, size=args.size)
    print(f"training ppo: {args.steps} steps on {args.mazes} mazes")
    _, stats = train_ppo(
        cfg, maze_seeds=list(range(args.mazes)), progress=True, checkpoint=args.out
    )
    print(f"saved {args.out}")
    print(f"  success rate {stats['success_rate']:.0%} over {stats['episodes']} episodes")
    return 0


def _cmd_dash(args) -> int:
    from .server.app import serve

    serve(host=args.host, port=args.port)
    return 0


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
