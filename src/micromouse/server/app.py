"""FastAPI backend for the dashboard.

Runs are executed synchronously and returned whole, trace included: a 16x16
attempt takes well under a second, and shipping the finished trajectory lets the
browser scrub and replay it without holding a socket open.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..bots import bot_info, get_bot
from ..generator import generate_maze
from ..maze import Direction, Maze
from ..runner import run_attempt
from ..tournament import run_tournament

__all__ = ["create_app", "serve"]

STATIC_DIR = Path(__file__).parent / "static"
MAX_RACE_MAZES = 50


class RunRequest(BaseModel):
    bot: str
    seed: int = 0
    size: int = Field(default=16, ge=2, le=32)
    loop_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    max_steps: int = Field(default=20_000, ge=1, le=200_000)
    budget_s: float = Field(default=600.0, gt=0, le=6_000)


class RaceRequest(BaseModel):
    bots: list[str] = Field(min_length=1)
    mazes: int = Field(default=5, ge=1, le=MAX_RACE_MAZES)
    size: int = Field(default=16, ge=2, le=32)
    loop_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


def _maze_payload(maze: Maze, seed: int) -> dict:
    optimal = maze.shortest_path(maze.start, maze.goal_cells)
    return {
        "seed": seed,
        "size": maze.size,
        # walls[y][x] is the cell's bitmask: N=1, E=2, S=4, W=8
        "walls": [[maze.walls_at(x, y) for x in range(maze.size)] for y in range(maze.size)],
        "start": list(maze.start),
        "goal": [list(c) for c in sorted(maze.goal_cells)],
        "optimal_cells": len(optimal) if optimal else None,
        "optimal_path": [list(c) for c in optimal] if optimal else [],
    }


def create_app() -> FastAPI:
    app = FastAPI(title="Micromouse", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/bots")
    def bots() -> dict:
        return {"bots": bot_info()}

    @app.get("/api/maze")
    def maze(seed: int = 0, size: int = 16, loop_ratio: float = 0.1) -> dict:
        if not 2 <= size <= 32:
            raise HTTPException(422, "size must be between 2 and 32")
        if not 0.0 <= loop_ratio <= 1.0:
            raise HTTPException(422, "loop_ratio must be between 0 and 1")
        return _maze_payload(generate_maze(seed=seed, size=size, loop_ratio=loop_ratio), seed)

    @app.post("/api/run")
    def run(req: RunRequest) -> dict:
        try:
            bot = get_bot(req.bot)
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from exc

        grid = generate_maze(seed=req.seed, size=req.size, loop_ratio=req.loop_ratio)
        result = run_attempt(
            bot, grid, seed=req.seed, budget_s=req.budget_s, max_steps=req.max_steps
        )
        return {"result": result.to_dict(), "maze": _maze_payload(grid, req.seed)}

    @app.post("/api/race")
    def race(req: RaceRequest) -> dict:
        unknown = set(req.bots) - {b["name"] for b in bot_info()}
        if unknown:
            raise HTTPException(400, f"unknown bots: {', '.join(sorted(unknown))}")

        result = run_tournament(
            bots=req.bots,
            seeds=list(range(req.mazes)),
            size=req.size,
            loop_ratio=req.loop_ratio,
        )
        payload = result.to_dict()
        # Traces across a whole tournament are large and the leaderboard view
        # does not need them; the run endpoint serves single-attempt replays.
        for attempt in payload["attempts"]:
            attempt.pop("trace", None)
        return payload

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    print(f"dashboard on http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
