"""Micromouse contest scoring.

Follows the classic handicapped-time formula::

    score = best run time + search penalty + touch penalty

where the search penalty is 1/30 of the total time spent in the maze, so a mouse
that explores forever cannot buy a fast run for free, and the touch penalty is
3 seconds plus a tenth of the run for a mouse that needed handling.
"""

from __future__ import annotations

__all__ = ["SEARCH_PENALTY_DIVISOR", "handicapped_score"]

SEARCH_PENALTY_DIVISOR = 30
TOUCH_FLAT_PENALTY = 3.0
TOUCH_RUN_FRACTION = 0.1


def handicapped_score(
    run_time: float | None,
    maze_time: float,
    touched: bool = False,
) -> float:
    """Handicapped time in seconds; ``inf`` if the mouse never reached the goal."""
    if maze_time < 0 or (run_time is not None and run_time < 0):
        raise ValueError("times must be non-negative")
    if run_time is None:
        return float("inf")

    score = run_time + maze_time / SEARCH_PENALTY_DIVISOR
    if touched:
        score += TOUCH_FLAT_PENALTY + TOUCH_RUN_FRACTION * run_time
    return score
