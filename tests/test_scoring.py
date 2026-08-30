"""Contest scoring: handicapped time = run time + search penalty + touches."""
import pytest

from micromouse.scoring import SEARCH_PENALTY_DIVISOR, handicapped_score


class TestHandicappedScore:
    def test_search_penalty_is_one_thirtieth_of_maze_time(self):
        assert SEARCH_PENALTY_DIVISOR == 30
        assert handicapped_score(run_time=10.0, maze_time=60.0) == pytest.approx(12.0)

    def test_a_fast_run_after_a_long_search_can_lose_to_a_steadier_mouse(self):
        speedy = handicapped_score(run_time=5.0, maze_time=300.0)   # 5 + 10
        steady = handicapped_score(run_time=9.0, maze_time=60.0)    # 9 + 2
        assert steady < speedy

    def test_touch_penalty_adds_three_seconds_plus_a_tenth_of_the_run(self):
        clean = handicapped_score(run_time=10.0, maze_time=30.0)
        touched = handicapped_score(run_time=10.0, maze_time=30.0, touched=True)
        assert touched - clean == pytest.approx(3.0 + 1.0)

    def test_unsolved_attempt_scores_infinity(self):
        assert handicapped_score(run_time=None, maze_time=600.0) == float("inf")

    def test_rejects_negative_times(self):
        with pytest.raises(ValueError):
            handicapped_score(run_time=-1.0, maze_time=10.0)
