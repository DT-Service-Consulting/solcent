from eval import metrics


def test_pinball_median_equals_half_abs_error():
    # at alpha=0.5 pinball is half the absolute error
    assert abs(metrics.pinball([10], [4], 0.5) - 3.0) < 1e-9


def test_coverage_full_and_empty():
    assert metrics.interval_coverage([1, 2, 3], [0, 0, 0], [9, 9, 9]) == 1.0
    assert metrics.interval_coverage([1, 2, 3], [5, 5, 5], [9, 9, 9]) == 0.0


def test_skill_score_sign():
    assert metrics.skill_score(0.5, 1.0) == 0.5   # half the reference error -> 0.5 skill
