import math

from email_ai_detector.evaluation.metrics import (
    average_precision_score,
    bootstrap_ci,
    localization_metrics,
    merge_intervals,
    roc_auc_score,
    roc_curve,
    spearman_correlation,
    threshold_metrics,
)

LABELS = [0, 0, 1, 1]
SCORES = [0.1, 0.4, 0.35, 0.8]


def test_roc_auc_matches_reference_value():
    assert math.isclose(roc_auc_score(LABELS, SCORES), 0.75, rel_tol=1e-9)


def test_roc_auc_is_undefined_for_single_class():
    assert math.isnan(roc_auc_score([1, 1, 1], [0.1, 0.2, 0.3]))


def test_average_precision_is_bounded():
    value = average_precision_score(LABELS, SCORES)
    assert 0.0 <= value <= 1.0


def test_roc_curve_endpoints():
    fpr, tpr, _ = roc_curve(LABELS, SCORES)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert math.isclose(fpr[-1], 1.0) and math.isclose(tpr[-1], 1.0)


def test_threshold_metrics_counts():
    metrics = threshold_metrics(LABELS, SCORES, 0.5)
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 0
    assert metrics["tn"] == 2


def test_bootstrap_ci_is_ordered():
    interval = bootstrap_ci(LABELS * 10, SCORES * 10, iterations=50)
    assert interval["low"] <= interval["high"]


def test_localization_metrics_on_exact_match():
    metrics = localization_metrics([(0, 10)], [(0, 10)])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["iou"] == 1.0


def test_merge_intervals_joins_overlaps():
    assert merge_intervals([(0, 5), (3, 9), (20, 25)]) == [(0, 9), (20, 25)]


def test_spearman_of_monotonic_series():
    assert math.isclose(spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, rel_tol=1e-9)
