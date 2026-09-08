from typing import Dict, List, Sequence

from .metrics import (
    localization_metrics,
    merge_intervals,
    roc_auc_score,
    spearman_correlation,
    threshold_metrics,
)


def predicted_intervals(result, threshold: float) -> List[Sequence[int]]:
    return merge_intervals(
        [(segment.start, segment.end) for segment in result.segments if segment.score >= threshold]
    )


def evaluate_partial(
    records: Sequence,
    results: Sequence,
    decision_threshold: float = 0.5,
    segment_threshold: float = 0.5,
) -> Dict[str, object]:
    paired = list(zip(records, results))

    mixed = [(record, result) for record, result in paired if record.label == "mixed"]
    human = [(record, result) for record, result in paired if record.label == "human"]
    full_ai = [(record, result) for record, result in paired if record.label == "ai"]

    mixed_vs_human_labels = [1] * len(mixed) + [0] * len(human)
    mixed_vs_human_scores = [result.score for _, result in mixed] + [result.score for _, result in human]

    ai_vs_mixed_labels = [1] * len(full_ai) + [0] * len(mixed)
    ai_vs_mixed_scores = [result.score for _, result in full_ai] + [result.score for _, result in mixed]

    fractions = [record.ai_char_fraction for record, _ in mixed]
    mixed_scores = [result.score for _, result in mixed]

    macro_precision: List[float] = []
    macro_recall: List[float] = []
    macro_f1: List[float] = []
    macro_iou: List[float] = []
    localized = 0

    for record, result in mixed:
        if not record.ai_char_intervals or not result.segments:
            continue
        predicted = predicted_intervals(result, segment_threshold)
        metrics = localization_metrics(predicted, record.ai_char_intervals)
        macro_precision.append(metrics["precision"])
        macro_recall.append(metrics["recall"])
        macro_f1.append(metrics["f1"])
        macro_iou.append(metrics["iou"])
        localized += 1

    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    report: Dict[str, object] = {
        "mixed_count": len(mixed),
        "human_count": len(human),
        "full_ai_count": len(full_ai),
        "roc_auc_mixed_vs_human": roc_auc_score(mixed_vs_human_labels, mixed_vs_human_scores)
        if mixed and human
        else float("nan"),
        "roc_auc_full_ai_vs_mixed": roc_auc_score(ai_vs_mixed_labels, ai_vs_mixed_scores)
        if mixed and full_ai
        else float("nan"),
        "mixed_detection_rate": (
            sum(1 for _, result in mixed if result.score >= decision_threshold) / len(mixed)
            if mixed
            else 0.0
        ),
        "mean_score_mixed": _mean(mixed_scores),
        "mean_score_human": _mean([result.score for _, result in human]),
        "mean_score_full_ai": _mean([result.score for _, result in full_ai]),
        "score_vs_ai_fraction_spearman": spearman_correlation(fractions, mixed_scores)
        if len(mixed) > 2
        else float("nan"),
        "localization": {
            "evaluated_emails": localized,
            "segment_threshold": segment_threshold,
            "precision_macro": _mean(macro_precision),
            "recall_macro": _mean(macro_recall),
            "f1_macro": _mean(macro_f1),
            "iou_macro": _mean(macro_iou),
        },
    }

    if mixed and human:
        report["threshold_metrics_mixed_vs_human"] = threshold_metrics(
            mixed_vs_human_labels, mixed_vs_human_scores, decision_threshold
        )

    return report


def segment_threshold_sweep(
    records: Sequence, results: Sequence, thresholds: Sequence[float], decision_threshold: float = 0.5
) -> List[Dict[str, object]]:
    sweep = []
    for threshold in thresholds:
        report = evaluate_partial(records, results, decision_threshold, threshold)
        localization = report["localization"]
        sweep.append(
            {
                "segment_threshold": threshold,
                "precision_macro": localization["precision_macro"],
                "recall_macro": localization["recall_macro"],
                "f1_macro": localization["f1_macro"],
                "iou_macro": localization["iou_macro"],
            }
        )
    return sweep
