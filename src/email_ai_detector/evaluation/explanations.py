from typing import Dict, List, Sequence

from ..features import EXPLANATION_CATEGORIES, normalize_category


def reference_categories(record) -> List[str]:
    normalized = {normalize_category(str(item).strip().lower()) for item in record.ai_elements}
    return sorted(name for name in normalized if name in EXPLANATION_CATEGORIES)


def predicted_categories(result, threshold: float) -> List[str]:
    return sorted(
        name
        for name, value in result.categories.items()
        if name in EXPLANATION_CATEGORIES and value >= threshold
    )


def evaluate_explanations(records: Sequence, results: Sequence, threshold: float = 0.5) -> Dict[str, object]:
    per_category = {
        name: {"reference": 0, "predicted": 0, "matched": 0} for name in EXPLANATION_CATEGORIES
    }

    macro_recall: List[float] = []
    macro_precision: List[float] = []
    macro_f1: List[float] = []
    covered = 0
    exact = 0
    evaluated = 0

    total_reference = 0
    total_predicted = 0
    total_matched = 0

    for record, result in zip(records, results):
        reference = set(reference_categories(record))
        if not reference:
            continue
        evaluated += 1
        predicted = set(predicted_categories(result, threshold))
        matched = reference & predicted

        total_reference += len(reference)
        total_predicted += len(predicted)
        total_matched += len(matched)

        recall = len(matched) / len(reference)
        precision = len(matched) / len(predicted) if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        macro_recall.append(recall)
        macro_precision.append(precision)
        macro_f1.append(f1)

        if matched:
            covered += 1
        if reference == predicted:
            exact += 1

        for name in reference:
            per_category[name]["reference"] += 1
            if name in predicted:
                per_category[name]["matched"] += 1
        for name in predicted:
            per_category[name]["predicted"] += 1

    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    micro_recall = total_matched / total_reference if total_reference else 0.0
    micro_precision = total_matched / total_predicted if total_predicted else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )

    categories_report = {}
    for name, counters in per_category.items():
        reference_count = counters["reference"]
        predicted_count = counters["predicted"]
        matched_count = counters["matched"]
        categories_report[name] = {
            "reference": reference_count,
            "predicted": predicted_count,
            "matched": matched_count,
            "recall": matched_count / reference_count if reference_count else 0.0,
            "precision": matched_count / predicted_count if predicted_count else 0.0,
        }

    return {
        "threshold": threshold,
        "evaluated_emails": evaluated,
        "explanation_recall_macro": _mean(macro_recall),
        "explanation_precision_macro": _mean(macro_precision),
        "explanation_f1_macro": _mean(macro_f1),
        "explanation_recall_micro": micro_recall,
        "explanation_precision_micro": micro_precision,
        "explanation_f1_micro": micro_f1,
        "coverage_at_least_one": covered / evaluated if evaluated else 0.0,
        "exact_set_match": exact / evaluated if evaluated else 0.0,
        "per_category": categories_report,
    }


def threshold_sweep(records: Sequence, results: Sequence, thresholds: Sequence[float]) -> List[Dict[str, object]]:
    sweep = []
    for threshold in thresholds:
        report = evaluate_explanations(records, results, threshold)
        sweep.append(
            {
                "threshold": threshold,
                "recall_macro": report["explanation_recall_macro"],
                "precision_macro": report["explanation_precision_macro"],
                "f1_macro": report["explanation_f1_macro"],
            }
        )
    return sweep
