from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..scoring.base import AI_THRESHOLD, MIXED_THRESHOLD
from ..utils import save_json
from .explanations import evaluate_explanations, threshold_sweep
from .masking import MASK_DESCRIPTIONS, MASKS, apply_mask
from .metrics import (
    average_precision_score,
    best_threshold,
    bootstrap_ci,
    brier_score,
    roc_auc_score,
    roc_curve,
    threshold_metrics,
)
from .partial import evaluate_partial, segment_threshold_sweep

EXPLANATION_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
SEGMENT_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.8)


def _group_auc(records: Sequence, scores: Sequence[float], key) -> Dict[str, Dict[str, float]]:
    grouped = defaultdict(lambda: {"labels": [], "scores": []})
    for record, score in zip(records, scores):
        name = key(record)
        if name is None:
            continue
        grouped[str(name)]["labels"].append(record.ai_binary)
        grouped[str(name)]["scores"].append(score)

    report = {}
    for name, payload in sorted(grouped.items()):
        labels = payload["labels"]
        if len(set(labels)) < 2:
            report[name] = {"count": len(labels), "roc_auc": None, "mean_score": sum(payload["scores"]) / len(labels)}
            continue
        report[name] = {
            "count": len(labels),
            "roc_auc": roc_auc_score(labels, payload["scores"]),
            "mean_score": sum(payload["scores"]) / len(labels),
        }
    return report


def _group_auc_vs_human(records: Sequence, scores: Sequence[float], key) -> Dict[str, Dict[str, float]]:
    human_scores = [score for record, score in zip(records, scores) if record.ai_binary == 0]
    grouped = defaultdict(list)
    for record, score in zip(records, scores):
        if record.ai_binary == 0:
            continue
        name = key(record)
        if name is None:
            continue
        grouped[str(name)].append(score)

    report = {}
    for name, values in sorted(grouped.items()):
        labels = [1] * len(values) + [0] * len(human_scores)
        combined = list(values) + human_scores
        report[name] = {
            "count": len(values),
            "roc_auc": roc_auc_score(labels, combined) if human_scores else None,
            "mean_score": sum(values) / len(values),
        }
    if human_scores:
        report["human"] = {
            "count": len(human_scores),
            "roc_auc": None,
            "mean_score": sum(human_scores) / len(human_scores),
        }
    return report


def _detection_rate_by_label(records: Sequence, scores: Sequence[float], threshold: float) -> Dict[str, float]:
    grouped = defaultdict(list)
    for record, score in zip(records, scores):
        grouped[record.label].append(score)
    return {
        label: sum(1 for score in values if score >= threshold) / len(values)
        for label, values in sorted(grouped.items())
    }


def evaluate_detection(records: Sequence, results: Sequence, threshold: float = 0.5) -> Dict[str, object]:
    labels = [record.ai_binary for record in records]
    scores = [result.score for result in results]

    report = {
        "count": len(records),
        "positives": sum(labels),
        "negatives": len(labels) - sum(labels),
        "roc_auc": roc_auc_score(labels, scores),
        "roc_auc_ci95": bootstrap_ci(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "brier": brier_score(labels, scores),
        "at_decision_threshold": threshold_metrics(labels, scores, threshold),
        "at_operating_threshold": threshold_metrics(labels, scores, AI_THRESHOLD),
        "best_f1": best_threshold(labels, scores, "f1"),
        "detection_rate_by_label": _detection_rate_by_label(records, scores, threshold),
        "by_language": _group_auc(records, scores, lambda record: record.lang),
        "by_generator_model": _group_auc_vs_human(records, scores, lambda record: record.model),
        "by_data_type": _group_auc(records, scores, lambda record: record.data_type),
        "by_origin": _group_auc_vs_human(records, scores, lambda record: record.ai_origin),
    }
    return report


def evaluate_masking(
    scorer,
    records: Sequence,
    baseline_results: Sequence,
    threshold: float = 0.5,
    masks: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    labels = [record.ai_binary for record in records]
    baseline_scores = [result.score for result in baseline_results]
    baseline_auc = roc_auc_score(labels, baseline_scores)
    baseline_recall = threshold_metrics(labels, baseline_scores, threshold)["recall"]

    report: Dict[str, object] = {
        "baseline_roc_auc": baseline_auc,
        "baseline_recall": baseline_recall,
        "threshold": threshold,
        "masks": {},
    }

    for name in masks or MASKS.keys():
        masked_records = apply_mask(name, list(records))
        masked_results = scorer.score_records(masked_records)
        masked_scores = [result.score for result in masked_results]
        masked_auc = roc_auc_score(labels, masked_scores)
        masked_metrics = threshold_metrics(labels, masked_scores, threshold)
        report["masks"][name] = {
            "description": MASK_DESCRIPTIONS.get(name, ""),
            "roc_auc": masked_auc,
            "roc_auc_delta": masked_auc - baseline_auc,
            "roc_auc_retention": masked_auc / baseline_auc if baseline_auc else float("nan"),
            "recall": masked_metrics["recall"],
            "recall_delta": masked_metrics["recall"] - baseline_recall,
            "precision": masked_metrics["precision"],
            "f1": masked_metrics["f1"],
        }

    values = [payload["roc_auc_retention"] for payload in report["masks"].values()]
    report["mean_roc_auc_retention"] = sum(values) / len(values) if values else float("nan")
    report["worst_mask"] = min(
        report["masks"].items(), key=lambda item: item[1]["roc_auc"]
    )[0] if report["masks"] else None
    return report


def run_evaluation(
    scorer,
    records: Sequence,
    split_name: str,
    dataset_name: str = "ai_assisted_spam",
    threshold: float = 0.5,
    explanation_threshold: float = 0.5,
    segment_threshold: float = 0.5,
    masks: Optional[Sequence[str]] = None,
    include_masking: bool = True,
) -> Dict[str, object]:
    results = scorer.score_records(records)
    labels = [record.ai_binary for record in records]
    scores = [result.score for result in results]
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)

    report: Dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "split": split_name,
        "scorer": scorer.name,
        "decision_threshold": threshold,
        "verdict_thresholds": {"mixed": MIXED_THRESHOLD, "ai": AI_THRESHOLD},
        "detection": evaluate_detection(records, results, threshold),
        "explanations": evaluate_explanations(records, results, explanation_threshold),
        "explanation_threshold_sweep": threshold_sweep(records, results, EXPLANATION_THRESHOLDS),
        "partial_generation": evaluate_partial(records, results, threshold, segment_threshold),
        "partial_segment_sweep": segment_threshold_sweep(records, results, SEGMENT_THRESHOLDS, threshold),
        "roc_curve": {
            "fpr": [round(value, 6) for value in false_positive_rate],
            "tpr": [round(value, 6) for value in true_positive_rate],
            "thresholds": [round(float(value), 6) if value != float("inf") else None for value in thresholds],
        },
    }

    if include_masking:
        report["masking_robustness"] = evaluate_masking(scorer, records, results, threshold, masks)

    return report


def save_report(report: Dict[str, object], directory: Path, prefix: str = "metrics") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return save_json(report, directory / ("%s.json" % prefix))


def plot_roc_curve(report: Dict[str, object], path: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    curve = report.get("roc_curve", {})
    if not curve:
        return None

    auc = report["detection"]["roc_auc"]
    figure, axes = plt.subplots(figsize=(6, 6))
    axes.plot(curve["fpr"], curve["tpr"], color="#1f77b4", linewidth=2, label="AUC-ROC = %.4f" % auc)
    axes.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("True Positive Rate")
    axes.set_title("ROC, %s, split %s" % (report["scorer"], report["split"]))
    axes.legend(loc="lower right")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
