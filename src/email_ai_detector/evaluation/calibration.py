import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from .explanations import evaluate_explanations
from .metrics import best_threshold
from .partial import evaluate_partial

PathLike = Union[str, Path]

DEFAULT_CALIBRATION = {
    "decision_threshold": 0.5,
    "explanation_threshold": 0.5,
    "segment_threshold": 0.5,
}

THRESHOLD_GRID = tuple(round(0.05 * step, 2) for step in range(2, 19))


def calibrate(scorer, records: Sequence) -> Dict[str, object]:
    results = scorer.score_records(records)
    labels = [record.ai_binary for record in records]
    scores = [result.score for result in results]

    decision = best_threshold(labels, scores, "balanced_accuracy")

    explanation_threshold = DEFAULT_CALIBRATION["explanation_threshold"]
    explanation_f1 = -1.0
    for threshold in THRESHOLD_GRID:
        report = evaluate_explanations(records, results, threshold)
        if report["explanation_f1_macro"] > explanation_f1:
            explanation_f1 = report["explanation_f1_macro"]
            explanation_threshold = threshold

    segment_threshold = DEFAULT_CALIBRATION["segment_threshold"]
    segment_f1 = -1.0
    for threshold in THRESHOLD_GRID:
        report = evaluate_partial(records, results, decision["threshold"], threshold)
        value = report["localization"]["f1_macro"]
        if value > segment_f1:
            segment_f1 = value
            segment_threshold = threshold

    return {
        "decision_threshold": round(float(decision["threshold"]), 4),
        "decision_f1": round(float(decision["f1"]), 4),
        "decision_balanced_accuracy": round(float(decision["balanced_accuracy"]), 4),
        "explanation_threshold": explanation_threshold,
        "explanation_f1_macro": round(explanation_f1, 4),
        "segment_threshold": segment_threshold,
        "segment_localization_f1": round(segment_f1, 4),
        "records": len(records),
        "scorer": scorer.name,
    }


def save_calibration(calibration: Dict[str, object], path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2, ensure_ascii=False)
    return path


def load_calibration(path: Optional[PathLike]) -> Dict[str, object]:
    if not path or not Path(path).exists():
        return dict(DEFAULT_CALIBRATION)
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    calibration = dict(DEFAULT_CALIBRATION)
    calibration.update({key: value for key, value in payload.items() if key in DEFAULT_CALIBRATION})
    return calibration
