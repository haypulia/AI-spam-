import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end + 2) / 2.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def roc_curve(labels: Sequence[int], scores: Sequence[float]) -> Tuple[List[float], List[float], List[float]]:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]

    true_positive = 0
    false_positive = 0
    fpr = [0.0]
    tpr = [0.0]
    thresholds = [float("inf")]
    previous_score = None

    for score, label in pairs:
        if previous_score is not None and score != previous_score:
            fpr.append(false_positive / negatives)
            tpr.append(true_positive / positives)
            thresholds.append(previous_score)
        if label == 1:
            true_positive += 1
        else:
            false_positive += 1
        previous_score = score

    fpr.append(false_positive / negatives)
    tpr.append(true_positive / positives)
    thresholds.append(previous_score if previous_score is not None else 0.0)
    return fpr, tpr, thresholds


def average_precision_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(1 for label in labels if label == 1)
    if positives == 0:
        return float("nan")

    true_positive = 0
    seen = 0
    previous_recall = 0.0
    total = 0.0
    index = 0

    while index < len(pairs):
        current_score = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == current_score:
            true_positive += pairs[index][1]
            seen += 1
            index += 1
        recall = true_positive / positives
        precision = true_positive / seen
        total += precision * (recall - previous_recall)
        previous_recall = recall

    return total


def confusion_at_threshold(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, int]:
    true_positive = false_positive = true_negative = false_negative = 0
    for label, score in zip(labels, scores):
        predicted = 1 if score >= threshold else 0
        if label == 1 and predicted == 1:
            true_positive += 1
        elif label == 1:
            false_negative += 1
        elif predicted == 1:
            false_positive += 1
        else:
            true_negative += 1
    return {"tp": true_positive, "fp": false_positive, "tn": true_negative, "fn": false_negative}


def threshold_metrics(labels: Sequence[int], scores: Sequence[float], threshold: float) -> Dict[str, float]:
    matrix = confusion_at_threshold(labels, scores, threshold)
    true_positive = matrix["tp"]
    false_positive = matrix["fp"]
    true_negative = matrix["tn"]
    false_negative = matrix["fn"]
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = len(labels) or 1
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": (true_positive + true_negative) / total,
        "balanced_accuracy": (recall + specificity) / 2,
        **{key: float(value) for key, value in matrix.items()},
    }


def best_threshold(labels: Sequence[int], scores: Sequence[float], metric: str = "f1") -> Dict[str, float]:
    candidates = sorted(set(round(float(score), 4) for score in scores))
    best = threshold_metrics(labels, scores, 0.5)
    for threshold in candidates:
        current = threshold_metrics(labels, scores, threshold)
        if current[metric] > best[metric]:
            best = current
    return best


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    if not labels:
        return float("nan")
    return sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)


def bootstrap_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    metric: Callable[[Sequence[int], Sequence[float]], float] = roc_auc_score,
    iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    generator = random.Random(seed)
    size = len(labels)
    samples: List[float] = []
    for _ in range(iterations):
        indices = [generator.randrange(size) for _ in range(size)]
        sampled_labels = [labels[index] for index in indices]
        sampled_scores = [scores[index] for index in indices]
        value = metric(sampled_labels, sampled_scores)
        if not math.isnan(value):
            samples.append(value)
    if not samples:
        return {"low": float("nan"), "high": float("nan"), "iterations": 0}
    samples.sort()
    low_index = int(alpha / 2 * len(samples))
    high_index = min(len(samples) - 1, int((1 - alpha / 2) * len(samples)))
    return {"low": samples[low_index], "high": samples[high_index], "iterations": len(samples)}


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) < 2:
        return float("nan")
    first_ranks = _ranks(first)
    second_ranks = _ranks(second)
    mean_first = sum(first_ranks) / len(first_ranks)
    mean_second = sum(second_ranks) / len(second_ranks)
    covariance = sum(
        (a - mean_first) * (b - mean_second) for a, b in zip(first_ranks, second_ranks)
    )
    variance_first = math.sqrt(sum((a - mean_first) ** 2 for a in first_ranks))
    variance_second = math.sqrt(sum((b - mean_second) ** 2 for b in second_ranks))
    if not variance_first or not variance_second:
        return float("nan")
    return covariance / (variance_first * variance_second)


def interval_overlap(first: Sequence[Sequence[int]], second: Sequence[Sequence[int]]) -> int:
    total = 0
    for start_a, end_a in first:
        for start_b, end_b in second:
            total += max(0, min(int(end_a), int(end_b)) - max(int(start_a), int(start_b)))
    return total


def interval_length(intervals: Sequence[Sequence[int]]) -> int:
    return sum(max(0, int(end) - int(start)) for start, end in intervals)


def merge_intervals(intervals: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    merged: List[Tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def localization_metrics(
    predicted: Sequence[Sequence[int]], reference: Sequence[Sequence[int]]
) -> Dict[str, float]:
    predicted_merged = merge_intervals(predicted)
    reference_merged = merge_intervals(reference)
    overlap = interval_overlap(predicted_merged, reference_merged)
    predicted_length = interval_length(predicted_merged)
    reference_length = interval_length(reference_merged)
    union = predicted_length + reference_length - overlap
    precision = overlap / predicted_length if predicted_length else 0.0
    recall = overlap / reference_length if reference_length else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": overlap / union if union else 0.0,
        "predicted_chars": float(predicted_length),
        "reference_chars": float(reference_length),
    }


def summarize_scores(labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    return {
        "count": float(len(labels)),
        "positives": float(sum(labels)),
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
        "brier": brier_score(labels, scores),
        **threshold_metrics(labels, scores, threshold),
    }
