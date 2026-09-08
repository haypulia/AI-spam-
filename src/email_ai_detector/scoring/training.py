from typing import Dict, List, Sequence, Tuple

from ..features import FEATURE_NAMES, SEGMENT_FEATURE_NAMES, extract_features, iter_segments
from ..features.perturbations import DEFAULT_SEEDS, augmented_records
from .explainer import CategoryExplainer, train_category_models
from .model import LinearModel, train_linear_model

SEGMENT_OVERLAP_THRESHOLD = 0.5

VOLUME_FEATURES = (
    "subject_len",
    "subject_word_count",
    "body_word_count_log",
    "html_html_len_log",
    "html_markup_text_ratio",
    "html_comment_count",
    "html_template_variable_count",
    "html_empty_cell_count",
    "html_missing_alt",
    "opener_len_log",
    "closer_len_log",
    "ocr_len_log",
    "segment_char_len_log",
    "segment_word_count_log",
)


def build_document_dataset(records: Sequence) -> Tuple[List[List[float]], List[int]]:
    matrix: List[List[float]] = []
    labels: List[int] = []
    for record in records:
        features = extract_features(
            text=record.text,
            subject=record.subject,
            html=record.html,
            ocr_text=record.ocr_text,
        )
        matrix.append(features.as_list(FEATURE_NAMES))
        labels.append(int(record.ai_binary))
    return matrix, labels


def _segment_label(record, start: int, end: int) -> int:
    if record.ai_binary == 0:
        return 0
    if not record.ai_char_intervals:
        return 1
    length = max(1, end - start)
    overlap = 0
    for interval_start, interval_end in record.ai_char_intervals:
        overlap += max(0, min(end, int(interval_end)) - max(start, int(interval_start)))
    return 1 if overlap / length >= SEGMENT_OVERLAP_THRESHOLD else 0


def build_segment_dataset(records: Sequence) -> Tuple[List[List[float]], List[int]]:
    matrix: List[List[float]] = []
    labels: List[int] = []
    for record in records:
        for segment in iter_segments(record.text):
            matrix.append([segment["features"].get(name, 0.0) for name in SEGMENT_FEATURE_NAMES])
            labels.append(_segment_label(record, segment["start"], segment["end"]))
    return matrix, labels


def train_document_model(
    records: Sequence,
    metadata: Dict[str, object] = None,
    augmentation_seeds: Sequence[int] = DEFAULT_SEEDS,
) -> LinearModel:
    expanded = augmented_records(records, augmentation_seeds) if augmentation_seeds else list(records)
    matrix, labels = build_document_dataset(expanded)
    payload = {
        "kind": "document",
        "samples": len(labels),
        "positives": sum(labels),
        "source_emails": len(records),
        "augmentation_seeds": list(augmentation_seeds or ()),
    }
    payload.update(metadata or {})
    return train_linear_model(matrix, labels, FEATURE_NAMES, metadata=payload, excluded=VOLUME_FEATURES)


def train_segment_model(
    records: Sequence,
    metadata: Dict[str, object] = None,
    augmentation_seeds: Sequence[int] = DEFAULT_SEEDS,
) -> LinearModel:
    expanded = augmented_records(records, augmentation_seeds) if augmentation_seeds else list(records)
    matrix, labels = build_segment_dataset(expanded)
    payload = {
        "kind": "segment",
        "samples": len(labels),
        "positives": sum(labels),
        "source_emails": len(records),
        "augmentation_seeds": list(augmentation_seeds or ()),
    }
    payload.update(metadata or {})
    return train_linear_model(matrix, labels, SEGMENT_FEATURE_NAMES, metadata=payload, excluded=VOLUME_FEATURES)


def train_explainer(records: Sequence, metadata: Dict[str, object] = None) -> CategoryExplainer:
    return train_category_models(records, metadata=metadata)
