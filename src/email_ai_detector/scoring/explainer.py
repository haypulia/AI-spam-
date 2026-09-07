import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..features import EXPLANATION_CATEGORIES, FEATURE_NAMES, extract_features, normalize_category
from .model import LinearModel, train_linear_model

PathLike = Union[str, Path]

CATEGORY_FEATURE_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "subject": ("subject_",),
    "opener": ("opener_", "body_opener_phrase", "body_generic_greeting", "body_sentence_capitalization_rate", "body_typo_marker_rate"),
    "body": ("body_",),
    "cta": ("body_cta_phrase_rate", "body_url_rate", "body_urgency_rate", "subject_cta", "subject_urgency"),
    "closer": ("closer_", "body_closer_phrase", "body_sentence_capitalization_rate"),
    "html_template": ("html_",),
    "image": ("ocr_",),
}


def category_feature_names(category: str) -> List[str]:
    prefixes = CATEGORY_FEATURE_PREFIXES.get(category, ())
    names = [name for name in FEATURE_NAMES if any(name.startswith(prefix) for prefix in prefixes)]
    return names or list(FEATURE_NAMES)


class CategoryExplainer:
    def __init__(self, models: Optional[Dict[str, LinearModel]] = None):
        self.models = models or {}

    def __bool__(self) -> bool:
        return bool(self.models)

    def predict(self, vector: Dict[str, float]) -> Dict[str, float]:
        scores = {}
        for category, model in self.models.items():
            values = [float(vector.get(name, 0.0)) for name in model.feature_names]
            scores[category] = model.predict_proba(values)
        return scores

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {category: model.to_dict() for category, model in self.models.items()}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "CategoryExplainer":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        models = {}
        for category, data in payload.items():
            models[category] = LinearModel(
                feature_names=list(data["feature_names"]),
                coefficients=[float(value) for value in data["coefficients"]],
                intercept=float(data.get("intercept", 0.0)),
                mean=[float(value) for value in data.get("mean", [])],
                scale=[float(value) for value in data.get("scale", [])],
                metadata=data.get("metadata", {}),
            )
        return cls(models)


def build_category_dataset(records: Sequence, category: str) -> Tuple[List[List[float]], List[int]]:
    names = category_feature_names(category)
    matrix: List[List[float]] = []
    labels: List[int] = []
    for record in records:
        features = extract_features(
            text=record.text, subject=record.subject, html=record.html, ocr_text=record.ocr_text
        )
        matrix.append([features.vector.get(name, 0.0) for name in names])
        reference = {normalize_category(str(item).strip().lower()) for item in record.ai_elements}
        labels.append(1 if category in reference else 0)
    return matrix, labels


def train_category_models(records: Sequence, metadata: Optional[dict] = None) -> CategoryExplainer:
    models: Dict[str, LinearModel] = {}
    for category in EXPLANATION_CATEGORIES:
        matrix, labels = build_category_dataset(records, category)
        if len(set(labels)) < 2:
            continue
        payload = {"category": category, "samples": len(labels), "positives": sum(labels)}
        payload.update(metadata or {})
        try:
            from .training import VOLUME_FEATURES

            models[category] = train_linear_model(
                matrix,
                labels,
                category_feature_names(category),
                metadata=payload,
                excluded=VOLUME_FEATURES,
            )
        except ValueError:
            continue
    return CategoryExplainer(models)
