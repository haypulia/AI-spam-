from pathlib import Path
from typing import Dict, Optional, Union

from ..features.image_signals import (
    DIMENSION_FEATURES,
    IMAGE_FEATURE_NAMES,
    extract_image_features,
    image_metadata,
)
from .artifacts import IMAGE_KIND, ArtifactAnalyzer, ArtifactResult
from .base import DEFAULT_THRESHOLDS, VerdictThresholds
from .model import LinearModel

PathLike = Union[str, Path]

REPORTED_SIGNALS = (
    "image_color_entropy",
    "image_noise_level",
    "image_edge_density",
    "image_dominant_color_share",
    "image_flat_area_ratio",
    "image_bytes_per_pixel",
    "image_has_software_tag",
)

SIGNAL_LABELS = {
    "image_color_entropy": "разнообразие цветов",
    "image_noise_level": "уровень шума",
    "image_edge_density": "плотность границ",
    "image_dominant_color_share": "доля доминирующего цвета",
    "image_flat_area_ratio": "доля плоских областей",
    "image_bytes_per_pixel": "плотность сжатия",
    "image_has_software_tag": "тег редактора в метаданных",
}

FALLBACK_WEIGHTS = {
    "image_color_entropy": 0.22,
    "image_noise_level": 0.02,
    "image_edge_density": 1.40,
    "image_dominant_color_share": -0.35,
}
FALLBACK_INTERCEPT = -0.35


class ImageAnalyzer(ArtifactAnalyzer):
    name = "image_signals"
    kind = IMAGE_KIND
    content_types = ("image/",)

    def __init__(self, model: Optional[LinearModel] = None, thresholds: Optional[VerdictThresholds] = None):
        self.model = model
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    @classmethod
    def from_settings(cls, settings) -> "ImageAnalyzer":
        path = settings.models_dir / "image_model.json"
        model = LinearModel.load(path) if path.exists() else None
        return cls(
            model=model,
            thresholds=VerdictThresholds(
                mixed=settings.verdict_mixed_threshold, ai=settings.verdict_ai_threshold
            ),
        )

    def _score(self, features: Dict[str, float]) -> float:
        if self.model is not None:
            return self.model.predict_proba([features.get(name, 0.0) for name in self.model.feature_names])
        total = FALLBACK_INTERCEPT + sum(
            weight * features.get(name, 0.0) for name, weight in FALLBACK_WEIGHTS.items()
        )
        return max(0.0, min(1.0, total))

    def _explanation(self, score: float, features: Dict[str, float], metadata: Dict[str, object]) -> str:
        if not features.get("image_decoded"):
            return "Изображение не удалось разобрать, оценка не выставлена."

        parts = ["Индекс генерации изображения: %s/100." % round(score * 100, 1)]
        parts.append(
            "Формат %s, размер %sx%s, вес %s Б."
            % (metadata.get("format", "?"), metadata.get("width", 0), metadata.get("height", 0), metadata.get("bytes", 0))
        )

        if self.model is not None:
            vector = [features.get(name, 0.0) for name in self.model.feature_names]
            drivers = [
                SIGNAL_LABELS.get(name, name)
                for name, contribution in self.model.top_contributions(vector, limit=4)
                if contribution > 0 and name in SIGNAL_LABELS
            ]
            if drivers:
                parts.append("Ключевые признаки: %s." % ", ".join(drivers))

        if features.get("image_has_software_tag"):
            parts.append("В метаданных указан редактор: %s." % (metadata.get("software") or "без названия"))

        return " ".join(parts)

    def analyze(self, payload: bytes, source: str = "", content_type: str = "") -> ArtifactResult:
        features = extract_image_features(payload)
        metadata = image_metadata(payload)
        decoded = bool(features.get("image_decoded"))
        score = self._score(features) if decoded else 0.0

        return ArtifactResult(
            kind=self.kind,
            source=source,
            content_type=content_type or ("image/%s" % str(metadata.get("format", "")).lower() if decoded else ""),
            score=score,
            confidence=0.5 + abs(score - 0.5) if decoded else 0.0,
            signals={name: features.get(name, 0.0) for name in REPORTED_SIGNALS},
            explanation=self._explanation(score, features, metadata),
            analyzer=self.name,
            thresholds=self.thresholds,
            meta={
                "decoded": decoded,
                "model": "linear" if self.model is not None else "rule_based",
                "excluded_features": list(DIMENSION_FEATURES),
                "feature_count": len(IMAGE_FEATURE_NAMES),
                **{key: value for key, value in metadata.items() if key != "decoded"},
            },
        )
