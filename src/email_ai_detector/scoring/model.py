import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]


STANDARDIZED_CLIP = 5.0


@dataclass
class LinearModel:
    feature_names: List[str]
    coefficients: List[float]
    intercept: float = 0.0
    mean: List[float] = field(default_factory=list)
    scale: List[float] = field(default_factory=list)
    clip: float = STANDARDIZED_CLIP
    metadata: Dict[str, object] = field(default_factory=dict)

    def _standardize(self, vector: Sequence[float]) -> List[float]:
        if not self.mean or not self.scale:
            return list(vector)
        standardized = []
        for index, value in enumerate(vector):
            scale = self.scale[index] if self.scale[index] else 1.0
            item = (value - self.mean[index]) / scale
            if self.clip:
                item = max(-self.clip, min(self.clip, item))
            standardized.append(item)
        return standardized

    def decision(self, vector: Sequence[float]) -> float:
        standardized = self._standardize(vector)
        return self.intercept + sum(
            coefficient * value for coefficient, value in zip(self.coefficients, standardized)
        )

    def predict_proba(self, vector: Sequence[float]) -> float:
        value = self.decision(vector)
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exponent = math.exp(value)
        return exponent / (1.0 + exponent)

    def contributions(self, vector: Sequence[float]) -> Dict[str, float]:
        standardized = self._standardize(vector)
        return {
            name: self.coefficients[index] * standardized[index]
            for index, name in enumerate(self.feature_names)
        }

    def top_contributions(self, vector: Sequence[float], limit: int = 6) -> List[Tuple[str, float]]:
        items = sorted(self.contributions(vector).items(), key=lambda item: abs(item[1]), reverse=True)
        return items[:limit]

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "mean": self.mean,
            "scale": self.scale,
            "clip": self.clip,
            "metadata": self.metadata,
        }

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "LinearModel":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            feature_names=list(payload["feature_names"]),
            coefficients=[float(value) for value in payload["coefficients"]],
            intercept=float(payload.get("intercept", 0.0)),
            mean=[float(value) for value in payload.get("mean", [])],
            scale=[float(value) for value in payload.get("scale", [])],
            clip=float(payload.get("clip", STANDARDIZED_CLIP)),
            metadata=payload.get("metadata", {}),
        )


def train_linear_model(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[int],
    feature_names: Sequence[str],
    regularization: float = 1.0,
    max_iter: int = 2000,
    metadata: Optional[dict] = None,
    excluded: Optional[Sequence[str]] = None,
) -> LinearModel:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(matrix, dtype=float)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    targets = np.asarray(labels, dtype=int)

    excluded = set(excluded or ())
    variance = features.std(axis=0)
    informative = variance > 0
    for index, name in enumerate(feature_names):
        if name in excluded:
            informative[index] = False
    if not informative.any():
        raise ValueError("no informative features in the training matrix")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features[:, informative])

    classifier = LogisticRegression(
        C=regularization,
        max_iter=max_iter,
        class_weight="balanced",
        solver="liblinear",
    )
    classifier.fit(scaled, targets)

    coefficients = np.zeros(features.shape[1], dtype=float)
    coefficients[informative] = classifier.coef_[0]

    mean = np.zeros(features.shape[1], dtype=float)
    mean[informative] = scaler.mean_

    scale = np.ones(features.shape[1], dtype=float)
    scale[informative] = [value if value else 1.0 for value in scaler.scale_]

    payload = {
        "dropped_features": [name for name, keep in zip(feature_names, informative) if not keep],
        "excluded_features": sorted(excluded),
    }
    payload.update(metadata or {})

    return LinearModel(
        feature_names=list(feature_names),
        coefficients=[float(value) for value in coefficients],
        intercept=float(classifier.intercept_[0]),
        mean=[float(value) for value in mean],
        scale=[float(value) for value in scale],
        metadata=payload,
    )
