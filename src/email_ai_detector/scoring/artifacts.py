from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .base import DEFAULT_THRESHOLDS, VerdictThresholds, verdict_for_score

IMAGE_KIND = "image"
ATTACHMENT_KIND = "attachment"


@dataclass
class ArtifactResult:
    kind: str
    source: str = ""
    content_type: str = ""
    score: float = 0.0
    verdict: str = ""
    confidence: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    analyzer: str = ""
    meta: Dict[str, object] = field(default_factory=dict)
    thresholds: VerdictThresholds = DEFAULT_THRESHOLDS

    def __post_init__(self):
        self.score = max(0.0, min(1.0, float(self.score)))
        if not self.verdict:
            self.verdict = verdict_for_score(self.score, self.thresholds)

    @property
    def score_percent(self) -> float:
        return round(self.score * 100, 1)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "source": self.source,
            "content_type": self.content_type,
            "analyzer": self.analyzer,
            "ai_score": round(self.score, 4),
            "ai_score_percent": self.score_percent,
            "verdict": self.verdict,
            "verdict_thresholds": self.thresholds.to_dict(),
            "confidence": round(self.confidence, 4),
            "signals": {name: round(float(value), 4) for name, value in self.signals.items()},
            "explanation": self.explanation,
            "meta": self.meta,
        }


class ArtifactAnalyzer(ABC):
    name = "base"
    kind = ATTACHMENT_KIND
    content_types: Tuple[str, ...] = ()
    thresholds: VerdictThresholds = DEFAULT_THRESHOLDS

    def supports(self, content_type: str = "") -> bool:
        if not self.content_types:
            return True
        lowered = (content_type or "").lower()
        return any(lowered.startswith(prefix) for prefix in self.content_types)

    @abstractmethod
    def analyze(self, payload: bytes, source: str = "", content_type: str = "") -> ArtifactResult:
        raise NotImplementedError


class ArtifactRegistry:
    def __init__(self, analyzers: Optional[Sequence[ArtifactAnalyzer]] = None):
        self.analyzers: List[ArtifactAnalyzer] = list(analyzers or ())

    def __len__(self) -> int:
        return len(self.analyzers)

    def register(self, analyzer: ArtifactAnalyzer) -> "ArtifactRegistry":
        self.analyzers.append(analyzer)
        return self

    def analyzer_for(self, content_type: str = "") -> Optional[ArtifactAnalyzer]:
        for analyzer in self.analyzers:
            if analyzer.supports(content_type):
                return analyzer
        return None

    def analyze(self, payload: bytes, source: str = "", content_type: str = "") -> Optional[ArtifactResult]:
        analyzer = self.analyzer_for(content_type)
        if analyzer is None:
            return None
        return analyzer.analyze(payload, source=source, content_type=content_type)

    def analyze_many(self, artifacts: Sequence[dict]) -> List[ArtifactResult]:
        results = []
        for artifact in artifacts:
            result = self.analyze(
                artifact.get("bytes", b""),
                source=artifact.get("source", ""),
                content_type=artifact.get("content_type", ""),
            )
            if result is not None:
                results.append(result)
        return results
