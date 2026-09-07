from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

HUMAN_VERDICT = "Написано человеком (Human-Written)"
MIXED_VERDICT = "Смешанный текст (AI-Assisted)"
AI_VERDICT = "Сгенерировано ИИ (AI-Generated)"

MIXED_THRESHOLD = 0.30
AI_THRESHOLD = 0.60


def verdict_for_score(score: float) -> str:
    if score >= AI_THRESHOLD:
        return AI_VERDICT
    if score >= MIXED_THRESHOLD:
        return MIXED_VERDICT
    return HUMAN_VERDICT


@dataclass
class Segment:
    index: int
    start: int
    end: int
    text: str
    score: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "score": round(self.score, 4),
        }


@dataclass
class ScoreResult:
    score: float
    verdict: str = ""
    confidence: float = 0.0
    categories: Dict[str, float] = field(default_factory=dict)
    signals: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    segments: List[Segment] = field(default_factory=list)
    scorer: str = ""
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.score = max(0.0, min(1.0, float(self.score)))
        if not self.verdict:
            self.verdict = verdict_for_score(self.score)

    @property
    def score_percent(self) -> float:
        return round(self.score * 100, 1)

    def reported_categories(self, threshold: float = 0.5) -> List[str]:
        return [name for name, value in sorted(self.categories.items()) if value >= threshold]

    def to_dict(self, threshold: float = 0.5) -> dict:
        return {
            "scorer": self.scorer,
            "ai_score": round(self.score, 4),
            "ai_score_percent": self.score_percent,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "categories": {name: round(value, 4) for name, value in self.categories.items()},
            "reported_categories": self.reported_categories(threshold),
            "signals": {name: round(float(value), 4) for name, value in self.signals.items()},
            "explanation": self.explanation,
            "segments": [segment.to_dict() for segment in self.segments],
            "meta": self.meta,
        }


class Scorer(ABC):
    name = "base"

    @abstractmethod
    def score_email(
        self,
        text: str = "",
        subject: str = "",
        html: str = "",
        ocr_text: str = "",
    ) -> ScoreResult:
        raise NotImplementedError

    def score_records(self, records, progress: Optional[callable] = None) -> List[ScoreResult]:
        results = []
        for index, record in enumerate(records):
            results.append(
                self.score_email(
                    text=getattr(record, "text", "") or "",
                    subject=getattr(record, "subject", "") or "",
                    html=getattr(record, "html", "") or "",
                    ocr_text=getattr(record, "ocr_text", "") or "",
                )
            )
            if progress:
                progress(index + 1, len(records))
        return results
