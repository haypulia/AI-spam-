from dataclasses import dataclass, field
from typing import List, Sequence

from .artifacts import ATTACHMENT_KIND, IMAGE_KIND, ArtifactResult
from .base import ScoreResult


@dataclass
class EmailAnalysis:
    text: ScoreResult
    images: List[ArtifactResult] = field(default_factory=list)
    attachments: List[ArtifactResult] = field(default_factory=list)

    @classmethod
    def from_results(cls, text: ScoreResult, artifacts: Sequence[ArtifactResult] = ()) -> "EmailAnalysis":
        return cls(
            text=text,
            images=[item for item in artifacts if item.kind == IMAGE_KIND],
            attachments=[item for item in artifacts if item.kind == ATTACHMENT_KIND],
        )

    @property
    def artifacts(self) -> List[ArtifactResult]:
        return list(self.images) + list(self.attachments)

    @property
    def max_artifact_score(self) -> float:
        scores = [item.score for item in self.artifacts]
        return max(scores) if scores else 0.0

    def to_dict(self, category_threshold: float = 0.5) -> dict:
        return {
            "text": self.text.to_dict(category_threshold),
            "images": [item.to_dict() for item in self.images],
            "attachments": [item.to_dict() for item in self.attachments],
        }
