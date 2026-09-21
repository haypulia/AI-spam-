import pytest

from email_ai_detector.scoring.analysis import EmailAnalysis
from email_ai_detector.scoring.artifacts import (
    ATTACHMENT_KIND,
    IMAGE_KIND,
    ArtifactAnalyzer,
    ArtifactRegistry,
    ArtifactResult,
)
from email_ai_detector.scoring.base import VerdictThresholds
from email_ai_detector.scoring.heuristic import HeuristicScorer


class StubImageAnalyzer(ArtifactAnalyzer):
    name = "stub"
    kind = IMAGE_KIND
    content_types = ("image/",)

    def analyze(self, payload: bytes, source: str = "", content_type: str = "") -> ArtifactResult:
        return ArtifactResult(
            kind=self.kind,
            source=source,
            content_type=content_type,
            score=len(payload) / 100.0,
            analyzer=self.name,
        )


def test_result_takes_verdict_from_thresholds():
    result = ArtifactResult(kind=IMAGE_KIND, score=0.5, thresholds=VerdictThresholds(0.4, 0.8))
    assert result.verdict == ArtifactResult(kind=IMAGE_KIND, score=0.45, thresholds=VerdictThresholds(0.4, 0.8)).verdict
    assert result.to_dict()["verdict_thresholds"] == {"mixed": 0.4, "ai": 0.8}


def test_registry_picks_analyzer_by_content_type():
    registry = ArtifactRegistry([StubImageAnalyzer()])
    assert registry.analyzer_for("image/png") is not None
    assert registry.analyzer_for("application/pdf") is None
    assert registry.analyze(b"x" * 50, content_type="application/pdf") is None


def test_registry_analyzes_only_supported_artifacts():
    registry = ArtifactRegistry([StubImageAnalyzer()])
    results = registry.analyze_many(
        [
            {"bytes": b"x" * 50, "source": "banner.png", "content_type": "image/png"},
            {"bytes": b"y" * 10, "source": "doc.pdf", "content_type": "application/pdf"},
        ]
    )
    assert [item.source for item in results] == ["banner.png"]
    assert results[0].score == 0.5


def test_empty_registry_keeps_pipeline_silent():
    assert ArtifactRegistry().analyze_many([{"bytes": b"x", "content_type": "image/png"}]) == []


def test_email_analysis_splits_artifacts_by_kind():
    text = HeuristicScorer().score_email(text="Короткое письмо.", subject="Тема")
    analysis = EmailAnalysis.from_results(
        text,
        [
            ArtifactResult(kind=IMAGE_KIND, source="a.png", score=0.9),
            ArtifactResult(kind=ATTACHMENT_KIND, source="b.pdf", score=0.2),
        ],
    )
    payload = analysis.to_dict()
    assert [item["source"] for item in payload["images"]] == ["a.png"]
    assert [item["source"] for item in payload["attachments"]] == ["b.pdf"]
    assert payload["text"]["ai_score"] == text.to_dict()["ai_score"]
    assert analysis.max_artifact_score == 0.9


def test_analyzer_requires_implementation():
    with pytest.raises(TypeError):
        ArtifactAnalyzer()
