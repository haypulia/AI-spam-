import io

import pytest

from email_ai_detector.scoring.artifacts import IMAGE_KIND, ArtifactRegistry
from email_ai_detector.scoring.base import VerdictThresholds
from email_ai_detector.scoring.image import ImageAnalyzer

Image = pytest.importorskip("PIL.Image")


def encode(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def flat_logo() -> bytes:
    return encode(Image.new("RGB", (400, 160), (245, 245, 245)))


def busy_banner() -> bytes:
    image = Image.new("RGB", (600, 300))
    image.putdata([((x * 7) % 256, (y * 11) % 256, (x * y) % 256) for y in range(300) for x in range(600)])
    return encode(image)


def test_analyzer_declares_image_contract():
    analyzer = ImageAnalyzer()
    assert analyzer.kind == IMAGE_KIND
    assert analyzer.supports("image/png")
    assert not analyzer.supports("application/pdf")


def test_result_carries_source_and_metadata():
    result = ImageAnalyzer().analyze(flat_logo(), source="logo.png", content_type="image/png")
    assert result.source == "logo.png"
    assert result.meta["decoded"] is True
    assert result.meta["format"] == "PNG"
    assert (result.meta["width"], result.meta["height"]) == (400, 160)
    assert 0.0 <= result.score <= 1.0


def test_broken_payload_is_not_scored():
    result = ImageAnalyzer().analyze(b"definitely not an image", source="x.png", content_type="image/png")
    assert result.score == 0.0
    assert result.meta["decoded"] is False
    assert "не удалось разобрать" in result.explanation


def test_busy_image_scores_above_flat_one():
    analyzer = ImageAnalyzer()
    assert analyzer.analyze(busy_banner()).score > analyzer.analyze(flat_logo()).score


def test_thresholds_reach_the_verdict():
    analyzer = ImageAnalyzer(thresholds=VerdictThresholds(mixed=0.01, ai=0.02))
    result = analyzer.analyze(busy_banner(), content_type="image/png")
    assert result.to_dict()["verdict_thresholds"] == {"mixed": 0.01, "ai": 0.02}


def test_registry_routes_images_to_analyzer():
    registry = ArtifactRegistry([ImageAnalyzer()])
    results = registry.analyze_many(
        [
            {"bytes": flat_logo(), "source": "logo.png", "content_type": "image/png"},
            {"bytes": b"%PDF-1.4", "source": "doc.pdf", "content_type": "application/pdf"},
        ]
    )
    assert [item.source for item in results] == ["logo.png"]
    assert results[0].analyzer == "image_signals"


def test_dimension_features_are_reported_as_excluded():
    result = ImageAnalyzer().analyze(flat_logo())
    assert "image_width_log" in result.meta["excluded_features"]
