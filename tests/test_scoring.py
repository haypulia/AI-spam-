from email_ai_detector.scoring.base import AI_THRESHOLD, MIXED_THRESHOLD, verdict_for_score
from email_ai_detector.scoring.engine import analyze_chunk_vector, analyze_email_vector, normalize_score
from email_ai_detector.scoring.heuristic import HeuristicScorer


def test_score_normalization_accepts_percent_scale():
    assert normalize_score(80) == 0.8
    assert normalize_score("bad", default=0.1) == 0.1
    assert normalize_score(150) == 1.0
    assert normalize_score(0.42) == 0.42


def test_verdict_thresholds():
    assert verdict_for_score(AI_THRESHOLD) != verdict_for_score(MIXED_THRESHOLD)
    assert verdict_for_score(0.0) == verdict_for_score(MIXED_THRESHOLD - 0.01)


def test_chunk_vector_weights_sum_to_one():
    result = analyze_chunk_vector(
        {
            "llm_probability": 100,
            "template_variable": True,
            "repeating_pattern": True,
            "suspicious_comments": True,
            "deep_nesting": True,
            "duplicated_styles": True,
            "empty_cell": True,
        }
    )
    assert result["AI_Score"] == 1.0


def test_email_vector_uses_ocr_weights():
    without_ocr = analyze_email_vector(0.8, 0.2, has_ocr=False)
    with_ocr = analyze_email_vector(0.8, 0.2, has_ocr=True)
    assert without_ocr["AI_Score"] > with_ocr["AI_Score"]


def test_heuristic_scorer_returns_bounded_score():
    scorer = HeuristicScorer()
    result = scorer.score_email(text="Hello. This is a short note.", subject="Note", html="<p>note</p>")
    assert 0.0 <= result.score <= 1.0
    assert result.explanation
    assert result.segments
