from typing import Optional

from .base import (
    AI_THRESHOLD,
    DEFAULT_THRESHOLDS,
    MIXED_THRESHOLD,
    ScoreResult,
    Scorer,
    Segment,
    VerdictThresholds,
    verdict_for_score,
)
from .engine import analyze_chunk_vector, analyze_email_vector
from .ensemble import EnsembleScorer
from .heuristic import HeuristicScorer
from .model import LinearModel, train_linear_model

SCORER_NAMES = ("heuristic", "llm", "ensemble")


def build_scorer(name: str, settings, model: Optional[str] = None) -> Scorer:
    name = (name or "heuristic").lower()
    if name == "heuristic":
        return HeuristicScorer.from_settings(settings)
    if name == "llm":
        from .llm import LLMScorer

        return LLMScorer.from_settings(settings, model=model)
    if name == "ensemble":
        from .llm import LLMScorer

        return EnsembleScorer(
            [HeuristicScorer.from_settings(settings), LLMScorer.from_settings(settings, model=model)],
            weights={"heuristic": 0.5, "llm": 0.5},
        )
    raise ValueError("unknown scorer: %s" % name)


__all__ = [
    "AI_THRESHOLD",
    "DEFAULT_THRESHOLDS",
    "MIXED_THRESHOLD",
    "VerdictThresholds",
    "ScoreResult",
    "Scorer",
    "Segment",
    "verdict_for_score",
    "analyze_chunk_vector",
    "analyze_email_vector",
    "EnsembleScorer",
    "HeuristicScorer",
    "LinearModel",
    "train_linear_model",
    "build_scorer",
    "SCORER_NAMES",
]

from .training import build_document_dataset, build_segment_dataset, train_document_model, train_segment_model

__all__ += [
    "build_document_dataset",
    "build_segment_dataset",
    "train_document_model",
    "train_segment_model",
]
