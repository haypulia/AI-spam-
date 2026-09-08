from typing import Dict, List, Optional

from .base import DEFAULT_THRESHOLDS, Scorer, ScoreResult, VerdictThresholds


class EnsembleScorer(Scorer):
    name = "ensemble"

    def __init__(
        self,
        scorers: List[Scorer],
        weights: Dict[str, float] = None,
        thresholds: Optional[VerdictThresholds] = None,
    ):
        if not scorers:
            raise ValueError("ensemble requires at least one scorer")
        self.scorers = scorers
        self.weights = weights or {scorer.name: 1.0 / len(scorers) for scorer in scorers}
        self.thresholds = thresholds or getattr(scorers[0], "thresholds", DEFAULT_THRESHOLDS)

    def score_email(
        self,
        text: str = "",
        subject: str = "",
        html: str = "",
        ocr_text: str = "",
    ) -> ScoreResult:
        results = [
            scorer.score_email(text=text, subject=subject, html=html, ocr_text=ocr_text)
            for scorer in self.scorers
        ]

        total_weight = sum(self.weights.get(result.scorer, 0.0) for result in results) or 1.0
        score = sum(result.score * self.weights.get(result.scorer, 0.0) for result in results) / total_weight

        categories: Dict[str, float] = {}
        signals: Dict[str, float] = {}
        for result in results:
            for name, value in result.categories.items():
                categories[name] = max(categories.get(name, 0.0), value)
            for name, value in result.signals.items():
                signals["%s_%s" % (result.scorer, name)] = value

        segments = []
        for result in results:
            if result.segments:
                segments = result.segments
                break

        return ScoreResult(
            score=score,
            confidence=sum(result.confidence for result in results) / len(results),
            categories=categories,
            signals=signals,
            explanation=" ".join(result.explanation for result in results if result.explanation),
            segments=segments,
            scorer=self.name,
            thresholds=self.thresholds,
            meta={"components": {result.scorer: round(result.score, 4) for result in results}},
        )
