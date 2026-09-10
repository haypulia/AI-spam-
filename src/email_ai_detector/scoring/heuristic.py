from pathlib import Path
from typing import Dict, List, Optional, Union

from ..features import (
    EXPLANATION_CATEGORIES,
    FEATURE_NAMES,
    SEGMENT_FEATURE_NAMES,
    extract_features,
    iter_segments,
)
from .base import DEFAULT_THRESHOLDS, Scorer, ScoreResult, Segment, VerdictThresholds
from .explainer import CategoryExplainer
from .model import LinearModel

PathLike = Union[str, Path]

OBFUSCATION_LIMITS = {
    "obfuscation_intraword_invisible": 0.05,
    "obfuscation_mixed_script_rate": 0.02,
    "obfuscation_inner_capital_rate": 0.05,
    "obfuscation_invisible_rate": 0.02,
}

OBFUSCATION_LABELS = {
    "obfuscation_intraword_invisible": "невидимые символы внутри слов",
    "obfuscation_mixed_script_rate": "слова из смешанных алфавитов",
    "obfuscation_inner_capital_rate": "подмена букв похожими начертаниями",
    "obfuscation_invisible_rate": "служебные невидимые символы в тексте",
}

FALLBACK_CATEGORY_WEIGHTS = {
    "body": 0.35,
    "html_template": 0.15,
    "subject": 0.15,
    "cta": 0.10,
    "opener": 0.10,
    "closer": 0.10,
    "image": 0.05,
}

CATEGORY_LABELS = {
    "subject": "тема письма построена по шаблонной маркетинговой схеме",
    "opener": "приветствие оформлено формульно и без индивидуальных деталей",
    "body": "основной текст отличается ровной стилистикой без человеческих нерегулярностей",
    "cta": "призыв к действию собран из типовых конструкций",
    "closer": "подпись и завершающая формула воспроизводят стандартный шаблон",
    "html_template": "в разметке присутствуют признаки шаблонной генерации",
    "image": "текст на изображении повторяет генеративные речевые обороты",
}

FEATURE_LABELS = {
    "body_sentence_capitalization_rate": "единообразная капитализация предложений",
    "body_sentence_terminator_rate": "последовательная финальная пунктуация",
    "body_typo_marker_rate": "отсутствие опечаток и небрежностей",
    "body_marketing_phrase_rate": "маркетинговые клише",
    "body_connective_rate": "связочные обороты",
    "body_burstiness": "выровненная длина предложений",
    "body_type_token_ratio": "лексическое разнообразие",
    "body_cta_phrase_rate": "типовые призывы к действию",
    "body_closer_phrase": "шаблонная завершающая формула",
    "body_opener_phrase": "шаблонное приветствие",
    "html_template_variable": "неразрешённые шаблонные переменные",
    "html_suspicious_comment": "служебные комментарии в разметке",
    "html_duplicated_styles": "дублирование инлайновых стилей",
    "html_deep_nesting": "избыточная вложенность таблиц",
    "html_empty_cell": "пустые ячейки таблиц",
    "subject_marketing": "маркетинговые клише в теме",
    "subject_cta": "призыв к действию в теме",
    "ocr_present": "текстовый слой изображения",
}


class HeuristicScorer(Scorer):
    name = "heuristic"

    def __init__(
        self,
        model: Optional[LinearModel] = None,
        segment_model: Optional[LinearModel] = None,
        explainer: Optional[CategoryExplainer] = None,
        category_threshold: float = 0.5,
        thresholds: Optional[VerdictThresholds] = None,
    ):
        self.model = model
        self.segment_model = segment_model
        self.explainer = explainer
        self.category_threshold = category_threshold
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    @classmethod
    def from_paths(
        cls,
        model_path: Optional[PathLike] = None,
        segment_model_path: Optional[PathLike] = None,
        explainer_path: Optional[PathLike] = None,
        category_threshold: float = 0.5,
        thresholds: Optional[VerdictThresholds] = None,
    ) -> "HeuristicScorer":
        model = LinearModel.load(model_path) if model_path and Path(model_path).exists() else None
        segment_model = (
            LinearModel.load(segment_model_path)
            if segment_model_path and Path(segment_model_path).exists()
            else None
        )
        explainer = (
            CategoryExplainer.load(explainer_path)
            if explainer_path and Path(explainer_path).exists()
            else None
        )
        return cls(
            model=model,
            segment_model=segment_model,
            explainer=explainer,
            category_threshold=category_threshold,
            thresholds=thresholds,
        )

    @classmethod
    def from_settings(cls, settings, category_threshold: float = 0.5) -> "HeuristicScorer":
        return cls.from_paths(
            model_path=settings.heuristic_model_path,
            segment_model_path=settings.models_dir / "segment_model.json",
            explainer_path=settings.models_dir / "category_models.json",
            category_threshold=category_threshold,
            thresholds=VerdictThresholds(
                mixed=settings.verdict_mixed_threshold, ai=settings.verdict_ai_threshold
            ),
        )

    def _document_score(self, features) -> float:
        if self.model is not None:
            return self.model.predict_proba(features.as_list(self.model.feature_names))
        total = sum(
            FALLBACK_CATEGORY_WEIGHTS.get(name, 0.0) * value for name, value in features.categories.items()
        )
        return max(0.0, min(1.0, total))

    def score_segments(self, text: str) -> List[Segment]:
        segments: List[Segment] = []
        for item in iter_segments(text):
            if self.segment_model is not None:
                vector = [item["features"].get(name, 0.0) for name in self.segment_model.feature_names]
                score = self.segment_model.predict_proba(vector)
            else:
                features = item["features"]
                score = max(
                    0.0,
                    min(
                        1.0,
                        0.5 * features.get("segment_sentence_capitalization_rate", 0.0)
                        + 0.2 * features.get("segment_sentence_terminator_rate", 0.0)
                        + 0.2 * min(1.0, features.get("segment_marketing_phrase_rate", 0.0))
                        - 0.4 * min(1.0, features.get("segment_typo_marker_rate", 0.0))
                        + 0.3,
                    ),
                )
            segments.append(
                Segment(index=item["index"], start=item["start"], end=item["end"], text=item["text"], score=score)
            )
        return segments

    def _obfuscation_notes(self, features) -> List[str]:
        return [
            OBFUSCATION_LABELS[name]
            for name, limit in OBFUSCATION_LIMITS.items()
            if features.vector.get(name, 0.0) >= limit
        ]

    def _explanation(self, score: float, categories: Dict[str, float], features) -> str:
        parts = ["Итоговый индекс AI-генерации: %s/100." % round(score * 100, 1)]

        triggered = [
            CATEGORY_LABELS[name]
            for name in EXPLANATION_CATEGORIES
            if categories.get(name, 0.0) >= self.category_threshold and name in CATEGORY_LABELS
        ]
        if triggered:
            parts.append("Зоны письма с признаками генерации: %s." % "; ".join(triggered))

        if self.model is not None:
            vector = features.as_list(self.model.feature_names)
            drivers = [
                FEATURE_LABELS.get(name, name)
                for name, contribution in self.model.top_contributions(vector, limit=5)
                if contribution > 0
            ]
            if drivers:
                parts.append("Ключевые признаки: %s." % ", ".join(drivers))

        notes = self._obfuscation_notes(features)
        if notes:
            parts.append("Признаки маскировки текста: %s." % ", ".join(notes))

        if not triggered and not notes and score < 0.3:
            parts.append("Стилистика и разметка соответствуют письму, написанному человеком.")

        return " ".join(parts)

    def score_email(
        self,
        text: str = "",
        subject: str = "",
        html: str = "",
        ocr_text: str = "",
    ) -> ScoreResult:
        features = extract_features(text=text, subject=subject, html=html, ocr_text=ocr_text)
        score = self._document_score(features)
        categories = dict(features.categories)
        if self.explainer:
            categories.update(self.explainer.predict(features.vector))
        segments = self.score_segments(text or "")

        signals = {
            name: features.vector.get(name, 0.0)
            for name in (
                "body_sentence_capitalization_rate",
                "body_typo_marker_rate",
                "body_marketing_phrase_rate",
                "body_burstiness",
                "html_template_variable",
                "html_suspicious_comment",
                "ocr_present",
                "obfuscation_intraword_invisible",
                "obfuscation_mixed_script_rate",
                "obfuscation_inner_capital_rate",
            )
        }

        return ScoreResult(
            score=score,
            confidence=0.5 + abs(score - 0.5),
            categories=categories,
            signals=signals,
            explanation=self._explanation(score, categories, features),
            segments=segments,
            scorer=self.name,
            thresholds=self.thresholds,
            meta={
                "model": "linear" if self.model is not None else "rule_based",
                "segment_model": "linear" if self.segment_model is not None else "rule_based",
                "explainer": "linear" if self.explainer else "rule_based",
                "feature_count": len(FEATURE_NAMES),
                "segment_feature_count": len(SEGMENT_FEATURE_NAMES),
            },
        )
