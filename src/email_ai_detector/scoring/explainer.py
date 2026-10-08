import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ..features import (
    EXPLANATION_CATEGORIES,
    FEATURE_NAMES,
    EmailFeatures,
    extract_features,
    normalize_category,
)
from .model import LinearModel, train_linear_model

PathLike = Union[str, Path]

CATEGORY_FEATURE_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "subject": ("subject_",),
    "opener": (
        "opener_",
        "body_opener_phrase",
        "body_generic_greeting",
        "body_sentence_capitalization_rate",
        "body_typo_marker_rate",
    ),
    "body": ("body_",),
    "cta": (
        "body_cta_phrase_rate",
        "body_imperative_opening_rate",
        "body_urgency_rate",
        "body_url_rate",
        "subject_cta",
        "subject_urgency",
    ),
    "closer": (
        "closer_",
        "body_closer_phrase",
        "body_sentence_capitalization_rate",
        "body_sentence_terminator_rate",
        "body_typo_marker_rate",
    ),
    "html_template": ("html_",),
}

FEATURE_EXPLANATIONS: Dict[str, str] = {
    "body_cta_phrase_rate": "Присутствие шаблонных фраз побуждения к действию (CTA)",
    "body_imperative_opening_rate": "Синтаксис предложений начинается с директивных императивов",
    "body_urgency_rate": "Искусственное нагнетание срочности (психологическое давление)",
    "body_url_rate": "Аномальная плотность гиперссылок",
    "subject_cta": "Тема письма содержит призыв к немедленному действию",
    "closer_signoff": "Стандартная формула вежливости в подписи",
    "closer_capitalized": "Строгое следование правилам капитализации подписи (отсутствие небрежности)",
    "body_closer_phrase": "Лексический маркер закрытия письма характерен для LLM-шаблонов",
    "body_typo_marker_rate": "Синтетическая чистота текста (полное отсутствие опечаток)",
    "html_template_variable": "Обнаружены переменные шаблонизации или незаполненные плейсхолдеры",
    "html_suspicious_comment": "Подозрительные служебные маркеры/комментарии в разметке",
}


@dataclass
class SOCZoneExplanation:
    category: str
    probability: float
    is_ai_generated: bool
    reasons: List[str]
    snippet: Optional[str] = None


@dataclass
class SOCAnalysisReport:
    email_verdict: str
    severity: str
    flagged_categories: List[str]
    zones: Dict[str, SOCZoneExplanation]
    analyst_summary: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def explain_for_soc(
    scores: Dict[str, float],
    vector: Dict[str, float],
    evidence: Optional[Dict] = None,
    threshold: float = 0.55,
) -> SOCAnalysisReport:
    evidence = evidence or {}
    zones: Dict[str, SOCZoneExplanation] = {}
    flagged: List[str] = []
    analyst_summary: List[str] = []

    for category, prob in scores.items():
        is_ai = prob >= threshold
        reasons = []

        prefixes = CATEGORY_FEATURE_PREFIXES.get(category, ())
        for feat_name, val in vector.items():
            if any(feat_name.startswith(p) for p in prefixes) and val > 0.0:
                readable = FEATURE_EXPLANATIONS.get(feat_name)
                if readable:
                    reasons.append(f"{readable} (значение: {val:.2f})")

        snippet = None
        if category == "cta":
            snippets = evidence.get("cta_snippets", [])
            snippet = " | ".join(snippets) if snippets else None
        elif category == "closer":
            snippet = evidence.get("closer_snippet")
        elif category == "subject":
            snippet = evidence.get("subject")

        if is_ai:
            flagged.append(category)
            if not reasons:
                reasons.append("Модель зафиксировала аномально низкую энтропию/высокую предсказуемость токенов")
            analyst_summary.append(
                f"Зона [{category.upper()}]: вероятность генерации {prob:.1%}. Факторы: {'; '.join(reasons[:2])}."
            )

        zones[category] = SOCZoneExplanation(
            category=category,
            probability=round(prob, 4),
            is_ai_generated=is_ai,
            reasons=reasons,
            snippet=snippet,
        )

    # Итоговый вердикт и критичность (вынесены за пределы цикла for)
    if not flagged:
        verdict = "HUMAN_WRITTEN"
        severity = "LOW"
    elif "cta" in flagged:
        verdict = "AI_ASSISTED_PHISHING_OR_SPAM"
        severity = "HIGH"
    elif len(flagged) >= 3:
        verdict = "FULLY_AI_GENERATED"
        severity = "HIGH"
    else:
        verdict = "PARTIALLY_AI_GENERATED"
        severity = "MEDIUM"

    return SOCAnalysisReport(
        email_verdict=verdict,
        severity=severity,
        flagged_categories=flagged,
        zones=zones,
        analyst_summary=analyst_summary if analyst_summary else ["Все зоны выглядят естественно."],
    )


def category_feature_names(category: str) -> List[str]:
    prefixes = CATEGORY_FEATURE_PREFIXES.get(category, ())
    names = [name for name in FEATURE_NAMES if any(name.startswith(prefix) for prefix in prefixes)]
    return names or list(FEATURE_NAMES)


class CategoryExplainer:
    def __init__(self, models: Optional[Dict[str, LinearModel]] = None):
        self.models = models or {}

    def __bool__(self) -> bool:
        return bool(self.models)

    def explain(
        self,
        features: EmailFeatures,
        threshold: float = 0.55,
    ) -> SOCAnalysisReport:
        scores = self.predict(features.vector)
        for cat, heur_score in features.categories.items():
            if cat not in scores:
                scores[cat] = heur_score
        return explain_for_soc(
            scores=scores,
            vector=features.vector,
            evidence=features.evidence,
            threshold=threshold,
        )

    def predict(self, vector: Dict[str, float]) -> Dict[str, float]:
        scores = {}
        for category, model in self.models.items():
            values = [float(vector.get(name, 0.0)) for name in model.feature_names]
            scores[category] = model.predict_proba(values)
        return scores

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {category: model.to_dict() for category, model in self.models.items()}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: PathLike) -> "CategoryExplainer":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        models = {}
        for category, data in payload.items():
            models[category] = LinearModel(
                feature_names=list(data["feature_names"]),
                coefficients=[float(value) for value in data["coefficients"]],
                intercept=float(data.get("intercept", 0.0)),
                mean=[float(value) for value in data.get("mean", [])],
                scale=[float(value) for value in data.get("scale", [])],
                metadata=data.get("metadata", {}),
            )
        return cls(models)


def build_category_dataset(records: Sequence, category: str) -> Tuple[List[List[float]], List[int]]:
    names = category_feature_names(category)
    matrix: List[List[float]] = []
    labels: List[int] = []
    for record in records:
        features = extract_features(
            text=record.text, subject=record.subject, html=record.html
        )
        matrix.append([features.vector.get(name, 0.0) for name in names])
        reference = {normalize_category(str(item).strip().lower()) for item in record.ai_elements}
        labels.append(1 if category in reference else 0)
    return matrix, labels


def train_category_models(records: Sequence, metadata: Optional[dict] = None) -> CategoryExplainer:
    models: Dict[str, LinearModel] = {}
    for category in EXPLANATION_CATEGORIES:
        matrix, labels = build_category_dataset(records, category)
        if len(set(labels)) < 2:
            continue
        payload = {"category": category, "samples": len(labels), "positives": sum(labels)}
        payload.update(metadata or {})
        try:
            from .training import VOLUME_FEATURES

            models[category] = train_linear_model(
                matrix,
                labels,
                category_feature_names(category),
                metadata=payload,
                excluded=VOLUME_FEATURES,
            )
        except ValueError:
            continue
    return CategoryExplainer(models)