from typing import Dict, Optional

from .base import AI_THRESHOLD, MIXED_THRESHOLD, verdict_for_score

CHUNK_WEIGHTS = {
    "llm_score": 0.35,
    "template_variable": 0.20,
    "repeating_pattern": 0.15,
    "suspicious_comments": 0.10,
    "deep_nesting": 0.05,
    "duplicated_styles": 0.05,
    "empty_cell": 0.10,
}

CHUNK_AI_THRESHOLD = 0.70
CHUNK_MIXED_THRESHOLD = 0.40

SIGNAL_DESCRIPTIONS = {
    "template_variable": "неразрешённые шаблонные переменные",
    "repeating_pattern": "повторяющиеся текстовые или структурные паттерны",
    "suspicious_comments": "подозрительные комментарии в разметке",
    "deep_nesting": "глубокая вложенность таблиц",
    "duplicated_styles": "дублирование инлайновых стилей",
    "empty_cell": "пустые ячейки таблиц",
}


def normalize_score(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value > 1:
        value /= 100
    return max(0.0, min(1.0, value))


def confidence_from_signals(signals) -> float:
    values = list(signals)
    if not values:
        return 0.5
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return max(0.50, min(0.99, 0.98 - variance * 0.8))


def analyze_chunk_vector(chunk_data: dict, weights: Optional[Dict[str, float]] = None) -> dict:
    weights = weights or CHUNK_WEIGHTS

    vector = {
        "llm_score": normalize_score(chunk_data.get("llm_probability", 0.0)),
        "template_variable": 1.0 if chunk_data.get("template_variable") else 0.0,
        "repeating_pattern": 1.0 if chunk_data.get("repeating_pattern") else 0.0,
        "suspicious_comments": 1.0 if chunk_data.get("suspicious_comments") else 0.0,
        "deep_nesting": 1.0 if chunk_data.get("deep_nesting") else 0.0,
        "duplicated_styles": 1.0 if chunk_data.get("duplicated_styles") else 0.0,
        "empty_cell": 1.0 if chunk_data.get("empty_cell") else 0.0,
    }

    ai_score = sum(vector[key] * weights.get(key, 0.0) for key in vector)
    ai_score = max(0.0, min(1.0, ai_score))
    confidence = confidence_from_signals(vector.values())

    if ai_score >= CHUNK_AI_THRESHOLD:
        verdict = verdict_for_score(1.0)
    elif ai_score >= CHUNK_MIXED_THRESHOLD:
        verdict = verdict_for_score(MIXED_THRESHOLD)
    else:
        verdict = verdict_for_score(0.0)

    reasons = ["Индекс AI-генерации: %s/100." % round(ai_score * 100, 1)]
    if vector["llm_score"] > 0.7:
        reasons.append("Модель оценивает фрагмент как машинный текст с высокой вероятностью.")

    triggered = [
        SIGNAL_DESCRIPTIONS[key]
        for key in SIGNAL_DESCRIPTIONS
        if vector.get(key, 0.0) > 0
    ]
    if triggered:
        reasons.append("Сработавшие структурные сигналы: %s." % ", ".join(triggered))

    return {
        "AI_Score": round(ai_score, 3),
        "AI_Score_Percent": round(ai_score * 100, 1),
        "Verdict": verdict,
        "Confidence": round(confidence * 100, 1),
        "Explanation": " ".join(reasons),
        "Signals": {key: round(value, 3) for key, value in vector.items()},
    }


def analyze_email_vector(html_score, ocr_score=0.0, has_ocr: bool = False) -> dict:
    html_score = normalize_score(html_score)
    ocr_score = normalize_score(ocr_score)

    if not has_ocr:
        html_weight, ocr_weight = 1.0, 0.0
    elif html_score < 0.30:
        html_weight, ocr_weight = 0.30, 0.70
    else:
        html_weight, ocr_weight = 0.50, 0.50

    final_score = max(0.0, min(1.0, html_score * html_weight + ocr_score * ocr_weight))

    return {
        "AI_Score": round(final_score, 3),
        "AI_Score_Percent": round(final_score * 100, 1),
        "Verdict": verdict_for_score(final_score),
        "Signals": {
            "html_score": round(html_score, 3),
            "ocr_score": round(ocr_score, 3),
            "html_weight": html_weight,
            "ocr_weight": ocr_weight,
            "has_ocr": has_ocr,
        },
        "Explanation": (
            "Индекс AI-генерации: %s/100. HTML: %s/100. OCR: %s/100. Веса: HTML %.0f%%, OCR %.0f%%."
            % (
                round(final_score * 100, 1),
                round(html_score * 100, 1),
                round(ocr_score * 100, 1),
                html_weight * 100,
                ocr_weight * 100,
            )
        ),
    }
