def analyze_email_vector(
    html_score,
    ocr_score=0.0,
    has_ocr=False,
    html_weight=0.50,
    ocr_weight=0.50
):
    """
    Объединяет AI-generation score HTML и OCR
    в единый score письма.

    Если OCR-текст присутствует, но HTML содержит
    мало полезного содержимого, OCR получает больший вес.

    Все score внутри функции: 0.0 .. 1.0.
    """

    try:
        html_score = float(html_score)
    except (TypeError, ValueError):
        html_score = 0.0

    try:
        ocr_score = float(ocr_score)
    except (TypeError, ValueError):
        ocr_score = 0.0

    # Поддерживаем 0..1 и 0..100
    if html_score > 1:
        html_score /= 100

    if ocr_score > 1:
        ocr_score /= 100

    html_score = max(0.0, min(1.0, html_score))

    ocr_score = max(0.0, min(1.0, ocr_score))

    # Веса:

    if not has_ocr:
        # Нет OCR → полностью используем HTML
        html_weight = 1.0
        ocr_weight = 0.0

    elif html_score < 0.30:
        # если html как такового нет, то меняем распределение весов
        # OCR важнее HTML.
        html_weight = 0.30
        ocr_weight = 0.70

    else:
        # если есть OCR и HTML, то +- равеы данные
        html_weight = 0.50
        ocr_weight = 0.50

    # Итоговый score

    final_score = (html_score * html_weight + ocr_score * ocr_weight)

    final_score = max(0.0, min(1.0, final_score))

    # Verdict

    if final_score >= 0.60: verdict = "Сгенерировано ИИ (AI-Generated)"

    elif final_score >= 0.30: verdict = "Смешанный текст (AI-Assisted / Редактировано)"

    else: verdict = "Написано человеком (Human-Written)"

    return {
        "AI_Score": round(final_score, 3),

        "AI_Score_Percent": round(final_score * 100, 1),

        "Verdict": verdict,

        "Signals": {
            "html_score": round(html_score, 3),

            "ocr_score": round(ocr_score, 3),

            "html_weight": html_weight,

            "ocr_weight": ocr_weight,

            "has_ocr": has_ocr
        },

        "Explanation": (
            f"Индекс AI-генерации: "
            f"{round(final_score * 100, 1)}/100. "
            f"HTML: {round(html_score * 100, 1)}/100. "
            f"OCR: {round(ocr_score * 100, 1)}/100. "
            f"Веса: HTML {html_weight:.0%}, "
            f"OCR {ocr_weight:.0%}."
        )
    }

def analyze_chunk_vector(chunk_data, weights=None):
    
    # 1. Веса признаков (сумма = 1.0). Подстроено под флаги твоей команды.
    if weights is None:
        weights = {
            "llm_score": 0.35,           # Базовая оценка вероятности от DeepSeek
            "template_variable": 0.20,   # Шаблонная переменная (н-р, [ИМЯ]) - сильный маркер ИИ
            "repeating_pattern": 0.15,   # Повторяющийся паттерн (галлюцинация или цикл)
            "suspicious_comments": 0.10, # Подозрительные комментарии в коде письма
            "deep_nesting": 0.05,        # Слишком глубокая вложенность
            "duplicated_styles": 0.05,   # Дублирование стилей
            "empty_cell": 0.10           # Пустая ячейка (ошибка генерации таблиц)
        }

    llm_score = float(chunk_data.get("llm_probability", 0.0))

    if llm_score > 1: llm_score /= 100

    llm_score = max(0.0, min(1.0, llm_score))

    # 2. Формируем вектор признаков x (все значения от 0.0 до 1.0)
    x = {
        "llm_score": float(chunk_data.get("llm_probability", 0.0)),
        "template_variable": 1.0 if chunk_data.get("template_variable", False) else 0.0,
        "repeating_pattern": 1.0 if chunk_data.get("repeating_pattern", False) else 0.0,
        "suspicious_comments": 1.0 if chunk_data.get("suspicious_comments", False) else 0.0,
        "deep_nesting": 1.0 if chunk_data.get("deep_nesting", False) else 0.0,
        "duplicated_styles": 1.0 if chunk_data.get("duplicated_styles", False) else 0.0,
        "empty_cell": 1.0 if chunk_data.get("empty_cell", False) else 0.0
    }

    # 3. Скалярное произведение векторов
    ai_score = sum(x[key] * weights[key] for key in weights)

    # 4. Расчет процента уверенности (Confidence Score) через дисперсию
    signals = list(x.values())
    mean_signal = sum(signals) / len(signals)
    variance = sum((s - mean_signal) ** 2 for s in signals) / len(signals)
    
    base_confidence = 0.98 - (variance * 0.8)
    confidence_percentage = round(max(0.50, min(0.99, base_confidence)) * 100, 1)

    # 5. Категоризация вердикта по уровню ИИ-генерации
    if ai_score >= 0.70:
        verdict = "Сгенерировано ИИ (AI-Generated)"
    elif ai_score >= 0.40:
        verdict = "Смешанный текст (AI-Assisted / Редактировано)"
    else:
        verdict = "Написано человеком (Human-Written)"

    # 6. Генерация динамического объяснения на основе сработавших флагов
    reasons = [f"Индекс AI-генерации: {round(ai_score * 100, 1)}/100."]
    
    if x["llm_score"] > 0.7:
        reasons.append("Модель выявила высокую базовую вероятность машинного текста.")
    if x["template_variable"] > 0:
        reasons.append("Найдены неразрешенные шаблонные переменные (признак автоматизации).")
    if x["repeating_pattern"] > 0:
        reasons.append("Обнаружены неестественные повторяющиеся текстовые или структурные паттерны.")
    if x["suspicious_comments"] > 0:
        reasons.append("В разметке присутствуют подозрительные или автогенерируемые комментарии.")
    
    # Группируем мелкие структурные аномалии в одно красивое предложение
    structural_anomalies = []
    if x["deep_nesting"] > 0: structural_anomalies.append("глубокая вложенность")
    if x["duplicated_styles"] > 0: structural_anomalies.append("дублирование стилей")
    if x["empty_cell"] > 0: structural_anomalies.append("пустые ячейки")
    
    if structural_anomalies:
        reasons.append(f"Аномалии форматирования ({', '.join(structural_anomalies)}), характерные для машинной генерации HTML.")

    explanation_str = " ".join(reasons)

    return {
        "AI_Score": round(ai_score, 3),
        "AI_Score_Percent": round(ai_score * 100, 1),
        "Verdict": verdict,
        "Confidence": f"{confidence_percentage}%",
        "Explanation": explanation_str,
        "Signals": {key: round(value, 3) for key, value in x.items()}
    }