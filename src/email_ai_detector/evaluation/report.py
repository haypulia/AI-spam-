from pathlib import Path
from typing import Dict, List, Optional


def _format_number(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return ("%%.%df" % digits) % value
    return str(value)


def _table(headers: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def render_markdown(report: Dict[str, object], roc_image: Optional[str] = None) -> str:
    detection = report["detection"]
    explanations = report["explanations"]
    partial = report["partial_generation"]
    masking = report.get("masking_robustness")

    lines: List[str] = []
    lines.append("# Метрики детектора AI-generated писем")
    lines.append("")
    lines.append("- Датасет: `%s`" % report["dataset"])
    lines.append("- Выборка: `%s`, писем: %s" % (report["split"], detection["count"]))
    lines.append("- Скорер: `%s`" % report["scorer"])
    lines.append("- Порог решения: %s" % report["decision_threshold"])
    lines.append("- Дата расчёта: %s" % report["generated_at"])
    lines.append("")

    lines.append("## 1. Базовая метрика качества")
    lines.append("")
    ci = detection["roc_auc_ci95"]
    lines.extend(
        _table(
            ["Метрика", "Значение"],
            [
                ["AUC-ROC", _format_number(detection["roc_auc"])],
                [
                    "AUC-ROC, 95%% CI (bootstrap, %s итераций)" % ci.get("iterations", 0),
                    "%s - %s" % (_format_number(ci["low"]), _format_number(ci["high"])),
                ],
                ["PR-AUC", _format_number(detection["pr_auc"])],
                ["Brier score", _format_number(detection["brier"])],
                ["Положительных примеров", str(detection["positives"])],
                ["Отрицательных примеров", str(detection["negatives"])],
            ],
        )
    )

    threshold_rows = []
    for key, title in (
        ("at_decision_threshold", "калиброванный порог"),
        ("at_operating_threshold", "рабочий порог вердикта"),
        ("best_f1", "порог максимума F1"),
    ):
        payload = detection[key]
        threshold_rows.append(
            [
                title,
                _format_number(payload["threshold"], 3),
                _format_number(payload["precision"]),
                _format_number(payload["recall"]),
                _format_number(payload["f1"]),
                _format_number(payload["accuracy"]),
                _format_number(payload["balanced_accuracy"]),
            ]
        )
    lines.extend(
        _table(
            ["Режим", "Порог", "Precision", "Recall", "F1", "Accuracy", "Balanced acc."],
            threshold_rows,
        )
    )

    if roc_image:
        lines.append("![ROC](%s)" % roc_image)
        lines.append("")

    lines.append("### Разрезы по данным")
    lines.append("")
    for title, key in (
        ("Язык", "by_language"),
        ("Модель-генератор (против human)", "by_generator_model"),
        ("Тип письма", "by_data_type"),
        ("Способ генерации (против human)", "by_origin"),
    ):
        rows = [
            [name, str(payload["count"]), _format_number(payload["roc_auc"]), _format_number(payload["mean_score"])]
            for name, payload in detection[key].items()
        ]
        lines.append("**%s**" % title)
        lines.append("")
        lines.extend(_table(["Значение", "N", "AUC-ROC", "Средний скор"], rows))

    lines.append("## 2. Полнота объяснений")
    lines.append("")
    lines.append(
        "Объяснение считается полным, если детектор указал те же зоны письма (`subject`, `opener`, `body`, "
        "`cta`, `closer`, `html_template`, `image`), что размечены в датасете."
    )
    lines.append("")
    lines.extend(
        _table(
            ["Метрика", "Значение"],
            [
                ["Писем в оценке", str(explanations["evaluated_emails"])],
                ["Полнота объяснений (macro recall)", _format_number(explanations["explanation_recall_macro"])],
                ["Точность объяснений (macro precision)", _format_number(explanations["explanation_precision_macro"])],
                ["F1 объяснений (macro)", _format_number(explanations["explanation_f1_macro"])],
                ["Полнота объяснений (micro recall)", _format_number(explanations["explanation_recall_micro"])],
                ["Доля писем с хотя бы одной верной зоной", _format_number(explanations["coverage_at_least_one"])],
                ["Полное совпадение набора зон", _format_number(explanations["exact_set_match"])],
            ],
        )
    )

    lines.extend(
        _table(
            ["Зона письма", "В разметке", "Предсказано", "Совпало", "Recall", "Precision"],
            [
                [
                    name,
                    str(payload["reference"]),
                    str(payload["predicted"]),
                    str(payload["matched"]),
                    _format_number(payload["recall"]),
                    _format_number(payload["precision"]),
                ]
                for name, payload in report["explanations"]["per_category"].items()
            ],
        )
    )

    lines.append("## 3. Устойчивость к маскировке AI-признаков")
    lines.append("")
    if masking:
        lines.append(
            "Каждое преобразование меняет письмо, не меняя его метку, и имитирует попытку скрыть следы генерации."
        )
        lines.append("")
        rows = [
            [
                name,
                payload["description"],
                _format_number(payload["roc_auc"]),
                _format_number(payload["roc_auc_delta"]),
                _format_number(payload["roc_auc_retention"]),
                _format_number(payload["recall"]),
            ]
            for name, payload in masking["masks"].items()
        ]
        rows.insert(
            0,
            [
                "baseline",
                "без маскировки",
                _format_number(masking["baseline_roc_auc"]),
                _format_number(0.0),
                _format_number(1.0),
                _format_number(masking["baseline_recall"]),
            ],
        )
        lines.extend(_table(["Маскировка", "Описание", "AUC-ROC", "Δ AUC", "Retention", "Recall"], rows))
        lines.append(
            "Средняя сохранность AUC-ROC: **%s**, худший сценарий: `%s`."
            % (_format_number(masking["mean_roc_auc_retention"]), masking["worst_mask"])
        )
        lines.append("")
    else:
        lines.append("Оценка не выполнялась.")
        lines.append("")

    lines.append("## 4. Работа с частично сгенерированными письмами")
    lines.append("")
    localization = partial["localization"]
    lines.extend(
        _table(
            ["Метрика", "Значение"],
            [
                ["Частично сгенерированных писем", str(partial["mixed_count"])],
                ["AUC-ROC, mixed против human", _format_number(partial["roc_auc_mixed_vs_human"])],
                ["AUC-ROC, полностью AI против mixed", _format_number(partial["roc_auc_full_ai_vs_mixed"])],
                ["Доля обнаруженных mixed-писем", _format_number(partial["mixed_detection_rate"])],
                ["Средний скор: human", _format_number(partial["mean_score_human"])],
                ["Средний скор: mixed", _format_number(partial["mean_score_mixed"])],
                ["Средний скор: ai", _format_number(partial["mean_score_full_ai"])],
                ["Корреляция скора и доли AI-текста (Spearman)", _format_number(partial["score_vs_ai_fraction_spearman"])],
            ],
        )
    )

    lines.append("**Локализация сгенерированных фрагментов**")
    lines.append("")
    lines.extend(
        _table(
            ["Метрика", "Значение"],
            [
                ["Писем в оценке", str(localization["evaluated_emails"])],
                ["Порог сегмента", _format_number(localization["segment_threshold"], 2)],
                ["Precision по символам (macro)", _format_number(localization["precision_macro"])],
                ["Recall по символам (macro)", _format_number(localization["recall_macro"])],
                ["F1 по символам (macro)", _format_number(localization["f1_macro"])],
                ["IoU по символам (macro)", _format_number(localization["iou_macro"])],
            ],
        )
    )

    sweep = report.get("partial_segment_sweep") or []
    if sweep:
        lines.append("**Зависимость локализации от порога сегмента**")
        lines.append("")
        lines.extend(
            _table(
                ["Порог", "Precision", "Recall", "F1", "IoU"],
                [
                    [
                        _format_number(item["segment_threshold"], 2),
                        _format_number(item["precision_macro"]),
                        _format_number(item["recall_macro"]),
                        _format_number(item["f1_macro"]),
                        _format_number(item["iou_macro"]),
                    ]
                    for item in sweep
                ],
            )
        )

    sweep = report.get("explanation_threshold_sweep") or []
    if sweep:
        lines.append("**Зависимость объяснений от порога зоны**")
        lines.append("")
        lines.extend(
            _table(
                ["Порог", "Recall", "Precision", "F1"],
                [
                    [
                        _format_number(item["threshold"], 2),
                        _format_number(item["recall_macro"]),
                        _format_number(item["precision_macro"]),
                        _format_number(item["f1_macro"]),
                    ]
                    for item in sweep
                ],
            )
        )

    return "\n".join(lines).rstrip() + "\n"


def save_markdown(report: Dict[str, object], path: Path, roc_image: Optional[str] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_markdown(report, roc_image))
    return path
