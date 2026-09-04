# расширенная проверка качества dataset_v2.jsonl
# проверяет:
# 1. структуру JSONL
# 2. обязательные поля
# 3. пустые source/target
# 4. слишком короткие target
# 5. дубликаты
# 6. совпадение source и target
# 7. диапазоны similarity
# 8. соответствие valid и порогов
# 9. HTML-записи
# 10. наличие Markdown fences
# 11. распределение языков
# 12. распределение форматов
# 13. распределение доменов
# 14. распределение transformations
# 15. распределение моделей
# 16. подозрительно похожие пары
# 17. подозрительно изменённые пары
# 18. итоговую статистику качества

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "generated_dataset"
    / "dataset_v2.jsonl"
)


COSINE_THRESHOLD = 0.98
LEVENSHTEIN_THRESHOLD = 0.20

MIN_TARGET_LENGTH = 20

MIN_COSINE = -1.0
MAX_COSINE = 1.0

MIN_LEVENSHTEIN = 0.0
MAX_LEVENSHTEIN = 1.0


REQUIRED_FIELDS = {
    "id",
    "language",
    "domain",
    "format",
    "model",
    "transformation",
    "source",
    "target",
    "cosine_similarity",
    "normalized_levenshtein_distance",
    "valid",
}


def normalize_text(text: str) -> str:
    # нормализует текст для проверки совпадений

    if not isinstance(text, str):
        text = str(text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def strip_html(text: str) -> str:
    # удаляет HTML-теги для проверки содержимого

    if not isinstance(text, str):
        return ""

    return re.sub(
        r"<[^>]+>",
        "",
        text,
    ).strip()


def contains_html(text: str) -> bool:
    # проверяет, содержит ли строка HTML-теги

    if not isinstance(text, str):
        return False

    return bool(
        re.search(
            r"<[a-zA-Z][^>]*>",
            text,
        )
    )


def contains_markdown_fence(text: str) -> bool:
    # проверяет наличие ```...``` в target

    if not isinstance(text, str):
        return False

    return "```" in text


def is_number(value: Any) -> bool:
    # проверяет, является ли значение числом

    return isinstance(
        value,
        (int, float),
    ) and not isinstance(
        value,
        bool,
    )


def load_dataset() -> tuple[list[dict], list[str]]:
    # загружает dataset_v2.jsonl

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset не найден:\n{DATASET_PATH}"
        )

    records: list[dict] = []
    parsing_errors: list[str] = []

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:

                parsing_errors.append(
                    f"Строка {line_number}: "
                    f"{error}"
                )

                continue

            if not isinstance(record, dict):

                parsing_errors.append(
                    f"Строка {line_number}: "
                    f"запись не является JSON object"
                )

                continue

            records.append(record)

    return records, parsing_errors


def check_structure(
    records: list[dict],
) -> list[dict]:
    # проверяет наличие обязательных полей

    problems = []

    for record in records:

        record_id = record.get(
            "id",
            "<без id>",
        )

        missing = (
            REQUIRED_FIELDS
            - set(record.keys())
        )

        if missing:

            problems.append(
                {
                    "id": record_id,
                    "type": "missing_fields",
                    "details": (
                        ", ".join(
                            sorted(missing)
                        )
                    ),
                }
            )

    return problems


def check_values(
    records: list[dict],
) -> list[dict]:
    # проверяет значения полей

    problems = []

    for record in records:

        record_id = record.get(
            "id",
            "<без id>",
        )

        source = record.get(
            "source",
            "",
        )

        target = record.get(
            "target",
            "",
        )

        cosine = record.get(
            "cosine_similarity"
        )

        levenshtein = record.get(
            "normalized_levenshtein_distance"
        )

        valid = record.get(
            "valid"
        )

        if not isinstance(
            source,
            str,
        ) or not source.strip():

            problems.append(
                {
                    "id": record_id,
                    "type": "empty_source",
                    "details": (
                        "source пустой"
                    ),
                }
            )

        if not isinstance(
            target,
            str,
        ) or not target.strip():

            problems.append(
                {
                    "id": record_id,
                    "type": "empty_target",
                    "details": (
                        "target пустой"
                    ),
                }
            )

        elif len(
            normalize_text(target)
        ) < MIN_TARGET_LENGTH:

            problems.append(
                {
                    "id": record_id,
                    "type": "short_target",
                    "details": (
                        f"длина target меньше "
                        f"{MIN_TARGET_LENGTH}"
                    ),
                }
            )

        if (
            isinstance(source, str)
            and isinstance(target, str)
            and normalize_text(source)
            == normalize_text(target)
        ):

            problems.append(
                {
                    "id": record_id,
                    "type": "identical_pair",
                    "details": (
                        "source и target одинаковые"
                    ),
                }
            )

        if not is_number(cosine):

            problems.append(
                {
                    "id": record_id,
                    "type": "invalid_cosine",
                    "details": (
                        "cosine_similarity "
                        "не является числом"
                    ),
                }
            )

        else:

            if not (
                MIN_COSINE
                <= float(cosine)
                <= MAX_COSINE
            ):

                problems.append(
                    {
                        "id": record_id,
                        "type": "cosine_out_of_range",
                        "details": (
                            f"cosine={cosine}"
                        ),
                    }
                )

        if not is_number(
            levenshtein
        ):

            problems.append(
                {
                    "id": record_id,
                    "type": "invalid_levenshtein",
                    "details": (
                        "normalized_levenshtein_distance "
                        "не является числом"
                    ),
                }
            )

        else:

            if not (
                MIN_LEVENSHTEIN
                <= float(levenshtein)
                <= MAX_LEVENSHTEIN
            ):

                problems.append(
                    {
                        "id": record_id,
                        "type": "levenshtein_out_of_range",
                        "details": (
                            f"levenshtein={levenshtein}"
                        ),
                    }
                )

        if not isinstance(
            valid,
            bool,
        ):

            problems.append(
                {
                    "id": record_id,
                    "type": "invalid_valid_field",
                    "details": (
                        "valid должен быть bool"
                    ),
                }
            )

    return problems


def check_threshold_consistency(
    records: list[dict],
) -> list[dict]:
    # проверяет соответствие valid установленным порогам

    problems = []

    for record in records:

        record_id = record.get(
            "id",
            "<без id>",
        )

        cosine = record.get(
            "cosine_similarity"
        )

        levenshtein = record.get(
            "normalized_levenshtein_distance"
        )

        valid = record.get(
            "valid"
        )

        if (
            not is_number(cosine)
            or not is_number(levenshtein)
            or not isinstance(valid, bool)
        ):
            continue

        cosine_ok = (
            float(cosine)
            <= COSINE_THRESHOLD
        )

        levenshtein_ok = (
            float(levenshtein)
            >= LEVENSHTEIN_THRESHOLD
        )

        expected_valid = (
            cosine_ok
            and levenshtein_ok
        )

        if valid != expected_valid:

            problems.append(
                {
                    "id": record_id,
                    "type": "threshold_mismatch",
                    "details": (
                        f"valid={valid}, "
                        f"ожидалось={expected_valid}; "
                        f"cosine={float(cosine):.4f}, "
                        f"lev={float(levenshtein):.4f}"
                    ),
                }
            )

    return problems


def find_duplicates(
    records: list[dict],
) -> dict:
    # находит повторяющиеся source и target

    source_counter = Counter()
    target_counter = Counter()
    id_counter = Counter()

    for record in records:

        source = normalize_text(
            record.get(
                "source",
                "",
            )
        )

        target = normalize_text(
            record.get(
                "target",
                "",
            )
        )

        record_id = record.get(
            "id",
            "",
        )

        if source:
            source_counter[source] += 1

        if target:
            target_counter[target] += 1

        if record_id:
            id_counter[record_id] += 1

    duplicate_sources = {
        key: value
        for key, value
        in source_counter.items()
        if value > 1
    }

    duplicate_targets = {
        key: value
        for key, value
        in target_counter.items()
        if value > 1
    }

    duplicate_ids = {
        key: value
        for key, value
        in id_counter.items()
        if value > 1
    }

    return {
        "sources": duplicate_sources,
        "targets": duplicate_targets,
        "ids": duplicate_ids,
    }


def check_html(
    records: list[dict],
) -> dict:
    # проверяет HTML-записи

    html_records = 0
    html_without_tags = []
    html_with_fences = []

    for record in records:

        if record.get(
            "format"
        ) != "html":

            continue

        html_records += 1

        record_id = record.get(
            "id",
            "<без id>",
        )

        target = record.get(
            "target",
            "",
        )

        if not contains_html(target):

            html_without_tags.append(
                record_id
            )

        if contains_markdown_fence(
            target
        ):

            html_with_fences.append(
                record_id
            )

    return {
        "count": html_records,
        "without_tags": html_without_tags,
        "with_fences": html_with_fences,
    }


def find_short_targets(
    records: list[dict],
) -> list[dict]:
    # возвращает короткие target

    result = []

    for record in records:

        target = record.get(
            "target",
            "",
        )

        if not isinstance(
            target,
            str,
        ):
            continue

        length = len(
            normalize_text(target)
        )

        if length < MIN_TARGET_LENGTH:

            result.append(
                {
                    "id": record.get(
                        "id",
                        "<без id>",
                    ),
                    "length": length,
                    "target": target,
                }
            )

    return result


def find_suspicious_similarity(
    records: list[dict],
) -> list[dict]:
    # ищет записи, близкие к установленным границам

    suspicious = []

    for record in records:

        cosine = record.get(
            "cosine_similarity"
        )

        levenshtein = record.get(
            "normalized_levenshtein_distance"
        )

        if (
            not is_number(cosine)
            or not is_number(levenshtein)
        ):
            continue

        cosine = float(cosine)
        levenshtein = float(
            levenshtein
        )

        cosine_near_limit = (
            0.965
            <= cosine
            <= COSINE_THRESHOLD
        )

        levenshtein_near_limit = (
            LEVENSHTEIN_THRESHOLD
            <= levenshtein
            <= 0.23
        )

        if (
            cosine_near_limit
            or levenshtein_near_limit
        ):

            suspicious.append(
                {
                    "id": record.get(
                        "id",
                        "<без id>",
                    ),
                    "transformation": record.get(
                        "transformation",
                        "",
                    ),
                    "cosine": cosine,
                    "levenshtein": levenshtein,
                }
            )

    return suspicious


def calculate_distributions(
    records: list[dict],
) -> dict:

    languages = Counter()
    formats = Counter()
    domains = Counter()
    transformations = Counter()
    models = Counter()

    for record in records:

        languages[
            record.get(
                "language",
                "unknown",
            )
        ] += 1

        formats[
            record.get(
                "format",
                "unknown",
            )
        ] += 1

        domains[
            record.get(
                "domain",
                "unknown",
            )
        ] += 1

        transformations[
            record.get(
                "transformation",
                "unknown",
            )
        ] += 1

        models[
            record.get(
                "model",
                "unknown",
            )
        ] += 1

    return {
        "languages": languages,
        "formats": formats,
        "domains": domains,
        "transformations": transformations,
        "models": models,
    }


def calculate_similarity_statistics(
    records: list[dict],
) -> dict:

    cosine_values = []
    levenshtein_values = []

    for record in records:

        cosine = record.get(
            "cosine_similarity"
        )

        levenshtein = record.get(
            "normalized_levenshtein_distance"
        )

        if is_number(cosine):

            cosine_values.append(
                float(cosine)
            )

        if is_number(levenshtein):

            levenshtein_values.append(
                float(levenshtein)
            )

    result = {}

    if cosine_values:

        result["cosine"] = {
            "minimum": min(
                cosine_values
            ),
            "maximum": max(
                cosine_values
            ),
            "average": (
                sum(cosine_values)
                / len(cosine_values)
            ),
        }

    else:

        result["cosine"] = None

    if levenshtein_values:

        result["levenshtein"] = {
            "minimum": min(
                levenshtein_values
            ),
            "maximum": max(
                levenshtein_values
            ),
            "average": (
                sum(
                    levenshtein_values
                )
                / len(
                    levenshtein_values
                )
            ),
        }

    else:

        result["levenshtein"] = None

    return result


def calculate_similarity_by_transformation(
    records: list[dict],
) -> dict:

    grouped: dict[
        str,
        list[dict],
    ] = {}

    for record in records:

        transformation = record.get(
            "transformation",
            "unknown",
        )

        grouped.setdefault(
            transformation,
            [],
        ).append(record)

    result = {}

    for transformation, items in sorted(
        grouped.items()
    ):

        cosine_values = [
            float(
                item[
                    "cosine_similarity"
                ]
            )
            for item in items
            if is_number(
                item.get(
                    "cosine_similarity"
                )
            )
        ]

        levenshtein_values = [
            float(
                item[
                    "normalized_levenshtein_distance"
                ]
            )
            for item in items
            if is_number(
                item.get(
                    "normalized_levenshtein_distance"
                )
            )
        ]

        result[transformation] = {
            "count": len(items),
            "cosine_average": (
                sum(cosine_values)
                / len(cosine_values)
                if cosine_values
                else 0.0
            ),
            "levenshtein_average": (
                sum(levenshtein_values)
                / len(levenshtein_values)
                if levenshtein_values
                else 0.0
            ),
        }

    return result


def print_counter(
    counter: Counter,
) -> None:

    for key, value in counter.most_common():

        print(
            f"{str(key):30s} "
            f"{value:5d}"
        )


def print_problems(
    problems: list[dict],
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    if not problems:

        print()
        print(
            "✓ Проблем не найдено."
        )

        return

    print()
    print(
        f"Найдено проблем: "
        f"{len(problems)}"
    )

    print()

    for problem in problems[:50]:

        print(
            f"⚠ {problem['id']}: "
            f"{problem['type']} — "
            f"{problem['details']}"
        )

    if len(problems) > 50:

        print()

        print(
            f"... и ещё "
            f"{len(problems) - 50}"
        )


def main() -> None:

    print()
    print("=" * 80)
    print("РАСШИРЕННАЯ ПРОВЕРКА DATASET V2")
    print("=" * 80)
    print()

    print(
        f"Файл: {DATASET_PATH}"
    )

    print()

    records, parsing_errors = (
        load_dataset()
    )

    print("=" * 80)
    print("ЗАГРУЗКА")
    print("=" * 80)
    print()

    print(
        f"Корректных записей: "
        f"{len(records)}"
    )

    print(
        f"Ошибок чтения JSONL: "
        f"{len(parsing_errors)}"
    )

    if parsing_errors:

        print()

        for error in parsing_errors[:20]:

            print(
                f"⚠ {error}"
            )

    print()
    print("=" * 80)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 80)
    print()

    valid_true = sum(
        1
        for record in records
        if record.get("valid") is True
    )

    valid_false = sum(
        1
        for record in records
        if record.get("valid") is False
    )

    print(
        f"Всего записей: "
        f"{len(records)}"
    )

    print(
        f"valid=True: "
        f"{valid_true}"
    )

    print(
        f"valid=False: "
        f"{valid_false}"
    )

    structure_problems = (
        check_structure(records)
    )

    print_problems(
        structure_problems,
        "ПРОВЕРКА СТРУКТУРЫ",
    )

    value_problems = (
        check_values(records)
    )

    print_problems(
        value_problems,
        "ПРОВЕРКА ЗНАЧЕНИЙ",
    )

    threshold_problems = (
        check_threshold_consistency(
            records
        )
    )

    print_problems(
        threshold_problems,
        "ПРОВЕРКА SIMILARITY И ПОРОГОВ",
    )

    distributions = (
        calculate_distributions(
            records
        )
    )

    print()
    print("=" * 80)
    print("РАСПРЕДЕЛЕНИЕ ЯЗЫКОВ")
    print("=" * 80)
    print()

    print_counter(
        distributions["languages"]
    )

    print()
    print("=" * 80)
    print("РАСПРЕДЕЛЕНИЕ ФОРМАТОВ")
    print("=" * 80)
    print()

    print_counter(
        distributions["formats"]
    )

    print()
    print("=" * 80)
    print("РАСПРЕДЕЛЕНИЕ ДОМЕНОВ")
    print("=" * 80)
    print()

    print_counter(
        distributions["domains"]
    )

    print()
    print("=" * 80)
    print("РАСПРЕДЕЛЕНИЕ TRANSFORMATIONS")
    print("=" * 80)
    print()

    print_counter(
        distributions["transformations"]
    )

    print()
    print("=" * 80)
    print("РАСПРЕДЕЛЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)
    print()

    print_counter(
        distributions["models"]
    )

    similarity_stats = (
        calculate_similarity_statistics(
            records
        )
    )

    print()
    print("=" * 80)
    print("SIMILARITY")
    print("=" * 80)
    print()

    cosine_stats = similarity_stats.get(
        "cosine"
    )

    if cosine_stats:

        print("Cosine similarity:")

        print(
            f"  minimum: "
            f"{cosine_stats['minimum']:.4f}"
        )

        print(
            f"  maximum: "
            f"{cosine_stats['maximum']:.4f}"
        )

        print(
            f"  average: "
            f"{cosine_stats['average']:.4f}"
        )

    levenshtein_stats = (
        similarity_stats.get(
            "levenshtein"
        )
    )

    if levenshtein_stats:

        print()

        print(
            "Normalized Levenshtein:"
        )

        print(
            f"  minimum: "
            f"{levenshtein_stats['minimum']:.4f}"
        )

        print(
            f"  maximum: "
            f"{levenshtein_stats['maximum']:.4f}"
        )

        print(
            f"  average: "
            f"{levenshtein_stats['average']:.4f}"
        )

    transformation_stats = (
        calculate_similarity_by_transformation(
            records
        )
    )

    print()
    print("=" * 80)
    print("SIMILARITY ПО TRANSFORMATIONS")
    print("=" * 80)
    print()

    for transformation, stats in (
        transformation_stats.items()
    ):

        print(
            f"{transformation:25s} "
            f"count={stats['count']:3d} "
            f"cos={stats['cosine_average']:.4f} "
            f"lev={stats['levenshtein_average']:.4f}"
        )

    duplicates = find_duplicates(
        records
    )

    print()
    print("=" * 80)
    print("ДУБЛИКАТЫ")
    print("=" * 80)
    print()

    unique_sources = len(
        {
            normalize_text(
                record.get(
                    "source",
                    "",
                )
            )
            for record in records
        }
    )

    unique_targets = len(
        {
            normalize_text(
                record.get(
                    "target",
                    "",
                )
            )
            for record in records
        }
    )

    print(
        f"Уникальных source: "
        f"{unique_sources}"
    )

    print(
        f"Уникальных target: "
        f"{unique_targets}"
    )

    print(
        f"Повторяющихся source: "
        f"{len(duplicates['sources'])}"
    )

    print(
        f"Повторяющихся target: "
        f"{len(duplicates['targets'])}"
    )

    print(
        f"Повторяющихся ID: "
        f"{len(duplicates['ids'])}"
    )

    if duplicates["sources"]:

        print()
        print(
            "Примеры повторяющихся source:"
        )

        for source, count in list(
            duplicates["sources"].items()
        )[:10]:

            print()
            print(
                f"Количество: {count}"
            )
            print(source[:500])

    if duplicates["targets"]:

        print()
        print(
            "Примеры повторяющихся target:"
        )

        for target, count in list(
            duplicates["targets"].items()
        )[:10]:

            print()
            print(
                f"Количество: {count}"
            )
            print(target[:500])

    html_stats = check_html(
        records
    )

    print()
    print("=" * 80)
    print("ПРОВЕРКА HTML")
    print("=" * 80)
    print()

    print(
        f"HTML-записей: "
        f"{html_stats['count']}"
    )

    print(
        f"HTML target без тегов: "
        f"{len(html_stats['without_tags'])}"
    )

    print(
        f"HTML target с Markdown fences: "
        f"{len(html_stats['with_fences'])}"
    )

    if html_stats["without_tags"]:

        print()
        print(
            "Примеры HTML-записей "
            "без HTML в target:"
        )

        for record_id in (
            html_stats["without_tags"][:10]
        ):

            print(
                f"- {record_id}"
            )

    if html_stats["with_fences"]:

        print()
        print(
            "HTML-записи с Markdown fences:"
        )

        for record_id in (
            html_stats["with_fences"][:10]
        ):

            print(
                f"- {record_id}"
            )

    short_targets = (
        find_short_targets(
            records
        )
    )

    print()
    print("=" * 80)
    print("КОРОТКИЕ TARGET")
    print("=" * 80)
    print()

    if not short_targets:

        print(
            "✓ Коротких target не найдено."
        )

    else:

        print(
            f"Найдено коротких target: "
            f"{len(short_targets)}"
        )

        for item in short_targets[:20]:

            print()
            print(
                f"ID: {item['id']}"
            )

            print(
                f"Длина: {item['length']}"
            )

            print(
                item["target"][:500]
            )

    suspicious = (
        find_suspicious_similarity(
            records
        )
    )

    print()
    print("=" * 80)
    print("ПОДОЗРИТЕЛЬНО БЛИЗКИЕ К ПОРОГАМ")
    print("=" * 80)
    print()

    print(
        f"Найдено: "
        f"{len(suspicious)}"
    )

    for item in suspicious[:30]:

        print(
            f"{item['id']:25s} "
            f"{item['transformation']:22s} "
            f"cos={item['cosine']:.4f} "
            f"lev={item['levenshtein']:.4f}"
        )

    print()
    print("=" * 80)
    print("ПРИМЕРЫ ЗАПИСЕЙ")
    print("=" * 80)

    for record in records[:3]:

        print()
        print(
            f"ID: {record.get('id')}"
        )

        print(
            f"Language: "
            f"{record.get('language')}"
        )

        print(
            f"Domain: "
            f"{record.get('domain')}"
        )

        print(
            f"Format: "
            f"{record.get('format')}"
        )

        print(
            f"Transformation: "
            f"{record.get('transformation')}"
        )

        print()
        print("SOURCE:")
        print(
            record.get(
                "source",
                "",
            )
        )

        print()
        print("TARGET:")
        print(
            record.get(
                "target",
                ""
            )
        )

        print()

        cosine = record.get(
            "cosine_similarity"
        )

        levenshtein = record.get(
            "normalized_levenshtein_distance"
        )

        if is_number(cosine):

            print(
                f"Cosine: "
                f"{float(cosine):.6f}"
            )

        if is_number(levenshtein):

            print(
                f"Levenshtein: "
                f"{float(levenshtein):.6f}"
            )

        print()
        print("-" * 80)

    total_problems = (
        len(parsing_errors)
        + len(structure_problems)
        + len(value_problems)
        + len(threshold_problems)
        + len(duplicates["ids"])
        + len(html_stats["without_tags"])
        + len(html_stats["with_fences"])
        + len(short_targets)
    )

    print()
    print("=" * 80)
    print("ИТОГОВАЯ ОЦЕНКА")
    print("=" * 80)
    print()

    print(
        f"Всего записей: "
        f"{len(records)}"
    )

    print(
        f"Критических проблем: "
        f"{total_problems}"
    )

    print()

    if total_problems == 0:

        print(
            "✓ DATASET V2 ПРОШЁЛ "
            "РАСШИРЕННУЮ ПРОВЕРКУ."
        )

    else:

        print(
            "⚠ В DATASET V2 НАЙДЕНЫ "
            "ПРОБЛЕМЫ."
        )

        print(
            "Не все найденные предупреждения "
            "обязательно означают ошибку."
        )

    print()
    print(
        f"Пороги:"
    )

    print(
        f"  Cosine <= "
        f"{COSINE_THRESHOLD}"
    )

    print(
        f"  Levenshtein >= "
        f"{LEVENSHTEIN_THRESHOLD}"
    )

    print()
    print("=" * 80)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 80)
    print()


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Проверка остановлена пользователем."
        )

    except Exception as error:

        print()
        print("=" * 80)
        print("КРИТИЧЕСКАЯ ОШИБКА")
        print("=" * 80)
        print()
        print(
            f"{type(error).__name__}: "
            f"{error}"
        )
        print()