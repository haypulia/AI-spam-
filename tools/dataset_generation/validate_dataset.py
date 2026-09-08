# проверка качества сгенерированного датасета
# проверяет:
# 1. количество записей
# 2. распределение языков
# 3. распределение форматов
# 4. распределение доменов
# 5. распределение transformations
# 6. значения cosine similarity
# 7. значения normalized Levenshtein distance
# 8. дубликаты source
# 9. дубликаты target
# 10. одинаковые source/target
# 11. слишком короткие фрагменты
# 12. подозрительные HTML-записи
# 13. записи с некорректными similarity
# 14. общую статистику

import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DATASET_PATH = (
    PROJECT_ROOT / "dataset.jsonl"
)


MIN_TEXT_LENGTH = 20

COSINE_MAX = 0.98

LEVENSHTEIN_MIN = 0.20


def load_dataset() -> list[dict]:
    # загружает dataset.jsonl

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Файл dataset.jsonl не найден:\n"
            f"{DATASET_PATH}"
        )

    records = []

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

                print(
                    f"⚠ Некорректный JSON "
                    f"в строке {line_number}: "
                    f"{error}"
                )

                continue

            if not isinstance(record, dict):

                print(
                    f"⚠ Строка {line_number} "
                    f"не является объектом JSON."
                )

                continue

            records.append(record)

    return records


def remove_html_tags(
    text: str,
) -> str:
    # удаляет HTML-теги для анализа длины текста

    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def print_basic_statistics(
    records: list[dict],
) -> None:

    print()
    print("=" * 70)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 70)

    print()

    print(
        f"Всего записей: {len(records)}"
    )

    valid_count = sum(
        1
        for record in records
        if record.get("valid") is True
    )

    print(
        f"valid=True: {valid_count}"
    )

    print(
        f"valid=False: "
        f"{len(records) - valid_count}"
    )


def print_counter(
    title: str,
    counter: Counter,
) -> None:

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    total = sum(counter.values())

    for value, count in counter.most_common():

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{str(value):25} "
            f"{count:5} "
            f"({percentage:6.2f}%)"
        )


def print_distributions(
    records: list[dict],
) -> None:

    languages = Counter(
        record.get(
            "language",
            "UNKNOWN",
        )
        for record in records
    )

    formats = Counter(
        record.get(
            "format",
            "UNKNOWN",
        )
        for record in records
    )

    domains = Counter(
        record.get(
            "domain",
            "UNKNOWN",
        )
        for record in records
    )

    transformations = Counter(
        record.get(
            "transformation",
            "UNKNOWN",
        )
        for record in records
    )

    models = Counter(
        record.get(
            "model",
            "UNKNOWN",
        )
        for record in records
    )

    print_counter(
        "ЯЗЫКИ",
        languages,
    )

    print_counter(
        "ФОРМАТЫ",
        formats,
    )

    print_counter(
        "ДОМЕНЫ",
        domains,
    )

    print_counter(
        "TRANSFORMATIONS",
        transformations,
    )

    print_counter(
        "МОДЕЛИ",
        models,
    )


def print_similarity_statistics(
    records: list[dict],
) -> None:

    cosine_values = []
    levenshtein_values = []

    for record in records:

        cosine = record.get(
            "cosine_similarity"
        )

        levenshtein = record.get(
            "normalized_levenshtein_distance"
        )

        if isinstance(
            cosine,
            (int, float),
        ):

            cosine_values.append(
                float(cosine)
            )

        if isinstance(
            levenshtein,
            (int, float),
        ):

            levenshtein_values.append(
                float(levenshtein)
            )

    print()
    print("=" * 70)
    print("SIMILARITY")
    print("=" * 70)

    print()

    if cosine_values:

        print(
            "Cosine similarity:"
        )

        print(
            f"  minimum: "
            f"{min(cosine_values):.4f}"
        )

        print(
            f"  maximum: "
            f"{max(cosine_values):.4f}"
        )

        print(
            f"  average: "
            f"{sum(cosine_values) / len(cosine_values):.4f}"
        )

    else:

        print(
            "Cosine similarity: "
            "данные отсутствуют."
        )

    print()

    if levenshtein_values:

        print(
            "Normalized Levenshtein:"
        )

        print(
            f"  minimum: "
            f"{min(levenshtein_values):.4f}"
        )

        print(
            f"  maximum: "
            f"{max(levenshtein_values):.4f}"
        )

        print(
            f"  average: "
            f"{sum(levenshtein_values) / len(levenshtein_values):.4f}"
        )

    else:

        print(
            "Normalized Levenshtein: "
            "данные отсутствуют."
        )


def find_duplicates(
    records: list[dict],
) -> None:

    source_counter = Counter()
    target_counter = Counter()

    for record in records:

        source = record.get(
            "source",
            "",
        )

        target = record.get(
            "target",
            "",
        )

        if source:
            source_counter[source] += 1

        if target:
            target_counter[target] += 1

    duplicate_sources = [
        (
            text,
            count,
        )
        for text, count
        in source_counter.items()
        if count > 1
    ]

    duplicate_targets = [
        (
            text,
            count,
        )
        for text, count
        in target_counter.items()
        if count > 1
    ]

    print()
    print("=" * 70)
    print("ДУБЛИКАТЫ")
    print("=" * 70)

    print()

    print(
        f"Уникальных source: "
        f"{len(source_counter)}"
    )

    print(
        f"Уникальных target: "
        f"{len(target_counter)}"
    )

    print()

    print(
        f"Повторяющихся source: "
        f"{len(duplicate_sources)}"
    )

    print(
        f"Повторяющихся target: "
        f"{len(duplicate_targets)}"
    )

    if duplicate_sources:

        print()
        print(
            "Примеры повторяющихся source:"
        )

        for text, count in duplicate_sources[:10]:

            print()
            print(
                f"Количество: {count}"
            )

            print(
                text[:300]
            )

    if duplicate_targets:

        print()
        print(
            "Примеры повторяющихся target:"
        )

        for text, count in duplicate_targets[:10]:

            print()
            print(
                f"Количество: {count}"
            )

            print(
                text[:300]
            )


def check_invalid_records(
    records: list[dict],
) -> None:

    problems = []

    for index, record in enumerate(
        records,
        start=1,
    ):

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

        record_id = record.get(
            "id",
            f"line_{index}",
        )

        if not isinstance(
            source,
            str,
        ) or not source.strip():

            problems.append(
                (
                    record_id,
                    "empty source",
                )
            )

            continue

        if not isinstance(
            target,
            str,
        ) or not target.strip():

            problems.append(
                (
                    record_id,
                    "empty target",
                )
            )

            continue

        if source.strip() == target.strip():

            problems.append(
                (
                    record_id,
                    "source == target",
                )
            )

        source_text = remove_html_tags(
            source
        )

        if len(source_text) < MIN_TEXT_LENGTH:

            problems.append(
                (
                    record_id,
                    "source слишком короткий",
                )
            )

        target_text = remove_html_tags(
            target
        )

        if len(target_text) < MIN_TEXT_LENGTH:

            problems.append(
                (
                    record_id,
                    "target слишком короткий",
                )
            )

        if not isinstance(
            cosine,
            (int, float),
        ):

            problems.append(
                (
                    record_id,
                    "cosine отсутствует",
                )
            )

        else:

            if cosine > COSINE_MAX:

                problems.append(
                    (
                        record_id,
                        "cosine выше установленного порога",
                    )
                )

            if cosine < -1 or cosine > 1:

                problems.append(
                    (
                        record_id,
                        "cosine вне диапазона [-1, 1]",
                    )
                )

        if not isinstance(
            levenshtein,
            (int, float),
        ):

            problems.append(
                (
                    record_id,
                    "Levenshtein отсутствует",
                )
            )

        else:

            if levenshtein < LEVENSHTEIN_MIN:

                problems.append(
                    (
                        record_id,
                        "Levenshtein ниже установленного порога",
                    )
                )

            if levenshtein < 0 or levenshtein > 1:

                problems.append(
                    (
                        record_id,
                        "Levenshtein вне диапазона [0, 1]",
                    )
                )

    print()
    print("=" * 70)
    print("ПРОБЛЕМНЫЕ ЗАПИСИ")
    print("=" * 70)

    print()

    if not problems:

        print(
            "✓ Проблемных записей не найдено."
        )

        return

    print(
        f"Найдено проблем: "
        f"{len(problems)}"
    )

    print()

    for record_id, reason in problems[:50]:

        print(
            f"⚠ {record_id}: {reason}"
        )

    if len(problems) > 50:

        print()

        print(
            f"... и ещё "
            f"{len(problems) - 50}"
        )


def check_html_records(
    records: list[dict],
) -> None:

    html_records = [
        record
        for record in records
        if record.get("format") == "html"
    ]

    if not html_records:

        return

    missing_html = []
    markdown_records = []

    for record in html_records:

        source = record.get(
            "source",
            "",
        )

        target = record.get(
            "target",
            "",
        )

        if "<" not in target or ">" not in target:

            missing_html.append(
                record
            )

        if "```" in target:

            markdown_records.append(
                record
            )

    print()
    print("=" * 70)
    print("ПРОВЕРКА HTML")
    print("=" * 70)

    print()

    print(
        f"HTML-записей: "
        f"{len(html_records)}"
    )

    print(
        f"HTML target без тегов: "
        f"{len(missing_html)}"
    )

    print(
        f"HTML target с Markdown fences: "
        f"{len(markdown_records)}"
    )

    if missing_html:

        print()
        print(
            "Примеры HTML-записей "
            "без HTML в target:"
        )

        for record in missing_html[:5]:

            print(
                f"- {record.get('id')}"
            )

            print(
                record.get(
                    "target",
                    "",
                )[:300]
            )

    if markdown_records:

        print()
        print(
            "Примеры target с ```:"
        )

        for record in markdown_records[:5]:

            print(
                f"- {record.get('id')}"
            )


def print_transformation_similarity(
    records: list[dict],
) -> None:

    print()
    print("=" * 70)
    print("SIMILARITY ПО TRANSFORMATIONS")
    print("=" * 70)

    grouped = {}

    for record in records:

        transformation = record.get(
            "transformation",
            "UNKNOWN",
        )

        cosine = record.get(
            "cosine_similarity"
        )

        levenshtein = record.get(
            "normalized_levenshtein_distance"
        )

        if not isinstance(
            cosine,
            (int, float),
        ):
            continue

        if not isinstance(
            levenshtein,
            (int, float),
        ):
            continue

        grouped.setdefault(
            transformation,
            {
                "cosine": [],
                "levenshtein": [],
            },
        )

        grouped[
            transformation
        ]["cosine"].append(
            float(cosine)
        )

        grouped[
            transformation
        ]["levenshtein"].append(
            float(levenshtein)
        )

    print()

    for transformation in sorted(
        grouped.keys()
    ):

        cosine_values = grouped[
            transformation
        ]["cosine"]

        levenshtein_values = grouped[
            transformation
        ]["levenshtein"]

        cosine_average = (
            sum(cosine_values)
            / len(cosine_values)
        )

        levenshtein_average = (
            sum(levenshtein_values)
            / len(levenshtein_values)
        )

        print(
            f"{transformation:25} "
            f"count={len(cosine_values):3} "
            f"cos={cosine_average:.4f} "
            f"lev={levenshtein_average:.4f}"
        )


def show_examples(
    records: list[dict],
) -> None:

    print()
    print("=" * 70)
    print("ПРИМЕРЫ ЗАПИСЕЙ")
    print("=" * 70)

    print()

    for record in records[:3]:

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
                "",
            )
        )

        print()

        print(
            f"Cosine: "
            f"{record.get('cosine_similarity')}"
        )

        print(
            f"Levenshtein: "
            f"{record.get('normalized_levenshtein_distance')}"
        )

        print()

        print("-" * 70)


def validate_dataset():
    # основная функция проверки

    print()
    print("=" * 70)
    print("ПРОВЕРКА DATASET")
    print("=" * 70)

    print()

    print(
        f"Файл: {DATASET_PATH}"
    )

    records = load_dataset()

    if not records:

        print()
        print(
            "✗ Dataset пуст."
        )

        return

    print_basic_statistics(
        records
    )

    print_distributions(
        records
    )

    print_similarity_statistics(
        records
    )

    print_transformation_similarity(
        records
    )

    find_duplicates(
        records
    )

    check_invalid_records(
        records
    )

    check_html_records(
        records
    )

    show_examples(
        records
    )

    print()
    print("=" * 70)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 70)
    print()


if __name__ == "__main__":

    try:

        validate_dataset()

    except KeyboardInterrupt:

        print()
        print(
            "Проверка остановлена пользователем."
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("КРИТИЧЕСКАЯ ОШИБКА")
        print("=" * 70)
        print()

        print(error)

        print()