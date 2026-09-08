# полная проверка dataset_v2.jsonl

# проверяет статистику, языки, форматы, домены,
# transformations, модели, similarity, дубликаты,
# длину target, HTML и базовые требования к записям

import json
import re
from collections import Counter
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

DATASET_PATH = (
    PROJECT_ROOT
    / "generated_dataset"
    / "dataset_v2.jsonl"
)


MIN_TARGET_LENGTH = 20

COSINE_MIN = 0.0
COSINE_MAX = 1.0

LEVENSHTEIN_MIN = 0.0
LEVENSHTEIN_MAX = 1.0

EXPECTED_LANGUAGES = {
    "ru",
    "en",
}

EXPECTED_FORMATS = {
    "html",
    "plain_text",
}

EXPECTED_TRANSFORMATIONS = {
    "paraphrase",
    "rewrite",
    "rephrase_structure",
    "shorten",
    "expand",
    "simplify",
    "formalize",
    "casualize",
    "friendly",
    "professional",
    "neutral",
    "neutralize",
    "tone_change",
    "grammar_improvement",
}


def load_dataset() -> list[dict]:
    # загружает dataset_v2.jsonl

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Файл dataset_v2.jsonl не найден:\n"
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
                    f"Некорректный JSON "
                    f"в строке {line_number}: "
                    f"{error}"
                )

                continue

            if not isinstance(record, dict):

                print(
                    f"Строка {line_number} "
                    f"не является объектом."
                )

                continue

            records.append(record)

    return records


def remove_html_tags(text: str) -> str:
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


def contains_html(text: str) -> bool:
    # проверяет, содержит ли текст HTML-теги

    if not isinstance(text, str):
        return False

    return bool(
        re.search(
            r"<[a-zA-Z][^>]*>",
            text,
        )
    )


def contains_markdown_fence(text: str) -> bool:
    # проверяет наличие markdown fences

    if not isinstance(text, str):
        return False

    return "```" in text


def validate_record(
    record: dict,
) -> list[str]:
    # проверяет одну запись

    problems = []

    record_id = record.get(
        "id",
        "<unknown>",
    )

    required_fields = [
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
    ]

    for field in required_fields:

        if field not in record:

            problems.append(
                f"{record_id}: "
                f"отсутствует поле '{field}'"
            )

    source = record.get(
        "source",
        "",
    )

    if not isinstance(source, str):

        problems.append(
            f"{record_id}: "
            "source не является строкой"
        )

    elif not source.strip():

        problems.append(
            f"{record_id}: "
            "source пустой"
        )

    target = record.get(
        "target",
        "",
    )

    if not isinstance(target, str):

        problems.append(
            f"{record_id}: "
            "target не является строкой"
        )

    elif not target.strip():

        problems.append(
            f"{record_id}: "
            "target пустой"
        )

    else:

        target_text = remove_html_tags(
            target
        )

        if len(target_text) < MIN_TARGET_LENGTH:

            problems.append(
                f"{record_id}: "
                "target слишком короткий"
            )

    language = record.get(
        "language"
    )

    if language not in EXPECTED_LANGUAGES:

        problems.append(
            f"{record_id}: "
            f"неизвестный язык: {language}"
        )

    email_format = record.get(
        "format"
    )

    if email_format not in EXPECTED_FORMATS:

        problems.append(
            f"{record_id}: "
            f"неизвестный формат: "
            f"{email_format}"
        )

    transformation = record.get(
        "transformation"
    )

    if transformation not in EXPECTED_TRANSFORMATIONS:

        problems.append(
            f"{record_id}: "
            f"неизвестный transformation: "
            f"{transformation}"
        )

    cosine = record.get(
        "cosine_similarity"
    )

    if not isinstance(
        cosine,
        (int, float),
    ):

        problems.append(
            f"{record_id}: "
            "cosine_similarity не число"
        )

    else:

        if not (
            COSINE_MIN
            <= cosine
            <= COSINE_MAX
        ):

            problems.append(
                f"{record_id}: "
                f"cosine_similarity вне диапазона: "
                f"{cosine}"
            )

    levenshtein = record.get(
        "normalized_levenshtein_distance"
    )

    if not isinstance(
        levenshtein,
        (int, float),
    ):

        problems.append(
            f"{record_id}: "
            "normalized_levenshtein_distance "
            "не число"
        )

    else:

        if not (
            LEVENSHTEIN_MIN
            <= levenshtein
            <= LEVENSHTEIN_MAX
        ):

            problems.append(
                f"{record_id}: "
                "Levenshtein вне диапазона: "
                f"{levenshtein}"
            )

    valid = record.get(
        "valid"
    )

    if valid is not True:

        problems.append(
            f"{record_id}: "
            "valid != True"
        )

    return problems


def find_duplicates(
    records: list[dict],
    field: str,
) -> Counter:
    # находит повторяющиеся значения указанного поля

    counter = Counter()

    for record in records:

        value = record.get(
            field
        )

        if isinstance(
            value,
            str,
        ):

            counter[value] += 1

    return Counter(
        {
            value: count
            for value, count
            in counter.items()
            if count > 1
        }
    )


def print_counter(
    counter: Counter,
    total: int,
) -> None:
    # выводит Counter с процентами

    for key, count in counter.most_common():

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{str(key):28s}"
            f"{count:5d}"
            f" ({percentage:6.2f}%)"
        )


def similarity_statistics(
    records: list[dict],
):
    # возвращает значения similarity

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

    return (
        cosine_values,
        levenshtein_values,
    )


def print_transformation_similarity(
    records: list[dict],
):
    # показывает similarity для каждого transformation

    grouped = {}

    for record in records:

        transformation = record.get(
            "transformation"
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

        if transformation not in grouped:

            grouped[
                transformation
            ] = {
                "cosine": [],
                "levenshtein": [],
            }

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

    for transformation in sorted(
        grouped.keys()
    ):

        cosine_values = grouped[
            transformation
        ]["cosine"]

        levenshtein_values = grouped[
            transformation
        ]["levenshtein"]

        count = len(
            cosine_values
        )

        cosine_average = (
            sum(cosine_values)
            / count
        )

        levenshtein_average = (
            sum(levenshtein_values)
            / count
        )

        print(
            f"{transformation:28s}"
            f"count={count:3d} "
            f"cos={cosine_average:.4f} "
            f"lev={levenshtein_average:.4f}"
        )


def check_html_records(
    records: list[dict],
):
    # проверяет HTML-записи

    html_records = [
        record
        for record in records
        if record.get("format") == "html"
    ]

    missing_html = []
    markdown_fences = []

    for record in html_records:

        target = record.get(
            "target",
            "",
        )

        if not contains_html(
            target
        ):

            missing_html.append(
                record
            )

        if contains_markdown_fence(
            target
        ):

            markdown_fences.append(
                record
            )

    return (
        html_records,
        missing_html,
        markdown_fences,
    )


def print_duplicate_examples(
    duplicates: Counter,
    title: str,
    limit: int = 10,
):
    # показывает примеры дубликатов

    if not duplicates:

        print(
            "Дубликатов не найдено."
        )

        return

    for value, count in duplicates.most_common(
        limit
    ):

        print()
        print(
            f"Количество: {count}"
        )

        print(value)


def print_examples(
    records: list[dict],
    count: int = 3,
):
    # выводит несколько записей

    print()

    for record in records[:count]:

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
            "Cosine: "
            f"{record.get('cosine_similarity')}"
        )

        print(
            "Levenshtein: "
            f"{record.get('normalized_levenshtein_distance')}"
        )

        print()
        print("-" * 70)


def validate_dataset():
    # основная функция проверки

    print()
    print("=" * 70)
    print("ПРОВЕРКА DATASET V2")
    print("=" * 70)
    print()

    print(
        f"Файл: {DATASET_PATH}"
    )

    print()

    records = load_dataset()

    total = len(records)

    print("=" * 70)
    print("ОБЩАЯ СТАТИСТИКА")
    print("=" * 70)
    print()

    print(
        f"Всего записей: {total}"
    )

    valid_count = sum(
        1
        for record in records
        if record.get("valid") is True
    )

    invalid_count = (
        total
        - valid_count
    )

    print(
        f"valid=True: {valid_count}"
    )

    print(
        f"valid=False: {invalid_count}"
    )

    print()
    print("=" * 70)
    print("ЯЗЫКИ")
    print("=" * 70)
    print()

    languages = Counter(
        record.get(
            "language",
            "unknown",
        )
        for record in records
    )

    print_counter(
        languages,
        total,
    )

    print()
    print("=" * 70)
    print("ФОРМАТЫ")
    print("=" * 70)
    print()

    formats = Counter(
        record.get(
            "format",
            "unknown",
        )
        for record in records
    )

    print_counter(
        formats,
        total,
    )

    print()
    print("=" * 70)
    print("ДОМЕНЫ")
    print("=" * 70)
    print()

    domains = Counter(
        record.get(
            "domain",
            "unknown",
        )
        for record in records
    )

    print_counter(
        domains,
        total,
    )

    print()
    print("=" * 70)
    print("TRANSFORMATIONS")
    print("=" * 70)
    print()

    transformations = Counter(
        record.get(
            "transformation",
            "unknown",
        )
        for record in records
    )

    print_counter(
        transformations,
        total,
    )

    print()
    print("=" * 70)
    print("МОДЕЛИ")
    print("=" * 70)
    print()

    models = Counter(
        record.get(
            "model",
            "unknown",
        )
        for record in records
    )

    print_counter(
        models,
        total,
    )

    (
        cosine_values,
        levenshtein_values,
    ) = similarity_statistics(
        records
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

    if levenshtein_values:

        print()

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

    print()
    print("=" * 70)
    print("SIMILARITY ПО TRANSFORMATIONS")
    print("=" * 70)
    print()

    print_transformation_similarity(
        records
    )

    print()
    print("=" * 70)
    print("ДУБЛИКАТЫ")
    print("=" * 70)
    print()

    source_duplicates = find_duplicates(
        records,
        "source",
    )

    target_duplicates = find_duplicates(
        records,
        "target",
    )

    unique_sources = len(
        {
            record.get("source")
            for record in records
            if isinstance(
                record.get("source"),
                str,
            )
        }
    )

    unique_targets = len(
        {
            record.get("target")
            for record in records
            if isinstance(
                record.get("target"),
                str,
            )
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
        f"{len(source_duplicates)}"
    )

    print(
        f"Повторяющихся target: "
        f"{len(target_duplicates)}"
    )

    if source_duplicates:

        print()
        print(
            "Примеры повторяющихся source:"
        )

        print_duplicate_examples(
            source_duplicates,
            "source",
        )

    if target_duplicates:

        print()
        print(
            "Примеры повторяющихся target:"
        )

        print_duplicate_examples(
            target_duplicates,
            "target",
        )

    print()
    print("=" * 70)
    print("ПРОБЛЕМНЫЕ ЗАПИСИ")
    print("=" * 70)
    print()

    all_problems = []

    for record in records:

        problems = validate_record(
            record
        )

        all_problems.extend(
            problems
        )

    if all_problems:

        print(
            f"Найдено проблем: "
            f"{len(all_problems)}"
        )

        print()

        for problem in all_problems:

            print(
                f" {problem}"
            )

    else:

        print(
            " Базовая проверка "
            "пройдена без ошибок."
        )

    print()
    print("=" * 70)
    print("ПРОВЕРКА HTML")
    print("=" * 70)
    print()

    (
        html_records,
        missing_html,
        markdown_fences,
    ) = check_html_records(
        records
    )

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
        f"{len(markdown_fences)}"
    )

    if missing_html:

        print()
        print(
            "Примеры HTML-записей "
            "без HTML в target:"
        )

        for record in missing_html[:10]:

            print(
                f"- {record.get('id')}"
            )

            print(
                remove_html_tags(
                    record.get(
                        "target",
                        "",
                    )
                )
            )

    if markdown_fences:

        print()
        print(
            "Примеры HTML-записей "
            "с Markdown fences:"
        )

        for record in markdown_fences[:10]:

            print(
                f"- {record.get('id')}"
            )

            print(
                record.get(
                    "target",
                    "",
                )
            )

    short_targets = []

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

        target_text = remove_html_tags(
            target
        )

        if (
            target_text
            and len(target_text)
            < MIN_TARGET_LENGTH
        ):

            short_targets.append(
                record
            )

    print()
    print("=" * 70)
    print("КОРОТКИЕ TARGET")
    print("=" * 70)
    print()

    print(
        f"Минимальная длина: "
        f"{MIN_TARGET_LENGTH}"
    )

    print(
        f"Найдено коротких target: "
        f"{len(short_targets)}"
    )

    for record in short_targets[:20]:

        target = remove_html_tags(
            record.get(
                "target",
                "",
            )
        )

        print()
        print(
            f"ID: {record.get('id')}"
        )

        print(
            f"Длина: {len(target)}"
        )

        print(
            f"Target: {target}"
        )

    print()
    print("=" * 70)
    print("ПРИМЕРЫ ЗАПИСЕЙ")
    print("=" * 70)

    print_examples(
        records,
        count=3,
    )

    print()
    print("=" * 70)
    print("ПРОВЕРКА ЗАВЕРШЕНА")
    print("=" * 70)
    print()

    if all_problems:

        print(
            " Обнаружены проблемы. "
            "Посмотри раздел "
            "'ПРОБЛЕМНЫЕ ЗАПИСИ'."
        )

    elif missing_html:

        print(
            " Базовая проверка пройдена, "
            "но часть HTML target "
            "потеряла HTML-разметку."
        )

    elif short_targets:

        print(
            " Базовая проверка пройдена, "
            "но есть слишком короткие target."
        )

    else:

        print(
            " DATASET V2 ПРОШЁЛ "
            "БАЗОВУЮ ПРОВЕРКУ."
        )

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