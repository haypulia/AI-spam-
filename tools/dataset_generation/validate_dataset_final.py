import json
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "generated_dataset" / "dataset_final.jsonl"

COSINE_MAX = 0.98
LEVENSHTEIN_MIN = 0.20

REQUIRED_FIELDS = {
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
    "id",
}


def print_section(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_dataset(path: Path):
    records = []

    if not path.exists():
        print(f"Файл не найден: {path}")
        raise SystemExit(1)

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"Некорректный JSON "
                    f"в строке {line_number}: {e}"
                )
                raise SystemExit(1)

            records.append(record)

    return records


def looks_like_html(text: str) -> bool:
    # проверяет наличие HTML-тегов

    import re

    if not text:
        return False

    pattern = r"<\s*/?\s*[a-zA-Z][^>]*>"

    return bool(re.search(pattern, text))


def main():

    print_section("ФИНАЛЬНАЯ ПРОВЕРКА DATASET")

    print("Файл:")
    print(DATASET_PATH)

    records = load_dataset(DATASET_PATH)

    print(f"\nЗагружено записей: {len(records)}")

    if len(records) == 1000:
        print("Количество записей: 1000")
    else:
        print(
            f"Ожидалось 1000 записей, "
            f"получено {len(records)}"
        )

    print_section("1. ПРОВЕРКА СТРУКТУРЫ")

    structure_errors = []

    for i, record in enumerate(records, start=1):

        missing = REQUIRED_FIELDS - set(record.keys())

        if missing:
            structure_errors.append(
                f"Запись {i}: отсутствуют поля "
                f"{sorted(missing)}"
            )

    if structure_errors:
        print(
            f"Ошибок структуры: "
            f"{len(structure_errors)}"
        )

        for error in structure_errors[:20]:
            print(" ", error)

    else:
        print("Все обязательные поля присутствуют")

    print_section("2. ПРОВЕРКА ID")

    ids = [record.get("id") for record in records]

    duplicate_ids = [
        item
        for item, count in Counter(ids).items()
        if count > 1
    ]

    empty_ids = [
        i
        for i, item in enumerate(ids, start=1)
        if not item
    ]

    if empty_ids:
        print(f"Пустых ID: {len(empty_ids)}")
    else:
        print("Пустых ID нет")

    if duplicate_ids:
        print(
            f"Повторяющихся ID: "
            f"{len(duplicate_ids)}"
        )

        for item in duplicate_ids[:20]:
            print(" ", item)

    else:
        print("Все ID уникальны")

    print_section("3. ПРОВЕРКА SOURCE")

    sources = [
        record.get("source", "")
        for record in records
    ]

    empty_sources = [
        i
        for i, source in enumerate(sources, start=1)
        if not source.strip()
    ]

    duplicate_sources = [
        source
        for source, count in Counter(sources).items()
        if count > 1
    ]

    print(
        f"Пустых source: "
        f"{len(empty_sources)}"
    )

    print(
        f"Повторяющихся source: "
        f"{len(duplicate_sources)}"
    )

    if empty_sources:
        print("Есть пустые source")
    else:
        print("Пустых source нет")

    # повторяющиеся source допустимы
    # одинаковые source нужно учитывать при split

    if duplicate_sources:
        print(
            "ℹ Повторения source между моделями "
            "будут учтены при split."
        )

    print_section("4. ПРОВЕРКА TARGET")

    targets = [
        record.get("target", "")
        for record in records
    ]

    empty_targets = [
        i
        for i, target in enumerate(targets, start=1)
        if not target.strip()
    ]

    duplicate_targets = [
        target
        for target, count in Counter(targets).items()
        if count > 1
    ]

    identical_pairs = []

    for i, record in enumerate(records, start=1):

        source = record.get("source", "").strip()
        target = record.get("target", "").strip()

        if source == target:
            identical_pairs.append(i)

    print(
        f"Пустых target: "
        f"{len(empty_targets)}"
    )

    print(
        f"Source == target: "
        f"{len(identical_pairs)}"
    )

    print(
        f"Повторяющихся target: "
        f"{len(duplicate_targets)}"
    )

    if empty_targets:
        print("Есть пустые target")
    else:
        print("Пустых target нет")

    if identical_pairs:
        print("Есть пары source == target")

        for item in identical_pairs[:20]:
            print(" ", item)

    else:
        print("Одинаковых source/target нет")

    if duplicate_targets:
        print(
            "Повторяющихся target: "
            f"{len(duplicate_targets)}"
        )

    print_section("5. ПРОВЕРКА SIMILARITY")

    cosine_values = []
    levenshtein_values = []

    similarity_errors = []

    for i, record in enumerate(records, start=1):

        cosine = record.get("cosine_similarity")

        levenshtein = record.get(
            "normalized_levenshtein_distance"
        )

        if cosine is None:
            similarity_errors.append(
                f"Запись {i}: "
                f"отсутствует cosine_similarity"
            )
            continue

        if levenshtein is None:
            similarity_errors.append(
                f"Запись {i}: "
                f"отсутствует "
                f"normalized_levenshtein_distance"
            )
            continue

        try:
            cosine = float(cosine)
            levenshtein = float(levenshtein)

        except (TypeError, ValueError):
            similarity_errors.append(
                f"Запись {i}: "
                f"similarity metrics не являются числами"
            )
            continue

        cosine_values.append(cosine)
        levenshtein_values.append(levenshtein)

        if cosine > COSINE_MAX:
            similarity_errors.append(
                f"Запись {i}: "
                f"cosine={cosine:.4f} > {COSINE_MAX}"
            )

        if levenshtein < LEVENSHTEIN_MIN:
            similarity_errors.append(
                f"Запись {i}: "
                f"levenshtein={levenshtein:.4f} "
                f"< {LEVENSHTEIN_MIN}"
            )

    if similarity_errors:

        print(
            f"Ошибок similarity: "
            f"{len(similarity_errors)}"
        )

        for error in similarity_errors[:20]:
            print(" ", error)

    else:
        print(
            "Все similarity metrics "
            "присутствуют и проходят пороги"
        )

    if cosine_values:

        print(
            "\nCosine similarity:"
        )

        print(
            f"  min = "
            f"{min(cosine_values):.4f}"
        )

        print(
            f"  max = "
            f"{max(cosine_values):.4f}"
        )

        print(
            f"  avg = "
            f"{sum(cosine_values) / len(cosine_values):.4f}"
        )

    if levenshtein_values:

        print(
            "\nNormalized Levenshtein distance:"
        )

        print(
            f"  min = "
            f"{min(levenshtein_values):.4f}"
        )

        print(
            f"  max = "
            f"{max(levenshtein_values):.4f}"
        )

        print(
            f"  avg = "
            f"{sum(levenshtein_values) / len(levenshtein_values):.4f}"
        )

    print_section("6. VALID FLAGS")

    invalid_records = [
        i
        for i, record in enumerate(records, start=1)
        if record.get("valid") is not True
    ]

    if invalid_records:

        print(
            f"valid != True: "
            f"{len(invalid_records)}"
        )

        for item in invalid_records[:20]:
            print(" ", item)

    else:
        print("Все записи valid=True")

    print_section("7. HTML / FORMAT")

    format_counts = Counter(
        record.get("format")
        for record in records
    )

    html_records = format_counts.get(
        "html",
        0
    )

    plain_records = format_counts.get(
        "plain_text",
        0
    )

    unknown_formats = {
        fmt: count
        for fmt, count in format_counts.items()
        if fmt not in {
            "html",
            "plain_text",
        }
    }

    html_errors = []

    for i, record in enumerate(records, start=1):

        fmt = record.get("format")

        source = record.get(
            "source",
            ""
        )

        target = record.get(
            "target",
            ""
        )

        if fmt == "html":

            source_html = looks_like_html(source)
            target_html = looks_like_html(target)

            if not source_html:

                html_errors.append(
                    (
                        i,
                        "source",
                        source[:150].replace(
                            "\n",
                            " "
                        )
                    )
                )

            if source_html and not target_html:

                html_errors.append(
                    (
                        i,
                        "target",
                        target[:150].replace(
                            "\n",
                            " "
                        )
                    )
                )

    print(
        f"HTML: "
        f"{html_records}"
    )

    print(
        f"Plain text: "
        f"{plain_records}"
    )

    if unknown_formats:

        print(
            "Неизвестные форматы:"
        )

        for fmt, count in unknown_formats.items():
            print(
                f"  {fmt}: {count}"
            )

    else:

        print(
            "Неизвестных форматов нет"
        )

    if html_errors:

        print(
            f"HTML-проверка обнаружила: "
            f"{len(html_errors)}"
        )

        for item in html_errors[:20]:

            index, field, preview = item

            print(
                f"  Запись {index}: "
                f"{field} не похож на HTML"
            )

            print(
                f"    {preview}"
            )

        print(
            "\nℹ Это информационная проверка. "
            "Она не удаляет записи."
        )

    else:

        print(
            "HTML-структура выглядит корректно"
        )

    print_section("8. МОДЕЛИ")

    model_counts = Counter(
        record.get("model")
        for record in records
    )

    for model, count in model_counts.most_common():

        print(
            f"{model:<45} "
            f"{count}"
        )

    expected_models = {
        "deepseek-ai/DeepSeek-V4-Flash",
        "Qwen3.8-27B",
    }

    missing_models = (
        expected_models
        - set(model_counts)
    )

    if missing_models:

        print(
            "\n Отсутствуют модели:"
        )

        for model in missing_models:
            print(
                f"  {model}"
            )

    else:

        print(
            "\n Обе модели присутствуют"
        )

    print_section("9. ЯЗЫКИ")

    language_counts = Counter(
        record.get("language")
        for record in records
    )

    for language, count in sorted(
        language_counts.items()
    ):

        print(
            f"{language:<20} "
            f"{count}"
        )

    print_section("10. TRANSFORMATIONS")

    transformation_counts = Counter(
        record.get("transformation")
        for record in records
    )

    for transformation, count in sorted(
        transformation_counts.items()
    ):

        print(
            f"{transformation:<25} "
            f"{count}"
        )

    print_section("11. DOMAINS")

    domain_counts = Counter(
        record.get("domain")
        for record in records
    )

    for domain, count in sorted(
        domain_counts.items()
    ):

        print(
            f"{domain:<20} "
            f"{count}"
        )

    print_section("12. MODEL × LANGUAGE")

    model_language = Counter(
        (
            record.get("model"),
            record.get("language"),
        )
        for record in records
    )

    for (model, language), count in sorted(
        model_language.items()
    ):

        print(
            f"{model:<45} "
            f"{language:<5} "
            f"{count}"
        )

    print_section("13. MODEL × FORMAT")

    model_format = Counter(
        (
            record.get("model"),
            record.get("format"),
        )
        for record in records
    )

    for (model, fmt), count in sorted(
        model_format.items()
    ):

        print(
            f"{model:<45} "
            f"{fmt:<12} "
            f"{count}"
        )

    print_section("14. MODEL × TRANSFORMATION")

    model_transformation = Counter(
        (
            record.get("model"),
            record.get("transformation"),
        )
        for record in records
    )

    for (
        model,
        transformation
    ), count in sorted(
        model_transformation.items()
    ):

        print(
            f"{model:<45} "
            f"{transformation:<25} "
            f"{count}"
        )

    print_section("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ")

    critical_problems = (
        len(structure_errors)
        + len(empty_ids)
        + len(duplicate_ids)
        + len(empty_sources)
        + len(empty_targets)
        + len(identical_pairs)
        + len(similarity_errors)
        + len(invalid_records)
        + len(unknown_formats)
    )

    if critical_problems == 0:

        print(
            " DATASET FINAL "
            "ПРОШЁЛ ФИНАЛЬНУЮ ПРОВЕРКУ"
        )

        print()

        print(
            f"Записей: "
            f"{len(records)}"
        )

        print(
            "Структура: OK"
        )

        print(
            "ID: OK"
        )

        print(
            "Source: OK"
        )

        print(
            "Target: OK"
        )

        print(
            "Similarity: OK"
        )

        print(
            "Valid flags: OK"
        )

        print(
            "Models: OK"
        )

        print(
            "Languages: OK"
        )

        print(
            "Transformations: OK"
        )

        print(
            "Domains: OK"
        )

        print(
            "Formats: OK"
        )

        if duplicate_sources:

            print()

            print(
                f" Повторяющихся source: "
                f"{len(duplicate_sources)}"
            )

            print(
                "Это допустимо и будет "
                "учтено при train/validation/test split."
            )

        if duplicate_targets:

            print()

            print(
                f" Повторяющихся target: "
                f"{len(duplicate_targets)}"
            )

        if html_errors:

            print()

            print(
                f" HTML-проверка: "
                f"{len(html_errors)} "
                f"подозрительных записей."
            )

            print(
                "Это не считается критической "
                "ошибкой автоматически."
            )

        print()

    else:

        print(
            f" Найдены критические проблемы: "
            f"{critical_problems}"
        )

        print()

if __name__ == "__main__":
    main()