# объединяет: generated_dataset/dataset_v2.jsonl и generated_dataset/dataset_qwen.jsonl
# в: generated_dataset/dataset_final.jsonl

import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "generated_dataset"

DATASET_V2_PATH = DATASET_DIR / "dataset_v2.jsonl"

DATASET_QWEN_PATH = DATASET_DIR / "dataset_qwen.jsonl"

DATASET_FINAL_PATH = DATASET_DIR / "dataset_final.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    # загружает JSONL-файл

    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден:\n{path}"
        )

    records = []

    with path.open(
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

                raise RuntimeError(
                    f"Ошибка JSON в файле {path}, "
                    f"строка {line_number}:\n"
                    f"{error}"
                ) from error

            if not isinstance(record, dict):

                raise RuntimeError(
                    f"Запись в {path}, "
                    f"строка {line_number}, "
                    f"не является объектом JSON."
                )

            records.append(record)

    return records


def check_required_fields(
    records: list[dict],
    dataset_name: str,
) -> None:
    # проверяет наличие основных полей

    required_fields = {
        "id",
        "language",
        "domain",
        "format",
        "model",
        "transformation",
        "source",
        "target",
    }

    problems = []

    for index, record in enumerate(
        records,
        start=1,
    ):

        missing = [
            field
            for field in required_fields
            if field not in record
        ]

        if missing:

            problems.append(
                f"Запись {index}: "
                f"отсутствуют поля {missing}"
            )

    if problems:

        print()
        print(
            f"ОШИБКИ СТРУКТУРЫ: {dataset_name}"
        )

        for problem in problems[:20]:
            print(problem)

        if len(problems) > 20:
            print(
                f"... и ещё "
                f"{len(problems) - 20}"
            )

        raise RuntimeError(
            f"В {dataset_name} обнаружены "
            f"проблемы структуры."
        )


def find_duplicates(
    records: list[dict],
    field: str,
) -> list[str]:
    # возвращает повторяющиеся значения поля

    values = [
        record.get(field)
        for record in records
    ]

    counter = Counter(values)

    return [
        value
        for value, count in counter.items()
        if count > 1
    ]


def check_dataset(
    records: list[dict],
    dataset_name: str,
) -> None:

    print()
    print(
        f"Проверка: {dataset_name}"
    )

    print(
        f"Записей: {len(records)}"
    )

    check_required_fields(
        records,
        dataset_name,
    )

    for field in (
        "id",
        "source",
        "target",
    ):

        duplicates = find_duplicates(
            records,
            field,
        )

        if duplicates:

            print(
                f"⚠ Дубли {field}: "
                f"{len(duplicates)}"
            )

            for value in duplicates[:10]:
                print(
                    f"  {value}"
                )

        else:

            print(
                f"✓ Уникальные {field}"
            )


def check_cross_duplicates(
    dataset_v2: list[dict],
    dataset_qwen: list[dict],
) -> None:

    print()
    print("=" * 70)
    print("ПРОВЕРКА ПЕРЕСЕЧЕНИЙ")
    print("=" * 70)

    for field in (
        "id",
        "source",
        "target",
    ):

        values_v2 = {
            record.get(field)
            for record in dataset_v2
        }

        values_qwen = {
            record.get(field)
            for record in dataset_qwen
        }

        intersection = (
            values_v2
            & values_qwen
        )

        if intersection:

            print(
                f"⚠ Пересечения {field}: "
                f"{len(intersection)}"
            )

            for value in list(
                intersection
            )[:10]:

                print(
                    f"  {value}"
                )

        else:

            print(
                f"✓ Пересечений {field} нет"
            )


def normalize_qwen_ids(
    records: list[dict],
) -> list[dict]:
    # приводит Qwen ID к безопасному формату

    normalized = []

    for index, record in enumerate(
        records,
        start=1,
    ):

        record = dict(record)

        if not record.get("id"):

            record["id"] = (
                f"generated_qwen_"
                f"{index:06d}"
            )

        normalized.append(record)

    return normalized


def merge_datasets(
    dataset_v2: list[dict],
    dataset_qwen: list[dict],
) -> list[dict]:

    dataset_qwen = normalize_qwen_ids(
        dataset_qwen
    )

    merged = []

    merged.extend(dataset_v2)

    merged.extend(dataset_qwen)

    return merged


def validate_final_dataset(
    records: list[dict],
) -> None:

    print()
    print("=" * 70)
    print("ФИНАЛЬНАЯ ПРОВЕРКА")
    print("=" * 70)

    # IDs

    ids = [
        record.get("id")
        for record in records
    ]

    duplicate_ids = [
        value
        for value, count
        in Counter(ids).items()
        if count > 1
    ]

    if duplicate_ids:

        print(
            f"✗ Повторяющихся ID: "
            f"{len(duplicate_ids)}"
        )

        for value in duplicate_ids[:10]:
            print(
                f"  {value}"
            )

        raise RuntimeError(
            "В итоговом датасете есть "
            "дублирующиеся ID."
        )

    print(
        "✓ ID уникальны"
    )

    # source

    sources = [
        record.get("source")
        for record in records
    ]

    duplicate_sources = [
        value
        for value, count
        in Counter(sources).items()
        if count > 1
    ]

    if duplicate_sources:

        print(
            f"⚠ Повторяющихся source: "
            f"{len(duplicate_sources)}"
        )

        for value in duplicate_sources[:5]:
            print()
            print(value)

    else:

        print(
            "✓ Все source уникальны"
        )

    # target

    targets = [
        record.get("target")
        for record in records
    ]

    duplicate_targets = [
        value
        for value, count
        in Counter(targets).items()
        if count > 1
    ]

    if duplicate_targets:

        print(
            f"⚠ Повторяющихся target: "
            f"{len(duplicate_targets)}"
        )

        for value in duplicate_targets[:5]:
            print()
            print(value)

    else:

        print(
            "✓ Все target уникальны"
        )

    # models

    models = Counter(
        record.get("model")
        for record in records
    )

    print()
    print("МОДЕЛИ:")

    for model, count in models.most_common():

        print(
            f"  {model:<45} {count}"
        )

    # valid

    invalid = [
        record
        for record in records
        if record.get("valid") is False
    ]

    if invalid:

        print()
        print(
            f"⚠ valid=False: "
            f"{len(invalid)}"
        )

    else:

        print()
        print(
            "✓ Все записи valid=True"
        )


def save_jsonl(
    records: list[dict],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_statistics(
    records: list[dict],
) -> None:

    print()
    print("=" * 70)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 70)

    print()
    print(
        f"Всего записей: {len(records)}"
    )

    print()
    print("ЯЗЫКИ:")

    languages = Counter(
        record.get("language")
        for record in records
    )

    for language, count in languages.most_common():

        print(
            f"  {language:<20} {count}"
        )

    print()
    print("ФОРМАТЫ:")

    formats = Counter(
        record.get("format")
        for record in records
    )

    for email_format, count in formats.most_common():

        print(
            f"  {email_format:<20} {count}"
        )

    print()
    print("МОДЕЛИ:")

    models = Counter(
        record.get("model")
        for record in records
    )

    for model, count in models.most_common():

        print(
            f"  {model:<45} {count}"
        )

    print()
    print("TRANSFORMATIONS:")

    transformations = Counter(
        record.get("transformation")
        for record in records
    )

    for transformation, count in (
        transformations.most_common()
    ):

        print(
            f"  {transformation:<25} {count}"
        )

    print()
    print("ДОМЕНЫ:")

    domains = Counter(
        record.get("domain")
        for record in records
    )

    for domain, count in domains.most_common():

        print(
            f"  {domain:<20} {count}"
        )


def main():

    print()
    print("=" * 70)
    print("ОБЪЕДИНЕНИЕ DATASET V2 + QWEN")
    print("=" * 70)

    print()
    print(
        f"Dataset V2:"
        f"\n{DATASET_V2_PATH}"
    )

    print()
    print(
        f"Dataset Qwen:"
        f"\n{DATASET_QWEN_PATH}"
    )

    print()
    print(
        f"Итоговый файл:"
        f"\n{DATASET_FINAL_PATH}"
    )

    # load

    print()
    print("=" * 70)
    print("ЗАГРУЗКА")
    print("=" * 70)

    dataset_v2 = load_jsonl(
        DATASET_V2_PATH
    )

    dataset_qwen = load_jsonl(
        DATASET_QWEN_PATH
    )

    print()
    print(
        f"Dataset V2: "
        f"{len(dataset_v2)} записей"
    )

    print(
        f"Dataset Qwen: "
        f"{len(dataset_qwen)} записей"
    )

    # check individual datasets

    check_dataset(
        dataset_v2,
        "dataset_v2",
    )

    check_dataset(
        dataset_qwen,
        "dataset_qwen",
    )

    # cross check

    check_cross_duplicates(
        dataset_v2,
        dataset_qwen,
    )

    # merge

    print()
    print("=" * 70)
    print("ОБЪЕДИНЕНИЕ")
    print("=" * 70)

    merged = merge_datasets(
        dataset_v2,
        dataset_qwen,
    )

    print()
    print(
        f"Dataset V2: "
        f"{len(dataset_v2)}"
    )

    print(
        f"Qwen: "
        f"{len(dataset_qwen)}"
    )

    print(
        f"Итого: "
        f"{len(merged)}"
    )

    # validate

    validate_final_dataset(
        merged
    )

    # save

    save_jsonl(
        merged,
        DATASET_FINAL_PATH,
    )

    # statistics

    print_statistics(
        merged
    )

    # finish

    print()
    print("=" * 70)
    print("ГОТОВО")
    print("=" * 70)

    print()
    print(
        f"Итоговый dataset:"
        f"\n{DATASET_FINAL_PATH}"
    )

    print()
    print(
        "✓ Dataset V2 сохранён без изменений."
    )

    print(
        "✓ Qwen-записи добавлены."
    )

    print(
        "✓ Итоговый JSONL создан."
    )

    print()


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Операция остановлена пользователем."
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("КРИТИЧЕСКАЯ ОШИБКА")
        print("=" * 70)
        print()
        print(error)
        print()
        raise