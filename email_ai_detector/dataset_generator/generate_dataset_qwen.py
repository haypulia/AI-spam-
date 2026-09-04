# массовая генерация второго набора датасета через Qwen3.8-27B
# использует те же исходные письма и transformations, что и основной генератор

import json
import random
import re
from pathlib import Path

from dataset_generator.config import (
    SOURCE_EMAILS_FILE,
    OUTPUT_DIR,
    EMBEDDING_MODEL_NAME,
    COSINE_SIMILARITY_THRESHOLD,
    NORMALIZED_LEVENSHTEIN_THRESHOLD,
)

from dataset_generator.generator import (
    DeepCodeGenerator,
)

from dataset_generator.prompts import (
    build_transformation_prompt,
)

from dataset_generator.transformations import (
    TRANSFORMATIONS,
)

from dataset_generator.similarity import (
    SimilarityCalculator,
    is_sufficiently_different,
)


MODEL_NAME = "Qwen3.8-27B"

TARGET_RECORDS = 495

MAX_ATTEMPTS_PER_PAIR = 5

RANDOM_SEED = 42

OUTPUT_FILENAME = "dataset_qwen.jsonl"


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

SOURCE_EMAILS_PATH = (
    PROJECT_ROOT / SOURCE_EMAILS_FILE
)

OUTPUT_DIR_PATH = (
    PROJECT_ROOT / OUTPUT_DIR
)

OUTPUT_FILE_PATH = (
    OUTPUT_DIR_PATH / OUTPUT_FILENAME
)


def load_source_emails() -> list[dict]:
    # загружает исходные письма из JSONL

    if not SOURCE_EMAILS_PATH.exists():
        raise FileNotFoundError(
            f"Файл с исходными письмами не найден:\n"
            f"{SOURCE_EMAILS_PATH}"
        )

    records = []

    with SOURCE_EMAILS_PATH.open(
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

            except json.JSONDecodeError:
                print(
                    f"⚠ Пропущена некорректная "
                    f"JSON-запись, строка {line_number}"
                )
                continue

            if isinstance(record, dict):
                records.append(record)

    if not records:
        raise RuntimeError(
            "В source_emails_new.jsonl "
            "нет корректных записей."
        )

    return records


def extract_html_fragments(
    text: str,
) -> list[str]:
    # извлекает содержательные HTML-фрагменты

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    candidates = []

    for line in lines:

        without_tags = re.sub(
            r"<[^>]+>",
            "",
            line,
        )

        without_tags = without_tags.strip()

        if not without_tags:
            continue

        if len(without_tags) < 20:
            continue

        candidates.append(line)

    return candidates


def extract_plain_text_fragments(
    text: str,
) -> list[str]:
    # извлекает содержательные фрагменты обычного текста

    text = text.strip()

    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    valid_paragraphs = []

    for paragraph in paragraphs:

        if len(paragraph) >= 20:
            valid_paragraphs.append(
                paragraph
            )

    if valid_paragraphs:
        return valid_paragraphs

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    valid_lines = [
        line
        for line in lines
        if len(line) >= 20
    ]

    if valid_lines:
        return valid_lines

    if len(text) >= 20:
        return [text]

    return []


def extract_fragments(
    text: str,
    email_format: str,
) -> list[str]:
    # возвращает список подходящих фрагментов

    if not isinstance(text, str):
        return []

    text = text.strip()

    if not text:
        return []

    if email_format == "html":
        return extract_html_fragments(text)

    return extract_plain_text_fragments(text)


def build_source_pool(
    records: list[dict],
) -> list[dict]:
    # создаёт пул уникальных source-фрагментов

    pool = []

    seen_sources = set()

    for record in records:

        text = record.get(
            "text",
            "",
        )

        email_format = record.get(
            "format",
            "plain_text",
        )

        fragments = extract_fragments(
            text=text,
            email_format=email_format,
        )

        for fragment in fragments:

            normalized = fragment.strip()

            if not normalized:
                continue

            if normalized in seen_sources:
                continue

            seen_sources.add(
                normalized
            )

            pool.append(
                {
                    "record": record,
                    "source": normalized,
                }
            )

    return pool


def clean_target(
    target: str,
) -> str:
    # очищает ответ модели от markdown fences

    if not isinstance(target, str):
        return ""

    target = target.strip()

    if target.startswith("```html"):

        target = target[
            len("```html"):
        ].strip()

        if target.endswith("```"):
            target = target[:-3].strip()

    elif target.startswith("```"):

        target = target[3:].strip()

        if target.endswith("```"):
            target = target[:-3].strip()

    return target.strip()


def validate_target(
    source: str,
    target: str,
    email_format: str,
) -> bool:
    # выполняет базовую проверку target

    if not target:
        return False

    if len(target.strip()) < 20:
        return False

    if target.strip() == source.strip():
        return False

    if email_format == "html":

        source_has_html = bool(
            re.search(
                r"<[a-zA-Z][^>]*>",
                source,
            )
        )

        target_has_html = bool(
            re.search(
                r"<[a-zA-Z][^>]*>",
                target,
            )
        )

        if source_has_html and not target_has_html:
            return False

        if target.startswith("```"):
            return False

    return True


def get_text_length(
    text: str,
) -> int:
    # возвращает длину текстового содержимого

    if not isinstance(text, str):
        return 0

    clean = re.sub(
        r"<[^>]+>",
        "",
        text,
    )

    return len(clean.strip())


def generate_pair(
    item: dict,
    transformation: str,
    generator: DeepCodeGenerator,
    similarity_calculator: SimilarityCalculator,
) -> dict | None:

    record = item["record"]

    source = item["source"]

    language = record.get(
        "language",
        "ru",
    )

    domain = record.get(
        "domain",
        "unknown",
    )

    email_format = record.get(
        "format",
        "plain_text",
    )

    prompt = build_transformation_prompt(
        text=source,
        transformation=transformation,
        language=language,
        email_format=email_format,
    )

    for attempt in range(
        1,
        MAX_ATTEMPTS_PER_PAIR + 1,
    ):

        print()
        print(
            f"ГЕНЕРАЦИЯ "
            f"(попытка {attempt}/"
            f"{MAX_ATTEMPTS_PER_PAIR})"
        )

        try:

            target = generator.generate(
                prompt=prompt
            )

        except Exception as error:

            print(
                f"⚠ Ошибка генерации: "
                f"{error}"
            )

            continue

        target = clean_target(
            target
        )

        if not validate_target(
            source=source,
            target=target,
            email_format=email_format,
        ):

            print(
                "⚠ Target не прошёл "
                "базовую проверку."
            )

            continue

        print()
        print("SOURCE:")
        print(source)

        print()
        print("TARGET:")
        print(target)

        try:

            is_valid, metrics = (
                is_sufficiently_different(
                    source=source,
                    target=target,
                    cosine_threshold=(
                        COSINE_SIMILARITY_THRESHOLD
                    ),
                    levenshtein_threshold=(
                        NORMALIZED_LEVENSHTEIN_THRESHOLD
                    ),
                    similarity_calculator=(
                        similarity_calculator
                    ),
                )
            )

        except Exception as error:

            print(
                f"⚠ Ошибка similarity: "
                f"{error}"
            )

            continue

        cosine = metrics[
            "cosine_similarity"
        ]

        levenshtein = metrics[
            "normalized_levenshtein_distance"
        ]

        print()
        print(
            f"Cosine similarity: "
            f"{cosine:.4f}"
        )

        print(
            f"Normalized Levenshtein: "
            f"{levenshtein:.4f}"
        )

        if is_valid:

            print()
            print(
                "✓ ПАРА ПРОШЛА ФИЛЬТР"
            )

            return {
                "language": language,
                "domain": domain,
                "format": email_format,
                "model": MODEL_NAME,
                "transformation": transformation,
                "source": source,
                "target": target,
                "cosine_similarity": cosine,
                "normalized_levenshtein_distance": (
                    levenshtein
                ),
                "valid": True,
            }

        print()
        print(
            "✗ Пара не прошла фильтр."
        )

        print(
            f"Причина: "
            f"{metrics['reason']}"
        )

    return None


def save_record(
    record: dict,
) -> None:
    # добавляет запись в dataset_qwen.jsonl

    OUTPUT_DIR_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def get_existing_record_count() -> int:
    # возвращает количество сохранённых Qwen-записей

    if not OUTPUT_FILE_PATH.exists():
        return 0

    count = 0

    with OUTPUT_FILE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            if line.strip():
                count += 1

    return count


def get_existing_sources() -> set[str]:
    # загружает source уже существующих Qwen-записей

    if not OUTPUT_FILE_PATH.exists():
        return set()

    sources = set()

    with OUTPUT_FILE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                continue

            source = record.get(
                "source"
            )

            if isinstance(source, str):
                sources.add(
                    source.strip()
                )

    return sources


def generate_dataset():
    # генерирует TARGET_RECORDS записей Qwen

    print()
    print("=" * 80)
    print("МАССОВАЯ ГЕНЕРАЦИЯ DATASET — QWEN")
    print("=" * 80)
    print()

    print(
        f"Модель: {MODEL_NAME}"
    )

    print(
        f"Целевое количество новых записей: "
        f"{TARGET_RECORDS}"
    )

    print(
        f"Файл результата: "
        f"{OUTPUT_FILE_PATH}"
    )

    print()

    random.seed(
        RANDOM_SEED
    )

    records = load_source_emails()

    print(
        f"Исходных писем: "
        f"{len(records)}"
    )

    source_pool = build_source_pool(
        records
    )

    print(
        f"Уникальных source-фрагментов: "
        f"{len(source_pool)}"
    )

    if not source_pool:
        raise RuntimeError(
            "Не найдено подходящих "
            "source-фрагментов."
        )

    OUTPUT_DIR_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_count = (
        get_existing_record_count()
    )

    existing_sources = (
        get_existing_sources()
    )

    print(
        f"Уже есть Qwen-записей: "
        f"{existing_count}"
    )

    print(
        f"Уже использовано Qwen source: "
        f"{len(existing_sources)}"
    )

    if existing_sources:

        source_pool = [
            item
            for item in source_pool
            if item["source"].strip()
            not in existing_sources
        ]

    print(
        f"Доступно новых source-фрагментов: "
        f"{len(source_pool)}"
    )

    if not source_pool:
        raise RuntimeError(
            "Все доступные source уже "
            "использованы в Qwen dataset."
        )

    print()
    print(
        "Инициализируем Qwen generator..."
    )

    generator = DeepCodeGenerator(
        model=MODEL_NAME
    )

    print(
        "Инициализируем similarity calculator..."
    )

    similarity_calculator = (
        SimilarityCalculator(
            EMBEDDING_MODEL_NAME
        )
    )

    print()

    generated = 0

    failed = 0

    attempts = 0

    used_sources = set(
        existing_sources
    )

    transformation_counts = {
        transformation: 0
        for transformation in TRANSFORMATIONS
    }

    language_counts = {}

    format_counts = {}

    domain_counts = {}

    while generated < TARGET_RECORDS:

        available_items = [
            item
            for item in source_pool
            if item["source"] not in used_sources
        ]

        if not available_items:

            print()
            print(
                "⚠ Закончились уникальные "
                "source-фрагменты."
            )

            break

        item = random.choice(
            available_items
        )

        source = item["source"]

        record = item["record"]

        transformation = random.choice(
            TRANSFORMATIONS
        )

        attempts += 1

        print()
        print("=" * 80)
        print(
            f"ПРОГРЕСС: "
            f"{generated}/{TARGET_RECORDS}"
        )
        print(
            f"Попытка: {attempts}"
        )
        print(
            f"Transformation: "
            f"{transformation}"
        )
        print(
            f"Language: "
            f"{record.get('language', 'unknown')}"
        )
        print(
            f"Domain: "
            f"{record.get('domain', 'unknown')}"
        )
        print(
            f"Format: "
            f"{record.get('format', 'unknown')}"
        )
        print("=" * 80)

        result = generate_pair(
            item=item,
            transformation=transformation,
            generator=generator,
            similarity_calculator=(
                similarity_calculator
            ),
        )

        if result is not None:

            generated += 1

            record_id = (
                f"generated_qwen_"
                f"{existing_count + generated:06d}"
            )

            result["id"] = record_id

            save_record(
                result
            )

            used_sources.add(
                source
            )

            transformation_counts[
                transformation
            ] += 1

            language = result[
                "language"
            ]

            language_counts[
                language
            ] = (
                language_counts.get(
                    language,
                    0,
                )
                + 1
            )

            email_format = result[
                "format"
            ]

            format_counts[
                email_format
            ] = (
                format_counts.get(
                    email_format,
                    0,
                )
                + 1
            )

            domain = result[
                "domain"
            ]

            domain_counts[
                domain
            ] = (
                domain_counts.get(
                    domain,
                    0,
                )
                + 1
            )

            print()
            print(
                f"✓ ЗАПИСЬ СОХРАНЕНА: "
                f"{record_id}"
            )

            print(
                f"Прогресс: "
                f"{generated}/{TARGET_RECORDS}"
            )

        else:

            failed += 1

            print()
            print(
                "✗ Не удалось создать "
                "валидную пару."
            )

    print()
    print("=" * 80)
    print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)
    print()

    print(
        f"Модель: {MODEL_NAME}"
    )

    print(
        f"Исходных писем: "
        f"{len(records)}"
    )

    print(
        f"Source-фрагментов: "
        f"{len(source_pool)}"
    )

    print(
        f"Создано новых записей: "
        f"{generated}"
    )

    print(
        f"Неудачных попыток: "
        f"{failed}"
    )

    print(
        f"Всего попыток: "
        f"{attempts}"
    )

    print(
        f"Всего Qwen-записей: "
        f"{existing_count + generated}"
    )

    print()

    print(
        "Файл результата:"
    )

    print(
        OUTPUT_FILE_PATH
    )

    print()
    print("=" * 80)
    print("СТАТИСТИКА НОВОЙ ГЕНЕРАЦИИ")
    print("=" * 80)

    print()
    print("ЯЗЫКИ:")

    for language, count in sorted(
        language_counts.items()
    ):

        print(
            f"  {language:<20} "
            f"{count}"
        )

    print()
    print("ФОРМАТЫ:")

    for email_format, count in sorted(
        format_counts.items()
    ):

        print(
            f"  {email_format:<20} "
            f"{count}"
        )

    print()
    print("ДОМЕНЫ:")

    for domain, count in sorted(
        domain_counts.items()
    ):

        print(
            f"  {domain:<20} "
            f"{count}"
        )

    print()
    print("TRANSFORMATIONS:")

    for transformation, count in sorted(
        transformation_counts.items()
    ):

        print(
            f"  {transformation:<28} "
            f"{count}"
        )

    print()
    print("МОДЕЛЬ:")

    print(
        f"  {MODEL_NAME:<45} "
        f"{generated}"
    )

    print()
    print("=" * 80)
    print()


if __name__ == "__main__":

    try:

        generate_dataset()

    except KeyboardInterrupt:

        print()
        print(
            "Генерация остановлена пользователем."
        )

    except Exception as error:

        print()
        print("=" * 80)
        print("КРИТИЧЕСКАЯ ОШИБКА")
        print("=" * 80)
        print()
        print(error)
        print()