# массовая генерация датасета:

import json
import random
import re
from pathlib import Path
from collections import Counter

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


MAX_ATTEMPTS_PER_PAIR = 5

RANDOM_SEED = 42

# мин длина target без html-тегов
MIN_TARGET_TEXT_LENGTH = 20

# макс длина target без html-тегов
MAX_TARGET_TEXT_LENGTH = 5000

# макс количество использований одного source
MAX_SOURCE_USES = 2

# целевой размер датасета
TARGET_DATASET_SIZE = 500

# макс количество попыток
MAX_TOTAL_ATTEMPTS = 1500

# добавлять новые записи в существующий датасет
APPEND_TO_EXISTING = False

# имя нового датасета
OUTPUT_FILE = "dataset_v2.jsonl"


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
    OUTPUT_DIR_PATH / OUTPUT_FILE
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

            if not isinstance(record, dict):
                continue

            text = record.get("text", "")

            if not isinstance(text, str):
                continue

            if not text.strip():
                continue

            records.append(record)

    if not records:
        raise RuntimeError(
            "В source_emails_new.jsonl "
            "нет корректных записей."
        )

    return records


def strip_html_tags(text: str) -> str:
    # удаляет html-теги и возвращает только текст

    if not isinstance(text, str):
        return ""

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = re.sub(
        r"&nbsp;",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"&amp;",
        "&",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"&lt;",
        "<",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"&gt;",
        ">",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def contains_html_tags(text: str) -> bool:
    # проверяет наличие html-тегов

    if not isinstance(text, str):
        return False

    return bool(
        re.search(
            r"<[a-zA-Z][^>]*>",
            text,
        )
    )


def count_html_tags(text: str) -> int:
    # считает количество html-тегов

    if not isinstance(text, str):
        return 0

    return len(
        re.findall(
            r"<[a-zA-Z][^>]*>",
            text,
        )
    )


def extract_html_fragments(
    text: str,
) -> list[str]:
    # извлекает содержательные html-фрагменты

    if not isinstance(text, str):
        return []

    text = text.strip()

    if not text:
        return []

    candidates = []

    block_pattern = re.compile(
        r"<(?:p|div|td|th|li|h[1-6]|section|article|"
        r"span|blockquote)[^>]*>.*?</(?:p|div|td|th|li|"
        r"h[1-6]|section|article|span|blockquote)>",
        re.IGNORECASE | re.DOTALL,
    )

    blocks = block_pattern.findall(text)

    for block in blocks:

        plain_text = strip_html_tags(block)

        if (
            MIN_TARGET_TEXT_LENGTH
            <= len(plain_text)
            <= MAX_TARGET_TEXT_LENGTH
        ):
            candidates.append(
                block.strip()
            )

    if not candidates:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        for line in lines:

            plain_text = strip_html_tags(line)

            if len(plain_text) < MIN_TARGET_TEXT_LENGTH:
                continue

            if len(plain_text) > MAX_TARGET_TEXT_LENGTH:
                continue

            if not contains_html_tags(line):
                continue

            candidates.append(line)

    unique = []

    seen = set()

    for candidate in candidates:

        normalized = re.sub(
            r"\s+",
            " ",
            candidate,
        ).strip()

        if normalized in seen:
            continue

        seen.add(normalized)
        unique.append(candidate)

    return unique


def split_into_sentences(
    text: str,
) -> list[str]:
    # разделяет текст на предложения

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def extract_plain_text_fragments(
    text: str,
) -> list[str]:
    # извлекает содержательные фрагменты обычного текста

    if not isinstance(text, str):
        return []

    text = text.strip()

    if not text:
        return []

    candidates = []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            text,
        )
        if paragraph.strip()
    ]

    for paragraph in paragraphs:

        if (
            MIN_TARGET_TEXT_LENGTH
            <= len(paragraph)
            <= MAX_TARGET_TEXT_LENGTH
        ):
            candidates.append(
                paragraph
            )

    sentences = split_into_sentences(text)

    for size in (1, 2, 3):

        for index in range(
            0,
            len(sentences) - size + 1,
        ):

            fragment = " ".join(
                sentences[
                    index:index + size
                ]
            ).strip()

            if (
                MIN_TARGET_TEXT_LENGTH
                <= len(fragment)
                <= MAX_TARGET_TEXT_LENGTH
            ):
                candidates.append(
                    fragment
                )

    if not candidates:

        if (
            MIN_TARGET_TEXT_LENGTH
            <= len(text)
            <= MAX_TARGET_TEXT_LENGTH
        ):
            candidates.append(text)

    unique = []

    seen = set()

    for candidate in candidates:

        normalized = re.sub(
            r"\s+",
            " ",
            candidate,
        ).strip()

        if normalized in seen:
            continue

        seen.add(normalized)
        unique.append(candidate)

    return unique


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


def clean_target(
    target: str,
) -> str:
    # очищает ответ модели

    if not isinstance(target, str):
        return ""

    target = target.strip()

    target = re.sub(
        r"^```html\s*",
        "",
        target,
        flags=re.IGNORECASE,
    )

    target = re.sub(
        r"^```\s*",
        "",
        target,
    )

    target = re.sub(
        r"\s*```\s*$",
        "",
        target,
    )

    target = target.strip()

    prefixes = [
        "Вот изменённый текст:",
        "Вот измененный текст:",
        "Изменённый вариант:",
        "Измененный вариант:",
        "Here is the revised text:",
        "Here is the modified text:",
    ]

    for prefix in prefixes:

        if target.lower().startswith(
            prefix.lower()
        ):

            target = target[
                len(prefix):
            ].strip()

    return target.strip()


def validate_target_length(
    target: str,
    email_format: str,
) -> tuple[bool, str]:
    # проверяет длину target без html-тегов

    if not target.strip():
        return False, "empty target"

    if email_format == "html":
        plain_text = strip_html_tags(target)
    else:
        plain_text = target

    plain_text = re.sub(
        r"\s+",
        " ",
        plain_text,
    ).strip()

    length = len(plain_text)

    if length < MIN_TARGET_TEXT_LENGTH:
        return (
            False,
            f"target too short ({length} chars)",
        )

    if length > MAX_TARGET_TEXT_LENGTH:
        return (
            False,
            f"target too long ({length} chars)",
        )

    return True, "ok"


def validate_html_target(
    source: str,
    target: str,
) -> tuple[bool, str]:
    # проверяет сохранение html-структуры

    if not contains_html_tags(target):
        return (
            False,
            "HTML target lost HTML tags",
        )

    source_tag_count = count_html_tags(
        source
    )

    target_tag_count = count_html_tags(
        target
    )

    if target_tag_count == 0:
        return (
            False,
            "HTML target has zero tags",
        )

    source_tags = re.findall(
        r"<([a-zA-Z][a-zA-Z0-9]*)\b",
        source,
    )

    target_tags = re.findall(
        r"<([a-zA-Z][a-zA-Z0-9]*)\b",
        target,
    )

    source_tags = {
        tag.lower()
        for tag in source_tags
    }

    target_tags = {
        tag.lower()
        for tag in target_tags
    }

    structural_tags = {
        "p",
        "div",
        "td",
        "th",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "span",
    }

    source_structural = (
        source_tags
        & structural_tags
    )

    target_structural = (
        target_tags
        & structural_tags
    )

    if source_structural and not target_structural:
        return (
            False,
            "HTML structural tags were lost",
        )

    if (
        source_tag_count <= 4
        and target_tag_count > 20
    ):
        return (
            False,
            "target introduced too many HTML tags",
        )

    return True, "ok"


def validate_target(
    source: str,
    target: str,
    email_format: str,
) -> tuple[bool, str]:
    # выполняет базовую проверку target

    if not isinstance(target, str):
        return False, "target is not a string"

    if not target.strip():
        return False, "empty target"

    if target.strip() == source.strip():
        return False, "target identical to source"

    if "```" in target:
        return False, "target contains markdown fence"

    length_ok, length_reason = (
        validate_target_length(
            target,
            email_format,
        )
    )

    if not length_ok:
        return False, length_reason

    if email_format == "html":

        html_ok, html_reason = (
            validate_html_target(
                source,
                target,
            )
        )

        if not html_ok:
            return False, html_reason

    else:

        if contains_html_tags(target):
            return (
                False,
                "plain_text target contains HTML",
            )

    return True, "ok"


def build_source_pool(
    records: list[dict],
) -> list[dict]:
    # создаёт пул исходных фрагментов

    pool = []

    seen = set()

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

            normalized = re.sub(
                r"\s+",
                " ",
                fragment,
            ).strip()

            key = (
                record.get("id"),
                normalized,
            )

            if key in seen:
                continue

            seen.add(key)

            pool.append(
                {
                    "record": record,
                    "fragment": fragment,
                }
            )

    return pool


def select_source(
    source_pool: list[dict],
    source_usage: Counter,
) -> dict | None:
    # выбирает source с минимальным количеством использований

    available = [
        item
        for item in source_pool
        if source_usage[
            item["fragment"]
        ] < MAX_SOURCE_USES
    ]

    if not available:
        return None

    min_usage = min(
        source_usage[
            item["fragment"]
        ]
        for item in available
    )

    least_used = [
        item
        for item in available
        if source_usage[
            item["fragment"]
        ] == min_usage
    ]

    return random.choice(
        least_used
    )


def select_transformation(
    transformation_usage: Counter,
) -> str:
    # выбирает transformation с минимальным количеством примеров

    min_usage = min(
        transformation_usage[
            transformation
        ]
        for transformation in TRANSFORMATIONS
    )

    candidates = [
        transformation
        for transformation in TRANSFORMATIONS
        if transformation_usage[
            transformation
        ] == min_usage
    ]

    return random.choice(
        candidates
    )


def generate_pair(
    source_item: dict,
    transformation: str,
    similarity_calculator: SimilarityCalculator,
    generator_cache: dict,
) -> dict | None:
    # генерирует одну валидную source -> target пару

    record = source_item["record"]

    source = source_item["fragment"]

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

    model = record.get(
        "model",
        "deepseek-ai/DeepSeek-V4-Flash",
    )

    print()
    print("-" * 80)

    print(
        f"SOURCE ID: {record.get('id')}"
    )

    print(
        f"Language: {language}"
    )

    print(
        f"Domain: {domain}"
    )

    print(
        f"Format: {email_format}"
    )

    print(
        f"Transformation: {transformation}"
    )

    print()
    print("SOURCE:")
    print(source)

    prompt = build_transformation_prompt(
        text=source,
        transformation=transformation,
        language=language,
        email_format=email_format,
    )

    if model not in generator_cache:

        generator_cache[model] = (
            DeepCodeGenerator(
                model=model
            )
        )

    generator = generator_cache[
        model
    ]

    for attempt in range(
        1,
        MAX_ATTEMPTS_PER_PAIR + 1,
    ):

        print()
        print(
            f"ГЕНЕРАЦИЯ TARGET "
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

        target_ok, target_reason = (
            validate_target(
                source=source,
                target=target,
                email_format=email_format,
            )
        )

        if not target_ok:

            print()
            print(
                f"⚠ TARGET отклонён: "
                f"{target_reason}"
            )

            continue

        print()
        print("TARGET:")
        print(target)

        print()
        print(
            "Проверяем similarity..."
        )

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

        cosine_value = metrics[
            "cosine_similarity"
        ]

        levenshtein_value = metrics[
            "normalized_levenshtein_distance"
        ]

        print(
            "Normalized Levenshtein: "
            f"{levenshtein_value:.4f}"
        )

        print(
            "Cosine similarity: "
            f"{cosine_value:.4f}"
        )

        if is_valid:

            print()
            print(
                "✓ ПАРА ПРОШЛА ПРОВЕРКУ"
            )

            return {
                "language": language,
                "domain": domain,
                "format": email_format,
                "model": model,
                "transformation": transformation,
                "source": source,
                "target": target,
                "cosine_similarity": cosine_value,
                "normalized_levenshtein_distance": (
                    levenshtein_value
                ),
                "valid": True,
            }

        print()
        print(
            "✗ Пара не прошла similarity-фильтр."
        )

        print(
            f"Причина: "
            f"{metrics['reason']}"
        )

    return None


def load_existing_records() -> list[dict]:
    # загружает существующий dataset_v2.jsonl

    if not OUTPUT_FILE_PATH.exists():
        return []

    records = []

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

            if isinstance(record, dict):
                records.append(record)

    return records


def save_record(
    record: dict,
) -> None:
    # добавляет запись в dataset_v2.jsonl

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


def get_next_id(
    existing_records: list[dict],
) -> int:
    # определяет следующий numeric ID

    max_id = 0

    for record in existing_records:

        record_id = str(
            record.get(
                "id",
                "",
            )
        )

        match = re.search(
            r"(\d+)$",
            record_id,
        )

        if not match:
            continue

        number = int(
            match.group(1)
        )

        max_id = max(
            max_id,
            number,
        )

    return max_id + 1


def build_usage_statistics(
    existing_records: list[dict],
) -> tuple[Counter, Counter]:
    # строит статистику source и transformations

    source_usage = Counter()

    transformation_usage = Counter()

    for record in existing_records:

        source = record.get(
            "source",
            "",
        )

        transformation = record.get(
            "transformation",
            "",
        )

        if source:
            source_usage[
                source
            ] += 1

        if transformation:
            transformation_usage[
                transformation
            ] += 1

    return (
        source_usage,
        transformation_usage,
    )


def print_generation_statistics(
    generated_records: list[dict],
) -> None:
    # выводит статистику созданных записей

    if not generated_records:
        return

    print()
    print("=" * 80)
    print("СТАТИСТИКА НОВОЙ ГЕНЕРАЦИИ")
    print("=" * 80)

    languages = Counter(
        record.get(
            "language",
            "unknown",
        )
        for record in generated_records
    )

    formats = Counter(
        record.get(
            "format",
            "unknown",
        )
        for record in generated_records
    )

    domains = Counter(
        record.get(
            "domain",
            "unknown",
        )
        for record in generated_records
    )

    transformations = Counter(
        record.get(
            "transformation",
            "unknown",
        )
        for record in generated_records
    )

    models = Counter(
        record.get(
            "model",
            "unknown",
        )
        for record in generated_records
    )

    print()
    print("ЯЗЫКИ:")

    for key, value in sorted(
        languages.items()
    ):
        print(
            f"  {key:<20} {value}"
        )

    print()
    print("ФОРМАТЫ:")

    for key, value in sorted(
        formats.items()
    ):
        print(
            f"  {key:<20} {value}"
        )

    print()
    print("ДОМЕНЫ:")

    for key, value in sorted(
        domains.items()
    ):
        print(
            f"  {key:<20} {value}"
        )

    print()
    print("TRANSFORMATIONS:")

    for key, value in sorted(
        transformations.items()
    ):
        print(
            f"  {key:<25} {value}"
        )

    print()
    print("МОДЕЛИ:")

    for key, value in sorted(
        models.items()
    ):
        print(
            f"  {key:<45} {value}"
        )


def generate_dataset():
    # основная функция генерации

    print()
    print("=" * 80)
    print("МАССОВАЯ ГЕНЕРАЦИЯ DATASET V2")
    print("=" * 80)
    print()

    random.seed(
        RANDOM_SEED
    )

    records = load_source_emails()

    print(
        f"Исходных писем найдено: "
        f"{len(records)}"
    )

    source_pool = build_source_pool(
        records
    )

    if not source_pool:
        raise RuntimeError(
            "Не удалось извлечь ни одного "
            "подходящего source-фрагмента."
        )

    print(
        f"Доступно source-фрагментов: "
        f"{len(source_pool)}"
    )

    existing_records = []

    if APPEND_TO_EXISTING:

        existing_records = (
            load_existing_records()
        )

    else:

        if OUTPUT_FILE_PATH.exists():

            print()
            print(
                "⚠ Существующий dataset_v2.jsonl "
                "будет перезаписан."
            )

            OUTPUT_FILE_PATH.unlink()

    existing_count = len(
        existing_records
    )

    print(
        f"Существующих записей: "
        f"{existing_count}"
    )

    records_needed = max(
        0,
        TARGET_DATASET_SIZE
        - existing_count,
    )

    print(
        f"Нужно создать новых записей: "
        f"{records_needed}"
    )

    if records_needed == 0:

        print()
        print(
            "Целевой размер датасета уже достигнут."
        )

        return

    (
        source_usage,
        transformation_usage,
    ) = build_usage_statistics(
        existing_records
    )

    print()
    print(
        "Инициализируем similarity calculator..."
    )

    similarity_calculator = (
        SimilarityCalculator(
            EMBEDDING_MODEL_NAME
        )
    )

    generator_cache = {}

    generated_records = []

    failed_attempts = 0

    total_attempts = 0

    next_id = get_next_id(
        existing_records
    )

    while (
        len(generated_records)
        < records_needed
        and total_attempts
        < MAX_TOTAL_ATTEMPTS
    ):

        total_attempts += 1

        print()
        print("=" * 80)

        print(
            f"ПРОГРЕСС: "
            f"{len(generated_records)}/"
            f"{records_needed}"
        )

        print(
            f"Попытка: "
            f"{total_attempts}/"
            f"{MAX_TOTAL_ATTEMPTS}"
        )

        print("=" * 80)

        source_item = select_source(
            source_pool=source_pool,
            source_usage=source_usage,
        )

        if source_item is None:

            print()
            print(
                "⚠ Больше нет доступных "
                "source-фрагментов."
            )

            break

        transformation = (
            select_transformation(
                transformation_usage
            )
        )

        result = generate_pair(
            source_item=source_item,
            transformation=transformation,
            similarity_calculator=(
                similarity_calculator
            ),
            generator_cache=generator_cache,
        )

        if result is None:

            failed_attempts += 1

            print()
            print(
                "✗ Пара не создана."
            )

            continue

        result["id"] = (
            f"generated_v2_"
            f"{next_id:06d}"
        )

        next_id += 1

        save_record(
            result
        )

        generated_records.append(
            result
        )

        source_usage[
            source_item["fragment"]
        ] += 1

        transformation_usage[
            transformation
        ] += 1

        print()
        print(
            f"✓ ЗАПИСЬ СОХРАНЕНА: "
            f"{result['id']}"
        )

        print(
            f"Прогресс: "
            f"{len(generated_records)}/"
            f"{records_needed}"
        )

    print()
    print("=" * 80)
    print("ГЕНЕРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80)
    print()

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
        f"{len(generated_records)}"
    )

    print(
        f"Неудачных попыток: "
        f"{failed_attempts}"
    )

    print(
        f"Всего попыток: "
        f"{total_attempts}"
    )

    print(
        f"Всего записей в dataset_v2: "
        f"{existing_count + len(generated_records)}"
    )

    print()
    print(
        f"Файл результата:"
    )

    print(
        OUTPUT_FILE_PATH
    )

    print_generation_statistics(
        generated_records
    )

    if len(generated_records) < records_needed:

        print()
        print("=" * 80)
        print("⚠ ЦЕЛЕВОЙ РАЗМЕР НЕ ДОСТИГНУТ")
        print("=" * 80)
        print()

        print(
            f"Хотели создать: "
            f"{records_needed}"
        )

        print(
            f"Создали: "
            f"{len(generated_records)}"
        )

        print(
            "Возможная причина:"
        )

        print(
            "- слишком строгий similarity-фильтр;"
        )

        print(
            "- слишком мало подходящих source-фрагментов;"
        )

        print(
            "- модель часто генерирует "
            "слишком похожие ответы;"
        )

        print(
            "- модель нарушает HTML-структуру;"
        )

        print(
            "- закончились доступные source-фрагменты."
        )

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

        print(
            error
        )

        print()