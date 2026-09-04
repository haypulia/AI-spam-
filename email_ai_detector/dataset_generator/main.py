# тестовая генерация одной записи

import json
import random
from pathlib import Path

from dataset_generator.config import (
    SOURCE_EMAILS_FILE,
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


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

SOURCE_EMAILS_PATH = (
    PROJECT_ROOT / SOURCE_EMAILS_FILE
)


def load_source_emails() -> list[dict]:
    # загружает исходные письма из JSONL

    if not SOURCE_EMAILS_PATH.exists():
        raise FileNotFoundError(
            f"Файл не найден:\n"
            f"{SOURCE_EMAILS_PATH}"
        )

    records = []

    with SOURCE_EMAILS_PATH.open(
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

    if not records:
        raise RuntimeError(
            "В source_emails_new.jsonl "
            "нет корректных записей."
        )

    return records


def extract_fragment(
    text: str,
) -> str:
    # выбирает небольшой содержательный фрагмент письма

    text = text.strip()

    if not text:
        raise RuntimeError(
            "Исходное письмо пустое."
        )

    # HTML

    if "<" in text and ">" in text:

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        candidates = []

        for line in lines:

            lowered = line.lower()

            has_text = False

            # проверяет наличие видимого текста

            import re

            visible_text = re.sub(
                r"<[^>]+>",
                "",
                line,
            ).strip()

            if visible_text:
                has_text = True

            if not has_text:
                continue

            if (
                "<p" in lowered
                or "<div" in lowered
                or "<li" in lowered
                or "<td" in lowered
                or "<h1" in lowered
                or "<h2" in lowered
                or "<h3" in lowered
                or "<a" in lowered
            ):
                candidates.append(line)

        if candidates:

            # предпочитаем фрагменты достаточной длины

            good_candidates = [
                candidate
                for candidate in candidates
                if len(candidate.strip()) >= 20
            ]

            if good_candidates:
                return random.choice(
                    good_candidates
                )

            return random.choice(candidates)

        # ищем любую строку с текстом

        text_candidates = []

        for line in lines:

            visible_text = re.sub(
                r"<[^>]+>",
                "",
                line,
            ).strip()

            if visible_text:
                text_candidates.append(line)

        if text_candidates:
            return random.choice(
                text_candidates
            )

        return lines[
            min(2, len(lines) - 1)
        ]

    # plain text

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    # сначала ищем достаточно содержательный абзац

    good_paragraphs = [
        paragraph
        for paragraph in paragraphs
        if len(paragraph) >= 20
    ]

    if good_paragraphs:
        return random.choice(
            good_paragraphs
        )

    if paragraphs:
        return random.choice(
            paragraphs
        )

    # sentences

    sentences = [
        sentence.strip()
        for sentence in text.split(".")
        if sentence.strip()
    ]

    if sentences:

        return sentences[0] + "."

    return text


def validate_fragment(
    fragment: str,
) -> bool:

    if not fragment:
        return False

    if len(fragment.strip()) < 20:
        return False

    return True


def clean_target(
    target: str,
) -> str:
    # очищает ответ модели от Markdown fences и лишних пробелов

    if not isinstance(target, str):
        return ""

    target = target.strip()

    # Markdown HTML fence

    if target.startswith("```html"):

        target = target[
            len("```html"):
        ].strip()

        if target.endswith("```"):
            target = target[:-3].strip()

    # generic Markdown fence

    elif target.startswith("```"):

        target = target[3:].strip()

        if target.endswith("```"):
            target = target[:-3].strip()

    return target


def generate_one():

    print()
    print("=" * 70)
    print("ТЕСТ ГЕНЕРАЦИИ ОДНОЙ ПАРЫ")
    print("=" * 70)
    print()

    # load

    records = load_source_emails()

    print(
        f"Исходных писем найдено: "
        f"{len(records)}"
    )

    # select

    record = random.choice(
        records
    )

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
    )

    text = record.get(
        "text",
        "",
    )

    print(
        f"Выбрано письмо ID: "
        f"{record.get('id')}"
    )

    print(
        f"Язык: {language}"
    )

    print(
        f"Домен: {domain}"
    )

    print(
        f"Формат: {email_format}"
    )

    print(
        f"Модель исходного письма: "
        f"{model}"
    )

    print()

    # fragment

    source = extract_fragment(
        text
    )

    if not validate_fragment(
        source
    ):
        raise RuntimeError(
            "Не удалось получить "
            "достаточно длинный фрагмент."
        )

    print("=" * 70)
    print("SOURCE")
    print("=" * 70)
    print()
    print(source)
    print()

    # transformation

    transformation = random.choice(
        TRANSFORMATIONS
    )

    print("=" * 70)
    print("TRANSFORMATION")
    print("=" * 70)
    print()

    print(
        transformation
    )

    print()

    # prompt

    prompt = build_transformation_prompt(
        text=source,
        transformation=transformation,
        language=language,
        email_format=email_format,
    )

    # generator

    generator = DeepCodeGenerator(
        model=model
    )

    # generation with retries

    MAX_GENERATION_ATTEMPTS = 5

    target = ""

    for attempt in range(
        1,
        MAX_GENERATION_ATTEMPTS + 1,
    ):

        print("=" * 70)
        print(
            f"ГЕНЕРАЦИЯ TARGET "
            f"(попытка {attempt}/{MAX_GENERATION_ATTEMPTS})"
        )
        print("=" * 70)
        print()

        try:

            target = generator.generate(
                prompt=prompt
            )

        except Exception as error:

            print(
                f"Ошибка генерации: {error}"
            )

            target = ""

        target = clean_target(
            target
        )

        if not target:

            print(
                "DeepCode вернул пустой target."
            )

            if attempt < MAX_GENERATION_ATTEMPTS:
                print(
                    "Повторяем запрос..."
                )
                print()

                continue

            raise RuntimeError(
                "Не удалось получить "
                "непустой target."
            )

        break

    # target

    print("=" * 70)
    print("TARGET")
    print("=" * 70)
    print()

    print(target)
    print()

    # difference

    print("=" * 70)
    print("ПРОВЕРКА ПОХОЖЕСТИ")
    print("=" * 70)
    print()

    print(
        "Вычисляем similarity..."
    )

    similarity_calculator = (
        SimilarityCalculator(
            EMBEDDING_MODEL_NAME
        )
    )

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

    print()

    print(
        "Normalized Levenshtein distance: "
        f"{metrics['normalized_levenshtein_distance']:.4f}"
    )

    print(
        "Cosine similarity: "
        f"{metrics['cosine_similarity']:.4f}"
    )

    print()

    # result

    if is_valid:

        print(
            "✓ ПАРА ПРОШЛА ПРОВЕРКУ"
        )

    else:

        print(
            "✗ ПАРА НЕ ПРОШЛА ПРОВЕРКУ"
        )

        print(
            f"Причина: "
            f"{metrics['reason']}"
        )

    print()

    # JSON

    result = {
        "id": "test_000001",
        "language": language,
        "domain": domain,
        "format": email_format,
        "model": model,
        "transformation": transformation,
        "source": source,
        "target": target,
        "cosine_similarity": (
            metrics[
                "cosine_similarity"
            ]
        ),
        "normalized_levenshtein_distance": (
            metrics[
                "normalized_levenshtein_distance"
            ]
        ),
        "valid": is_valid,
    }

    print("=" * 70)
    print("ИТОГОВАЯ ЗАПИСЬ")
    print("=" * 70)
    print()

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()


if __name__ == "__main__":

    try:

        generate_one()

    except KeyboardInterrupt:

        print()
        print(
            "Остановлено пользователем."
        )

    except Exception as error:

        print()
        print("=" * 70)
        print("ОШИБКА")
        print("=" * 70)
        print()
        print(error)
        print()