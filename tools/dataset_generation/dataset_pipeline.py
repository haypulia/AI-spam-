# основной pipeline генерации датасета
# на текущем этапе используется только: deepseek-ai/DeepSeek-V4-Flash

import json
import random
import re
import traceback

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dataset_generation.config import (
    MODELS,
    COSINE_SIMILARITY_THRESHOLD,
    NORMALIZED_LEVENSHTEIN_THRESHOLD,
    OUTPUT_DIR,
    OUTPUT_FILE,
)

from dataset_generation.generator import (
    DeepCodeGenerator,
)

from dataset_generation.prompts import (
    build_prompt,
)

from dataset_generation.similarity import (
    calculate_similarity,
)

from dataset_generation.transformations import (
    TRANSFORMATIONS,
    select_tone,
)

from dataset_generation.email_sources import (
    EMAIL_SOURCES,
)


TARGET_SAMPLES = 10

MAX_PIPELINE_ATTEMPTS = 50

MAX_GENERATION_ATTEMPTS = 2

MIN_FRAGMENT_SENTENCES = 1

MAX_FRAGMENT_SENTENCES = 2

MIN_FRAGMENT_LENGTH = 20

MAX_FRAGMENT_LENGTH = 500


def get_model() -> str:
    # возвращает единственную доступную модель

    if not MODELS:
        raise RuntimeError(
            "MODELS пуст."
        )

    if len(MODELS) != 1:
        raise RuntimeError(
            "Ожидается ровно одна модель."
        )

    return MODELS[0]


def normalize_text(
    text: str,
) -> str:

    if text is None:
        return ""

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    return text.strip()


def detect_language(
    text: str,
) -> str:

    russian_chars = len(
        re.findall(
            r"[А-Яа-яЁё]",
            text,
        )
    )

    english_chars = len(
        re.findall(
            r"[A-Za-z]",
            text,
        )
    )

    if russian_chars > english_chars:
        return "ru"

    if english_chars > russian_chars:
        return "en"

    return "unknown"


def split_sentence_spans(
    text: str,
) -> list[tuple[int, int, str]]:

    text = normalize_text(text)

    if not text:
        return []

    pattern = re.compile(
        r".+?(?:[.!?]+(?=\s|$)|$)",
        flags=re.DOTALL,
    )

    spans = []

    for match in pattern.finditer(text):

        start = match.start()
        end = match.end()

        sentence = text[
            start:end
        ]

        stripped = sentence.strip()

        if not stripped:
            continue

        leading_spaces = (
            len(sentence)
            - len(sentence.lstrip())
        )

        actual_start = (
            start
            + leading_spaces
        )

        actual_end = (
            actual_start
            + len(stripped)
        )

        spans.append(
            (
                actual_start,
                actual_end,
                stripped,
            )
        )

    return spans


def choose_plain_text_fragment(
    email: str,
) -> str:

    email = normalize_text(
        email
    )

    spans = split_sentence_spans(
        email
    )

    if not spans:
        raise ValueError(
            "Не удалось найти предложения."
        )

    candidates = []

    for amount in range(
        MIN_FRAGMENT_SENTENCES,
        MAX_FRAGMENT_SENTENCES + 1,
    ):

        if amount > len(spans):
            continue

        for start_index in range(
            0,
            len(spans) - amount + 1,
        ):

            first_start = spans[
                start_index
            ][0]

            last_end = spans[
                start_index
                + amount
                - 1
            ][1]

            fragment = email[
                first_start:last_end
            ].strip()

            if (
                MIN_FRAGMENT_LENGTH
                <= len(fragment)
                <= MAX_FRAGMENT_LENGTH
            ):

                candidates.append(
                    fragment
                )

    if not candidates:

        candidates = [
            sentence
            for _, _, sentence
            in spans
            if (
                MIN_FRAGMENT_LENGTH
                <= len(sentence)
                <= MAX_FRAGMENT_LENGTH
            )
        ]

    if not candidates:
        raise ValueError(
            "Не найден подходящий фрагмент."
        )

    return random.choice(
        candidates
    )


def choose_html_fragment(
    html: str,
) -> str:

    html = normalize_text(
        html
    )

    pattern = re.compile(
        r"(<(?:p|div|td|li|span|h[1-6])"
        r"\b[^>]*>)"
        r"(.*?)"
        r"(</(?:p|div|td|li|span|h[1-6])>)",
        flags=re.IGNORECASE
        | re.DOTALL,
    )

    candidates = []

    for match in pattern.finditer(
        html
    ):

        inner_text = match.group(2)

        visible_text = re.sub(
            r"<[^>]+>",
            "",
            inner_text,
        )

        visible_text = (
            visible_text.strip()
        )

        if not visible_text:
            continue

        if (
            MIN_FRAGMENT_LENGTH
            <= len(visible_text)
            <= MAX_FRAGMENT_LENGTH
        ):

            candidates.append(
                visible_text
            )

    if not candidates:
        raise ValueError(
            "Не удалось найти подходящий "
            "текстовый фрагмент внутри HTML."
        )

    return random.choice(
        candidates
    )


def choose_fragment(
    email: str,
    email_format: str,
) -> str:

    if email_format == "plain_text":

        return choose_plain_text_fragment(
            email
        )

    if email_format in {
        "html",
        "html_with_images",
        "html_without_images",
    }:

        return choose_html_fragment(
            email
        )

    raise ValueError(
        f"Неизвестный формат письма: "
        f"{email_format}"
    )


def replace_html_fragment(
    html: str,
    original_fragment: str,
    generated_fragment: str,
) -> str:

    if original_fragment in html:

        return html.replace(
            original_fragment,
            generated_fragment,
            1,
        )

    normalized_original = re.sub(
        r"\s+",
        " ",
        original_fragment.strip(),
    )

    pattern = re.compile(
        r">([^<>]+)<",
        flags=re.DOTALL,
    )

    for match in pattern.finditer(
        html
    ):

        visible = match.group(1)

        normalized_visible = re.sub(
            r"\s+",
            " ",
            visible.strip(),
        )

        if (
            normalized_visible
            == normalized_original
        ):

            start = match.start(1)
            end = match.end(1)

            return (
                html[:start]
                + generated_fragment
                + html[end:]
            )

    raise ValueError(
        "Исходный HTML-фрагмент "
        "не найден в письме."
    )


def replace_fragment(
    email: str,
    original_fragment: str,
    generated_fragment: str,
    email_format: str,
) -> str:

    if email_format == "plain_text":

        if original_fragment not in email:

            raise ValueError(
                "Исходный фрагмент "
                "не найден в письме."
            )

        return email.replace(
            original_fragment,
            generated_fragment,
            1,
        )

    return replace_html_fragment(
        email,
        original_fragment,
        generated_fragment,
    )


def choose_transformation() -> str:

    available = list(
        TRANSFORMATIONS.keys()
    )

    if not available:
        raise RuntimeError(
            "TRANSFORMATIONS пуст."
        )

    return random.choice(
        available
    )


def choose_source() -> dict:

    if not EMAIL_SOURCES:
        raise RuntimeError(
            "EMAIL_SOURCES пуст."
        )

    return random.choice(
        EMAIL_SOURCES
    )


def choose_tone_for_transformation(
    transformation: str,
) -> Optional[str]:

    if transformation != "tone_change":
        return None

    return select_tone()


def validate_generated_fragment(
    original_fragment: str,
    generated_fragment: str,
) -> bool:

    if not generated_fragment:
        return False

    generated_fragment = (
        generated_fragment.strip()
    )

    if not generated_fragment:
        return False

    if (
        generated_fragment
        == original_fragment.strip()
    ):

        print()
        print(
            "SAMPLE ОТБРОШЕН."
        )

        print(
            "Причина: "
            "generated_equals_original"
        )

        return False

    return True


def generate_one_sample(
    source: dict,
    model: Optional[str] = None,
    transformation: Optional[str] = None,
) -> Optional[dict]:

    email = normalize_text(
        source["email"]
    )

    language = source.get(
        "language"
    ) or detect_language(
        email
    )

    domain = source.get(
        "domain",
        "unknown",
    )

    email_format = source.get(
        "email_format",
        "plain_text",
    )

    html = (
        email_format != "plain_text"
    )

    if model is None:
        model = get_model()

    try:

        original_fragment = (
            choose_fragment(
                email,
                email_format,
            )
        )

    except Exception as exc:

        print()
        print(
            "Не удалось выбрать фрагмент:"
        )

        print(str(exc))

        return None

    if transformation is None:

        transformation = (
            choose_transformation()
        )

    tone = (
        choose_tone_for_transformation(
            transformation
        )
    )

    print()
    print("=" * 70)

    print(
        "ИСТОЧНИК:"
    )

    print(
        source.get(
            "name",
            "unknown",
        )
    )

    print()
    print(
        "ИСХОДНЫЙ ФРАГМЕНТ:"
    )

    print(
        original_fragment
    )

    print()
    print(
        "TRANSFORMATION:"
    )

    print(
        transformation
    )

    print()
    print(
        "MODEL:"
    )

    print(
        model
    )

    print()
    print(
        "LANGUAGE:"
    )

    print(
        language
    )

    print()
    print(
        "DOMAIN:"
    )

    print(
        domain
    )

    print()
    print(
        "EMAIL FORMAT:"
    )

    print(
        email_format
    )

    prompt = build_prompt(
        text=original_fragment,
        transformation=transformation,
        tone=tone,
    )

    print()
    print(
        "PROMPT:"
    )

    print(
        prompt
    )

    try:

        generator = (
            DeepCodeGenerator(
                model=model
            )
        )

    except Exception as exc:

        print()
        print(
            "Ошибка при создании "
            "DeepCodeGenerator:"
        )

        print(str(exc))

        return None

    generated_fragment = None

    for attempt in range(
        1,
        MAX_GENERATION_ATTEMPTS + 1,
    ):

        print()
        print(
            f"Попытка генерации "
            f"{attempt}/"
            f"{MAX_GENERATION_ATTEMPTS}"
        )

        try:

            generated_fragment = (
                generator.generate(
                    text=original_fragment,
                    transformation=transformation,
                    tone=tone,
                    model=model,
                )
            )

        except Exception as exc:

            print()
            print(
                "Ошибка при обращении "
                "к модели:"
            )

            print(str(exc))

            if (
                attempt
                < MAX_GENERATION_ATTEMPTS
            ):

                print(
                    "Повторяем запрос..."
                )

                continue

            return None

        if generated_fragment:
            break

    if not generated_fragment:

        print(
            "Модель не вернула текст."
        )

        return None

    generated_fragment = (
        normalize_text(
            generated_fragment
        )
    )

    print()
    print(
        "СГЕНЕРИРОВАННЫЙ ФРАГМЕНТ:"
    )

    print(
        generated_fragment
    )

    if not validate_generated_fragment(
        original_fragment,
        generated_fragment,
    ):

        return None

    print()
    print(
        "Считаем метрики сходства..."
    )

    try:

        metrics = calculate_similarity(
            original_fragment,
            generated_fragment,
        )

    except Exception as exc:

        print()
        print(
            "Ошибка при вычислении "
            "similarity:"
        )

        print(str(exc))

        return None

    print()
    print(
        "МЕТРИКИ:"
    )

    print(
        f"Levenshtein: "
        f"{metrics['levenshtein']}"
    )

    print(
        f"Normalized Levenshtein: "
        f"{metrics['normalized_levenshtein']:.4f}"
    )

    print(
        f"Cosine similarity: "
        f"{metrics['cosine_similarity']:.4f}"
    )

    cosine = metrics[
        "cosine_similarity"
    ]

    if (
        cosine
        > COSINE_SIMILARITY_THRESHOLD
    ):

        print()
        print(
            "✗ SAMPLE ОТБРОШЕН."
        )

        print(
            "Причина: "
            "cosine_similarity_too_high"
        )

        print(
            f"{cosine:.4f} > "
            f"{COSINE_SIMILARITY_THRESHOLD}"
        )

        return None

    normalized_levenshtein = (
        metrics[
            "normalized_levenshtein"
        ]
    )

    if (
        normalized_levenshtein
        < NORMALIZED_LEVENSHTEIN_THRESHOLD
    ):

        print()
        print(
            "✗ SAMPLE ОТБРОШЕН."
        )

        print(
            "Причина: "
            "text_changed_too_little"
        )

        print(
            f"{normalized_levenshtein:.4f} < "
            f"{NORMALIZED_LEVENSHTEIN_THRESHOLD}"
        )

        return None

    try:

        generated_email = (
            replace_fragment(
                email=email,
                original_fragment=(
                    original_fragment
                ),
                generated_fragment=(
                    generated_fragment
                ),
                email_format=email_format,
            )
        )

    except Exception as exc:

        print()
        print(
            "Не удалось собрать "
            "итоговое письмо:"
        )

        print(str(exc))

        return None

    if generated_email == email:

        print()
        print(
            "SAMPLE ОТБРОШЕН."
        )

        print(
            "Причина: "
            "email_was_not_changed"
        )

        return None

    sample = {

        "id": None,

        "original_email":
            email,

        "generated_email":
            generated_email,

        "original_fragment":
            original_fragment,

        "generated_fragment":
            generated_fragment,

        "model":
            model,

        "transformation":
            transformation,

        "tone":
            tone,

        "language":
            language,

        "domain":
            domain,

        "email_format":
            email_format,

        "ai_generated":
            True,

        "html":
            html,

        "levenshtein":
            metrics[
                "levenshtein"
            ],

        "normalized_levenshtein":
            metrics[
                "normalized_levenshtein"
            ],

        "cosine_similarity":
            cosine,

        "cosine_threshold":
            COSINE_SIMILARITY_THRESHOLD,

        "normalized_levenshtein_threshold":
            NORMALIZED_LEVENSHTEIN_THRESHOLD,

        "created_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    print()
    print("=" * 70)

    print(
        "✓ SAMPLE УСПЕШНО СОЗДАН."
    )

    print("=" * 70)

    return sample


def save_sample(
    sample: dict,
    output_file: Optional[str | Path] = None,
) -> None:

    if output_file is None:

        output_file = (
            Path(OUTPUT_DIR)
            / OUTPUT_FILE
        )

    output_file = Path(
        output_file
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if sample.get("id") is None:

        try:

            with open(
                output_file,
                "r",
                encoding="utf-8",
            ) as f:

                existing = sum(
                    1
                    for _ in f
                )

        except FileNotFoundError:

            existing = 0

        sample["id"] = (
            f"sample_{existing + 1:06d}"
        )

    with open(
        output_file,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                sample,
                ensure_ascii=False,
            )
        )

        f.write("\n")

    print()
    print("=" * 70)

    print(
        "SAMPLE СОХРАНЁН:"
    )

    print(
        output_file
    )

    print("=" * 70)


def print_dataset_statistics(
    output_file: Path,
) -> None:

    if not output_file.exists():

        print(
            "\nДатасет пока не существует."
        )

        return

    samples = []

    with open(
        output_file,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                samples.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:

                continue

    if not samples:

        print(
            "\nВ датасете нет samples."
        )

        return

    print()
    print("=" * 70)
    print("СТАТИСТИКА DATASET")
    print("=" * 70)

    print()
    print(
        f"Всего samples: "
        f"{len(samples)}"
    )

    models = {}

    for sample in samples:

        value = sample.get(
            "model",
            "unknown",
        )

        models[value] = (
            models.get(value, 0)
            + 1
        )

    print()
    print("МОДЕЛИ:")

    for key, value in sorted(
        models.items()
    ):

        print(
            f"  {key}: {value}"
        )

    transformations = {}

    for sample in samples:

        value = sample.get(
            "transformation",
            "unknown",
        )

        transformations[value] = (
            transformations.get(
                value,
                0,
            )
            + 1
        )

    print()
    print(
        "TRANSFORMATIONS:"
    )

    for key, value in sorted(
        transformations.items()
    ):

        print(
            f"  {key}: {value}"
        )

    languages = {}

    for sample in samples:

        value = sample.get(
            "language",
            "unknown",
        )

        languages[value] = (
            languages.get(
                value,
                0,
            )
            + 1
        )

    print()
    print("ЯЗЫКИ:")

    for key, value in sorted(
        languages.items()
    ):

        print(
            f"  {key}: {value}"
        )

    domains = {}

    for sample in samples:

        value = sample.get(
            "domain",
            "unknown",
        )

        domains[value] = (
            domains.get(
                value,
                0,
            )
            + 1
        )

    print()
    print("ДОМЕНЫ:")

    for key, value in sorted(
        domains.items()
    ):

        print(
            f"  {key}: {value}"
        )

    formats = {}

    for sample in samples:

        value = sample.get(
            "email_format",
            "unknown",
        )

        formats[value] = (
            formats.get(
                value,
                0,
            )
            + 1
        )

    print()
    print("ФОРМАТЫ:")

    for key, value in sorted(
        formats.items()
    ):

        print(
            f"  {key}: {value}"
        )

    print()
    print("=" * 70)


def main():

    print("=" * 70)

    print(
        "ЗАПУСК DATASET PIPELINE"
    )

    print("=" * 70)

    output_file = (
        Path(OUTPUT_DIR)
        / OUTPUT_FILE
    )

    model = get_model()

    print()
    print(
        "ИСПОЛЬЗУЕМАЯ МОДЕЛЬ:"
    )

    print(
        model
    )

    print()
    print(
        "ЦЕЛЬ:"
    )

    print(
        f"{TARGET_SAMPLES} новых samples"
    )

    successful = 0
    attempts = 0

    while (
        successful < TARGET_SAMPLES
        and attempts
        < MAX_PIPELINE_ATTEMPTS
    ):

        attempts += 1

        print()
        print("=" * 70)

        print(
            f"PIPELINE ПОПЫТКА "
            f"{attempts}/"
            f"{MAX_PIPELINE_ATTEMPTS}"
        )

        print(
            f"УСПЕШНО СОЗДАНО: "
            f"{successful}/"
            f"{TARGET_SAMPLES}"
        )

        print("=" * 70)

        try:

            source = choose_source()

            sample = (
                generate_one_sample(
                    source=source,
                    model=model,
                )
            )

        except KeyboardInterrupt:

            print()
            print(
                "Генерация остановлена "
                "пользователем."
            )

            break

        except Exception as exc:

            print()
            print(
                "ОШИБКА PIPELINE:"
            )

            print(
                str(exc)
            )

            traceback.print_exc()

            continue

        if sample is None:

            print()
            print(
                "Sample не создан."
            )

            continue

        try:

            save_sample(
                sample,
                output_file,
            )

            successful += 1

        except Exception as exc:

            print()
            print(
                "ОШИБКА СОХРАНЕНИЯ:"
            )

            print(
                str(exc)
            )

            traceback.print_exc()

    print()
    print("=" * 70)

    print(
        "ГЕНЕРАЦИЯ ЗАВЕРШЕНА"
    )

    print("=" * 70)

    print()
    print(
        f"Новых успешных samples: "
        f"{successful}"
    )

    print(
        f"Попыток pipeline: "
        f"{attempts}"
    )

    print(
        f"Файл: "
        f"{output_file}"
    )

    print_dataset_statistics(
        output_file
    )


if __name__ == "__main__":
    main()