# сборка итогового письма после генерации

from __future__ import annotations

from datetime import datetime
from typing import Any


def normalize_spaces(
    text: str,
) -> str:
    # нормализация используется только для анализа
    if not text:
        return ""

    import re

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def find_fragment_position(
    email: str,
    fragment: str,
) -> tuple[int, int]:
    if not email:
        raise ValueError(
            "Исходное письмо пустое."
        )

    if not fragment:
        raise ValueError(
            "Исходный фрагмент пуст."
        )

    # сначала ищем точное совпадение
    start = email.find(
        fragment
    )

    if start != -1:
        return (
            start,
            start + len(fragment),
        )

    # затем ищем с учётом различий в пробелах
    import re

    fragment_parts = re.split(
        r"\s+",
        fragment.strip(),
    )

    if not fragment_parts:
        raise ValueError(
            "Не удалось разбить фрагмент."
        )

    pattern = r"\s+".join(
        re.escape(part)
        for part in fragment_parts
        if part
    )

    if not pattern:
        raise ValueError(
            "Пустой шаблон фрагмента."
        )

    match = re.search(
        pattern,
        email,
        flags=re.DOTALL,
    )

    if match:
        return (
            match.start(),
            match.end(),
        )

    raise ValueError(
        "Исходный фрагмент не найден "
        "в письме.\n\n"
        f"Фрагмент:\n{fragment!r}\n\n"
        f"Исходное письмо:\n{email!r}"
    )


def replace_fragment(
    email: str,
    original_fragment: str,
    generated_fragment: str,
) -> tuple[
    str,
    int,
    int,
]:
    if not email:
        raise ValueError(
            "email пустой."
        )

    if not original_fragment:
        raise ValueError(
            "original_fragment пуст."
        )

    if generated_fragment is None:
        raise ValueError(
            "generated_fragment равен None."
        )

    generated_fragment = (
        generated_fragment.strip()
    )

    if not generated_fragment:
        raise ValueError(
            "generated_fragment пуст."
        )

    # находим исходный фрагмент
    start, end = find_fragment_position(
        email,
        original_fragment,
    )

    # заменяем только найденный фрагмент
    generated_email = (
        email[:start]
        + generated_fragment
        + email[end:]
    )

    return (
        generated_email,
        start,
        end,
    )


def build_sample(
    original_email: str,
    original_fragment: str,
    generated_fragment: str,
    transformation: str,
    model: str,
    language: str = "ru",
    domain: str = "corporate",
    similarity: dict[str, Any] | None = None,
    *,
    prompt: str | None = None,
    sample_id: str | None = None,
) -> dict[str, Any]:
    if similarity is None:
        similarity = {}

    # заменяем исходный фрагмент
    (
        generated_email,
        start_position,
        end_position,
    ) = replace_fragment(
        email=original_email,
        original_fragment=original_fragment,
        generated_fragment=generated_fragment,
    )

    # проверяем результат
    if generated_email == original_email:
        raise ValueError(
            "Сгенерированное письмо полностью "
            "совпадает с исходным."
        )

    # проверяем наличие сгенерированного фрагмента
    generated_position = (
        generated_email.find(
            generated_fragment
        )
    )

    if generated_position == -1:
        raise ValueError(
            "Сгенерированный фрагмент "
            "не найден в итоговом письме."
        )

    # создаём id
    if sample_id is None:
        sample_id = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )

    sample = {
        "id": sample_id,
        "original_email": original_email,
        "generated_email": generated_email,
        "original_fragment": (
            original_fragment
        ),
        "generated_fragment": (
            generated_fragment
        ),
        "fragment_start": (
            start_position
        ),
        "fragment_end": (
            start_position
            + len(generated_fragment)
        ),
        "transformation": (
            transformation
        ),
        "model": model,
        "language": language,
        "domain": domain,
        "similarity": similarity,
        "levenshtein": similarity.get(
            "levenshtein"
        ),
        "normalized_levenshtein": (
            similarity.get(
                "normalized_levenshtein"
            )
        ),
        "cosine_similarity": (
            similarity.get(
                "cosine_similarity"
            )
        ),
        "prompt": prompt,
        "label": 1,
        "has_generated_fragment": True,
        "generated_fragment_start": (
            start_position
        ),
        "generated_fragment_end": (
            start_position
            + len(generated_fragment)
        ),
        "created_at": (
            datetime.now().isoformat()
        ),
    }

    return sample