# проверка сходства исходного и сгенерированного текста
# используются:
# 1. нормализованное расстояние Левенштейна
# 2. cosine similarity

from __future__ import annotations

import re

from typing import Optional
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_COSINE_THRESHOLD = 0.98

DEFAULT_LEVENSHTEIN_THRESHOLD = 0.20

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


_embedding_model: Optional[SentenceTransformer] = None


def normalize_text(text: str) -> str:
    # нормализует текст перед сравнением

    if not isinstance(text, str):

        text = str(text)

    # нормализуем переносы строк

    text = text.replace("\r\n", "\n")

    text = text.replace("\r", "\n")

    # последовательности пробелов и табов заменяются одним пробелом

    text = re.sub(r"[ \t]+", " ", text)

    # убираем пробелы в начале и конце строк

    text = re.sub(r" *\n *", "\n", text)

    # оставляем максимум две последовательные пустые строки

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalized_levenshtein(
    source: str,
    target: str,
) -> float:
    # вычисляет нормализованное расстояние Левенштейна

    source = normalize_text(source)

    target = normalize_text(target)

    if not source and not target:

        return 0.0

    if not source or not target:

        return 1.0

    distance = Levenshtein.distance(
        source,
        target,
    )

    max_length = max(
        len(source),
        len(target),
    )

    return distance / max_length


def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    # загружает embedding model один раз

    global _embedding_model

    if _embedding_model is None:

        print()

        print("Загружаем embedding model...")

        print(
            f"Embedding model: {model_name}"
        )

        _embedding_model = SentenceTransformer(
            model_name
        )

    return _embedding_model


def cosine_text_similarity(
    source: str,
    target: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> float:
    # вычисляет cosine similarity между source и target

    source = normalize_text(source)

    target = normalize_text(target)

    if not source or not target:

        return 0.0

    model = get_embedding_model(
        model_name=model_name
    )

    embeddings = model.encode(
        [
            source,
            target,
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    similarity = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1),
    )[0][0]

    return float(similarity)


class SimilarityCalculator:
    # обёртка над embedding model

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):

        self.model_name = model_name

    def cosine_similarity(
        self,
        source: str,
        target: str,
    ) -> float:
        # возвращает cosine similarity

        return cosine_text_similarity(
            source,
            target,
            model_name=self.model_name,
        )


def is_sufficiently_different(
    source: str,
    target: str,
    cosine_threshold: float,
    levenshtein_threshold: float,
    similarity_calculator: SimilarityCalculator,
) -> tuple[bool, dict]:
    # проверяет, достаточно ли отличаются source и target
    # пара принимается, если cosine similarity не выше порога
    # и normalized levenshtein distance не ниже порога

    # проверяем типы входных данных

    if not isinstance(source, str):

        raise TypeError(
            "source должен быть строкой."
        )

    if not isinstance(target, str):

        raise TypeError(
            "target должен быть строкой."
        )

    # проверяем пустой source

    if not source.strip():

        return (
            False,
            {
                "cosine_similarity": 1.0,
                "normalized_levenshtein_distance": 0.0,
                "valid": False,
                "reason": "empty source",
            },
        )

    # проверяем пустой target

    if not target.strip():

        return (
            False,
            {
                "cosine_similarity": 1.0,
                "normalized_levenshtein_distance": 0.0,
                "valid": False,
                "reason": "empty target",
            },
        )

    # проверяем полное совпадение текстов

    if normalize_text(source) == normalize_text(target):

        return (
            False,
            {
                "cosine_similarity": 1.0,
                "normalized_levenshtein_distance": 0.0,
                "valid": False,
                "reason": "identical source and target",
            },
        )

    levenshtein_distance = normalized_levenshtein(
        source,
        target,
    )

    cosine_similarity_value = (
        similarity_calculator.cosine_similarity(
            source,
            target,
        )
    )

    cosine_ok = (
        cosine_similarity_value
        <= cosine_threshold
    )

    levenshtein_ok = (
        levenshtein_distance
        >= levenshtein_threshold
    )

    valid = (
        cosine_ok
        and levenshtein_ok
    )

    if valid:

        reason = "accepted"

    elif not cosine_ok and not levenshtein_ok:

        reason = (
            "cosine too high and "
            "levenshtein too low"
        )

    elif not cosine_ok:

        reason = (
            "cosine similarity too high"
        )

    else:

        reason = (
            "levenshtein distance too low"
        )

    return (
        valid,
        {
            "cosine_similarity": (
                cosine_similarity_value
            ),
            "normalized_levenshtein_distance": (
                levenshtein_distance
            ),
            "valid": valid,
            "reason": reason,
        },
    )
