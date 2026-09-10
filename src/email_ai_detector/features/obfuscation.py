import re
from typing import Dict, Iterator, Tuple

from .normalize import INVISIBLE_PATTERN

OBFUSCATION_FEATURE_NAMES: Tuple[str, ...] = (
    "obfuscation_invisible_rate",
    "obfuscation_intraword_invisible",
    "obfuscation_mixed_script_rate",
    "obfuscation_inner_capital_rate",
)

WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\S+@\S+")
CYRILLIC_PATTERN = re.compile(r"[Ѐ-ӿ]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
INTRAWORD_INVISIBLE_PATTERN = re.compile(
    r"(?<=[^\W\d_])%s(?=[^\W\d_])" % INVISIBLE_PATTERN.pattern, re.UNICODE
)

TOKEN_TRIM = "\"'«»„“”().,;:!?[]{}<>-–—…\\/|*_"
CONFUSABLE_CAPITALS = frozenset("IOІОЅЕАСМНТ")
BRAND_EXCEPTIONS = frozenset(
    (
        "linkedin",
        "directindustry",
        "ebay",
        "paypal",
        "iphone",
        "ipad",
        "ios",
        "macos",
        "youtube",
        "whatsapp",
        "tiktok",
        "sberid",
        "vkid",
    )
)
MIN_CONFUSABLE_WORD = 4
MIN_LOWERCASE_LETTERS = 3


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def prose_words(text: str) -> Iterator[str]:
    cleaned = EMAIL_PATTERN.sub(" ", URL_PATTERN.sub(" ", text or ""))
    for token in cleaned.split():
        token = token.strip(TOKEN_TRIM)
        if token and token.isalpha():
            yield token


def is_confusable_word(word: str) -> bool:
    if len(word) < MIN_CONFUSABLE_WORD:
        return False

    if word.lower() in BRAND_EXCEPTIONS:
        return False

    inner = [(index, char) for index, char in enumerate(word) if char.isupper() and index > 0]
    if len(inner) != 1 or inner[0][1] not in CONFUSABLE_CAPITALS:
        return False

    return sum(1 for char in word if char.islower()) >= MIN_LOWERCASE_LETTERS


def is_mixed_script_word(word: str) -> bool:
    return bool(CYRILLIC_PATTERN.search(word) and LATIN_PATTERN.search(word))


def mixed_script_words(text: str) -> int:
    return sum(1 for word in prose_words(text) if is_mixed_script_word(word))


def inner_capital_words(text: str) -> int:
    return sum(1 for word in prose_words(text) if is_confusable_word(word))


def _source_text(text: str, subject: str, html: str) -> str:
    source = "\n".join(part for part in (subject or "", text or "") if part).strip()
    return source or (html or "")


def extract_obfuscation_features(text: str = "", subject: str = "", html: str = "") -> Dict[str, float]:
    source = _source_text(text, subject, html)
    word_count = len(WORD_PATTERN.findall(source))

    return {
        "obfuscation_invisible_rate": _safe_div(len(INVISIBLE_PATTERN.findall(source)), len(source)),
        "obfuscation_intraword_invisible": _safe_div(
            len(INTRAWORD_INVISIBLE_PATTERN.findall(source)), word_count
        ),
        "obfuscation_mixed_script_rate": _safe_div(mixed_script_words(source), word_count),
        "obfuscation_inner_capital_rate": _safe_div(inner_capital_words(source), word_count),
    }


def obfuscation_evidence(text: str = "", subject: str = "", html: str = "") -> Dict[str, list]:
    source = _source_text(text, subject, html)
    return {
        "mixed_script_words": [word for word in prose_words(source) if is_mixed_script_word(word)][:5],
        "inner_capital_words": [word for word in prose_words(source) if is_confusable_word(word)][:5],
    }
