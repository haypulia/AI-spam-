import math
from typing import Dict, List, Sequence, Tuple

from .lexicons import (
    CALL_TO_ACTION,
    CLOSERS,
    CONNECTIVES,
    FUNCTION_WORDS,
    GENERIC_GREETINGS,
    IMPERATIVE_OPENINGS,
    MARKETING_PHRASES,
    OPENERS,
    TYPO_TOKENS,
    URGENCY,
)

from .patterns import (
    CLAUSE_PATTERN,
    CURRENCY_PATTERN,
    DASH_PATTERN,
    DATE_PATTERN,
    DOUBLE_SPACE_PATTERN,
    EMOJI_PATTERN,
    IDENTIFIER_PATTERN,
    REPEATED_PUNCT_PATTERN,
    SENTENCE_PATTERN,
    SMART_QUOTES_PATTERN,
    TEXT_URL_PATTERN,
    TOKEN_PATTERN,
)

FUNCTION_WORD_SET = frozenset(FUNCTION_WORDS)
IMPERATIVE_SET = frozenset(IMPERATIVE_OPENINGS)

TEXT_FEATURE_NAMES = (
    "word_count_log",
    "avg_sentence_len",
    "sentence_len_std",
    "burstiness",
    "type_token_ratio",
    "hapax_ratio",
    "avg_word_len",
    "sentence_capitalization_rate",
    "sentence_terminator_rate",
    "typo_marker_rate",
    "repeated_punct_rate",
    "double_space_rate",
    "uppercase_ratio",
    "digit_ratio",
    "comma_rate",
    "exclamation_rate",
    "emoji_rate",
    "smart_punctuation_rate",
    "spaced_dash_rate",
    "marketing_phrase_rate",
    "connective_rate",
    "urgency_rate",
    "cta_phrase_rate",
    "opener_phrase",
    "closer_phrase",
    "generic_greeting",
    "repeated_bigram_ratio",
    "url_rate",
    "function_word_ratio",
    "long_word_ratio",
    "clause_density",
    "specificity_rate",
    "digit_token_ratio",
    "identifier_rate",
    "sentence_opening_repetition",
    "imperative_opening_rate",
)


def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    for match in SENTENCE_PATTERN.finditer(text or ""):
        chunk = match.group()
        stripped = chunk.strip()
        if not stripped:
            continue
        start = match.start() + (len(chunk) - len(chunk.lstrip()))
        end = start + len(stripped)
        spans.append((start, end, stripped))
    return spans


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _phrase_hits(lowered: str, phrases: Sequence[str]) -> int:
    return sum(1 for phrase in phrases if phrase in lowered)


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _typo_markers(text: str, lowered: str, sentences: Sequence[Tuple[int, int, str]]) -> int:
    markers = 0
    markers += len(REPEATED_PUNCT_PATTERN.findall(text))
    markers += sum(1 for token in TYPO_TOKENS if token in lowered)
    markers += sum(1 for _, _, sentence in sentences if sentence and sentence[0].islower())
    return markers


def _opening_repetition(sentences: Sequence[Tuple[int, int, str]]) -> float:
    openings = []
    for _, _, sentence in sentences:
        tokens = tokenize(sentence)[:2]
        if tokens:
            openings.append(tuple(tokens))
    if len(openings) < 2:
        return 0.0
    return 1.0 - len(set(openings)) / len(openings)


def extract_text_features(text: str) -> Dict[str, float]:
    text = text or ""
    lowered = text.lower()
    sentences = split_sentences(text)
    tokens = tokenize(text)
    token_count = len(tokens)
    char_count = len(text)
    sentence_count = len(sentences)
    unique_tokens = set(tokens)
    lengths = [len(tokenize(sentence)) for _, _, sentence in sentences]
    mean_length = _safe_div(sum(lengths), len(lengths))
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    hapax = sum(1 for value in counts.values() if value == 1)
    bigrams = [tuple(tokens[index : index + 2]) for index in range(max(0, token_count - 1))]
    unique_bigrams = set(bigrams)

    features = {
        "word_count_log": math.log1p(token_count),
        "avg_sentence_len": mean_length,
        "sentence_len_std": _std(lengths),
        "burstiness": _safe_div(_std(lengths), mean_length),
        "type_token_ratio": _safe_div(len(unique_tokens), token_count),
        "hapax_ratio": _safe_div(hapax, token_count),
        "avg_word_len": _safe_div(sum(len(token) for token in tokens), token_count),
        "sentence_capitalization_rate": _safe_div(
            sum(1 for _, _, sentence in sentences if sentence[:1].isupper()), sentence_count
        ),
        "sentence_terminator_rate": _safe_div(
            sum(1 for _, _, sentence in sentences if sentence.endswith((".", "!", "?"))), sentence_count
        ),
        "typo_marker_rate": _safe_div(_typo_markers(text, lowered, sentences), max(1, sentence_count)),
        "repeated_punct_rate": _safe_div(len(REPEATED_PUNCT_PATTERN.findall(text)), max(1, sentence_count)),
        "double_space_rate": _safe_div(len(DOUBLE_SPACE_PATTERN.findall(text)), max(1, sentence_count)),
        "uppercase_ratio": _safe_div(sum(1 for char in text if char.isupper()), char_count),
        "digit_ratio": _safe_div(sum(1 for char in text if char.isdigit()), char_count),
        "comma_rate": _safe_div(text.count(","), max(1, sentence_count)),
        "exclamation_rate": _safe_div(text.count("!"), max(1, sentence_count)),
        "emoji_rate": _safe_div(len(EMOJI_PATTERN.findall(text)), max(1, sentence_count)),
        "smart_punctuation_rate": _safe_div(len(SMART_QUOTES_PATTERN.findall(text)), max(1, sentence_count)),
        "spaced_dash_rate": _safe_div(len(DASH_PATTERN.findall(text)), max(1, sentence_count)),
        "marketing_phrase_rate": _safe_div(_phrase_hits(lowered, MARKETING_PHRASES), max(1, sentence_count)),
        "connective_rate": _safe_div(_phrase_hits(lowered, CONNECTIVES), max(1, sentence_count)),
        "urgency_rate": _safe_div(_phrase_hits(lowered, URGENCY), max(1, sentence_count)),
        "cta_phrase_rate": _safe_div(_phrase_hits(lowered, CALL_TO_ACTION), max(1, sentence_count)),
        "opener_phrase": 1.0 if _phrase_hits(lowered[:160], OPENERS) else 0.0,
        "closer_phrase": 1.0 if _phrase_hits(lowered[-200:], CLOSERS) else 0.0,
        "generic_greeting": 1.0 if _phrase_hits(lowered, GENERIC_GREETINGS) else 0.0,
        "repeated_bigram_ratio": 1.0 - _safe_div(len(unique_bigrams), len(bigrams)) if bigrams else 0.0,
        "url_rate": _safe_div(len(TEXT_URL_PATTERN.findall(text)), max(1, sentence_count)),
        "function_word_ratio": _safe_div(
            sum(1 for token in tokens if token in FUNCTION_WORD_SET), token_count
        ),
        "long_word_ratio": _safe_div(sum(1 for token in tokens if len(token) > 7), token_count),
        "clause_density": _safe_div(len(CLAUSE_PATTERN.findall(text)), max(1, sentence_count)),
        "specificity_rate": _safe_div(
            len(IDENTIFIER_PATTERN.findall(text))
            + len(DATE_PATTERN.findall(text))
            + len(CURRENCY_PATTERN.findall(text)),
            max(1, sentence_count),
        ),
        "digit_token_ratio": _safe_div(
            sum(1 for token in tokens if any(char.isdigit() for char in token)), token_count
        ),
        "identifier_rate": _safe_div(len(IDENTIFIER_PATTERN.findall(text)), max(1, sentence_count)),
        "sentence_opening_repetition": _opening_repetition(sentences),
        "imperative_opening_rate": _safe_div(
            sum(
                1
                for _, _, sentence in sentences
                if tokenize(sentence)[:1] and tokenize(sentence)[0] in IMPERATIVE_SET
            ),
            max(1, sentence_count),
        ),
    }
    return {name: float(features.get(name, 0.0)) for name in TEXT_FEATURE_NAMES}


def extract_subject_features(subject: str) -> Dict[str, float]:
    subject = subject or ""
    lowered = subject.lower()
    tokens = tokenize(subject)
    letters = [char for char in subject if char.isalpha()]
    words = subject.split()
    title_words = sum(1 for word in words if word[:1].isupper())
    return {
        "subject_len": float(len(subject)),
        "subject_word_count": float(len(tokens)),
        "subject_title_rate": _safe_div(title_words, len(words)),
        "subject_uppercase_ratio": _safe_div(sum(1 for char in letters if char.isupper()), len(letters)),
        "subject_exclamation": float(subject.count("!")),
        "subject_digit_ratio": _safe_div(sum(1 for char in subject if char.isdigit()), len(subject)),
        "subject_marketing": float(_phrase_hits(lowered, MARKETING_PHRASES)),
        "subject_urgency": float(_phrase_hits(lowered, URGENCY)),
        "subject_cta": float(_phrase_hits(lowered, CALL_TO_ACTION)),
        "subject_dash": float(len(DASH_PATTERN.findall(subject))),
    }


SUBJECT_FEATURE_NAMES = tuple(sorted(extract_subject_features("").keys()))
