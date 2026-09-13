import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .html_signals import HTML_FEATURE_NAMES, extract_html_features, html_evidence, strip_tags
from .lexicons import CLOSERS, OPENERS
from .normalize import normalize_text
from .obfuscation import (
    OBFUSCATION_FEATURE_NAMES,
    extract_obfuscation_features,
    obfuscation_evidence,
)
from .patterns import REPEATED_PUNCT_PATTERN
from .text_signals import (
    SUBJECT_FEATURE_NAMES,
    TEXT_FEATURE_NAMES,
    extract_subject_features,
    extract_text_features,
    split_sentences,
)

EDGE_FEATURE_NAMES = (
    "opener_greeting",
    "opener_capitalized",
    "opener_terminated",
    "opener_len_log",
    "opener_typo",
    "closer_signoff",
    "closer_capitalized",
    "closer_terminated",
    "closer_len_log",
    "closer_typo",
)

EXPLANATION_CATEGORIES = ("subject", "opener", "body", "cta", "closer", "html_template")

CATEGORY_ALIASES = {"ai_cta": "cta", "cta": "cta", "html": "html_template"}

FEATURE_NAMES: Tuple[str, ...] = (
    tuple("subject_%s" % name.replace("subject_", "") for name in SUBJECT_FEATURE_NAMES)
    + tuple("body_%s" % name for name in TEXT_FEATURE_NAMES)
    + tuple("html_%s" % name for name in HTML_FEATURE_NAMES)
    + EDGE_FEATURE_NAMES
    + OBFUSCATION_FEATURE_NAMES
)


@dataclass
class EmailFeatures:
    vector: Dict[str, float] = field(default_factory=dict)
    categories: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, list] = field(default_factory=dict)

    def as_list(self, names: Optional[Sequence[str]] = None) -> List[float]:
        names = names or FEATURE_NAMES
        return [float(self.vector.get(name, 0.0)) for name in names]


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _edge_features(sentences: Sequence[Tuple[int, int, str]]) -> Dict[str, float]:
    if not sentences:
        return {name: 0.0 for name in EDGE_FEATURE_NAMES}

    first = sentences[0][2]
    last = sentences[-1][2]
    first_lowered = first.lower()
    last_lowered = last.lower()

    return {
        "opener_greeting": 1.0 if any(phrase in first_lowered for phrase in OPENERS) else 0.0,
        "opener_capitalized": 1.0 if first[:1].isupper() else 0.0,
        "opener_terminated": 1.0 if first.endswith((".", "!", "?")) else 0.0,
        "opener_len_log": math.log1p(len(first)),
        "opener_typo": float(len(REPEATED_PUNCT_PATTERN.findall(first))),
        "closer_signoff": 1.0 if any(phrase in last_lowered for phrase in CLOSERS) else 0.0,
        "closer_capitalized": 1.0 if last[:1].isupper() else 0.0,
        "closer_terminated": 1.0 if last.endswith((".", "!", "?")) else 0.0,
        "closer_len_log": math.log1p(len(last)),
        "closer_typo": float(len(REPEATED_PUNCT_PATTERN.findall(last))),
    }


def _polish_score(text_features: Dict[str, float]) -> float:
    return _clip(
        0.5 * text_features["sentence_capitalization_rate"]
        + 0.3 * text_features["sentence_terminator_rate"]
        + 0.2 * (1.0 - _clip(text_features["typo_marker_rate"]))
    )


def _subject_score(subject_features: Dict[str, float]) -> float:
    return _clip(
        0.4 * _clip(subject_features["subject_marketing"])
        + 0.25 * _clip(subject_features["subject_cta"])
        + 0.2 * _clip(subject_features["subject_urgency"])
        + 0.15 * subject_features["subject_title_rate"]
    )


def _opener_score(text_features: Dict[str, float], first_sentence: str) -> float:
    return _clip(
        0.6 * text_features["opener_phrase"]
        + 0.4 * (1.0 if first_sentence[:1].isupper() and first_sentence.endswith((".", "!", "?")) else 0.0)
    )


def _closer_score(text_features: Dict[str, float], last_sentence: str) -> float:
    return _clip(
        0.7 * text_features["closer_phrase"]
        + 0.3 * (1.0 if last_sentence[:1].isupper() else 0.0)
    )


def _cta_score(text_features: Dict[str, float], subject_features: Dict[str, float]) -> float:
    return _clip(
        0.7 * _clip(text_features["cta_phrase_rate"] * 2.0)
        + 0.3 * _clip(subject_features["subject_cta"])
    )


def _body_score(text_features: Dict[str, float]) -> float:
    return _clip(
        0.45 * _polish_score(text_features)
        + 0.25 * _clip(text_features["marketing_phrase_rate"] * 2.0)
        + 0.2 * _clip(text_features["connective_rate"] * 2.0)
        + 0.1 * _clip(text_features["burstiness"])
    )


def _html_template_score(html_features: Dict[str, float]) -> float:
    return _clip(
        0.3 * html_features["template_variable"]
        + 0.25 * html_features["suspicious_comment"]
        + 0.15 * html_features["deep_nesting"]
        + 0.15 * html_features["duplicated_styles"]
        + 0.15 * html_features["empty_cell"]
    )


def _category_scores(
    subject_features: Dict[str, float],
    text_features: Dict[str, float],
    html_features: Dict[str, float],
    sentences: Sequence[Tuple[int, int, str]],
) -> Dict[str, float]:
    first_sentence = sentences[0][2] if sentences else ""
    last_sentence = sentences[-1][2] if sentences else ""
    return {
        "subject": _subject_score(subject_features),
        "opener": _opener_score(text_features, first_sentence),
        "body": _body_score(text_features),
        "cta": _cta_score(text_features, subject_features),
        "closer": _closer_score(text_features, last_sentence),
        "html_template": _html_template_score(html_features),
    }


def extract_features(
    text: str = "",
    subject: str = "",
    html: str = "",
) -> EmailFeatures:
    obfuscation_features = extract_obfuscation_features(text=text, subject=subject, html=html)

    text = normalize_text(text)
    subject = normalize_text(subject)

    if not text and html:
        text = strip_tags(html)

    subject_features = extract_subject_features(subject)
    text_features = extract_text_features(text)
    html_features = extract_html_features(html)
    sentences = split_sentences(text)
    edge_features = _edge_features(sentences)

    vector: Dict[str, float] = {}
    vector.update({name: value for name, value in subject_features.items()})
    vector.update({"body_%s" % name: value for name, value in text_features.items()})
    vector.update({"html_%s" % name: value for name, value in html_features.items()})
    vector.update(edge_features)
    vector.update(obfuscation_features)

    categories = _category_scores(subject_features, text_features, html_features, sentences)

    evidence = html_evidence(html)
    evidence["sentences"] = [sentence for _, _, sentence in sentences[:10]]
    evidence.update(obfuscation_evidence(text=text, subject=subject, html=html))

    return EmailFeatures(vector=vector, categories=categories, evidence=evidence)


SEGMENT_FEATURE_NAMES: Tuple[str, ...] = tuple("segment_%s" % name for name in TEXT_FEATURE_NAMES) + (
    "segment_relative_position",
    "segment_is_first",
    "segment_is_last",
    "segment_char_len_log",
)


def extract_segment_features(sentence: str, index: int, total: int) -> Dict[str, float]:
    features = {"segment_%s" % name: value for name, value in extract_text_features(sentence).items()}
    features["segment_relative_position"] = (index / max(1, total - 1)) if total > 1 else 0.0
    features["segment_is_first"] = 1.0 if index == 0 else 0.0
    features["segment_is_last"] = 1.0 if index == total - 1 else 0.0
    features["segment_char_len_log"] = math.log1p(len(sentence))
    return {name: float(features.get(name, 0.0)) for name in SEGMENT_FEATURE_NAMES}


def iter_segments(text: str):
    text = normalize_text(text)
    sentences = split_sentences(text)
    total = len(sentences)
    for index, (start, end, sentence) in enumerate(sentences):
        yield {
            "index": index,
            "start": start,
            "end": end,
            "text": sentence,
            "features": extract_segment_features(sentence, index, total),
        }


def normalize_category(name: str) -> str:
    return CATEGORY_ALIASES.get(name, name)
