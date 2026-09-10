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
from .text_signals import (
    REPEATED_PUNCT,
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

OCR_FEATURE_NAMES = (
    "ocr_present",
    "ocr_len_log",
    "ocr_marketing_rate",
    "ocr_cta_rate",
    "ocr_capitalization_rate",
    "ocr_typo_rate",
)

EXPLANATION_CATEGORIES = ("subject", "opener", "body", "cta", "closer", "html_template", "image")

CATEGORY_ALIASES = {"ai_cta": "cta", "cta": "cta", "html": "html_template"}

FEATURE_NAMES: Tuple[str, ...] = (
    tuple("subject_%s" % name.replace("subject_", "") for name in SUBJECT_FEATURE_NAMES)
    + tuple("body_%s" % name for name in TEXT_FEATURE_NAMES)
    + tuple("html_%s" % name for name in HTML_FEATURE_NAMES)
    + EDGE_FEATURE_NAMES
    + OCR_FEATURE_NAMES
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


def _ocr_features(ocr_text: str) -> Dict[str, float]:
    ocr_text = (ocr_text or "").strip()
    if not ocr_text:
        return {name: 0.0 for name in OCR_FEATURE_NAMES}
    text_features = extract_text_features(ocr_text)
    return {
        "ocr_present": 1.0,
        "ocr_len_log": math.log1p(len(ocr_text)),
        "ocr_marketing_rate": text_features["marketing_phrase_rate"],
        "ocr_cta_rate": text_features["cta_phrase_rate"],
        "ocr_capitalization_rate": text_features["sentence_capitalization_rate"],
        "ocr_typo_rate": text_features["typo_marker_rate"],
    }


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
        "opener_typo": float(len(REPEATED_PUNCT.findall(first))),
        "closer_signoff": 1.0 if any(phrase in last_lowered for phrase in CLOSERS) else 0.0,
        "closer_capitalized": 1.0 if last[:1].isupper() else 0.0,
        "closer_terminated": 1.0 if last.endswith((".", "!", "?")) else 0.0,
        "closer_len_log": math.log1p(len(last)),
        "closer_typo": float(len(REPEATED_PUNCT.findall(last))),
    }


def _category_scores(
    subject_features: Dict[str, float],
    text_features: Dict[str, float],
    html_features: Dict[str, float],
    ocr_features: Dict[str, float],
    sentences: Sequence[Tuple[int, int, str]],
) -> Dict[str, float]:
    polish = _clip(
        0.5 * text_features["sentence_capitalization_rate"]
        + 0.3 * text_features["sentence_terminator_rate"]
        + 0.2 * (1.0 - _clip(text_features["typo_marker_rate"]))
    )

    first_sentence = sentences[0][2] if sentences else ""
    last_sentence = sentences[-1][2] if sentences else ""

    subject_score = _clip(
        0.4 * _clip(subject_features["subject_marketing"])
        + 0.25 * _clip(subject_features["subject_cta"])
        + 0.2 * _clip(subject_features["subject_urgency"])
        + 0.15 * subject_features["subject_title_rate"]
    )

    opener_score = _clip(
        0.6 * text_features["opener_phrase"]
        + 0.4 * (1.0 if first_sentence[:1].isupper() and first_sentence.endswith((".", "!", "?")) else 0.0)
    )

    closer_score = _clip(
        0.7 * text_features["closer_phrase"]
        + 0.3 * (1.0 if last_sentence[:1].isupper() else 0.0)
    )

    cta_score = _clip(
        0.7 * _clip(text_features["cta_phrase_rate"] * 2.0)
        + 0.3 * _clip(subject_features["subject_cta"])
    )

    body_score = _clip(
        0.45 * polish
        + 0.25 * _clip(text_features["marketing_phrase_rate"] * 2.0)
        + 0.2 * _clip(text_features["connective_rate"] * 2.0)
        + 0.1 * _clip(text_features["burstiness"])
    )

    html_score = _clip(
        0.3 * html_features["template_variable"]
        + 0.25 * html_features["suspicious_comment"]
        + 0.15 * html_features["deep_nesting"]
        + 0.15 * html_features["duplicated_styles"]
        + 0.15 * html_features["empty_cell"]
    )

    image_score = _clip(
        ocr_features["ocr_present"]
        * (0.4 + 0.3 * ocr_features["ocr_capitalization_rate"] + 0.3 * _clip(ocr_features["ocr_marketing_rate"] + ocr_features["ocr_cta_rate"]))
    )

    return {
        "subject": subject_score,
        "opener": opener_score,
        "body": body_score,
        "cta": cta_score,
        "closer": closer_score,
        "html_template": html_score,
        "image": image_score,
    }


def extract_features(
    text: str = "",
    subject: str = "",
    html: str = "",
    ocr_text: str = "",
) -> EmailFeatures:
    obfuscation_features = extract_obfuscation_features(text=text, subject=subject, html=html)

    text = normalize_text(text)
    subject = normalize_text(subject)
    ocr_text = normalize_text(ocr_text)

    if not text and html:
        text = strip_tags(html)

    subject_features = extract_subject_features(subject)
    text_features = extract_text_features(text)
    html_features = extract_html_features(html)
    ocr_features = _ocr_features(ocr_text)
    sentences = split_sentences(text)
    edge_features = _edge_features(sentences)

    vector: Dict[str, float] = {}
    vector.update({name: value for name, value in subject_features.items()})
    vector.update({"body_%s" % name: value for name, value in text_features.items()})
    vector.update({"html_%s" % name: value for name, value in html_features.items()})
    vector.update(edge_features)
    vector.update(ocr_features)
    vector.update(obfuscation_features)

    categories = _category_scores(subject_features, text_features, html_features, ocr_features, sentences)

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
