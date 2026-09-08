import dataclasses
import random
import re
from typing import Callable, Dict, List

from ..features.lexicons import CLOSERS, CONNECTIVES, MARKETING_PHRASES, OPENERS

COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
STYLE_ATTR_PATTERN = re.compile(r'\s*style\s*=\s*"[^"]*"', re.IGNORECASE)
TEMPLATE_PATTERN = re.compile(r"\{\{\s*([\w.\-]+)\s*\}\}|\[\s*(NAME|EMAIL|COMPANY|ИМЯ|КОМПАНИЯ)\s*\]", re.IGNORECASE)
TAG_SPLIT_PATTERN = re.compile(r"(<[^>]+>)")
SMART_QUOTES = {"«": '"', "»": '"', "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'"}
DASHES = {"—": "-", "–": "-"}

TEMPLATE_FILLERS = ("Alex", "Morgan", "Sam", "Иван", "Мария")

NEUTRAL_REPLACEMENTS = {
    "exclusive opportunity": "offer",
    "limited seats": "few places",
    "do not miss out": "check it",
    "exceptional gains": "returns",
    "we are pleased": "we write",
    "we are excited": "we write",
    "rest assured": "note",
    "уникальная возможность": "предложение",
    "ограниченное предложение": "предложение",
    "не упустите": "посмотрите",
    "рады сообщить": "сообщаем",
    "кроме того": "и",
    "также напоминаем": "напоминаем",
    "обращаем ваше внимание": "внимание",
    "moreover": "and",
    "furthermore": "and",
    "additionally": "also",
    "please note": "note",
}

SENTENCE_PATTERN = re.compile(r"[^\n.!?]+[.!?]*", re.UNICODE)


def map_text_nodes(html: str, transform: Callable[[str], str]) -> str:
    if not html:
        return html
    parts = TAG_SPLIT_PATTERN.split(html)
    return "".join(part if part.startswith("<") else transform(part) for part in parts)


def _replace_case_insensitive(text: str, mapping: Dict[str, str]) -> str:
    for source, target in mapping.items():
        text = re.sub(re.escape(source), target, text, flags=re.IGNORECASE)
    return text


def normalize_typography(text: str) -> str:
    for source, target in SMART_QUOTES.items():
        text = text.replace(source, target)
    for source, target in DASHES.items():
        text = text.replace(source, target)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def humanize_case(text: str, seed: int = 13) -> str:
    generator = random.Random(seed)

    def transform(match):
        sentence = match.group()
        stripped = sentence.lstrip()
        if not stripped or not stripped[0].isupper():
            return sentence
        if generator.random() < 0.6:
            offset = len(sentence) - len(stripped)
            return sentence[:offset] + stripped[0].lower() + stripped[1:]
        return sentence

    return SENTENCE_PATTERN.sub(transform, text)


def inject_typos(text: str, seed: int = 17) -> str:
    generator = random.Random(seed)
    words = text.split(" ")
    for index, word in enumerate(words):
        if len(word) < 6 or not word.isalpha():
            continue
        if generator.random() < 0.12:
            position = generator.randrange(1, len(word) - 1)
            words[index] = word[:position] + word[position + 1 :]
    text = " ".join(words)
    text = re.sub(r"(?<=[a-zа-я])\.(?=\s)", "..", text, count=2)
    return text


def strip_boilerplate(text: str) -> str:
    sentences = SENTENCE_PATTERN.findall(text)
    kept = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(phrase in lowered for phrase in OPENERS + CLOSERS):
            continue
        kept.append(sentence.strip())
    result = "\n\n".join(item for item in kept if item)
    return result or text


def neutralize_phrases(text: str) -> str:
    return _replace_case_insensitive(text, NEUTRAL_REPLACEMENTS)


def strip_markup_artifacts(html: str) -> str:
    html = COMMENT_PATTERN.sub("", html or "")
    html = STYLE_ATTR_PATTERN.sub("", html)
    html = TEMPLATE_PATTERN.sub(TEMPLATE_FILLERS[0], html)
    return html


def _record_with(record, subject=None, text=None, html=None):
    return dataclasses.replace(
        record,
        subject=record.subject if subject is None else subject,
        text=record.text if text is None else text,
        html=record.html if html is None else html,
    )


def mask_markup(record):
    return _record_with(record, html=strip_markup_artifacts(record.html))


def mask_typography(record):
    return _record_with(
        record,
        subject=normalize_typography(record.subject),
        text=normalize_typography(record.text),
        html=map_text_nodes(record.html, normalize_typography),
    )


def mask_case(record):
    return _record_with(
        record,
        text=humanize_case(record.text),
        html=map_text_nodes(record.html, humanize_case),
    )


def mask_typos(record):
    return _record_with(
        record,
        text=inject_typos(record.text),
        html=map_text_nodes(record.html, inject_typos),
    )


def mask_boilerplate(record):
    return _record_with(record, text=strip_boilerplate(record.text))


def mask_phrases(record):
    return _record_with(
        record,
        subject=neutralize_phrases(record.subject),
        text=neutralize_phrases(record.text),
        html=map_text_nodes(record.html, neutralize_phrases),
    )


def mask_combined(record):
    masked = mask_markup(record)
    masked = mask_typography(masked)
    masked = mask_phrases(masked)
    masked = mask_boilerplate(masked)
    masked = mask_case(masked)
    return mask_typos(masked)


MASKS: Dict[str, Callable] = {
    "markup_cleanup": mask_markup,
    "typography_normalization": mask_typography,
    "case_humanization": mask_case,
    "typo_injection": mask_typos,
    "boilerplate_removal": mask_boilerplate,
    "phrase_neutralization": mask_phrases,
    "combined_evasion": mask_combined,
}

MASK_DESCRIPTIONS = {
    "markup_cleanup": "удаление служебных комментариев, инлайновых стилей и шаблонных переменных",
    "typography_normalization": "приведение кавычек, тире и пробелов к нейтральному виду",
    "case_humanization": "перевод части предложений в строчное начало",
    "typo_injection": "внесение опечаток и небрежной пунктуации",
    "boilerplate_removal": "удаление шаблонных приветствий и подписей",
    "phrase_neutralization": "замена маркетинговых клише и связок нейтральными формулировками",
    "combined_evasion": "последовательное применение всех перечисленных приёмов",
}


def apply_mask(name: str, records: List) -> List:
    mask = MASKS[name]
    return [mask(record) for record in records]
