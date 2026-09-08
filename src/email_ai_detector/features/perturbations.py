import random
import re
from typing import List, Sequence

SENTENCE_PATTERN = re.compile(r"[^\n.!?]+[.!?]*", re.UNICODE)
WORD_PATTERN = re.compile(r"\w{4,}", re.UNICODE)
TAG_SPLIT_PATTERN = re.compile(r"(<[^>]+>)")

DEFAULT_SEEDS = (101, 202, 303)


def lowercase_openings(text: str, generator: random.Random, rate: float = 0.5) -> str:
    def transform(match):
        sentence = match.group()
        stripped = sentence.lstrip()
        if not stripped or not stripped[0].isupper() or generator.random() > rate:
            return sentence
        offset = len(sentence) - len(stripped)
        return sentence[:offset] + stripped[0].lower() + stripped[1:]

    return SENTENCE_PATTERN.sub(transform, text)


def character_noise(text: str, generator: random.Random, rate: float = 0.08) -> str:
    words = text.split(" ")
    for index, word in enumerate(words):
        if not WORD_PATTERN.fullmatch(word) or generator.random() > rate:
            continue
        position = generator.randrange(1, len(word) - 1)
        action = generator.choice(("delete", "swap", "double"))
        if action == "delete":
            words[index] = word[:position] + word[position + 1 :]
        elif action == "swap":
            words[index] = word[:position] + word[position + 1 : position + 2] + word[position : position + 1] + word[position + 2 :]
        else:
            words[index] = word[:position] + word[position] + word[position:]
    return " ".join(words)


def punctuation_noise(text: str, generator: random.Random, rate: float = 0.3) -> str:
    def transform(match):
        sentence = match.group()
        if generator.random() > rate:
            return sentence
        if sentence.endswith((".", "!", "?")):
            return sentence[:-1] if generator.random() < 0.5 else sentence + sentence[-1]
        return sentence

    return SENTENCE_PATTERN.sub(transform, text)


def spacing_noise(text: str, generator: random.Random, rate: float = 0.15) -> str:
    parts = text.split(" ")
    result = []
    for part in parts:
        result.append(part)
        if generator.random() < rate:
            result.append("")
    return " ".join(result)


def perturb_text(text: str, seed: int) -> str:
    if not text:
        return text
    generator = random.Random(seed)
    noisy = lowercase_openings(text, generator)
    noisy = character_noise(noisy, generator)
    noisy = punctuation_noise(noisy, generator)
    return spacing_noise(noisy, generator)


def perturb_html(html: str, seed: int) -> str:
    if not html:
        return html
    generator = random.Random(seed + 1)
    parts = TAG_SPLIT_PATTERN.split(html)
    rebuilt = []
    for part in parts:
        if part.startswith("<"):
            rebuilt.append(part)
            continue
        noisy = lowercase_openings(part, generator)
        rebuilt.append(character_noise(noisy, generator))
    return "".join(rebuilt)


def perturb_record(record, seed: int):
    import dataclasses

    return dataclasses.replace(
        record,
        subject=perturb_text(record.subject, seed + 7),
        text=perturb_text(record.text, seed),
        html=perturb_html(record.html, seed),
    )


def augmented_records(records: Sequence, seeds: Sequence[int] = DEFAULT_SEEDS) -> List:
    expanded = list(records)
    for seed in seeds:
        expanded.extend(perturb_record(record, seed) for record in records)
    return expanded
