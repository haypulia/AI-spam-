import re

INVISIBLE_PATTERN = re.compile(
    "["
    "­͏؜ᅟᅠ឴឵"
    "᠋-᠏​-‏‪-‮⁠-⁯"
    "ㅤ︀-️﻿ﾠ￰-￸"
    "]"
)

INLINE_SPACE_PATTERN = re.compile(r"[^\S\n]+")
BLANK_LINES_PATTERN = re.compile(r"\n{3,}")
ALL_SPACE_PATTERN = re.compile(r"[\s ]+")


def drop_invisible(text: str) -> str:
    return INVISIBLE_PATTERN.sub("", text or "")


def collapse_inline_spaces(text: str) -> str:
    text = INLINE_SPACE_PATTERN.sub(" ", text or "")
    return BLANK_LINES_PATTERN.sub("\n\n", text)


def collapse_all_spaces(text: str) -> str:
    return ALL_SPACE_PATTERN.sub(" ", text or "").strip()


def normalize_text(text: str) -> str:
    return collapse_inline_spaces(drop_invisible(text)).strip()


def invisible_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(INVISIBLE_PATTERN.findall(text)) / len(text)
