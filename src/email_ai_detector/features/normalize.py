from .patterns import (
    ALL_SPACE_PATTERN,
    BLANK_LINES_PATTERN,
    INLINE_SPACE_PATTERN,
    INVISIBLE_PATTERN,
)


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
