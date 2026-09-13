import re

SENTENCE_PATTERN = re.compile(r"[^\n.!?]+[.!?]*", re.UNICODE)
TOKEN_PATTERN = re.compile(r"[\w'’-]+", re.UNICODE)
LETTER_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)

TEXT_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
LINK_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\S+@\S+")

EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]",
    re.UNICODE,
)
REPEATED_PUNCT_PATTERN = re.compile(r"([.,!?])\1+")
DOUBLE_SPACE_PATTERN = re.compile(r"[ ]{2,}")
SMART_QUOTES_PATTERN = re.compile(r"[«»“”„‟‘’]")
DASH_PATTERN = re.compile(r"\s[—–-]\s")

IDENTIFIER_PATTERN = re.compile(r"\b[A-ZА-Я]{2,}[-_ ]?\d{3,}\b|\b\d{4,}\b", re.UNICODE)
DATE_PATTERN = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?\b|\b\d{1,2}\s+(?:янв|фев|мар|апр|мая|июн|июл|авг|сен|окт|ноя|дек|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
    re.IGNORECASE,
)
CURRENCY_PATTERN = re.compile(r"[$€₽£]\s?\d|\b\d[\d\s.,]*\s?(?:руб|₽|usd|eur|долл)", re.IGNORECASE)
CLAUSE_PATTERN = re.compile(
    r"[,;:]|\b(?:and|or|but|that|which|и|или|но|что|который|которая)\b", re.IGNORECASE
)

CYRILLIC_PATTERN = re.compile(r"[Ѐ-ӿ]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")

INVISIBLE_PATTERN = re.compile(
    "["
    "­͏؜ᅟᅠ឴឵"
    "᠋-᠏​-‏‪-‮⁠-⁯"
    "ㅤ︀-️﻿ﾠ￰-￸"
    "]"
)
INTRAWORD_INVISIBLE_PATTERN = re.compile(
    r"(?<=[^\W\d_])%s(?=[^\W\d_])" % INVISIBLE_PATTERN.pattern, re.UNICODE
)
INLINE_SPACE_PATTERN = re.compile(r"[^\S\n]+")
BLANK_LINES_PATTERN = re.compile(r"\n{3,}")
ALL_SPACE_PATTERN = re.compile(r"[\s ]+")

COMMENT_PATTERN = re.compile(r"<!--(.*?)-->", re.DOTALL)
TAG_PATTERN = re.compile(r"<\s*(/?)\s*([a-zA-Z0-9]+)", re.IGNORECASE)
TEXT_TAG_PATTERN = re.compile(r"<[^>]+>")
STYLE_PATTERN = re.compile(r'style\s*=\s*"([^"]*)"', re.IGNORECASE)
EMPTY_CELL_PATTERN = re.compile(r"<td[^>]*>\s*(?:&nbsp;|\s)*</td>", re.IGNORECASE)
IMG_PATTERN = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
ALT_PATTERN = re.compile(r'alt\s*=\s*"[^"]+"', re.IGNORECASE)
NON_CONTENT_PATTERN = re.compile(
    r"<(style|script|head)\b[^>]*>.*?</\1\s*>|<(style|script)\b[^>]*>.*",
    re.DOTALL | re.IGNORECASE,
)

TABLE_BLOCK_PATTERN = re.compile(r"(<table[^>]*>.*?</table>)", re.DOTALL | re.IGNORECASE)
SECTION_BLOCK_PATTERN = re.compile(
    r"(<div[^>]*class=[\"'](?:section|content|main|header|footer)[^\"']*[\"'][^>]*>.*?</div>)",
    re.DOTALL | re.IGNORECASE,
)
LIST_BLOCK_PATTERN = re.compile(r"(<ul[^>]*>.*?</ul>|<ol[^>]*>.*?</ol>)", re.DOTALL | re.IGNORECASE)
