import math
import re
from html import unescape
from typing import Dict, List

from .lexicons import SUSPICIOUS_COMMENT_MARKERS, TEMPLATE_PATTERNS
from .normalize import collapse_all_spaces, drop_invisible

COMMENT_PATTERN = re.compile(r"<!--(.*?)-->", re.DOTALL)
TAG_PATTERN = re.compile(r"<\s*(/?)\s*([a-zA-Z0-9]+)", re.IGNORECASE)
STYLE_PATTERN = re.compile(r'style\s*=\s*"([^"]*)"', re.IGNORECASE)
EMPTY_CELL_PATTERN = re.compile(r"<td[^>]*>\s*(?:&nbsp;|\s)*</td>", re.IGNORECASE)
IMG_PATTERN = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
ALT_PATTERN = re.compile(r'alt\s*=\s*"[^"]+"', re.IGNORECASE)
TEXT_TAG_PATTERN = re.compile(r"<[^>]+>")
NON_CONTENT_PATTERN = re.compile(
    r"<(style|script|head)\b[^>]*>.*?</\1\s*>|<(style|script)\b[^>]*>.*",
    re.DOTALL | re.IGNORECASE,
)
TEMPLATE_REGEXES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in TEMPLATE_PATTERNS)

HTML_FEATURE_NAMES = (
    "html_len_log",
    "template_variable",
    "template_variable_count",
    "suspicious_comment",
    "comment_count",
    "table_depth",
    "deep_nesting",
    "duplicated_styles",
    "duplicate_style_ratio",
    "empty_cell",
    "empty_cell_count",
    "inline_style_overuse",
    "inline_style_rate",
    "structural_redundancy",
    "missing_alt",
    "tag_diversity",
    "markup_text_ratio",
)


def strip_tags(html: str) -> str:
    cleaned = COMMENT_PATTERN.sub(" ", html or "")
    cleaned = NON_CONTENT_PATTERN.sub(" ", cleaned)
    cleaned = TEXT_TAG_PATTERN.sub(" ", cleaned)
    return collapse_all_spaces(drop_invisible(unescape(cleaned)))


def find_template_variables(html: str) -> List[str]:
    matches = []
    for regex in TEMPLATE_REGEXES:
        matches.extend(regex.findall(html or ""))
    return matches


def find_suspicious_comments(html: str) -> List[str]:
    suspicious = []
    for comment in COMMENT_PATTERN.findall(html or ""):
        lowered = comment.lower()
        if any(marker in lowered for marker in SUSPICIOUS_COMMENT_MARKERS):
            suspicious.append(comment.strip())
    return suspicious


def max_table_depth(html: str) -> int:
    depth = 0
    max_depth = 0
    for closing, tag in TAG_PATTERN.findall(html or ""):
        if tag.lower() != "table":
            continue
        if closing:
            depth = max(0, depth - 1)
        else:
            depth += 1
            max_depth = max(max_depth, depth)
    return max_depth


def duplicate_style_stats(html: str):
    styles = [style.strip() for style in STYLE_PATTERN.findall(html or "") if style.strip()]
    if not styles:
        return 0, 0.0
    unique = len(set(styles))
    return len(styles), 1.0 - unique / len(styles)


def extract_html_features(html: str) -> Dict[str, float]:
    html = html or ""
    text = strip_tags(html)
    tags = [tag.lower() for closing, tag in TAG_PATTERN.findall(html) if not closing]
    tag_count = len(tags)
    template_variables = find_template_variables(html)
    suspicious_comments = find_suspicious_comments(html)
    comments = COMMENT_PATTERN.findall(html)
    style_count, duplicate_ratio = duplicate_style_stats(html)
    empty_cells = EMPTY_CELL_PATTERN.findall(html)
    images = IMG_PATTERN.findall(html)
    missing_alt = sum(1 for image in images if not ALT_PATTERN.search(image))
    depth = max_table_depth(html)
    div_count = tags.count("div")
    table_count = tags.count("table")

    features = {
        "html_len_log": math.log1p(len(html)),
        "template_variable": 1.0 if template_variables else 0.0,
        "template_variable_count": float(len(template_variables)),
        "suspicious_comment": 1.0 if suspicious_comments else 0.0,
        "comment_count": float(len(comments)),
        "table_depth": float(depth),
        "deep_nesting": 1.0 if depth > 5 else 0.0,
        "duplicated_styles": 1.0 if duplicate_ratio >= 0.5 and style_count >= 4 else 0.0,
        "duplicate_style_ratio": float(duplicate_ratio),
        "empty_cell": 1.0 if empty_cells else 0.0,
        "empty_cell_count": float(len(empty_cells)),
        "inline_style_overuse": 1.0 if tag_count and style_count / tag_count > 0.6 else 0.0,
        "inline_style_rate": float(style_count / tag_count) if tag_count else 0.0,
        "structural_redundancy": 1.0 if div_count + table_count > 0 and (div_count + table_count) / max(1, tag_count) > 0.6 else 0.0,
        "missing_alt": float(missing_alt),
        "tag_diversity": float(len(set(tags)) / tag_count) if tag_count else 0.0,
        "markup_text_ratio": float(len(html) / max(1, len(text.strip()))),
    }
    return {name: float(features.get(name, 0.0)) for name in HTML_FEATURE_NAMES}


def html_evidence(html: str) -> Dict[str, list]:
    return {
        "template_variables": find_template_variables(html)[:10],
        "suspicious_comments": find_suspicious_comments(html)[:10],
        "table_depth": [max_table_depth(html)],
        "empty_cells": EMPTY_CELL_PATTERN.findall(html or "")[:5],
    }
