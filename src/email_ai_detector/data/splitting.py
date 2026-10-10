"""Групповое разбиение без удаления писем основного emails.jsonl."""

import hashlib
import html
import random
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def normalize_text(row: dict) -> str:
    text = row.get("text") or re.sub(r"<[^>]+>", " ", row.get("html") or "")
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    text = re.sub(r"[\u200b-\u200d\ufeff]", "", text)
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    return " ".join(re.findall(r"\w+", text.casefold()))


def _shingles(text: str) -> set:
    words = text.split()
    if len(words) < 3:
        return {text}
    return {tuple(words[i:i + 3]) for i in range(len(words) - 2)}


def validate_rows(rows: List[dict]) -> None:
    if not rows:
        raise ValueError("датасет пуст")
    ids = set()
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("пустой или повторяющийся id")
        ids.add(identifier)
        if row.get("label") not in ("human", "mixed", "ai"):
            raise ValueError("неизвестная метка: " + identifier)
        if row.get("ai_binary") != int(row["label"] != "human"):
            raise ValueError("несогласованная бинарная метка: " + identifier)
        if not normalize_text(row):
            raise ValueError("нет текста для сравнения: " + identifier)
        text = row.get("text") or ""
        for interval in row.get("ai_char_intervals", []):
            if (len(interval) != 2 or any(type(v) is not int for v in interval)
                    or not 0 <= interval[0] < interval[1] <= len(text)):
                raise ValueError("неверные символьные интервалы: " + identifier)


def find_groups(rows: List[dict], threshold: float = 0.85):
    if not 0 < threshold <= 1:
        raise ValueError("порог сходства должен быть в (0, 1]")
    validate_rows(rows)
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[max(first, second)] = min(first, second)

    keys = {}
    normalized = [normalize_text(row) for row in rows]
    for index, row in enumerate(rows):
        values = [("family", row["id"]), ("text", normalized[index])]
        if row.get("seed_id"):
            values.append(("family", row["seed_id"]))
        for field in ("topic_id", "template_id", "group_id"):
            if row.get(field):
                values.append((field, row[field]))
        for key in values:
            if key in keys:
                union(index, keys[key])
            else:
                keys[key] = index

    shingles = [_shingles(text) for text in normalized]
    inverted = defaultdict(list)
    similar = []
    for index, current in enumerate(shingles):
        candidates = set()
        for token in current:
            candidates.update(inverted[token])
        for other in sorted(candidates):
            previous = shingles[other]
            if min(len(current), len(previous)) / max(len(current), len(previous)) < threshold:
                continue
            score = len(current & previous) / len(current | previous)
            if score >= threshold:
                union(index, other)
                similar.append((other, index, score))
        for token in current:
            inverted[token].append(index)
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    return list(groups.values()), similar


def prepare_split(rows: List[dict], threshold: float = 0.85, seed: int = 42):
    rows = sorted(rows, key=lambda row: row.get("id", ""))
    groups, _ = find_groups(rows, threshold)
    retained_groups = groups
    if len(groups) < 3:
        raise ValueError("меньше трёх независимых групп: разбиение без утечек невозможно")
    labels = Counter(row["label"] for row in rows)
    total = len(rows)
    best = None
    for attempt in range(32):
        rng = random.Random(seed + attempt)
        order = list(range(len(groups)))
        rng.shuffle(order)
        order.sort(key=lambda g: -len(retained_groups[g]))
        counts = {name: Counter() for name in SPLIT_RATIOS}
        sizes = Counter()
        assignment = {}

        def cost():
            return sum(
                ((sizes[name] - ratio * total) / max(1, ratio * total)) ** 2
                + sum(((counts[name][label] - ratio * count) / max(1, ratio * count)) ** 2
                      for label, count in labels.items())
                for name, ratio in SPLIT_RATIOS.items()
            )

        for position, group_index in enumerate(order):
            indices = retained_groups[group_index]
            local = Counter(rows[i]["label"] for i in indices)
            empty = [name for name in SPLIT_RATIOS if not sizes[name]]
            choices = empty if len(order) - position == len(empty) else list(SPLIT_RATIOS)
            scores = []
            for name in choices:
                sizes[name] += len(indices)
                counts[name].update(local)
                scores.append((cost(), name))
                sizes[name] -= len(indices)
                counts[name].subtract(local)
            _, name = min(scores)
            assignment[group_index] = name
            sizes[name] += len(indices)
            counts[name].update(local)
        candidate = (cost(), assignment, dict(sizes))
        if best is None or candidate[0] < best[0]:
            best = candidate
    output = []
    for group_index, indices in enumerate(retained_groups):
        group_id = hashlib.sha256("\n".join(rows[i]["id"] for i in groups[group_index]).encode()).hexdigest()[:16]
        for index in indices:
            row = dict(rows[index], split=best[1][group_index], group_id=group_id)
            output.append(row)
    output.sort(key=lambda row: row["id"])
    audit = audit_split(output, threshold)
    if audit["cross_split_groups"]:
        raise ValueError("обнаружены утечки после разбиения")
    report = {
        "seed": seed, "similarity_threshold": threshold, "target_ratios": SPLIT_RATIOS,
        "input_count": len(rows), "output_count": len(output),
        "removed": [],
        "deduplication": "disabled: сохраняются все исходные письма",
        "groups": len(groups), "largest_group": max(map(len, groups)),
        "split_sizes": best[2], "audit": audit,
    }
    return output, report


def audit_split(rows: List[dict], threshold: float = 0.85) -> Dict[str, object]:
    groups, similar = find_groups(rows, threshold)
    leaks = [group for group in groups if len({rows[i].get("split") for i in group}) > 1]
    invalid = [row["id"] for row in rows if row.get("split") not in SPLIT_RATIOS]
    return {
        "count": len(rows), "groups": len(groups), "similar_pairs": len(similar),
        "cross_split_groups": len(leaks), "invalid_split_ids": invalid,
        "cross_split_examples": [[rows[i]["id"] for i in group] for group in leaks[:10]],
        "split_labels": {name: dict(Counter(row["label"] for row in rows if row.get("split") == name))
                         for name in SPLIT_RATIOS},
    }
