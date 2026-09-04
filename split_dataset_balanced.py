import json
import random
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("generated_dataset/dataset_deepseek.jsonl")

TRAIN_FILE = Path("generated_dataset/dataset_train.jsonl")
VAL_FILE = Path("generated_dataset/dataset_val.jsonl")
TEST_FILE = Path("generated_dataset/dataset_test.jsonl")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

REQUIRED_FIELDS = [
    "id",
    "original_email",
    "generated_email",
    "original_fragment",
    "generated_fragment",
    "model",
    "transformation",
    "language",
    "domain",
    "email_format",
]


# ============================================================
# HELPERS
# ============================================================

def print_separator():
    print("=" * 70)


def load_dataset(path):
    data = []
    errors = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                errors += 1
                print(f"JSON error in line {line_number}")

    return data, errors


def save_dataset(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_values(data, field):
    return {
        item.get(field)
        for item in data
        if item.get(field) is not None
    }


def intersection(a, b):
    return len(a & b)


def count_duplicates(data, field):
    counter = Counter(
        item.get(field)
        for item in data
        if item.get(field) is not None
    )

    return sum(
        1
        for value, count in counter.items()
        if count > 1
    )


def print_distribution(title, data, field):
    print()
    print(f"{title} — {field}:")

    counter = Counter(
        item.get(field)
        for item in data
    )

    for value, count in counter.most_common():
        print(f"  {value}: {count}")


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]

        return x

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a

        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# BUILD LEAKAGE-SAFE GROUPS
# ============================================================

def build_groups(data):
    """
    Samples are connected if they share:

        1. original_email
        OR
        2. original_fragment

    Connected components are then treated as indivisible groups.

    This guarantees that neither original_email nor
    original_fragment can appear in multiple splits.
    """

    n = len(data)

    uf = UnionFind(n)

    email_owner = {}
    fragment_owner = {}

    for i, item in enumerate(data):

        original_email = item.get("original_email")
        original_fragment = item.get("original_fragment")

        if original_email:
            if original_email in email_owner:
                uf.union(i, email_owner[original_email])
            else:
                email_owner[original_email] = i

        if original_fragment:
            if original_fragment in fragment_owner:
                uf.union(i, fragment_owner[original_fragment])
            else:
                fragment_owner[original_fragment] = i

    groups = defaultdict(list)

    for i in range(n):
        root = uf.find(i)
        groups[root].append(data[i])

    return list(groups.values())


# ============================================================
# GROUP INFORMATION
# ============================================================

def group_signature(group):
    emails = {
        item.get("original_email")
        for item in group
        if item.get("original_email")
    }

    fragments = {
        item.get("original_fragment")
        for item in group
        if item.get("original_fragment")
    }

    return {
        "size": len(group),
        "emails": emails,
        "fragments": fragments,
    }


# ============================================================
# SPLIT SCORE
# ============================================================

def split_score(train_n, val_n, test_n, total):
    """
    Lower is better.

    Main objective:
        stay close to 70 / 15 / 15.

    We use squared error so large deviations
    are strongly penalized.
    """

    target_train = total * TRAIN_RATIO
    target_val = total * VAL_RATIO
    target_test = total * TEST_RATIO

    error = (
        (train_n - target_train) ** 2
        + (val_n - target_val) ** 2
        + (test_n - target_test) ** 2
    )

    return error


# ============================================================
# FIND BEST GROUP ASSIGNMENT
# ============================================================

def find_best_split(groups, seed=42, iterations=50000):

    rng = random.Random(seed)

    total = sum(len(group) for group in groups)

    target_train = total * TRAIN_RATIO
    target_val = total * VAL_RATIO
    target_test = total * TEST_RATIO

    best_assignment = None
    best_score = float("inf")

    group_indices = list(range(len(groups)))

    # --------------------------------------------------------
    # Strategy:
    #
    # We repeatedly shuffle groups and assign each group to
    # the currently best split.
    #
    # Then we keep the best solution found.
    # --------------------------------------------------------

    for iteration in range(iterations):

        order = group_indices.copy()
        rng.shuffle(order)

        # Large groups first.
        order.sort(
            key=lambda i: len(groups[i]),
            reverse=True
        )

        train = []
        val = []
        test = []

        train_n = 0
        val_n = 0
        test_n = 0

        for group_index in order:

            size = len(groups[group_index])

            candidates = []

            # Try putting group into each split.
            for split_name in ["train", "val", "test"]:

                new_train = train_n
                new_val = val_n
                new_test = test_n

                if split_name == "train":
                    new_train += size

                elif split_name == "val":
                    new_val += size

                else:
                    new_test += size

                score = split_score(
                    new_train,
                    new_val,
                    new_test,
                    total
                )

                candidates.append(
                    (score, split_name)
                )

            # Small random noise prevents always choosing
            # exactly the same split when scores are equal.
            candidates.sort(
                key=lambda x: x[0] + rng.random() * 0.01
            )

            chosen = candidates[0][1]

            if chosen == "train":
                train.append(group_index)
                train_n += size

            elif chosen == "val":
                val.append(group_index)
                val_n += size

            else:
                test.append(group_index)
                test_n += size

        score = split_score(
            train_n,
            val_n,
            test_n,
            total
        )

        # ----------------------------------------------------
        # Require all three splits to contain samples.
        # ----------------------------------------------------

        if train_n == 0 or val_n == 0 or test_n == 0:
            continue

        if score < best_score:
            best_score = score

            best_assignment = {
                "train": train.copy(),
                "val": val.copy(),
                "test": test.copy(),
                "train_n": train_n,
                "val_n": val_n,
                "test_n": test_n,
            }

    if best_assignment is None:
        raise RuntimeError(
            "Не удалось построить корректный split."
        )

    return best_assignment


# ============================================================
# MATERIALIZE SPLIT
# ============================================================

def materialize(groups, assignment):

    train = []
    val = []
    test = []

    for index in assignment["train"]:
        train.extend(groups[index])

    for index in assignment["val"]:
        val.extend(groups[index])

    for index in assignment["test"]:
        test.extend(groups[index])

    return train, val, test


# ============================================================
# LEAKAGE CHECK
# ============================================================

def check_intersection(train, val, test, field):

    train_values = get_values(train, field)
    val_values = get_values(val, field)
    test_values = get_values(test, field)

    tv = train_values & val_values
    tt = train_values & test_values
    vt = val_values & test_values

    print(f"{field}:")
    print(f"  Train ∩ Validation: {len(tv)}")
    print(f"  Train ∩ Test:       {len(tt)}")
    print(f"  Validation ∩ Test:  {len(vt)}")

    return len(tv), len(tt), len(vt)


def check_all_leakage(train, val, test):

    print_separator()
    print("4. ПРОВЕРКА DATA LEAKAGE")
    print_separator()

    fields = [
        "original_email",
        "original_fragment",
        "generated_email",
        "generated_fragment",
    ]

    leakage_found = False

    for field in fields:

        result = check_intersection(
            train,
            val,
            test,
            field
        )

        if any(value > 0 for value in result):
            leakage_found = True

        print()

    return leakage_found


# ============================================================
# ID CHECK
# ============================================================

def check_ids(train, val, test):

    print_separator()
    print("5. ПРОВЕРКА ID")
    print_separator()

    datasets = {
        "TRAIN": train,
        "VALIDATION": val,
        "TEST": test,
    }

    all_ids = []

    for name, data in datasets.items():

        ids = [
            item.get("id")
            for item in data
        ]

        duplicates = len(ids) - len(set(ids))

        print(
            f"{name}: {len(ids)} IDs, "
            f"дубликатов: {duplicates}"
        )

        all_ids.extend(ids)

    duplicate_global = (
        len(all_ids) - len(set(all_ids))
    )

    print()
    print(
        f"Дубликатов ID между split: "
        f"{duplicate_global}"
    )

    return duplicate_global == 0


# ============================================================
# INTERNAL DUPLICATES
# ============================================================

def check_internal_duplicates(name, data):

    print()
    print(name)

    for field in [
        "original_fragment",
        "generated_fragment",
        "original_email",
        "generated_email",
    ]:

        duplicates = count_duplicates(
            data,
            field
        )

        print(
            f"  Duplicate {field}: {duplicates}"
        )


# ============================================================
# DISTRIBUTION
# ============================================================

def print_all_distributions(train, val, test):

    print_separator()
    print("6. РАСПРЕДЕЛЕНИЕ")
    print_separator()

    for field in [
        "transformation",
        "language",
        "domain",
        "email_format",
        "model",
    ]:

        print()
        print("-" * 70)
        print(field.upper())
        print("-" * 70)

        print_distribution(
            "TRAIN",
            train,
            field
        )

        print_distribution(
            "VALIDATION",
            val,
            field
        )

        print_distribution(
            "TEST",
            test,
            field
        )


# ============================================================
# HTML CHECK
# ============================================================

def contains_html(text):

    if not isinstance(text, str):
        return False

    html_markers = [
        "<html",
        "<body",
        "<table",
        "<div",
        "<p>",
        "<h1",
        "<a ",
        "<td",
        "<tr",
    ]

    text_lower = text.lower()

    return any(
        marker in text_lower
        for marker in html_markers
    )


def check_html(name, data):

    declared = 0
    actual = 0

    for item in data:

        if item.get("email_format") == "html":
            declared += 1

        original = item.get(
            "original_email",
            ""
        )

        generated = item.get(
            "generated_email",
            ""
        )

        if contains_html(original) or contains_html(generated):
            actual += 1

    print(
        f"{name}: declared HTML={declared}, "
        f"actual HTML={actual}"
    )

    return declared == actual


# ============================================================
# METRICS
# ============================================================

def print_metrics(name, data):

    cosine_values = []

    for item in data:

        value = item.get("cosine_similarity")

        if isinstance(value, (int, float)):
            cosine_values.append(value)

    print()
    print(f"{name} — METRICS:")

    if not cosine_values:
        print("  Cosine: нет данных")
        return

    minimum = min(cosine_values)
    maximum = max(cosine_values)
    average = sum(cosine_values) / len(cosine_values)

    suspicious = sum(
        1
        for item in data
        if (
            isinstance(
                item.get("cosine_similarity"),
                (int, float)
            )
            and
            item.get("cosine_similarity") >= 0.95
            and
            isinstance(
                item.get(
                    "normalized_levenshtein"
                ),
                (int, float)
            )
            and
            item.get(
                "normalized_levenshtein"
            ) < 0.20
        )
    )

    print(f"  Cosine min: {minimum:.4f}")
    print(f"  Cosine max: {maximum:.4f}")
    print(f"  Cosine average: {average:.4f}")
    print(f"  Suspicious samples: {suspicious}")


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator()
    print("LEAKAGE-SAFE BALANCED DATASET SPLIT")
    print_separator()

    print()
    print(f"Input: {INPUT_FILE}")
    print()
    print("Target proportions:")
    print("  Train:      70%")
    print("  Validation: 15%")
    print("  Test:       15%")
    print()
    print(f"Random seed: {SEED}")

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    print()
    print_separator()
    print("1. LOAD")
    print_separator()

    data, json_errors = load_dataset(
        INPUT_FILE
    )

    print(f"Samples: {len(data)}")
    print(f"JSON errors: {json_errors}")

    if json_errors > 0:
        raise RuntimeError(
            "Есть ошибки JSON. Split отменён."
        )

    if not data:
        raise RuntimeError(
            "Dataset пуст."
        )

    # --------------------------------------------------------
    # REQUIRED FIELDS
    # --------------------------------------------------------

    missing_fields = []

    for index, item in enumerate(data):

        for field in REQUIRED_FIELDS:

            if field not in item:
                missing_fields.append(
                    (index, field)
                )

    if missing_fields:

        print()
        print("ОТСУТСТВУЮЩИЕ ПОЛЯ:")

        for index, field in missing_fields[:20]:
            print(
                f"  sample index {index}: {field}"
            )

        raise RuntimeError(
            "В dataset отсутствуют обязательные поля."
        )

    # --------------------------------------------------------
    # 2. BUILD GROUPS
    # --------------------------------------------------------

    print()
    print_separator()
    print("2. ПОСТРОЕНИЕ LEAKAGE-SAFE GROUPS")
    print_separator()

    print()
    print(
        "Samples будут объединены в одну группу, "
        "если они имеют:"
    )

    print("  • одинаковый original_email")
    print("  • ИЛИ одинаковый original_fragment")

    print()
    print(
        "Это гарантирует отсутствие leakage "
        "по обоим полям."
    )

    groups = build_groups(data)

    print()
    print(f"Всего samples: {len(data)}")
    print(f"Групп: {len(groups)}")

    group_sizes = sorted(
        [len(group) for group in groups],
        reverse=True
    )

    print(
        f"Largest group: {group_sizes[0]} samples"
    )

    print(
        f"Smallest group: {group_sizes[-1]} samples"
    )

    print()
    print("Размеры групп:")

    for i, size in enumerate(group_sizes, start=1):
        print(
            f"  Group {i}: {size}"
        )

    # --------------------------------------------------------
    # 3. FIND BEST SPLIT
    # --------------------------------------------------------

    print()
    print_separator()
    print("3. ПОИСК СБАЛАНСИРОВАННОГО SPLIT")
    print_separator()

    assignment = find_best_split(
        groups,
        seed=SEED,
        iterations=50000
    )

    train, val, test = materialize(
        groups,
        assignment
    )

    total = len(data)

    train_pct = len(train) / total * 100
    val_pct = len(val) / total * 100
    test_pct = len(test) / total * 100

    print()
    print(
        f"Train:      {len(train)} "
        f"({train_pct:.1f}%)"
    )

    print(
        f"Validation: {len(val)} "
        f"({val_pct:.1f}%)"
    )

    print(
        f"Test:       {len(test)} "
        f"({test_pct:.1f}%)"
    )

    print(
        f"Total:      {len(train) + len(val) + len(test)}"
    )

    print()
    print("Target:")
    print(
        f"  Train:      {total * TRAIN_RATIO:.1f}"
    )
    print(
        f"  Validation: {total * VAL_RATIO:.1f}"
    )
    print(
        f"  Test:       {total * TEST_RATIO:.1f}"
    )

    # --------------------------------------------------------
    # 4. LEAKAGE
    # --------------------------------------------------------

    leakage_found = check_all_leakage(
        train,
        val,
        test
    )

    if leakage_found:

        print_separator()
        print("✗ LEAKAGE DETECTED")
        print_separator()

        print()
        print(
            "Файлы НЕ будут сохранены."
        )

        raise RuntimeError(
            "LEAKAGE DETECTED"
        )

    print("✓ Leakage не обнаружен.")

    # --------------------------------------------------------
    # 5. IDs
    # --------------------------------------------------------

    ids_ok = check_ids(
        train,
        val,
        test
    )

    if not ids_ok:

        print()
        print(
            "✗ Ошибка ID."
        )

        raise RuntimeError(
            "Duplicate IDs detected."
        )

    print()
    print("✓ IDs корректны.")

    # --------------------------------------------------------
    # 6. INTERNAL DUPLICATES
    # --------------------------------------------------------

    print()
    print_separator()
    print("6. ПРОВЕРКА ДУБЛИКАТОВ ВНУТРИ SPLIT")
    print_separator()

    check_internal_duplicates(
        "TRAIN",
        train
    )

    check_internal_duplicates(
        "VALIDATION",
        val
    )

    check_internal_duplicates(
        "TEST",
        test
    )

    # --------------------------------------------------------
    # 7. DISTRIBUTION
    # --------------------------------------------------------

    print_all_distributions(
        train,
        val,
        test
    )

    # --------------------------------------------------------
    # 8. HTML
    # --------------------------------------------------------

    print()
    print_separator()
    print("8. HTML")
    print_separator()

    html_train_ok = check_html(
        "TRAIN",
        train
    )

    html_val_ok = check_html(
        "VALIDATION",
        val
    )

    html_test_ok = check_html(
        "TEST",
        test
    )

    if not (
        html_train_ok
        and html_val_ok
        and html_test_ok
    ):

        print()
        print(
            "⚠ В HTML-разметке есть несоответствия."
        )

    # --------------------------------------------------------
    # 9. METRICS
    # --------------------------------------------------------

    print()
    print_separator()
    print("9. METRICS")
    print_separator()

    print_metrics(
        "TRAIN",
        train
    )

    print_metrics(
        "VALIDATION",
        val
    )

    print_metrics(
        "TEST",
        test
    )

    # --------------------------------------------------------
    # 10. FINAL SAFETY CHECK
    # --------------------------------------------------------

    print()
    print_separator()
    print("10. ФИНАЛЬНАЯ ПРОВЕРКА")
    print_separator()

    all_samples = train + val + test

    if len(all_samples) != len(data):
        raise RuntimeError(
            "Количество samples после split "
            "не совпадает с исходным."
        )

    original_ids = {
        item["id"]
        for item in data
    }

    split_ids = {
        item["id"]
        for item in all_samples
    }

    if original_ids != split_ids:
        raise RuntimeError(
            "Некоторые samples потерялись "
            "или появились новые."
        )

    print(
        f"Исходных samples: {len(data)}"
    )

    print(
        f"После split:       {len(all_samples)}"
    )

    print(
        f"Train:             {len(train)}"
    )

    print(
        f"Validation:        {len(val)}"
    )

    print(
        f"Test:              {len(test)}"
    )

    print()
    print("✓ Все samples сохранены.")
    print("✓ ID корректны.")
    print("✓ original_email leakage отсутствует.")
    print("✓ original_fragment leakage отсутствует.")
    print("✓ generated_email leakage отсутствует.")
    print("✓ generated_fragment leakage отсутствует.")

    # --------------------------------------------------------
    # 11. SAVE
    # --------------------------------------------------------

    print()
    print_separator()
    print("11. СОХРАНЕНИЕ")
    print_separator()

    save_dataset(
        TRAIN_FILE,
        train
    )

    save_dataset(
        VAL_FILE,
        val
    )

    save_dataset(
        TEST_FILE,
        test
    )

    print()
    print(f"✓ Train:      {TRAIN_FILE}")
    print(f"✓ Validation: {VAL_FILE}")
    print(f"✓ Test:       {TEST_FILE}")

    # --------------------------------------------------------
    # 12. VERIFY FILES
    # --------------------------------------------------------

    print()
    print_separator()
    print("12. ПРОВЕРКА СОХРАНЁННЫХ ФАЙЛОВ")
    print_separator()

    train_check, train_errors = load_dataset(
        TRAIN_FILE
    )

    val_check, val_errors = load_dataset(
        VAL_FILE
    )

    test_check, test_errors = load_dataset(
        TEST_FILE
    )

    if (
        train_errors
        or val_errors
        or test_errors
    ):

        raise RuntimeError(
            "После сохранения обнаружены "
            "JSON errors."
        )

    if (
        len(train_check) != len(train)
        or
        len(val_check) != len(val)
        or
        len(test_check) != len(test)
    ):

        raise RuntimeError(
            "Количество samples в сохранённых "
            "файлах не совпадает."
        )

    print(
        f"Train:      {len(train_check)} samples"
    )

    print(
        f"Validation: {len(val_check)} samples"
    )

    print(
        f"Test:       {len(test_check)} samples"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print_separator()
    print("✓ SPLIT УСПЕШНО ЗАВЕРШЁН")
    print_separator()

    print()
    print("Исходный dataset НЕ изменён:")
    print(
        f"  {INPUT_FILE}"
    )

    print()
    print("Созданы:")

    print(
        f"  {TRAIN_FILE}"
    )

    print(
        f"  {VAL_FILE}"
    )

    print(
        f"  {TEST_FILE}"
    )

    print()
    print(
        "Теперь можно запускать check_split.py."
    )


if __name__ == "__main__":
    main()