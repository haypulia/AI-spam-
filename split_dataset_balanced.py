import json
import random
from collections import Counter, defaultdict
from pathlib import Path


# config

INPUT_FILE = Path(
    "email_ai_detector/generated_dataset/dataset_final.jsonl"
)

TRAIN_FILE = Path(
    "email_ai_detector/generated_dataset/dataset_train.jsonl"
)

VAL_FILE = Path(
    "email_ai_detector/generated_dataset/dataset_val.jsonl"
)

TEST_FILE = Path(
    "email_ai_detector/generated_dataset/dataset_test.jsonl"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

GROUP_KEYS = [
    "source",
    "target",
]

BALANCE_FIELDS = [
    "model",
    "language",
    "domain",
    "transformation",
    "format",
]

REQUIRED_FIELDS = {
    "id",
    "source",
    "target",
    "model",
    "language",
    "domain",
    "transformation",
    "format",
    "cosine_similarity",
    "normalized_levenshtein_distance",
    "valid",
}


def load_dataset(path):
    data = []
    json_errors = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError:
                json_errors += 1
                print(f"JSON error at line {line_number}")

    return data, json_errors


def validate_required_fields(data):
    missing = []

    for index, item in enumerate(data):
        missing_fields = REQUIRED_FIELDS - set(item.keys())

        if missing_fields:
            missing.append(
                {
                    "index": index,
                    "missing": sorted(missing_fields),
                }
            )

    return missing


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

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


def build_groups(data):
    uf = UnionFind(len(data))
    value_to_index = {}

    for index, item in enumerate(data):
        for key in GROUP_KEYS:
            value = item.get(key)

            if not value:
                continue

            lookup_key = (key, value)

            if lookup_key in value_to_index:
                uf.union(index, value_to_index[lookup_key])
            else:
                value_to_index[lookup_key] = index

    groups = defaultdict(list)

    for index in range(len(data)):
        root = uf.find(index)
        groups[root].append(index)

    return list(groups.values())


def calculate_global_distributions(data):
    distributions = {}

    for field in BALANCE_FIELDS:
        distributions[field] = Counter(
            item.get(field) for item in data
        )

    return distributions


def calculate_group_statistics(groups, data):
    stats = []

    for group_id, indices in enumerate(groups):
        field_counts = {}

        for field in BALANCE_FIELDS:
            field_counts[field] = Counter(
                data[index].get(field)
                for index in indices
            )

        stats.append(
            {
                "id": group_id,
                "indices": indices,
                "size": len(indices),
                "field_counts": field_counts,
            }
        )

    return stats


def target_sizes(total):
    train = int(round(total * TRAIN_RATIO))
    val = int(round(total * VAL_RATIO))
    test = total - train - val

    return train, val, test


def distribution_distance(actual, expected):
    distance = 0.0

    for key, expected_value in expected.items():
        actual_value = actual.get(key, 0)
        distance += abs(actual_value - expected_value)

    return distance


def split_score(
    split_counts,
    split_field_counts,
    target_counts,
    global_distributions,
):
    score = 0.0

    split_names = ["train", "val", "test"]

    for split_name in split_names:
        actual_size = split_counts[split_name]
        expected_size = target_counts[split_name]

        score += abs(actual_size - expected_size) * 1000

    for split_name in split_names:
        split_size = split_counts[split_name]

        if split_size == 0:
            score += 1000000
            continue

        for field in BALANCE_FIELDS:
            global_counts = global_distributions[field]

            expected = {
                value: count * split_size / sum(global_counts.values())
                for value, count in global_counts.items()
            }

            actual = split_field_counts[split_name][field]

            score += distribution_distance(actual, expected)

    return score


def add_group_to_split(
    group,
    split_name,
    split_indices,
    split_counts,
    split_field_counts,
    data,
):
    split_indices[split_name].extend(group["indices"])
    split_counts[split_name] += group["size"]

    for field in BALANCE_FIELDS:
        for value, count in group["field_counts"][field].items():
            split_field_counts[split_name][field][value] += count


def remove_group_from_split(
    group,
    split_name,
    split_indices,
    split_counts,
    split_field_counts,
    data,
):
    group_set = set(group["indices"])

    split_indices[split_name] = [
        index
        for index in split_indices[split_name]
        if index not in group_set
    ]

    split_counts[split_name] -= group["size"]

    for field in BALANCE_FIELDS:
        for value, count in group["field_counts"][field].items():
            split_field_counts[split_name][field][value] -= count

            if split_field_counts[split_name][field][value] <= 0:
                del split_field_counts[split_name][field][value]


def initialize_split_state():
    return (
        {
            "train": [],
            "val": [],
            "test": [],
        },
        {
            "train": 0,
            "val": 0,
            "test": 0,
        },
        {
            "train": {
                field: Counter()
                for field in BALANCE_FIELDS
            },
            "val": {
                field: Counter()
                for field in BALANCE_FIELDS
            },
            "test": {
                field: Counter()
                for field in BALANCE_FIELDS
            },
        },
    )


def choose_split_for_group(
    group,
    split_counts,
    target_counts,
    split_field_counts,
    global_distributions,
):
    candidates = []

    for split_name in ["train", "val", "test"]:
        new_size = split_counts[split_name] + group["size"]
        target_size = target_counts[split_name]

        size_error = abs(new_size - target_size)

        balance_error = 0.0

        for field in BALANCE_FIELDS:
            global_counts = global_distributions[field]

            expected = {
                value: count * new_size / sum(global_counts.values())
                for value, count in global_counts.items()
            }

            current = split_field_counts[split_name][field].copy()

            for value, count in group["field_counts"][field].items():
                current[value] += count

            balance_error += distribution_distance(
                current,
                expected,
            )

        overflow = max(0, new_size - target_size)

        score = (
            overflow * 10000
            + size_error * 100
            + balance_error
        )

        candidates.append((score, split_name))

    candidates.sort()

    return candidates[0][1]


def greedy_split(groups, data, seed):
    rng = random.Random(seed)

    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)

    shuffled_groups.sort(
        key=lambda group: (
            -group["size"],
            rng.random(),
        )
    )

    total = len(data)

    train_target, val_target, test_target = target_sizes(total)

    target_counts = {
        "train": train_target,
        "val": val_target,
        "test": test_target,
    }

    (
        split_indices,
        split_counts,
        split_field_counts,
    ) = initialize_split_state()

    global_distributions = calculate_global_distributions(data)

    for group in shuffled_groups:
        split_name = choose_split_for_group(
            group,
            split_counts,
            target_counts,
            split_field_counts,
            global_distributions,
        )

        add_group_to_split(
            group,
            split_name,
            split_indices,
            split_counts,
            split_field_counts,
            data,
        )

    return (
        split_indices,
        split_counts,
        split_field_counts,
    )


def rebalance_sizes(
    groups,
    data,
    split_indices,
    split_counts,
    split_field_counts,
):
    target_train, target_val, target_test = target_sizes(len(data))

    targets = {
        "train": target_train,
        "val": target_val,
        "test": target_test,
    }

    group_by_index = {}

    for group in groups:
        for index in group["indices"]:
            group_by_index[index] = group

    max_iterations = 10000

    for _ in range(max_iterations):
        changed = False

        oversized = [
            name
            for name in ["train", "val", "test"]
            if split_counts[name] > targets[name]
        ]

        undersized = [
            name
            for name in ["train", "val", "test"]
            if split_counts[name] < targets[name]
        ]

        if not oversized or not undersized:
            break

        best_move = None

        current_size_error = sum(
            abs(split_counts[name] - targets[name])
            for name in ["train", "val", "test"]
        )

        for source_name in oversized:
            source_group_ids = set()

            for index in split_indices[source_name]:
                source_group_ids.add(
                    group_by_index[index]["id"]
                )

            candidate_groups = [
                group
                for group in groups
                if group["id"] in source_group_ids
            ]

            for group in candidate_groups:
                for destination_name in undersized:
                    new_source_size = (
                        split_counts[source_name]
                        - group["size"]
                    )

                    new_destination_size = (
                        split_counts[destination_name]
                        + group["size"]
                    )

                    new_error = (
                        abs(
                            new_source_size
                            - targets[source_name]
                        )
                        + abs(
                            new_destination_size
                            - targets[destination_name]
                        )
                    )

                    if new_error >= current_size_error:
                        continue

                    move = (
                        new_error,
                        group["size"],
                        source_name,
                        destination_name,
                        group,
                    )

                    if best_move is None or move[:2] < best_move[:2]:
                        best_move = move

        if best_move is None:
            break

        _, _, source_name, destination_name, group = best_move

        remove_group_from_split(
            group,
            source_name,
            split_indices,
            split_counts,
            split_field_counts,
            data,
        )

        add_group_to_split(
            group,
            destination_name,
            split_indices,
            split_counts,
            split_field_counts,
            data,
        )

        changed = True

        if not changed:
            break

    return (
        split_indices,
        split_counts,
        split_field_counts,
    )


def build_best_split(data, groups, attempts=1000):
    group_stats = calculate_group_statistics(
        groups,
        data,
    )

    best_result = None
    best_score = float("inf")

    print(f"Trying {attempts} split variants...")

    for attempt in range(attempts):
        (
            split_indices,
            split_counts,
            split_field_counts,
        ) = greedy_split(
            group_stats,
            data,
            SEED + attempt,
        )

        (
            split_indices,
            split_counts,
            split_field_counts,
        ) = rebalance_sizes(
            group_stats,
            data,
            split_indices,
            split_counts,
            split_field_counts,
        )

        target_train, target_val, target_test = target_sizes(
            len(data)
        )

        target_counts = {
            "train": target_train,
            "val": target_val,
            "test": target_test,
        }

        global_distributions = calculate_global_distributions(
            data
        )

        score = split_score(
            split_counts,
            split_field_counts,
            target_counts,
            global_distributions,
        )

        if score < best_score:
            best_score = score

            best_result = (
                split_indices,
                split_counts,
                split_field_counts,
            )

    return best_result, best_score


def check_leakage(data, split_indices):
    print()
    print("=" * 70)
    print("LEAKAGE CHECK")
    print("=" * 70)

    leakage_found = False

    for key in GROUP_KEYS:
        print()
        print(f"{key}:")

        values = {}

        for split_name in ["train", "val", "test"]:
            values[split_name] = set(
                data[index].get(key)
                for index in split_indices[split_name]
                if data[index].get(key) is not None
            )

        train_val = values["train"] & values["val"]
        train_test = values["train"] & values["test"]
        val_test = values["val"] & values["test"]

        print(f"  Train ∩ Validation: {len(train_val)}")
        print(f"  Train ∩ Test: {len(train_test)}")
        print(f"  Validation ∩ Test: {len(val_test)}")

        if train_val or train_test or val_test:
            leakage_found = True

    if leakage_found:
        print()
        print("✗ LEAKAGE DETECTED")
    else:
        print()
        print("✓ No leakage detected.")

    return not leakage_found


def check_ids(data, split_indices):
    print()
    print("=" * 70)
    print("ID CHECK")
    print("=" * 70)

    all_ids = set()
    duplicate_ids = 0

    for split_name in ["train", "val", "test"]:
        ids = [
            data[index].get("id")
            for index in split_indices[split_name]
        ]

        unique_ids = set(ids)
        duplicates = len(ids) - len(unique_ids)

        print(
            f"{split_name.capitalize():12}"
            f"{len(ids):4} IDs | duplicates: {duplicates}"
        )

        duplicate_ids += duplicates
        all_ids.update(unique_ids)

    total_ids = sum(
        len(split_indices[name])
        for name in ["train", "val", "test"]
    )

    cross_split_duplicates = total_ids - len(all_ids)

    print()
    print(
        f"Duplicates between splits: "
        f"{cross_split_duplicates}"
    )

    return duplicate_ids == 0 and cross_split_duplicates == 0


def print_distribution(
    data,
    indices,
    split_name,
):
    print()
    print("-" * 70)
    print(split_name.upper())
    print("-" * 70)

    print(f"Samples: {len(indices)}")

    for field in BALANCE_FIELDS:
        counts = Counter(
            data[index].get(field)
            for index in indices
        )

        print()
        print(f"{field}:")

        for value, count in counts.most_common():
            print(f"  {value}: {count}")


def print_all_distributions(data, split_indices):
    print()
    print("=" * 70)
    print("FINAL DISTRIBUTIONS")
    print("=" * 70)

    print_distribution(
        data,
        split_indices["train"],
        "Train",
    )

    print_distribution(
        data,
        split_indices["val"],
        "Validation",
    )

    print_distribution(
        data,
        split_indices["test"],
        "Test",
    )


def validate_split_sizes(split_counts, total):
    expected = target_sizes(total)

    actual = (
        split_counts["train"],
        split_counts["val"],
        split_counts["test"],
    )

    print()
    print("=" * 70)
    print("SIZE CHECK")
    print("=" * 70)

    print(
        f"Expected: Train {expected[0]}, "
        f"Validation {expected[1]}, "
        f"Test {expected[2]}"
    )

    print(
        f"Actual:   Train {actual[0]}, "
        f"Validation {actual[1]}, "
        f"Test {actual[2]}"
    )

    exact = actual == expected

    if exact:
        print()
        print("✓ Split sizes are exact.")
    else:
        print()
        print("⚠ Exact target sizes were not possible with group constraints.")

    return exact


def validate_total(split_counts, total):
    actual_total = sum(split_counts.values())

    print()
    print(
        f"Total samples in splits: "
        f"{actual_total} / {total}"
    )

    if actual_total != total:
        print("✗ Some samples are missing or duplicated.")
        return False

    print("✓ All samples are present exactly once.")
    return True


def save_jsonl(path, data, indices):
    with open(path, "w", encoding="utf-8") as f:
        for index in indices:
            f.write(
                json.dumps(
                    data[index],
                    ensure_ascii=False,
                )
                + "\n"
            )


def validate_saved_file(path):
    if not path.exists():
        return False

    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    return count > 0


def main():
    print("=" * 70)
    print("LEAKAGE-SAFE DATASET SPLIT")
    print("=" * 70)

    print()
    print(f"Input: {INPUT_FILE}")

    print()
    print("Grouping keys:")

    for key in GROUP_KEYS:
        print(f"  {key}")

    print()
    print("Balance fields:")

    for field in BALANCE_FIELDS:
        print(f"  {field}")

    print()
    print("Target proportions:")
    print(f"  Train:      {TRAIN_RATIO:.0%}")
    print(f"  Validation: {VAL_RATIO:.0%}")
    print(f"  Test:       {TEST_RATIO:.0%}")

    print()
    print(f"Random seed: {SEED}")

    print()
    print("=" * 70)
    print("1. LOAD")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print()
        print(f"✗ Input file not found: {INPUT_FILE}")
        return 1

    data, json_errors = load_dataset(INPUT_FILE)

    print(f"Samples: {len(data)}")
    print(f"JSON errors: {json_errors}")

    missing_fields = validate_required_fields(data)

    if missing_fields:
        print()
        print("✗ Required fields are missing.")

        for error in missing_fields[:10]:
            print(
                f"  Record {error['index']}: "
                f"{', '.join(error['missing'])}"
            )

        return 1

    print("Required fields: OK")

    train_target, val_target, test_target = target_sizes(
        len(data)
    )

    print()
    print("Target sample counts:")
    print(f"  Train       {train_target}")
    print(f"  Validation  {val_target}")
    print(f"  Test        {test_target}")

    print()
    print("=" * 70)
    print("2. BUILD LEAKAGE-SAFE GROUPS")
    print("=" * 70)

    groups = build_groups(data)

    print()
    print("=" * 70)
    print("GROUP STATISTICS")
    print("=" * 70)

    group_sizes = Counter(len(group) for group in groups)

    print(f"Leakage-safe groups: {len(groups)}")
    print(f"Largest group: {max(group_sizes)}")
    print(f"Smallest group: {min(group_sizes)}")

    average_size = len(data) / len(groups)

    print(f"Average group size: {average_size:.2f}")

    print()
    print("Group size distribution:")

    for size, count in sorted(group_sizes.items()):
        print(
            f"  {size} samples: {count} groups"
        )

    print()
    print("=" * 70)
    print("3. SEARCH FOR BALANCED SPLIT")
    print("=" * 70)

    result, best_score = build_best_split(
        data,
        groups,
        attempts=1000,
    )

    if result is None:
        print()
        print("✗ Failed to build split.")
        return 1

    (
        split_indices,
        split_counts,
        split_field_counts,
    ) = result

    print()
    print(f"Best score: {best_score:.4f}")

    print()
    print("Result:")

    for name, label in [
        ("train", "Train"),
        ("val", "Validation"),
        ("test", "Test"),
    ]:
        count = split_counts[name]
        percentage = count / len(data) * 100

        print(
            f"  {label:12}"
            f"{count:4} ({percentage:.1f}%)"
        )

    print(
        f"  {'Total':12}"
        f"{sum(split_counts.values()):4}"
    )

    print()
    print("=" * 70)
    print("4. VALIDATION BEFORE SAVE")
    print("=" * 70)

    sizes_ok = validate_split_sizes(
        split_counts,
        len(data),
    )

    total_ok = validate_total(
        split_counts,
        len(data),
    )

    leakage_ok = check_leakage(
        data,
        split_indices,
    )

    ids_ok = check_ids(
        data,
        split_indices,
    )

    print_all_distributions(
        data,
        split_indices,
    )

    if not total_ok:
        print()
        print("✗ Validation failed: total sample count mismatch.")
        return 1

    if not leakage_ok:
        print()
        print("✗ Validation failed: leakage detected.")
        return 1

    if not ids_ok:
        print()
        print("✗ Validation failed: duplicate IDs detected.")
        return 1

    if not sizes_ok:
        print()
        print(
            "✗ Validation failed: "
            "train/validation/test sizes are not exact."
        )
        print()
        print(
            "The dataset contains leakage groups that prevent "
            "the requested 70/15/15 split."
        )
        return 1

    print()
    print("✓ All pre-save checks passed.")

    print()
    print("=" * 70)
    print("5. SAVE")
    print("=" * 70)

    save_jsonl(
        TRAIN_FILE,
        data,
        split_indices["train"],
    )

    save_jsonl(
        VAL_FILE,
        data,
        split_indices["val"],
    )

    save_jsonl(
        TEST_FILE,
        data,
        split_indices["test"],
    )

    print(f"✓ Train:      {TRAIN_FILE}")
    print(f"✓ Validation: {VAL_FILE}")
    print(f"✓ Test:       {TEST_FILE}")

    saved_ok = (
        validate_saved_file(TRAIN_FILE)
        and validate_saved_file(VAL_FILE)
        and validate_saved_file(TEST_FILE)
    )

    if not saved_ok:
        print()
        print("✗ One or more output files are empty or missing.")
        return 1

    print()
    print("=" * 70)
    print("✓ SPLIT УСПЕШНО ЗАВЕРШЁН")
    print("=" * 70)

    print()
    print("Исходный dataset НЕ изменён:")
    print(f"  {INPUT_FILE}")

    print()
    print("Созданы:")
    print(f"  {TRAIN_FILE}")
    print(f"  {VAL_FILE}")
    print(f"  {TEST_FILE}")

    print()
    print("Финальные размеры:")
    print(f"  Train       {split_counts['train']}")
    print(f"  Validation  {split_counts['val']}")
    print(f"  Test        {split_counts['test']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
