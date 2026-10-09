from email_ai_detector.data.splitting import audit_split, prepare_split


def record(identifier, text, **kwargs):
    return dict(id=identifier, text=text, label="human", ai_binary=0,
                ai_elements=[], ai_char_intervals=[], **kwargs)


def examples():
    return [
        record("a", "Invoice 123 please pay at https://one.example", topic_id="invoice", split="train"),
        record("b", "Invoice 456 please pay at https://two.example", topic_id="other", split="test"),
        record("c", "A completely different message concerning office maintenance", split="validation"),
        record("d", "Please review the quarterly financial statement before Friday", split="test"),
        record("e", "An unrelated email about a family birthday party", split="train"),
    ]


def test_audit_detects_normalized_duplicates_across_splits():
    report = audit_split(examples())
    assert report["cross_split_groups"] == 1


def test_split_deduplicates_and_is_reproducible():
    rows = examples()
    output, report = prepare_split(rows)
    reverse, reverse_report = prepare_split(list(reversed(rows)))
    assert output == reverse
    assert report == reverse_report
    assert report["removed"] == [{"id": "b", "representative": "a"}]
    assert report["audit"]["cross_split_groups"] == 0
    assert set(row["split"] for row in output) == {"train", "validation", "test"}
    assert rows[1]["split"] == "test"


def test_seed_and_template_links_are_transitive():
    rows = examples()
    rows[0]["template_id"] = "template"
    rows[2]["template_id"] = "template"
    rows[3]["seed_id"] = "c"
    report = audit_split(rows)
    assert report["cross_split_groups"] == 1
    assert set(report["cross_split_examples"][0]) == {"a", "b", "c", "d"}


def test_conflicting_labels_are_kept_in_same_split():
    rows = examples()
    rows[1].update(label="ai", ai_binary=1, ai_elements=["body"])
    output, report = prepare_split(rows)
    by_id = {row["id"]: row for row in output}
    assert report["removed"] == []
    assert by_id["a"]["split"] == by_id["b"]["split"]


def test_seed_chains_stay_together():
    rows = examples()
    rows[2]["seed_id"] = "a"
    rows[3]["seed_id"] = "c"
    report = audit_split(rows)
    assert set(report["cross_split_examples"][0]) == {"a", "b", "c", "d"}


def test_near_duplicates_are_detected_without_exact_match():
    text = " ".join("token" + chr(97 + i // 26) + chr(97 + i % 26) for i in range(40))
    rows = examples()
    rows[0]["text"] = text
    rows[1]["text"] = text + " additional"
    assert audit_split(rows)["cross_split_groups"] == 1


def test_invalid_inputs_and_insufficient_groups_are_rejected():
    for rows in ([examples()[0]], [examples()[0], examples()[0]]):
        try:
            prepare_split(rows)
        except ValueError:
            pass
        else:
            raise AssertionError("некорректный датасет принят")
    rows = examples()
    rows[0]["ai_char_intervals"] = [(0, 10000)]
    try:
        prepare_split(rows)
    except ValueError:
        pass
    else:
        raise AssertionError("неверные интервалы приняты")
