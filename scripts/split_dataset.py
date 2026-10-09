"""Проверка и разбиение основного датасета; исходный файл не изменяется."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import _bootstrap  # noqa: F401
from email_ai_detector.data.splitting import audit_split, prepare_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/datasets/ai_assisted_spam_dataset.zip"))
    parser.add_argument("--output", type=Path, help="новый пустой каталог для emails.jsonl и отчёта")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.input.suffix.lower() == ".zip":
        with zipfile.ZipFile(args.input) as bundle:
            names = [name for name in bundle.namelist() if name.endswith("/emails.jsonl") or name == "emails.jsonl"]
            if len(names) != 1:
                parser.error("архив должен содержать один emails.jsonl")
            content = bundle.read(names[0]).decode("utf-8")
    else:
        content = args.input.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if args.output is None:
        report = audit_split(rows, args.threshold)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(bool(report["cross_split_groups"] or report["invalid_split_ids"]))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("выходной каталог должен быть пустым")
    output, report = prepare_split(rows, args.threshold, args.seed)
    report["input_file"] = str(args.input)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "emails.jsonl").open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
