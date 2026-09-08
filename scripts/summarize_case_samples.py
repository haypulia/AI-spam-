import argparse
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
from email_ai_detector.config import get_settings
from email_ai_detector.utils import load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Сводка по результатам анализа писем кейса")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--category-threshold", type=float, default=0.5)
    return parser.parse_args()


def group_of(report, root: Path) -> str:
    path = Path(report.get("path") or report.get("file", "")).resolve()
    try:
        relative = path.relative_to(Path(root).resolve())
    except ValueError:
        relative = path
    parts = relative.parts
    return parts[0] if len(parts) > 1 else "root"


def main() -> int:
    args = parse_args()
    settings = get_settings()
    source = args.input or (settings.processed_dir / "case_samples_analysis.json")
    reports = load_json(source)
    root = settings.case_samples_dir

    grouped = defaultdict(list)
    for report in reports:
        grouped[group_of(report, root)].append(report)

    try:
        source_name = str(Path(source).resolve().relative_to(Path(settings.project_root).resolve()))
    except ValueError:
        source_name = Path(source).name

    summary = {"source": source_name, "total": len(reports), "groups": {}}

    for name, items in sorted(grouped.items()):
        scores = [item["analysis"]["ai_score"] for item in items]
        verdicts = Counter(item["analysis"]["verdict"] for item in items)
        categories = Counter()
        for item in items:
            for category, value in item["analysis"]["categories"].items():
                if value >= args.category_threshold:
                    categories[category] += 1
        summary["groups"][name] = {
            "count": len(items),
            "mean_ai_score": sum(scores) / len(scores),
            "median_ai_score": sorted(scores)[len(scores) // 2],
            "min_ai_score": min(scores),
            "max_ai_score": max(scores),
            "verdicts": dict(verdicts),
            "reported_zones": dict(categories.most_common()),
        }

    destination = args.output or (settings.reports_dir / "case_samples_summary.json")
    save_json(summary, destination)
    print("сводка: %s" % destination)
    for name, payload in summary["groups"].items():
        print("%s: писем %d, средний индекс %.3f" % (name, payload["count"], payload["mean_ai_score"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
