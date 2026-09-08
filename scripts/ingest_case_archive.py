import argparse
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
from email_ai_detector.config import get_settings
from email_ai_detector.data.eml import load_email_from_file
from email_ai_detector.utils import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Распаковка архива с письмами кейса в data/raw")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    target = ensure_dir(args.output or settings.case_samples_dir)

    with zipfile.ZipFile(args.archive) as bundle:
        bundle.extractall(target)

    index = []
    for path in sorted(target.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        email = load_email_from_file(path)
        if email is None:
            continue
        index.append(
            {
                "path": str(path.relative_to(target)),
                "group": path.parent.name,
                "subject": email.subject,
                "sender": email.sender,
                "html_length": len(email.html),
                "images": len(email.images),
            }
        )

    index_path = args.index or (settings.interim_dir / "case_samples_index.json")
    save_json(index, index_path)
    print("писем разобрано: %d" % len(index))
    print("индекс: %s" % index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
