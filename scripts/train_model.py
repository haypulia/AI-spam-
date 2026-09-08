import sys

import _bootstrap  # noqa: F401
from email_ai_detector.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train"] + sys.argv[1:]))
