"""Validate aggregate evaluation/status evidence without running restricted data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=ROOT / "results" / "current_evaluation.json")
    args = parser.parse_args(argv)
    try:
        record = json.loads(args.path.read_text(encoding="utf-8"))
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from evaluation import validate_result
        validate_result(record)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f"Result validation failed: {error}", file=sys.stderr)
        return 1
    print(f"Result schema verified; status={record['status']}. Schema validity is not empirical completion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
