"""Fail fast when the project is run outside its documented environment."""

from importlib.metadata import version
import sys


EXPECTED_PYTHON = (3, 11, 16)
EXPECTED_PACKAGES = {
    "numpy": "2.4.6",
    "pandas": "3.0.6",
    "scikit-learn": "1.9.1",
    "xgboost": "3.2.0",
}


def main() -> int:
    actual_python = sys.version_info[:3]
    if actual_python != EXPECTED_PYTHON:
        print(f"Python mismatch: expected {EXPECTED_PYTHON}, found {actual_python}", file=sys.stderr)
        return 1
    mismatches = []
    for distribution, expected in EXPECTED_PACKAGES.items():
        actual = version(distribution)
        if actual != expected:
            mismatches.append(f"{distribution}: expected {expected}, found {actual}")
    if mismatches:
        print("Dependency version mismatch:\n- " + "\n- ".join(mismatches), file=sys.stderr)
        return 1
    print(f"Environment OK: Python {'.'.join(map(str, actual_python))}; {len(EXPECTED_PACKAGES)} direct dependencies match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
