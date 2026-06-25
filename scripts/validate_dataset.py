"""Validate the JSONL instruction dataset."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Any, Dict, List

from app.inference.formatting import REQUIRED_FIELDS, missing_required_fields, parse_json_output
from scripts.common import read_jsonl

REQUIRED_ROW_FIELDS = ["instruction", "input", "output"]


def validate_record(record: Dict[str, Any], row_number: int) -> List[str]:
    """Return validation errors for one dataset row."""
    errors: List[str] = []

    for field in REQUIRED_ROW_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"row {row_number}: '{field}' must be a non-empty string")

    output = record.get("output")
    if isinstance(output, str) and output.strip():
        try:
            parsed = parse_json_output(output)
        except ValueError as exc:
            errors.append(f"row {row_number}: output must be valid JSON object: {exc}")
        else:
            missing = missing_required_fields(parsed, REQUIRED_FIELDS)
            if missing:
                errors.append(f"row {row_number}: output missing required fields: {missing}")

    return errors


def validate_dataset(path: str | Path) -> Dict[str, Any]:
    """Validate all rows in a dataset file."""
    records = read_jsonl(path)
    errors: List[str] = []
    for idx, record in enumerate(records, start=1):
        errors.extend(validate_record(record, idx))

    return {
        "path": str(path),
        "rows": len(records),
        "valid": len(errors) == 0,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate support dataset JSONL schema.")
    parser.add_argument("--data", default="data/raw/sample_support_dataset.jsonl", help="Path to JSONL dataset")
    parser.add_argument("--report", default=None, help="Optional path to save validation report JSON")
    args = parser.parse_args()

    report = validate_dataset(args.data)
    print(json.dumps(report, indent=2))

    if args.report:
        output_path = Path(args.report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
