"""Simple structured-output evaluation metrics."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from app.inference.formatting import REQUIRED_FIELDS, missing_required_fields, parse_json_output


def safe_parse(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON output and return None if parsing fails."""
    try:
        return parse_json_output(text)
    except ValueError:
        return None


def valid_json_rate(outputs: Iterable[str]) -> float:
    """Fraction of outputs that contain a valid JSON object."""
    outputs_list = list(outputs)
    if not outputs_list:
        return 0.0
    valid = sum(1 for output in outputs_list if safe_parse(output) is not None)
    return valid / len(outputs_list)


def field_completeness(outputs: Iterable[str]) -> float:
    """Average fraction of required fields present and non-empty."""
    outputs_list = list(outputs)
    if not outputs_list:
        return 0.0

    total = 0.0
    for output in outputs_list:
        parsed = safe_parse(output)
        if parsed is None:
            continue
        missing = missing_required_fields(parsed, REQUIRED_FIELDS)
        total += (len(REQUIRED_FIELDS) - len(missing)) / len(REQUIRED_FIELDS)
    return total / len(outputs_list)


def _category_from_expected_output(expected_output: str) -> Optional[str]:
    parsed = safe_parse(expected_output)
    if parsed is None:
        return None
    value = parsed.get("category")
    return str(value).strip().lower() if value else None


def category_match_rate(outputs: Iterable[str], expected_outputs: Iterable[str]) -> float:
    """Category exact-match rate when reference categories are available."""
    pairs = list(zip(outputs, expected_outputs))
    if not pairs:
        return 0.0

    scored = 0
    matched = 0
    for output, expected in pairs:
        expected_category = _category_from_expected_output(expected)
        parsed_output = safe_parse(output)
        if expected_category is None or parsed_output is None:
            continue
        predicted_category = str(parsed_output.get("category", "")).strip().lower()
        scored += 1
        if predicted_category == expected_category:
            matched += 1

    return matched / scored if scored else 0.0


def compute_metrics(outputs: List[str], expected_outputs: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute all available metrics."""
    metrics = {
        "valid_json_rate": round(valid_json_rate(outputs), 4),
        "field_completeness": round(field_completeness(outputs), 4),
    }
    if expected_outputs is not None:
        metrics["category_match_rate"] = round(category_match_rate(outputs, expected_outputs), 4)
    return metrics


def write_evaluation_report(path: str, report: Dict[str, Any]) -> None:
    """Save an evaluation report as JSON."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
