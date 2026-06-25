"""Prompt formatting and JSON parsing utilities."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

REQUIRED_FIELDS = ["category", "priority", "suggested_reply", "next_action"]

DEFAULT_INSTRUCTION = (
    "Classify the customer support message and return only valid JSON with exactly these fields: "
    "category, priority, suggested_reply, next_action. Do not include markdown or extra text."
)


def format_prompt(
    customer_message: str,
    instruction: str = DEFAULT_INSTRUCTION,
    include_response_marker: bool = True,
) -> str:
    """Create the instruction prompt used for training and inference."""
    if not customer_message or not customer_message.strip():
        raise ValueError("customer_message must be a non-empty string.")

    prompt = (
        "### Instruction:\n"
        f"{instruction.strip()}\n\n"
        "### Customer Message:\n"
        f"{customer_message.strip()}\n"
    )
    if include_response_marker:
        prompt += "\n### Response:\n"
    return prompt


def format_training_text(instruction: str, customer_message: str, output: str, eos_token: str = "") -> str:
    """Format a full supervised fine-tuning example."""
    prompt = format_prompt(customer_message=customer_message, instruction=instruction)
    return f"{prompt}{output.strip()}{eos_token}"


def _extract_first_json_object(text: str) -> Optional[str]:
    """Return the first balanced JSON object substring from text."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        char = text[idx]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def parse_json_output(text: str) -> Dict[str, Any]:
    """Parse the first JSON object from a model output.

    Raises:
        ValueError: if no valid JSON object can be parsed.
    """
    if not isinstance(text, str):
        raise ValueError("Model output must be a string.")

    candidate = _extract_first_json_object(text.strip())
    if candidate is None:
        raise ValueError("No JSON object found in model output.")

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON output: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON must be an object.")
    return parsed


def missing_required_fields(parsed: Dict[str, Any], required_fields: Iterable[str] = REQUIRED_FIELDS) -> list[str]:
    """Return required fields that are missing or empty."""
    missing = []
    for field in required_fields:
        value = parsed.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return missing


def normalize_structured_response(parsed: Dict[str, Any]) -> Dict[str, str]:
    """Normalize parsed JSON into the exact response schema.

    Missing values are filled with a safe fallback string so the API can return
    a predictable schema while still exposing the raw model output.
    """
    return {
        "category": str(parsed.get("category", "Unknown")).strip() or "Unknown",
        "priority": str(parsed.get("priority", "Medium")).strip() or "Medium",
        "suggested_reply": str(parsed.get("suggested_reply", "Unable to generate a reply.")).strip()
        or "Unable to generate a reply.",
        "next_action": str(parsed.get("next_action", "Review manually")).strip() or "Review manually",
    }
