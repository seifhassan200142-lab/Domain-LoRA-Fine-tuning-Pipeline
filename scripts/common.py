"""Shared script utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read JSONL records."""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write records to JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    """Set Python random seed."""
    random.seed(seed)
