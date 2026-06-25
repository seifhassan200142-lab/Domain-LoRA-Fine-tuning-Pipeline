"""Create train/validation JSONL splits."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import random
from typing import Any, Dict, List, Tuple

from scripts.common import read_jsonl, write_jsonl


def split_records(
    records: List[Dict[str, Any]],
    validation_size: float = 0.25,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records into train and validation sets."""
    if not records:
        raise ValueError("Cannot split an empty dataset.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    items = list(records)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(items)

    validation_count = max(1, int(round(len(items) * validation_size)))
    validation_records = items[:validation_count]
    train_records = items[validation_count:]

    if not train_records:
        raise ValueError("Split produced no training rows. Add more records or reduce validation_size.")

    return train_records, validation_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset into train and validation JSONL files.")
    parser.add_argument("--data", default="data/raw/sample_support_dataset.jsonl")
    parser.add_argument("--train-out", default="data/splits/train.jsonl")
    parser.add_argument("--validation-out", default="data/splits/validation.jsonl")
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.data)
    train_records, validation_records = split_records(
        records,
        validation_size=args.validation_size,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    write_jsonl(args.train_out, train_records)
    write_jsonl(args.validation_out, validation_records)

    summary = {
        "source": args.data,
        "train_path": args.train_out,
        "validation_path": args.validation_out,
        "train_rows": len(train_records),
        "validation_rows": len(validation_records),
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
