"""Run the base model before fine-tuning and save outputs."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from app.inference.formatting import format_prompt
from scripts.common import load_yaml, read_jsonl, write_jsonl
from scripts.model_utils import generate_text, load_tokenizer_and_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline outputs from the base model.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--output", default="outputs/baseline_outputs.jsonl")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_cfg = config["model"]
    inference_cfg = config.get("inference", {})
    validation_file = args.validation_file or config["training"]["validation_file"]

    records = read_jsonl(validation_file)[: args.limit]
    tokenizer, model = load_tokenizer_and_model(
        model_cfg["base_model_name"],
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )

    outputs = []
    for record in records:
        prompt = format_prompt(record["input"], instruction=record["instruction"])
        raw_output = generate_text(
            tokenizer,
            model,
            prompt,
            max_new_tokens=int(inference_cfg.get("max_new_tokens", 220)),
        )
        outputs.append(
            {
                "instruction": record["instruction"],
                "input": record["input"],
                "expected_output": record["output"],
                "baseline_output": raw_output,
            }
        )

    write_jsonl(args.output, outputs)
    summary = {"saved_to": args.output, "rows": len(outputs)}
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
