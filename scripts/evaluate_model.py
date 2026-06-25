"""Evaluate baseline and optional LoRA-adapted model outputs."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from peft import PeftModel

from app.inference.formatting import format_prompt
from scripts.common import load_yaml, read_jsonl, write_jsonl
from scripts.metrics import compute_metrics, write_evaluation_report
from scripts.model_utils import generate_text, load_tokenizer_and_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structured model outputs.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--baseline-file", default="outputs/baseline_outputs.jsonl")
    parser.add_argument("--output-jsonl", default="outputs/finetuned_outputs.jsonl")
    parser.add_argument("--report", default="outputs/evaluation_report.json")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_cfg = config["model"]
    train_cfg = config["training"]
    inference_cfg = config.get("inference", {})

    validation_file = args.validation_file or train_cfg["validation_file"]
    adapter_path = args.adapter_path or train_cfg["output_dir"]

    records = read_jsonl(validation_file)[: args.limit]
    tokenizer, model = load_tokenizer_and_model(
        model_cfg["base_model_name"],
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )

    if Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print(f"Warning: adapter path '{adapter_path}' does not exist. Evaluating base model only.")

    finetuned_rows = []
    finetuned_outputs = []
    expected_outputs = []

    for record in records:
        prompt = format_prompt(record["input"], instruction=record["instruction"])
        output = generate_text(
            tokenizer,
            model,
            prompt,
            max_new_tokens=int(inference_cfg.get("max_new_tokens", 220)),
        )
        finetuned_outputs.append(output)
        expected_outputs.append(record["output"])
        finetuned_rows.append(
            {
                "instruction": record["instruction"],
                "input": record["input"],
                "expected_output": record["output"],
                "model_output": output,
            }
        )

    write_jsonl(args.output_jsonl, finetuned_rows)

    report = {
        "model": model_cfg["base_model_name"],
        "adapter_path": adapter_path,
        "num_examples": len(finetuned_rows),
        "finetuned_or_current_model": compute_metrics(finetuned_outputs, expected_outputs),
        "notes": [
            "Metrics are simple local checks, not benchmark claims.",
            "Category match is exact string match against the sample labels only.",
        ],
    }

    if Path(args.baseline_file).exists():
        baseline_rows = read_jsonl(args.baseline_file)[: args.limit]
        baseline_outputs = [row.get("baseline_output", "") for row in baseline_rows]
        baseline_expected = [row.get("expected_output", "") for row in baseline_rows]
        report["baseline_model"] = compute_metrics(baseline_outputs, baseline_expected)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_evaluation_report(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
