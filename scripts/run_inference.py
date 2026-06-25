"""Command-line inference for the LoRA-adapted support model."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from app.inference.predictor import SupportPredictor
from scripts.common import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one structured prediction.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--text", required=True, help="Customer complaint or question")
    parser.add_argument("--adapter-path", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_cfg = config["model"]
    train_cfg = config["training"]
    inference_cfg = config.get("inference", {})

    predictor = SupportPredictor(
        base_model_name=model_cfg["base_model_name"],
        adapter_path=args.adapter_path or train_cfg["output_dir"],
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    result = predictor.predict(args.text, max_new_tokens=int(inference_cfg.get("max_new_tokens", 220)))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
