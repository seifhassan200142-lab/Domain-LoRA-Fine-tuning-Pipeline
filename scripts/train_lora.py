"""Train a LoRA adapter for structured customer support formatting."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import inspect
import json
from typing import Any, Dict

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from app.inference.formatting import format_training_text
from scripts.common import load_yaml, set_seed


def _filter_training_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only arguments supported by the installed Transformers version."""
    signature = inspect.signature(TrainingArguments.__init__)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def tokenize_dataset(dataset: Any, tokenizer: Any, max_seq_length: int) -> Any:
    """Convert instruction rows into tokenized causal-LM training examples."""
    eos_token = tokenizer.eos_token or ""

    def tokenize_record(record: Dict[str, str]) -> Dict[str, Any]:
        text = format_training_text(
            instruction=record["instruction"],
            customer_message=record["input"],
            output=record["output"],
            eos_token=eos_token,
        )
        return tokenizer(text, truncation=True, max_length=max_seq_length)

    return dataset.map(tokenize_record, remove_columns=dataset["train"].column_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA adapter.")
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model_name"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_arg: Any = "auto"
    if model_cfg.get("torch_dtype", "auto") != "auto":
        dtype_arg = getattr(torch, model_cfg["torch_dtype"])

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_name"],
        torch_dtype=dtype_arg,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        target_modules=list(lora_cfg["target_modules"]),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_cfg["train_file"],
            "validation": train_cfg["validation_file"],
        },
    )
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, int(model_cfg.get("max_seq_length", 512)))

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(train_cfg["logging_dir"]).mkdir(parents=True, exist_ok=True)

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "logging_dir": train_cfg["logging_dir"],
        "num_train_epochs": train_cfg.get("num_train_epochs", 1),
        "max_steps": train_cfg.get("max_steps", -1),
        "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": train_cfg.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 1),
        "learning_rate": train_cfg.get("learning_rate", 2e-4),
        "warmup_steps": train_cfg.get("warmup_steps", 0),
        "weight_decay": train_cfg.get("weight_decay", 0.0),
        "logging_steps": train_cfg.get("logging_steps", 10),
        "save_steps": train_cfg.get("save_steps", 50),
        "eval_steps": train_cfg.get("eval_steps", 50),
        "save_total_limit": train_cfg.get("save_total_limit", 2),
        "fp16": bool(train_cfg.get("fp16", False)),
        "bf16": bool(train_cfg.get("bf16", False)),
        "report_to": [],
        "remove_unused_columns": False,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        training_args_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in signature.parameters:
        training_args_kwargs["evaluation_strategy"] = "steps"

    training_args = TrainingArguments(**_filter_training_args(training_args_kwargs))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    config_copy_path = output_dir / "training_config_used.json"
    config_copy_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({"adapter_saved_to": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
