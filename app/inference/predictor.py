"""Model loading and prediction utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.inference.formatting import format_prompt, normalize_structured_response, parse_json_output

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional until inference with adapters is used
    PeftModel = None  # type: ignore


class SupportPredictor:
    """Load a base model plus optional LoRA adapter and generate structured output."""

    def __init__(
        self,
        base_model_name: str,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> None:
        """Load tokenizer, base model, and optional PEFT adapter."""
        dtype_arg: Any = "auto"
        if self.torch_dtype != "auto":
            dtype_arg = getattr(torch, self.torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=dtype_arg,
            trust_remote_code=self.trust_remote_code,
        )

        if self.adapter_path and Path(self.adapter_path).exists():
            if PeftModel is None:
                raise ImportError("peft is required to load a LoRA adapter. Install requirements.txt first.")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, customer_message: str, max_new_tokens: int = 220) -> Dict[str, Any]:
        """Generate and parse a structured support response."""
        if self.model is None or self.tokenizer is None:
            self.load()

        prompt = format_prompt(customer_message)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        raw_output = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()

        try:
            parsed = parse_json_output(raw_output)
            normalized = normalize_structured_response(parsed)
        except ValueError:
            normalized = normalize_structured_response({})

        return {
            "prediction": normalized,
            "raw_output": raw_output,
            "model_info": {
                "base_model_name": self.base_model_name,
                "adapter_path": self.adapter_path or "none",
                "device": self.device,
            },
        }


def build_predictor_from_env() -> SupportPredictor:
    """Create predictor using environment variables."""
    return SupportPredictor(
        base_model_name=os.getenv("BASE_MODEL_NAME", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        adapter_path=os.getenv("LORA_ADAPTER_PATH", "outputs/lora_adapter"),
        device=os.getenv("DEVICE") or None,
        torch_dtype=os.getenv("TORCH_DTYPE", "auto"),
        trust_remote_code=os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true",
    )
