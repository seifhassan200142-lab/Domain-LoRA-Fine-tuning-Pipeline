"""Utilities for model loading and text generation in scripts."""

from __future__ import annotations

from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer_and_model(
    base_model_name: str,
    torch_dtype: str = "auto",
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load tokenizer and causal language model."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_arg: Any = "auto"
    if torch_dtype != "auto":
        dtype_arg = getattr(torch, torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype_arg,
        trust_remote_code=trust_remote_code,
    )
    return tokenizer, model


def generate_text(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int = 220,
    device: Optional[str] = None,
) -> str:
    """Generate text from a prompt and remove the prompt from decoded text when possible."""
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(selected_device)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(selected_device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()
