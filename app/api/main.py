"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Domain LoRA Fine-tuning Pipeline",
    description="Structured customer support response generation using a LoRA-adapted language model.",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "project": "domain-lora-finetuning-pipeline",
        "docs": "/docs",
        "predict_endpoint": "/api/predict",
    }
