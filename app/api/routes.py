"""FastAPI routes for model inference."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from app.inference.predictor import SupportPredictor, build_predictor_from_env
from app.schemas.models import PredictionRequest, PredictionResponse

router = APIRouter()


@lru_cache(maxsize=1)
def get_predictor() -> SupportPredictor:
    """Cache the predictor so the model is loaded once per API process."""
    return build_predictor_from_env()


@router.get("/health")
def health() -> dict[str, str]:
    """Basic service health check."""
    return {"status": "ok"}


@router.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Run structured customer support response prediction."""
    try:
        result = get_predictor().predict(
            customer_message=request.text,
            max_new_tokens=request.max_new_tokens or 220,
        )
        return PredictionResponse(**result)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise HTTPException(status_code=500, detail=str(exc)) from exc
