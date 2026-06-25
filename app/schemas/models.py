"""Pydantic schemas for API and inference responses."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for customer support complaint formatting."""

    text: str = Field(..., min_length=1, description="Customer complaint or question")
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=16,
        le=512,
        description="Optional generation token limit override",
    )


class StructuredSupportResponse(BaseModel):
    """Structured response expected from the fine-tuned model."""

    category: str = Field(..., description="Support category")
    priority: str = Field(..., description="Priority such as Low, Medium, or High")
    suggested_reply: str = Field(..., description="Support agent reply draft")
    next_action: str = Field(..., description="Recommended operational action")


class PredictionResponse(BaseModel):
    """API response with parsed and raw model outputs."""

    prediction: StructuredSupportResponse
    raw_output: str
    model_info: Dict[str, str] = Field(default_factory=dict)
