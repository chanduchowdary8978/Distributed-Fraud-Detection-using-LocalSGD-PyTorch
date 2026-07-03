"""
schemas.py

Purpose:
    Pydantic request/response models for the Model Serving Layer. Every
    request body, response body, and error shape returned by api/routes.py
    is defined here so validation and OpenAPI documentation stay in one
    place.

Phase 8 Scope:
    Schemas only -- no inference logic (api/services.py) and no checkpoint
    handling (api/model_manager.py) lives here.

Public Interface:
    PredictionRequest
    BatchPredictionRequest
    PredictionResponse
    BatchPredictionResponse
    HealthResponse
    ModelInfoResponse
    MetricsResponse
    ReloadResponse
    ErrorResponse
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """A single transaction's feature vector.

    The trained model's feature schema (column order, one-hot expansion
    of `type`, etc.) is produced dynamically by training.dataset.FraudDataset
    from the shard CSV at training time and is not persisted as named
    columns in any artifact -- so this API accepts an ordered flat vector
    of length ``input_dim`` rather than named fields. Callers are
    responsible for producing features in the same order used at
    training time. See GET /model/info for the expected length.
    """

    features: List[float] = Field(
        ...,
        description=(
            "Ordered feature vector for one transaction, matching the "
            "input_dim reported by GET /model/info."
        ),
        examples=[[0.0, 1.0, 9839.64, 170136.0, 160296.36, 0.0, 0.0, 0.0]],
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Decision threshold applied to the fraud probability.",
    )

    @field_validator("features")
    @classmethod
    def _features_non_empty(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("features must be a non-empty list of numbers")
        return v


class BatchPredictionRequest(BaseModel):
    """A batch of transactions to score in a single request."""

    instances: List[List[float]] = Field(
        ...,
        description="List of feature vectors, each matching input_dim.",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("instances")
    @classmethod
    def _instances_non_empty(cls, v: List[List[float]]) -> List[List[float]]:
        if not v:
            raise ValueError("instances must be a non-empty list of feature vectors")
        for i, row in enumerate(v):
            if not row:
                raise ValueError(f"instances[{i}] must be a non-empty feature vector")
        return v


class PredictionResponse(BaseModel):
    """Result of scoring a single transaction."""

    fraud_probability: float = Field(..., description="Predicted fraud probability in [0, 1].")
    is_fraud: bool = Field(..., description="fraud_probability >= threshold.")
    threshold: float
    latency_ms: float = Field(..., description="Server-side inference latency in milliseconds.")
    model_version: str
    request_id: str


class BatchPredictionResponse(BaseModel):
    """Result of scoring a batch of transactions."""

    predictions: List[PredictionResponse]
    count: int
    total_latency_ms: float = Field(..., description="Total server-side latency for the whole batch.")
    request_id: str


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = Field(..., description="'ok' if serving requests, 'degraded' if model is not loaded.")
    model_loaded: bool
    uptime_seconds: float
    device: str


class ModelInfoResponse(BaseModel):
    """GET /model/info response: everything about the currently loaded model."""

    model_loaded: bool
    checkpoint_path: str
    model_config_path: Optional[str]
    checkpoint_format: Optional[str]
    device: str
    model_version: str
    loaded_at: Optional[float]
    architecture: Dict[str, Any]
    training_extra: Dict[str, Any]
    load_count: int


class MetricsResponse(BaseModel):
    """GET /metrics response."""

    request_count: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float
    uptime_seconds: float
    model_loaded: bool
    model_version: str
    system: Dict[str, Any]


class ReloadResponse(BaseModel):
    """POST /model/reload response."""

    status: str
    model_version: str
    checkpoint_path: str
    loaded_at: Optional[float]


class ErrorResponse(BaseModel):
    """Uniform error shape for all non-2xx responses. Never includes a
    stack trace or other internal detail -- see api/main.py's exception
    handlers."""

    error: str
    detail: str
    request_id: Optional[str] = None
