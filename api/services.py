"""
services.py

Purpose:
    The inference and monitoring business logic behind api/routes.py:
    request validation/preprocessing, running the model, latency
    measurement, and in-process request metrics. No FastAPI-specific code
    (routing, dependency wiring) and no checkpoint I/O lives here -- those
    belong to api/routes.py and api/model_manager.py respectively.

Phase 8 Scope:
    Reuses models.fraud_mlp.FraudMLP.predict_proba/.predict as-is (never
    reimplements the forward pass or decision-threshold logic) and
    api.model_manager.ModelManager for the model instance. No training
    logic.

Public Interface:
    class PredictionError(Exception)
    class PredictionService
        Methods:
            predict(request: PredictionRequest) -> PredictionResponse
            predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse
    class MetricsTracker
        Methods:
            record_request(success: bool, latency_ms: float) -> None
            snapshot() -> dict
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from api.model_manager import ModelManager
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)


class PredictionError(ValueError):
    """Raised for any request-level validation failure (bad shape, bad
    batch size, etc.). Callers (api/routes.py) map this to HTTP 422/400;
    it is never allowed to surface as an unhandled 500."""


@dataclass
class PredictionServiceConfig:
    """Configuration for :class:`PredictionService`.

    Attributes:
        max_batch_size: Maximum number of instances accepted by
            predict_batch() in a single request.
    """

    max_batch_size: int = 512

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {self.max_batch_size!r}")


class PredictionService:
    """Validates, preprocesses, and scores requests against the model
    owned by a :class:`~api.model_manager.ModelManager`.

    Args:
        model_manager: Provides the loaded ``FraudMLP`` instance. Never
            constructed here -- this class only ever calls
            ``model_manager.get_model()``.
        config: Optional :class:`PredictionServiceConfig`.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        config: PredictionServiceConfig | None = None,
    ) -> None:
        self.model_manager = model_manager
        self.config = config or PredictionServiceConfig()

    # ------------------------------------------------------------------
    # Validation / preprocessing
    # ------------------------------------------------------------------

    def _validate_and_build_tensor(self, rows: List[List[float]]) -> torch.Tensor:
        model = self.model_manager.get_model()
        expected_dim = model.input_dim

        for i, row in enumerate(rows):
            if len(row) != expected_dim:
                raise PredictionError(
                    f"features at index {i} has {len(row)} values, expected "
                    f"{expected_dim} (model.input_dim); see GET /model/info"
                )
            if not all(np.isfinite(v) for v in row):
                raise PredictionError(
                    f"features at index {i} contains NaN or infinite values"
                )

        array = np.asarray(rows, dtype=np.float32)
        return torch.from_numpy(array)

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Validate, preprocess, run inference, and return a fraud
        probability + decision for one transaction.

        Raises:
            PredictionError: If ``request.features`` has the wrong
                length for the loaded model, or contains non-finite
                values.
        """
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        model = self.model_manager.get_model()
        tensor = self._validate_and_build_tensor([request.features])

        probs = model.predict_proba(tensor)
        probability = float(probs[0, 0].item())
        is_fraud = probability >= request.threshold

        latency_ms = (time.perf_counter() - start) * 1000.0

        return PredictionResponse(
            fraud_probability=probability,
            is_fraud=is_fraud,
            threshold=request.threshold,
            latency_ms=latency_ms,
            model_version=self.model_manager.metadata["model_version"],
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Validate, preprocess, run inference, and return fraud
        probabilities + decisions for a batch of transactions.

        Raises:
            PredictionError: If the batch exceeds max_batch_size, or any
                instance has the wrong length / non-finite values.
        """
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        if len(request.instances) > self.config.max_batch_size:
            raise PredictionError(
                f"batch size {len(request.instances)} exceeds max_batch_size "
                f"({self.config.max_batch_size}); see config/api/api.yaml"
            )

        model = self.model_manager.get_model()
        tensor = self._validate_and_build_tensor(request.instances)

        probs = model.predict_proba(tensor)
        probs_np = probs.detach().cpu().numpy().reshape(-1)

        predictions: List[PredictionResponse] = []
        model_version = self.model_manager.metadata["model_version"]
        for probability in probs_np:
            probability = float(probability)
            predictions.append(
                PredictionResponse(
                    fraud_probability=probability,
                    is_fraud=probability >= request.threshold,
                    threshold=request.threshold,
                    latency_ms=0.0,  # per-item latency is not meaningful for a batched forward pass
                    model_version=model_version,
                    request_id=request_id,
                )
            )

        total_latency_ms = (time.perf_counter() - start) * 1000.0

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions),
            total_latency_ms=total_latency_ms,
            request_id=request_id,
        )


class MetricsTracker:
    """Thread-safe counters for GET /metrics.

    Tracks request counts (successful/failed), a running average
    latency, and process start time (for uptime). Deliberately in-memory
    only, matching Phase 8's "no database" constraint.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._request_count = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_latency_ms = 0.0

    def record_request(self, success: bool, latency_ms: float) -> None:
        with self._lock:
            self._request_count += 1
            self._total_latency_ms += latency_ms
            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def snapshot(self) -> dict:
        with self._lock:
            avg_latency = (
                self._total_latency_ms / self._request_count if self._request_count else 0.0
            )
            return {
                "request_count": self._request_count,
                "successful_requests": self._successful_requests,
                "failed_requests": self._failed_requests,
                "average_latency_ms": avg_latency,
                "uptime_seconds": self.uptime_seconds,
            }
