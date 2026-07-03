"""
routes.py

Purpose:
    REST endpoints for the Model Serving Layer. Thin: every endpoint
    validates via api/schemas.py, delegates to api/services.py or
    api/model_manager.py, and returns a schema instance. No inference
    logic, no checkpoint I/O, and no app/startup wiring lives here (that
    belongs to api/services.py, api/model_manager.py, and api/main.py
    respectively).

Public Interface (never rename):
    GET  /
    GET  /health
    GET  /model/info
    POST /predict
    POST /predict/batch
    POST /model/reload
    GET  /metrics
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request, status

from api.model_manager import CheckpointNotFoundError, ModelManager
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    MetricsResponse,
    PredictionRequest,
    PredictionResponse,
    ReloadResponse,
)
from api.services import PredictionError, PredictionService

logger = logging.getLogger(__name__)

router = APIRouter()


# ----------------------------------------------------------------------
# Dependency accessors -- read singletons off app.state, set once at
# startup by api/main.py. Never construct a ModelManager/PredictionService
# here; that would defeat the "singleton model instance" requirement.
# ----------------------------------------------------------------------


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


def get_prediction_service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


def get_metrics_tracker(request: Request):
    return request.app.state.metrics_tracker


def get_app_start_time(request: Request) -> float:
    return request.app.state.start_time


# ----------------------------------------------------------------------
# GET /
# ----------------------------------------------------------------------


@router.get("/", tags=["meta"], summary="Service banner")
def root(request: Request) -> dict:
    """Basic liveness/identification endpoint."""
    return {
        "service": "fraud-detection-model-serving",
        "status": "running",
        "docs": "/docs",
    }


# ----------------------------------------------------------------------
# GET /health
# ----------------------------------------------------------------------


@router.get("/health", tags=["meta"], response_model=HealthResponse, summary="Health check")
def health(request: Request) -> HealthResponse:
    """Report whether the service is up and whether a model is loaded.

    Always returns HTTP 200 (this is a liveness/readiness signal, not a
    prediction endpoint) -- check ``model_loaded`` / ``status`` in the
    body to distinguish "up with a model" from "up but not yet serving
    predictions".
    """
    model_manager: ModelManager = request.app.state.model_manager
    metadata = model_manager.metadata
    return HealthResponse(
        status="ok" if model_manager.is_loaded else "degraded",
        model_loaded=model_manager.is_loaded,
        uptime_seconds=time.time() - request.app.state.start_time,
        device=metadata["device"],
    )


# ----------------------------------------------------------------------
# GET /model/info
# ----------------------------------------------------------------------


@router.get(
    "/model/info", tags=["model"], response_model=ModelInfoResponse, summary="Model metadata"
)
def model_info(request: Request) -> ModelInfoResponse:
    """Return checkpoint path, architecture, and training metadata for
    the currently loaded model (loading it first if not already loaded).
    """
    model_manager: ModelManager = request.app.state.model_manager
    try:
        model_manager.get_model()
    except (CheckpointNotFoundError, ValueError) as exc:
        logger.warning("model_info: model unavailable: %s", exc)
        # Still return what we know (model_loaded=False) rather than a
        # hard error -- callers use this endpoint to discover *why* a
        # model isn't loaded.
        metadata = model_manager.metadata
        return ModelInfoResponse(model_loaded=False, **metadata)

    metadata = model_manager.metadata
    return ModelInfoResponse(model_loaded=model_manager.is_loaded, **metadata)


# ----------------------------------------------------------------------
# POST /predict
# ----------------------------------------------------------------------


@router.post(
    "/predict",
    tags=["inference"],
    response_model=PredictionResponse,
    summary="Score a single transaction",
)
def predict(request: Request, body: PredictionRequest) -> PredictionResponse:
    prediction_service: PredictionService = request.app.state.prediction_service
    metrics_tracker = request.app.state.metrics_tracker

    start = time.perf_counter()
    try:
        result = prediction_service.predict(body)
    except PredictionError as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics_tracker.record_request(success=False, latency_ms=latency_ms)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except (CheckpointNotFoundError, ValueError) as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics_tracker.record_request(success=False, latency_ms=latency_ms)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    metrics_tracker.record_request(success=True, latency_ms=result.latency_ms)
    return result


# ----------------------------------------------------------------------
# POST /predict/batch
# ----------------------------------------------------------------------


@router.post(
    "/predict/batch",
    tags=["inference"],
    response_model=BatchPredictionResponse,
    summary="Score a batch of transactions",
)
def predict_batch(request: Request, body: BatchPredictionRequest) -> BatchPredictionResponse:
    prediction_service: PredictionService = request.app.state.prediction_service
    metrics_tracker = request.app.state.metrics_tracker

    start = time.perf_counter()
    try:
        result = prediction_service.predict_batch(body)
    except PredictionError as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics_tracker.record_request(success=False, latency_ms=latency_ms)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except (CheckpointNotFoundError, ValueError) as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics_tracker.record_request(success=False, latency_ms=latency_ms)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    metrics_tracker.record_request(success=True, latency_ms=result.total_latency_ms)
    return result


# ----------------------------------------------------------------------
# POST /model/reload
# ----------------------------------------------------------------------


@router.post(
    "/model/reload",
    tags=["model"],
    response_model=ReloadResponse,
    summary="Reload the model checkpoint from disk",
)
def reload_model(request: Request) -> ReloadResponse:
    """Force a fresh read of the configured checkpoint, replacing the
    in-memory model. Intended for picking up a newly (re)trained
    checkpoint without restarting the process.
    """
    model_manager: ModelManager = request.app.state.model_manager
    try:
        model_manager.reload_model()
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    metadata = model_manager.metadata
    return ReloadResponse(
        status="reloaded",
        model_version=metadata["model_version"],
        checkpoint_path=metadata["checkpoint_path"],
        loaded_at=metadata["loaded_at"],
    )


# ----------------------------------------------------------------------
# GET /metrics
# ----------------------------------------------------------------------


@router.get(
    "/metrics", tags=["meta"], response_model=MetricsResponse, summary="Service metrics"
)
def metrics(request: Request) -> MetricsResponse:
    """Expose request counts, latency, uptime, and loaded model version.

    System resource figures (CPU/RAM/GPU) are best-effort, reusing
    monitoring.system_monitor.SystemMonitor.collect() -- if that
    collection fails for any reason, ``system`` is returned empty rather
    than failing the whole endpoint.
    """
    model_manager: ModelManager = request.app.state.model_manager
    metrics_tracker = request.app.state.metrics_tracker
    snap = metrics_tracker.snapshot()

    system_stats: dict = {}
    try:
        from monitoring.system_monitor import SystemMonitor

        system_stats = SystemMonitor().collect()
    except Exception as exc:  # pragma: no cover - best-effort only
        logger.warning("metrics: system stats collection failed: %s", exc)

    return MetricsResponse(
        request_count=snap["request_count"],
        successful_requests=snap["successful_requests"],
        failed_requests=snap["failed_requests"],
        average_latency_ms=snap["average_latency_ms"],
        uptime_seconds=snap["uptime_seconds"],
        model_loaded=model_manager.is_loaded,
        model_version=model_manager.metadata["model_version"],
        system=system_stats,
    )
