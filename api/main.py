"""
main.py

Purpose:
    Assemble the FastAPI application for the Model Serving Layer:
    configuration loading, structured logging, startup/shutdown events
    (model loading/unloading), dependency singletons on app.state, router
    registration, and centralized exception handling.

Phase 8 Scope:
    No training logic. No auth/db/queues/docker (see project-level DO NOT
    list). Reuses api/model_manager.py and api/services.py for everything
    model- and inference-related; this file only wires them together.

Run:
    uvicorn api.main:app --reload
    (or `python -m api.main` for a non-reload run using config/api/api.yaml's
    server.host/port/workers)

Public Interface:
    app: FastAPI
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.model_manager import CheckpointNotFoundError, ModelManager, ModelManagerConfig
from api.routes import router
from api.schemas import ErrorResponse
from api.services import MetricsTracker, PredictionService, PredictionServiceConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "api" / "api.yaml"

logger = logging.getLogger("api")


# ----------------------------------------------------------------------
# TASK 10 -- Configuration loading (no hardcoded values)
# ----------------------------------------------------------------------


def load_config(config_path: Path = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load config/api/api.yaml.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        The parsed config dict.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the file does not parse to a dict.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"API config not found at {config_path}; expected config/api/api.yaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"API config at {config_path} did not parse to a mapping")
    return data


# ----------------------------------------------------------------------
# TASK 8 -- Structured logging
# ----------------------------------------------------------------------


class _JSONLogFormatter(logging.Formatter):
    """Minimal structured (JSON) log formatter: one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "endpoint", "response_time_ms", "prediction_result"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            # Deliberately no traceback text here -- structured logs are
            # written to disk/stdout (an operator surface), but Task 7
            # ("never expose internal stack traces") is enforced at the
            # HTTP-response boundary in the exception handlers below, and
            # keeping tracebacks out of routine request logs too avoids
            # leaking them into any log aggregator with looser access
            # control than the server itself.
            payload["error_type"] = str(record.exc_info[0].__name__) if record.exc_info[0] else None
        return json.dumps(payload)


def configure_logging(level: str, json_logs: bool) -> None:
    """Configure the root logger once at startup.

    Args:
        level: Logging level name (e.g. "info", "debug").
        json_logs: If True, use structured JSON log lines; otherwise
            plain text.
    """
    handler = logging.StreamHandler()
    if json_logs:
        handler.setFormatter(_JSONLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


# ----------------------------------------------------------------------
# TASK 1 -- Startup / shutdown
# ----------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.state.config

    logger.info("Starting Model Serving Layer")
    model_manager: ModelManager = app.state.model_manager
    try:
        model_manager.load_model()
        logger.info("Model loaded at startup from %s", model_manager.metadata["checkpoint_path"])
    except CheckpointNotFoundError as exc:
        # Per Task 2 (lazy loading) and the acceptance criteria, the app
        # must still come up (and expose /health, /model/info, OpenAPI
        # docs) even before a model has ever been trained -- it will
        # lazily retry on the first request that needs the model, or
        # after POST /model/reload once a checkpoint exists.
        logger.warning(
            "No checkpoint available at startup (%s); server will start without a "
            "loaded model and retry lazily on first use or POST /model/reload", exc,
        )

    yield

    logger.info("Shutting down Model Serving Layer")
    model_manager.unload_model()


# ----------------------------------------------------------------------
# TASK 1 -- Application factory
# ----------------------------------------------------------------------


def create_app(config_path: Path = _DEFAULT_CONFIG_PATH) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        config_path: Path to config/api/api.yaml.
    """
    config = load_config(config_path)

    logging_cfg = config.get("logging", {})
    configure_logging(
        level=logging_cfg.get("level", "info"),
        json_logs=bool(logging_cfg.get("json_logs", True)),
    )

    app = FastAPI(
        title="Fraud Detection Model Serving API",
        description=(
            "Phase 8 Model Serving Layer for the Communication-Efficient "
            "Distributed Fraud Detection project. Exposes the trained "
            "FraudMLP model for single and batch inference."
        ),
        version="8.0.0",
        lifespan=lifespan,
    )

    app.state.config = config
    app.state.start_time = time.time()

    model_cfg = config.get("model", {})
    app.state.model_manager = ModelManager(
        config=ModelManagerConfig(
            checkpoint_path=model_cfg.get("checkpoint_path", "artifacts/models/best_model.pt"),
            model_config_path=model_cfg.get(
                "model_config_path", "artifacts/centralized_config.json"
            ),
            device=model_cfg.get("device", "auto"),
            model_version=model_cfg.get("model_version", "unversioned"),
        ),
        lazy=True,
    )

    inference_cfg = config.get("inference", {})
    app.state.prediction_service = PredictionService(
        model_manager=app.state.model_manager,
        config=PredictionServiceConfig(
            max_batch_size=inference_cfg.get("max_batch_size", 512),
        ),
    )

    app.state.metrics_tracker = MetricsTracker()

    app.include_router(router)
    _register_middleware(app)
    _register_exception_handlers(app)

    return app


# ----------------------------------------------------------------------
# TASK 8 -- Per-request logging middleware
# ----------------------------------------------------------------------


def _register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        request.state.request_id = request_id

        response = await call_next(request)

        response_time_ms = (time.perf_counter() - start) * 1000.0
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request handled",
            extra={
                "request_id": request_id,
                "endpoint": request.url.path,
                "response_time_ms": round(response_time_ms, 3),
            },
        )
        return response


# ----------------------------------------------------------------------
# TASK 7 -- Centralized exception handling
# ----------------------------------------------------------------------


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", None)
        logger.info(
            "request validation failed",
            extra={"request_id": request_id, "endpoint": request.url.path},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="validation_error",
                detail=str(exc.errors()),
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        request_id = getattr(request.state, "request_id", None)
        logger.info(
            "request failed",
            extra={"request_id": request_id, "endpoint": request.url.path},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="request_error",
                detail=str(exc.detail),
                request_id=request_id,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception(
            "unhandled exception",
            extra={"request_id": request_id, "endpoint": request.url.path},
        )
        # Never expose internal stack traces or exception internals to
        # the client (Task 7) -- a fixed, generic message only.
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_server_error",
                detail="An unexpected error occurred while processing the request.",
                request_id=request_id,
            ).model_dump(),
        )


app = create_app()


def main() -> None:
    """CLI entry point for a non-reload run: `python -m api.main`."""
    import uvicorn

    server_cfg = load_config(_DEFAULT_CONFIG_PATH).get("server", {})
    uvicorn.run(
        "api.main:app",
        host=server_cfg.get("host", "0.0.0.0"),
        port=int(server_cfg.get("port", 8000)),
        workers=int(server_cfg.get("workers", 1)),
        log_level=server_cfg.get("log_level", "info"),
        reload=bool(server_cfg.get("reload", False)),
    )


if __name__ == "__main__":
    main()
