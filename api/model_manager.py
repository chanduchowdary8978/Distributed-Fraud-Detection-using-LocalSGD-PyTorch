"""
model_manager.py

Purpose:
    Own the lifecycle of the single in-process FraudMLP instance used to
    serve predictions: locating and loading the best checkpoint, device
    placement, lazy loading, and reload/unload. This is the ONLY place in
    Phase 8 that touches a checkpoint file or constructs a FraudMLP for
    serving.

Phase 8 Scope:
    No training, no optimizer, no loss function, no dataset access.
    Reuses models.fraud_mlp.FraudMLP as-is (never duplicates its
    architecture or inference logic) and training.utils.get_device for
    device resolution, the same helper training/ already uses.

Checkpoint formats supported:
    1. Trainer-style (training.trainer.Trainer.save_checkpoint): a dict
       with a "model_state_dict" key and no embedded architecture config.
       This is what every training strategy in training/ actually
       produces. Requires a companion JSON file (model_config_path)
       containing the FraudMLP constructor kwargs under a "model" key
       (as written by training.centralized_baseline.CentralizedTrainer.run(),
       for example), or the raw kwargs at the JSON's top level.
    2. Self-contained (models.fraud_mlp.FraudMLP.save/.load): a dict with
       "config" and "state_dict" keys. Loaded directly via
       FraudMLP.load(), no companion file needed.

    The manager tries format 2 first (it's self-describing), then falls
    back to format 1.

Public Interface:
    class ModelManager
        Methods:
            load_model()
            reload_model()
            unload_model()
            get_model()
        Properties:
            is_loaded
            metadata
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from models.fraud_mlp import FraudMLP
from training.utils import get_device

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ModelNotLoadedError(RuntimeError):
    """Raised when an operation requires a loaded model but none is loaded."""


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when the configured checkpoint (or its companion config) is missing."""


@dataclass
class ModelManagerConfig:
    """Configuration for :class:`ModelManager`.

    Attributes:
        checkpoint_path: Path to the model checkpoint (Trainer-style or
            FraudMLP-self-contained). Relative paths are resolved against
            the project root.
        model_config_path: Path to a JSON file providing FraudMLP
            constructor kwargs, used only for Trainer-style checkpoints
            (which don't embed the architecture config themselves).
        device: "auto", "cpu", "cuda", or "cuda:N". "auto" defers to
            training.utils.get_device (CUDA if available, else CPU).
        model_version: Human-readable label surfaced via metadata /
            GET /model/info. Purely informational.
    """

    checkpoint_path: Union[str, Path] = "artifacts/models/best_model.pt"
    model_config_path: Union[str, Path] = "artifacts/centralized_config.json"
    device: str = "auto"
    model_version: str = "unversioned"

    def __post_init__(self) -> None:
        self.checkpoint_path = _resolve(self.checkpoint_path)
        self.model_config_path = _resolve(self.model_config_path)


def _resolve(path: Union[str, Path]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (_PROJECT_ROOT / path)


@dataclass
class _ModelMetadata:
    """Snapshot of everything known about the currently loaded model."""

    checkpoint_path: str
    model_config_path: Optional[str]
    device: str
    model_version: str
    loaded_at: Optional[float] = None
    checkpoint_format: Optional[str] = None
    architecture: Dict[str, Any] = field(default_factory=dict)
    training_extra: Dict[str, Any] = field(default_factory=dict)
    load_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "model_config_path": self.model_config_path,
            "device": self.device,
            "model_version": self.model_version,
            "loaded_at": self.loaded_at,
            "checkpoint_format": self.checkpoint_format,
            "architecture": self.architecture,
            "training_extra": self.training_extra,
            "load_count": self.load_count,
        }


class ModelManager:
    """Owns the single serving-time FraudMLP instance.

    Thread-safe: a lock guards load/reload/unload so concurrent requests
    never observe a half-swapped model. The model is never reloaded on a
    per-request basis -- callers hold onto the instance returned by
    :meth:`get_model` for the duration of a single request only, and
    always go back through :meth:`get_model` for the next one, so a
    reload takes effect for subsequent requests without restarting the
    process.

    Args:
        config: Optional :class:`ModelManagerConfig`. Defaults to
            ``ModelManagerConfig()`` (i.e. the repository's default
            checkpoint locations) if not given.
        lazy: If True (default), no checkpoint is read until the first
            :meth:`get_model` call or an explicit :meth:`load_model`
            call. If False, :meth:`load_model` is called immediately in
            ``__init__``.

    Raises:
        CheckpointNotFoundError: If ``lazy=False`` and the checkpoint (or
            its companion config, for Trainer-style checkpoints) does not
            exist.
    """

    def __init__(self, config: Optional[ModelManagerConfig] = None, lazy: bool = True) -> None:
        self.config = config or ModelManagerConfig()
        self._lock = threading.RLock()
        self._model: Optional[FraudMLP] = None
        self._device = self._resolve_device(self.config.device)
        self._metadata = _ModelMetadata(
            checkpoint_path=str(self.config.checkpoint_path),
            model_config_path=str(self.config.model_config_path),
            device=str(self._device),
            model_version=self.config.model_version,
        )

        logger.info(
            "ModelManager initialized (lazy=%s, checkpoint_path=%s, device=%s)",
            lazy, self.config.checkpoint_path, self._device,
        )

        if not lazy:
            self.load_model()

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device in (None, "auto"):
            return get_device(None)
        return get_device(device)

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    @property
    def metadata(self) -> Dict[str, Any]:
        with self._lock:
            return self._metadata.as_dict()

    def load_model(self) -> FraudMLP:
        """Load the checkpoint into memory if not already loaded.

        Idempotent: if a model is already loaded, it is returned as-is
        without touching disk again. Use :meth:`reload_model` to force a
        fresh read from disk.

        Returns:
            The loaded ``FraudMLP`` instance (in eval mode, on the
            resolved device).

        Raises:
            CheckpointNotFoundError: If the checkpoint (or, for
                Trainer-style checkpoints, its companion model config
                JSON) does not exist on disk.
            ValueError: If the checkpoint/config is malformed.
        """
        with self._lock:
            if self._model is not None:
                logger.debug("load_model() called but a model is already loaded; no-op")
                return self._model
            return self._load_locked()

    def reload_model(self) -> FraudMLP:
        """Force a fresh load from disk, replacing any currently loaded model.

        Intended for POST /model/reload: swap in a newly trained
        checkpoint without restarting the server. If the reload fails,
        the previously loaded model (if any) is left in place so a
        transient/bad reload never leaves the service without a model.

        Returns:
            The newly loaded ``FraudMLP`` instance.

        Raises:
            CheckpointNotFoundError: If the checkpoint/config is missing.
            ValueError: If the checkpoint/config is malformed.
        """
        with self._lock:
            previous = self._model
            self._model = None
            try:
                return self._load_locked()
            except Exception:
                logger.exception(
                    "reload_model() failed; keeping previously loaded model in place"
                )
                self._model = previous
                raise

    def unload_model(self) -> None:
        """Release the currently loaded model, freeing its memory.

        Safe to call when no model is loaded (no-op). A subsequent
        :meth:`get_model` call will lazily reload from disk.
        """
        with self._lock:
            if self._model is None:
                logger.debug("unload_model() called but no model was loaded; no-op")
                return
            self._model = None
            self._metadata.loaded_at = None
            self._metadata.checkpoint_format = None
            logger.info("Model unloaded")

    def get_model(self) -> FraudMLP:
        """Return the loaded model, lazily loading it on first access.

        Returns:
            The loaded ``FraudMLP`` instance.

        Raises:
            CheckpointNotFoundError: If no model is loaded and the
                checkpoint cannot be found on disk.
            ValueError: If no model is loaded and the checkpoint/config is
                malformed.
        """
        with self._lock:
            if self._model is None:
                return self._load_locked()
            return self._model

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_locked(self) -> FraudMLP:
        """Load a model from disk. Caller must hold self._lock."""
        checkpoint_path: Path = self.config.checkpoint_path
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(
                f"No checkpoint found at {checkpoint_path}. Train a model first "
                f"(e.g. `python -m training.centralized_baseline`) or point "
                f"model.checkpoint_path in config/api/api.yaml at an existing "
                f"checkpoint."
            )

        raw = torch.load(checkpoint_path, map_location=self._device)
        if not isinstance(raw, dict):
            raise ValueError(
                f"Checkpoint at {checkpoint_path} is not a dict-shaped torch "
                f"checkpoint (got {type(raw)!r}); cannot load"
            )

        if "config" in raw and "state_dict" in raw:
            model, fmt, extra = self._load_self_contained(raw)
        elif "model_state_dict" in raw:
            model, fmt, extra = self._load_trainer_style(raw)
        else:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} has unrecognized keys "
                f"{sorted(raw.keys())}; expected either "
                f"('config', 'state_dict') [FraudMLP.save format] or "
                f"('model_state_dict', ...) [Trainer.save_checkpoint format]"
            )

        model.to(self._device)
        model.eval()

        self._model = model
        self._metadata.loaded_at = time.time()
        self._metadata.checkpoint_format = fmt
        self._metadata.architecture = dict(model.config)
        self._metadata.training_extra = extra
        self._metadata.load_count += 1
        self._metadata.device = str(self._device)

        logger.info(
            "Model loaded from %s (format=%s, device=%s, architecture=%s)",
            checkpoint_path, fmt, self._device, model.config,
        )
        return model

    def _load_self_contained(self, raw: Dict[str, Any]) -> tuple:
        """Load a checkpoint saved via FraudMLP.save()."""
        model = FraudMLP(**raw["config"])
        model.load_state_dict(raw["state_dict"])
        extra = {k: v for k, v in raw.items() if k not in ("config", "state_dict")}
        return model, "fraud_mlp_self_contained", extra

    def _load_trainer_style(self, raw: Dict[str, Any]) -> tuple:
        """Load a checkpoint saved via training.trainer.Trainer.save_checkpoint()."""
        model_config = self._read_model_config()
        model = FraudMLP(**model_config)
        model.load_state_dict(raw["model_state_dict"])
        extra = {
            "epoch": raw.get("epoch"),
            **(raw.get("extra") or {}),
        }
        return model, "trainer_checkpoint", extra

    def _read_model_config(self) -> Dict[str, Any]:
        config_path: Path = self.config.model_config_path
        if not config_path.exists():
            raise CheckpointNotFoundError(
                f"Checkpoint at {self.config.checkpoint_path} does not embed an "
                f"architecture config, and no companion model config was found "
                f"at {config_path}. Set model.model_config_path in "
                f"config/api/api.yaml to the JSON file written alongside "
                f"training (e.g. artifacts/centralized_config.json)."
            )
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # training.centralized_baseline.CentralizedTrainer.run() writes the
        # architecture kwargs under a "model" key alongside other run
        # metadata; other producers may write the kwargs directly at the
        # top level.
        model_config = data.get("model", data)

        required = {"input_dim"}
        missing = required - set(model_config)
        if missing:
            raise ValueError(
                f"Model config at {config_path} is missing required key(s) "
                f"{sorted(missing)}; found {sorted(model_config)}"
            )
        return model_config
