"""
utils.py

Purpose:
    Shared, dependency-free helpers used by trainer.py and dataset.py:
    device selection, reproducibility seeding, and early stopping.

Phase 3 Scope:
    No training loop, model, or data logic lives here -- only small
    utilities that would otherwise be duplicated across future training
    strategies (Centralized, LocalSGD, FedAvg).

Public Interface:
    Functions:
        get_device(preferred: str | None = None) -> torch.device
        set_seed(seed: int = 42) -> None

    Classes:
        EarlyStopping
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from models.fraud_mlp import FraudMLP

logger = logging.getLogger(__name__)


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve the torch device to train/evaluate on.

    Args:
        preferred: Optional explicit device string (e.g. "cpu", "cuda",
            "cuda:0"). If given, it is used as-is (no availability check
            beyond what torch.device itself performs). If ``None``
            (default), CUDA is used when available, otherwise CPU.

    Returns:
        A ``torch.device`` instance. Never hardcodes a device name when
        ``preferred`` is not given.
    """
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Seed all RNGs used during training for reproducible runs.

    This is a thin wrapper around ``FraudMLP.set_seed`` (models/fraud_mlp.py)
    rather than a reimplementation, so there is exactly one place that
    seeds random/numpy/torch/cuDNN across the whole project.

    Note on scope: calling this before ``Trainer.fit()`` makes data
    shuffling, dropout masks, and any weight init that happens *after*
    this call deterministic. It cannot retroactively make a model's
    initial weights deterministic if that model was already constructed
    (and thus already initialized) before this function ran -- seed
    before constructing the model for full reproducibility.

    Args:
        seed: Seed value for random / numpy / torch (CPU + CUDA).
    """
    FraudMLP.set_seed(seed)


class EarlyStopping:
    """Tracks a monitored metric across epochs and signals when to stop.

    Args:
        patience: Number of consecutive non-improving epochs to tolerate
            before signaling a stop. Must be a positive integer.
        mode: ``"min"`` if lower values of the monitored metric are
            better (e.g. loss), ``"max"`` if higher is better (e.g.
            ROC-AUC).
        min_delta: Minimum absolute change to count as an improvement.
            Defaults to ``0.0``.

    Raises:
        ValueError: If ``patience`` is not positive or ``mode`` is not
            one of ``{"min", "max"}``.
    """

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 0.0) -> None:
        if patience <= 0:
            raise ValueError(f"patience must be a positive integer, got {patience!r}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.patience = patience
        self.mode = mode
        self.min_delta = float(min_delta)

        self.best: Optional[float] = None
        self.num_bad_epochs: int = 0
        self.should_stop: bool = False

    def step(self, value: float) -> bool:
        """Register a new metric value; return True if training should stop.

        Args:
            value: The monitored metric's value for the current epoch.

        Returns:
            ``True`` once ``patience`` consecutive epochs have failed to
            improve on the best seen value by at least ``min_delta``;
            ``False`` otherwise.
        """
        if self.best is None or self._is_improvement(value):
            self.best = value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        self.should_stop = self.num_bad_epochs >= self.patience
        return self.should_stop

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < (self.best - self.min_delta)
        return value > (self.best + self.min_delta)
