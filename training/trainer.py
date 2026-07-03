"""
trainer.py

Purpose:
    The reusable training framework: optimizer/loss factories, metrics,
    and a generic ``Trainer`` that Centralized, LocalSGD, and FedAvg
    (future phases) will import and use as-is. Nothing in this file is
    specific to any one training strategy.

Phase 3 Scope:
    No Centralized/LocalSGD/FedAvg logic, no FastAPI, no Docker, no
    latency/fault-injection, no experiments. Only the framework pieces
    listed in the Phase 3 spec.

IMPORTANT -- Loss function incompatibility (Task 4):
    models/fraud_mlp.py's FraudMLP ends its forward pass with an
    explicit `nn.Sigmoid()`, so `forward()` returns probabilities in
    [0, 1], not raw logits. `nn.BCEWithLogitsLoss` internally applies
    its own sigmoid to its input; feeding it already-sigmoided
    probabilities double-applies the sigmoid, which:
      - compresses the effective input range and distorts gradients
        (most values get pushed toward the loss's linear-ish middle
        region instead of spanning the full logit range), and
      - is numerically inconsistent with how the model is evaluated
        elsewhere (FraudMLP.predict_proba / .predict both assume
        `forward()` output is already a probability).

    Per the spec, models/ is out of scope for Phase 3 (not modified),
    so this is documented rather than silently worked around. Two
    ways to resolve it existed:
      (a) Use `nn.BCELoss` against the model's probability output
          (mathematically correct for a Sigmoid-terminated model, no
          model changes needed), or
      (b) Invert the model's Sigmoid at loss-computation time
          (`logit = log(p / (1 - p))`) to force-fit
          `BCEWithLogitsLoss`.

    (b) is strictly worse than (a): it reduces to the same loss
    mathematically but adds a numerically unstable inverse-sigmoid
    (blows up as p -> 0 or 1) for no benefit, and couples this file to
    the model's internal Sigmoid choice. `create_loss_fn()` therefore
    defaults to (a), `"bce"` -> `nn.BCELoss`, and exposes
    `"bce_with_logits"` -> `nn.BCEWithLogitsLoss` only for a *future*
    model that outputs raw logits (at which point flipping the config
    string is the only change needed here).

Public Interface:
    Functions:
        create_optimizer(...) -> torch.optim.Optimizer
        create_loss_fn(...) -> torch.nn.Module
        compute_metrics(...) -> dict

    Classes:
        TrainerConfig
        Trainer
"""

from __future__ import annotations

import csv
import logging
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from training.utils import EarlyStopping, get_device, set_seed

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts"

_METRIC_LOG_FIELDS = [
    "epoch",
    "split",
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "learning_rate",
    "epoch_time_seconds",
]


# ----------------------------------------------------------------------
# TASK 3 -- Optimizer factory
# ----------------------------------------------------------------------

_OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def create_optimizer(
    params: Iterable[torch.nn.Parameter],
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """Build an optimizer by name.

    Args:
        params: Iterable of model parameters to optimize (e.g.
            ``model.parameters()``).
        optimizer_name: One of ``{"adam", "adamw", "sgd"}`` (case
            insensitive).
        lr: Learning rate.
        weight_decay: Weight decay (L2 penalty). Applies to all three
            supported optimizers.
        momentum: Momentum factor. Only used when
            ``optimizer_name == "sgd"``; ignored otherwise.
        **kwargs: Extra keyword arguments forwarded to the underlying
            ``torch.optim`` constructor (e.g. ``betas`` for Adam).

    Returns:
        A configured ``torch.optim.Optimizer``.

    Raises:
        ValueError: If ``optimizer_name`` is not supported.
    """
    key = optimizer_name.lower()
    if key not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"optimizer_name must be one of {sorted(_OPTIMIZER_REGISTRY)}, got {optimizer_name!r}"
        )

    if key == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs
        )
    return _OPTIMIZER_REGISTRY[key](params, lr=lr, weight_decay=weight_decay, **kwargs)


# ----------------------------------------------------------------------
# TASK 4 -- Loss function factory (see module docstring for rationale)
# ----------------------------------------------------------------------

_LOSS_REGISTRY = {
    "bce": nn.BCELoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
}


def create_loss_fn(loss_type: str = "bce", **kwargs: Any) -> nn.Module:
    """Build the training loss function.

    Defaults to ``"bce"`` (``nn.BCELoss``) because ``FraudMLP`` (Phase 2)
    outputs post-Sigmoid probabilities, not logits -- see the
    module-level docstring for why ``"bce_with_logits"`` would be
    incorrect against the current model. ``"bce_with_logits"`` is kept
    available for a future model variant that outputs raw logits.

    Args:
        loss_type: One of ``{"bce", "bce_with_logits"}``.
        **kwargs: Extra keyword arguments forwarded to the underlying
            ``torch.nn`` loss constructor (e.g. ``pos_weight``).

    Returns:
        An instantiated loss module.

    Raises:
        ValueError: If ``loss_type`` is not supported.
    """
    key = loss_type.lower()
    if key not in _LOSS_REGISTRY:
        raise ValueError(f"loss_type must be one of {sorted(_LOSS_REGISTRY)}, got {loss_type!r}")
    if key == "bce_with_logits":
        logger.warning(
            "create_loss_fn('bce_with_logits') assumes the model outputs raw logits. "
            "The current FraudMLP (Phase 2) outputs Sigmoid probabilities -- pairing it "
            "with BCEWithLogitsLoss double-applies sigmoid and is mathematically wrong. "
            "Use 'bce' unless the model has been changed to output logits."
        )
    return _LOSS_REGISTRY[key](**kwargs)


# ----------------------------------------------------------------------
# TASK 5 -- Metrics
# ----------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute classification metrics for binary fraud prediction.

    Args:
        y_true: Array-like of ground-truth binary labels, shape (N,) or
            (N, 1).
        y_prob: Array-like of predicted fraud probabilities in [0, 1],
            same shape as ``y_true``.
        threshold: Decision threshold applied to ``y_prob`` to obtain
            hard predictions for accuracy/precision/recall/F1.

    Returns:
        Dict with keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``roc_auc``, ``pr_auc``. ``roc_auc`` and ``pr_auc`` are
        undefined (mathematically, not a bug) when ``y_true`` contains
        only one class -- e.g. a tiny or heavily imbalanced batch. In
        that case they are reported as ``float("nan")`` and a warning is
        logged, rather than raising, so a single degenerate batch does
        not crash an otherwise-valid training run.

    Raises:
        ValueError: If ``y_true`` and ``y_prob`` have different lengths.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            f"y_true and y_prob must have the same length, got {y_true.shape[0]} vs {y_prob.shape[0]}"
        )

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=UndefinedMetricWarning)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except (ValueError, UndefinedMetricWarning):
            logger.warning("roc_auc undefined (y_true has a single class in this split); using NaN")
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except (ValueError, UndefinedMetricWarning):
            logger.warning("pr_auc undefined (y_true has a single class in this split); using NaN")
            metrics["pr_auc"] = float("nan")

    return metrics


# ----------------------------------------------------------------------
# TASK 6/7 -- Trainer
# ----------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Configuration for :class:`Trainer`. All fields are optional/defaulted.

    Attributes:
        artifacts_dir: Root directory for this Trainer's outputs
            (models/, logs/, metrics/ subdirectories are created under
            it). Defaults to the project's top-level ``artifacts/``.
            Callers running multiple training strategies or multiple
            data centers (a future phase) should pass distinct
            subdirectories (e.g. ``artifacts/dc_1``) so runs don't
            overwrite each other's checkpoints.
        grad_clip_norm: Max gradient L2 norm for clipping. ``None``
            disables clipping.
        use_amp: Whether to use CUDA mixed precision. ``None`` (default)
            auto-enables it iff CUDA is available; explicit ``True``/
            ``False`` overrides that.
        early_stopping_patience: Epochs to wait for improvement before
            stopping. ``None`` disables early stopping.
        early_stopping_mode: ``"min"`` or ``"max"`` for the monitored
            metric.
        seed: If set, seeds RNGs (see training.utils.set_seed) at the
            start of ``fit()``.
    """

    artifacts_dir: Path = _DEFAULT_ARTIFACTS_DIR
    grad_clip_norm: Optional[float] = None
    use_amp: Optional[bool] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_mode: str = "min"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.artifacts_dir = Path(self.artifacts_dir)
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive, got {self.grad_clip_norm!r}")
        if self.early_stopping_patience is not None and self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be positive, got {self.early_stopping_patience!r}"
            )
        if self.early_stopping_mode not in ("min", "max"):
            raise ValueError(
                f"early_stopping_mode must be 'min' or 'max', got {self.early_stopping_mode!r}"
            )


class Trainer(object):
    """Generic supervised training loop for ``FraudMLP``-shaped models.

    Owns the epoch loop, metric logging, checkpointing, gradient
    clipping, mixed precision, early stopping, and LR scheduling. Future
    training strategies (Centralized, LocalSGD, FedAvg) are expected to
    build/own their own model+optimizer and delegate the actual
    epoch-level training to an instance of this class, rather than
    reimplementing any of the above.

    Args:
        model: A ``torch.nn.Module`` whose ``forward(x)`` returns
            per-sample probabilities of shape ``(batch, 1)`` (matches
            ``FraudMLP``). Moved to the resolved device in ``__init__``.
        optimizer: A ``torch.optim.Optimizer`` already bound to
            ``model``'s parameters (e.g. via ``create_optimizer``).
        loss_fn: A loss module (e.g. via ``create_loss_fn``).
        device: Optional explicit device string. Auto-detected via
            ``training.utils.get_device`` if not given.
        config: Optional ``TrainerConfig``. Defaults to
            ``TrainerConfig()`` if not given.
        lr_scheduler: Optional ``torch.optim.lr_scheduler`` instance,
            already bound to ``optimizer``. Stepped once per epoch
            inside ``fit()``. ``ReduceLROnPlateau`` is stepped with the
            epoch's validation loss (if a val_loader was given) or
            training loss otherwise; any other scheduler type is
            stepped with no arguments.

    Raises:
        ValueError: If ``model`` has no ``input_dim`` attribute (used
            for batch validation).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: Optional[str] = None,
        config: Optional[TrainerConfig] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> None:
        if not hasattr(model, "input_dim"):
            raise ValueError(
                "model must expose an `input_dim` attribute (as FraudMLP does) "
                "so the Trainer can validate incoming batch shapes"
            )

        self.config = config or TrainerConfig()
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(self.device)
        self.lr_scheduler = lr_scheduler

        self.use_amp = (
            self.config.use_amp if self.config.use_amp is not None else self.device.type == "cuda"
        )
        if self.use_amp and self.device.type != "cuda":
            logger.warning("use_amp=True has no effect on device %r; AMP requires CUDA", self.device.type)
            self.use_amp = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.models_dir = self.config.artifacts_dir / "models"
        self.logs_dir = self.config.artifacts_dir / "logs"
        self.metrics_dir = self.config.artifacts_dir / "metrics"
        for d in (self.models_dir, self.logs_dir, self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.metrics_csv_path = self.metrics_dir / "training_metrics.csv"
        self.best_model_path = self.models_dir / "best_model.pt"
        self.last_checkpoint_path = self.models_dir / "last_checkpoint.pt"

        self.early_stopper: Optional[EarlyStopping] = None
        if self.config.early_stopping_patience is not None:
            self.early_stopper = EarlyStopping(
                patience=self.config.early_stopping_patience, mode=self.config.early_stopping_mode
            )

        self._best_metric: Optional[float] = None
        self.history: List[Dict[str, float]] = []

        logger.info("Trainer ready on device=%s, amp=%s, artifacts_dir=%s", self.device, self.use_amp, self.config.artifacts_dir)

    # ------------------------------------------------------------------
    # TASK 11 -- Validation helpers
    # ------------------------------------------------------------------

    def _validate_batch(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.dim() != 2:
            raise ValueError(f"Expected 2D feature batch (batch, features), got shape {tuple(features.shape)}")
        if features.shape[1] != self.model.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: batch has {features.shape[1]} features, "
                f"model.input_dim={self.model.input_dim}"
            )
        if labels.dim() != 2 or labels.shape[1] != 1:
            raise ValueError(f"Expected labels of shape (batch, 1), got {tuple(labels.shape)}")
        if labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"Batch size mismatch between features ({features.shape[0]}) and labels ({labels.shape[0]})"
            )

    # ------------------------------------------------------------------
    # TASK 6 -- Core epoch loops
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run one training epoch (forward + backward + optimizer step per batch).

        Args:
            dataloader: Yields ``(features, labels)`` batches, e.g. from
                ``create_dataloader(FraudDataset(...), shuffle=True)``.

        Returns:
            Dict of aggregated epoch metrics (loss + compute_metrics
            output), plus ``learning_rate`` and ``epoch_time_seconds``.
        """
        self.model.train()
        return self._run_epoch(dataloader, train=True)

    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run one evaluation epoch with gradients disabled (no optimizer step).

        Args:
            dataloader: Yields ``(features, labels)`` batches, typically
                with ``shuffle=False``.

        Returns:
            Dict of aggregated epoch metrics, same shape as
            ``train_epoch``'s return value.
        """
        self.model.eval()
        return self._run_epoch(dataloader, train=False)

    def _run_epoch(self, dataloader: DataLoader, train: bool) -> Dict[str, float]:
        start = time.time()
        total_loss = 0.0
        total_samples = 0
        all_labels: List[np.ndarray] = []
        all_probs: List[np.ndarray] = []

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for features, labels in dataloader:
                self._validate_batch(features, labels)
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                batch_size = features.shape[0]

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    probs = self.model(features)
                    loss = self.loss_fn(probs, labels)

                if train:
                    self.scaler.scale(loss).backward()
                    if self.config.grad_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * batch_size
                total_samples += batch_size
                all_labels.append(labels.detach().cpu().numpy())
                all_probs.append(probs.detach().cpu().numpy())

        if total_samples == 0:
            raise ValueError("dataloader produced zero samples; cannot compute epoch metrics")

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        metrics = compute_metrics(y_true, y_prob)
        metrics["loss"] = total_loss / total_samples
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        metrics["epoch_time_seconds"] = time.time() - start
        return metrics

    # ------------------------------------------------------------------
    # TASK 6/7 -- fit / evaluate / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_metric: str = "loss",
        resume_from: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, float]]:
        """Run the full training loop.

        Args:
            train_loader: Training batches.
            val_loader: Optional validation batches. If given,
                ``validate_epoch`` runs each epoch and its metrics
                (prefixed nowhere -- see logged CSV's ``split`` column)
                drive early stopping / best-checkpoint selection. If
                omitted, training metrics drive both instead.
            epochs: Number of epochs to train for (in addition to any
                epochs already completed if resuming).
            early_stopping_metric: Key into the epoch metrics dict
                (e.g. ``"loss"``, ``"f1"``, ``"roc_auc"``) used for
                early stopping and best-checkpoint selection.
            resume_from: Optional checkpoint path to resume from before
                training starts (see ``load_checkpoint``).

        Returns:
            List of per-epoch metric dicts (one train dict, and one val
            dict if ``val_loader`` was given, per epoch) -- the same
            records written to ``training_metrics.csv``.
        """
        if self.config.seed is not None:
            set_seed(self.config.seed)

        start_epoch = 0
        if resume_from is not None:
            state = self.load_checkpoint(resume_from)
            start_epoch = state["epoch"] + 1
            logger.info("Resumed from %s at epoch %d", resume_from, start_epoch)

        write_header = not self.metrics_csv_path.exists()
        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            csv_writer = csv.DictWriter(f, fieldnames=_METRIC_LOG_FIELDS)
            if write_header:
                csv_writer.writeheader()

            for epoch in range(start_epoch, start_epoch + epochs):
                train_metrics = self.train_epoch(train_loader)
                self._log_epoch(csv_writer, f, epoch, "train", train_metrics)
                self.history.append({"epoch": epoch, "split": "train", **train_metrics})

                monitored = train_metrics
                if val_loader is not None:
                    val_metrics = self.validate_epoch(val_loader)
                    self._log_epoch(csv_writer, f, epoch, "val", val_metrics)
                    self.history.append({"epoch": epoch, "split": "val", **val_metrics})
                    monitored = val_metrics

                logger.info(
                    "epoch %d: train_loss=%.4f%s",
                    epoch,
                    train_metrics["loss"],
                    f" val_loss={monitored['loss']:.4f}" if val_loader is not None else "",
                )

                if early_stopping_metric not in monitored:
                    raise ValueError(
                        f"early_stopping_metric {early_stopping_metric!r} not in computed "
                        f"metrics {sorted(monitored)}"
                    )
                metric_value = monitored[early_stopping_metric]
                is_best = self._best_metric is None or self._is_better(metric_value)
                if is_best:
                    self._best_metric = metric_value
                    self.save_checkpoint(self.best_model_path, epoch, extra={"metrics": monitored})

                self.save_checkpoint(self.last_checkpoint_path, epoch, extra={"metrics": monitored})

                if self.lr_scheduler is not None:
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(metric_value)
                    else:
                        self.lr_scheduler.step()

                if self.early_stopper is not None and self.early_stopper.step(metric_value):
                    logger.info("Early stopping triggered at epoch %d (no improvement for %d epochs)", epoch, self.early_stopper.patience)
                    break

        return self.history

    def _is_better(self, value: float) -> bool:
        if self.config.early_stopping_mode == "min":
            return value < self._best_metric
        return value > self._best_metric

    def _log_epoch(self, csv_writer, f, epoch: int, split: str, metrics: Dict[str, float]) -> None:
        row = {"epoch": epoch, "split": split, **{k: metrics.get(k) for k in _METRIC_LOG_FIELDS if k not in ("epoch", "split")}}
        csv_writer.writerow(row)
        f.flush()

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Final, no-grad evaluation pass. Semantically identical to
        ``validate_epoch`` -- separate method so callers have a
        stable, purpose-named entry point that doesn't imply it's part
        of the per-epoch training loop.

        Args:
            dataloader: Batches to evaluate on.

        Returns:
            Dict of aggregated metrics, same shape as ``train_epoch``.
        """
        return self.validate_epoch(dataloader)

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Run inference and return predicted probabilities (no labels needed).

        Args:
            dataloader: Yields either ``(features, labels)`` or bare
                ``features`` batches; labels (if present) are ignored.

        Returns:
            1D numpy array of predicted fraud probabilities, in the
            order the dataloader produced batches (do not use
            ``shuffle=True`` if order must match the source dataset).
        """
        self.model.eval()
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in dataloader:
                features = batch[0] if isinstance(batch, (list, tuple)) else batch
                features = features.to(self.device, non_blocking=True)
                probs = self.model(features)
                all_probs.append(probs.detach().cpu().numpy())
        return np.concatenate(all_probs).reshape(-1)

    # ------------------------------------------------------------------
    # TASK 6/7 -- Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, path: Union[str, Path], epoch: int, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model/optimizer/scheduler/scaler state to ``path``.

        Args:
            path: Destination file path. Parent directories are created
                if needed.
            epoch: Epoch index to record in the checkpoint.
            extra: Optional extra JSON-serializable-ish data to store
                alongside the checkpoint (e.g. the epoch's metrics).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "extra": extra or {},
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load a checkpoint written by ``save_checkpoint`` into this Trainer.

        Restores model weights, optimizer state, and (if present at
        both save and load time) scaler / lr_scheduler state.

        Args:
            path: Path to a checkpoint file.

        Returns:
            The raw checkpoint dict (``epoch``, ``extra``, ...), so
            callers (e.g. ``fit(resume_from=...)``) can inspect it.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.use_amp and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if self.lr_scheduler is not None and checkpoint.get("lr_scheduler_state_dict") is not None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        return checkpoint
