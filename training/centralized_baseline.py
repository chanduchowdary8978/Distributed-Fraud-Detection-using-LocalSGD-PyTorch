"""
centralized_baseline.py

Purpose:
    Train the fraud detection model on the full, unpartitioned dataset
    (all five data-center shards combined) in a single process. This is
    the accuracy upper-bound reference baseline that LocalSGD and
    FedAvg (future phases) are compared against.

Phase 4 Scope:
    This file only assembles Phase 1-3 building blocks -- it contains no
    training loop of its own. Data loading/splitting and config/metrics
    plumbing are the only new logic here; the epoch loop, optimizer,
    loss, and metrics all come from training/trainer.py unmodified.

Design decisions worth flagging explicitly (not silently assumed):

    1. FraudDataset (training/dataset.py) only accepts a CSV file path
       -- it has no in-memory / DataFrame constructor, and it cannot be
       modified in this phase. Combining five shards into "one
       centralized dataset" therefore means: concatenate the five raw
       shard DataFrames in memory, split them, and write the two splits
       out as CSV files, which FraudDataset then reads like any other
       shard. Nothing under data/ is touched -- the intermediate files
       are written to artifacts/centralized/data/.

    2. Task 7 lists checkpoint/metric paths as flat under
       artifacts/centralized/ (e.g. "artifacts/centralized/best_model.pt").
       training/trainer.py -- which this phase must reuse unmodified --
       actually writes checkpoints to
       "{artifacts_dir}/models/{best_model.pt,last_checkpoint.pt}" and
       training_metrics.csv to "{artifacts_dir}/metrics/". Reproducing
       the flat layout would require either duplicating Trainer's
       checkpoint/logging code (forbidden: "do NOT duplicate the
       training loop") or editing trainer.py (forbidden outright). The
       correct resolution is to defer to Trainer's existing, tested
       contract rather than fight it: with
       artifacts_dir=artifacts/centralized, the real paths are
       artifacts/centralized/models/best_model.pt,
       artifacts/centralized/models/last_checkpoint.pt, and
       artifacts/centralized/metrics/training_metrics.csv.
       evaluation_metrics.json and centralized_config.json are written
       by *this* file (not Trainer), so those two do land exactly at
       the flat paths Task 7 specifies:
       artifacts/centralized/evaluation_metrics.json and
       artifacts/centralized/centralized_config.json.

Public Interface:
    CentralizedTrainer

    Methods:
        train() -> List[Dict[str, float]]
        evaluate() -> Dict[str, float]
        run() -> Dict[str, Any]
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Running this file directly (`python training/centralized_baseline.py`, per
# the Phase 4 acceptance criteria) puts only this file's own directory
# (training/) on sys.path, not the repository root -- so the `data.*` /
# `models.*` / `training.*` absolute imports below would otherwise fail.
# Running it as a module (`python -m training.centralized_baseline`) or via
# an already-installed package does not need this, so the insert is guarded.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data.partition_regions import PROCESSED_DIR, SHARD_NAMES
from models.fraud_mlp import FraudMLP
from training.dataset import FraudDataset, create_dataloader
from training.trainer import Trainer, TrainerConfig, create_loss_fn, create_optimizer
from training.utils import set_seed

logger = logging.getLogger(__name__)

_DEFAULT_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts" / "centralized"

_SCHEDULER_REGISTRY = {
    "step": torch.optim.lr_scheduler.StepLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}


def _create_scheduler(optimizer, scheduler_name: Optional[str], scheduler_kwargs: Dict[str, Any]):
    """Build an optional LR scheduler by name for Trainer to step each epoch.

    Not part of the Phase 3 framework (Trainer accepts a pre-built
    scheduler rather than constructing one), so this small factory lives
    here rather than in training/trainer.py.

    Args:
        optimizer: Optimizer the scheduler will adjust.
        scheduler_name: One of ``{None, "step", "plateau", "cosine"}``.
        scheduler_kwargs: Keyword arguments forwarded to the underlying
            ``torch.optim.lr_scheduler`` constructor.

    Returns:
        A scheduler instance, or ``None`` if ``scheduler_name`` is ``None``.

    Raises:
        ValueError: If ``scheduler_name`` is not supported.
    """
    if scheduler_name is None:
        return None
    key = scheduler_name.lower()
    if key not in _SCHEDULER_REGISTRY:
        raise ValueError(f"scheduler_name must be one of {sorted(_SCHEDULER_REGISTRY)} or None, got {scheduler_name!r}")
    return _SCHEDULER_REGISTRY[key](optimizer, **scheduler_kwargs)


@dataclass
class CentralizedConfig:
    """All configurable knobs for the centralized baseline (Task 5).

    Attributes:
        val_split: Fraction of the combined dataset held out for
            validation.
        seed: Random seed for the train/val split and all RNGs (see
            training.utils.set_seed).
        batch_size: DataLoader batch size for both splits.
        num_workers: DataLoader worker processes.
        optimizer_name: One of ``{"adam", "adamw", "sgd"}``.
        lr: Learning rate.
        weight_decay: Optimizer weight decay.
        momentum: SGD momentum (ignored for adam/adamw).
        loss_type: One of ``{"bce", "bce_with_logits"}`` -- see
            training/trainer.py's module docstring; "bce" is correct
            for the current Sigmoid-terminated FraudMLP.
        scheduler_name: One of ``{None, "step", "plateau", "cosine"}``.
        scheduler_kwargs: Kwargs for the chosen scheduler.
        epochs: Training epochs.
        grad_clip_norm: Max gradient norm, or ``None`` to disable.
        early_stopping_patience: Epochs of no improvement before
            stopping, or ``None`` to disable.
        early_stopping_metric: Metric key early stopping monitors.
        early_stopping_mode: ``"min"`` or ``"max"`` for that metric.
        use_amp: ``None`` auto-enables mixed precision iff CUDA is
            available; explicit ``True``/``False`` overrides that.
        hidden_dims: FraudMLP hidden layer widths.
        dropout_rate: FraudMLP dropout rate.
        activation: FraudMLP activation function name.
        use_batch_norm: FraudMLP batch norm toggle.
        use_dropout: FraudMLP dropout toggle.
        artifacts_dir: Root directory for this run's artifacts.
    """

    val_split: float = 0.2
    seed: int = 42

    batch_size: int = 64
    num_workers: int = 0

    optimizer_name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9

    loss_type: str = "bce"

    scheduler_name: Optional[str] = None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    epochs: int = 20
    grad_clip_norm: Optional[float] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_metric: str = "loss"
    early_stopping_mode: str = "min"
    use_amp: Optional[bool] = None

    hidden_dims: Sequence[int] = (128, 64)
    dropout_rate: float = 0.3
    activation: str = "relu"
    use_batch_norm: bool = True
    use_dropout: bool = True

    artifacts_dir: Path = _DEFAULT_ARTIFACTS_DIR

    def __post_init__(self) -> None:
        self.artifacts_dir = Path(self.artifacts_dir)
        if not (0.0 < self.val_split < 1.0):
            raise ValueError(f"val_split must be in (0.0, 1.0), got {self.val_split!r}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs!r}")


class CentralizedTrainer:
    """Centralized training baseline: all five shards combined, one model.

    Args:
        config: Optional ``CentralizedConfig``. Defaults to
            ``CentralizedConfig()`` if not given.
    """

    def __init__(self, config: Optional[CentralizedConfig] = None) -> None:
        self.config = config or CentralizedConfig()

        self.model: Optional[FraudMLP] = None
        self.trainer: Optional[Trainer] = None
        self.val_loader = None
        self.dataset_stats: Dict[str, Any] = {}
        self.history: List[Dict[str, float]] = []
        self._eval_metrics: Optional[Dict[str, float]] = None
        self._train_duration_seconds: Optional[float] = None

    # ------------------------------------------------------------------
    # TASK 1/2 -- Load, merge, split
    # ------------------------------------------------------------------

    def _load_and_split(self) -> tuple[Path, Path]:
        """Load all five shards, merge, stratified-split, write two CSVs.

        Returns:
            (train_csv_path, val_csv_path)

        Raises:
            FileNotFoundError: If any shard CSV is missing.
            ValueError: If the combined dataset is empty or missing the
                label column.
        """
        shard_paths = [PROCESSED_DIR / f"{name}.csv" for name in SHARD_NAMES]
        missing = [p for p in shard_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing processed shard(s): {missing}. Run data/partition_regions.py first."
            )

        shard_frames = [pd.read_csv(p) for p in shard_paths]
        combined_df = pd.concat(shard_frames, ignore_index=True)
        if combined_df.empty:
            raise ValueError("Combined dataset is empty after merging all shards")
        if "isFraud" not in combined_df.columns:
            raise ValueError("Combined dataset is missing the 'isFraud' label column")

        try:
            train_df, val_df = train_test_split(
                combined_df,
                test_size=self.config.val_split,
                stratify=combined_df["isFraud"],
                random_state=self.config.seed,
            )
        except ValueError as exc:
            logger.warning(
                "Stratified split failed (%s); falling back to a non-stratified split", exc
            )
            train_df, val_df = train_test_split(
                combined_df, test_size=self.config.val_split, random_state=self.config.seed
            )

        data_dir = self.config.artifacts_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        train_csv_path = data_dir / "train.csv"
        val_csv_path = data_dir / "val.csv"
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)

        self.dataset_stats = {
            "total_samples": len(combined_df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "total_fraud": int(combined_df["isFraud"].sum()),
            "train_fraud": int(train_df["isFraud"].sum()),
            "val_fraud": int(val_df["isFraud"].sum()),
        }
        logger.info(
            "Merged %d shards: total=%d, train=%d, val=%d",
            len(shard_paths),
            self.dataset_stats["total_samples"],
            self.dataset_stats["train_samples"],
            self.dataset_stats["val_samples"],
        )
        return train_csv_path, val_csv_path

    # ------------------------------------------------------------------
    # TASK 3/4/9/10 -- Build model + framework components, train
    # ------------------------------------------------------------------

    def train(self) -> List[Dict[str, float]]:
        """Load data, build the model, and train via training.trainer.Trainer.

        Returns:
            The training history (per-epoch metric dicts), as returned
            by ``Trainer.fit``.

        Raises:
            ValueError: If the model's input dimension does not match
                the dataset's feature dimension, or if the two splits
                disagree on feature dimension.
            RuntimeError: If training produced no history, or if
                checkpoints were not written to disk.
        """
        set_seed(self.config.seed)  # before model construction, for reproducible init

        train_csv_path, val_csv_path = self._load_and_split()
        train_dataset = FraudDataset(train_csv_path)
        val_dataset = FraudDataset(val_csv_path)

        if train_dataset.feature_dim != val_dataset.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch between splits: "
                f"train={train_dataset.feature_dim}, val={val_dataset.feature_dim}"
            )
        self.dataset_stats["feature_dim"] = train_dataset.feature_dim
        logger.info("Feature dimension: %d", train_dataset.feature_dim)

        self.model = FraudMLP(
            input_dim=train_dataset.feature_dim,
            hidden_dims=list(self.config.hidden_dims),
            dropout_rate=self.config.dropout_rate,
            activation=self.config.activation,
            use_batch_norm=self.config.use_batch_norm,
            use_dropout=self.config.use_dropout,
        )
        if self.model.input_dim != train_dataset.feature_dim:
            raise ValueError(
                f"Model input_dim ({self.model.input_dim}) does not match dataset "
                f"feature_dim ({train_dataset.feature_dim})"
            )
        logger.info("Model configuration: %s", self.model.config)

        optimizer = create_optimizer(
            self.model.parameters(),
            optimizer_name=self.config.optimizer_name,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            momentum=self.config.momentum,
        )
        logger.info(
            "Optimizer configuration: name=%s lr=%s weight_decay=%s momentum=%s",
            self.config.optimizer_name, self.config.lr, self.config.weight_decay, self.config.momentum,
        )
        loss_fn = create_loss_fn(self.config.loss_type)
        scheduler = _create_scheduler(optimizer, self.config.scheduler_name, self.config.scheduler_kwargs)

        trainer_config = TrainerConfig(
            artifacts_dir=self.config.artifacts_dir,
            grad_clip_norm=self.config.grad_clip_norm,
            use_amp=self.config.use_amp,
            early_stopping_patience=self.config.early_stopping_patience,
            early_stopping_mode=self.config.early_stopping_mode,
            seed=self.config.seed,
        )
        self.trainer = Trainer(self.model, optimizer, loss_fn, config=trainer_config, lr_scheduler=scheduler)

        train_loader = create_dataloader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers
        )
        self.val_loader = create_dataloader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers
        )

        start = time.time()
        self.history = self.trainer.fit(
            train_loader,
            val_loader=self.val_loader,
            epochs=self.config.epochs,
            early_stopping_metric=self.config.early_stopping_metric,
        )
        self._train_duration_seconds = time.time() - start
        logger.info("Training duration: %.2f seconds", self._train_duration_seconds)

        if not self.history:
            raise RuntimeError("Trainer.fit() returned an empty history; training did not run")
        if not self.trainer.best_model_path.exists() or not self.trainer.last_checkpoint_path.exists():
            raise RuntimeError(
                f"Expected checkpoints not found: {self.trainer.best_model_path}, "
                f"{self.trainer.last_checkpoint_path}"
            )

        return self.history

    # ------------------------------------------------------------------
    # TASK 6 -- Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained model on the held-out validation split.

        Reuses ``Trainer.evaluate`` (which reuses ``compute_metrics``);
        no metric computation is duplicated here.

        Returns:
            Dict with keys: loss, accuracy, precision, recall, f1,
            roc_auc, pr_auc, learning_rate, epoch_time_seconds.

        Raises:
            RuntimeError: If called before ``train()``.
        """
        if self.trainer is None or self.val_loader is None:
            raise RuntimeError("evaluate() called before train(); call train() first")

        self._eval_metrics = self.trainer.evaluate(self.val_loader)
        logger.info("Final evaluation metrics: %s", self._eval_metrics)

        eval_path = self.config.artifacts_dir / "evaluation_metrics.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(self._eval_metrics, f, indent=2)

        return self._eval_metrics

    # ------------------------------------------------------------------
    # TASK 7/8 -- Orchestration, config persistence, logging summary
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the full pipeline: train(), evaluate(), save configuration.

        Returns:
            Summary dict with dataset_stats, final evaluation metrics,
            training duration, and the resolved configuration.
        """
        self.train()
        eval_metrics = self.evaluate()

        config_dict = asdict(self.config)
        config_dict["artifacts_dir"] = str(self.config.artifacts_dir)
        config_dict["model"] = self.model.config
        config_dict["train_duration_seconds"] = self._train_duration_seconds

        config_path = self.config.artifacts_dir / "centralized_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        summary = {
            "dataset_stats": self.dataset_stats,
            "evaluation_metrics": eval_metrics,
            "train_duration_seconds": self._train_duration_seconds,
            "config": config_dict,
        }
        logger.info(
            "Centralized baseline complete: total=%d train=%d val=%d feature_dim=%d duration=%.2fs",
            self.dataset_stats["total_samples"],
            self.dataset_stats["train_samples"],
            self.dataset_stats["val_samples"],
            self.dataset_stats["feature_dim"],
            self._train_duration_seconds,
        )
        return summary


def main() -> None:
    """CLI entry point: run the centralized baseline end-to-end."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    CentralizedTrainer().run()


if __name__ == "__main__":
    main()
