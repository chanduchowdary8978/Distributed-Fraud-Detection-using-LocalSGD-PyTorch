"""
local_sgd.py

Purpose:
    Implement the LocalSGD distributed training strategy: one worker per
    processed data-center shard trains locally for K epochs, after which
    all workers' parameters are averaged and broadcast back out. This is
    a purely logical, single-process simulation of multi-data-center
    LocalSGD -- no networking, RPC, or multiprocessing is involved.

Phase 5 Scope:
    This file only orchestrates Phase 0-4 building blocks -- it contains
    no training-loop, optimizer, loss, metric, or model logic of its
    own. Per-epoch training and evaluation are delegated entirely to
    ``training.trainer.Trainer`` (one Trainer instance per worker); the
    only new logic here is worker construction, weight cloning, the
    averaging/broadcast step, and config/metrics/checkpoint plumbing.

Design decisions worth flagging explicitly (not silently assumed):

    1. Mirrors training/centralized_baseline.py's structure and
       artifact-path resolution rationale: Trainer (unmodified, reused
       as-is) writes checkpoints to
       "{artifacts_dir}/models/{best_model.pt,last_checkpoint.pt}" and
       "{artifacts_dir}/metrics/training_metrics.csv", not to flat
       paths. Each worker gets its own artifacts_dir
       (artifacts/local_sgd/workers/<shard_name>/) so five Trainers
       never collide on the same checkpoint files. The *global* model's
       checkpoints/metrics/config (best_global_model.pt,
       last_global_model.pt, training_metrics.csv,
       local_sgd_config.json, evaluation_metrics.json) are written by
       *this* file directly under artifacts/local_sgd/{models,metrics}/,
       matching Task 8's flat layout.

    2. "Reuse Trainer.train_epoch()" (Task 3) is taken literally: each
       worker's Trainer.train_epoch() is called K times per
       communication round (not Trainer.fit(), which owns its own
       early-stopping/best-checkpoint bookkeeping oriented around a
       single, non-distributed run). Per-epoch worker losses are still
       logged (Task 9) and appended to that worker's own history.

    3. "Reuse Trainer.evaluate()" (Task 7) is applied to a global
       evaluation Trainer that wraps the global model -- constructed
       once and re-synced with the averaged weights after every
       round -- rather than instantiating a fresh Trainer per round,
       so evaluation checkpoint bookkeeping stays consistent.

    4. Global evaluation uses a held-out validation split carved out of
       the five shards the same way training/centralized_baseline.py
       does (stratified train/val split, written to
       artifacts/local_sgd/data/), since FraudDataset only reads from a
       CSV path and none of the raw shard CSVs are themselves
       validation-only. Nothing under data/ is touched.

    5. Deterministic synchronization (Task 4) means: parameters are
       averaged in the fixed SHARD_NAMES order with a plain arithmetic
       mean (equal weighting per Task 4's "average every trainable
       parameter equally"), performed under torch.no_grad() with no
       reliance on floating-point-order-dependent reductions beyond
       ordinary IEEE-754 summation, which is itself deterministic for a
       fixed input order.

Phase 7.5 Scope (Network Monitoring Integration):
    LocalSGDTrainer now automatically drives the Phase 7 network layer
    (network.network_simulator.NetworkSimulator,
    network.communication.CommunicationManager) and
    monitoring.system_monitor.SystemMonitor around its existing
    synchronize()/train() logic -- no changes to the averaging math or
    training loop structure itself, no manual calls required by
    callers. Both are dependency-injectable (constructor arguments)
    so experiments/experiment_runner.py can inject a single shared
    SystemMonitor per run and standalone use (`python
    training/local_sgd.py`) still works with sensible defaults built
    automatically. See LocalSGDTrainer.__init__,
    _build_network_layer, and the network/monitoring calls inside
    train()/run().

Public Interface:
    LocalSGDTrainer

    Methods:
        train() -> Dict[str, Any]
        synchronize() -> Dict[str, float]
        evaluate() -> Dict[str, float]
        run() -> Dict[str, Any]
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Running this file directly (`python training/local_sgd.py`, per the
# Phase 5 acceptance criteria) puts only this file's own directory
# (training/) on sys.path, not the repository root -- so the `data.*` /
# `models.*` / `training.*` absolute imports below would otherwise fail.
# Running it as a module (`python -m training.local_sgd`) or via an
# already-installed package does not need this, so the insert is guarded.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from data.partition_regions import PROCESSED_DIR, SHARD_NAMES
from models.fraud_mlp import FraudMLP
from training.dataset import FraudDataset, create_dataloader
from training.trainer import (
    Trainer,
    TrainerConfig,
    create_loss_fn,
    create_optimizer,
)
from training.utils import set_seed

logger = logging.getLogger(__name__)

_DEFAULT_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts" / "local_sgd"

_SCHEDULER_REGISTRY = {
    "step": torch.optim.lr_scheduler.StepLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}


def _create_scheduler(optimizer, scheduler_name: Optional[str], scheduler_kwargs: Dict[str, Any]):
    """Build an optional LR scheduler by name for a worker's Trainer to step.

    Not part of the Phase 3 framework (Trainer accepts a pre-built
    scheduler rather than constructing one), so this small factory is
    duplicated here from centralized_baseline.py's identical helper
    rather than importing a private helper across sibling strategy
    modules.

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
        raise ValueError(
            f"scheduler_name must be one of {sorted(_SCHEDULER_REGISTRY)} or None, got {scheduler_name!r}"
        )
    return _SCHEDULER_REGISTRY[key](optimizer, **scheduler_kwargs)


@dataclass
class LocalSGDConfig:
    """All configurable knobs for LocalSGD (Task 6).

    Attributes:
        local_epochs: K -- number of local epochs each worker trains
            for between synchronization rounds.
        communication_rounds: Number of synchronization rounds to run.
        val_split: Fraction of each shard held out (and combined) for
            global evaluation. The held-out rows are removed from every
            worker's local training data before shards are written out,
            so validation rows are never trained on by any worker.
        seed: Random seed for the train/val split and all RNGs (see
            training.utils.set_seed). Also what makes model
            initialization and synchronization deterministic.
        batch_size: DataLoader batch size for every worker and for
            global evaluation.
        num_workers: DataLoader worker processes.
        optimizer_name: One of ``{"adam", "adamw", "sgd"}``, applied
            identically to every worker.
        lr: Learning rate, applied identically to every worker.
        weight_decay: Optimizer weight decay.
        momentum: SGD momentum (ignored for adam/adamw).
        loss_type: One of ``{"bce", "bce_with_logits"}`` -- see
            training/trainer.py's module docstring; "bce" is correct
            for the current Sigmoid-terminated FraudMLP.
        scheduler_name: One of ``{None, "step", "plateau", "cosine"}``,
            applied identically to every worker.
        scheduler_kwargs: Kwargs for the chosen scheduler.
        grad_clip_norm: Max gradient norm, or ``None`` to disable.
        early_stopping_patience: Rounds of no improvement in the
            global evaluation metric before stopping early, or
            ``None`` to disable. Evaluated against the *global* model
            after each synchronization round (not per-worker).
        early_stopping_metric: Metric key (from compute_metrics) early
            stopping and best-checkpoint selection monitor.
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

    local_epochs: int = 1
    communication_rounds: int = 10

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

    grad_clip_norm: Optional[float] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_metric: str = "loss"
    early_stopping_mode: str = "min"
    use_amp: Optional[bool] = False

    hidden_dims: Sequence[int] = (128, 64)
    dropout_rate: float = 0.3
    activation: str = "relu"
    use_batch_norm: bool = True
    use_dropout: bool = True

    artifacts_dir: Path = _DEFAULT_ARTIFACTS_DIR

    # ------------------------------------------------------------------
    # Phase 7.5 -- Network Monitoring Integration (see module docstring
    # "Phase 7.5 Scope" below). All fields are plain, JSON-serializable
    # values (no live objects) so LocalSGDConfig keeps working with
    # asdict()/json.dump() unchanged; the live NetworkSimulator /
    # CommunicationManager / SystemMonitor objects themselves are
    # dependency-injected into LocalSGDTrainer's constructor instead
    # (see LocalSGDTrainer.__init__), not stored on this dataclass.
    # ------------------------------------------------------------------
    enable_network_simulation: bool = True
    """Whether every synchronization round automatically also drives
    network.communication.CommunicationManager (Phase 7.5, Task 2).
    Disabling this only turns off the *simulated-network* accounting;
    the actual parameter averaging in synchronize() is unaffected."""

    network_config_path: Optional[str] = None
    """Path to a config/network/*.yaml file (see
    network.network_simulator.NetworkSimulator.from_config). Defaults
    to config/network/fully_connected.yaml, which already matches
    SHARD_NAMES's 5 workers -- see LocalSGDTrainer._build_network_layer."""

    def __post_init__(self) -> None:
        self.artifacts_dir = Path(self.artifacts_dir)
        if self.local_epochs <= 0:
            raise ValueError(f"local_epochs must be positive, got {self.local_epochs!r}")
        if self.communication_rounds <= 0:
            raise ValueError(
                f"communication_rounds must be positive, got {self.communication_rounds!r}"
            )
        if not (0.0 < self.val_split < 1.0):
            raise ValueError(f"val_split must be in (0.0, 1.0), got {self.val_split!r}")


class LocalSGDTrainer:
    """LocalSGD distributed training strategy: K local epochs per worker,
    then average-and-broadcast parameter synchronization, repeated for a
    configurable number of communication rounds.

    Args:
        config: Optional ``LocalSGDConfig``. Defaults to
            ``LocalSGDConfig()`` if not given.
        communication_manager: Optional, pre-built
            ``network.communication.CommunicationManager`` (Phase 7.5).
            If omitted and ``config.enable_network_simulation`` is
            ``True`` (the default), one is built automatically from
            ``config.network_config_path`` the first time ``train()``
            runs -- no manual call is required either way. Passing one
            in (the pattern experiments.experiment_runner.ExperimentRunner
            uses) lets a caller own/export a single simulator instance
            across a run.
        system_monitor: Optional, pre-built
            ``monitoring.system_monitor.SystemMonitor`` (Phase 7.5).
            If given, this trainer records synchronization
            duration/communication cost into it during training but
            does NOT call ``start()``/``stop()``/``mark_training_*()``
            on it (the injecting caller owns that lifecycle -- see
            ExperimentRunner). If omitted, a monitor is built and its
            full lifecycle (start/mark/stop/export) is owned by this
            trainer, so standalone use still gets full monitoring with
            no manual calls required.
    """

    def __init__(
        self,
        config: Optional[LocalSGDConfig] = None,
        communication_manager: Optional[Any] = None,
        system_monitor: Optional[Any] = None,
    ) -> None:
        self.config = config or LocalSGDConfig()

        # Per-worker state, index-aligned with self.shard_names.
        self.shard_names: List[str] = list(SHARD_NAMES)
        self.worker_models: List[FraudMLP] = []
        self.worker_trainers: List[Trainer] = []
        self.worker_loaders: List[Any] = []

        self.global_model: Optional[FraudMLP] = None
        self.global_trainer: Optional[Trainer] = None
        self.val_loader = None

        self.dataset_stats: Dict[str, Any] = {}
        self.round_metrics: List[Dict[str, Any]] = []
        self._eval_metrics: Optional[Dict[str, float]] = None
        self._train_duration_seconds: Optional[float] = None
        self._sync_durations: List[float] = []

        self._feature_dim: Optional[int] = None

        # Phase 7.5 -- Network Monitoring Integration. communication_manager
        # is None until _build_network_layer() runs (lazily, inside
        # train()) unless the caller already injected one.
        self.communication_manager = communication_manager
        self.system_monitor = system_monitor
        self._owns_system_monitor = system_monitor is None
        self._network_summary: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # TASK 10 -- Validation helpers
    # ------------------------------------------------------------------

    def _validate_shards_exist(self) -> List[Path]:
        """Confirm every shard CSV exists before doing any work.

        Returns:
            List of shard CSV paths, index-aligned with self.shard_names.

        Raises:
            FileNotFoundError: If any shard CSV is missing.
        """
        shard_paths = [PROCESSED_DIR / f"{name}.csv" for name in self.shard_names]
        missing = [p for p in shard_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing processed shard(s): {missing}. Run data/partition_regions.py first."
            )
        return shard_paths

    def _validate_identical_init(self) -> None:
        """Confirm every worker model started from identical initial weights.

        Raises:
            RuntimeError: If any worker's initial parameters differ from
                worker 0's, or if no workers exist.
        """
        if not self.worker_models:
            raise RuntimeError("No worker models were constructed; cannot validate initialization")

        reference_state = self.worker_models[0].state_dict()
        for shard_name, model in zip(self.shard_names[1:], self.worker_models[1:]):
            state = model.state_dict()
            for key, ref_tensor in reference_state.items():
                if not torch.equal(ref_tensor, state[key]):
                    raise RuntimeError(
                        f"Worker {shard_name!r} does not share identical initial weights "
                        f"with worker {self.shard_names[0]!r} at parameter {key!r}"
                    )

    def _validate_feature_dims(self, datasets: Sequence[FraudDataset]) -> int:
        """Confirm every worker dataset agrees on feature dimension.

        Args:
            datasets: One ``FraudDataset`` per worker, index-aligned
                with ``self.shard_names``.

        Returns:
            The shared feature dimension.

        Raises:
            ValueError: If any dataset's feature_dim disagrees with the
                first dataset's.
        """
        reference_dim = datasets[0].feature_dim
        for shard_name, ds in zip(self.shard_names[1:], datasets[1:]):
            if ds.feature_dim != reference_dim:
                raise ValueError(
                    f"Feature dimension mismatch: shard {self.shard_names[0]!r} has "
                    f"{reference_dim} features, shard {shard_name!r} has {ds.feature_dim}"
                )
        return reference_dim

    def _validate_synchronized(self) -> None:
        """Confirm every worker received identical averaged weights.

        Raises:
            RuntimeError: If any worker's post-sync parameters differ
                from the global model's.
        """
        if self.global_model is None:
            raise RuntimeError("synchronize() called before workers were built")

        global_state = self.global_model.state_dict()
        for shard_name, model in zip(self.shard_names, self.worker_models):
            state = model.state_dict()
            for key, global_tensor in global_state.items():
                if not torch.equal(global_tensor, state[key]):
                    raise RuntimeError(
                        f"Post-synchronization mismatch: worker {shard_name!r} parameter "
                        f"{key!r} does not match the global model"
                    )

    # ------------------------------------------------------------------
    # TASK 1/2 -- Load shards, carve out a validation split, build workers
    # ------------------------------------------------------------------

    def _prepare_worker_and_val_data(self, shard_paths: Sequence[Path]) -> Path:
        """Remove a stratified validation slice from each shard, write the
        remaining per-shard rows back out as worker-local training CSVs,
        and write the combined validation slice out as one CSV.

        Nothing under data/ is touched -- all intermediate files are
        written to ``artifacts_dir/data/``.

        Args:
            shard_paths: Raw shard CSV paths (index-aligned with
                ``self.shard_names``), as returned by
                ``_validate_shards_exist``.

        Returns:
            Path to the combined validation CSV.

        Raises:
            ValueError: If any shard is empty or missing the label
                column.
        """
        data_dir = self.config.artifacts_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        val_frames = []
        total_rows = 0
        total_fraud = 0

        for shard_name, shard_path in zip(self.shard_names, shard_paths):
            df = pd.read_csv(shard_path)
            if df.empty:
                raise ValueError(f"Shard {shard_path} contains no rows")
            if "isFraud" not in df.columns:
                raise ValueError(f"Shard {shard_path} is missing the 'isFraud' label column")

            try:
                train_df, val_df = train_test_split(
                    df,
                    test_size=self.config.val_split,
                    stratify=df["isFraud"],
                    random_state=self.config.seed,
                )
            except ValueError as exc:
                logger.warning(
                    "Stratified split failed for shard %s (%s); falling back to a "
                    "non-stratified split",
                    shard_name,
                    exc,
                )
                train_df, val_df = train_test_split(
                    df, test_size=self.config.val_split, random_state=self.config.seed
                )

            worker_csv_path = data_dir / f"{shard_name}_train.csv"
            train_df.to_csv(worker_csv_path, index=False)
            val_frames.append(val_df)

            total_rows += len(df)
            total_fraud += int(df["isFraud"].sum())

            self.dataset_stats.setdefault("per_shard", {})[shard_name] = {
                "total_samples": len(df),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "fraud_samples": int(df["isFraud"].sum()),
            }

        val_combined = pd.concat(val_frames, ignore_index=True)
        val_csv_path = data_dir / "val.csv"
        val_combined.to_csv(val_csv_path, index=False)

        self.dataset_stats["total_samples"] = total_rows
        self.dataset_stats["total_fraud"] = total_fraud
        self.dataset_stats["val_samples"] = len(val_combined)
        self.dataset_stats["val_fraud"] = int(val_combined["isFraud"].sum())

        logger.info(
            "Prepared %d worker shard(s) + combined val split: total=%d val=%d",
            len(self.shard_names), total_rows, len(val_combined),
        )
        return val_csv_path

    def _build_workers(self, val_csv_path: Path) -> None:
        """Construct one FraudDataset/DataLoader/FraudMLP/Optimizer/Trainer
        per shard, all workers starting from a cloned copy of the same
        initial model weights (Task 2), plus the global model/evaluation
        Trainer.

        Args:
            val_csv_path: Path to the combined validation CSV, as
                returned by ``_prepare_worker_and_val_data``.

        Raises:
            ValueError: If worker feature dimensions disagree with each
                other or with the validation set's.
        """
        data_dir = self.config.artifacts_dir / "data"
        worker_datasets = [
            FraudDataset(data_dir / f"{name}_train.csv") for name in self.shard_names
        ]
        feature_dim = self._validate_feature_dims(worker_datasets)

        val_dataset = FraudDataset(val_csv_path)
        if val_dataset.feature_dim != feature_dim:
            raise ValueError(
                f"Validation set feature_dim ({val_dataset.feature_dim}) does not match "
                f"worker feature_dim ({feature_dim})"
            )
        self._feature_dim = feature_dim
        self.dataset_stats["feature_dim"] = feature_dim
        logger.info("Feature dimension: %d", feature_dim)

        # TASK 2 -- build exactly one initial model and clone its weights
        # into every worker (and the global model), rather than letting
        # each worker initialize independently.
        set_seed(self.config.seed)
        seed_model = FraudMLP(
            input_dim=feature_dim,
            hidden_dims=list(self.config.hidden_dims),
            dropout_rate=self.config.dropout_rate,
            activation=self.config.activation,
            use_batch_norm=self.config.use_batch_norm,
            use_dropout=self.config.use_dropout,
        )
        initial_state = copy.deepcopy(seed_model.state_dict())

        self.val_loader = create_dataloader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        self.worker_models = []
        self.worker_trainers = []
        self.worker_loaders = []

        for shard_name, dataset in zip(self.shard_names, worker_datasets):
            model = FraudMLP(
                input_dim=feature_dim,
                hidden_dims=list(self.config.hidden_dims),
                dropout_rate=self.config.dropout_rate,
                activation=self.config.activation,
                use_batch_norm=self.config.use_batch_norm,
                use_dropout=self.config.use_dropout,
            )
            model.load_state_dict(copy.deepcopy(initial_state))

            optimizer = create_optimizer(
                model.parameters(),
                optimizer_name=self.config.optimizer_name,
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum,
            )
            loss_fn = create_loss_fn(self.config.loss_type)
            scheduler = _create_scheduler(optimizer, self.config.scheduler_name, self.config.scheduler_kwargs)

            trainer_config = TrainerConfig(
                artifacts_dir=self.config.artifacts_dir / "workers" / shard_name,
                grad_clip_norm=self.config.grad_clip_norm,
                use_amp=self.config.use_amp,
                # Early stopping is evaluated at the *global* level
                # (Task 7/10 evaluate the global model each round), not
                # per-worker, so it is intentionally left disabled here.
                early_stopping_patience=None,
                seed=self.config.seed,
            )
            trainer = Trainer(model, optimizer, loss_fn, config=trainer_config, lr_scheduler=scheduler)

            loader = create_dataloader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
            )

            self.worker_models.append(model)
            self.worker_trainers.append(trainer)
            self.worker_loaders.append(loader)

        self._validate_identical_init()

        # Global model: same architecture, same initial weights, evaluated
        # (never locally trained) via its own Trainer wrapper so Task 7
        # can reuse Trainer.evaluate() unmodified.
        self.global_model = FraudMLP(
            input_dim=feature_dim,
            hidden_dims=list(self.config.hidden_dims),
            dropout_rate=self.config.dropout_rate,
            activation=self.config.activation,
            use_batch_norm=self.config.use_batch_norm,
            use_dropout=self.config.use_dropout,
        )
        self.global_model.load_state_dict(copy.deepcopy(initial_state))

        global_optimizer = create_optimizer(
            self.global_model.parameters(),
            optimizer_name=self.config.optimizer_name,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            momentum=self.config.momentum,
        )
        global_trainer_config = TrainerConfig(
            artifacts_dir=self.config.artifacts_dir,
            grad_clip_norm=self.config.grad_clip_norm,
            use_amp=self.config.use_amp,
            early_stopping_patience=self.config.early_stopping_patience,
            early_stopping_mode=self.config.early_stopping_mode,
            seed=self.config.seed,
        )
        self.global_trainer = Trainer(
            self.global_model,
            global_optimizer,
            create_loss_fn(self.config.loss_type),
            config=global_trainer_config,
        )

    # ------------------------------------------------------------------
    # TASK 4/5 -- Synchronization
    # ------------------------------------------------------------------

    def synchronize(self) -> Dict[str, float]:
        """Average every worker's trainable parameters equally, store the
        result in the global model, and broadcast it back to every
        worker (Task 4/5).

        Deterministic: parameters are summed in the fixed
        ``self.shard_names`` order and divided by the worker count, with
        no other source of nondeterminism.

        Returns:
            Dict with ``sync_time_seconds`` for logging (Task 9).

        Raises:
            RuntimeError: If called before workers are built, or if
                post-sync weights are not identical across all workers.
        """
        if not self.worker_models or self.global_model is None:
            raise RuntimeError("synchronize() called before workers were built")

        start = time.time()
        with torch.no_grad():
            worker_states = [m.state_dict() for m in self.worker_models]
            averaged_state = {}
            for key in worker_states[0].keys():
                stacked = torch.stack([state[key].float() for state in worker_states], dim=0)
                averaged = stacked.mean(dim=0).to(worker_states[0][key].dtype)
                averaged_state[key] = averaged

            self.global_model.load_state_dict(averaged_state)

            broadcast_state = copy.deepcopy(self.global_model.state_dict())
            for model in self.worker_models:
                model.load_state_dict(copy.deepcopy(broadcast_state))

        sync_time = time.time() - start
        self._sync_durations.append(sync_time)

        self._validate_synchronized()
        logger.info("Synchronization complete in %.4fs", sync_time)
        return {"sync_time_seconds": sync_time}

    # ------------------------------------------------------------------
    # Phase 7.5 -- Network Monitoring Integration
    # ------------------------------------------------------------------

    def _build_network_layer(self) -> None:
        """Lazily build ``self.communication_manager``/``self.system_monitor``
        from ``self.config`` if they weren't dependency-injected (Task
        2/3 -- "No manual calls should ever be required"). Never fatal:
        a broken/missing network config disables network simulation
        for this run (logged) rather than aborting training, since the
        network layer is diagnostic/simulated accounting on top of
        real training, not a training dependency.
        """
        if self.config.enable_network_simulation and self.communication_manager is None:
            try:
                from network.communication import CommunicationManager
                from network.network_simulator import NetworkSimulator

                config_path = self.config.network_config_path or str(
                    _PROJECT_ROOT / "config" / "network" / "fully_connected.yaml"
                )
                simulator = NetworkSimulator.from_config(config_path)
                self.communication_manager = CommunicationManager(
                    simulator, synchronization_interval=self.config.local_epochs,
                )
                logger.info(
                    "Network simulation enabled: %d worker(s), config=%s",
                    simulator.topology.num_workers, config_path,
                )
            except Exception:
                logger.exception(
                    "Failed to build NetworkSimulator/CommunicationManager from %r; "
                    "continuing WITHOUT network simulation for this run",
                    self.config.network_config_path,
                )
                self.communication_manager = None

        if self.system_monitor is None:
            from monitoring.system_monitor import SystemMonitor

            self.system_monitor = SystemMonitor()
            self._owns_system_monitor = True

    def _record_network_sync(self, round_idx: int) -> Dict[str, Any]:
        """Drive one simulated synchronization round through
        ``self.communication_manager`` for the just-completed
        ``synchronize()`` call (Task 2/3), recording latency,
        bandwidth, transfer time, and bytes/parameters transferred,
        and feeding synchronization duration + communication cost into
        ``self.system_monitor`` (Task 3).

        No-op (returns ``{}``) if network simulation is disabled or
        unavailable for this run.
        """
        if self.communication_manager is None:
            return {}

        param_count = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)
        try:
            net_round = self.communication_manager.synchronize(
                num_parameters=param_count, bytes_per_parameter=4,
            )
        except Exception:
            logger.exception(
                "Network synchronize() failed for round %d; continuing without network "
                "accounting for this round", round_idx,
            )
            return {}

        if self.system_monitor is not None:
            self.system_monitor.record_sync_duration(net_round["total_sync_duration_seconds"])
            self.system_monitor.record_communication_cost(net_round["transmitted_bytes"])

        return {
            "network_sync_duration_seconds": net_round["total_sync_duration_seconds"],
            "network_transmitted_bytes": net_round["transmitted_bytes"],
            "network_transmitted_parameters": net_round["transmitted_parameters"],
        }

    def _export_network_and_monitoring_artifacts(self) -> None:
        """Automatic artifact + plot generation (Task 4/5) at experiment
        completion. Writes to both the project-root canonical locations
        (``artifacts/network/``, ``analysis/network_plots/`` -- exactly
        where the spec names them) and a copy scoped under this run's
        own ``artifacts_dir`` (so a multi-experiment/multi-seed sweep
        driven by ExperimentRunner keeps every run's network artifacts,
        not just the last one to finish).
        """
        root_network_dir = _PROJECT_ROOT / "artifacts" / "network"
        root_plots_dir = _PROJECT_ROOT / "analysis" / "network_plots"
        run_network_dir = self.config.artifacts_dir / "network"
        run_plots_dir = self.config.artifacts_dir / "network_plots"

        if self.communication_manager is not None:
            try:
                self.communication_manager.export(
                    root_network_dir, system_monitor=self.system_monitor, plots_dir=root_plots_dir,
                )
                self.communication_manager.export(
                    run_network_dir, system_monitor=self.system_monitor, plots_dir=run_plots_dir,
                )
                self._network_summary = self.communication_manager.get_metrics_summary()
            except Exception:
                logger.exception(
                    "Network artifact export failed for %s; training artifacts are still saved",
                    self.config.artifacts_dir,
                )
        elif self.system_monitor is not None and self.system_monitor.samples:
            # No network simulation this run, but system monitoring
            # still ran -- export it on its own so CPU/RAM/GPU metrics
            # and plots are never silently dropped (Task 3/4/5).
            try:
                self.system_monitor.export(root_network_dir)
                self.system_monitor.generate_visualizations(root_plots_dir)
                self.system_monitor.export(run_network_dir)
                self.system_monitor.generate_visualizations(run_plots_dir)
            except Exception:
                logger.exception(
                    "System monitor export failed for %s; training artifacts are still saved",
                    self.config.artifacts_dir,
                )

    # ------------------------------------------------------------------
    # TASK 3/9 -- Local training
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Run the full LocalSGD pipeline: build workers, then for each
        communication round run K local epochs per worker (Task 3),
        synchronize (Task 4/5), and evaluate the global model (Task 7).

        Returns:
            Dict with ``round_metrics`` (one entry per communication
            round) and ``train_duration_seconds``.

        Raises:
            FileNotFoundError: If any shard CSV is missing.
            RuntimeError: If training produced no round metrics, or if
                global checkpoints were not written to disk.
        """
        set_seed(self.config.seed)

        shard_paths = self._validate_shards_exist()
        val_csv_path = self._prepare_worker_and_val_data(shard_paths)
        self._build_workers(val_csv_path)

        # Phase 7.5 -- Network Monitoring Integration: build (if not
        # injected) and start monitoring before any training happens,
        # entirely automatically (Task 2/3).
        self._build_network_layer()
        if self.system_monitor is not None and self._owns_system_monitor:
            self.system_monitor.start()
            self.system_monitor.mark_training_start()
            self.system_monitor.collect()  # guarantee >=1 sample even for very short runs

        overall_start = time.time()
        self.round_metrics = []

        for round_idx in range(self.config.communication_rounds):
            round_start = time.time()
            worker_losses: Dict[str, List[float]] = {name: [] for name in self.shard_names}

            for shard_name, trainer, loader in zip(
                self.shard_names, self.worker_trainers, self.worker_loaders
            ):
                for local_epoch in range(self.config.local_epochs):
                    metrics = trainer.train_epoch(loader)
                    worker_losses[shard_name].append(metrics["loss"])
                    logger.info(
                        "round %d/%d | worker=%s | local_epoch %d/%d | loss=%.4f",
                        round_idx + 1, self.config.communication_rounds,
                        shard_name, local_epoch + 1, self.config.local_epochs,
                        metrics["loss"],
                    )

            sync_info = self.synchronize()

            # Phase 7.5, Task 2/3 -- every synchronization round
            # automatically drives NetworkSimulator/CommunicationManager
            # and feeds SystemMonitor; no-op if network simulation is
            # disabled/unavailable.
            net_info = self._record_network_sync(round_idx)

            global_metrics = self.evaluate()

            round_time = time.time() - round_start
            round_record = {
                "round": round_idx,
                "local_epochs": self.config.local_epochs,
                "worker_losses": worker_losses,
                "global_metrics": global_metrics,
                "sync_time_seconds": sync_info["sync_time_seconds"],
                "round_time_seconds": round_time,
                **net_info,
            }
            self.round_metrics.append(round_record)

            logger.info(
                "round %d/%d complete | global_loss=%.4f global_f1=%.4f | round_time=%.2fs",
                round_idx + 1, self.config.communication_rounds,
                global_metrics["loss"], global_metrics["f1"], round_time,
            )

            is_best = (
                self.global_trainer._best_metric is None
                or self.global_trainer._is_better(global_metrics[self.config.early_stopping_metric])
            )
            if is_best:
                self.global_trainer._best_metric = global_metrics[self.config.early_stopping_metric]
                self._save_global_checkpoint(self._best_global_model_path(), round_idx, global_metrics)
            self._save_global_checkpoint(self._last_global_model_path(), round_idx, global_metrics)

            if (
                self.global_trainer.early_stopper is not None
                and self.global_trainer.early_stopper.step(
                    global_metrics[self.config.early_stopping_metric]
                )
            ):
                logger.info(
                    "Early stopping triggered after round %d (no improvement for %d rounds)",
                    round_idx, self.global_trainer.early_stopper.patience,
                )
                break

        self._train_duration_seconds = time.time() - overall_start
        self._write_round_metrics_csv()

        # Phase 7.5 -- stop monitoring automatically when training
        # finishes (Task 3), only if this trainer owns the monitor's
        # lifecycle (an injected monitor is stopped by its owner).
        if self.system_monitor is not None and self._owns_system_monitor:
            self.system_monitor.mark_training_end()
            self.system_monitor.stop()

        if not self.round_metrics:
            raise RuntimeError("Training produced no round metrics; training did not run")
        if not self._best_global_model_path().exists() or not self._last_global_model_path().exists():
            raise RuntimeError(
                f"Expected global checkpoints not found: {self._best_global_model_path()}, "
                f"{self._last_global_model_path()}"
            )

        logger.info("Total training time: %.2fs", self._train_duration_seconds)
        return {
            "round_metrics": self.round_metrics,
            "train_duration_seconds": self._train_duration_seconds,
        }

    # ------------------------------------------------------------------
    # TASK 7 -- Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the global model on the held-out validation split.

        Reuses ``Trainer.evaluate`` (which reuses ``compute_metrics``);
        no metric computation is duplicated here.

        Returns:
            Dict with keys: loss, accuracy, precision, recall, f1,
            roc_auc, pr_auc, learning_rate, epoch_time_seconds.

        Raises:
            RuntimeError: If called before workers/val_loader are built.
        """
        if self.global_trainer is None or self.val_loader is None:
            raise RuntimeError("evaluate() called before train(); call train() first")

        self._eval_metrics = self.global_trainer.evaluate(self.val_loader)
        return self._eval_metrics

    # ------------------------------------------------------------------
    # TASK 8 -- Checkpoints
    # ------------------------------------------------------------------

    def _best_global_model_path(self) -> Path:
        return self.config.artifacts_dir / "models" / "best_global_model.pt"

    def _last_global_model_path(self) -> Path:
        return self.config.artifacts_dir / "models" / "last_global_model.pt"

    def _save_global_checkpoint(self, path: Path, round_idx: int, metrics: Dict[str, float]) -> None:
        """Save the global model's config + weights + round metrics.

        Args:
            path: Destination checkpoint path.
            round_idx: Communication round index being saved.
            metrics: The global model's evaluation metrics for this
                round, stored alongside the weights.

        Raises:
            RuntimeError: If the checkpoint file is not present on disk
                after saving (Task 10).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "round": round_idx,
            "config": self.global_model.config,
            "state_dict": self.global_model.state_dict(),
            "metrics": metrics,
        }
        torch.save(checkpoint, path)
        if not path.exists():
            raise RuntimeError(f"Failed to save global checkpoint to {path}")

    def _write_round_metrics_csv(self) -> None:
        """Flatten per-round global metrics + worker losses to
        ``metrics/training_metrics.csv`` (Task 8).
        """
        metrics_dir = self.config.artifacts_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        csv_path = metrics_dir / "training_metrics.csv"

        fieldnames = (
            ["round", "sync_time_seconds", "round_time_seconds"]
            + [f"worker_{name}_mean_loss" for name in self.shard_names]
            + ["loss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        )

        rows = []
        for record in self.round_metrics:
            row: Dict[str, Any] = {
                "round": record["round"],
                "sync_time_seconds": record["sync_time_seconds"],
                "round_time_seconds": record["round_time_seconds"],
            }
            for name in self.shard_names:
                losses = record["worker_losses"][name]
                row[f"worker_{name}_mean_loss"] = sum(losses) / len(losses) if losses else float("nan")
            for key in ("loss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
                row[key] = record["global_metrics"].get(key)
            rows.append(row)

        pd.DataFrame(rows, columns=fieldnames).to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run the full pipeline: train(), a final evaluate(), save
        configuration (Task 8).

        Returns:
            Summary dict with dataset_stats, final evaluation metrics,
            training duration, per-round metrics, and the resolved
            configuration.
        """
        train_result = self.train()
        eval_metrics = self.evaluate()

        # Phase 7.5, Task 4/5 -- automatic artifact + plot generation at
        # experiment completion (network_metrics.csv, communication_log.csv,
        # network_summary.json, network_config.json, and every Task 5 plot).
        self._export_network_and_monitoring_artifacts()

        eval_path = self.config.artifacts_dir / "evaluation_metrics.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)

        config_dict = asdict(self.config)
        config_dict["artifacts_dir"] = str(self.config.artifacts_dir)
        config_dict["model"] = self.global_model.config
        config_dict["shard_names"] = self.shard_names
        config_dict["train_duration_seconds"] = train_result["train_duration_seconds"]
        config_dict["total_sync_time_seconds"] = sum(self._sync_durations)

        config_path = self.config.artifacts_dir / "local_sgd_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        summary = {
            "dataset_stats": self.dataset_stats,
            "evaluation_metrics": eval_metrics,
            "train_duration_seconds": train_result["train_duration_seconds"],
            "round_metrics": self.round_metrics,
            "config": config_dict,
            # Phase 7.5 -- network/monitoring summaries, populated by
            # _export_network_and_monitoring_artifacts() above; None
            # when network simulation is disabled/unavailable.
            "network_summary": self._network_summary,
            "system_monitor_summary": (
                self.system_monitor.get_summary()
                if self.system_monitor is not None and self.system_monitor.samples
                else None
            ),
        }
        logger.info(
            "LocalSGD complete: rounds=%d workers=%d feature_dim=%d duration=%.2fs "
            "final_loss=%.4f final_f1=%.4f",
            len(self.round_metrics), len(self.shard_names), self.dataset_stats.get("feature_dim", -1),
            train_result["train_duration_seconds"], eval_metrics["loss"], eval_metrics["f1"],
        )
        return summary


def main() -> None:
    """CLI entry point: run LocalSGD end-to-end.

    Phase 7.5, Task 1: standalone invocations (as opposed to being
    driven by experiments.experiment_runner.ExperimentRunner, which
    cleans once per session itself) clean previous generated artifacts
    once here, so `python training/local_sgd.py` alone still always
    starts from a fresh, reproducible state.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from experiments.cleanup import clean_generated_artifacts

    clean_generated_artifacts()
    LocalSGDTrainer().run()


if __name__ == "__main__":
    main()