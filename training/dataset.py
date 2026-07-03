"""
dataset.py

Purpose:
    PyTorch Dataset for the cleaned, partitioned PaySim shards produced by
    data/partition_regions.py (data/processed/dc_*.csv), plus a reusable
    DataLoader factory.

Phase 3 Scope:
    data/partition_regions.py performs verification, cleaning, and
    non-IID partitioning only -- it does NOT encode categorical columns
    or drop identifier columns. Those shard CSVs are therefore still raw
    at the column level (mixed numeric / string / identifier columns).
    This module is the first point in the pipeline that turns a shard
    into model-ready float tensors, and it does so entirely in memory
    (the CSV files on disk are never modified):

      - Identifier columns (nameOrig, nameDest) are dropped: they are
        near-unique strings with no generalizable signal and are not
        used by models/fraud_mlp.py.
      - The categorical `type` column is one-hot encoded against a
        FIXED, hardcoded vocabulary (PaySim has exactly 5 transaction
        types, all visible in data/partition_regions.py's
        TYPE_ALLOCATION). A fixed vocabulary -- rather than fitting
        `pd.get_dummies` per shard -- is required for correctness: any
        one-hot encoding fit independently per shard could disagree on
        column count/order across dc_1..dc_5, silently breaking
        `input_dim` consistency for federated aggregation in a later
        phase. This is the one non-obvious design decision in this
        file; everything else is direct column selection.
      - All remaining numeric columns become features as-is.

    `isFlaggedFraud` is kept as a feature (it is a genuine PaySim column,
    already numeric). It is PaySim's own rule-based flag for large
    transfers and is correlated with the label; whether to use it as a
    feature vs. drop it as a leakage risk is a modeling decision for a
    later phase, not a Phase 3 framework decision, so it is not silently
    dropped here.

Public Interface:
    Classes:
        FraudDataset(torch.utils.data.Dataset)

    Functions:
        create_dataloader(...) -> torch.utils.data.DataLoader
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Columns that identify a transaction/party rather than describe it --
# always excluded from features.
ID_COLUMNS: List[str] = ["nameOrig", "nameDest"]

# The categorical column and its fixed vocabulary. Fixed (not fit per
# shard) so every shard produces the same feature columns in the same
# order -- see module docstring.
TYPE_COLUMN = "type"
TRANSACTION_TYPES: List[str] = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

DEFAULT_LABEL_COLUMN = "isFraud"


class FraudDataset(Dataset):
    """PyTorch Dataset over a single processed PaySim shard CSV.

    Args:
        csv_path: Path to a shard CSV (e.g. data/processed/dc_1.csv).
        label_column: Name of the binary target column. Defaults to
            ``"isFraud"``.
        exclude_columns: Additional column names to exclude from
            features, beyond ``ID_COLUMNS`` and ``label_column``.
            Defaults to none.

    Attributes:
        feature_names: Ordered list of the engineered feature column
            names (after one-hot expansion of `type`).
        feature_dim: Number of features, i.e. ``len(feature_names)``.
            Callers use this to construct ``FraudMLP(input_dim=...)``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If the CSV is empty, missing ``label_column`` or the
            `type` column, contains unseen transaction types, contains
            non-binary labels, or contains NaN/inf after conversion to
            float.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        label_column: str = DEFAULT_LABEL_COLUMN,
        exclude_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.label_column = label_column
        self._exclude_columns = list(exclude_columns) if exclude_columns else []

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Shard CSV not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        if df.empty:
            raise ValueError(f"Shard CSV at {self.csv_path} contains no rows")

        if self.label_column not in df.columns:
            raise ValueError(
                f"label_column {self.label_column!r} not found in {self.csv_path}; "
                f"available columns: {list(df.columns)}"
            )
        if TYPE_COLUMN not in df.columns:
            raise ValueError(
                f"Expected categorical column {TYPE_COLUMN!r} not found in "
                f"{self.csv_path}; available columns: {list(df.columns)}"
            )

        labels = pd.to_numeric(df[self.label_column], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(
                f"label_column {self.label_column!r} in {self.csv_path} must be "
                f"binary (0/1) with no missing values"
            )

        unseen_types = set(df[TYPE_COLUMN].unique()) - set(TRANSACTION_TYPES)
        if unseen_types:
            raise ValueError(
                f"{self.csv_path} contains transaction type(s) {sorted(unseen_types)} "
                f"not in the fixed vocabulary TRANSACTION_TYPES={TRANSACTION_TYPES}"
            )

        drop_cols = set(ID_COLUMNS) | {self.label_column, TYPE_COLUMN} | set(self._exclude_columns)
        numeric_cols = [c for c in df.columns if c not in drop_cols]

        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if numeric_df.isna().any().any():
            bad_cols = numeric_df.columns[numeric_df.isna().any()].tolist()
            raise ValueError(
                f"Non-numeric or missing values found in feature column(s) "
                f"{bad_cols} of {self.csv_path}"
            )

        type_dummies = pd.get_dummies(
            pd.Categorical(df[TYPE_COLUMN], categories=TRANSACTION_TYPES)
        ).reset_index(drop=True)
        type_dummies.columns = [f"{TYPE_COLUMN}_{c}" for c in type_dummies.columns]

        features_df = pd.concat([numeric_df.reset_index(drop=True), type_dummies], axis=1)

        feature_array = features_df.to_numpy(dtype=np.float32)
        if not np.isfinite(feature_array).all():
            raise ValueError(f"Non-finite feature values found in {self.csv_path}")

        self.feature_names: List[str] = list(features_df.columns)
        self.feature_dim: int = len(self.feature_names)

        self.features: torch.Tensor = torch.from_numpy(feature_array)
        self.labels: torch.Tensor = torch.tensor(
            labels.to_numpy(dtype=np.float32), dtype=torch.float32
        ).unsqueeze(1)  # shape (N, 1) to match FraudMLP's output_dim=1

        logger.info(
            "Loaded %s: %d rows, %d features", self.csv_path, len(self), self.feature_dim
        )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a ``DataLoader`` over a ``FraudDataset`` (or any Dataset).

    Args:
        dataset: A ``torch.utils.data.Dataset`` instance.
        batch_size: Samples per batch. Must be positive.
        shuffle: Whether to shuffle at every epoch (typically ``True``
            for training, ``False`` for validation/test).
        num_workers: Subprocesses for data loading.
        pin_memory: Whether to pin memory (only useful when training on
            CUDA).
        drop_last: Whether to drop the last incomplete batch.

    Returns:
        A configured ``DataLoader``.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size!r}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )