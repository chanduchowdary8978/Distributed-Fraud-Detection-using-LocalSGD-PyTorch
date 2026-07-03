"""
partition_regions.py

Purpose:
    Clean the raw PaySim dataset and partition it into five simulated,
    non-IID data-center shards (dc_1 .. dc_5) used by later phases for
    per-data-center local training.

Phase 1 Scope:
    PaySim contains no geographic information, so shards are NOT regional
    in a literal sense -- they are five statistically distinct synthetic
    partitions of the same underlying transaction population, standing in
    for five data centers. No modeling, scaling, encoding, or feature
    engineering happens here; only verification, minimal cleaning, and
    partitioning.

Partition Strategy (see partition_dataset() for full detail):
    - Fraud rows (isFraud == 1) are unevenly distributed across the five
      shards using a fixed allocation vector, so each shard has a
      different fraud ratio.
    - Non-fraud rows are distributed per-transaction-type using a fixed
      allocation matrix, so each shard has a different transaction-type
      mix (e.g. one shard skews toward CASH_OUT/TRANSFER, another toward
      PAYMENT/CASH_IN).
    - All splits use a fixed random seed (RANDOM_SEED) for determinism
      and reproducibility across runs.

Public Interface:
    Functions:
        load_dataset() -> pd.DataFrame
        clean_dataset(df: pd.DataFrame) -> pd.DataFrame
        partition_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]
        save_partitions(shards: dict[str, pd.DataFrame]) -> dict[str, Path]
        generate_metadata(shards: dict[str, pd.DataFrame]) -> dict
        main() -> None
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# download_data.py exposes RAW_DATA_PATH / REQUIRED_COLUMNS / verify_dataset.
# Support both "python data/partition_regions.py" (script directory is
# added to sys.path) and "import data.partition_regions" (package) styles.
try:  # pragma: no cover - import shim, not core logic
    from data.download_data import RAW_DATA_PATH, REQUIRED_COLUMNS, verify_dataset
except ImportError:  # pragma: no cover
    from download_data import RAW_DATA_PATH, REQUIRED_COLUMNS, verify_dataset

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
METADATA_PATH = PROCESSED_DIR / "partition_metadata.json"

RANDOM_SEED = 42
SHARD_NAMES = ["dc_1", "dc_2", "dc_3", "dc_4", "dc_5"]

# --- Non-IID partition assumptions -----------------------------------------
#
# FRAUD_ALLOCATION: fraction of all fraudulent transactions routed to each
# shard (index-aligned with SHARD_NAMES). Deliberately unequal so shards
# have different fraud ratios, mimicking real deployments where some
# regions/data centers see disproportionately more fraud.
FRAUD_ALLOCATION = [0.35, 0.25, 0.20, 0.12, 0.08]

# TYPE_ALLOCATION: for each PaySim transaction type, the fraction of that
# type's *non-fraud* rows routed to each shard (index-aligned with
# SHARD_NAMES). Deliberately unequal per type so each shard has a distinct
# transaction-type mix. Any type not listed here falls back to an equal
# split (DEFAULT_TYPE_ALLOCATION).
TYPE_ALLOCATION: Dict[str, list] = {
    "CASH_IN":  [0.35, 0.30, 0.15, 0.10, 0.10],
    "CASH_OUT": [0.10, 0.15, 0.35, 0.25, 0.15],
    "DEBIT":    [0.20, 0.20, 0.20, 0.20, 0.20],
    "PAYMENT":  [0.30, 0.10, 0.10, 0.30, 0.20],
    "TRANSFER": [0.05, 0.35, 0.10, 0.10, 0.40],
}
DEFAULT_TYPE_ALLOCATION = [0.20, 0.20, 0.20, 0.20, 0.20]


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Verify and load the raw PaySim dataset into memory.

    Args:
        path: Location of the raw PaySim CSV file.

    Returns:
        The raw dataset as a DataFrame.

    Raises:
        FileNotFoundError: If the raw dataset file does not exist.
        ValueError: If the dataset fails verification.
    """
    verify_dataset(path)
    logger.info("Loading raw dataset from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded raw dataset with shape %s", df.shape)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal, non-destructive-to-signal cleaning to the dataset.

    Operations (each logged):
        1. Drop exact duplicate rows.
        2. Drop rows with missing values in required columns.
        3. Coerce numeric columns and drop rows that fail coercion or
           contain negative amounts/balances.
        4. Validate target/flag labels (isFraud, isFlaggedFraud in {0, 1})
           and drop rows with invalid labels.

    No normalization, scaling, encoding, or feature engineering is
    performed. The original raw file on disk is never modified; this
    function only transforms an in-memory copy.

    Args:
        df: Raw dataset as loaded by load_dataset().

    Returns:
        A cleaned copy of the dataset.

    Raises:
        ValueError: If cleaning removes the entire dataset.
    """
    initial_rows = len(df)
    logger.info("Starting cleaning on %d rows", initial_rows)
    cleaned = df.copy()

    # 1. Exact duplicate rows.
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    logger.info("Removed %d exact duplicate rows", before - len(cleaned))

    # 2. Missing values in required columns.
    missing_counts = cleaned[REQUIRED_COLUMNS].isna().sum()
    total_missing = int(missing_counts.sum())
    if total_missing > 0:
        logger.info(
            "Found %d missing values across required columns: %s",
            total_missing,
            {c: int(n) for c, n in missing_counts.items() if n > 0},
        )
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=REQUIRED_COLUMNS)
    logger.info("Removed %d rows with missing required values", before - len(cleaned))

    # 3. Numeric column validation (amount + balance columns must be
    #    numeric and non-negative; PaySim balances/amounts are inherently
    #    non-negative currency values).
    numeric_cols = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=numeric_cols)
    logger.info(
        "Removed %d rows with non-numeric values in %s", before - len(cleaned), numeric_cols
    )

    before = len(cleaned)
    negative_mask = (cleaned[numeric_cols] < 0).any(axis=1)
    cleaned = cleaned.loc[~negative_mask]
    logger.info("Removed %d rows with negative amount/balance values", before - len(cleaned))

    # 4. Target/flag label validation.
    for label_col in ["isFraud", "isFlaggedFraud"]:
        cleaned[label_col] = pd.to_numeric(cleaned[label_col], errors="coerce")
    before = len(cleaned)
    valid_label_mask = cleaned["isFraud"].isin([0, 1]) & cleaned["isFlaggedFraud"].isin([0, 1])
    cleaned = cleaned.loc[valid_label_mask]
    logger.info("Removed %d rows with invalid isFraud/isFlaggedFraud labels", before - len(cleaned))

    cleaned["isFraud"] = cleaned["isFraud"].astype(int)
    cleaned["isFlaggedFraud"] = cleaned["isFlaggedFraud"].astype(int)
    cleaned = cleaned.reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("Cleaning removed all rows from the dataset; nothing left to partition.")

    logger.info(
        "Cleaning complete: %d rows -> %d rows (%d removed total)",
        initial_rows,
        len(cleaned),
        initial_rows - len(cleaned),
    )
    return cleaned


def _split_indices(n: int, allocation: list, seed: int) -> list:
    """Deterministically shuffle ``n`` indices and cut them into len(allocation)
    contiguous, proportionally-sized groups.

    Args:
        n: Number of items to split.
        allocation: Fractions (summing to ~1.0) describing group sizes.
        seed: Seed for the shuffle, so the split is reproducible.

    Returns:
        A list of index arrays, one per allocation entry.
    """
    rng = np.random.RandomState(seed)
    permuted = rng.permutation(n)
    boundaries = np.cumsum([0] + [round(frac * n) for frac in allocation])
    boundaries[-1] = n  # guard against rounding drift
    return [permuted[boundaries[i]:boundaries[i + 1]] for i in range(len(allocation))]


def partition_dataset(df: pd.DataFrame, seed: int = RANDOM_SEED) -> Dict[str, pd.DataFrame]:
    """Partition a cleaned dataset into five non-IID shards.

    Strategy:
        - Fraud rows are split across shards using FRAUD_ALLOCATION, so
          fraud ratio differs per shard.
        - Non-fraud rows are split *within each transaction type* using
          TYPE_ALLOCATION, so transaction-type mix differs per shard.
        - Each shard is row-shuffled (seeded) before being returned so
          fraud/normal rows are interleaved rather than block-ordered.

    This is deterministic: the same input dataframe and seed always
    produce the same five shards.

    Args:
        df: Cleaned dataset (output of clean_dataset()).
        seed: Base random seed controlling all shuffles/splits.

    Returns:
        Mapping of shard name (e.g. "dc_1") to its DataFrame partition.
    """
    logger.info("Partitioning dataset of %d rows into %d shards", len(df), len(SHARD_NAMES))

    fraud_df = df.loc[df["isFraud"] == 1].reset_index(drop=True)
    normal_df = df.loc[df["isFraud"] == 0].reset_index(drop=True)

    shard_frames = {name: [] for name in SHARD_NAMES}

    # Distribute fraud rows per FRAUD_ALLOCATION.
    fraud_groups = _split_indices(len(fraud_df), FRAUD_ALLOCATION, seed=seed)
    for shard_name, idx in zip(SHARD_NAMES, fraud_groups):
        shard_frames[shard_name].append(fraud_df.iloc[idx])

    # Distribute non-fraud rows per-type using TYPE_ALLOCATION.
    for txn_type, type_df in normal_df.groupby("type"):
        allocation = TYPE_ALLOCATION.get(txn_type, DEFAULT_TYPE_ALLOCATION)
        # Derive a distinct but deterministic seed per type so different
        # types don't share an identical shuffle pattern. Python's builtin
        # hash() is randomized per-process (PYTHONHASHSEED) for strings, so
        # a stable hash (md5) is used instead to keep this reproducible
        # across runs and machines.
        type_hash = int(hashlib.md5(txn_type.encode("utf-8")).hexdigest(), 16)
        type_seed = seed + (type_hash % 10_000)
        type_groups = _split_indices(len(type_df), allocation, seed=type_seed)
        type_df = type_df.reset_index(drop=True)
        for shard_name, idx in zip(SHARD_NAMES, type_groups):
            shard_frames[shard_name].append(type_df.iloc[idx])

    shards: Dict[str, pd.DataFrame] = {}
    for i, shard_name in enumerate(SHARD_NAMES):
        shard_df = pd.concat(shard_frames[shard_name], ignore_index=True)
        # Final seeded shuffle so rows are interleaved, not grouped by
        # how they were assembled above.
        shard_df = shard_df.sample(frac=1.0, random_state=seed + i).reset_index(drop=True)
        shards[shard_name] = shard_df
        logger.info(
            "Shard %s: %d rows (%d fraud, %.4f%% fraud ratio)",
            shard_name,
            len(shard_df),
            int(shard_df["isFraud"].sum()),
            100.0 * shard_df["isFraud"].mean() if len(shard_df) else 0.0,
        )

    total_out = sum(len(s) for s in shards.values())
    logger.info("Partitioning complete: %d input rows -> %d rows across shards", len(df), total_out)
    return shards


def save_partitions(
    shards: Dict[str, pd.DataFrame], output_dir: Path = PROCESSED_DIR
) -> Dict[str, Path]:
    """Write each shard to its own CSV file under ``output_dir``.

    Args:
        shards: Mapping of shard name to DataFrame, as returned by
            partition_dataset().
        output_dir: Directory to write shard CSV files into. Created if
            it does not already exist.

    Returns:
        Mapping of shard name to the Path it was written to.

    Raises:
        OSError: If a shard file cannot be written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for shard_name, shard_df in shards.items():
        out_path = output_dir / f"{shard_name}.csv"
        try:
            shard_df.to_csv(out_path, index=False)
        except OSError as exc:
            raise OSError(f"Failed to write shard '{shard_name}' to {out_path}: {exc}") from exc
        written[shard_name] = out_path
        logger.info("Wrote shard %s (%d rows) to %s", shard_name, len(shard_df), out_path)
    return written


def generate_metadata(
    shards: Dict[str, pd.DataFrame], output_path: Path = METADATA_PATH
) -> dict:
    """Compute and persist summary metadata describing the partition.

    Args:
        shards: Mapping of shard name to DataFrame, as returned by
            partition_dataset().
        output_path: File path to write the metadata JSON to. Parent
            directory is created if needed.

    Returns:
        The metadata dictionary that was written to disk.
    """
    shard_summaries = {}
    for shard_name, shard_df in shards.items():
        n = len(shard_df)
        fraud_count = int(shard_df["isFraud"].sum())
        normal_count = n - fraud_count
        fraud_pct = (100.0 * fraud_count / n) if n else 0.0

        type_counts = shard_df["type"].value_counts().to_dict()
        type_distribution = {
            str(t): {"count": int(c), "percentage": round(100.0 * c / n, 4) if n else 0.0}
            for t, c in type_counts.items()
        }

        shard_summaries[shard_name] = {
            "sample_count": n,
            "fraud_count": fraud_count,
            "normal_count": normal_count,
            "fraud_percentage": round(fraud_pct, 4),
            "class_distribution": {
                "fraud": fraud_count,
                "normal": normal_count,
            },
            "transaction_type_distribution": type_distribution,
            "amount_stats": {
                "mean": round(float(shard_df["amount"].mean()), 2) if n else 0.0,
                "median": round(float(shard_df["amount"].median()), 2) if n else 0.0,
                "std": round(float(shard_df["amount"].std()), 2) if n and n > 1 else 0.0,
                "min": round(float(shard_df["amount"].min()), 2) if n else 0.0,
                "max": round(float(shard_df["amount"].max()), 2) if n else 0.0,
            },
        }

    total_samples = sum(s["sample_count"] for s in shard_summaries.values())
    total_fraud = sum(s["fraud_count"] for s in shard_summaries.values())

    metadata = {
        "random_seed": RANDOM_SEED,
        "num_shards": len(shards),
        "shard_names": list(shards.keys()),
        "total_samples": total_samples,
        "total_fraud": total_fraud,
        "overall_fraud_percentage": round(
            100.0 * total_fraud / total_samples, 4
        ) if total_samples else 0.0,
        "partition_assumptions": {
            "geographic_data": (
                "PaySim contains no geographic fields; shards are simulated "
                "non-IID data-center partitions, not real regional splits."
            ),
            "fraud_allocation": dict(zip(SHARD_NAMES, FRAUD_ALLOCATION)),
            "type_allocation": TYPE_ALLOCATION,
            "default_type_allocation": DEFAULT_TYPE_ALLOCATION,
            "notes": (
                "Fraud rows are distributed across shards via FRAUD_ALLOCATION, "
                "producing a different fraud ratio per shard. Non-fraud rows are "
                "distributed per transaction type via TYPE_ALLOCATION, producing "
                "a different transaction-type mix per shard. All splits use a "
                "fixed random seed for determinism."
            ),
        },
        "shards": shard_summaries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote partition metadata to %s", output_path)

    return metadata


def main() -> None:
    """CLI entry point: load, clean, partition, save, and summarize."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        raw_df = load_dataset()
        cleaned_df = clean_dataset(raw_df)
        shards = partition_dataset(cleaned_df)
        save_partitions(shards)
        generate_metadata(shards)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Partitioning failed: %s", exc)
        raise SystemExit(1) from exc

    logger.info("SUCCESS: dataset cleaned, partitioned, and saved to %s", PROCESSED_DIR)


if __name__ == "__main__":
    main()
