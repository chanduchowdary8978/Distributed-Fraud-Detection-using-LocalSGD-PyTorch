"""
download_data.py

Purpose:
    Verify that the raw PaySim transaction dataset is present, readable,
    non-empty, and contains all columns required by downstream processing
    (partition_regions.py).

Phase 1 Scope:
    The PaySim dataset is manually staged by the user at a fixed location
    (data/raw/PaySim.csv). This module does NOT fetch data from any remote
    source (no Kaggle API, no HTTP downloads). "download_dataset" instead
    means "confirm the manually-provided dataset is ready to use" -- this
    keeps the public interface stable for a future phase that may want to
    replace manual staging with an automated fetch, without touching
    callers in partition_regions.py or elsewhere.

Public Interface:
    Functions:
        download_dataset() -> Path
        verify_dataset(path: Path) -> None
        main() -> None
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Fixed dataset location. Per project spec this path always exists and is
# manually managed by the user; it is intentionally the one hardcoded path
# allowed in this module.
RAW_DATA_PATH = Path(__file__).resolve().parent / "raw" / "PaySim.csv"

# Columns required by the PaySim schema. Downstream partitioning and future
# training phases depend on these being present.
REQUIRED_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def verify_dataset(path: Path = RAW_DATA_PATH) -> None:
    """Validate that the raw dataset at ``path`` is usable.

    Checks performed:
        1. The path exists and points to a regular file.
        2. The file can be read as CSV (readability check).
        3. All columns in REQUIRED_COLUMNS are present.
        4. The dataset contains at least one data row (not empty).

    Args:
        path: Location of the raw PaySim CSV file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file exists but is not a regular file, cannot
            be parsed as CSV, is missing required columns, or is empty.
    """
    logger.info("Verifying dataset at %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at expected location: {path}. "
            "The PaySim.csv file must be manually placed at this path "
            "before running download_data.py or partition_regions.py."
        )

    if not path.is_file():
        raise ValueError(f"Expected a file at {path}, but found something else.")

    try:
        header_df = pd.read_csv(path, nrows=5)
    except Exception as exc:  # noqa: BLE001 - surface any parse failure clearly
        raise ValueError(f"Dataset at {path} could not be read as CSV: {exc}") from exc

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in header_df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset at {path} is missing required columns: {missing_columns}. "
            f"Required columns are: {REQUIRED_COLUMNS}"
        )

    if header_df.empty:
        raise ValueError(f"Dataset at {path} appears to be empty (no data rows).")

    # header_df only samples a few rows; confirm the file is not truncated
    # to just a header with no data by checking the total row count cheaply.
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            row_count = sum(1 for _ in f) - 1
    except OSError as exc:
        raise ValueError(f"Could not read dataset at {path} to count rows: {exc}") from exc

    if row_count <= 0:
        raise ValueError(f"Dataset at {path} contains a header but no data rows.")

    logger.info(
        "Dataset verification passed: %s (%d data rows, %d required columns present)",
        path,
        row_count,
        len(REQUIRED_COLUMNS),
    )


def download_dataset(path: Path = RAW_DATA_PATH) -> Path:
    """Confirm the manually-staged raw dataset is ready for use.

    No network access or Kaggle API calls are performed. This function
    exists to give downstream code (and future phases) a single, stable
    entry point for "the dataset is ready", regardless of how it was
    obtained.

    Args:
        path: Location of the raw PaySim CSV file.

    Returns:
        The verified path to the raw dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file fails any verification check.
    """
    logger.info("Preparing raw dataset (manual staging expected at %s)", path)
    verify_dataset(path)
    logger.info("Raw dataset is staged and verified at %s", path)
    return path


def main() -> None:
    """CLI entry point: verify the raw dataset and report status."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        path = download_dataset()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Dataset verification failed: %s", exc)
        raise SystemExit(1) from exc

    logger.info("SUCCESS: dataset verified and ready at %s", path)


if __name__ == "__main__":
    main()
