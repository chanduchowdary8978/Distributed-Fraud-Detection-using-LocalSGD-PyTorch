"""
plots.py

Purpose:
    Generate every plot required by the Phase 6 spec (Task 5) from
    experiment results produced by experiments/experiment_runner.py:
    per-run training curves (Loss vs Epoch, Accuracy vs Epoch,
    Precision vs Recall) and cross-experiment communication-accuracy
    tradeoff plots (F1 / ROC-AUC / Communication Cost / Training Time,
    each vs Synchronization Interval).

The prior stub in this file named three different, narrower functions
(plot_accuracy_vs_communication, plot_fault_tolerance,
plot_drift_timeline) for experiments this phase does not implement
(fault tolerance and drift/anomaly injection are explicitly out of
scope -- see the Phase 6 spec's DO NOT list). Those are superseded here
by the functions Task 5 actually specifies; nothing in the Phase 6 spec
lists analysis/plots.py's prior stub names as a protected public API
(only ExperimentRunner's methods are).

Data source note -- Loss/Accuracy/Precision/Recall curves:
    training/trainer.py logs one row per (epoch, split) for centralized
    runs, and training/local_sgd.py logs one row per communication
    round (global-model evaluation only, no train/val split) for
    LocalSGD runs -- two different CSV schemas (see
    training/centralized_baseline.py and training/local_sgd.py's module
    docstrings; both are reused unmodified). ``_load_curve`` below
    normalizes both into a common (step, loss, accuracy, precision,
    recall, f1, roc_auc, pr_auc) frame using the val-split rows for
    centralized and the per-round global metrics for LocalSGD, so a
    single plotting function can serve both without duplicating
    per-strategy branching in every plot function.

Data source note -- Precision vs Recall:
    Reusing Trainer/LocalSGDTrainer unmodified means only one
    (precision, recall) pair per epoch/round is available -- there is
    no per-threshold sweep recorded (that would require re-running
    inference to get raw probabilities post-hoc, which is new logic
    outside training/trainer.py's existing public API and would
    duplicate evaluation logic Task 3 explicitly forbids duplicating).
    "Precision vs Recall" here is therefore the training-time
    trajectory of the two, in step order, not a threshold-swept PR
    curve.

Public Interface:
    Functions:
        generate_experiment_plots(training_metrics_csv, strategy, output_dir) -> dict[str, Path]
        generate_cross_experiment_plots(results, output_dir) -> dict[str, Path]
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import matplotlib

matplotlib.use("Agg")  # headless: this process never opens a display.
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.metrics import summarize_experiments  # noqa: E402

logger = logging.getLogger(__name__)

_CURVE_COLUMNS = ["step", "loss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


# ----------------------------------------------------------------------
# Per-run training curves (Loss vs Epoch, Accuracy vs Epoch, Precision vs Recall)
# ----------------------------------------------------------------------


def _load_curve(training_metrics_csv: Union[str, Path], strategy: str) -> pd.DataFrame:
    """Normalize a single run's training_metrics.csv into a common frame
    with a ``step`` column (see module docstring for the two source
    schemas).

    Args:
        training_metrics_csv: Path to a run's
            ``metrics/training_metrics.csv``, as written by
            ``training.trainer.Trainer`` (centralized) or
            ``training.local_sgd.LocalSGDTrainer`` (local_sgd).
        strategy: ``"centralized"`` or ``"local_sgd"``.

    Returns:
        DataFrame with columns ``_CURVE_COLUMNS``, one row per
        epoch (centralized, val split) or round (local_sgd).

    Raises:
        FileNotFoundError: If ``training_metrics_csv`` does not exist.
        ValueError: If ``strategy`` is unsupported, or the CSV is
            missing an expected column for that strategy.
    """
    training_metrics_csv = Path(training_metrics_csv)
    if not training_metrics_csv.exists():
        raise FileNotFoundError(f"No training metrics CSV at {training_metrics_csv}")

    df = pd.read_csv(training_metrics_csv)

    if strategy == "centralized":
        if "split" not in df.columns:
            raise ValueError(f"{training_metrics_csv} missing 'split' column for strategy='centralized'")
        df = df[df["split"] == "val"].reset_index(drop=True)
        df = df.rename(columns={"epoch": "step"})
    elif strategy == "local_sgd":
        df = df.rename(columns={"round": "step"})
    else:
        raise ValueError(f"strategy must be 'centralized' or 'local_sgd', got {strategy!r}")

    missing = set(_CURVE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"{training_metrics_csv} is missing expected column(s) {sorted(missing)} "
            f"for strategy={strategy!r}"
        )
    if df.empty:
        raise ValueError(f"{training_metrics_csv} produced zero rows for strategy={strategy!r}")

    return df[_CURVE_COLUMNS].sort_values("step").reset_index(drop=True)


def _save_fig(output_path: Union[str, Path]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    if not output_path.exists():
        raise RuntimeError(f"Failed to save plot to {output_path}")
    return output_path


def plot_loss_vs_epoch(df: pd.DataFrame, strategy: str, output_path: Union[str, Path]) -> Path:
    """Loss vs Epoch (or vs Round for local_sgd -- see module docstring)."""
    xlabel = "Round" if strategy == "local_sgd" else "Epoch"
    plt.figure(figsize=(6, 4))
    plt.plot(df["step"], df["loss"], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title(f"Loss vs {xlabel}")
    plt.grid(True, alpha=0.3)
    return _save_fig(output_path)


def plot_accuracy_vs_epoch(df: pd.DataFrame, strategy: str, output_path: Union[str, Path]) -> Path:
    """Accuracy vs Epoch (or vs Round for local_sgd -- see module docstring)."""
    xlabel = "Round" if strategy == "local_sgd" else "Epoch"
    plt.figure(figsize=(6, 4))
    plt.plot(df["step"], df["accuracy"], marker="o", color="tab:green")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs {xlabel}")
    plt.grid(True, alpha=0.3)
    return _save_fig(output_path)


def plot_precision_recall(df: pd.DataFrame, strategy: str, output_path: Union[str, Path]) -> Path:
    """Precision vs Recall trajectory across training steps (see module
    docstring for why this is a trajectory, not a threshold-swept curve).
    """
    plt.figure(figsize=(6, 4))
    plt.plot(df["recall"], df["precision"], marker="o", color="tab:purple")
    for i, step in enumerate(df["step"]):
        if i == 0 or i == len(df) - 1:
            plt.annotate(str(int(step)), (df["recall"][i], df["precision"][i]))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision vs Recall (training trajectory)")
    plt.grid(True, alpha=0.3)
    return _save_fig(output_path)


def generate_experiment_plots(
    training_metrics_csv: Union[str, Path], strategy: str, output_dir: Union[str, Path]
) -> Dict[str, Path]:
    """Generate the three per-run curve plots for one experiment run.

    Args:
        training_metrics_csv: Path to that run's training_metrics.csv.
        strategy: ``"centralized"`` or ``"local_sgd"``.
        output_dir: Directory the three PNGs are written into.

    Returns:
        Dict mapping plot name to the written file path.
    """
    output_dir = Path(output_dir)
    df = _load_curve(training_metrics_csv, strategy)
    paths = {
        "loss_vs_epoch": plot_loss_vs_epoch(df, strategy, output_dir / "loss_vs_epoch.png"),
        "accuracy_vs_epoch": plot_accuracy_vs_epoch(df, strategy, output_dir / "accuracy_vs_epoch.png"),
        "precision_vs_recall": plot_precision_recall(df, strategy, output_dir / "precision_vs_recall.png"),
    }
    logger.info("Generated %d per-run plot(s) in %s", len(paths), output_dir)
    return paths


# ----------------------------------------------------------------------
# Cross-experiment tradeoff plots (vs Synchronization Interval)
# ----------------------------------------------------------------------


def plot_metric_vs_sync_interval(
    summaries: Sequence[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Union[str, Path],
) -> Path:
    """Scatter/line plot of one metric against synchronization interval
    (K), restricted to local_sgd experiments -- centralized has no
    synchronization interval (K=None by construction) and is excluded
    rather than silently coerced to a placeholder K value.

    Args:
        summaries: Per-experiment summaries, as returned by
            ``analysis.metrics.summarize_experiments``.
        metric_key: Key into each summary dict to plot on the y-axis.
        ylabel: Y-axis label.
        title: Plot title.
        output_path: Destination PNG path.

    Returns:
        The written file path.

    Raises:
        ValueError: If no local_sgd summaries are present.
    """
    points = sorted(
        (s["synchronization_interval"], s[metric_key])
        for s in summaries
        if s["strategy"] == "local_sgd"
    )
    if not points:
        raise ValueError(
            f"No local_sgd experiments found to plot {metric_key!r} vs synchronization interval"
        )

    xs, ys = zip(*points)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o", color="tab:orange")
    plt.xlabel("Synchronization Interval (K, local epochs per round)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return _save_fig(output_path)


def generate_cross_experiment_plots(
    results: Sequence[Dict[str, Any]], output_dir: Union[str, Path]
) -> Dict[str, Path]:
    """Generate the four required cross-experiment tradeoff plots (Task 5):
    F1, ROC-AUC, Communication Cost, and Training Time, each vs
    Synchronization Interval.

    Args:
        results: Flat list of per-seed run records across every
            experiment (as produced by ``ExperimentRunner.run_all``).
        output_dir: Directory the four PNGs are written into
            (``analysis/plots/`` per the Task 5 spec).

    Returns:
        Dict mapping plot name to the written file path.
    """
    output_dir = Path(output_dir)
    summaries = summarize_experiments(results)

    specs = [
        ("f1_vs_sync_interval", "f1", "F1 Score", "F1 vs Synchronization Interval"),
        ("roc_auc_vs_sync_interval", "roc_auc", "ROC-AUC", "ROC-AUC vs Synchronization Interval"),
        (
            "communication_cost_vs_sync_interval",
            "communication_cost_params",
            "Communication Cost (parameter values transmitted)",
            "Communication Cost vs Synchronization Interval",
        ),
        (
            "training_time_vs_sync_interval",
            "training_time_seconds",
            "Training Time (seconds)",
            "Training Time vs Synchronization Interval",
        ),
    ]

    paths: Dict[str, Path] = {}
    for name, metric_key, ylabel, title in specs:
        paths[name] = plot_metric_vs_sync_interval(
            summaries, metric_key, ylabel, title, output_dir / f"{name}.png"
        )

    logger.info("Generated %d cross-experiment plot(s) in %s", len(paths), output_dir)
    return paths
