"""
metrics.py

Purpose:
    Pure, I/O-free computation used by experiments/experiment_runner.py
    and analysis/report_generator.py / analysis/plots.py: a
    communication-cost model, and reduction of repeated-seed runs into
    per-experiment summary statistics.

Scope:
    No file I/O, no plotting, no training/orchestration logic -- those
    live in experiment_runner.py, plots.py, and report_generator.py
    respectively. Keeping this module pure makes it independently
    testable (see the __main__ self-check at the bottom) and reusable
    from every caller without duplicating the same reduction logic.

Communication Cost Model (Engineering Heuristic, not a measurement):
    Per the Phase 6 spec's DO NOT list, this phase implements no
    networking/RPC/latency/bandwidth simulation -- network/ is out of
    scope. training/local_sgd.py's synchronize() (unmodified, reused
    as-is) is a single-process simulation: no bytes actually cross a
    network. What IS well-defined from that code is the *protocol*:
    once per communication round, every worker's full parameter vector
    is conceptually sent to a coordinator, which averages them and
    sends the result back to every worker. This module counts the
    scalar parameter values implied by that protocol:

        communication_cost(rounds, num_workers, param_count)
            = rounds * num_workers * param_count      (upload)
            + rounds * num_workers * param_count      (broadcast)
            = 2 * rounds * num_workers * param_count

    This is a parameter-count proxy for communication volume: it
    ignores per-message protocol/header overhead, gradient/weight
    compression, and quantization -- all of which a real deployment
    would use to reduce actual bytes. Multiplying by BYTES_PER_PARAM
    (4, for float32 -- FraudMLP's native dtype) gives an approximate
    byte count under a "send raw float32 weights" assumption, which is
    the worst-case (no compression) baseline. Centralized training
    performs no cross-process communication, so its cost is always 0
    by construction (experiments/experiment_runner.py always passes
    communication_rounds=0 for the centralized strategy).

Public Interface:
    Functions:
        count_model_parameters(model) -> int
        communication_cost(...) -> int
        communication_bytes(...) -> int
        aggregate_seeds(records) -> dict
        summarize_experiments(results) -> list[dict]
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

BYTES_PER_PARAM = 4  # float32, matching FraudMLP's native parameter dtype.


# ----------------------------------------------------------------------
# Communication cost model
# ----------------------------------------------------------------------


def count_model_parameters(model: Any) -> int:
    """Count trainable parameters in a torch.nn.Module.

    Args:
        model: Any object exposing ``.parameters()`` (e.g. FraudMLP).

    Returns:
        Sum of ``numel()`` over all parameters with ``requires_grad``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def communication_cost(communication_rounds: int, num_workers: int, param_count: int) -> int:
    """Scalar parameter values transmitted across all rounds.

    See the module docstring for the protocol this models and its
    stated assumptions/limitations.

    Args:
        communication_rounds: Number of synchronization rounds (0 for
            centralized training, which performs no synchronization).
        num_workers: Number of participating workers.
        param_count: Trainable parameter count of the model being
            synchronized.

    Returns:
        ``2 * communication_rounds * num_workers * param_count``.

    Raises:
        ValueError: If any argument is negative.
    """
    if communication_rounds < 0 or num_workers < 0 or param_count < 0:
        raise ValueError(
            "communication_rounds, num_workers, param_count must all be >= 0, got "
            f"({communication_rounds}, {num_workers}, {param_count})"
        )
    return 2 * communication_rounds * num_workers * param_count


def communication_bytes(
    communication_rounds: int,
    num_workers: int,
    param_count: int,
    bytes_per_param: int = BYTES_PER_PARAM,
) -> int:
    """``communication_cost(...)`` converted to approximate bytes
    (float32, uncompressed, by default -- see module docstring).
    """
    return communication_cost(communication_rounds, num_workers, param_count) * bytes_per_param


# ----------------------------------------------------------------------
# Reproducibility: reducing repeated-seed runs (Task 8)
# ----------------------------------------------------------------------


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def aggregate_seeds(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute mean/std across repeated-seed runs of the same experiment.

    Args:
        records: One metrics dict per seed (e.g. the per-run records
            produced by ``ExperimentRunner.run_experiment``). Only
            fields that are numeric (int/float, not bool) in EVERY
            record are aggregated; non-numeric fields (e.g. ``name``,
            ``strategy``) are silently skipped -- callers needing those
            should read them directly from ``records[0]``.

    Returns:
        Dict mapping each numeric field name to
        ``{"mean": ..., "std": ..., "n": <non-NaN sample count>}``.
        NaN values (e.g. ROC-AUC on a degenerate split -- see
        ``training/trainer.py::compute_metrics``) are excluded from a
        field's mean/std rather than propagating NaN through the whole
        aggregate; a field with zero non-NaN samples reports
        ``mean``/``std`` as NaN with ``n=0``. ``std`` of a single
        sample is reported as ``0.0`` (a single observation has no
        estimate of spread) rather than NaN.

    Raises:
        ValueError: If ``records`` is empty.
    """
    if not records:
        raise ValueError("aggregate_seeds() requires at least one record")

    numeric_fields = {k for k, v in records[0].items() if _is_numeric(v)}
    for record in records[1:]:
        numeric_fields &= {k for k, v in record.items() if _is_numeric(v)}

    result: Dict[str, Dict[str, float]] = {}
    for field in sorted(numeric_fields):
        values = [r[field] for r in records if not _is_nan(r[field])]
        n = len(values)
        if n == 0:
            result[field] = {"mean": float("nan"), "std": float("nan"), "n": 0}
            continue
        mean = sum(values) / n
        if n == 1:
            std = 0.0
        else:
            variance = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(variance)
        result[field] = {"mean": mean, "std": std, "n": n}
    return result


def summarize_experiments(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group per-seed run records by experiment name and reduce each
    group to one representative record.

    Constant fields (fields that must not vary across seeds of the same
    experiment -- name, strategy, synchronization_interval,
    communication_rounds, param_count) are carried through unchanged
    after verifying they actually agree across seeds. Every other
    numeric field is reduced via ``aggregate_seeds``'s mean, with a
    ``"<field>_std"`` companion key.

    Args:
        results: Flat list of per-seed run records across all
            experiments (as written to results.json by
            ``ExperimentRunner.run_all``).

    Returns:
        One summary dict per distinct experiment ``name``.

    Raises:
        ValueError: If ``results`` is empty, or a constant field
            disagrees across seeds of the same experiment (a config or
            bookkeeping bug, not a legitimate seed-to-seed difference).
    """
    if not results:
        raise ValueError("summarize_experiments() requires at least one record")

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        groups.setdefault(r["name"], []).append(r)

    constant_fields = (
        "name",
        "strategy",
        "synchronization_interval",
        "communication_rounds",
        "param_count",
    )

    summaries: List[Dict[str, Any]] = []
    for name, group in groups.items():
        for field in constant_fields:
            values = {g[field] for g in group}
            if len(values) != 1:
                raise ValueError(
                    f"Experiment {name!r} has inconsistent {field!r} across its seeds: "
                    f"{values}. This indicates a config-loading bug, not a legitimate "
                    f"seed-dependent difference."
                )

        agg = aggregate_seeds(group)
        summary: Dict[str, Any] = {field: group[0][field] for field in constant_fields}
        summary["num_seeds"] = len(group)
        for metric, stats in agg.items():
            if metric in constant_fields:
                continue
            summary[metric] = stats["mean"]
            summary[f"{metric}_std"] = stats["std"]
        summaries.append(summary)

    return summaries


if __name__ == "__main__":
    # Minimal reproducibility self-check (Task/Verification requirement):
    # fails loudly if the aggregation logic is broken, without requiring
    # a full experiment run or a test framework.
    logging.basicConfig(level=logging.INFO)

    assert communication_cost(0, 5, 1000) == 0, "centralized cost must be 0"
    assert communication_cost(10, 5, 1000) == 100_000, "2 * 10 * 5 * 1000"

    seed_records = [
        {"name": "x", "strategy": "centralized", "synchronization_interval": None,
         "communication_rounds": 0, "param_count": 10, "f1": 0.8, "roc_auc": float("nan")},
        {"name": "x", "strategy": "centralized", "synchronization_interval": None,
         "communication_rounds": 0, "param_count": 10, "f1": 0.6, "roc_auc": 0.9},
    ]
    agg = aggregate_seeds(seed_records)
    assert math.isclose(agg["f1"]["mean"], 0.7), agg
    assert agg["roc_auc"]["n"] == 1, "the NaN roc_auc must be excluded from the mean"

    summaries = summarize_experiments(seed_records)
    assert len(summaries) == 1 and math.isclose(summaries[0]["f1"], 0.7), summaries

    print("analysis/metrics.py self-check passed.")
