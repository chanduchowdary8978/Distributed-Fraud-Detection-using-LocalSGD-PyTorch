"""
report_generator.py

Purpose:
    Turn the flat list of per-seed run records produced by
    experiments/experiment_runner.py into the two required Phase 6
    report artifacts: analysis/report.json (machine-readable) and
    analysis/report.md (human-readable).

Ranking Methodology (Engineering Heuristic -- stated explicitly, not
manufactured confidence):
    "Experiment Ranking" and "Best Overall Configuration" both rank by
    F1 (descending), not accuracy. PaySim-derived fraud data is heavily
    class-imbalanced (fraud is a small minority of transactions); a
    classifier that always predicts "not fraud" scores high accuracy
    with zero recall, which is not what this project is trying to
    detect. F1 balances precision and recall and is the standard
    single-number choice for imbalanced binary classification -- but it
    is still one metric among several reported here (accuracy,
    precision, recall, ROC-AUC, PR-AUC, training time, communication
    cost), chosen as the ranking key rather than derived from a fitted
    or theoretically-justified multi-objective weighting. Communication
    cost and training time are reported alongside the ranking as
    tradeoffs for the reader to weigh, not folded into an opaque
    combined score -- there is no principled way to trade off "F1
    points" against "parameters transmitted" without a stated business
    cost per unit of each, which this framework does not have.

    Every reported number is a MEAN across an experiment's configured
    seeds (see analysis.metrics.summarize_experiments); no single-seed
    result is reported as if it were the experiment's performance
    (Task 8 -- reproducibility).

Public Interface:
    Functions:
        build_report(results) -> dict
        write_report(results, output_dir) -> tuple[Path, Path]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from analysis.metrics import summarize_experiments

logger = logging.getLogger(__name__)

RANKING_METRIC = "f1"

_BEST_PER_METRIC = {
    "accuracy": "Best Accuracy",
    "precision": "Best Precision",
    "recall": "Best Recall",
    "f1": "Best F1",
    "roc_auc": "Best ROC-AUC",
    "pr_auc": "Best PR-AUC",
}


def build_report(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the report data structure (shared by both report.json and
    report.md -- report.md is a rendering of exactly this dict, so the
    two outputs can never disagree with each other).

    Args:
        results: Flat list of per-seed run records across every
            experiment, as produced by ``ExperimentRunner.run_all``.

    Returns:
        Dict with keys: generated_at, num_experiments, ranking_metric,
        experiment_ranking, Best Accuracy/Precision/Recall/F1/ROC-AUC/
        PR-AUC, Fastest Training, Lowest Communication Cost,
        Best Overall Configuration, configuration_summary.

    Raises:
        ValueError: If ``results`` is empty (propagated from
            ``summarize_experiments``).
    """
    summaries = summarize_experiments(results)  # raises ValueError if results is empty

    ranking = sorted(summaries, key=lambda s: s[RANKING_METRIC], reverse=True)

    best_per_metric = {}
    for metric, title in _BEST_PER_METRIC.items():
        best = max(summaries, key=lambda s: s[metric])
        best_per_metric[title] = {"experiment": best["name"], "value": best[metric]}

    fastest = min(summaries, key=lambda s: s["training_time_seconds"])
    lowest_comm = min(summaries, key=lambda s: s["communication_cost_params"])
    best_overall = ranking[0]

    # Phase 7.5, Task 7 -- network/system statistics, automatically
    # available because experiments/experiment_runner.py now attaches
    # these fields to every per-seed record and analysis.metrics's
    # generic aggregate_seeds()/summarize_experiments() mean-reduces
    # any numeric field with no changes needed here. 0.0 for
    # centralized experiments and any local_sgd experiment where
    # network simulation was disabled/unavailable (see
    # ExperimentRunner.run_experiment).
    _NETWORK_STAT_KEYS = (
        "network_avg_latency_seconds",
        "network_max_latency_seconds",
        "network_min_latency_seconds",
        "network_synchronization_cost_seconds",
        "network_total_bytes_transferred",
        "network_avg_bandwidth_mbps",
        "experiment_duration_seconds",
        "cpu_percent_mean",
        "ram_percent_mean",
        "gpu_percent_mean",
    )
    network_statistics = [
        {"experiment": s["name"], "strategy": s["strategy"], **{k: s.get(k, 0.0) for k in _NETWORK_STAT_KEYS}}
        for s in summaries
    ]

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_experiments": len(summaries),
        "ranking_metric": RANKING_METRIC,
        "experiment_ranking": [
            {
                "rank": i + 1,
                "experiment": s["name"],
                "strategy": s["strategy"],
                RANKING_METRIC: s[RANKING_METRIC],
            }
            for i, s in enumerate(ranking)
        ],
        **best_per_metric,
        "Fastest Training": {
            "experiment": fastest["name"],
            "training_time_seconds": fastest["training_time_seconds"],
        },
        "Lowest Communication Cost": {
            "experiment": lowest_comm["name"],
            "communication_cost_params": lowest_comm["communication_cost_params"],
        },
        "Best Overall Configuration": {
            "experiment": best_overall["name"],
            "strategy": best_overall["strategy"],
            "f1": best_overall["f1"],
            "accuracy": best_overall["accuracy"],
            "roc_auc": best_overall["roc_auc"],
            "training_time_seconds": best_overall["training_time_seconds"],
            "communication_cost_params": best_overall["communication_cost_params"],
        },
        "configuration_summary": [
            {
                "experiment": s["name"],
                "strategy": s["strategy"],
                "synchronization_interval": s["synchronization_interval"],
                "communication_rounds": s["communication_rounds"],
                "num_seeds": s["num_seeds"],
                "param_count": s["param_count"],
            }
            for s in summaries
        ],
        "network_statistics": network_statistics,
    }
    return report


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Phase 6 Experiment Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Experiments: {report['num_experiments']}",
        f"Ranking metric: {report['ranking_metric']} (see report_generator.py's module "
        f"docstring for why F1, not accuracy, is the ranking key)",
        "",
        "## Experiment Ranking (by F1, descending, mean across seeds)",
        "",
        "| Rank | Experiment | Strategy | F1 |",
        "|---|---|---|---|",
    ]
    for row in report["experiment_ranking"]:
        lines.append(f"| {row['rank']} | {row['experiment']} | {row['strategy']} | {row['f1']:.4f} |")

    lines += ["", "## Best Per Metric"]
    for title in _BEST_PER_METRIC.values():
        entry = report[title]
        lines.append(f"- **{title}**: {entry['experiment']} ({entry['value']:.4f})")

    ft = report["Fastest Training"]
    lc = report["Lowest Communication Cost"]
    bo = report["Best Overall Configuration"]
    lines += [
        f"- **Fastest Training**: {ft['experiment']} ({ft['training_time_seconds']:.2f}s)",
        f"- **Lowest Communication Cost**: {lc['experiment']} "
        f"({lc['communication_cost_params']:,.0f} parameter values transmitted -- "
        f"centralized is 0 by construction; see communication_rounds=0)",
        "",
        "## Best Overall Configuration",
        "",
        f"**{bo['experiment']}** (strategy={bo['strategy']}), ranked by F1 -- see Ranking "
        f"Methodology in this module's docstring for what this ranking does and does not claim.",
        "",
        f"- F1: {bo['f1']:.4f}",
        f"- Accuracy: {bo['accuracy']:.4f}",
        f"- ROC-AUC: {bo['roc_auc']:.4f}",
        f"- Training time: {bo['training_time_seconds']:.2f}s",
        f"- Communication cost: {bo['communication_cost_params']:,.0f} parameter values",
        "",
        "## Configuration Summary",
        "",
        "| Experiment | Strategy | Sync Interval (K) | Comm Rounds | Seeds | Trainable Params |",
        "|---|---|---|---|---|---|",
    ]
    for c in report["configuration_summary"]:
        k = c["synchronization_interval"] if c["synchronization_interval"] is not None else "N/A"
        lines.append(
            f"| {c['experiment']} | {c['strategy']} | {k} | {c['communication_rounds']} | "
            f"{c['num_seeds']} | {c['param_count']:,} |"
        )

    # Phase 7.5, Task 7 -- Network & System Statistics.
    lines += [
        "",
        "## Network & System Statistics",
        "",
        "0.0 for centralized experiments and any local_sgd experiment where network "
        "simulation was disabled/unavailable (see experiments/experiment_runner.py).",
        "",
        "| Experiment | Strategy | Avg Latency (s) | Max Latency (s) | Min Latency (s) | "
        "Sync Cost (s) | Bytes Transferred | Avg Bandwidth (MB/s) | Exp Duration (s) | "
        "CPU % | RAM % | GPU % |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n in report["network_statistics"]:
        lines.append(
            f"| {n['experiment']} | {n['strategy']} | {n['network_avg_latency_seconds']:.4f} | "
            f"{n['network_max_latency_seconds']:.4f} | {n['network_min_latency_seconds']:.4f} | "
            f"{n['network_synchronization_cost_seconds']:.4f} | "
            f"{n['network_total_bytes_transferred']:,.0f} | "
            f"{n['network_avg_bandwidth_mbps']:.2f} | {n['experiment_duration_seconds']:.2f} | "
            f"{n['cpu_percent_mean']:.1f} | {n['ram_percent_mean']:.1f} | {n['gpu_percent_mean']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(results: Sequence[Dict[str, Any]], output_dir: Any) -> Tuple[Path, Path]:
    """Build the report and write both report.json and report.md.

    Args:
        results: Flat list of per-seed run records, as produced by
            ``ExperimentRunner.run_all``.
        output_dir: Directory to write into (``analysis/`` per the
            Task 6 spec).

    Returns:
        ``(json_path, md_path)``.

    Raises:
        ValueError: If ``results`` is empty.
        RuntimeError: If either output file is missing after writing
            (Task 10 -- validation).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(results)

    json_path = output_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = output_dir / "report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(report))

    missing = [p for p in (json_path, md_path) if not p.exists()]
    if missing:
        raise RuntimeError(f"Report generation did not produce expected file(s): {missing}")

    logger.info("Report written: %s, %s", json_path, md_path)
    return json_path, md_path
