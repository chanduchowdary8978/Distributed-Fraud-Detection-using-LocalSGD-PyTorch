"""
experiment_runner.py

Purpose:
    The permanent experiment/benchmarking framework for this project.
    Loads experiment configurations from config/experiments/*.yaml,
    runs each configured strategy (reusing CentralizedTrainer and
    LocalSGDTrainer unmodified -- no training logic is duplicated
    here), collects a common metric record per run, and drives
    analysis/plots.py and analysis/report_generator.py to produce the
    project's plots and reports.

    Future work adds new experiment configurations under
    config/experiments/; this file does not need to change to support
    them (Task 1/3 -- no hardcoded experiment settings, no per-strategy
    branching beyond the two strategies training/ currently
    implements).

Phase 6 Scope:
    Per the spec's DO NOT list: no FedAvg, networking, RPC, latency/
    bandwidth simulation, fault injection, FastAPI, or Docker. Those
    live (or will live) in api/, network/, monitoring/, and
    training/fedavg_baseline.py, none of which this file imports or
    duplicates.

Design decisions worth flagging explicitly (not silently assumed):

    1. Config schema (Task 1): each YAML file has three top-level keys
       -- `name`, `strategy` (`"centralized"` or `"local_sgd"`), and
       `params` (forwarded, verbatim and unmodified, to
       `CentralizedConfig(**params)` or `LocalSGDConfig(**params)`).
       This is a direct pass-through rather than a translation layer:
       every field either dataclass already validates (see their
       `__post_init__`) is validated exactly once, in exactly one
       place, with no risk of this file's schema silently drifting
       from the dataclasses it configures.

    2. Artifact layout (Task 7): each run's `CentralizedTrainer`/
       `LocalSGDTrainer` is given `artifacts_dir =
       artifacts/experiments/<name>/seed_<seed>/`. Those classes
       already write `evaluation_metrics.json`,
       `metrics/training_metrics.csv`, `models/`, and their own
       `{centralized,local_sgd}_config.json` there (Task 3 forbids
       duplicating that logic) -- this file adds exactly one more file
       per run, `experiment_metadata.json` (the record described in
       Task 4), and per-run plots under `plots/`. `results.csv` /
       `results.json` (Task 4) and cross-experiment plots / reports
       (Tasks 5/6) are collection-wide, not per-run, so they are
       written at `artifacts/experiments/` and `analysis/`
       respectively -- the spec does not pin an exact path for
       results.csv/json beyond "Task 4", and colocating them with the
       per-experiment directories they summarize is the least
       surprising choice.

    3. Communication accounting (Task 4): `communication_rounds=0` and
       `synchronization_interval=None` for every centralized run by
       construction -- centralized training is single-process and
       performs no parameter synchronization, so this is not an
       omission but the correct value. See analysis/metrics.py's
       module docstring for the communication-cost formula itself.

Public API (Task -- "PUBLIC API"; names below are never renamed):
    ExperimentRunner
        run_experiment(name, seed=None) -> dict
        run_all(seeds=None) -> list[dict]
        repeat_experiment(name, seeds=None) -> dict
        generate_report(results=None) -> tuple[Path, Path]
        generate_plots(results=None) -> dict
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Running this file directly (`python experiments/experiment_runner.py`,
# per the Task acceptance criteria) puts only this file's own directory
# on sys.path, not the repository root -- so the absolute imports below
# would otherwise fail. Mirrors the identical guard already used in
# training/centralized_baseline.py and training/local_sgd.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml  # noqa: E402

from analysis.metrics import (  # noqa: E402
    BYTES_PER_PARAM,
    communication_cost,
    count_model_parameters,
)
from analysis.plots import generate_cross_experiment_plots, generate_experiment_plots  # noqa: E402
from analysis.report_generator import write_report  # noqa: E402
from experiments.cleanup import clean_generated_artifacts  # noqa: E402
from monitoring.system_monitor import SystemMonitor  # noqa: E402
from training.centralized_baseline import CentralizedConfig, CentralizedTrainer  # noqa: E402
from training.local_sgd import LocalSGDConfig, LocalSGDTrainer  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "config"
_DEFAULT_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts" / "experiments"
_DEFAULT_ANALYSIS_DIR = _PROJECT_ROOT / "analysis"
_DEFAULT_NETWORK_PLOTS_DIR = _PROJECT_ROOT / "analysis" / "network_plots"
_DEFAULT_NETWORK_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts" / "network"

_STRATEGIES = {"centralized", "local_sgd"}

_RESULT_FIELDS = [
    "name",
    "strategy",
    "seed",
    "loss",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "training_time_seconds",
    "communication_rounds",
    "synchronization_interval",
    "communication_cost_params",
    "communication_cost_bytes",
    "param_count",
    "run_duration_seconds",
    # Phase 7.5 -- Network Monitoring Integration (Task 7): 0.0 for
    # centralized runs and for local_sgd runs where network simulation
    # was disabled/unavailable, by construction (see run_experiment).
    "network_avg_latency_seconds",
    "network_max_latency_seconds",
    "network_min_latency_seconds",
    "network_avg_bandwidth_mbps",
    "network_total_bytes_transferred",
    "network_synchronization_cost_seconds",
    "cpu_percent_mean",
    "ram_percent_mean",
    "gpu_percent_mean",
    "experiment_duration_seconds",
]


class ExperimentRunner:
    """Loads config/experiments/*.yaml, runs CentralizedTrainer/
    LocalSGDTrainer, and produces results/plots/reports.

    Args:
        config_dir: Directory containing experiment YAML files.
            Defaults to ``config/experiments/``.
        artifacts_dir: Root directory for per-run and per-experiment
            artifacts. Defaults to ``artifacts/experiments/``.
        analysis_dir: Root directory for plots.py/report_generator.py
            output. Defaults to ``analysis/`` (so plots land in
            ``analysis/plots/`` and reports in ``analysis/`` per the
            Task 5/6 spec).
        clean_on_init: Phase 7.5, Task 1 -- automatically clean every
            previously-generated artifact (artifacts/,
            analysis/network_plots/, analysis/plots/,
            analysis/reports/, logs/) once when this runner is
            constructed, so a session started via ``ExperimentRunner()``
            always begins from a completely fresh, reproducible state
            with no manual cleanup commands required. Set to ``False``
            to opt out (e.g. constructing multiple ``ExperimentRunner``
            instances against the same artifacts in one process).
    """

    def __init__(
        self,
        config_dir: Path = _DEFAULT_CONFIG_DIR,
        artifacts_dir: Path = _DEFAULT_ARTIFACTS_DIR,
        analysis_dir: Path = _DEFAULT_ANALYSIS_DIR,
        clean_on_init: bool = True,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.analysis_dir = Path(analysis_dir)
        self.plots_dir = self.analysis_dir / "plots"
        self._results: List[Dict[str, Any]] = []

        if clean_on_init:
            clean_generated_artifacts(_PROJECT_ROOT)

    # ------------------------------------------------------------------
    # TASK 1 -- Configuration loading
    # ------------------------------------------------------------------

    def list_experiment_names(self) -> List[str]:
        """List every experiment name discoverable under config_dir
        (Task 3 -- adding future experiments requires no runner change).

        Returns:
            Sorted list of experiment names (YAML stem names).

        Raises:
            FileNotFoundError: If config_dir does not exist.
        """
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Experiment config directory not found: {self.config_dir}")
        return sorted(p.stem for p in self.config_dir.glob("*.yaml"))

    def _load_config(self, name: str) -> Dict[str, Any]:
        """Load and minimally validate one experiment's YAML config.

        Args:
            name: Experiment name (YAML filename stem under config_dir).

        Returns:
            Dict with keys ``name``, ``strategy``, ``seeds``, ``params``.

        Raises:
            FileNotFoundError: If ``config/experiments/<name>.yaml``
                does not exist.
            ValueError: If required keys are missing or malformed.
        """
        config_path = self.config_dir / f"{name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No experiment config found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for key in ("name", "strategy", "seeds", "params"):
            if key not in config:
                raise ValueError(f"{config_path} is missing required key {key!r}")
        if config["strategy"] not in _STRATEGIES:
            raise ValueError(
                f"{config_path}: strategy must be one of {sorted(_STRATEGIES)}, "
                f"got {config['strategy']!r}"
            )
        if not isinstance(config["seeds"], list) or not config["seeds"]:
            raise ValueError(f"{config_path}: 'seeds' must be a non-empty list")
        if not isinstance(config["params"], dict):
            raise ValueError(f"{config_path}: 'params' must be a mapping")

        return config

    # ------------------------------------------------------------------
    # TASK 2/3 -- Single-experiment execution (reuses CentralizedTrainer /
    # LocalSGDTrainer -- no training logic is duplicated here)
    # ------------------------------------------------------------------

    def run_experiment(self, name: str, seed: Optional[int] = None) -> Dict[str, Any]:
        """Run one experiment config for one seed.

        Args:
            name: Experiment name (matches a YAML file under
                config_dir).
            seed: Seed to run with. Defaults to the first entry in that
                config's ``seeds`` list.

        Returns:
            A metric record with the fields in ``_RESULT_FIELDS``.

        Raises:
            FileNotFoundError: If the config is missing.
            ValueError: If the config is malformed.
            RuntimeError: If the run completed without producing the
                artifacts CentralizedTrainer/LocalSGDTrainer are
                contracted to write (Task 10 -- validation).
            Exception: Any exception raised by CentralizedTrainer/
                LocalSGDTrainer.run() itself is logged and re-raised
                unchanged, not swallowed.
        """
        config = self._load_config(name)
        seed = seed if seed is not None else config["seeds"][0]
        params = dict(config["params"])
        params.setdefault("use_amp", False)
        params["seed"] = seed
        

        run_dir = self.artifacts_dir / name / f"seed_{seed}"
        strategy = config["strategy"]

        logger.info("=== Experiment start: %s (strategy=%s, seed=%d) ===", name, strategy, seed)
        logger.info("Configuration: %s", params)
        start = time.time()

        # Phase 7.5, Task 3 -- automatically start monitoring when
        # training begins for EVERY experiment (both strategies), and
        # stop it when training finishes, with no manual calls
        # required from the caller. LocalSGDTrainer additionally feeds
        # per-round sync duration/communication cost into this same
        # monitor (see LocalSGDConfig/LocalSGDTrainer's
        # system_monitor= injection below).
        monitor = SystemMonitor()
        monitor.start()
        monitor.mark_training_start()
        monitor.collect()  # guarantee >=1 sample even for very short runs
        try:
            if strategy == "centralized":
                trainer = CentralizedTrainer(CentralizedConfig(artifacts_dir=run_dir, **params))
                summary = trainer.run()
                model = trainer.model
                communication_rounds = 0
                synchronization_interval = None
                num_workers = 1
                communication_manager = None
            else:  # "local_sgd"
                trainer = LocalSGDTrainer(
                    LocalSGDConfig(artifacts_dir=run_dir, **params),
                    system_monitor=monitor,
                )
                summary = trainer.run()
                model = trainer.global_model
                communication_rounds = trainer.config.communication_rounds
                synchronization_interval = trainer.config.local_epochs
                num_workers = len(trainer.shard_names)
                communication_manager = trainer.communication_manager
        except Exception:
            logger.exception("Experiment %s (seed=%d) failed", name, seed)
            raise
        finally:
            monitor.mark_training_end()
            monitor.stop()
        run_duration = time.time() - start

        # Phase 7.5, Task 4/5/7 -- for centralized runs (no
        # LocalSGDTrainer to export it), export this run's monitoring
        # artifacts/plots directly; local_sgd already did this inside
        # trainer.run() (network + monitor together, since they share
        # one CommunicationManager.export() call).
        if strategy == "centralized" and monitor.samples:
            try:
                monitor.export(run_dir / "monitoring")
                monitor.generate_visualizations(run_dir / "network_plots")
                monitor.export(_DEFAULT_NETWORK_ARTIFACTS_DIR)
                monitor.generate_visualizations(_DEFAULT_NETWORK_PLOTS_DIR)
            except Exception:
                logger.exception("System monitor export failed for %s (seed=%d)", name, seed)

        network_summary = communication_manager.get_metrics_summary() if communication_manager else None
        monitor_summary = monitor.get_summary() if monitor.samples else None

        eval_metrics = summary["evaluation_metrics"]
        param_count = count_model_parameters(model)
        comm_cost = communication_cost(communication_rounds, num_workers, param_count)

        record: Dict[str, Any] = {
            "name": name,
            "strategy": strategy,
            "seed": seed,
            "loss": eval_metrics["loss"],
            "accuracy": eval_metrics["accuracy"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "f1": eval_metrics["f1"],
            "roc_auc": eval_metrics["roc_auc"],
            "pr_auc": eval_metrics["pr_auc"],
            "training_time_seconds": summary["train_duration_seconds"],
            "communication_rounds": communication_rounds,
            "synchronization_interval": synchronization_interval,
            "communication_cost_params": comm_cost,
            "communication_cost_bytes": comm_cost * BYTES_PER_PARAM,
            "param_count": param_count,
            "run_duration_seconds": run_duration,
            # Phase 7.5, Task 7 -- network statistics (0.0 by
            # construction for centralized runs and any local_sgd run
            # where network simulation was disabled/unavailable).
            "network_avg_latency_seconds": (network_summary or {}).get("avg_latency_seconds", 0.0),
            "network_max_latency_seconds": (network_summary or {}).get("max_latency_seconds", 0.0),
            "network_min_latency_seconds": (network_summary or {}).get("min_latency_seconds", 0.0),
            "network_avg_bandwidth_mbps": (network_summary or {}).get("avg_bandwidth_mbps", 0.0),
            "network_total_bytes_transferred": (network_summary or {}).get("transmitted_bytes", 0.0),
            "network_synchronization_cost_seconds": (network_summary or {}).get(
                "total_sync_duration_seconds", 0.0
            ),
            "cpu_percent_mean": ((monitor_summary or {}).get("cpu_percent") or {}).get("mean") or 0.0,
            "ram_percent_mean": ((monitor_summary or {}).get("ram_percent") or {}).get("mean") or 0.0,
            "gpu_percent_mean": (
                ((monitor_summary or {}).get("gpu_utilization_percent") or {}).get("mean") or 0.0
            ),
            "experiment_duration_seconds": (monitor_summary or {}).get(
                "experiment_duration_seconds"
            ) or run_duration,
        }

        self._validate_run_artifacts(run_dir)

        metadata_path = run_dir / "experiment_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        try:
            plot_paths = generate_experiment_plots(
                run_dir / "metrics" / "training_metrics.csv", strategy, run_dir / "plots"
            )
            logger.info("Per-run plots: %s", list(plot_paths))
        except Exception:
            # Plot generation is diagnostic, not part of the trained
            # artifact -- a plotting failure should not discard a
            # successfully completed and already-persisted training run.
            logger.exception("Per-run plot generation failed for %s (seed=%d); run metrics are still saved", name, seed)

        logger.info(
            "=== Experiment end: %s (seed=%d) duration=%.2fs loss=%.4f f1=%.4f accuracy=%.4f ===",
            name, seed, run_duration, record["loss"], record["f1"], record["accuracy"],
        )
        return record

    def _validate_run_artifacts(self, run_dir: Path) -> None:
        """Confirm the trainer actually wrote what it's contracted to
        write before this file adds its own metadata on top (Task 10).

        Raises:
            RuntimeError: If any expected artifact is missing.
        """
        required = [
            run_dir / "evaluation_metrics.json",
            run_dir / "metrics" / "training_metrics.csv",
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise RuntimeError(
                f"Experiment at {run_dir} completed without producing expected artifact(s): {missing}"
            )

    # ------------------------------------------------------------------
    # TASK 2/8 -- Repeated runs with reproducibility statistics
    # ------------------------------------------------------------------

    def repeat_experiment(self, name: str, seeds: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """Run one experiment config across multiple seeds and aggregate.

        Args:
            name: Experiment name.
            seeds: Seeds to run. Defaults to that config's ``seeds`` list.

        Returns:
            Dict with ``runs`` (one record per seed) and ``aggregate``
            (mean/std per metric, from ``analysis.metrics.aggregate_seeds``).

        Raises:
            Same as ``run_experiment`` for any individual seed's run.
        """
        from analysis.metrics import aggregate_seeds  # local import: avoids a module-level

        config = self._load_config(name)
        seeds = list(seeds) if seeds is not None else config["seeds"]

        records = [self.run_experiment(name, seed) for seed in seeds]
        aggregate = aggregate_seeds(records)

        exp_dir = self.artifacts_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "aggregate_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"runs": records, "aggregate": aggregate}, f, indent=2)

        logger.info(
            "Experiment %s: %d seed(s) complete, mean f1=%.4f (std=%.4f)",
            name, len(records), aggregate["f1"]["mean"], aggregate["f1"]["std"],
        )
        return {"runs": records, "aggregate": aggregate}

    # ------------------------------------------------------------------
    # TASK 3 -- Run every configured experiment
    # ------------------------------------------------------------------

    def run_all(self, seeds: Optional[Sequence[int]] = None) -> List[Dict[str, Any]]:
        """Run every experiment config found under config_dir.

        A failure in one experiment is logged and that experiment is
        skipped (with its error recorded), rather than aborting the
        entire benchmark sweep -- so a single misconfigured K value
        does not prevent reporting on the others.

        Args:
            seeds: If given, overrides every config's own ``seeds``
                list. If omitted, each config's own seeds are used.

        Returns:
            Flat list of per-seed run records across every
            successfully completed experiment. Also stored on
            ``self`` for subsequent ``generate_plots()``/
            ``generate_report()`` calls with no arguments.

        Raises:
            RuntimeError: If every configured experiment failed (there
                is nothing to plot or report).
        """
        names = self.list_experiment_names()
        if not names:
            raise RuntimeError(f"No experiment configs found under {self.config_dir}")

        all_records: List[Dict[str, Any]] = []
        errors: Dict[str, str] = {}
        for name in names:
            try:
                result = self.repeat_experiment(name, seeds)
                all_records.extend(result["runs"])
            except Exception as exc:  # noqa: BLE001 -- intentionally broad: isolate one bad config
                logger.error("Skipping experiment %r due to error: %s", name, exc)
                errors[name] = str(exc)

        if not all_records:
            raise RuntimeError(f"All {len(names)} experiment(s) failed: {errors}")
        if errors:
            logger.warning("run_all completed with %d failing experiment(s): %s", len(errors), errors)

        self._results = all_records
        self._write_results(all_records)
        return all_records

    def _write_results(self, records: List[Dict[str, Any]]) -> None:
        """Write results.csv and results.json (Task 4)."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        json_path = self.artifacts_dir / "results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        csv_path = self.artifacts_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_RESULT_FIELDS)
            writer.writeheader()
            for record in records:
                writer.writerow({k: record.get(k) for k in _RESULT_FIELDS})

        logger.info("Results written: %s, %s (%d run(s))", json_path, csv_path, len(records))

    # ------------------------------------------------------------------
    # TASK 5 -- Plots
    # ------------------------------------------------------------------

    def generate_plots(self, results: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate the cross-experiment tradeoff plots (Task 5) under
        analysis/plots/. Per-run curve plots are already generated
        inside ``run_experiment``.

        Args:
            results: Results to plot from. Defaults to the results of
                the most recent ``run_all()`` call on this instance.

        Returns:
            Dict mapping plot name to written file path.

        Raises:
            RuntimeError: If no results are available.
        """
        results = results if results is not None else self._results
        if not results:
            raise RuntimeError(
                "generate_plots() has no results to plot; call run_all() first or pass results="
            )
        paths = generate_cross_experiment_plots(results, self.plots_dir)
        missing = [p for p in paths.values() if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"generate_plots() did not produce expected file(s): {missing}")
        return paths

    # ------------------------------------------------------------------
    # TASK 6 -- Reports
    # ------------------------------------------------------------------

    def generate_report(self, results: Optional[Sequence[Dict[str, Any]]] = None):
        """Generate analysis/report.json and analysis/report.md (Task 6).

        Args:
            results: Results to report on. Defaults to the results of
                the most recent ``run_all()`` call on this instance.

        Returns:
            ``(report_json_path, report_md_path)``.

        Raises:
            RuntimeError: If no results are available.
        """
        results = results if results is not None else self._results
        if not results:
            raise RuntimeError(
                "generate_report() has no results to report on; call run_all() first or pass results="
            )
        return write_report(results, self.analysis_dir)


def main() -> None:
    """CLI entry point (Task acceptance criteria): run every configured
    experiment end-to-end, then generate plots and reports.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    runner = ExperimentRunner()
    results = runner.run_all()
    runner.generate_plots(results)
    runner.generate_report(results)
    logger.info("Phase 6 experiment sweep complete: %d run(s) across %d experiment(s)",
                len(results), len({r["name"] for r in results}))


if __name__ == "__main__":
    main()
