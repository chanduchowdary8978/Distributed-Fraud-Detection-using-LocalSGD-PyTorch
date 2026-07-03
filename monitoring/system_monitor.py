"""
system_monitor.py

Purpose:
    Resource monitoring for the Distributed System Simulation Layer
    (Task 7): CPU/RAM/GPU utilization and memory, synchronization
    duration, communication cost, training duration, and experiment
    duration, collected as a time series plus a summary. Pure
    monitoring -- no training, networking, or ML logic lives here.

Dependencies:
    Uses ``psutil`` for CPU/RAM (added to requirements.txt). GPU
    metrics are best-effort: ``torch.cuda`` is used if importable and
    a CUDA device is available, otherwise ``nvidia-smi`` is tried via
    subprocess, otherwise GPU fields are reported as unavailable
    (``gpu_available: False``) rather than raising -- a machine with
    no GPU is a normal, supported environment for this simulation.

Public Interface:
    class SystemMonitor
        Methods:
            start()
            stop()
            collect() -> dict
            export(output_dir) -> Path
"""

from __future__ import annotations

import csv
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil

logger = logging.getLogger(__name__)

_SYSTEM_METRICS_FIELDS = [
    "timestamp",
    "elapsed_seconds",
    "cpu_percent",
    "ram_percent",
    "ram_used_mb",
    "gpu_available",
    "gpu_utilization_percent",
    "gpu_memory_used_mb",
    "gpu_memory_total_mb",
]


@dataclass
class SystemMonitorConfig:
    """Configuration for :class:`SystemMonitor` (Task 9).

    Attributes:
        sampling_interval_seconds: Time between background samples
            while ``start()``/``stop()`` are active.
        artifacts_dir: Default export directory (``export()`` accepts
            an override).
    """

    sampling_interval_seconds: float = 1.0
    artifacts_dir: Path = Path(__file__).resolve().parent.parent / "artifacts" / "network"

    def __post_init__(self) -> None:
        self.artifacts_dir = Path(self.artifacts_dir)
        if self.sampling_interval_seconds <= 0:
            raise ValueError(
                f"sampling_interval_seconds must be positive, got {self.sampling_interval_seconds!r}"
            )


class SystemMonitor:
    """Samples system resource utilization on a background thread and
    tracks synchronization/communication/training/experiment duration
    (Task 7).

    Args:
        config: Optional ``SystemMonitorConfig``.

    Usage:
        >>> monitor = SystemMonitor()
        >>> monitor.start()
        >>> monitor.mark_training_start()
        ... # training / communication happens ...
        >>> monitor.record_sync_duration(0.42)
        >>> monitor.record_communication_cost(transmitted_bytes=1_048_576)
        >>> monitor.mark_training_end()
        >>> monitor.stop()
        >>> monitor.export("artifacts/network")
    """

    def __init__(self, config: Optional[SystemMonitorConfig] = None) -> None:
        self.config = config or SystemMonitorConfig()

        self.samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        self._experiment_start: Optional[float] = None
        self._experiment_end: Optional[float] = None
        self._training_start: Optional[float] = None
        self._training_end: Optional[float] = None
        self._sync_durations: List[float] = []
        self._communication_costs_bytes: List[float] = []

        self._gpu_backend = self._detect_gpu_backend()

    # ------------------------------------------------------------------
    # TASK 7 -- GPU backend detection (best-effort, never fatal)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_gpu_backend() -> Optional[str]:
        try:
            import torch  # noqa: F401
            if torch.cuda.is_available():
                return "torch"
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                return "nvidia-smi"
        except Exception:
            pass
        return None

    def _gpu_snapshot(self) -> Dict[str, Any]:
        if self._gpu_backend == "torch":
            try:
                import torch

                util = None
                try:
                    util = torch.cuda.utilization()
                except Exception:
                    pass
                mem_used_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                mem_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                return {
                    "gpu_available": True,
                    "gpu_utilization_percent": util,
                    "gpu_memory_used_mb": mem_used_mb,
                    "gpu_memory_total_mb": mem_total_mb,
                }
            except Exception:
                logger.debug("torch GPU snapshot failed; falling back to unavailable", exc_info=True)

        if self._gpu_backend == "nvidia-smi":
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=2,
                )
                util_str, used_str, total_str = result.stdout.strip().split(",")[:3]
                return {
                    "gpu_available": True,
                    "gpu_utilization_percent": float(util_str),
                    "gpu_memory_used_mb": float(used_str),
                    "gpu_memory_total_mb": float(total_str),
                }
            except Exception:
                logger.debug("nvidia-smi GPU snapshot failed; falling back to unavailable", exc_info=True)

        return {
            "gpu_available": False,
            "gpu_utilization_percent": None,
            "gpu_memory_used_mb": None,
            "gpu_memory_total_mb": None,
        }

    # ------------------------------------------------------------------
    # TASK 7 -- Single snapshot
    # ------------------------------------------------------------------

    def collect(self) -> Dict[str, Any]:
        """Take one resource-utilization snapshot (Task 7).

        Returns:
            Dict matching ``_SYSTEM_METRICS_FIELDS``.
        """
        vm = psutil.virtual_memory()
        sample = {
            "timestamp": time.time(),
            "elapsed_seconds": (
                time.time() - self._experiment_start if self._experiment_start is not None else 0.0
            ),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": vm.percent,
            "ram_used_mb": vm.used / (1024 ** 2),
        }
        sample.update(self._gpu_snapshot())
        with self._lock:
            self.samples.append(sample)
        return sample

    # ------------------------------------------------------------------
    # TASK 7 -- Background sampling
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background sampling and mark the experiment start
        time (Task 7). Idempotent -- calling ``start()`` while already
        running is a no-op (with a warning).
        """
        if self._running:
            logger.warning("SystemMonitor.start() called while already running; ignoring")
            return
        self._experiment_start = time.time()
        self._stop_event.clear()
        self._running = True
        # Prime psutil.cpu_percent's internal baseline so the first
        # background sample isn't a meaningless 0.0 comparing against
        # process start.
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info("SystemMonitor started (sampling every %.2fs)", self.config.sampling_interval_seconds)

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.collect()
            except Exception:
                logger.exception("SystemMonitor background sample failed; continuing")
            self._stop_event.wait(self.config.sampling_interval_seconds)

    def stop(self) -> None:
        """Stop background sampling and mark the experiment end time
        (Task 7). Safe to call even if ``start()`` was never called.
        """
        if not self._running:
            logger.warning("SystemMonitor.stop() called while not running; ignoring")
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.config.sampling_interval_seconds * 2)
        self._running = False
        self._experiment_end = time.time()
        logger.info(
            "SystemMonitor stopped: %d sample(s), experiment_duration=%.2fs",
            len(self.samples), self.experiment_duration_seconds or 0.0,
        )

    # ------------------------------------------------------------------
    # TASK 7 -- Duration / cost tracking hooks
    # ------------------------------------------------------------------

    def mark_training_start(self) -> None:
        """Record the start of a training phase."""
        self._training_start = time.time()

    def mark_training_end(self) -> None:
        """Record the end of a training phase.

        Raises:
            RuntimeError: If called before ``mark_training_start()``.
        """
        if self._training_start is None:
            raise RuntimeError("mark_training_end() called before mark_training_start()")
        self._training_end = time.time()

    def record_sync_duration(self, seconds: float) -> None:
        """Record one synchronization event's duration (typically the
        ``total_sync_duration_seconds`` from
        ``network.communication.CommunicationManager.synchronize()``).

        Raises:
            ValueError: If ``seconds`` is negative.
        """
        if seconds < 0:
            raise ValueError(f"sync duration must be non-negative, got {seconds!r}")
        self._sync_durations.append(seconds)

    def record_communication_cost(self, transmitted_bytes: float) -> None:
        """Record one synchronization event's communication cost in
        bytes (typically ``transmitted_bytes`` from the same
        ``synchronize()`` call).

        Raises:
            ValueError: If ``transmitted_bytes`` is negative.
        """
        if transmitted_bytes < 0:
            raise ValueError(f"transmitted_bytes must be non-negative, got {transmitted_bytes!r}")
        self._communication_costs_bytes.append(transmitted_bytes)

    @property
    def training_duration_seconds(self) -> Optional[float]:
        if self._training_start is None:
            return None
        end = self._training_end if self._training_end is not None else time.time()
        return end - self._training_start

    @property
    def experiment_duration_seconds(self) -> Optional[float]:
        if self._experiment_start is None:
            return None
        end = self._experiment_end if self._experiment_end is not None else time.time()
        return end - self._experiment_start

    # ------------------------------------------------------------------
    # TASK 11 -- Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Sanity-check collected state before export (Task 11).

        Raises:
            RuntimeError: If no samples were collected and no manual
                ``collect()`` call was ever made (nothing to export).
        """
        if not self.samples:
            raise RuntimeError(
                "SystemMonitor has no samples; call start()/stop() or collect() before export()"
            )

    # ------------------------------------------------------------------
    # TASK 7 -- Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of tracked durations, communication cost,
        and resource utilization statistics (Task 7).
        """
        cpu_vals = [s["cpu_percent"] for s in self.samples]
        ram_vals = [s["ram_percent"] for s in self.samples]
        gpu_vals = [s["gpu_utilization_percent"] for s in self.samples if s.get("gpu_utilization_percent") is not None]

        def _stats(values: List[float]) -> Dict[str, Optional[float]]:
            if not values:
                return {"mean": None, "max": None, "min": None}
            return {"mean": sum(values) / len(values), "max": max(values), "min": min(values)}

        return {
            "num_samples": len(self.samples),
            "sampling_interval_seconds": self.config.sampling_interval_seconds,
            "cpu_percent": _stats(cpu_vals),
            "ram_percent": _stats(ram_vals),
            "gpu_utilization_percent": _stats(gpu_vals),
            "gpu_available": bool(self._gpu_backend),
            "training_duration_seconds": self.training_duration_seconds,
            "experiment_duration_seconds": self.experiment_duration_seconds,
            "synchronization_duration": {
                "count": len(self._sync_durations),
                "total_seconds": sum(self._sync_durations),
                **_stats(self._sync_durations),
            },
            "communication_cost": {
                "count": len(self._communication_costs_bytes),
                "total_bytes": sum(self._communication_costs_bytes),
                **_stats(self._communication_costs_bytes),
            },
        }

    # ------------------------------------------------------------------
    # Phase 7.5, Task 5 -- Visualizations (cpu_ram_usage, gpu_memory_usage,
    # system_resource_usage). Mirrors the plotting style already used by
    # network.network_simulator.NetworkSimulator's visualize_* methods.
    # ------------------------------------------------------------------

    def visualize_cpu_ram_usage(self, output_path: Optional[Union[str, Path]] = None):
        """Plot CPU and RAM utilization (%) over elapsed time."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [s["elapsed_seconds"] for s in self.samples]
        cpu = [s["cpu_percent"] for s in self.samples]
        ram = [s["ram_percent"] for s in self.samples]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, cpu, marker="o", markersize=3, color="steelblue", label="CPU %")
        ax.plot(xs, ram, marker="o", markersize=3, color="seagreen", label="RAM %")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Utilization (%)")
        ax.set_title("CPU / RAM Usage")
        ax.legend()
        fig.tight_layout()
        return self._finish_plot(fig, output_path)

    def visualize_gpu_memory_usage(self, output_path: Optional[Union[str, Path]] = None):
        """Plot GPU memory used (MB) over elapsed time. Returns ``None``
        without writing anything if no GPU-bearing sample was ever
        collected (Task 5 -- "skip GPU plots gracefully" rather than
        failing on a machine with no GPU).
        """
        gpu_samples = [s for s in self.samples if s.get("gpu_available")]
        if not gpu_samples:
            logger.info("No GPU available in any sample; skipping gpu_memory_usage plot")
            return None

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [s["elapsed_seconds"] for s in gpu_samples]
        used = [s["gpu_memory_used_mb"] for s in gpu_samples]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, used, marker="o", markersize=3, color="darkorange")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("GPU memory used (MB)")
        ax.set_title("GPU Memory Usage")
        fig.tight_layout()
        return self._finish_plot(fig, output_path)

    def visualize_system_resource_usage(self, output_path: Optional[Union[str, Path]] = None):
        """Combined CPU / RAM / GPU utilization overview in one figure
        (three stacked panels; the GPU panel is omitted, not left
        blank, when no GPU is available -- Task 5).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [s["elapsed_seconds"] for s in self.samples]
        cpu = [s["cpu_percent"] for s in self.samples]
        ram = [s["ram_percent"] for s in self.samples]
        gpu_samples = [s for s in self.samples if s.get("gpu_available")]

        n_panels = 3 if gpu_samples else 2
        fig, axes = plt.subplots(n_panels, 1, figsize=(8, 3 * n_panels), sharex=True)

        axes[0].plot(xs, cpu, color="steelblue")
        axes[0].set_ylabel("CPU %")
        axes[0].set_title("System Resource Usage")

        axes[1].plot(xs, ram, color="seagreen")
        axes[1].set_ylabel("RAM %")

        if gpu_samples:
            gxs = [s["elapsed_seconds"] for s in gpu_samples]
            gutil = [s["gpu_utilization_percent"] for s in gpu_samples]
            axes[2].plot(gxs, gutil, color="darkorange")
            axes[2].set_ylabel("GPU %")

        axes[-1].set_xlabel("Elapsed time (s)")
        fig.tight_layout()
        return self._finish_plot(fig, output_path)

    @staticmethod
    def _finish_plot(fig, output_path: Optional[Union[str, Path]]):
        import matplotlib.pyplot as plt

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=120)
            plt.close(fig)
            return None
        return fig

    def generate_visualizations(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """Generate the Task 5 system-resource plots into ``output_dir``
        (typically ``analysis/network_plots/``, alongside the network
        plots -- Task 5 lists them in the same directory).

        GPU plots are skipped gracefully (no exception, no empty file)
        when no GPU was ever available during sampling.

        Raises:
            RuntimeError: If no samples were collected.
        """
        self.validate()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, Path] = {}
        cpu_ram_path = output_dir / "cpu_ram_usage.png"
        self.visualize_cpu_ram_usage(cpu_ram_path)
        paths["cpu_ram_usage"] = cpu_ram_path

        gpu_path = output_dir / "gpu_memory_usage.png"
        self.visualize_gpu_memory_usage(gpu_path)
        if gpu_path.exists():
            paths["gpu_memory_usage"] = gpu_path

        resource_path = output_dir / "system_resource_usage.png"
        self.visualize_system_resource_usage(resource_path)
        paths["system_resource_usage"] = resource_path

        logger.info("Generated %d system-resource visualization(s) in %s", len(paths), output_dir)
        return paths

    # ------------------------------------------------------------------
    # TASK 10 -- Export
    # ------------------------------------------------------------------

    def export(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Write ``system_metrics.csv`` (the collected time series) to
        ``output_dir`` (Task 10). Also writes a companion
        ``system_summary.json`` alongside it for convenience.

        Args:
            output_dir: Destination directory. Defaults to
                ``self.config.artifacts_dir`` (``artifacts/network/``).

        Returns:
            Path to ``system_metrics.csv``.

        Raises:
            RuntimeError: If no samples have been collected.
        """
        self.validate()
        output_dir = Path(output_dir) if output_dir is not None else self.config.artifacts_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "system_metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_SYSTEM_METRICS_FIELDS)
            writer.writeheader()
            for sample in self.samples:
                writer.writerow({k: sample.get(k) for k in _SYSTEM_METRICS_FIELDS})

        summary_path = output_dir / "system_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.get_summary(), f, indent=2)

        if not csv_path.exists():
            raise RuntimeError(f"Failed to write {csv_path}")
        logger.info("System metrics exported: %s (%d samples)", csv_path, len(self.samples))
        return csv_path
