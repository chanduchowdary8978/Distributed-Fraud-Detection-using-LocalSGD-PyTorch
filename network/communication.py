"""
communication.py

Purpose:
    Reusable communication primitives (send/receive/broadcast/
    synchronize) for the Distributed System Simulation Layer. This is
    the primary entry point future phases (Phase 8+) should import to
    add simulated networking to a distributed training strategy --
    ``training/local_sgd.py`` and friends are never modified; this
    module only wraps around them from the outside, given plain
    "how many parameters / bytes am I sending" numbers.

Relationship to network_simulator.py:
    ``NetworkSimulator`` computes the timing/outcome of a single
    logical transfer (Task 2/3/6). ``CommunicationManager`` is the
    higher-level engine (Task 4) that turns "worker 3 sends its model
    to the coordinator" into one or more simulator calls, records the
    resulting event (Task 5), and accumulates the running metrics
    (Task 5) and CSV/JSON artifacts (Task 10) a caller needs.

Public Interface:
    class CommunicationManager
        Methods:
            send(src, dst, payload_bytes, num_parameters=None, metadata=None) -> dict
            receive(event) -> dict
            broadcast(src, payload_bytes, targets=None, num_parameters=None, metadata=None) -> list[dict]
            synchronize(num_parameters, bytes_per_parameter=4, participant_ids=None, coordinator=None, metadata=None) -> dict
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from network.network_simulator import NetworkSimulator

logger = logging.getLogger(__name__)

_NETWORK_METRICS_FIELDS = [
    "round",
    "communication_rounds_so_far",
    "synchronization_interval",
    "participants",
    "transmitted_parameters",
    "transmitted_bytes",
    "avg_latency_seconds",
    "max_latency_seconds",
    "bandwidth_mbps",
    "serialization_time_seconds",
    "deserialization_time_seconds",
    "transfer_time_seconds",
    "total_sync_duration_seconds",
    "dropped_events",
    "delayed_events",
    "disconnected_events",
]

_COMMUNICATION_LOG_FIELDS = [
    "event_id",
    "event_type",
    "round",
    "src",
    "dst",
    "status",
    "num_parameters",
    "payload_bytes",
    "latency_seconds",
    "bandwidth_mbps",
    "serialization_time_seconds",
    "deserialization_time_seconds",
    "transfer_time_seconds",
    "total_time_seconds",
    "sim_timestamp_seconds",
]


class CommunicationManager:
    """High-level communication engine over a ``NetworkSimulator``
    (Task 4). Owns the running communication metrics (Task 5) and the
    on-disk artifacts described in Task 10.

    Args:
        simulator: A ``NetworkSimulator`` instance (topology + latency
            + bandwidth + failure models already configured).
        synchronization_interval: Optional label for how many local
            steps occur between synchronizations (purely descriptive
            metadata written into ``network_metrics.csv`` / summary --
            this module has no opinion on training-loop structure).
        coordinator: Default aggregation point for ``synchronize()``/
            ``broadcast()``. Defaults to
            ``simulator.topology.coordinator_index``.
    """

    def __init__(
        self,
        simulator: NetworkSimulator,
        synchronization_interval: Optional[int] = None,
        coordinator: Optional[int] = None,
    ) -> None:
        self.simulator = simulator
        self.synchronization_interval = synchronization_interval
        self.coordinator = (
            coordinator if coordinator is not None else simulator.topology.coordinator_index
        )
        self.simulator.topology.validate_worker_id(self.coordinator)

        self.round_records: List[Dict[str, Any]] = []
        self._communication_rounds = 0
        self._total_transmitted_parameters = 0
        self._total_transmitted_bytes = 0.0

    # ------------------------------------------------------------------
    # TASK 4 -- send / receive
    # ------------------------------------------------------------------

    def send(
        self,
        src: int,
        dst: int,
        payload_bytes: float,
        num_parameters: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Simulate sending ``payload_bytes`` from ``src`` to ``dst``,
        applying the configured latency/bandwidth/failure models and
        recording the resulting event (Task 4/5/6).

        Returns:
            The recorded event dict (see
            ``NetworkSimulator.record_event`` for the schema). If the
            failure model drops the message, ``status`` will be
            ``"dropped"``/``"disconnected"`` and ``payload_bytes`` in
            the *transmitted* metrics will not include it (the event
            itself still records the attempted size).
        """
        self.simulator.topology.validate_worker_id(src)
        self.simulator.topology.validate_worker_id(dst)

        failure = self.simulator.apply_failure(src, dst)
        status = failure["status"]

        if status in ("dropped", "disconnected"):
            event = self.simulator.record_event(
                event_type="send", src=src, dst=dst, payload_bytes=payload_bytes,
                latency_seconds=0.0, transfer_time_seconds=0.0,
                num_parameters=num_parameters, status=status, extra=metadata,
            )
            logger.info("send(%s -> %s) %s (payload=%d bytes not delivered)", src, dst, status, payload_bytes)
            return event

        latency_s = self.simulator.simulate_latency(src, dst) * failure["delay_multiplier"]
        transfer_s = self.simulator.estimate_transfer_time(payload_bytes, src, dst)
        event = self.simulator.record_event(
            event_type="send", src=src, dst=dst, payload_bytes=payload_bytes,
            latency_seconds=latency_s, transfer_time_seconds=transfer_s,
            num_parameters=num_parameters, status=status, extra=metadata,
        )
        self._accumulate(event)
        return event

    def receive(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Acknowledge receipt of a previously recorded ``send``/
        ``broadcast`` event (Task 4). This is a logical companion to
        ``send`` -- since everything is simulated in a single process
        there is nothing to actually poll for, so this simply
        validates the event was delivered and returns it, giving
        callers a symmetric send/receive API to build on.

        Args:
            event: An event dict previously returned by ``send`` or
                one entry from ``broadcast``'s returned list.

        Returns:
            The same event dict, unchanged.

        Raises:
            ValueError: If the event does not look like a delivered
                communication event (missing required keys, or its
                status indicates it never arrived).
        """
        required = {"event_id", "status", "payload_bytes"}
        if not required.issubset(event.keys()):
            raise ValueError(f"receive() requires an event dict with keys {required}, got {set(event.keys())}")
        if event["status"] in ("dropped", "disconnected"):
            raise ValueError(
                f"Cannot receive event {event['event_id']}: message never arrived (status={event['status']!r})"
            )
        return event

    # ------------------------------------------------------------------
    # TASK 4 -- broadcast
    # ------------------------------------------------------------------

    def broadcast(
        self,
        src: int,
        payload_bytes: float,
        targets: Optional[Sequence[int]] = None,
        num_parameters: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Simulate ``src`` sending ``payload_bytes`` to every worker in
        ``targets`` (defaults to every other worker) -- e.g. the
        coordinator broadcasting averaged weights back out after a
        synchronization round (Task 4).

        Returns:
            List of event dicts, one per target, index-aligned with
            (the resolved) ``targets``.
        """
        self.simulator.topology.validate_worker_id(src)
        targets = list(targets) if targets is not None else [
            w for w in range(self.simulator.topology.num_workers) if w != src
        ]
        events = [
            self.send(src, dst, payload_bytes, num_parameters=num_parameters, metadata=metadata)
            for dst in targets
        ]
        return events

    # ------------------------------------------------------------------
    # TASK 4/5 -- synchronize
    # ------------------------------------------------------------------

    def synchronize(
        self,
        num_parameters: int,
        bytes_per_parameter: int = 4,
        participant_ids: Optional[Sequence[int]] = None,
        coordinator: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Simulate one full synchronization round: every participant
        uploads its parameters to the coordinator, then the
        coordinator broadcasts the aggregated result back out (Task
        4). Applies failure simulation per-participant (a dropped
        synchronization for a worker means that worker's upload is
        skipped; a disconnected worker is skipped entirely for the
        round -- Task 6), and records a per-round aggregate metric
        (Task 5).

        Args:
            num_parameters: Number of scalar trainable parameters in
                the model being synchronized (e.g.
                ``sum(p.numel() for p in model.parameters())``).
            bytes_per_parameter: Bytes per parameter (4 for float32).
            participant_ids: Workers taking part. Defaults to every
                worker except the coordinator.
            coordinator: Aggregation point. Defaults to
                ``self.coordinator``.
            metadata: Optional extra fields merged into every
                underlying send/broadcast event.

        Returns:
            A round-summary dict (also appended to
            ``self.round_records``) with keys: ``round``,
            ``participants``, ``transmitted_parameters``,
            ``transmitted_bytes``, ``upload_time_seconds``,
            ``broadcast_time_seconds``, ``total_sync_duration_seconds``,
            ``dropped_workers``, ``disconnected_workers``.

        Raises:
            ValueError: If ``num_parameters`` is not positive.
        """
        if num_parameters <= 0:
            raise ValueError(f"num_parameters must be positive, got {num_parameters!r}")

        coordinator = coordinator if coordinator is not None else self.coordinator
        self.simulator.topology.validate_worker_id(coordinator)
        participants = list(participant_ids) if participant_ids is not None else [
            w for w in range(self.simulator.topology.num_workers) if w != coordinator
        ]

        round_index = self.simulator.next_round()
        payload_bytes = num_parameters * bytes_per_parameter

        upload_events: List[Dict[str, Any]] = []
        dropped_workers: List[int] = []
        for worker in participants:
            if self.simulator.sync_round_dropped(worker):
                dropped_workers.append(worker)
                continue
            event = self.send(worker, coordinator, payload_bytes, num_parameters=num_parameters, metadata=metadata)
            event["event_type"] = "synchronize"
            upload_events.append(event)

        active_participants = [e["src"] for e in upload_events if e["status"] == "ok"]
        broadcast_events = self.broadcast(
            coordinator, payload_bytes, targets=active_participants,
            num_parameters=num_parameters, metadata=metadata,
        )
        for event in broadcast_events:
            event["event_type"] = "synchronize"

        delivered = [e for e in upload_events + broadcast_events if e["status"] == "ok"]
        disconnected = [e for e in upload_events + broadcast_events if e["status"] == "disconnected"]

        upload_time = max((e["total_time_seconds"] for e in upload_events if e["status"] == "ok"), default=0.0)
        broadcast_time = max((e["total_time_seconds"] for e in broadcast_events if e["status"] == "ok"), default=0.0)
        total_duration = upload_time + broadcast_time

        self._communication_rounds += 1
        record = {
            "round": round_index,
            "coordinator": coordinator,
            "participants": participants,
            "active_participants": active_participants,
            "transmitted_parameters": num_parameters * len(delivered),
            "transmitted_bytes": payload_bytes * len(delivered),
            "upload_time_seconds": upload_time,
            "broadcast_time_seconds": broadcast_time,
            "total_sync_duration_seconds": total_duration,
            "dropped_workers": dropped_workers,
            "disconnected_events": len(disconnected),
        }
        self.round_records.append(record)
        logger.info(
            "synchronize() round=%d participants=%d delivered=%d duration=%.4fs",
            round_index, len(participants), len(delivered), total_duration,
        )
        return record

    # ------------------------------------------------------------------
    # Internal metrics accumulation
    # ------------------------------------------------------------------

    def _accumulate(self, event: Dict[str, Any]) -> None:
        if event["status"] == "ok":
            self._total_transmitted_parameters += event.get("num_parameters") or 0
            self._total_transmitted_bytes += event["payload_bytes"]

    # ------------------------------------------------------------------
    # TASK 5 -- Metrics summary
    # ------------------------------------------------------------------

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Aggregate running communication metrics across every event
        recorded so far (Task 5).
        """
        events = self.simulator.events
        ok_events = [e for e in events if e["status"] == "ok"]
        latencies = [e["latency_seconds"] for e in ok_events]
        return {
            "total_events": len(events),
            "delivered_events": len(ok_events),
            "dropped_events": sum(1 for e in events if e["status"] == "dropped"),
            "delayed_events": sum(1 for e in events if e["status"] == "delayed"),
            "disconnected_events": sum(1 for e in events if e["status"] == "disconnected"),
            "communication_rounds": self._communication_rounds,
            "synchronization_interval": self.synchronization_interval,
            "transmitted_parameters": self._total_transmitted_parameters,
            "transmitted_bytes": self._total_transmitted_bytes,
            "avg_latency_seconds": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "max_latency_seconds": max(latencies) if latencies else 0.0,
            # Phase 7.5 addition (Task 7 requires min latency in the
            # report alongside avg/max; the underlying event data was
            # always there, this just also reduces it with min()).
            "min_latency_seconds": min(latencies) if latencies else 0.0,
            "avg_bandwidth_mbps": self.simulator.bandwidth_config.bandwidth_mbps,
            "total_sync_duration_seconds": sum(
                r["total_sync_duration_seconds"] for r in self.round_records
            ),
        }

    # ------------------------------------------------------------------
    # TASK 10 -- Artifact export
    # ------------------------------------------------------------------

    def export(
        self,
        output_dir: Union[str, Path],
        system_monitor: Optional[Any] = None,
        plots_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Write every Task 10 artifact for this run:
        ``network_metrics.csv``, ``communication_log.csv``,
        ``network_summary.json``, ``network_config.json`` (plus
        ``system_metrics.csv`` if ``system_monitor`` is given), and
        generate the Task 8 visualizations.

        Args:
            output_dir: Root directory for artifacts (typically
                ``artifacts/network/``).
            system_monitor: Optional ``monitoring.system_monitor.SystemMonitor``
                whose time-series is exported alongside and whose
                summary is merged into ``network_summary.json``.
            plots_dir: Directory for Task 8 visualizations. Defaults
                to ``analysis/network_plots/``.

        Returns:
            Dict mapping artifact name to written file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}

        # communication_log.csv -- raw per-event log
        log_path = output_dir / "communication_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_COMMUNICATION_LOG_FIELDS)
            writer.writeheader()
            for event in self.simulator.events:
                writer.writerow({k: event.get(k) for k in _COMMUNICATION_LOG_FIELDS})
        paths["communication_log"] = log_path

        # network_metrics.csv -- per-round aggregate
        metrics_path = output_dir / "network_metrics.csv"
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_NETWORK_METRICS_FIELDS)
            writer.writeheader()
            for i, record in enumerate(self.round_records, start=1):
                row = {
                    "round": record["round"],
                    "communication_rounds_so_far": i,
                    "synchronization_interval": self.synchronization_interval,
                    "participants": len(record["participants"]),
                    "transmitted_parameters": record["transmitted_parameters"],
                    "transmitted_bytes": record["transmitted_bytes"],
                    "avg_latency_seconds": None,
                    "max_latency_seconds": None,
                    "bandwidth_mbps": self.simulator.bandwidth_config.bandwidth_mbps,
                    "serialization_time_seconds": self.simulator.bandwidth_config.serialization_overhead_ms / 1000.0,
                    "deserialization_time_seconds": self.simulator.bandwidth_config.deserialization_overhead_ms / 1000.0,
                    "transfer_time_seconds": record["upload_time_seconds"] + record["broadcast_time_seconds"],
                    "total_sync_duration_seconds": record["total_sync_duration_seconds"],
                    "dropped_events": len(record["dropped_workers"]),
                    "delayed_events": None,
                    "disconnected_events": record["disconnected_events"],
                }
                writer.writerow(row)
        paths["network_metrics"] = metrics_path

        # network_summary.json
        summary = {
            "metrics": self.get_metrics_summary(),
            "round_records": self.round_records,
        }
        if system_monitor is not None:
            summary["system_monitor"] = system_monitor.get_summary()
        summary_path = output_dir / "network_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        paths["network_summary"] = summary_path

        # network_config.json
        config_path = output_dir / "network_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.simulator.to_config_dict(), f, indent=2)
        paths["network_config"] = config_path

        if system_monitor is not None:
            paths["system_metrics"] = system_monitor.export(output_dir)

        # Task 8 -- visualizations
        plots_dir = Path(plots_dir) if plots_dir is not None else (
            Path(__file__).resolve().parent.parent / "analysis" / "network_plots"
        )
        try:
            plot_paths = self.simulator.generate_visualizations(plots_dir)
            paths.update({f"plot_{k}": v for k, v in plot_paths.items()})
        except Exception:
            logger.exception("Network visualization generation failed; artifacts are still saved")

        # Phase 7.5, Task 5 -- system-resource plots (cpu_ram_usage.png,
        # gpu_memory_usage.png, system_resource_usage.png) live alongside
        # the network plots. Reuses SystemMonitor.generate_visualizations
        # (Task 5's "skip GPU plots gracefully if unavailable" is handled
        # there) rather than duplicating plotting logic here.
        if system_monitor is not None:
            try:
                sys_plot_paths = system_monitor.generate_visualizations(plots_dir)
                paths.update({f"plot_{k}": v for k, v in sys_plot_paths.items()})
            except Exception:
                logger.exception("System monitor visualization generation failed; artifacts are still saved")

        self._validate_export(paths)
        logger.info("Network artifacts exported to %s (%d files)", output_dir, len(paths))
        return paths

    # ------------------------------------------------------------------
    # TASK 11 -- Validation
    # ------------------------------------------------------------------

    def _validate_export(self, paths: Dict[str, Path]) -> None:
        """Confirm every required Task 10 artifact was actually written.

        Raises:
            RuntimeError: If a required artifact is missing on disk.
        """
        required = ["communication_log", "network_metrics", "network_summary", "network_config"]
        missing = [name for name in required if name not in paths or not Path(paths[name]).exists()]
        if missing:
            raise RuntimeError(f"CommunicationManager.export() failed to produce required artifact(s): {missing}")
