"""
network_simulator.py

Purpose:
    Simulate realistic communication behaviour between distributed data
    centers on top of a ``network.topology.Topology``: latency,
    bandwidth/transfer-time estimation, optional failure injection, and
    per-event metric recording. Everything here is computed, not
    measured -- there is no real networking, no ``time.sleep()`` by
    default, and no dependency on Phase 0-6 ML code beyond duck-typed
    "how many bytes/parameters am I sending" numbers supplied by the
    caller (e.g. ``network.communication.CommunicationManager``, used
    by future phases wrapping ``training.local_sgd.LocalSGDTrainer``).

Determinism (Task 12):
    Every source of randomness (latency sampling, failure injection)
    is drawn from a ``numpy.random.RandomState`` seeded once at
    construction time -- never the shared global RNG -- so identical
    seeds reproduce identical simulated timings and failure decisions
    regardless of call order elsewhere in the process.

Public Interface:
    class LatencyConfig / BandwidthConfig / FailureConfig (dataclasses)

    class NetworkSimulator
        Methods:
            simulate_latency(src, dst) -> float
            simulate_bandwidth(src=None, dst=None) -> float
            estimate_transfer_time(message_size_bytes, src=None, dst=None) -> float
            estimate_sync_time(message_size_bytes, participant_ids=None, coordinator=None) -> dict
            record_event(...) -> dict
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from network.topology import Topology

logger = logging.getLogger(__name__)

try:  # Reuse the project's single seeding utility when it's importable
    # (Task 12). training.utils imports models.fraud_mlp, which requires
    # torch -- the network layer itself must not hard-require torch, so
    # this is best-effort: a NetworkSimulator seeded explicitly (the
    # normal usage pattern -- see `seed=` below) is fully deterministic
    # on its own RandomState either way.
    from training.utils import set_seed as _project_set_seed
except Exception:  # pragma: no cover - torch not installed / not on path
    _project_set_seed = None


_MS_PER_S = 1000.0
_BYTES_PER_MB = 1_000_000  # decimal MB, matching typical "MB/s" bandwidth ratings


# ----------------------------------------------------------------------
# TASK 2 -- Latency model
# ----------------------------------------------------------------------


@dataclass
class LatencyConfig:
    """Configuration for inter-worker latency (Task 2).

    Attributes:
        mode: One of ``{"constant", "uniform", "gaussian", "matrix"}``.
        constant_ms: Latency (milliseconds) used when ``mode == "constant"``.
        uniform_low_ms / uniform_high_ms: Bounds for ``mode == "uniform"``.
        gaussian_mean_ms / gaussian_std_ms: Parameters for ``mode == "gaussian"``
            (negative samples are clamped to 0).
        matrix: Required iff ``mode == "matrix"``: a user-defined,
            symmetric, zero-diagonal ``(n, n)`` latency matrix in
            milliseconds.
        seed: RNG seed for sampled modes (uniform/gaussian). Ignored
            for constant/matrix, which are already deterministic.
    """

    mode: str = "constant"
    constant_ms: float = 50.0
    uniform_low_ms: float = 10.0
    uniform_high_ms: float = 100.0
    gaussian_mean_ms: float = 50.0
    gaussian_std_ms: float = 10.0
    matrix: Optional[List[List[float]]] = None
    seed: int = 42

    _MODES = ("constant", "uniform", "gaussian", "matrix")

    def __post_init__(self) -> None:
        if self.mode not in self._MODES:
            raise ValueError(f"latency mode must be one of {self._MODES}, got {self.mode!r}")
        if self.mode == "matrix" and self.matrix is None:
            raise ValueError("latency mode 'matrix' requires 'matrix' to be provided")
        if self.mode == "uniform" and self.uniform_low_ms > self.uniform_high_ms:
            raise ValueError(
                f"uniform_low_ms ({self.uniform_low_ms}) must be <= uniform_high_ms ({self.uniform_high_ms})"
            )
        if self.constant_ms < 0 or self.uniform_low_ms < 0 or self.gaussian_mean_ms < 0:
            raise ValueError("latency values must be non-negative")


# ----------------------------------------------------------------------
# TASK 3 -- Bandwidth model
# ----------------------------------------------------------------------


@dataclass
class BandwidthConfig:
    """Configuration for link bandwidth and (de)serialization overhead
    (Task 3).

    Attributes:
        bandwidth_mbps: Effective link bandwidth in MB/s (decimal
            megabytes/second). Applied uniformly to every link unless
            ``matrix`` is given.
        matrix: Optional per-link bandwidth override, ``(n, n)`` MB/s,
            symmetric, zero diagonal.
        serialization_overhead_ms: Fixed cost paid by the sender to
            serialize a message, independent of size.
        deserialization_overhead_ms: Fixed cost paid by the receiver to
            deserialize a message, independent of size.
    """

    bandwidth_mbps: float = 100.0
    matrix: Optional[List[List[float]]] = None
    serialization_overhead_ms: float = 5.0
    deserialization_overhead_ms: float = 5.0

    def __post_init__(self) -> None:
        if self.bandwidth_mbps <= 0:
            raise ValueError(f"bandwidth_mbps must be positive, got {self.bandwidth_mbps!r}")
        if self.serialization_overhead_ms < 0 or self.deserialization_overhead_ms < 0:
            raise ValueError("serialization/deserialization overhead must be non-negative")


# ----------------------------------------------------------------------
# TASK 6 -- Failure model
# ----------------------------------------------------------------------


@dataclass
class FailureConfig:
    """Configuration for optional failure simulation (Task 6).
    Disabled by default; every probability is independently applied
    per simulated message.

    Attributes:
        enabled: Master switch. If ``False``, every other field is
            ignored and no failure is ever injected.
        packet_loss_prob: Probability a message is lost in transit
            (recorded as ``status="dropped"``, contributes 0 bytes to
            transmitted totals).
        delayed_sync_prob: Probability a message is delayed (its
            latency is multiplied by ``delay_multiplier``) rather than
            lost outright.
        delay_multiplier: Multiplier applied to latency when a message
            is delayed.
        dropped_sync_prob: Probability an entire synchronization round
            is dropped for a given worker (that worker sits the round
            out, as if it never attempted to sync).
        disconnected_workers: Worker ids that are disconnected for a
            configurable round window.
        disconnect_start_round / disconnect_end_round: Inclusive round
            range (0-indexed) during which ``disconnected_workers`` are
            unreachable. ``None`` for either bound means "from the
            start" / "through the end". A worker's temporary recovery
            is simply this window ending before training does.
        seed: RNG seed for failure sampling.
    """

    enabled: bool = False
    packet_loss_prob: float = 0.0
    delayed_sync_prob: float = 0.0
    delay_multiplier: float = 3.0
    dropped_sync_prob: float = 0.0
    disconnected_workers: List[int] = field(default_factory=list)
    disconnect_start_round: Optional[int] = None
    disconnect_end_round: Optional[int] = None
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("packet_loss_prob", "delayed_sync_prob", "dropped_sync_prob"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if self.delay_multiplier <= 0:
            raise ValueError(f"delay_multiplier must be positive, got {self.delay_multiplier!r}")


class NetworkSimulator:
    """Ties a ``Topology`` together with latency/bandwidth/failure
    models to simulate the timing and outcome of individual
    communication events (Task 2/3/4/5/6).

    Args:
        topology: A ``Topology`` instance. ``build()`` is called on it
            automatically if not already built.
        latency_config: Optional ``LatencyConfig``. Defaults to a
            constant-50ms model.
        bandwidth_config: Optional ``BandwidthConfig``.
        failure_config: Optional ``FailureConfig`` (disabled by
            default -- Task 6).
        simulate_real_time: If ``True``, ``simulate_latency`` actually
            sleeps for the simulated duration (useful for demos /
            wall-clock-sensitive downstream tooling). Defaults to
            ``False`` (Task 2): only simulated communication time is
            computed, nothing actually blocks.
        seed: Top-level seed. Used to seed this simulator's own
            ``RandomState`` and, when not explicitly overridden on the
            sub-configs, propagated to ``latency_config.seed`` and
            ``failure_config.seed`` for a single source of truth
            (Task 12).

    Raises:
        ValueError: If a latency/bandwidth matrix's shape does not
            match the topology's worker count.
    """

    def __init__(
        self,
        topology: Topology,
        latency_config: Optional[LatencyConfig] = None,
        bandwidth_config: Optional[BandwidthConfig] = None,
        failure_config: Optional[FailureConfig] = None,
        simulate_real_time: bool = False,
        seed: int = 42,
    ) -> None:
        self.topology = topology
        if self.topology.adjacency is None:
            self.topology.build()

        self.seed = seed
        self.latency_config = latency_config or LatencyConfig(seed=seed)
        self.bandwidth_config = bandwidth_config or BandwidthConfig()
        self.failure_config = failure_config or FailureConfig(seed=seed)
        self.simulate_real_time = simulate_real_time

        if _project_set_seed is not None:
            try:
                _project_set_seed(seed)
            except Exception:  # pragma: no cover - defensive; never fatal for the network layer
                logger.debug("training.utils.set_seed(%d) unavailable/failed; continuing", seed)

        self._latency_rng = np.random.RandomState(self.latency_config.seed)
        self._failure_rng = np.random.RandomState(self.failure_config.seed)

        self._latency_matrix_ms = self._resolve_latency_matrix()
        self._bandwidth_matrix_mbps = self._resolve_bandwidth_matrix()

        self.events: List[Dict[str, Any]] = []
        self._event_counter = 0
        self._round_index = 0
        self._sim_clock = 0.0  # cumulative simulated seconds, for event timestamps

    # ------------------------------------------------------------------
    # TASK 2 -- Latency
    # ------------------------------------------------------------------

    def _resolve_latency_matrix(self) -> np.ndarray:
        """Precompute a deterministic ``(n, n)`` latency matrix in
        milliseconds from ``self.latency_config`` (Task 2/12).

        Raises:
            ValueError: If a user-supplied matrix's shape does not
                match the topology's worker count.
        """
        n = self.topology.num_workers
        cfg = self.latency_config

        if cfg.mode == "constant":
            matrix = np.full((n, n), cfg.constant_ms, dtype=float)
        elif cfg.mode == "uniform":
            matrix = self._latency_rng.uniform(cfg.uniform_low_ms, cfg.uniform_high_ms, size=(n, n))
        elif cfg.mode == "gaussian":
            matrix = self._latency_rng.normal(cfg.gaussian_mean_ms, cfg.gaussian_std_ms, size=(n, n))
            matrix = np.clip(matrix, 0.0, None)
        else:  # matrix
            matrix = np.array(cfg.matrix, dtype=float)
            if matrix.shape != (n, n):
                raise ValueError(f"latency matrix shape {matrix.shape} does not match ({n}, {n})")

        matrix = (matrix + matrix.T) / 2.0  # symmetrize sampled modes
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def simulate_latency(self, src: int, dst: int) -> float:
        """Return the simulated one-way latency between ``src`` and
        ``dst`` in seconds, following the shortest path on the
        topology (so STAR/TREE/RING correctly accumulate multi-hop
        latency rather than assuming a direct link).

        If ``simulate_real_time`` is ``True``, this method also sleeps
        for that duration.

        Args:
            src: Source worker id.
            dst: Destination worker id.

        Returns:
            Latency in seconds.
        """
        self.topology.validate_worker_id(src)
        self.topology.validate_worker_id(dst)
        path = self.topology.shortest_path(src, dst)
        total_ms = sum(
            self._latency_matrix_ms[path[i], path[i + 1]] for i in range(len(path) - 1)
        )
        latency_s = total_ms / _MS_PER_S
        if self.simulate_real_time and latency_s > 0:
            time.sleep(latency_s)
        return latency_s

    # ------------------------------------------------------------------
    # TASK 3 -- Bandwidth / transfer time
    # ------------------------------------------------------------------

    def _resolve_bandwidth_matrix(self) -> np.ndarray:
        """Precompute a ``(n, n)`` bandwidth matrix in MB/s.

        Raises:
            ValueError: If a user-supplied matrix's shape does not
                match the topology's worker count.
        """
        n = self.topology.num_workers
        cfg = self.bandwidth_config
        if cfg.matrix is None:
            matrix = np.full((n, n), cfg.bandwidth_mbps, dtype=float)
        else:
            matrix = np.array(cfg.matrix, dtype=float)
            if matrix.shape != (n, n):
                raise ValueError(f"bandwidth matrix shape {matrix.shape} does not match ({n}, {n})")
        np.fill_diagonal(matrix, cfg.bandwidth_mbps)
        return matrix

    def simulate_bandwidth(self, src: Optional[int] = None, dst: Optional[int] = None) -> float:
        """Return the simulated link bandwidth in MB/s between ``src``
        and ``dst`` (or the configured default if either is omitted).
        """
        if src is None or dst is None:
            return float(self.bandwidth_config.bandwidth_mbps)
        self.topology.validate_worker_id(src)
        self.topology.validate_worker_id(dst)
        return float(self._bandwidth_matrix_mbps[src, dst])

    def estimate_transfer_time(
        self, message_size_bytes: float, src: Optional[int] = None, dst: Optional[int] = None
    ) -> float:
        """Estimate the time to transfer ``message_size_bytes`` bytes,
        including serialization + deserialization overhead (Task 3).
        Does not include network latency -- see ``estimate_sync_time``
        for the combined figure.

        Args:
            message_size_bytes: Payload size in bytes.
            src: Optional source worker id (for per-link bandwidth).
            dst: Optional destination worker id.

        Returns:
            Estimated transfer time in seconds.

        Raises:
            ValueError: If ``message_size_bytes`` is negative.
        """
        if message_size_bytes < 0:
            raise ValueError(f"message_size_bytes must be non-negative, got {message_size_bytes!r}")
        bandwidth_mbps = self.simulate_bandwidth(src, dst)
        raw_transfer_s = message_size_bytes / _BYTES_PER_MB / bandwidth_mbps
        overhead_s = (
            self.bandwidth_config.serialization_overhead_ms
            + self.bandwidth_config.deserialization_overhead_ms
        ) / _MS_PER_S
        return raw_transfer_s + overhead_s

    # ------------------------------------------------------------------
    # TASK 3/4 -- Combined synchronization time estimate
    # ------------------------------------------------------------------

    def estimate_sync_time(
        self,
        message_size_bytes: float,
        participant_ids: Optional[Sequence[int]] = None,
        coordinator: Optional[int] = None,
    ) -> Dict[str, float]:
        """Estimate the total wall-clock time for one synchronization
        round: every participant uploads ``message_size_bytes`` to the
        coordinator (in parallel), then the coordinator broadcasts the
        aggregated result back out (in parallel). Because uploads
        happen concurrently, the round is bottlenecked by the slowest
        participant in each phase (Task 3/4).

        Args:
            message_size_bytes: Size of one worker's parameter payload,
                in bytes.
            participant_ids: Workers taking part. Defaults to every
                worker except ``coordinator``.
            coordinator: Aggregation point. Defaults to
                ``self.topology.coordinator_index``.

        Returns:
            Dict with ``upload_time_seconds``, ``broadcast_time_seconds``,
            ``total_sync_time_seconds``, and ``communication_time_seconds``
            (== total_sync_time_seconds; kept as an explicit alias
            since Task 3 names both quantities separately).
        """
        coordinator = coordinator if coordinator is not None else self.topology.coordinator_index
        self.topology.validate_worker_id(coordinator)
        participants = (
            list(participant_ids)
            if participant_ids is not None
            else [w for w in range(self.topology.num_workers) if w != coordinator]
        )

        upload_times = []
        broadcast_times = []
        for worker in participants:
            latency_s = self.simulate_latency(worker, coordinator)
            transfer_s = self.estimate_transfer_time(message_size_bytes, worker, coordinator)
            upload_times.append(latency_s + transfer_s)
            broadcast_times.append(latency_s + transfer_s)  # symmetric link assumption

        upload_time = max(upload_times) if upload_times else 0.0
        broadcast_time = max(broadcast_times) if broadcast_times else 0.0
        total = upload_time + broadcast_time

        return {
            "upload_time_seconds": upload_time,
            "broadcast_time_seconds": broadcast_time,
            "total_sync_time_seconds": total,
            "communication_time_seconds": total,
        }

    # ------------------------------------------------------------------
    # TASK 6 -- Failure simulation
    # ------------------------------------------------------------------

    def is_worker_disconnected(self, worker_id: int, round_index: Optional[int] = None) -> bool:
        """Return whether ``worker_id`` is disconnected during
        ``round_index`` (defaults to the simulator's current round)
        per ``self.failure_config``.
        """
        cfg = self.failure_config
        if not cfg.enabled or worker_id not in cfg.disconnected_workers:
            return False
        r = round_index if round_index is not None else self._round_index
        if cfg.disconnect_start_round is not None and r < cfg.disconnect_start_round:
            return False
        if cfg.disconnect_end_round is not None and r > cfg.disconnect_end_round:
            return False
        return True

    def apply_failure(self, src: int, dst: int) -> Dict[str, Any]:
        """Sample whether a message from ``src`` to ``dst`` is dropped
        or delayed, given ``self.failure_config`` (Task 6). Disabled
        (returns "ok") unless ``failure_config.enabled`` is ``True``.

        Returns:
            Dict with ``status`` (one of ``"ok"``, ``"disconnected"``,
            ``"dropped"``, ``"delayed"``) and ``delay_multiplier``
            (``1.0`` unless ``status == "delayed"``).
        """
        cfg = self.failure_config
        if not cfg.enabled:
            return {"status": "ok", "delay_multiplier": 1.0}

        if self.is_worker_disconnected(src) or self.is_worker_disconnected(dst):
            return {"status": "disconnected", "delay_multiplier": 1.0}
        if self._failure_rng.random_sample() < cfg.packet_loss_prob:
            return {"status": "dropped", "delay_multiplier": 1.0}
        if self._failure_rng.random_sample() < cfg.delayed_sync_prob:
            return {"status": "delayed", "delay_multiplier": cfg.delay_multiplier}
        return {"status": "ok", "delay_multiplier": 1.0}

    def sync_round_dropped(self, worker_id: int) -> bool:
        """Sample whether ``worker_id`` should sit out this
        synchronization round entirely (Task 6 -- "dropped
        synchronization"). Disabled unless ``failure_config.enabled``.
        """
        cfg = self.failure_config
        if not cfg.enabled:
            return False
        return bool(self._failure_rng.random_sample() < cfg.dropped_sync_prob)

    def next_round(self) -> int:
        """Advance and return the simulator's internal round counter
        (used for disconnect-window scheduling and event metadata).
        """
        self._round_index += 1
        return self._round_index

    # ------------------------------------------------------------------
    # TASK 5 -- Event recording
    # ------------------------------------------------------------------

    def record_event(
        self,
        event_type: str,
        src: Any,
        dst: Any,
        payload_bytes: float,
        latency_seconds: float,
        transfer_time_seconds: float,
        num_parameters: Optional[int] = None,
        status: str = "ok",
        round_index: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record one communication event's metadata (Task 4/5). Every
        ``send``/``broadcast``/``synchronize`` call in
        ``network.communication.CommunicationManager`` routes through
        here so every event is captured uniformly.

        Returns:
            The recorded event dict (also appended to ``self.events``).
        """
        self._event_counter += 1
        total_time = latency_seconds + transfer_time_seconds
        self._sim_clock += total_time
        event = {
            "event_id": self._event_counter,
            "event_type": event_type,
            "round": round_index if round_index is not None else self._round_index,
            "src": src,
            "dst": dst,
            "status": status,
            "num_parameters": num_parameters,
            "payload_bytes": payload_bytes,
            "latency_seconds": latency_seconds,
            "bandwidth_mbps": self.simulate_bandwidth(
                src if isinstance(src, int) else None, dst if isinstance(dst, int) else None
            ),
            "serialization_time_seconds": self.bandwidth_config.serialization_overhead_ms / _MS_PER_S,
            "deserialization_time_seconds": self.bandwidth_config.deserialization_overhead_ms / _MS_PER_S,
            "transfer_time_seconds": transfer_time_seconds,
            "total_time_seconds": total_time,
            "sim_timestamp_seconds": self._sim_clock,
        }
        if extra:
            event.update(extra)
        self.events.append(event)
        return event

    # ------------------------------------------------------------------
    # Reporting / export helpers
    # ------------------------------------------------------------------

    def to_config_dict(self) -> Dict[str, Any]:
        """Return the fully resolved configuration for this simulator
        (Task 9/10/12 -- what ``network_config.json`` is built from).
        """
        return {
            "seed": self.seed,
            "simulate_real_time": self.simulate_real_time,
            "topology": self.topology.to_dict(),
            "latency": asdict(self.latency_config),
            "bandwidth": asdict(self.bandwidth_config),
            "failure": asdict(self.failure_config),
        }

    def events_dataframe(self):
        """Return ``self.events`` as a ``pandas.DataFrame`` (imported
        lazily so this module has no hard pandas dependency at import
        time beyond what the rest of the project already requires).
        """
        import pandas as pd

        return pd.DataFrame(self.events)

    # ------------------------------------------------------------------
    # TASK 8 -- Visualizations
    # ------------------------------------------------------------------

    def visualize_latency_timeline(self, output_path: Optional[Union[str, Path]] = None):
        """Plot latency over communication rounds/events (Task 8)."""
        return self._timeline_plot(
            "latency_seconds", "Latency (s)", "Latency Timeline", output_path,
        )

    def visualize_bandwidth_utilization(self, output_path: Optional[Union[str, Path]] = None):
        """Plot per-event effective bandwidth utilization: payload size
        divided by transfer time, over the event sequence (Task 8).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.events:
            logger.warning("No events recorded yet; bandwidth utilization plot will be empty")
        xs = [e["event_id"] for e in self.events]
        ys = [
            (e["payload_bytes"] / _BYTES_PER_MB) / e["transfer_time_seconds"]
            if e["transfer_time_seconds"] > 0 else 0.0
            for e in self.events
        ]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, ys, marker="o", markersize=3, color="seagreen")
        ax.set_xlabel("Event #")
        ax.set_ylabel("Effective bandwidth (MB/s)")
        ax.set_title("Bandwidth Utilization")
        fig.tight_layout()
        return self._finish_plot(fig, output_path)

    def visualize_sync_duration(self, output_path: Optional[Union[str, Path]] = None):
        """Plot synchronize()-event total duration per round (Task 8)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sync_events = [e for e in self.events if e["event_type"] == "synchronize"]
        xs = [e["round"] for e in sync_events]
        ys = [e["total_time_seconds"] for e in sync_events]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, ys, marker="o", color="darkorange")
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Synchronization duration (s)")
        ax.set_title("Synchronization Duration per Round")
        fig.tight_layout()
        return self._finish_plot(fig, output_path)

    def visualize_communication_cost(self, output_path: Optional[Union[str, Path]] = None):
        """Plot cumulative transmitted bytes over communication rounds
        (Task 8).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rounds: Dict[int, float] = {}
        for e in self.events:
            if e["status"] == "ok":
                rounds[e["round"]] = rounds.get(e["round"], 0.0) + e["payload_bytes"]
        xs = sorted(rounds.keys())
        cumulative = np.cumsum([rounds[r] for r in xs]) if xs else []

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, cumulative, marker="o", color="crimson")
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Cumulative bytes transmitted")
        ax.set_title("Communication Cost over Time")
        fig.tight_layout()
        return self._finish_plot(fig, output_path)

    def _timeline_plot(self, field_name: str, ylabel: str, title: str, output_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [e["event_id"] for e in self.events]
        ys = [e[field_name] for e in self.events]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(xs, ys, marker="o", markersize=3, color="steelblue")
        ax.set_xlabel("Event #")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
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
        """Generate every Task 8 plot (topology graph + the four
        event-derived plots above) into ``output_dir`` (typically
        ``analysis/network_plots/``).

        Returns:
            Dict mapping plot name to the written file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "topology": output_dir / "network_topology.png",
            "latency_timeline": output_dir / "latency_timeline.png",
            "bandwidth_utilization": output_dir / "bandwidth_utilization.png",
            "sync_duration": output_dir / "sync_duration.png",
            "communication_cost": output_dir / "communication_cost.png",
        }
        self.topology.visualize(paths["topology"])
        self.visualize_latency_timeline(paths["latency_timeline"])
        self.visualize_bandwidth_utilization(paths["bandwidth_utilization"])
        self.visualize_sync_duration(paths["sync_duration"])
        self.visualize_communication_cost(paths["communication_cost"])
        logger.info("Generated %d network visualization(s) in %s", len(paths), output_dir)
        return paths

    # ------------------------------------------------------------------
    # TASK 9 -- Configuration loading
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Union[str, Path, Dict[str, Any]]) -> "NetworkSimulator":
        """Build a fully wired ``NetworkSimulator`` (topology built,
        configs validated) from a config dict or a path to a
        ``config/network/*.yaml`` file (Task 9).

        Raises:
            ValueError: If required top-level keys are missing.
        """
        if isinstance(config, (str, Path)):
            config = load_network_config(config)
        config = dict(config)
        for key in ("topology",):
            if key not in config:
                raise ValueError(f"network config is missing required section {key!r}")

        topology = Topology.from_config(config["topology"])
        topology.build()

        top_seed = config.get("seed", 42)
        latency_cfg = LatencyConfig(**{**{"seed": top_seed}, **config.get("latency", {})})
        bandwidth_cfg = BandwidthConfig(**config.get("bandwidth", {}))
        failure_cfg = FailureConfig(**{**{"seed": top_seed}, **config.get("failure", {})})

        return cls(
            topology=topology,
            latency_config=latency_cfg,
            bandwidth_config=bandwidth_cfg,
            failure_config=failure_cfg,
            simulate_real_time=config.get("simulate_real_time", False),
            seed=top_seed,
        )


def load_network_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a ``config/network/*.yaml`` file into a plain dict (Task 9).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not parse to a mapping.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Network config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return config
