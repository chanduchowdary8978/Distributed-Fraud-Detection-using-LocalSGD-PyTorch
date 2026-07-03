"""
topology.py

Purpose:
    Implement configurable logical network topologies for the
    Distributed System Simulation Layer (Phase 7). A topology defines
    which pairs of simulated data centers ("workers") may communicate
    directly, as a plain adjacency matrix. Nothing in this module
    touches machine-learning logic, sockets, or real networking --
    everything here is a logical, in-process description of who is
    allowed to talk to whom.

Scope:
    Phase 7 only. No RPC/sockets/gRPC/MPI/real networking of any kind.
    This module is pure data + validation + (optional) visualization.

Public Interface:
    class TopologyType(str, Enum)

    class Topology
        Methods:
            build() -> np.ndarray
            neighbors(worker_id) -> List[int]
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


class TopologyType(str, Enum):
    """Supported logical network topologies (Task 1)."""

    FULLY_CONNECTED = "fully_connected"
    RING = "ring"
    STAR = "star"
    TREE = "tree"
    CUSTOM = "custom"


class Topology:
    """A logical network topology over ``num_workers`` simulated data
    centers.

    Each worker represents one logical data center. The topology
    determines which worker pairs may communicate directly (an edge in
    the adjacency matrix), which in turn determines communication
    paths used by ``network.network_simulator.NetworkSimulator`` and
    ``network.communication.CommunicationManager``.

    Args:
        num_workers: Number of simulated data centers. Must be >= 2
            (>= 1 is technically buildable but a single-node "network"
            has no communication to simulate, so 2 is the practical
            floor for every topology except CUSTOM with a
            pre-validated single-node matrix).
        topology_type: One of the values in ``TopologyType`` (or the
            equivalent string).
        custom_adjacency: Required iff ``topology_type == "custom"``.
            A square, symmetric, zero-diagonal 0/1 matrix (list of
            lists or ``np.ndarray``) of shape ``(num_workers,
            num_workers)``.
        coordinator_index: Index of the hub/coordinator worker, used
            only by the STAR topology (and as the default
            synchronization coordinator elsewhere). Defaults to ``0``.
        branching_factor: Number of children per internal node, used
            only by the TREE topology. Defaults to ``2`` (binary tree).
        worker_names: Optional human-readable names, index-aligned with
            worker ids ``0..num_workers-1``. Defaults to
            ``["worker_0", "worker_1", ...]``.

    Raises:
        ValueError: If ``num_workers < 2``, if ``topology_type`` is not
            supported, or if CUSTOM is chosen without a valid
            ``custom_adjacency``.
    """

    def __init__(
        self,
        num_workers: int,
        topology_type: Union[str, TopologyType] = TopologyType.FULLY_CONNECTED,
        custom_adjacency: Optional[Union[Sequence[Sequence[int]], np.ndarray]] = None,
        coordinator_index: int = 0,
        branching_factor: int = 2,
        worker_names: Optional[Sequence[str]] = None,
    ) -> None:
        if not isinstance(num_workers, int) or num_workers < 2:
            raise ValueError(f"num_workers must be an integer >= 2, got {num_workers!r}")

        self.topology_type = TopologyType(topology_type)
        self.num_workers = num_workers
        self.custom_adjacency = custom_adjacency
        self.branching_factor = branching_factor

        if not (0 <= coordinator_index < num_workers):
            raise ValueError(
                f"coordinator_index must be in [0, {num_workers}), got {coordinator_index!r}"
            )
        self.coordinator_index = coordinator_index

        if self.topology_type == TopologyType.TREE and branching_factor < 1:
            raise ValueError(f"branching_factor must be >= 1, got {branching_factor!r}")

        if self.topology_type == TopologyType.CUSTOM and custom_adjacency is None:
            raise ValueError("custom_adjacency is required when topology_type='custom'")

        if worker_names is not None:
            if len(worker_names) != num_workers:
                raise ValueError(
                    f"worker_names has {len(worker_names)} entries, expected {num_workers}"
                )
            self.worker_names = list(worker_names)
        else:
            self.worker_names = [f"worker_{i}" for i in range(num_workers)]

        self.adjacency: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # TASK 1 -- Topology construction
    # ------------------------------------------------------------------

    def build(self) -> np.ndarray:
        """Construct the adjacency matrix for the configured topology.

        Returns:
            A ``(num_workers, num_workers)`` symmetric, zero-diagonal
            0/1 ``np.ndarray`` (``dtype=int``). Stored on
            ``self.adjacency`` and also returned.

        Raises:
            ValueError: If the resulting adjacency matrix fails
                validation (wrong shape, asymmetric, self-loops, or
                disconnected graph).
        """
        n = self.num_workers
        if self.topology_type == TopologyType.FULLY_CONNECTED:
            adjacency = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)

        elif self.topology_type == TopologyType.RING:
            adjacency = np.zeros((n, n), dtype=int)
            for i in range(n):
                adjacency[i, (i + 1) % n] = 1
                adjacency[(i + 1) % n, i] = 1
            if n == 2:
                # A 2-node "ring" degenerates to a single undirected edge;
                # the loop above already sets it exactly once each way.
                pass

        elif self.topology_type == TopologyType.STAR:
            adjacency = np.zeros((n, n), dtype=int)
            hub = self.coordinator_index
            for i in range(n):
                if i == hub:
                    continue
                adjacency[hub, i] = 1
                adjacency[i, hub] = 1

        elif self.topology_type == TopologyType.TREE:
            adjacency = np.zeros((n, n), dtype=int)
            b = self.branching_factor
            for i in range(1, n):
                parent = (i - 1) // b
                adjacency[parent, i] = 1
                adjacency[i, parent] = 1

        elif self.topology_type == TopologyType.CUSTOM:
            adjacency = np.array(self.custom_adjacency, dtype=int)

        else:  # pragma: no cover - unreachable given TopologyType(...) coercion
            raise ValueError(f"Unsupported topology_type: {self.topology_type!r}")

        self._validate_adjacency(adjacency)
        self.adjacency = adjacency
        logger.info(
            "Built %s topology over %d workers (%d edges)",
            self.topology_type.value, n, int(adjacency.sum()) // 2,
        )
        return adjacency

    # ------------------------------------------------------------------
    # TASK 11 -- Validation
    # ------------------------------------------------------------------

    def _validate_adjacency(self, adjacency: np.ndarray) -> None:
        """Validate an adjacency matrix (Task 11).

        Raises:
            ValueError: If the matrix is not square, not the expected
                size, not 0/1-valued, not symmetric, has a self-loop,
                or does not describe a connected graph.
        """
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Adjacency matrix must be square, got shape {adjacency.shape}")
        if adjacency.shape[0] != self.num_workers:
            raise ValueError(
                f"Adjacency matrix shape {adjacency.shape} does not match "
                f"num_workers={self.num_workers}"
            )
        if not np.array_equal(adjacency, adjacency.T):
            raise ValueError("Adjacency matrix must be symmetric (communication is bidirectional)")
        if np.any(np.diag(adjacency) != 0):
            raise ValueError("Adjacency matrix must have a zero diagonal (no self-loops)")
        unique_vals = set(np.unique(adjacency).tolist())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(f"Adjacency matrix must be 0/1-valued, got values {sorted(unique_vals)}")
        if not self._is_connected(adjacency):
            raise ValueError(
                "Topology is not fully connected: at least one worker cannot reach the "
                "others through any path. Every worker must be reachable, directly or "
                "indirectly, for synchronization to be possible."
            )

    @staticmethod
    def _is_connected(adjacency: np.ndarray) -> bool:
        n = adjacency.shape[0]
        visited = {0}
        frontier = [0]
        while frontier:
            node = frontier.pop()
            neighbors = np.nonzero(adjacency[node])[0]
            for nb in neighbors:
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    frontier.append(nb)
        return len(visited) == n

    def validate_worker_id(self, worker_id: int) -> None:
        """Raise if ``worker_id`` is out of range for this topology.

        Raises:
            ValueError: If ``worker_id`` is not in ``[0, num_workers)``.
        """
        if not (0 <= worker_id < self.num_workers):
            raise ValueError(
                f"worker_id {worker_id!r} out of range for {self.num_workers} workers"
            )

    # ------------------------------------------------------------------
    # TASK 1 -- Public query API
    # ------------------------------------------------------------------

    def neighbors(self, worker_id: int) -> List[int]:
        """Return the ids of every worker directly reachable from
        ``worker_id`` under this topology.

        Args:
            worker_id: Index of the worker to query.

        Returns:
            Sorted list of directly-connected worker ids.

        Raises:
            RuntimeError: If ``build()`` has not been called yet.
            ValueError: If ``worker_id`` is out of range.
        """
        if self.adjacency is None:
            raise RuntimeError("Topology.build() must be called before neighbors()")
        self.validate_worker_id(worker_id)
        return [int(j) for j in np.nonzero(self.adjacency[worker_id])[0]]

    def shortest_path(self, src: int, dst: int) -> List[int]:
        """Breadth-first shortest path from ``src`` to ``dst`` (used by
        the communication layer to route multi-hop messages on
        non-fully-connected topologies, e.g. STAR/TREE/RING).

        Args:
            src: Source worker id.
            dst: Destination worker id.

        Returns:
            List of worker ids from ``src`` to ``dst`` inclusive (a
            single-element list ``[src]`` if ``src == dst``).

        Raises:
            RuntimeError: If ``build()`` has not been called yet.
            ValueError: If ``src``/``dst`` are out of range.
        """
        if self.adjacency is None:
            raise RuntimeError("Topology.build() must be called before shortest_path()")
        self.validate_worker_id(src)
        self.validate_worker_id(dst)
        if src == dst:
            return [src]

        from collections import deque

        parent: Dict[int, int] = {}
        visited = {src}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for nb in np.nonzero(self.adjacency[node])[0]:
                nb = int(nb)
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = node
                    if nb == dst:
                        queue.clear()
                        break
                    queue.append(nb)

        if dst not in parent and dst != src:
            # Unreachable should be impossible post-validation, but guard anyway.
            raise RuntimeError(f"No path found from worker {src} to worker {dst}")

        path = [dst]
        while path[-1] != src:
            path.append(parent[path[-1]])
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # TASK 9/12 -- Configuration & reproducibility
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this topology's configuration (not the built
        adjacency matrix's dense NxN form, to keep this compact --
        ``adjacency`` is included separately when non-``None``).
        """
        return {
            "topology_type": self.topology_type.value,
            "num_workers": self.num_workers,
            "coordinator_index": self.coordinator_index,
            "branching_factor": self.branching_factor,
            "worker_names": self.worker_names,
            "custom_adjacency": (
                np.array(self.custom_adjacency).tolist() if self.custom_adjacency is not None else None
            ),
            "adjacency": self.adjacency.tolist() if self.adjacency is not None else None,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Topology":
        """Build a ``Topology`` from a config dict (Task 9), e.g. the
        ``topology:`` section of a ``config/network/*.yaml`` file.

        Args:
            config: Dict with keys ``type`` (or ``topology_type``),
                ``num_workers``, and optionally ``coordinator_index``,
                ``branching_factor``, ``custom_adjacency``,
                ``worker_names``.

        Returns:
            A ``Topology`` instance (``build()`` not yet called).

        Raises:
            ValueError: If required keys are missing.
        """
        config = dict(config)
        topology_type = config.get("type", config.get("topology_type"))
        num_workers = config.get("num_workers")
        if topology_type is None:
            raise ValueError("network topology config is missing 'type'")
        if num_workers is None:
            raise ValueError("network topology config is missing 'num_workers'")
        return cls(
            num_workers=num_workers,
            topology_type=topology_type,
            custom_adjacency=config.get("custom_adjacency"),
            coordinator_index=config.get("coordinator_index", 0),
            branching_factor=config.get("branching_factor", 2),
            worker_names=config.get("worker_names"),
        )

    def save(self, path: Union[str, Path]) -> None:
        """Write this topology's configuration + built adjacency (if
        any) to a JSON file (Task 10).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # TASK 8 -- Visualization
    # ------------------------------------------------------------------

    def visualize(self, output_path: Optional[Union[str, Path]] = None, title: Optional[str] = None):
        """Render the topology as a simple circular-layout node/edge
        graph using matplotlib (no extra graph-drawing dependency).

        Args:
            output_path: If given, the figure is saved there (parent
                directories created as needed) and the figure is
                closed. If ``None``, the ``matplotlib.figure.Figure``
                is returned without being saved or closed.
            title: Optional plot title. Defaults to a description of
                the topology type and worker count.

        Returns:
            The ``matplotlib.figure.Figure`` if ``output_path`` is
            ``None``, otherwise ``None``.

        Raises:
            RuntimeError: If ``build()`` has not been called yet.
        """
        if self.adjacency is None:
            raise RuntimeError("Topology.build() must be called before visualize()")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = self.num_workers
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs, ys = np.cos(angles), np.sin(angles)

        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(n):
            for j in range(i + 1, n):
                if self.adjacency[i, j]:
                    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="steelblue", alpha=0.6, zorder=1)

        node_colors = [
            "orange" if (self.topology_type == TopologyType.STAR and i == self.coordinator_index) else "steelblue"
            for i in range(n)
        ]
        ax.scatter(xs, ys, s=600, color=node_colors, edgecolors="black", zorder=2)
        for i in range(n):
            ax.text(xs[i], ys[i], self.worker_names[i], ha="center", va="center", zorder=3, fontsize=8, color="white")

        ax.set_title(title or f"{self.topology_type.value} topology ({n} workers)")
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=120)
            plt.close(fig)
            logger.info("Topology visualization saved to %s", output_path)
            return None
        return fig
