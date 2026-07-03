"""
latency_matrix.py

Purpose:
    Model the network latency between the five simulated data centers
    and the coordinator, to be used when simulating synchronization
    communication cost.

Future Responsibility:
    - Define/load a latency matrix between all data center pairs.
    - Expose latency values to training/local_sgd.py and
      training/fedavg_baseline.py for communication cost accounting.

TODO:
    - Decide on realistic latency values between the five regions.
    - Implement matrix construction/loading logic.
    - Implement a lookup interface for other modules to query latency.

Public Interface:
    LatencyMatrix

    Methods:
        build()
        load()
        get_latency()
"""


class LatencyMatrix:
    """Inter-data-center latency model. Logic to be implemented in Phase 4."""

    def build(self, *args, **kwargs):
        ...

    def load(self, *args, **kwargs):
        ...

    def get_latency(self, *args, **kwargs):
        ...
