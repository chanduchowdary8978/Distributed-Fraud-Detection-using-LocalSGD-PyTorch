"""
drift_monitor.py

Purpose:
    Monitor data and/or model drift across data centers over the course
    of training, to detect when regional fraud patterns diverge.

Future Responsibility:
    - Track statistics of incoming regional data over time.
    - Compare distributions across data centers and across time.
    - Surface drift signals for use by monitoring dashboards or
      experiments/run_anomaly_injection.py.

TODO:
    - Decide on drift detection method(s).
    - Implement statistics tracking per data center.
    - Implement drift alerting/reporting interface.

Public Interface:
    DriftMonitor

    Methods:
        update()
        detect_drift()
        report()
"""


class DriftMonitor:
    """Data/model drift monitor. Logic to be implemented in Phase 5."""

    def update(self, *args, **kwargs):
        ...

    def detect_drift(self, *args, **kwargs):
        ...

    def report(self, *args, **kwargs):
        ...
