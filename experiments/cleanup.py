"""
cleanup.py

Purpose:
    Phase 7.5, Task 1 -- automatic cleanup of previously-generated
    artifacts before every experiment/training run, so every run
    starts from a completely fresh, reproducible state (no appended
    logs, no stale metrics, no duplicate artifacts left over from a
    prior invocation).

Scope:
    This module only ever deletes *generated output*, never source
    code. It is intentionally conservative:

      - Only the directories explicitly listed in
        ``GENERATED_OUTPUT_DIRS`` are touched (``artifacts/``,
        ``analysis/network_plots/``, ``analysis/plots/``,
        ``analysis/reports/``, ``logs/``) -- everything else
        (``config/``, ``data/raw/``, ``models/``, ``training/``,
        ``analysis/*.py``, ``network/*.py``, ``api/``) is never
        walked, let alone deleted.
      - Within those directories, only files/subdirectories are
        removed -- the directory itself is recreated so downstream
        writers never have to guess whether it exists.
      - Non-generated documentation files that happen to live inside a
        generated-output directory (e.g. ``artifacts/README.md``) are
        preserved rather than deleted, since Task 1 says "do not
        delete source code" and checked-in documentation is source,
        not a generated artifact.

Who calls this:
    ``experiments.experiment_runner.ExperimentRunner.__init__`` calls
    this once per runner session (Task 1/6 -- "no manual commands
    should ever be necessary"). ``training.local_sgd.LocalSGDTrainer``
    does NOT call this internally, because it is driven by
    ExperimentRunner across many (name, seed) runs in a single
    session and per-run cleanup would delete the very sibling
    experiments' artifacts a multi-experiment sweep is trying to
    accumulate. Standalone entry points (``python
    training/local_sgd.py``) call this once from their own ``main()``
    instead, matching "before every new experiment/training run" for
    that run's session.

Public Interface:
    Functions:
        clean_generated_artifacts(project_root=None) -> dict
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories whose CONTENTS (not the directories themselves) are wiped
# before every run (Task 1). Generated-output locations only.
GENERATED_OUTPUT_DIRS = [
    "artifacts",
    "analysis/network_plots",
    "analysis/plots",
    "analysis/reports",
    "logs",
]

# The canonical generated-output skeleton, recreated (empty) after
# every clean so every writer in the pipeline has somewhere to write
# without needing its own mkdir(parents=True) fallback logic.
_SKELETON_DIRS = GENERATED_OUTPUT_DIRS + [
    "artifacts/network",
    "artifacts/experiments",
    "artifacts/plots",
    "artifacts/metrics",
    "artifacts/logs",
    "artifacts/models",
]

# Documentation/meta files that must survive a clean even though they
# live inside a directory this module otherwise empties.
_PRESERVE_FILENAMES = {"readme.md", ".gitkeep", ".gitignore"}


def _clean_directory(root: Path) -> Dict[str, int]:
    """Remove every file/subdirectory directly and transitively under
    ``root``, except files named in ``_PRESERVE_FILENAMES`` at the top
    level. The directory itself is left in place (created if missing).
    """
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return {"removed_files": 0, "removed_dirs": 0}

    removed_files = 0
    removed_dirs = 0
    for child in sorted(root.iterdir()):
        if child.is_file() or child.is_symlink():
            if child.name.lower() in _PRESERVE_FILENAMES:
                continue
            child.unlink()
            removed_files += 1
        elif child.is_dir():
            shutil.rmtree(child)
            removed_dirs += 1
    return {"removed_files": removed_files, "removed_dirs": removed_dirs}


def clean_generated_artifacts(project_root: Optional[Path] = None) -> Dict[str, Dict[str, int]]:
    """Wipe every generated-output directory (Task 1), leaving source
    code (``config/``, ``data/raw/``, ``models/``, ``training/``,
    ``analysis/*.py``, ``network/*.py``, ``api/``) untouched, then
    recreate the empty output skeleton so the run starts from a
    completely fresh, reproducible state.

    Args:
        project_root: Repository root. Defaults to this file's
            grandparent directory (the project root).

    Returns:
        Dict mapping each cleaned directory (relative path) to
        ``{"removed_files": N, "removed_dirs": M}``.
    """
    project_root = Path(project_root) if project_root is not None else _PROJECT_ROOT

    report: Dict[str, Dict[str, int]] = {}
    for rel_dir in GENERATED_OUTPUT_DIRS:
        stats = _clean_directory(project_root / rel_dir)
        report[rel_dir] = stats
        logger.info(
            "Cleaned %s: removed %d file(s), %d subdirectory(ies)",
            project_root / rel_dir, stats["removed_files"], stats["removed_dirs"],
        )

    for rel_dir in _SKELETON_DIRS:
        (project_root / rel_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Cleanup complete: project is in a fresh, reproducible state")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    clean_generated_artifacts()
