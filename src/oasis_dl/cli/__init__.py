"""Hydra entrypoints. One per pipeline stage.

Config groups live in ``configs/`` at the repo root, reached by a path relative to this
package. That resolves under the editable install the Makefile performs
(``pip install -e .``), which ADR-0008 already requires; a non-editable install would
need ``--config-dir`` pointed at the repo.
"""

from pathlib import Path

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")
"""Absolute path to the repo's ``configs/`` directory."""
