"""Portable defaults for datasets and pretrained parameters.

Set environment variables before importing the training configuration.
"""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def dataset_path(relative: str = "") -> str:
    root = Path(os.environ.get("FUTURE_DATA_ROOT", str(REPO_ROOT / "dataset"))).expanduser()
    return str(root / relative)


def checkpoint_path(relative: str) -> str:
    root = Path(os.environ.get("FUTURE_CHECKPOINT_ROOT", str(REPO_ROOT / "checkpoints"))).expanduser()
    return str(root / relative)


def pretrained_params(model: str) -> str:
    """Use an explicit local/remote override or the upstream public checkpoint."""
    return os.path.expanduser(
        os.environ.get(f"FUTURE_{model.upper()}_PARAMS", f"gs://openpi-assets/checkpoints/{model}_base/params")
    )
