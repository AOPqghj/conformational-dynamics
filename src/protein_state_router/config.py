"""Small YAML configuration loader."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a mapping-valued YAML configuration."""
    with Path(path).open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must be a mapping")
    return config
