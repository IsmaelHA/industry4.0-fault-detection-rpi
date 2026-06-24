"""Config loading and RNG seeding."""

from __future__ import annotations
from pathlib import Path
from typing import Union

import numpy as np
import yaml


def load_config(path: Union[str, Path] = "config.yaml") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)
