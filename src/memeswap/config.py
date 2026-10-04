"""Load repo-relative YAML config."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def repo_root() -> Path:
    env = os.environ.get("MEMESWAP_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "default.yaml").is_file():
            return parent
    return Path.cwd()


def load_config(path: Path | None = None) -> dict:
    root = repo_root()
    cfg_path = Path(path) if path else root / "configs" / "default.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {cfg_path}")
    data["_config_path"] = str(cfg_path.resolve())
    data["_root"] = str(root)
    return data


def root_path(config: dict, key: str, default: str) -> Path:
    root = Path(config["_root"])
    value = config.get(key) or default
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path
