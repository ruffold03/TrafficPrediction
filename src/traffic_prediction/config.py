from __future__ import annotations

import os
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent)
    env_url = os.environ.get("DATABASE_URL")
    if env_url:
        cfg["database_url"] = env_url
    return cfg


def resolve_path(cfg: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(cfg["_root"]) / path
