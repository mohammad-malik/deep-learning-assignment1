from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def save_json(data: Any, path: Path, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent))


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def save_yaml(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
