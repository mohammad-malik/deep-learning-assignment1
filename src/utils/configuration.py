from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def update_dataclass(instance: Any, updates: Mapping[str, Any]) -> Any:
    for field in fields(instance):
        name = field.name
        if name not in updates:
            continue
        current_value = getattr(instance, name)
        new_value = updates[name]

        if is_dataclass(current_value):
            update_dataclass(current_value, new_value)
            continue

        if isinstance(current_value, Path):
            setattr(instance, name, Path(new_value))
        else:
            setattr(instance, name, new_value)
    return instance
