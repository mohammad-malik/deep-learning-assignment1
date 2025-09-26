from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass(frozen=True)
class FaceSample:
    """Container for the metadata associated with a single facial image sample."""

    sample_id: str
    image_path: Path
    expression: int
    valence: float
    arousal: float
    landmark_path: Optional[Path] = None
    face_bbox: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "image_path": str(self.image_path),
            "expression": self.expression,
            "valence": self.valence,
            "arousal": self.arousal,
            "landmark_path": str(self.landmark_path) if self.landmark_path else None,
            "face_bbox": self.face_bbox,
        }
