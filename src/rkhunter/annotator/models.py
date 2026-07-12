from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(slots=True)
class BoxProposal:
    """A pixel-space axis-aligned box proposed by a model or human."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None = None
    source: str = "manual"
    model_id: str | None = None
    model_revision_id: str | None = None
    warning: str | None = None

    def validate(self, image_width: int, image_height: int) -> None:
        if not all(math.isfinite(value) for value in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("box coordinates must be finite")
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative")
        if not (0 <= self.x1 < self.x2 <= image_width):
            raise ValueError(f"invalid x coordinates: {self.x1}, {self.x2}")
        if not (0 <= self.y1 < self.y2 <= image_height):
            raise ValueError(f"invalid y coordinates: {self.y1}, {self.y2}")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not (0 <= self.confidence <= 1)
        ):
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelDescriptor:
    id: str
    revision_id: str
    name: str
    adapter: str
    weights_path: str
    version: str
    sha256: str
    config: dict[str, Any]
    active: bool = True


@dataclass(slots=True)
class ProjectDescriptor:
    id: str
    name: str
    dataset_root: str
    image_dir: str
    label_dir: str
    classes: dict[int, str]
