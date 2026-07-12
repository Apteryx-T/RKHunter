from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..models import BoxProposal, ModelDescriptor


@dataclass(slots=True)
class AdapterContext:
    model: ModelDescriptor
    device: str = "cpu"


class AutoLabelAdapter(Protocol):
    """Stable provider contract for local or externally supplied label models."""

    adapter_name: str

    def propose(
        self,
        image_path: Path,
        *,
        image_width: int,
        image_height: int,
        params: dict[str, Any],
    ) -> list[BoxProposal]: ...
