"""Auto-label adapter registry."""

from .base import AutoLabelAdapter, AdapterContext
from .registry import AdapterRegistry, default_registry
# Import built-ins for registration. Heavy model libraries remain lazy-loaded.
from . import grounding_dino as _grounding_dino  # noqa: F401,E402
from . import ultralytics_yolo as _ultralytics_yolo  # noqa: F401,E402

__all__ = ["AdapterContext", "AdapterRegistry", "AutoLabelAdapter", "default_registry"]
