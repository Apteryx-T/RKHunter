from __future__ import annotations

from importlib.metadata import entry_points
from typing import Callable

from .base import AdapterContext, AutoLabelAdapter

AdapterFactory = Callable[[AdapterContext], AutoLabelAdapter]


class AdapterRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, name: str, factory: AdapterFactory) -> None:
        if not name or name in self._factories:
            raise ValueError(f"adapter already registered or invalid: {name}")
        self._factories[name] = factory

    def names(self) -> list[str]:
        return sorted(self._factories)

    def version(self, name: str) -> str:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ValueError(f"unknown adapter {name}; available: {self.names()}") from exc
        return str(getattr(factory, "adapter_version", "unversioned"))

    def create(self, name: str, context: AdapterContext) -> AutoLabelAdapter:
        try:
            return self._factories[name](context)
        except KeyError as exc:
            raise ValueError(f"unknown adapter {name}; available: {self.names()}") from exc

    def load_entry_points(self) -> None:
        """Load future adapters without changing the core package."""
        for entry_point in entry_points(group="rkhunter.annotator_adapters"):
            if entry_point.name not in self._factories:
                self._factories[entry_point.name] = entry_point.load()


default_registry = AdapterRegistry()
