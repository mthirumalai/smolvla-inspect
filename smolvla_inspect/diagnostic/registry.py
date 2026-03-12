"""Primitive registry — central catalog of all diagnostic primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class PrimitiveSpec:
    """Describes a registered diagnostic primitive."""
    name: str
    fn: Callable
    category: str          # "scene" | "model" | "composite" | "counterfactual" | "dataset"
    requires_gpu: bool
    requires_model: bool
    requires_scene_models: bool
    cost: str              # "cheap" | "moderate" | "expensive"
    description: str


REGISTRY: dict[str, PrimitiveSpec] = {}


def register_primitive(name: str, *, category: str, cost: str = "cheap",
                       requires_gpu: bool = False, requires_model: bool = False,
                       requires_scene_models: bool = False, description: str = ""):
    """Decorator to register a function as a diagnostic primitive."""
    def decorator(fn: Callable) -> Callable:
        REGISTRY[name] = PrimitiveSpec(
            name=name, fn=fn, category=category,
            requires_gpu=requires_gpu, requires_model=requires_model,
            requires_scene_models=requires_scene_models,
            cost=cost, description=description,
        )
        return fn
    return decorator


def get_primitive(name: str) -> PrimitiveSpec:
    """Look up a primitive by name. Raises KeyError if not found."""
    return REGISTRY[name]


def list_primitives(category: str | None = None) -> list[PrimitiveSpec]:
    """List all registered primitives, optionally filtered by category."""
    specs = list(REGISTRY.values())
    if category is not None:
        specs = [s for s in specs if s.category == category]
    return sorted(specs, key=lambda s: s.name)
