"""Discovery and construction of bot plugins."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Type

from .base import Bot

__all__ = ["register_bot", "get_bot", "available_bots", "bot_info"]

_REGISTRY: dict[str, Type[Bot]] = {}
_DISCOVERED = False


def register_bot(cls: Type[Bot]) -> Type[Bot]:
    """Class decorator that makes a bot available by name."""
    if not issubclass(cls, Bot):
        raise TypeError(f"{cls.__name__} does not subclass Bot")
    _REGISTRY[cls.name] = cls
    return cls


def _discover() -> None:
    """Import every module in ``micromouse.bots`` so decorators run."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True  # set first: a failing import must not retrigger discovery
    package = importlib.import_module(__package__)
    for module in pkgutil.iter_modules(package.__path__):
        if module.name in {"base", "registry"}:
            continue
        importlib.import_module(f"{__package__}.{module.name}")


def available_bots() -> list[str]:
    """Names of every registered bot, sorted."""
    _discover()
    return sorted(_REGISTRY)


def get_bot(name: str, /, **kwargs) -> Bot:
    """Construct a registered bot by name."""
    _discover()
    if name not in _REGISTRY:
        raise KeyError(f"unknown bot {name!r}; available: {', '.join(available_bots())}")
    return _REGISTRY[name](**kwargs)


def bot_info() -> list[dict[str, str]]:
    """Metadata for every bot, for the dashboard and CLI listings."""
    _discover()
    return [
        {
            "name": cls.name,
            "style": cls.style,
            "description": cls.description,
        }
        for _, cls in sorted(_REGISTRY.items())
    ]
