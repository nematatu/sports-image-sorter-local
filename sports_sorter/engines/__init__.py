"""Pluggable classification engines."""

from .base import Engine, FakeEngine, Prediction, available_engines, get_engine, register

__all__ = [
    "Engine",
    "FakeEngine",
    "Prediction",
    "available_engines",
    "get_engine",
    "register",
]
