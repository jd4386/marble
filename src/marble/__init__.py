"""MARBLE: Multi-scale linear-time encoder for whole-slide image analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

__all__ = ["MarbleClassifier", "MarbleModel", "__version__"]

if TYPE_CHECKING:
    from marble.model import MarbleClassifier, MarbleModel


def __getattr__(name: str):
    if name in {"MarbleClassifier", "MarbleModel"}:
        from marble.model import MarbleClassifier, MarbleModel

        return MarbleClassifier if name == "MarbleClassifier" else MarbleModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
