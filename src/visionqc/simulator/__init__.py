"""Production-line simulator: virtual camera, trigger generator, virtual reject
station, and runtime fault injection.

The simulator drives the real :class:`~visionqc.orchestrator.Orchestrator` with
frames from a pluggable :class:`~visionqc.simulator.source.ImageSource`, so the
whole system can be demonstrated end-to-end with zero physical hardware — either
against a directory of real MVTec images or fully synthetic frames generated on
the fly.
"""

from __future__ import annotations

from .line import LineSimulator, LineState
from .source import (
    DirectoryImageSource,
    ImageSource,
    SourceImage,
    SyntheticImageSource,
    build_source,
)

__all__ = [
    "DirectoryImageSource",
    "ImageSource",
    "LineSimulator",
    "LineState",
    "SourceImage",
    "SyntheticImageSource",
    "build_source",
]
