"""VisionQC isolated GPU inference worker (separate OS process).

This package is intentionally decoupled from the ``visionqc`` line-controller
package. It owns the CUDA context and the loaded anomaly-detection model and
serves inference over localhost HTTP (see :mod:`visionqc_inference.worker`).

``anomalib`` (and torch) are **optional** imports — the modules here import
cleanly without the ``ai`` extra installed so the main test environment can
import and exercise the worker in ``--fake`` mode.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
