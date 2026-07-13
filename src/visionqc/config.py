"""Application configuration.

All settings are overridable through environment variables prefixed with
``VISIONQC_`` (e.g. ``VISIONQC_DB_PATH=/data/qc.db``) or a local ``.env`` file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the VisionQC line controller."""

    model_config = SettingsConfigDict(
        env_prefix="VISIONQC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Data layer -----------------------------------------------------
    db_path: Path = Field(
        default=Path("visionqc.db"),
        description="Path to the SQLite database file.",
    )
    evidence_dir: Path = Field(
        default=Path("evidence"),
        description="Root directory for evidence image storage.",
    )
    read_pool_size: int = Field(
        default=4,
        ge=1,
        description="Number of read-only SQLite connections in the query pool.",
    )

    # --- Inference worker ----------------------------------------------
    inference_worker_url: str = Field(
        default="http://127.0.0.1:8001/infer",
        description="URL of the localhost GPU inference worker.",
    )
    inference_timeout_s: float = Field(
        default=2.5,
        gt=0,
        description="Per-call timeout for the inference worker, in seconds.",
    )

    # --- Lifecycle ------------------------------------------------------
    lifecycle_timeout_s: float = Field(
        default=10.0,
        gt=0,
        description="Max time a product may stay in a non-terminal state before "
        "the watchdog forces a FAULT.",
    )
    watchdog_interval_s: float = Field(
        default=1.0,
        gt=0,
        description="How often the lifecycle watchdog scans for stuck products.",
    )

    # --- Event bus ------------------------------------------------------
    ws_queue_maxsize: int = Field(
        default=3,
        ge=1,
        description="Bounded per-client WebSocket queue size (drop-oldest on overflow).",
    )

    # --- Simulator (placeholder — built by a separate task) -------------
    simulator_enabled: bool = Field(
        default=False,
        description="Whether the virtual-line simulator starts with the app.",
    )
    simulator_interval_s: float = Field(
        default=1.0,
        gt=0,
        description="Interval between simulated triggers, in seconds.",
    )
    simulator_fault_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of simulated cycles that inject a fault.",
    )


def get_settings(**overrides: object) -> Settings:
    """Build a :class:`Settings` instance, applying optional overrides.

    Overrides are primarily useful for tests, which pass a temporary
    ``db_path`` / ``evidence_dir``.
    """

    return Settings(**overrides)  # type: ignore[arg-type]
