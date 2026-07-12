"""Panoptes exception hierarchy."""

from __future__ import annotations

__all__ = [
    "BackendUnavailableError",
    "CalibrationError",
    "ConfigError",
    "PanoptesError",
    "StreamSourceError",
]


class PanoptesError(Exception):
    """Base class for all Panoptes errors."""


class ConfigError(PanoptesError):
    """Invalid or inconsistent configuration."""


class BackendUnavailableError(PanoptesError):
    """A model backend (or its optional dependency) is not installed.

    Raised with an actionable message, e.g. the pip extra to install.
    """

    def __init__(self, backend: str, hint: str) -> None:
        self.backend = backend
        self.hint = hint
        super().__init__(f"backend '{backend}' unavailable: {hint}")


class StreamSourceError(PanoptesError):
    """Video source could not be opened or died irrecoverably."""


class CalibrationError(PanoptesError):
    """Camera calibration missing or geometrically degenerate."""
