"""Automatic license-plate recognition (detection -> OCR -> voting -> events).

Importable with base dependencies only; the optional runtimes behind the
``panoptes[alpr]`` extra are loaded lazily when :class:`AlprPipeline` is
constructed with ``enabled=True``.
"""

from panoptes.alpr.pipeline import AlprPipeline

__all__ = ["AlprPipeline"]
