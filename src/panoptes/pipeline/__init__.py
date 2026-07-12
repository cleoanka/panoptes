"""Panoptes pipeline: sources, frame governor, GPU micro-batching
scheduler, per-stream workers and the manager facade.

Facade::

    from panoptes.pipeline import PipelineManager
"""

from panoptes.pipeline.manager import PipelineManager

__all__ = ["PipelineManager"]
