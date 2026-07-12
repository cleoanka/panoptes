"""Vehicle attributes: color and make/model with honest track-level fusion.

Facade::

    from panoptes.attributes import AttributePipeline
    AttributePipeline(config: AttributesConfig)
        .process(frame, tracks, frame_index) -> None

The pipeline runs each enabled extractor on its own frame cadence
(``every_n_frames``) against ACTIVE tracks only, feeds observations into
an :class:`~panoptes.attributes.fusion.AttributeFuser`, and the fuser
writes the consensus into ``Track.attributes``. Importing this package
never touches optional runtimes (onnxruntime is loaded lazily inside the
model-backed extractors).
"""

from __future__ import annotations

import numpy as np

from panoptes.attributes.base import AttributeExtractor
from panoptes.attributes.color import HeuristicColorExtractor, OnnxColorExtractor
from panoptes.attributes.fusion import AttributeFuser
from panoptes.attributes.makemodel import MakeModelExtractor
from panoptes.core.config import AttributesConfig
from panoptes.core.types import Track, TrackState

__all__ = [
    "AttributeExtractor",
    "AttributeFuser",
    "AttributePipeline",
    "HeuristicColorExtractor",
    "MakeModelExtractor",
    "OnnxColorExtractor",
]


class AttributePipeline:
    """Cadenced attribute extraction + fusion for one stream worker.

    Owned by a single worker thread, like every per-stream component —
    no locking. One instance per stream: track ids are only unique per
    tracker, and the fuser keys evidence by ``track_id``.
    """

    def __init__(self, config: AttributesConfig) -> None:
        self.config = config
        self.fuser = AttributeFuser()
        self._extractors: list[AttributeExtractor] = []
        self._cadence: list[int] = []
        self._next_run: list[int] = []
        self._last_frame_index: int | None = None

        if config.color.enabled:
            color: AttributeExtractor
            if config.color.method == "model":
                color = OnnxColorExtractor(config.color)
            else:
                color = HeuristicColorExtractor(config.color)
            self._add(color, config.color.every_n_frames)
        if config.makemodel.enabled:
            self._add(MakeModelExtractor(config.makemodel), config.makemodel.every_n_frames)

    def _add(self, extractor: AttributeExtractor, every_n_frames: int) -> None:
        self._extractors.append(extractor)
        self._cadence.append(max(1, int(every_n_frames)))
        self._next_run.append(0)

    def process(self, frame: np.ndarray, tracks: list[Track], frame_index: int) -> None:
        """Observe ACTIVE tracks on cadence; fused values land in
        ``track.attributes``. FINISHED tracks in ``tracks`` release their
        fusion state."""
        # A stream restart rewinds frame_index; reset the schedule or the
        # extractors would sleep until the previous high-water mark.
        if self._last_frame_index is not None and frame_index < self._last_frame_index:
            self._next_run = [frame_index] * len(self._next_run)
        self._last_frame_index = frame_index

        active = [t for t in tracks if t.state is TrackState.ACTIVE]
        for i, extractor in enumerate(self._extractors):
            # Skip without consuming the cadence slot when the scene is
            # empty, so a freshly confirmed track is observed immediately.
            if frame_index < self._next_run[i] or not active:
                continue
            self._next_run[i] = frame_index + self._cadence[i]
            for track in active:
                observation = extractor.extract(frame, track)
                if observation is None:
                    continue
                value, confidence = observation
                self.fuser.observe(track, extractor.attribute_key, value, confidence)

        for track in tracks:
            if track.state is TrackState.FINISHED:
                self.fuser.forget(track.track_id)

    def forget(self, track_id: int) -> None:
        """Release fusion state for a finished track. The worker should
        call this for tracks the tracker retires without them appearing
        in a later ``process()`` call."""
        self.fuser.forget(track_id)

    def close(self) -> None:
        for extractor in self._extractors:
            extractor.close()
