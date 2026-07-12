"""Event-triggered JPEG snapshots.

Files land under ``media_dir/<stream_id>/<YYYYMMDD>/<event_id>.jpg``
(UTC date). :meth:`SnapshotSaver.maybe_save` *returns* the path relative
to ``media_dir`` — :class:`~panoptes.core.events.Event` is frozen, so
the caller attaches it via ``dataclasses.replace(event,
snapshot_path=path)``. Relative paths keep DB rows portable and map 1:1
onto the API's ``GET /media/{path}`` route.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from panoptes.core.config import SnapshotConfig
from panoptes.core.events import Event
from panoptes.core.types import Track
from panoptes.pipeline.annotate import annotate

__all__ = ["SnapshotSaver"]

_JPEG_QUALITY = 90
_RATE_WINDOW_S = 60.0


class SnapshotSaver:
    """Owned by a single worker thread — no locking needed."""

    def __init__(self, config: SnapshotConfig, media_dir: str | Path) -> None:
        self._config = config
        self._media_dir = Path(media_dir)
        self._on_events = set(config.on_events)
        # per-stream monotonic save times inside the rate window
        self._recent: dict[str, deque[float]] = defaultdict(deque)

    def maybe_save(
        self, event: Event, frame: np.ndarray, tracks: list[Track]
    ) -> str | None:
        """Save a JPEG for qualifying events; return the media-relative path."""
        if not self._config.enabled or frame is None:
            return None
        wanted = event.type.value in self._on_events or bool(
            event.data.get("snapshot_requested")
        )
        if not wanted:
            return None
        if not self._admit(event.stream_id):
            return None

        day = datetime.fromtimestamp(event.wall_ts, tz=UTC).strftime("%Y%m%d")
        rel = Path(event.stream_id) / day / f"{event.id}.jpg"
        full = self._media_dir / rel
        full.parent.mkdir(parents=True, exist_ok=True)

        image = annotate(frame, tracks, None, None) if self._config.annotate else frame
        ok = cv2.imwrite(str(full), image, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if not ok:
            return None
        return rel.as_posix()

    def _admit(self, stream_id: str) -> bool:
        # wall-rate limit, so it uses the monotonic clock (not stream PTS)
        now = time.monotonic()
        window = self._recent[stream_id]
        while window and now - window[0] > _RATE_WINDOW_S:
            window.popleft()
        if len(window) >= self._config.max_per_minute:
            return False
        window.append(now)
        return True
