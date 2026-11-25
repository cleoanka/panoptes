"""Outbound plate redaction for the live event feeds (SSE + WebSocket).

With ``privacy.plate_storage == "hashed"`` the storage layer replaces
plate text with ``sha256(salt + plate)`` at write time, so history
endpoints already return the digest. The live feeds serialize the
in-memory :class:`~panoptes.core.events.Event` directly and must apply
the same scrub — otherwise readable plates would leave the system via
``/api/v1/events/stream`` and ``/api/v1/events/ws`` in the exact mode
whose promise is that they do not (see docs/PRIVACY.md).

Plain mode is a passthrough: the original dict is returned untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from panoptes.core.types import PLATE_DATA_KEYS

__all__ = ["redact_event_dict"]


def _scrub(value: Any, hasher: Callable[[str], str], key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _scrub(v, hasher, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, hasher, key) for v in value]
    if isinstance(value, str) and key in PLATE_DATA_KEYS:
        return hasher(value)
    return value


def redact_event_dict(event_dict: dict[str, Any], state: Any) -> dict[str, Any]:
    """Hash plate text inside an outbound event dict when the app runs in
    hashed plate-storage mode; identity in plain mode.

    Never mutates the input: ``Event.to_dict()`` shares the ``data`` dict
    with the live in-memory event.
    """
    if getattr(state.config.privacy, "plate_storage", "plain") != "hashed":
        return event_dict
    data = event_dict.get("data")
    if not data:
        return event_dict
    hasher: Callable[[str], str] | None = getattr(state.db, "hash_plate", None)
    if hasher is None:
        # Fail closed: hashed mode without a hasher (e.g. mid-shutdown)
        # must still never emit readable plates.
        def hasher(_text: str) -> str:
            return "***"

    out = dict(event_dict)
    out["data"] = _scrub(data, hasher)
    return out
