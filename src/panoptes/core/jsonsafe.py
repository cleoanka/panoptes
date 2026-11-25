"""Strict-JSON helpers for event payloads.

NaN/Infinity are not valid JSON: strict parsers reject them and
``json.dumps(..., allow_nan=False)`` (which httpx uses for ``json=``) raises.
A non-finite float can reach an event payload via ``event.data`` (e.g. a
speed estimate before enough samples), so every outbound serializer —
SSE, WebSocket, webhooks — coerces those to ``null`` first.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["finite"]


def finite(value: Any) -> Any:
    """Return *value* with every non-finite float (recursively) replaced by ``None``."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
