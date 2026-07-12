"""Panoptes HTTP surface: REST + SSE + WebSocket + MJPEG + jobs + dashboard."""

from __future__ import annotations

from panoptes.api.app import create_app

__all__ = ["create_app"]
