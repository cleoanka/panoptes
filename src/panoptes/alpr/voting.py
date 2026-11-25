"""Per-track plate consensus voting.

Single-frame OCR reads are noisy; a per-character, confidence-weighted
majority vote across a track's reads recovers the true plate even when no
single read is fully correct.

Algorithm
---------
1. Reads are grouped by text length and the *dominant-length* group is
   voted (most reads; ties broken by total confidence, then longer text).
   Cross-length character alignment is not attempted — a wrong-length read
   is treated as a whole-read outlier.
2. For each character slot, every read in the group adds its confidence as
   weight to the character it saw; the consensus character is the weight
   argmax (exact-weight ties break to the lower codepoint, so the result is
   a function of the read multiset alone — independent of arrival order).
3. Consensus confidence is the mean, over slots, of the winning character's
   weight share (``winner_weight / slot_total_weight``) — 1.0 means every
   read agreed on every slot.

A consensus is *emitted* (returned) once at least ``vote_min_reads`` reads
in the dominant group agree exactly with the consensus text, and again only
when the consensus text later changes. ``PlateRead.n_reads`` is that
agreeing-read count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from panoptes.core.types import PlateRead

__all__ = ["PlateVoter"]

# Zero-confidence reads still carry positional evidence; give them a
# negligible-but-nonzero weight so argmax stays well-defined.
_MIN_WEIGHT = 1e-6


@dataclass(slots=True)
class _TrackVotes:
    reads: list[tuple[str, float]] = field(default_factory=list)
    emitted: str | None = None


class PlateVoter:
    """Accumulates validated plate reads per track and emits consensuses."""

    def __init__(self, vote_min_reads: int) -> None:
        self.vote_min_reads = max(1, int(vote_min_reads))
        self._tracks: dict[int, _TrackVotes] = {}

    def add_read(self, track_id: int, text: str, conf: float) -> PlateRead | None:
        """Record one validated read; return a new consensus or ``None``.

        The returned :class:`PlateRead` carries only ``text``, ``confidence``
        and ``n_reads`` — the caller re-validates and fills country/validity.
        """
        if not text:
            return None
        state = self._tracks.setdefault(track_id, _TrackVotes())
        state.reads.append((text, max(float(conf), _MIN_WEIGHT)))

        group = self._dominant_group(state.reads)
        consensus, confidence = self._consensus(group)
        n_agree = sum(1 for t, _ in group if t == consensus)
        if n_agree < self.vote_min_reads or consensus == state.emitted:
            return None
        state.emitted = consensus
        return PlateRead(text=consensus, confidence=confidence, n_reads=n_agree)

    def forget(self, track_id: int) -> None:
        """Drop all voting state for a track (finished/pruned)."""
        self._tracks.pop(track_id, None)

    def tracked_ids(self) -> set[int]:
        """Track ids currently holding voting state (for pruning)."""
        return set(self._tracks)

    @staticmethod
    def _dominant_group(reads: list[tuple[str, float]]) -> list[tuple[str, float]]:
        by_len: dict[int, list[tuple[str, float]]] = {}
        for text, conf in reads:
            by_len.setdefault(len(text), []).append((text, conf))
        return max(
            by_len.values(),
            key=lambda g: (len(g), sum(c for _, c in g), len(g[0][0])),
        )

    @staticmethod
    def _consensus(group: list[tuple[str, float]]) -> tuple[str, float]:
        length = len(group[0][0])
        chars: list[str] = []
        shares: list[float] = []
        for i in range(length):
            weights: dict[str, float] = {}
            for text, conf in group:
                weights[text[i]] = weights.get(text[i], 0.0) + conf
            total = sum(weights.values())
            # Highest weight, then lowest codepoint: a deterministic, read-order
            # independent tie-break (bare ``max`` would pick the first-inserted
            # key, i.e. whichever character happened to arrive first).
            winner = min(weights, key=lambda ch: (-weights[ch], ch))
            chars.append(winner)
            shares.append(weights[winner] / total if total > 0 else 0.0)
        confidence = sum(shares) / len(shares) if shares else 0.0
        return "".join(chars), confidence
