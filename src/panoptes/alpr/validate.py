"""Country-aware plate validation and structure-based OCR correction.

Turkish plates (the default) have the structure::

    PP L NNNN    PP L NNNNN     (1 letter,  4-5 digits)
    PP LL NNN    PP LL NNNN     (2 letters, 3-4 digits)
    PP LLL NN    PP LLL NNN     (3 letters, 2-3 digits)

where ``PP`` is the province code 01-81 and letters are drawn from the
Turkish plate alphabet (no Q, W, X and no Ç, Ğ, İ, Ö, Ş, Ü).

Because OCR confuses look-alike glyphs (O<->0, I<->1, B<->8, S<->5,
Z<->2, G<->6), :func:`correct_and_validate` tries *structure-aware*
disambiguation: every segmentation consistent with a legal pattern is
scored and the best legal reading wins. This recovers a large share of
single-character OCR errors without any extra model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["PlateValidation", "correct_and_validate", "normalize", "validate"]

TR_LETTERS = "ABCDEFGHIJKLMNOPRSTUVYZ"  # legal Turkish plate letters

# (n_letters, min_digits, max_digits)
_TR_PATTERNS: tuple[tuple[int, int, int], ...] = (
    (1, 4, 5),
    (2, 3, 4),
    (3, 2, 3),
)

_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "B": "8", "S": "5", "Z": "2", "G": "6", "T": "7", "A": "4"}
_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G", "4": "A", "7": "T"}
# conservative subsets used before falling back to aggressive mapping
_TO_DIGIT_SAFE = {"O": "0", "Q": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "G": "6"}
_TO_LETTER_SAFE = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G"}

_EU_GENERIC = re.compile(r"^[A-Z0-9]{4,9}$")


@dataclass(frozen=True, slots=True)
class PlateValidation:
    text: str        # normalised (possibly corrected) plate text, no spaces
    valid: bool
    country: str | None
    corrected: bool  # True if OCR-confusion correction was applied


def normalize(text: str) -> str:
    """Uppercase, strip spaces/dashes/dots, map Turkish glyphs to ASCII."""
    t = text.upper()
    for src, dst in (("İ", "I"), ("Ş", "S"), ("Ğ", "G"), ("Ü", "U"), ("Ö", "O"), ("Ç", "C")):
        t = t.replace(src, dst)
    return re.sub(r"[^A-Z0-9]", "", t)


def _tr_match(text: str) -> bool:
    if len(text) < 5 or len(text) > 10:
        return False
    if not (text[:2].isdigit() and 1 <= int(text[:2]) <= 81):
        return False
    body = text[2:]
    for n_letters, dmin, dmax in _TR_PATTERNS:
        letters, digits = body[:n_letters], body[n_letters:]
        if (
            len(letters) == n_letters
            and all(c in TR_LETTERS for c in letters)
            and digits.isdigit()
            and dmin <= len(digits) <= dmax
        ):
            return True
    return False


def validate(text: str, country: str | None = "TR") -> bool:
    """Check a normalised plate against the country format."""
    t = normalize(text)
    if country == "TR":
        return _tr_match(t)
    if country is None:
        return bool(_EU_GENERIC.match(t))
    # unknown country code: fall back to the permissive generic pattern
    return bool(_EU_GENERIC.match(t))


def _coerce(text: str, n_letters: int, aggressive: bool) -> str | None:
    """Force ``text`` into the TR shape [2 digits][n letters][rest digits];
    return the corrected string or None if impossible."""
    to_digit = _TO_DIGIT if aggressive else _TO_DIGIT_SAFE
    to_letter = _TO_LETTER if aggressive else _TO_LETTER_SAFE
    # A digit->letter substitution is single-glyph OCR recovery, not plate
    # synthesis: require at least one *real* letter already anchoring the
    # letter slot, otherwise an all-digit blob would be fabricated into a
    # "valid" plate by inventing letters where OCR saw only digits.
    letter_slot = text[2 : 2 + n_letters]
    anchored = any(c in TR_LETTERS for c in letter_slot)
    out: list[str] = []
    for i, ch in enumerate(text):
        want_letter = 2 <= i < 2 + n_letters
        if want_letter:
            if ch in TR_LETTERS:
                out.append(ch)
            elif anchored and ch in to_letter and to_letter[ch] in TR_LETTERS:
                out.append(to_letter[ch])
            else:
                return None
        else:
            if ch.isdigit():
                out.append(ch)
            elif ch in to_digit:
                out.append(to_digit[ch])
            else:
                return None
    return "".join(out)


def correct_and_validate(text: str, country: str | None = "TR") -> PlateValidation:
    """Normalise, then try to reach a legal plate via structure-aware
    OCR-confusion correction (Turkish plates only; other countries just
    validate as-is)."""
    t = normalize(text)
    if country != "TR":
        ok = validate(t, country)
        return PlateValidation(text=t, valid=ok, country=country if ok else None, corrected=False)

    if _tr_match(t):
        return PlateValidation(text=t, valid=True, country="TR", corrected=False)

    if not 5 <= len(t) <= 10:
        return PlateValidation(text=t, valid=False, country=None, corrected=False)

    # Enumerate *every* legal (aggressive, n_letters) split — not just the
    # first that validates — and keep the least-corrupting reading. Accepting
    # the first match let a fewer-letters split win by coercing a genuine
    # letter into its look-alike digit (e.g. "O6AB1234" -> "06A81234", B->8)
    # before the no-letter-corruption 2-letter reading was ever tried. The
    # score is the number of coerced glyphs (chars left untouched cost 0);
    # ties prefer the safe mapping (aggressive=False sorts first), so a real
    # letter that already fits its slot is never swapped for a digit.
    best: tuple[int, bool, str] | None = None
    for aggressive in (False, True):
        for n_letters, dmin, dmax in _TR_PATTERNS:
            n_digits = len(t) - 2 - n_letters
            if not dmin <= n_digits <= dmax:
                continue
            candidate = _coerce(t, n_letters, aggressive)
            if candidate is None or not _tr_match(candidate):
                continue
            score = sum(a != b for a, b in zip(t, candidate, strict=True))
            key = (score, aggressive, candidate)
            if best is None or key < best:
                best = key
    if best is not None:
        return PlateValidation(text=best[2], valid=True, country="TR", corrected=True)

    return PlateValidation(text=t, valid=False, country=None, corrected=False)
