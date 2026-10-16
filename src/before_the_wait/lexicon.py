"""Reasoning-marker lexicon and matcher, ported verbatim from frictionmap.

Source: https://github.com/alatruwe/frictionmap/blob/6bc1552d335eab1434d93519644689f8e560e456/src/frictionmap/clusters.py
Commit: 6bc1552d335eab1434d93519644689f8e560e456

Copied unchanged, code and comments: _LEXICON, _MARKER_RE, _FENCE_RE,
_fenced_regions, _in_any_region, strip_code_fences, find_markers.
Not ported: the cluster-detection code in that module. Do not edit.
"""
from __future__ import annotations

import re

# Empirically calibrated against the attune corpus (April 2026, 30-block
# hand-tagged sample). "but" and "let me check" were dropped — both were
# dominated by forward-planning false positives, not re-evaluation. "let
# me think" and "however" were missing markers caught during the same
# pass.
_LEXICON = (
    "actually", "wait", "hmm", "no,",
    "let me reconsider", "on second thought", "scratch that",
    "hold on", "reconsidering", "i was wrong", "let me think",
    "now I'm realizing", "however",
)

# Longest-first so "let me reconsider" wins over "let me check" at the
# same offset; \b applies at outer boundaries only (the comma in "no,"
# and the apostrophe in "now I'm realizing" are matched literally).
_MARKER_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in sorted(_LEXICON, key=len, reverse=True)) + r")",
    re.IGNORECASE,
)

_FENCE_RE = re.compile(r"^```[\s\S]*?^```", re.MULTILINE)


def _fenced_regions(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _FENCE_RE.finditer(text)]


def _in_any_region(pos: int, regions: list[tuple[int, int]]) -> bool:
    for start, end in regions:
        if start <= pos < end:
            return True
    return False


def strip_code_fences(text: str) -> str:
    """Remove ``` fenced blocks. Used for per-100-words denominators."""
    return _FENCE_RE.sub("", text)


def find_markers(text: str) -> list[tuple[int, int, str]]:
    """Return (char_start, char_end, matched_text) per marker occurrence
    in original-text coordinates. Hits inside fenced code blocks are
    filtered out — downstream `_excerpts_for_block` consumes the same
    coordinate space, so no offset translation is needed.
    """
    regions = _fenced_regions(text)
    out: list[tuple[int, int, str]] = []
    for m in _MARKER_RE.finditer(text):
        if _in_any_region(m.start(), regions):
            continue
        out.append((m.start(), m.end(), m.group(0)))
    out.sort(key=lambda x: x[0])
    return out
