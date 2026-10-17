"""Map character offsets in decoded output back to generated-token indices.

The handoff requires an incremental decode of the generated ids (prefix
lengths), not a re-tokenization of the text: re-tokenizing can segment
differently from what the model actually produced, which would silently shift
every onset index.

Nothing here loads a model; everything takes a tokenizer-like object with a
`decode` method, so the tests run without touching the GPU.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Any, Protocol


class _Decoder(Protocol):
    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...


class AlignmentError(RuntimeError):
    """Raised when prefix decoding does not behave like a prefix."""


def prefix_lengths(tokenizer: Any, ids: list[int]) -> list[int]:
    """`out[k]` is the character length of decoding the first k ids.

    Token k therefore covers characters `[out[k], out[k+1])`.

    Raises AlignmentError if the decoded prefixes are not non-decreasing, which
    is how a partial multi-byte UTF-8 sequence (decoded as U+FFFD) would show
    up. Better to fail loudly than to map onsets to the wrong token.
    """
    out = [0]
    for k in range(1, len(ids) + 1):
        out.append(len(tokenizer.decode(ids[:k], skip_special_tokens=False)))
    for k in range(1, len(out)):
        if out[k] < out[k - 1]:
            raise AlignmentError(
                f"decoded prefix shrank at token {k}: {out[k - 1]} -> {out[k]}"
            )
    return out


def check_prefix_consistency(tokenizer: Any, ids: list[int], full_text: str) -> list[int]:
    """prefix_lengths plus the stronger claim: every prefix is a real prefix.

    Returns the token indices where the decoded prefix is NOT a prefix of the
    full text (empty list is the healthy case).
    """
    bad: list[int] = []
    for k in range(1, len(ids) + 1):
        piece = tokenizer.decode(ids[:k], skip_special_tokens=False)
        if not full_text.startswith(piece):
            bad.append(k)
    return bad


def token_index_for_char(prefix_len: list[int], char_pos: int) -> int:
    """The index of the token whose character span contains `char_pos`."""
    if char_pos < 0:
        raise ValueError(f"negative char position: {char_pos}")
    # prefix_len is non-decreasing; find the last boundary <= char_pos.
    t = bisect_right(prefix_len, char_pos) - 1
    # Zero-width tokens (a token that decodes to "") can leave several
    # boundaries at the same offset; step forward to the one that actually
    # covers the character.
    while t + 1 < len(prefix_len) and prefix_len[t + 1] <= char_pos:
        t += 1
    return max(0, min(t, len(prefix_len) - 2))


def hard_check(tokenizer: Any, ids: list[int], token_index: int, marker_text: str) -> bool:
    """The handoff's hard check.

    Decoding the generated ids from `token_index` onward must start with the
    marker text, case-insensitive, after stripping leading whitespace.
    """
    tail = tokenizer.decode(ids[token_index:], skip_special_tokens=False)
    return tail.lstrip().lower().startswith(marker_text.lower())


def is_mid_token_onset(prefix_len: list[int], token_index: int, char_start: int) -> bool:
    """True when the marker starts inside a token rather than at its boundary."""
    return prefix_len[token_index] != char_start
