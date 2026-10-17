"""Turn a trace into marker onsets with their local context.

Pure text and counting. No model, no hidden states.
"""
from __future__ import annotations

import string
from collections import Counter
from dataclasses import dataclass
from typing import Any

from before_the_wait.align import (
    hard_check,
    is_mid_token_onset,
    token_index_for_char,
)
from before_the_wait.lexicon import _LEXICON, find_markers

# Window size and repeat count for the `looped` flag. Pilot definition: any
# window of this many consecutive generated ids inside the reasoning span that
# occurs at least LOOP_REPEATS times verbatim.
LOOP_WINDOW = 30
LOOP_REPEATS = 3

TOKEN_KINDS = ("newline", "period", "comma", "other_punct", "digit", "word", "other")

_PUNCT = set(string.punctuation)


def token_kind(text: str) -> str:
    """Classify a token by its decoded text.

    Precedence is deliberate and documented in the report: a line break beats
    everything (it is the strongest structural cue), then sentence-final
    punctuation, then commas. `prev_token_text` is kept raw in the CSV so these
    categories can be revised later without regenerating anything.
    """
    if "\n" in text:
        return "newline"
    s = text.strip()
    if s == "":
        return "other"
    if "." in s:
        return "period"
    if "," in s:
        return "comma"
    if all(ch in _PUNCT for ch in s):
        return "other_punct"
    has_alpha = any(ch.isalpha() for ch in s)
    if any(ch.isdigit() for ch in s) and not has_alpha:
        return "digit"
    if has_alpha:
        return "word"
    return "other"


def is_paragraph_initial(reasoning_text: str, char_start: int) -> bool:
    """True if the hit is the first non-whitespace text after a \\n\\n."""
    before = reasoning_text[:char_start]
    cut = before.rfind("\n\n")
    if cut == -1:
        return False
    return before[cut + 2 :].strip() == ""


def same_paragraph(reasoning_text: str, a: int, b: int) -> bool:
    """True if no blank line separates two character positions."""
    lo, hi = (a, b) if a <= b else (b, a)
    return "\n\n" not in reasoning_text[lo:hi]


def is_looped(
    reasoning_ids: list[int], window: int = LOOP_WINDOW, repeats: int = LOOP_REPEATS
) -> bool:
    if len(reasoning_ids) < window:
        return False
    counts: Counter[tuple[int, ...]] = Counter()
    for i in range(len(reasoning_ids) - window + 1):
        counts[tuple(reasoning_ids[i : i + window])] += 1
        if counts[tuple(reasoning_ids[i : i + window])] >= repeats:
            return True
    return False


@dataclass
class Onset:
    dataset: str
    index: int
    token_index: int
    marker: str
    char_start: int
    prev_token_text: str
    prev_token_kind: str
    paragraph_initial: bool
    onset_mid_token: bool
    tokens_since_prev_hit: int | None
    tokens_to_next_hit: int | None
    same_paragraph_as_prev_hit: bool | None


@dataclass
class HardCheckFailure:
    dataset: str
    index: int
    token_index: int
    marker: str
    char_start: int
    mid_token: bool
    decoded_tail: str


def label_trace(
    tokenizer: Any,
    *,
    dataset: str,
    index: int,
    ids: list[int],
    output_text: str,
    prefix_len: list[int],
    reasoning_char_span: tuple[int, int],
    reasoning_token_span: tuple[int, int],
) -> tuple[list[Onset], list[HardCheckFailure]]:
    """Find markers in the reasoning text and map each to a generated token."""
    c0, c1 = reasoning_char_span
    reasoning_text = output_text[c0:c1]

    onsets: list[Onset] = []
    failures: list[HardCheckFailure] = []
    hits = find_markers(reasoning_text)

    for local_start, _local_end, marker in hits:
        char_start = c0 + local_start  # back to output_text coordinates
        t = token_index_for_char(prefix_len, char_start)
        mid = is_mid_token_onset(prefix_len, t, char_start)

        if not hard_check(tokenizer, ids, t, marker):
            failures.append(
                HardCheckFailure(
                    dataset=dataset,
                    index=index,
                    token_index=t,
                    marker=marker,
                    char_start=char_start,
                    mid_token=mid,
                    decoded_tail=tokenizer.decode(ids[t : t + 6], skip_special_tokens=False),
                )
            )
            continue

        prev_text = (
            tokenizer.decode([ids[t - 1]], skip_special_tokens=False) if t > 0 else ""
        )
        onsets.append(
            Onset(
                dataset=dataset,
                index=index,
                token_index=t,
                marker=marker.lower(),
                char_start=char_start,
                prev_token_text=prev_text,
                prev_token_kind=token_kind(prev_text) if t > 0 else "other",
                paragraph_initial=is_paragraph_initial(reasoning_text, local_start),
                onset_mid_token=mid,
                tokens_since_prev_hit=None,
                tokens_to_next_hit=None,
                same_paragraph_as_prev_hit=None,
            )
        )

    # Gaps, in reasoning-local character coordinates for the paragraph test.
    for i, o in enumerate(onsets):
        if i > 0:
            prev = onsets[i - 1]
            o.tokens_since_prev_hit = o.token_index - prev.token_index
            o.same_paragraph_as_prev_hit = same_paragraph(
                reasoning_text, prev.char_start - c0, o.char_start - c0
            )
        if i + 1 < len(onsets):
            o.tokens_to_next_hit = onsets[i + 1].token_index - o.token_index

    return onsets, failures


def negative_pool(
    tokenizer: Any,
    ids: list[int],
    reasoning_token_span: tuple[int, int],
    onset_token_indices: set[int],
    lookahead: int,
) -> tuple[Counter[str], int]:
    """Positions with no onset in the next `lookahead` tokens, by token kind.

    Only positions with a full lookahead window inside the reasoning span are
    counted; the second return value is the number of positions skipped for
    insufficient lookahead (the review asked for both numbers).
    """
    t0, t1 = reasoning_token_span
    by_kind: Counter[str] = Counter()
    skipped = 0
    for t in range(t0, t1):
        if t + lookahead >= t1:
            skipped += 1
            continue
        if any(x in onset_token_indices for x in range(t + 1, t + 1 + lookahead)):
            continue
        by_kind[token_kind(tokenizer.decode([ids[t]], skip_special_tokens=False))] += 1
    return by_kind, skipped


def paragraph_initial_ngrams(reasoning_text: str, n: int) -> Counter[str]:
    """Words (n=1) or bigrams (n=2) that open a paragraph, lower-cased.

    Used only to list what the lexicon might be missing. Nothing is added to
    the lexicon.
    """
    out: Counter[str] = Counter()
    for chunk in reasoning_text.split("\n\n")[1:]:
        words = chunk.strip().split()
        if len(words) < n:
            continue
        gram = " ".join(words[:n]).lower().strip(string.punctuation + " ")
        if gram:
            out[gram] += 1
    return out


def is_lexicon_gram(gram: str) -> bool:
    g = gram.lower()
    return any(g == m or g.startswith(m) or m.startswith(g) for m in _LEXICON)
