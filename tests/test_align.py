"""Alignment tests: char offsets -> generated token indices.

A fake tokenizer stands in for the real one so these run without the model.
Its decode is plain concatenation, which is the property the real byte-level
BPE decode has for the prefixes we rely on.
"""
from __future__ import annotations

import pytest

from before_the_wait.align import (
    AlignmentError,
    check_prefix_consistency,
    hard_check,
    is_mid_token_onset,
    prefix_lengths,
    token_index_for_char,
)


class FakeTokenizer:
    def __init__(self, pieces: list[str]):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self.pieces[i] for i in ids)


# "Okay. Wait, that is wrong" split the way a BPE tokenizer would.
PIECES = ["Okay", ".", " Wait", ",", " that", " is", " wrong"]
IDS = list(range(len(PIECES)))
TEXT = "".join(PIECES)


@pytest.fixture
def tok():
    return FakeTokenizer(PIECES)


def test_prefix_lengths_are_cumulative(tok):
    lens = prefix_lengths(tok, IDS)
    assert lens[0] == 0
    assert lens[-1] == len(TEXT)
    assert lens == sorted(lens)


def test_prefix_lengths_match_each_token(tok):
    lens = prefix_lengths(tok, IDS)
    for t, piece in enumerate(PIECES):
        assert TEXT[lens[t] : lens[t + 1]] == piece


def test_check_prefix_consistency_clean(tok):
    assert check_prefix_consistency(tok, IDS, TEXT) == []


def test_token_index_for_char_finds_covering_token(tok):
    lens = prefix_lengths(tok, IDS)
    # " Wait" is token 2; the W is one char into it (the leading space).
    w = TEXT.index("Wait")
    assert token_index_for_char(lens, w) == 2


def test_token_index_for_char_at_boundaries(tok):
    lens = prefix_lengths(tok, IDS)
    for t in range(len(PIECES)):
        assert token_index_for_char(lens, lens[t]) == t


def test_hard_check_passes_on_correct_token(tok):
    # Token 2 decodes to " Wait" -> lstrip -> "Wait" starts with "wait".
    assert hard_check(tok, IDS, 2, "wait") is True


def test_hard_check_is_case_insensitive(tok):
    assert hard_check(tok, IDS, 2, "WAIT") is True


def test_hard_check_fails_on_wrong_token(tok):
    """The negative case: a guard that only ever returns True proves nothing."""
    assert hard_check(tok, IDS, 0, "wait") is False
    assert hard_check(tok, IDS, 3, "wait") is False
    assert hard_check(tok, IDS, 4, "wait") is False


def test_mid_token_onset_detection(tok):
    lens = prefix_lengths(tok, IDS)
    w = TEXT.index("Wait")
    # The marker starts one char into token 2, so this is a mid-token onset by
    # the strict boundary test, even though the hard check passes.
    assert is_mid_token_onset(lens, 2, w) is True
    assert is_mid_token_onset(lens, 2, lens[2]) is False


def test_prefix_lengths_rejects_shrinking_decode():
    class ShrinkingTokenizer:
        def decode(self, ids, skip_special_tokens: bool = False) -> str:
            return "xxx" if len(ids) < 2 else "y"

    with pytest.raises(AlignmentError):
        prefix_lengths(ShrinkingTokenizer(), [0, 1])


def test_check_prefix_consistency_reports_bad_prefix(tok):
    assert check_prefix_consistency(tok, IDS, "completely different text") != []
