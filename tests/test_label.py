"""Labeling tests: token kinds, paragraph structure, loop detection, onsets."""
from __future__ import annotations

from before_the_wait.label import (
    LOOP_WINDOW,
    is_looped,
    is_paragraph_initial,
    label_trace,
    negative_pool,
    paragraph_initial_ngrams,
    same_paragraph,
    token_kind,
)


class FakeTokenizer:
    def __init__(self, pieces: list[str]):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(self.pieces[i] for i in ids)


def test_token_kind_newline_wins():
    assert token_kind("\n") == "newline"
    assert token_kind(".\n") == "newline"
    assert token_kind("\n\n") == "newline"


def test_token_kind_period_and_comma():
    assert token_kind(".") == "period"
    assert token_kind(" 1.") == "period"
    assert token_kind(",") == "comma"


def test_token_kind_other_punct():
    assert token_kind(")") == "other_punct"
    assert token_kind(" =") == "other_punct"


def test_token_kind_digit_and_word():
    assert token_kind(" 42") == "digit"
    assert token_kind(" the") == "word"
    assert token_kind("x2") == "word"


def test_token_kind_whitespace_is_other():
    assert token_kind("   ") == "other"
    assert token_kind("") == "other"


def test_paragraph_initial_true_after_blank_line():
    text = "first para\n\nWait, no"
    assert is_paragraph_initial(text, text.index("Wait")) is True


def test_paragraph_initial_false_mid_paragraph():
    text = "some text wait here"
    assert is_paragraph_initial(text, text.index("wait")) is False


def test_paragraph_initial_false_at_start_of_reasoning():
    # Literal reading of the handoff: paragraph-initial means after a \n\n.
    text = "wait at the very start"
    assert is_paragraph_initial(text, 0) is False


def test_same_paragraph():
    text = "a wait b\n\nc wait d"
    first = text.index("wait")
    second = text.index("wait", first + 1)
    assert same_paragraph(text, first, second) is False
    assert same_paragraph(text, 0, first) is True


def test_is_looped_detects_repeat():
    block = list(range(LOOP_WINDOW))
    assert is_looped(block * 3) is True


def test_is_looped_false_on_varied_text():
    assert is_looped(list(range(200))) is False


def test_is_looped_false_when_too_short():
    assert is_looped([1, 2, 3]) is False


def test_is_looped_needs_three_repeats():
    block = list(range(LOOP_WINDOW))
    # Two copies only.
    assert is_looped(block * 2) is False


def test_paragraph_initial_ngrams():
    text = "intro\n\nWait, this\n\nWait, that\n\nHowever it goes"
    unigrams = paragraph_initial_ngrams(text, 1)
    assert unigrams["wait"] == 2
    bigrams = paragraph_initial_ngrams(text, 2)
    assert bigrams["wait, this"] == 1


def test_label_trace_finds_and_maps_onset():
    pieces = ["Okay", ".", " Wait", ",", " that", " is", " wrong"]
    tok = FakeTokenizer(pieces)
    ids = list(range(len(pieces)))
    text = "".join(pieces)
    lens = [0]
    for k in range(1, len(ids) + 1):
        lens.append(len(tok.decode(ids[:k])))

    onsets, failures = label_trace(
        tok,
        dataset="test",
        index=0,
        ids=ids,
        output_text=text,
        prefix_len=lens,
        reasoning_char_span=(0, len(text)),
        reasoning_token_span=(0, len(ids)),
    )
    assert failures == []
    assert [o.marker for o in onsets] == ["wait"]
    assert onsets[0].token_index == 2
    assert onsets[0].prev_token_text == "."
    assert onsets[0].prev_token_kind == "period"


def test_negative_pool_counts_and_skips():
    pieces = [" a"] * 50
    tok = FakeTokenizer(pieces)
    ids = list(range(50))
    by_kind, skipped = negative_pool(tok, ids, (0, 50), onset_token_indices=set(), lookahead=10)
    # A full window at position t needs t+10 < 50, so t = 0..39 qualify and
    # the last 10 positions are skipped for insufficient lookahead.
    assert sum(by_kind.values()) == 40
    assert skipped == 10
    assert by_kind["word"] == 40


def test_negative_pool_excludes_positions_near_an_onset():
    pieces = [" a"] * 50
    tok = FakeTokenizer(pieces)
    ids = list(range(50))
    by_kind, _ = negative_pool(tok, ids, (0, 50), onset_token_indices={20}, lookahead=10)
    # Positions 10..19 see the onset at 20 within their lookahead, so 10 of
    # the 40 eligible positions drop out.
    assert sum(by_kind.values()) == 30
