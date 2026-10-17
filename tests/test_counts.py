"""Tests for the pre-registration count rules (§4, §6, §11)."""
from __future__ import annotations

import pytest

from before_the_wait.counts import (
    apply_margin,
    hanley_mcneil_se,
    marker_token_span,
    minimum_events,
    projected_validation_onsets,
    trace_count,
    z_for_interval,
)


def test_hanley_mcneil_known_value():
    """AUROC 0.70 with 100 positives and 100 negatives: SE ~= 0.036."""
    se = hanley_mcneil_se(0.70, 100, 100)
    assert se == pytest.approx(0.0368, abs=0.0005)
    assert round(se, 3) == 0.037


def test_hanley_mcneil_shrinks_with_n():
    assert hanley_mcneil_se(0.70, 400, 400) < hanley_mcneil_se(0.70, 100, 100)


def test_hanley_mcneil_perfect_auc_has_zero_se():
    assert hanley_mcneil_se(1.0, 50, 50) == pytest.approx(0.0, abs=1e-12)


def test_hanley_mcneil_halving_n_scales_se():
    """SE scales roughly as 1/sqrt(n) once n is large."""
    a = hanley_mcneil_se(0.70, 1000, 1000)
    b = hanley_mcneil_se(0.70, 4000, 4000)
    assert a / b == pytest.approx(2.0, rel=0.02)


def test_z_for_98_75_percent():
    assert z_for_interval(0.9875) == pytest.approx(2.498, abs=0.001)


def test_minimum_events_satisfies_its_own_condition():
    for delta in (0.03, 0.05, 0.08):
        m = minimum_events(delta)
        assert z_for_interval() * hanley_mcneil_se(0.70, m, m) <= delta
        # ...and m-1 does not, so it really is the smallest.
        assert z_for_interval() * hanley_mcneil_se(0.70, m - 1, m - 1) > delta


def test_minimum_events_monotone_in_delta():
    assert minimum_events(0.03) > minimum_events(0.05) > minimum_events(0.08)


def test_trace_count_rounds_up():
    # 100 / (0.35 * 0.5 * 4 * 0.5) = 285.7 -> 286
    assert trace_count(100, 0.35, 0.5, 4.0, 0.5) == 286


def test_trace_count_satisfies_the_inequality():
    n = trace_count(250, 0.35, 5 / 30, 114 / 13, 0.8)
    assert n * 0.35 * (1 - 5 / 30) * (114 / 13) * 0.8 >= 250


def test_apply_margin_the_preregistered_case():
    """D12: 157 -> ceil(1.2*157)=189 -> rounded up to ten = 190."""
    assert apply_margin(157) == 190


def test_apply_margin_leaves_an_exact_multiple_alone():
    # 150 * 1.2 = 180 exactly, already a multiple of ten.
    assert apply_margin(150) == 180


def test_apply_margin_rounds_up_not_to_nearest():
    # 101 * 1.2 = 121.2 -> 122 -> 130, not 120.
    assert apply_margin(101) == 130


def test_apply_margin_is_monotone():
    values = [apply_margin(n) for n in range(100, 200)]
    assert values == sorted(values)


def test_apply_margin_never_shrinks():
    for n in (1, 7, 100, 157, 999):
        assert apply_margin(n) >= n


def test_projected_validation_onsets():
    assert projected_validation_onsets(100, 0.0, 10.0, 0.15) == pytest.approx(150.0)


class FakeTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.pieces[i] for i in ids)


def test_marker_token_span_single_token():
    tok = FakeTokenizer([" Wait", ","])
    assert marker_token_span(tok, [0, 1], 0, "wait") == (0, 1)


def test_marker_token_span_multi_token():
    tok = FakeTokenizer([" hold", " on", " there"])
    assert marker_token_span(tok, [0, 1, 2], 0, "hold on") == (0, 2)


def test_marker_token_span_stops_at_sequence_end():
    tok = FakeTokenizer([" wa"])
    start, end = marker_token_span(tok, [0], 0, "wait")
    assert (start, end) == (0, 1)
