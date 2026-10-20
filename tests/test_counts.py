"""Tests for the pre-registration count rules (§4, §6, §11)."""
from __future__ import annotations

import pytest

from before_the_wait.counts import (
    apply_margin,
    band_positives,
    classify_band_position,
    gap_to_previous_hit,
    hanley_mcneil_se,
    lift_se,
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


def test_lift_se_at_rho_half_is_one_auroc_se():
    assert lift_se(0.70, 300, 300, rho=0.5) == pytest.approx(hanley_mcneil_se(0.70, 300, 300))


def test_lift_se_at_rho_030():
    """2 * (1 - 0.3) = 1.4, so SE_lift = sqrt(1.4) * SE."""
    se = hanley_mcneil_se(0.60, 500, 500)
    assert lift_se(0.60, 500, 500, rho=0.30) == pytest.approx(1.4 ** 0.5 * se)


def test_minimum_events_satisfies_its_own_condition():
    for delta in (0.03, 0.05, 0.08):
        m = minimum_events(delta)
        assert z_for_interval() * lift_se(0.60, m, m, 0.30) <= delta
        # ...and m-1 does not, so it really is the smallest.
        assert z_for_interval() * lift_se(0.60, m - 1, m - 1, 0.30) > delta


def test_minimum_events_old_assumptions_reproduce_338():
    """The 2026-10-17 run: AUROC 0.70, rho 0.5, delta 0.05 -> m = 338."""
    assert minimum_events(0.05, auc=0.70, rho=0.5) == 338


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


# --- §5/§6 band-positive rule (2026-10-19) ---------------------------------
#
# One trace: onset A at 10 opens a paragraph, a merged hit at 20 sits in the
# same paragraph, and onset B at 30 opens the next one. Every hit spans one
# token.
ONSET_A, MERGED, ONSET_B = 10, 20, 30
HITS = [ONSET_A, MERGED, ONSET_B]
SPANS = [(h, h + 1) for h in HITS]


def test_positions_before_a_merged_hit_do_not_count_for_the_next_onset():
    for pos in range(ONSET_A + 1, MERGED):
        assert classify_band_position(pos, ONSET_B, HITS, SPANS, 0) == "hit_between"
    assert classify_band_position(MERGED, ONSET_B, HITS, SPANS, 0) == "in_marker"
    for pos in range(MERGED + 1, ONSET_B):
        assert classify_band_position(pos, ONSET_B, HITS, SPANS, 0) is None


def test_band_positives_for_the_second_onset():
    assert band_positives(ONSET_B, 2, 5, HITS, SPANS, 0) == [28, 27, 26, 25]
    # 20 is the merged hit itself; 21-24 are after it.
    assert band_positives(ONSET_B, 6, 10, HITS, SPANS, 0) == [24, 23, 22, 21]
    # Everything 11-20 back lies at or before the merged hit.
    assert band_positives(ONSET_B, 11, 20, HITS, SPANS, 0) == []


def test_each_position_belongs_to_at_most_one_onset():
    """Position 5 is before onset A, so it can only be A's, never B's."""
    assert classify_band_position(5, ONSET_A, HITS, SPANS, 0) is None
    assert classify_band_position(5, ONSET_B, HITS, SPANS, 0) == "hit_between"
    owners = [o for o in (ONSET_A, ONSET_B)
              if classify_band_position(5, o, HITS, SPANS, 0) is None]
    assert owners == [ONSET_A]


def test_positions_before_the_reasoning_start_are_excluded():
    assert classify_band_position(3, ONSET_A, HITS, SPANS, 4) == "before_reasoning"
    assert band_positives(ONSET_A, 2, 9, HITS, SPANS, 4) == [8, 7, 6, 5, 4]


def test_gap_counts_merged_hits():
    assert gap_to_previous_hit(ONSET_B, HITS, 0) == ONSET_B - MERGED
    assert gap_to_previous_hit(ONSET_A, HITS, 0) == ONSET_A
    assert gap_to_previous_hit(ONSET_A, HITS, 4) == ONSET_A - 4
