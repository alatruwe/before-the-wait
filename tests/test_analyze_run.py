"""Tests for the counting helpers in scripts/analyze_run.py (handoff 08b).

The helpers that aren't straight reuse from src/: trace and onset tallies, the
CSV-row gap, the per-band table and flag, and the progress.log parser. The
pilot check reproduces the e_b values in results/count_calculations.md from
the committed c1 CSV alone.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("analyze_run", REPO / "scripts" / "analyze_run.py")
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)


def _row(finish="eos", looped=False, correct=True):
    return {"finish_reason": finish, "looped": looped, "correct": correct}


def test_trace_tally_separates_truncated_from_looped_capped():
    rows = [
        _row(),                                   # kept
        _row(correct=False),                      # kept, wrong
        _row("length"),                           # truncated (kept)
        _row("length", looped=True),              # looped and capped
        _row(looped=True),                        # looped, not capped
    ]
    t = ar.trace_tally(rows)
    assert t["n"] == 5 and t["kept"] == 3 and t["looped"] == 2
    assert t["truncated"] == 1
    assert t["looped_capped"] == 1 and t["looped_not_capped"] == 1
    assert t["loop_share"] == pytest.approx(0.4)
    assert t["correct_kept"] == 2 and t["accuracy_kept"] == pytest.approx(2 / 3)


def test_trace_tally_reads_csv_strings():
    rows = [{"finish_reason": "length", "looped": "False", "correct": "True"},
            {"finish_reason": "eos", "looped": "True", "correct": "False"}]
    t = ar.trace_tally(rows)
    assert (t["kept"], t["looped"], t["truncated"], t["correct_kept"]) == (1, 1, 1, 1)


def test_onset_tally_counts_only_unmerged_rows():
    rows = [
        {"marker": "wait", "merged_into": ""},
        {"marker": "hmm", "merged_into": 10},
        {"marker": "hmm", "merged_into": ""},
        {"marker": "wait", "merged_into": ""},
    ]
    o = ar.onset_tally(rows, n_kept=2)
    assert o["hits"] == 4 and o["onsets"] == 3
    assert o["o"] == pytest.approx(1.5)
    assert o["wait_onsets"] == 2 and o["wait_share"] == pytest.approx(2 / 3)


def test_gap_from_onset_row_first_hit_uses_reasoning_start():
    assert ar.gap_from_onset_row({"tokens_since_prev_hit": "", "token_index": "37"}, 0) == 37
    assert ar.gap_from_onset_row({"tokens_since_prev_hit": None, "token_index": 37}, 5) == 32
    assert ar.gap_from_onset_row({"tokens_since_prev_hit": "12", "token_index": "99"}, 0) == 12


def test_band_table_strict_bound_and_flag():
    # gap > upper bound is eligible; equal to the bound is not.
    gaps = [5, 6, 10, 11, 20, 21, 40, 41]
    t = {b["band"]: b for b in ar.band_table(gaps, test_share=0.5, m=2)}
    assert [t[b]["eligible"] for b in ("2-5", "6-10", "11-20", "21-40")] == [7, 5, 3, 1]
    assert t["2-5"]["e_b"] == pytest.approx(7 / 8)
    assert t["21-40"]["projected"] == pytest.approx(0.5)
    assert t["11-20"]["projected"] == pytest.approx(1.5)
    # projected 3.5, 2.5 >= m = 2; 1.5, 0.5 < m.
    assert [t[b]["flag"] for b in ("2-5", "6-10", "11-20", "21-40")] == [False, False, True, True]


def test_band_table_flag_threshold_is_strict_less_than():
    t = ar.band_table([100] * 4, test_share=0.5, m=2)  # projected exactly 2.0 = m
    assert all(b["flag"] is False for b in t)


def test_progress_facts_parses_log_and_wall_time():
    log = ("i\tindex\tseed\tnew_tokens\tseconds\ttok_per_s\tfinish_reason\treasoning_complete\n"
           "0\t415\t20261016\t100\t1.5\t66.7\teos\tTrue\n"
           "1\t681\t20261017\t4096\t60.0\t68.3\tlength\tFalse\n")
    pf = ar.progress_facts(log, "...\n\ndone; total 77s\n")
    assert pf["rows"] == 2 and pf["tokens"] == 4196
    assert pf["gen_seconds"] == pytest.approx(61.5)
    assert pf["finish"] == {"eos": 1, "length": 1}
    assert pf["reasoning_complete"] == {"True": 1, "False": 1}
    assert pf["wall_seconds"] == 77
    assert ar.progress_facts(log, "no closing line")["wall_seconds"] is None


def test_pilot_e_b_reproduced_from_c1_csv():
    """The CSV path gives the pilot GSM8K e_b in results/count_calculations.md."""
    rows = [r for r in csv.DictReader((REPO / "results" / "c1_onsets_paragraph.csv").open())
            if r["dataset"] == "gsm8k" and r["merged_into"] == ""]
    gaps = []
    for r in rows:
        t = json.loads((REPO / "traces" / "pilot" / f"gsm8k_{int(r['index']):04d}.json").read_text())
        gaps.append(ar.gap_from_onset_row(r, t["reasoning_token_span"][0]))
    e_b = {b["band"]: round(b["e_b"], 4) for b in ar.band_table(gaps)}
    assert len(gaps) == 114
    assert e_b == ar.PILOT["e_b"]
