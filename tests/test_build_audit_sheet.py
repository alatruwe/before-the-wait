"""Tests for scripts/build_audit_sheet.py (handoff audit-sheet). Synthetic traces only."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bas", REPO / "scripts" / "build_audit_sheet.py")
bas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bas)

ONSET_COLS = ["dataset", "index", "token_index", "marker", "char_start", "prev_token_text",
              "prev_token_kind", "paragraph_initial", "tokens_since_prev_hit",
              "tokens_to_next_hit", "same_paragraph_as_prev_hit", "onset_mid_token",
              "merged_into"]
SECRET_INDEX, SECRET_TOKEN = 9871, 54321


def trace(paras: list[str], tail: str = "</think> The answer is \\boxed{77777}") -> dict:
    reasoning = "\n\n".join(paras)
    return {"index": SECRET_INDEX, "problem": "Tom has <3> apples & pears.",
            "output_text": reasoning + tail, "reasoning_char_span": [0, len(reasoning)],
            "gold_answer": "88888", "boxed_answer": "77777", "correct": True,
            "finish_reason": "eos"}


def onset_at(t: dict, para: int, word: str = "Wait") -> int:
    reasoning = t["output_text"][:t["reasoning_char_span"][1]]
    starts = [s for s, _ in bas.paragraphs(reasoning)]
    return starts[para] + t["output_text"][starts[para]:].index(word)


def paras_n(n: int, marker_at: int) -> list[str]:
    return [f"Para {k}. Wait, recheck." if k == marker_at else f"Para {k} text." for k in range(n)]


# ---------------------------------------------------------------- window

def test_window_middle():
    t = trace(paras_n(12, 7))
    w = bas.window(t, onset_at(t, 7), "wait")
    assert w["has_earlier"] and w["earlier"] == ["Para 0 text.", "Para 1 text."]
    assert w["before"] == [f"Para {k} text." for k in range(2, 7)]
    assert w["marker_para"] == ("Para 7. ", "Wait", ", recheck.")
    assert w["after"] == ["Para 8 text.", "Para 9 text."]
    assert not w["ends_inside"] and w["rest"] == ["Para 10 text.", "Para 11 text."]


def test_window_at_start_of_reasoning():
    t = trace(paras_n(6, 0))
    w = bas.window(t, onset_at(t, 0), "wait")
    assert not w["has_earlier"] and w["before"] == [] and w["earlier"] == []
    t = trace(paras_n(9, 5))  # exactly 5 before: no collapsed block
    w = bas.window(t, onset_at(t, 5), "wait")
    assert not w["has_earlier"] and len(w["before"]) == 5
    t = trace(paras_n(9, 6))  # 6 before: one collapsed
    w = bas.window(t, onset_at(t, 6), "wait")
    assert w["has_earlier"] and w["earlier"] == ["Para 0 text."]


def test_window_at_end_of_reasoning():
    t = trace(paras_n(4, 3))
    w = bas.window(t, onset_at(t, 3), "wait")
    assert w["after"] == [] and w["ends_inside"]
    t = trace(paras_n(5, 2))  # exactly 2 after: reasoning ends inside the shown text
    w = bas.window(t, onset_at(t, 2), "wait")
    assert len(w["after"]) == 2 and w["ends_inside"] and w["rest"] == []
    t = trace(paras_n(6, 2))
    assert not bas.window(t, onset_at(t, 2), "wait")["ends_inside"]


def test_marker_span_assertion():
    t = trace(paras_n(4, 1))
    with pytest.raises(AssertionError, match="marker span"):
        bas.window(t, onset_at(t, 1) + 1, "wait")


def test_paragraph_split_matches_same_paragraph_boundary():
    r = "a\n\nb\nc\n\n\nd"
    assert [p for _, p in bas.paragraphs(r)] == r.split("\n\n")
    for s, p in bas.paragraphs(r):
        assert r[s:s + len(p)] == p


# ---------------------------------------------------------------- HTML

def _html(practice: bool, n: int = 12, at: int = 7) -> str:
    t = trace(paras_n(n, at))
    t["problem"] = "Tom has <3> apples & pears. <script>x</script>"
    w = bas.window(t, onset_at(t, at), "wait")
    return bas.sheet_html("T", [("001", t["problem"], w)], practice)


def test_mark_only_this_span_and_escaping():
    h = _html(False)
    assert h.count("<mark>") == 1 and "<mark>Wait</mark>, recheck." in h
    assert "&lt;3&gt; apples &amp; pears." in h and "<script>" not in h
    assert "white-space: pre-wrap" in h and bas.esc(bas.QUESTION) in h


def test_hidden_fields_and_no_text_beyond_window():
    h = _html(False)
    for hidden in (str(SECRET_INDEX), str(SECRET_TOKEN), "77777", "88888", "boxed", "think",
                   "eos", "Para 10", "Para 11", "practice only"):
        assert hidden not in h, hidden
    assert "[reasoning continues]" in h and "Earlier reasoning" in h
    p = _html(True)
    assert "Rest of the reasoning (practice only)" in p and "Para 11 text." in p
    assert "</think>" not in p  # nothing after the reasoning span, even in practice


def test_start_and_end_lines():
    h = _html(False, n=3, at=1)
    assert "[start of reasoning]" in h and "[end of reasoning]" in h
    assert "Earlier reasoning" not in h


# ---------------------------------------------------------------- draw and files

def _fixture(tmp_path: Path, n_test: int = 6):
    traces = tmp_path / "traces"
    traces.mkdir()
    rows, split = [], {"train": [], "validation": [], "test": []}
    for i in range(10):
        sp = "test" if i < n_test else "train"
        split[sp].append(i)
        t = trace(paras_n(10, 4))
        t["index"] = i
        (traces / f"gsm8k_{i:04d}.json").write_text(json.dumps(t))
        cs = onset_at(t, 4)
        rows.append([("gsm8k", i, 30 + i, "wait", cs) + ("",) * 7 + ("",)])
        rows.append([("gsm8k", i, 99, "wait", cs) + ("",) * 7 + ("5",)])  # merged: excluded
    onsets = tmp_path / "onsets.csv"
    with open(onsets, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ONSET_COLS)
        for r in rows:
            w.writerow(list(r[0]))
    sj = tmp_path / "split.json"
    sj.write_text(json.dumps(split))
    dec = tmp_path / "decisions.md"
    dec.write_text("| D22 | x |\n| D23 | audit |\n")
    return onsets, sj, traces, dec


def test_population_and_draw_reproduce(tmp_path, monkeypatch):
    onsets, sj, traces, dec = _fixture(tmp_path)
    pop = bas.population(onsets, sj, "test")
    assert [(int(r["index"]), int(r["token_index"])) for r in pop] == [(i, 30 + i) for i in range(6)]
    monkeypatch.setattr(bas, "TEST_N", 4)
    out = bas.run_default(onsets, sj, traces, tmp_path / "audit", dec, expected_population=6)
    meta = json.loads((out / "draw.json").read_text())
    want = random.Random(20261020).sample(pop, 4)
    assert [(x["index"], x["token_index"]) for x in meta["items"]] == \
        [(int(r["index"]), int(r["token_index"])) for r in want]
    assert [x["item"] for x in meta["items"]] == ["001", "002", "003", "004"]
    lab = (out / "labels.csv").read_text().splitlines()
    assert lab[0] == "item,label,flag,note" and lab[1:] == ["001,,,", "002,,,", "003,,,", "004,,,"]


def test_default_rerun_byte_identical_and_guards(tmp_path, monkeypatch):
    onsets, sj, traces, dec = _fixture(tmp_path)
    monkeypatch.setattr(bas, "TEST_N", 4)
    a = tmp_path / "audit"
    sha = lambda: {f: hashlib.sha256((a / f).read_bytes()).hexdigest()  # noqa: E731
                   for f in ("draw.json", "sheet.html", "labels.csv")}
    bas.run_default(onsets, sj, traces, a, dec, expected_population=6)
    first = sha()
    bas.run_default(onsets, sj, traces, a, dec, expected_population=6)
    assert sha() == first
    with pytest.raises(AssertionError, match="population"):
        bas.run_default(onsets, sj, traces, a, dec, expected_population=806)
    nod23 = tmp_path / "d.md"
    nod23.write_text("| D22 | x |\n")
    with pytest.raises(AssertionError, match="D23"):
        bas.run_default(onsets, sj, traces, a, nod23, expected_population=6)
    (a / "labels.csv").write_text("item,label,flag,note\n001,real,,\n")
    with pytest.raises(AssertionError, match="already has labels"):
        bas.run_default(onsets, sj, traces, a, dec, expected_population=6)


def test_practice_rerun_byte_identical(tmp_path):
    onsets, sj, traces, _ = _fixture(tmp_path, n_test=2)  # 8 train traces
    out = tmp_path / "p.html"
    bas.run_practice(onsets, sj, traces, out)
    first = out.read_bytes()
    bas.run_practice(onsets, sj, traces, out)
    assert out.read_bytes() == first and first.count(b"<mark>") == 5
    assert b'class="num">P1<' in first and b'class="num">P5<' in first
