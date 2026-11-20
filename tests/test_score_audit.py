"""Tests for scripts/score_audit.py (handoff audit-sheet). Synthetic labels only."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sa", REPO / "scripts" / "score_audit.py")
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)


def rows(n=100, label="real", flag=""):
    return [{"item": f"{k:03d}", "label": label, "flag": flag, "note": ""} for k in range(1, n + 1)]


ITEMS = list(range(1, 101))


def problems_of(fn):
    with pytest.raises(sa.ValidationError) as e:
        fn()
    return e.value.problems


def test_valid_and_normalization():
    r = rows()
    r[0].update(item="1", label=" REAL ", flag=" Unsure")
    r[1].update(label="False", flag="MISHIT")
    v = sa.validate(r, ITEMS)
    assert v[0] == {"item": 1, "label": "real", "flag": "unsure"}
    assert v[1]["label"] == "false" and v[1]["flag"] == "mishit"


def test_each_validation_failure():
    assert problems_of(lambda: sa.validate(rows(99), ITEMS))["missing items"] == [100]
    r = rows()
    r[4]["item"] = "004"
    p = problems_of(lambda: sa.validate(r, ITEMS))
    assert p["duplicate items"] == [4] and p["missing items"] == [5]
    r = rows() + [{"item": "101", "label": "real", "flag": "", "note": ""}]
    p = problems_of(lambda: sa.validate(r, ITEMS))
    assert p["items not in draw.json"] == [101] and p["row count"] == [101]
    r = rows()
    r[2]["item"] = "x3"
    assert "x3" in problems_of(lambda: sa.validate(r, ITEMS))["item not an integer"]
    r = rows()
    r[6]["label"] = ""
    r[7]["label"] = "maybe"
    assert problems_of(lambda: sa.validate(r, ITEMS))["label not real/false"] == [7, 8]
    r = rows()
    r[9]["flag"] = "odd"
    assert problems_of(lambda: sa.validate(r, ITEMS))["flag not empty/unsure/mishit"] == [10]
    r = rows()
    r[11]["flag"] = "mishit"
    assert problems_of(lambda: sa.validate(r, ITEMS))["mishit not labelled false"] == [12]
    assert "draw.json items not 1-100" in problems_of(lambda: sa.validate(rows(), ITEMS[:-1] + [200]))


def test_bootstrap_hand_checkable():
    # two problems: A has 2 real of 2, B has 0 real of 2. Possible resamples: AA 1.0, AB 0.5, BB 0.0
    real = np.array([1.0, 1.0, 0.0, 0.0])
    unit = np.array([7, 7, 9, 9])
    lo, hi, prec = sa.bootstrap(real, unit, n=4000, seed=1)
    assert set(np.unique(prec)) <= {0.0, 0.5, 1.0}
    assert lo == 0.0 and hi == 1.0
    assert abs(np.mean(prec == 0.5) - 0.5) < 0.05
    # unequal sizes: precision is items-weighted, not a mean of problem shares
    real = np.array([1.0, 0.0, 0.0, 1.0])
    unit = np.array([1, 1, 1, 2])
    _, _, prec = sa.bootstrap(real, unit, n=2000, seed=2)
    assert set(np.round(np.unique(prec), 6)) <= {round(1 / 3, 6), 0.5, 1.0}
    rng = np.random.default_rng(2)
    d = rng.integers(0, 2, size=(2000, 2))
    rp, ip = np.array([1.0, 1.0]), np.array([3.0, 1.0])
    assert np.array_equal(prec, rp[d].sum(1) / ip[d].sum(1))


def test_score_end_to_end_and_no_write_on_failure(tmp_path):
    draw = {"items": [{"item": f"{k:03d}", "index": k // 2, "token_index": k} for k in ITEMS]}
    dp = tmp_path / "draw.json"
    dp.write_text(json.dumps(draw))
    lp = tmp_path / "labels.csv"
    r = rows(label="false")
    for k in range(25):
        r[k]["label"] = "real"
    r[50]["flag"] = "unsure"
    r[60]["flag"] = "mishit"
    with open(lp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item", "label", "flag", "note"])
        w.writeheader()
        w.writerows(r)
    out = tmp_path / "audit.md"
    res = sa.score(lp, dp, out, today="2026-01-01")
    assert (res["real"], res["false"], res["precision"]) == (25, 75, 0.25)
    assert res["problems"] == 51 and res["unsure"] == 1 and res["mishit"] == 1
    assert res["addon_runnable"] and res["lo"] < 0.25 < res["hi"]
    md = out.read_text()
    assert "Precision: 0.2500" in md and "Add-on (§12) runnable: yes" in md
    assert out.read_text() == (sa.score(lp, dp, out, today="2026-01-01") and out.read_text())
    out.unlink()
    r[0]["label"] = ""
    with open(lp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["item", "label", "flag", "note"])
        w.writeheader()
        w.writerows(r)
    with pytest.raises(sa.ValidationError):
        sa.score(lp, dp, out)
    assert not out.exists()
