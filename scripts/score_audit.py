"""Score the hand audit (pre-registration §12; handoff audit-sheet, stage 3).

Reads only audit/labels.csv and audit/draw.json. Validates the labels, then writes
results/audit.md: precision (real / 100) with a 95% percentile bootstrap interval over
whole problems (the distinct trace indices among the 100 items; 10,000 resamples,
numpy default_rng(20261020)). Written and tested on synthetic labels before any real
label exists; nothing about scoring is decided after labelling.

    .venv/bin/python scripts/score_audit.py
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = REPO_ROOT / "audit" / "labels.csv"
DRAW = REPO_ROOT / "audit" / "draw.json"
OUT = REPO_ROOT / "results" / "audit.md"

N_ITEMS = 100
SEED = 20261020
N_BOOT = 10_000
PCT = (2.5, 97.5)
ADDON_MIN = 20          # §12: the add-on runs only if both groups have at least 20 hits
LABEL_VALUES = {"real", "false"}
FLAG_VALUES = {"", "unsure", "mishit"}


class ValidationError(Exception):
    def __init__(self, problems: dict[str, list]):
        self.problems = problems
        super().__init__("; ".join(f"{k}: {v}" for k, v in problems.items()))


def norm(s) -> str:
    return (s or "").strip().lower()


def validate(rows: list[dict], draw_items: list[int]) -> list[dict]:
    """Returns rows as {item:int, label, flag} sorted by item; raises ValidationError
    listing the offending items by number."""
    problems: dict[str, list] = {}
    parsed, seen = [], {}
    for r in rows:
        raw = (r.get("item") or "").strip()
        try:
            item = int(raw)
        except ValueError:
            problems.setdefault("item not an integer", []).append(raw)
            continue
        seen[item] = seen.get(item, 0) + 1
        parsed.append({"item": item, "label": norm(r.get("label")), "flag": norm(r.get("flag"))})
    if len(rows) != N_ITEMS:
        problems["row count"] = [len(rows)]
    expected = set(draw_items)
    if expected != set(range(1, N_ITEMS + 1)):
        problems["draw.json items not 1-100"] = sorted(expected ^ set(range(1, N_ITEMS + 1)))
    dup = sorted(k for k, v in seen.items() if v > 1)
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    if dup:
        problems["duplicate items"] = dup
    if missing:
        problems["missing items"] = missing
    if extra:
        problems["items not in draw.json"] = extra
    bad_label = sorted({p["item"] for p in parsed if p["label"] not in LABEL_VALUES})
    bad_flag = sorted({p["item"] for p in parsed if p["flag"] not in FLAG_VALUES})
    mishit_real = sorted({p["item"] for p in parsed
                          if p["flag"] == "mishit" and p["label"] != "false"})
    if bad_label:
        problems["label not real/false"] = bad_label
    if bad_flag:
        problems["flag not empty/unsure/mishit"] = bad_flag
    if mishit_real:
        problems["mishit not labelled false"] = mishit_real
    if problems:
        raise ValidationError(problems)
    return sorted(parsed, key=lambda p: p["item"])


def bootstrap(real: np.ndarray, unit: np.ndarray, n: int = N_BOOT, seed: int = SEED):
    """Percentile interval of precision, resampling whole problems with replacement.
    real: 0/1 per item; unit: problem id per item. Draws one (n, n_units) matrix of unit
    positions with default_rng(seed).integers."""
    keys = sorted(set(unit.tolist()))
    pos = {k: j for j, k in enumerate(keys)}
    u = np.array([pos[x] for x in unit.tolist()])
    real_per = np.bincount(u, weights=real, minlength=len(keys))
    items_per = np.bincount(u, minlength=len(keys)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(n, len(keys)))
    prec = real_per[draws].sum(axis=1) / items_per[draws].sum(axis=1)
    lo, hi = np.percentile(prec, list(PCT))
    return float(lo), float(hi), prec


def score(labels_path: Path = LABELS, draw_path: Path = DRAW, out: Path = OUT,
          today: str | None = None) -> dict:
    with open(labels_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    meta = json.loads(draw_path.read_text())
    draw_items = [int(x["item"]) for x in meta["items"]]
    index_of = {int(x["item"]): int(x["index"]) for x in meta["items"]}
    v = validate(rows, draw_items)
    real = np.array([1.0 if p["label"] == "real" else 0.0 for p in v])
    unit = np.array([index_of[p["item"]] for p in v])
    lo, hi, _ = bootstrap(real, unit)
    n_real = int(real.sum())
    res = {"sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
           "real": n_real, "false": N_ITEMS - n_real, "precision": n_real / N_ITEMS,
           "lo": lo, "hi": hi, "problems": len(set(unit.tolist())),
           "unsure": sum(p["flag"] == "unsure" for p in v),
           "mishit": sum(p["flag"] == "mishit" for p in v)}
    res["addon_runnable"] = res["real"] >= ADDON_MIN and res["false"] >= ADDON_MIN
    date = today or datetime.date.today().isoformat()
    lines = [
        "# Hand audit: precision of the lexicon onsets (§12)", "",
        "Computed by `scripts/score_audit.py` from `audit/labels.csv` and `audit/draw.json`.", "",
        "## Data contact", "",
        "| Date | What was touched | What was not |", "|---|---|---|",
        f"| {date} | `audit/labels.csv` (Adeline's hand labels of 100 test-split onsets) and "
        "`audit/draw.json`. | Any probe, rival, activation or score; the examples; any other "
        "split. |", "",
        "## Result", "",
        f"- sha256 of `audit/labels.csv`: `{res['sha256']}`",
        f"- Real: {res['real']}; false: {res['false']}",
        f"- Precision: {res['precision']:.4f}",
        f"- 95% interval: {lo:.4f}–{hi:.4f} (percentile bootstrap over whole problems, "
        f"{N_BOOT:,} resamples, `numpy.random.default_rng({SEED})`)",
        f"- Distinct problems: {res['problems']}",
        f"- Flagged `unsure`: {res['unsure']}",
        f"- Flagged `mishit`: {res['mishit']}",
        f"- Add-on (§12) runnable: {'yes' if res['addon_runnable'] else 'no'} "
        f"(needs at least {ADDON_MIN} real and {ADDON_MIN} false; "
        f"{res['real']} real, {res['false']} false)", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return res


def main() -> None:
    try:
        score()
    except ValidationError as e:
        for k, items in e.problems.items():
            print(f"{k}: {', '.join(map(str, items))}", file=sys.stderr)
        sys.exit(1)
    print(OUT.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
