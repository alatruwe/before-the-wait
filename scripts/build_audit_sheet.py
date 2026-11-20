"""Hand-audit sheet (pre-registration §12, D1, D23; handoff audit-sheet).

Draws onsets of the paragraph-rule onset table and writes a self-contained HTML sheet
for labelling by hand. Reads only results/run_onsets_paragraph.csv, results/split.json,
traces/run/gsm8k_*.json and (import only) the lexicon. Never reads probes/, rival/,
activations/ or examples/, and shows no score, distance, band or answer.

    .venv/bin/python scripts/build_audit_sheet.py --practice   # 5 train onsets, Random(1)
    .venv/bin/python scripts/build_audit_sheet.py              # 100 test onsets, Random(20261020)

Deterministic: rerunning either mode gives byte-identical files.
"""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import platform
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from before_the_wait import lexicon  # noqa: E402,F401  (import only, per the handoff)

ONSETS_CSV = REPO_ROOT / "results" / "run_onsets_paragraph.csv"
SPLIT_JSON = REPO_ROOT / "results" / "split.json"
TRACES_DIR = REPO_ROOT / "traces" / "run"
DECISIONS = REPO_ROOT / "methodology" / "decisions.md"
AUDIT_DIR = REPO_ROOT / "audit"
PRACTICE_OUT = REPO_ROOT / "handoffs" / "audit-practice-sheet.html"

N_BEFORE = 5        # paragraphs shown before the marker's paragraph
N_AFTER = 2         # paragraphs shown after it
TEST_SEED, TEST_N, TEST_POPULATION = 20261020, 100, 806
PRACTICE_SEED, PRACTICE_N = 1, 5
SEP = "\n\n"        # the boundary label.same_paragraph uses

QUESTION = (
    "Real: after the marker, the trace changes something it had stated (a value, a step, a "
    "reading of the problem). The change must be visible in the marker's paragraph or the two "
    "paragraphs after it. Compare with what the trace held just before the marker. A position it "
    "had already dropped earlier does not count. Mark real only if you can point to what "
    "changed. Otherwise false. A "
    "re-check that confirms the prior statement is false. A marker matched inside another word "
    "or not used as a reconsideration is false, flag `mishit`. Flag `unsure` when the forced "
    "label was a close call.")
POPULATION_RULE = (
    "rows of results/run_onsets_paragraph.csv with empty merged_into whose index is in the "
    "chosen split of results/split.json, sorted ascending by (int(index), int(token_index))")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- population and draw

def population(onsets_csv: Path, split_json: Path, split: str) -> list[dict]:
    keep = set(json.loads(split_json.read_text())[split])
    with open(onsets_csv, newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["merged_into"] == "" and int(r["index"]) in keep]
    return sorted(rows, key=lambda r: (int(r["index"]), int(r["token_index"])))


def draw(pop: list[dict], seed: int, n: int) -> list[dict]:
    return random.Random(seed).sample(pop, n)  # item order = sample order; no second shuffle


# ---------------------------------------------------------------- window

def paragraphs(reasoning: str) -> list[tuple[int, str]]:
    """(start offset in reasoning, text) for each piece of reasoning.split('\\n\\n')."""
    out, pos = [], 0
    for piece in reasoning.split(SEP):
        out.append((pos, piece))
        pos += len(piece) + len(SEP)
    return out


def window(trace: dict, char_start: int, marker: str) -> dict:
    """Split the reasoning around the onset. char_start is in output_text coordinates."""
    c0, c1 = trace["reasoning_char_span"]
    text = trace["output_text"]
    assert text[char_start:char_start + len(marker)].lower() == marker, "marker span mismatch"
    reasoning = text[c0:c1]
    rel = char_start - c0
    assert 0 <= rel and rel + len(marker) <= len(reasoning), "onset outside reasoning span"
    paras = paragraphs(reasoning)
    m = max(k for k, (s, _) in enumerate(paras) if s <= rel)
    start, ptext = paras[m]
    local = rel - start
    assert local + len(marker) <= len(ptext), "marker crosses a paragraph boundary"
    lo = max(0, m - N_BEFORE)
    return {"earlier": [p for _, p in paras[:lo]],
            "before": [p for _, p in paras[lo:m]],
            "marker_para": (ptext[:local], ptext[local:local + len(marker)],
                            ptext[local + len(marker):]),
            "after": [p for _, p in paras[m + 1:m + 1 + N_AFTER]],
            "rest": [p for _, p in paras[m + 1 + N_AFTER:]],
            "ends_inside": m + 1 + N_AFTER >= len(paras),
            "has_earlier": m > N_BEFORE}


# ---------------------------------------------------------------- HTML

CSS = """
body { font-family: Georgia, serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem;
       background: #fdfdfb; color: #1b1b1b; line-height: 1.45; }
.q { border-left: 4px solid #888; padding: .4rem .9rem; background: #f2f2ee; }
.item { border-top: 2px solid #444; margin-top: 2.5rem; padding-top: .8rem; }
.num { font-family: monospace; font-size: 1.3rem; font-weight: bold; }
.label { font-family: sans-serif; font-size: .8rem; text-transform: uppercase; color: #666;
         margin-top: 1rem; }
.para { white-space: pre-wrap; margin: 0 0 1.45em 0; font-family: Menlo, monospace;
        font-size: .9rem; }
.edge { font-family: sans-serif; font-style: italic; color: #666; margin: .6rem 0 1.2rem; }
mark { background: #ffe066; color: #000; padding: 0 .1em; }
details { margin: .6rem 0 1.2rem; }
summary { cursor: pointer; font-family: sans-serif; color: #444; }
"""


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def para_html(text: str) -> str:
    return f'<div class="para">{esc(text)}</div>'


def item_html(label: str, problem: str, w: dict, practice: bool) -> str:
    out = [f'<section class="item"><div class="num">{esc(label)}</div>',
           '<div class="label">Problem</div>', para_html(problem),
           '<div class="label">Reasoning</div>']
    if w["has_earlier"]:
        out.append("<details><summary>Earlier reasoning</summary>"
                   + "".join(para_html(p) for p in w["earlier"]) + "</details>")
    else:
        out.append('<div class="edge">[start of reasoning]</div>')
    out += [para_html(p) for p in w["before"]]
    a, mk, b = w["marker_para"]
    out.append(f'<div class="para">{esc(a)}<mark>{esc(mk)}</mark>{esc(b)}</div>')
    out += [para_html(p) for p in w["after"]]
    out.append('<div class="edge">'
               + ("[end of reasoning]" if w["ends_inside"] else "[reasoning continues]")
               + "</div>")
    if practice:
        out.append("<details><summary>Rest of the reasoning (practice only)</summary>"
                   + "".join(para_html(p) for p in w["rest"]) + "</details>")
    out.append("</section>")
    return "\n".join(out)


def sheet_html(title: str, items: list[tuple[str, str, dict]], practice: bool) -> str:
    body = "\n".join(item_html(lab, prob, w, practice) for lab, prob, w in items)
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body>\n"
            f"<h1>{esc(title)}</h1>\n<div class=\"q\">{esc(QUESTION)}</div>\n"
            f"{body}\n</body></html>\n")


# ---------------------------------------------------------------- build

def load_trace(traces_dir: Path, index: int) -> dict:
    return json.loads((traces_dir / f"gsm8k_{index:04d}.json").read_text())


def build_items(drawn: list[dict], traces_dir: Path, labels: list[str]):
    items = []
    for lab, r in zip(labels, drawn):
        t = load_trace(traces_dir, int(r["index"]))
        assert int(t["index"]) == int(r["index"])
        items.append((lab, t["problem"], window(t, int(r["char_start"]), r["marker"])))
    return items


def labels_csv(n: int) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["item", "label", "flag", "note"])
    for k in range(1, n + 1):
        w.writerow([f"{k:03d}", "", "", ""])
    return buf.getvalue().encode()


def labels_started(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path, newline="") as fh:
        return any((r.get("label") or "").strip() for r in csv.DictReader(fh))


def has_d23(decisions: Path) -> bool:
    return any(line.startswith("| D23 |") for line in decisions.read_text().splitlines())


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def run_practice(onsets_csv=ONSETS_CSV, split_json=SPLIT_JSON, traces_dir=TRACES_DIR,
                 out=PRACTICE_OUT) -> Path:
    pop = population(onsets_csv, split_json, "train")
    drawn = draw(pop, PRACTICE_SEED, PRACTICE_N)
    items = build_items(drawn, traces_dir, [f"P{k}" for k in range(1, PRACTICE_N + 1)])
    write(out, sheet_html("Hand audit: practice sheet (train split)", items, True).encode())
    return out


def run_default(onsets_csv=ONSETS_CSV, split_json=SPLIT_JSON, traces_dir=TRACES_DIR,
                audit_dir=AUDIT_DIR, decisions=DECISIONS,
                expected_population=TEST_POPULATION) -> Path:
    assert has_d23(decisions), "methodology/decisions.md has no D23 row: refusing to draw"
    assert not labels_started(audit_dir / "labels.csv"), \
        "audit/labels.csv already has labels: refusing to overwrite"
    pop = population(onsets_csv, split_json, "test")
    assert len(pop) == expected_population, f"population {len(pop)} != {expected_population}"
    drawn = draw(pop, TEST_SEED, TEST_N)
    keys = [(int(r["index"]), int(r["token_index"])) for r in drawn]
    assert len(set(keys)) == TEST_N
    items = build_items(drawn, traces_dir, [f"{k:03d}" for k in range(1, TEST_N + 1)])
    meta = {
        "seed": TEST_SEED,
        "population_rule": POPULATION_RULE + "; split = test",
        "sample_call": f"random.Random({TEST_SEED}).sample(population, {TEST_N})",
        "python_version": platform.python_version(),
        "population_count": len(pop),
        "sha256_run_onsets_paragraph_csv": sha256_file(onsets_csv),
        "sha256_split_json": sha256_file(split_json),
        "items": [{"item": f"{k:03d}", "index": i, "token_index": ti}
                  for k, (i, ti) in enumerate(keys, start=1)],
    }
    write(audit_dir / "draw.json", (json.dumps(meta, indent=1) + "\n").encode())
    write(audit_dir / "sheet.html", sheet_html("Hand audit (§12)", items, False).encode())
    write(audit_dir / "labels.csv", labels_csv(TEST_N))
    return audit_dir


def main() -> None:
    if "--practice" in sys.argv:
        print(run_practice().relative_to(REPO_ROOT))
    else:
        print(run_default().relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
