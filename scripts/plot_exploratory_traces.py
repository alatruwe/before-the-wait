"""Step 14, analysis 3: per-trace plots (pre-registration §13 item 3, D25 X4). EXPLORATORY.

10 test problems drawn with random.Random(20261021).sample(sorted(test problems), 10).
For each, a teacher-forced pass of the 1.5B over the stored token ids (step 10's
extract_trace, unchanged), and the four frozen band probes scored at every reasoning token
(decision function, each at its own frozen layer). Check: at every example position the
trace has in activations/run/, the new score equals the score from the saved activation
(tolerance 1e-3, plan Q8). One self-contained HTML file per trace. No verdict.

    .venv/bin/python scripts/plot_exploratory_traces.py
"""
from __future__ import annotations

import csv
import html
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

import extract_activations as ea  # noqa: E402  (sets HF_HOME, offline)
import run_exploratory as rx  # noqa: E402
import run_gate as rg  # noqa: E402
import run_probes as rp  # noqa: E402
import run_verdicts as rv  # noqa: E402

SEED = 20261021                     # §18, per-trace plots
N_TRACES = 10
TOLERANCE = 1e-3                    # plan Q8: max |Δ decision function| allowed
TRACE_OUT = rx.OUT / "traces"
COLORS = {"2-5": "#1f77b4", "6-10": "#d62728", "11-20": "#2ca02c", "21-40": "#9467bd"}


def draw_traces(test_ids) -> list[int]:
    return random.Random(SEED).sample(sorted(test_ids), N_TRACES)


def onsets_for(tr: int) -> tuple[list[int], list[int]]:
    """(paragraph-rule onsets, merged hits) as generated-token indices."""
    on, merged = [], []
    with open(rx.ONSETS_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["index"]) == tr:
                (on if r["merged_into"] == "" else merged).append(int(r["token_index"]))
    return sorted(on), sorted(merged)


# ---------------------------------------------------------------- HTML

def svg_plot(t0: int, scores: dict[str, np.ndarray], onsets: list[int], layers: dict) -> str:
    W, H, ml, mr, mt, mb = 1100, 340, 56, 14, 14, 40
    n = len(next(iter(scores.values())))
    allv = np.concatenate(list(scores.values()))
    lo, hi = float(np.min(allv)), float(np.max(allv))
    pad = 0.05 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad
    x1 = t0 + max(n - 1, 1)

    def X(t):
        return ml + (t - t0) / (x1 - t0) * (W - ml - mr)

    def Y(v):
        return mt + (hi - v) / (hi - lo) * (H - mt - mb)

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="Exploratory: four frozen band probe scores by token">']
    parts.append(f'<rect x="{ml}" y="{mt}" width="{W - ml - mr}" height="{H - mt - mb}" '
                 'fill="#fff" stroke="#bbb"/>')
    for t in onsets:
        parts.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{mt}" y2="{H - mb}" '
                     'stroke="#f0a000" stroke-width="1"><title>onset t=' + str(t) + '</title></line>')
    if lo < 0 < hi:
        parts.append(f'<line x1="{ml}" x2="{W - mr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" '
                     'stroke="#888" stroke-dasharray="3,3"/>')
    for k in range(6):
        v = lo + (hi - lo) * k / 5
        parts.append(f'<text x="{ml - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" '
                     f'font-size="11">{v:.1f}</text>')
    for k in range(9):
        t = t0 + round((x1 - t0) * k / 8)
        parts.append(f'<text x="{X(t):.1f}" y="{H - mb + 16}" text-anchor="middle" '
                     f'font-size="11">{t}</text>')
    parts.append(f'<text x="{(W + ml) / 2:.0f}" y="{H - 6}" text-anchor="middle" '
                 'font-size="12">token index t (generated tokens, reasoning span)</text>')
    for b, s in scores.items():
        pts = " ".join(f"{X(t0 + i):.1f},{Y(v):.1f}" for i, v in enumerate(s))
        parts.append(f'<polyline fill="none" stroke="{COLORS[b]}" stroke-width="1" '
                     f'points="{pts}"><title>band {b} probe, layer {layers[b]}</title></polyline>')
    parts.append("</svg>")
    return "".join(parts)


def text_html(pieces: list[str], t0: int, onsets: set[int], merged: set[int]) -> str:
    out = []
    for i, p in enumerate(pieces):
        t = t0 + i
        e = html.escape(p)
        if t in onsets:
            out.append(f'<mark title="onset, t={t}">{e}</mark>')
        elif t in merged:
            out.append(f'<span class="merged" title="merged hit, t={t}">{e}</span>')
        else:
            out.append(e)
    return "".join(out)


def page(k: int, tr: int, problem: str, svg: str, body: str, layers: dict,
         n_onsets: int, check: str) -> str:
    legend = " ".join(f'<span class="key"><span class="sw" style="background:{COLORS[b]}">'
                      f'</span>band {b} (layer {layers[b]})</span>' for b in rp.BANDS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Exploratory trace {k:02d} (test problem {tr})</title>
<style>
body{{margin:0;padding:16px;background:#fafafa;color:#222;font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1150px;margin:0 auto}}
h1{{font-size:18px;margin:0 0 4px}} p.note{{color:#555;margin:4px 0 12px}}
.key{{margin-right:14px;white-space:nowrap}} .sw{{display:inline-block;width:14px;height:3px;
vertical-align:middle;margin-right:4px}}
.box{{background:#fff;border:1px solid #ddd;padding:10px 12px;margin:10px 0;overflow-x:auto}}
pre{{white-space:pre-wrap;word-wrap:break-word;font:13px/1.5 ui-monospace,Menlo,monospace;margin:0}}
mark{{background:#ffd27a;padding:0 1px}} .merged{{text-decoration:underline dotted #c98a00}}
</style></head><body><main>
<h1>Exploratory: per-trace probe scores, test problem {tr} (draw {k:02d} of 10)</h1>
<p class="note">Pre-registration §13 item 3 (D25 X4). Decision function of the four frozen band
probes at every reasoning token, from a teacher-forced pass of the 1.5B over the stored token
ids. Orange lines and highlights: the {n_onsets} paragraph-rule onsets; dotted underline: hits
merged into an onset. {check} No verdict is applied to any number here.</p>
<div>{legend}<span class="key"><span class="sw" style="background:#f0a000"></span>onset</span></div>
<div class="box">{svg}</div>
<h2 style="font-size:15px">Problem</h2>
<div class="box"><!--quoted--><pre>{html.escape(problem)}</pre><!--/quoted--></div>
<h2 style="font-size:15px">Reasoning</h2>
<div class="box"><!--quoted--><pre>{body}</pre><!--/quoted--></div>
</main></body></html>
"""


# ---------------------------------------------------------------- main

def main() -> None:
    from before_the_wait.align import prefix_lengths
    from before_the_wait.generate import load_model_and_tokenizer

    t_start = time.perf_counter()
    rx.frozen_check("a3", "before")
    split = json.loads(rx.SPLIT_JSON.read_text())
    draw = draw_traces(split["test"])
    rx.save_record("a3_draw", {"seed": SEED, "draw": draw})
    print("a3 draw", draw, flush=True)
    layers = {b: int(rv.probe_meta(b)["layer"][0]) for b in rp.BANDS}
    manifest = rg.load_manifest()
    tok, model = load_model_and_tokenizer()
    model._btw_real = True
    eos = int(tok.eos_token_id)
    TRACE_OUT.mkdir(parents=True, exist_ok=True)
    check_rows, rec = [], {"draw": draw, "layers": layers, "tolerance": TOLERANCE, "traces": []}
    pages = []
    for k, tr in enumerate(draw, 1):
        t1s = time.perf_counter()
        trace = json.loads((rx.TRACES_DIR / f"gsm8k_{tr:04d}.json").read_text())
        r0, r1 = trace["reasoning_token_span"]
        ts = list(range(r0, r1))
        arrays, checks = ea.extract_trace(model, trace, eos, ts)
        assert checks["wiring_max_abs_diff"] == 0.0 and checks["hidden_finite"], tr
        hid = arrays["hidden"]
        scores = {b: rv.probe_scores_for(b, hid[:, layers[b] - 1, :]) for b in rp.BANDS}
        path = rp.ACT_DIR / f"gsm8k_{tr:04d}.npz"
        assert rg.sha256_file(path) == manifest[tr]
        with np.load(path, allow_pickle=False) as z:
            ts_saved, saved = z["t"], z["hidden"]
            assert ts_saved.min() >= r0 and ts_saved.max() < r1, tr
            idx = ts_saved - r0
            trow = {"trace_index": tr, "draw": k, "reasoning_tokens": len(ts),
                    "positions": int(len(ts_saved))}
            for b in rp.BANDS:
                Ls = layers[b] - 1
                s_saved = rv.probe_scores_for(b, saved[:, Ls, :])
                ds = float(np.max(np.abs(scores[b][idx] - s_saved)))
                dh = float(np.max(np.abs(hid[idx, Ls, :].astype(np.float32)
                                         - saved[:, Ls, :].astype(np.float32))))
                trow[f"score_{b}"], trow[f"hidden_{b}"] = ds, dh
                check_rows.append([k, tr, b, layers[b], len(ts_saved), f"{ds:.3g}", f"{dh:.3g}"])
        del saved, hid, arrays
        rec["traces"].append(trow)
        worst = max(trow[f"score_{b}"] for b in rp.BANDS)
        print(f"[a3 {k:02d} trace {tr}] {len(ts)} tokens, {trow['positions']} saved positions, "
              f"max |Δscore| {worst:.3g}, {time.perf_counter() - t1s:.1f}s", flush=True)
        on, merged = onsets_for(tr)
        out_ids = list(trace["output_token_ids"])
        offs = prefix_lengths(tok, out_ids)
        assert tok.decode(out_ids, skip_special_tokens=False) == trace["output_text"], tr
        text = trace["output_text"]
        pieces = [text[offs[i]:offs[i + 1]] for i in range(r0, r1)]
        on_r = [t for t in on if r0 <= t < r1]
        pages.append((k, tr, trace["problem"], svg_plot(r0, scores, on_r, layers),
                      text_html(pieces, r0, set(on_r), set(merged)), len(on_r)))
    rx.write_csv(rx.OUT / "a3_check.csv", ["draw", "trace_index", "band", "layer",
                                           "positions_compared", "max_abs_score_diff",
                                           "max_abs_hidden_diff"], check_rows)
    worst = max(float(r[5]) for r in check_rows)
    rec["max_abs_score_diff"] = worst
    rec["pass"] = worst <= TOLERANCE
    rx.save_record("a3", rec)
    if not rec["pass"]:
        sys.exit(f"STOP (a3): max |Δscore| {worst} above tolerance {TOLERANCE}; no HTML written")
    note = (f"Check: at the saved example positions of this trace, the scores equal those from "
            f"the saved activations (largest difference over all 10 traces: {worst:.3g}).")
    sizes = {}
    for k, tr, problem, svg, body, n_on in pages:
        p = TRACE_OUT / f"trace_{k:02d}.html"
        p.write_text(page(k, tr, problem, svg, body, layers, n_on, note))
        sizes[p.name] = p.stat().st_size
    rec["html_bytes"] = sizes
    rx.save_record("a3", rec)
    rx.frozen_check("a3", "after")
    print(f"a3 done in {time.perf_counter() - t_start:.0f}s; html bytes {sizes}", flush=True)


if __name__ == "__main__":
    main()
