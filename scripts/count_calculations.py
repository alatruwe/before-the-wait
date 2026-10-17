"""Turn the pilot counts into the numbers the pre-registration's rules call for.

    .venv/bin/python scripts/count_calculations.py

Deterministic: no seeds, no sampling. Counts only — no model is loaded and no
activation is read. The tokenizer is used to decode individual tokens for their
kind and to measure how many tokens a marker's text spans.

Every number is computed from a rule quoted in results/count_calculations.md;
nothing here chooses a value.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
sys.path.insert(0, str(REPO_ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from before_the_wait.counts import (  # noqa: E402
    BANDS,
    INTERVAL,
    REQUIRED_BANDS,
    hanley_mcneil_se,
    marker_token_span,
    minimum_events,
    projected_validation_onsets,
    trace_count,
    z_for_interval,
)
from before_the_wait.generate import MODEL_ID, MODEL_REVISION  # noqa: E402
from before_the_wait.label import TOKEN_KINDS, token_kind  # noqa: E402

RESULTS = REPO_ROOT / "results"
TRACE_DIR = REPO_ROOT / "traces" / "pilot"

DELTA = 0.05
DELTA_SECONDARY = (0.03, 0.08)
TEST_SHARE = 0.35
VALIDATION_SHARE = 0.15
CAP = 1000  # §4 says "Capped at [DECIDE]"; not set, so this is the stated stand-in.
GENERATION_RATE = 63.0  # tokens/second, from the handoff
KIND_THRESHOLD = 20  # §6: a kind stands alone at >= 20 pilot positive positions


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    traces = {}
    for p in sorted(TRACE_DIR.glob("*.json")):
        if p.name.startswith("_"):
            continue
        t = json.loads(p.read_text())
        traces[(t["dataset"], str(t["index"]))] = t

    trace_rows = {(r["dataset"], r["index"]): r
                  for r in csv.DictReader((RESULTS / "c1_traces.csv").open())}
    looped = {k for k, r in trace_rows.items() if r["looped"] == "True"}
    kept = [k for k in trace_rows if k not in looped]

    all_hits: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in csv.DictReader((RESULTS / "c1_onsets.csv").open()):
        all_hits[(r["dataset"], r["index"])].append(r)
    onsets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in csv.DictReader((RESULTS / "c1_onsets_paragraph.csv").open()):
        if r["merged_into"] == "":
            onsets[(r["dataset"], r["index"])].append(r)
    for d in (all_hits, onsets):
        for k in d:
            d[k].sort(key=lambda r: int(r["token_index"]))

    doc: list[str] = []
    w = doc.append

    # ---------- 1. minimum events ----------
    z = z_for_interval()
    m_main = minimum_events(DELTA)
    m_alt = {d: minimum_events(d) for d in DELTA_SECONDARY}

    # ---------- 2. eligibility shares ----------
    def eligibility(dataset: str) -> dict[int, tuple[float, int, int]]:
        gaps: list[int] = []
        for k in kept:
            if k[0] != dataset:
                continue
            t0 = traces[k]["reasoning_token_span"][0]
            prev = t0
            for o in onsets[k]:
                gaps.append(int(o["token_index"]) - prev)
                prev = int(o["token_index"])
        out = {}
        for _name, _lo, hi in BANDS:
            n = sum(1 for g in gaps if g > hi)
            out[hi] = (n / len(gaps) if gaps else 0.0, n, len(gaps))
        return out

    elig = {ds: eligibility(ds) for ds in ("gsm8k", "math500")}

    # ---------- 3. trace count ----------
    loop_share = len(looped) / len(trace_rows)
    n_gsm_kept = sum(1 for k in kept if k[0] == "gsm8k")
    n_math_kept = sum(1 for k in kept if k[0] == "math500")
    o_gsm = sum(len(onsets[k]) for k in kept if k[0] == "gsm8k") / n_gsm_kept
    o_math = sum(len(onsets[k]) for k in kept if k[0] == "math500") / n_math_kept

    def per_band_n(o: float, e: dict[int, tuple[float, int, int]]) -> dict[str, int]:
        return {name: trace_count(m_main, TEST_SHARE, loop_share, o, e[hi][0])
                for name, _lo, hi in BANDS}

    n_gsm = per_band_n(o_gsm, elig["gsm8k"])
    binding_band = max(REQUIRED_BANDS, key=lambda b: n_gsm[b])
    n_required = n_gsm[binding_band]
    switched = n_required > CAP
    n_math = per_band_n(o_math, elig["math500"])
    binding_math = max(REQUIRED_BANDS, key=lambda b: n_math[b])

    final_n = n_math[binding_math] if switched else n_required
    final_o = o_math if switched else o_gsm
    final_ds = "math500" if switched else "gsm8k"

    # ---------- 4. split ----------
    val_onsets = projected_validation_onsets(final_n, loop_share, final_o, VALIDATION_SHARE)
    split = "50 / 15 / 35" if val_onsets >= 100 else "45 / 20 / 35"

    # ---------- 5. positive positions by kind ----------
    marker_spans: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for k in kept:
        ids = traces[k]["output_token_ids"]
        marker_spans[k] = [
            marker_token_span(tokenizer, ids, int(h["token_index"]), h["marker"])
            for h in all_hits[k]
        ]

    positives: dict[str, Counter[str]] = {name: Counter() for name, _, _ in BANDS}
    positives_by_ds: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    skipped_before_reasoning = 0
    skipped_in_marker = 0
    for k in kept:
        tr = traces[k]
        ids = tr["output_token_ids"]
        t0, _t1 = tr["reasoning_token_span"]
        spans = marker_spans[k]
        for o in onsets[k]:
            t = int(o["token_index"])
            for name, lo, hi in BANDS:
                for d in range(lo, hi + 1):
                    pos = t - d
                    if pos < t0:
                        skipped_before_reasoning += 1
                        continue
                    if any(s <= pos < e for s, e in spans):
                        skipped_in_marker += 1
                        continue
                    kind = token_kind(tokenizer.decode([ids[pos]], skip_special_tokens=False))
                    positives[name][kind] += 1
                    positives_by_ds[(tr["dataset"], name)][kind] += 1

    total_positives: Counter[str] = Counter()
    for c in positives.values():
        total_positives += c

    # ---------- 6. quiet zone ----------
    def negatives_at(lookahead: int) -> Counter[str]:
        out: Counter[str] = Counter()
        for k in kept:
            tr = traces[k]
            ids = tr["output_token_ids"]
            t0, t1 = tr["reasoning_token_span"]
            hits = {int(h["token_index"]) for h in all_hits[k]}
            for q in range(t0, t1):
                if q + lookahead >= t1:
                    continue
                if any(x in hits for x in range(q + 1, q + 1 + lookahead)):
                    continue
                out[token_kind(tokenizer.decode([ids[q]], skip_special_tokens=False))] += 1
        return out

    neg100 = negatives_at(100)
    largest_band = BANDS[-1][0]
    failing_kinds = [k for k in TOKEN_KINDS
                     if positives[largest_band].get(k, 0) > 0
                     and neg100.get(k, 0) < positives[largest_band].get(k, 0)]
    quiet_zone = 60 if failing_kinds else 100
    neg60 = negatives_at(60) if failing_kinds else None

    # ---------- 7. matching kinds ----------
    standalone = [k for k in TOKEN_KINDS if total_positives.get(k, 0) >= KIND_THRESHOLD]
    merged_kinds = [k for k in TOKEN_KINDS
                    if 0 < total_positives.get(k, 0) < KIND_THRESHOLD]
    final_kinds = standalone + (["other"] if "other" not in standalone else [])

    # ---------- 8. generation time ----------
    mean_tokens_ds = sum(int(trace_rows[k]["new_tokens"]) for k in kept
                         if k[0] == final_ds) / (n_math_kept if switched else n_gsm_kept)
    mean_tokens_all = sum(int(trace_rows[k]["new_tokens"]) for k in kept) / len(kept)
    hours = final_n * mean_tokens_ds / GENERATION_RATE / 3600
    hours_all = final_n * mean_tokens_all / GENERATION_RATE / 3600

    # ================= document =================
    w("# Count calculations for the pre-registration")
    w("")
    w("Computed by `scripts/count_calculations.py` from the pilot results. Every")
    w("number below is produced by a rule already written in")
    w("`methodology/pre-registration.md`; nothing here is chosen.")
    w("")
    w("## Summary")
    w("")
    w("| Quantity | Value | Rule |")
    w("|---|---|---|")
    w(f"| Minimum events m (δ = {DELTA}) | **{m_main}** per band | §11 |")
    w(f"| Trace count N | **{final_n}** {final_ds} traces | §4 |")
    w(f"| Binding band | {binding_math if switched else binding_band} | §4 |")
    w(f"| Split | **{split}** | §4 |")
    w(f"| Quiet zone | **{quiet_zone} tokens** | §6 |")
    w(f"| Matching kinds | **{', '.join(final_kinds)}** | §6 |")
    w(f"| Projected generation | **{hours:.1f} h** | handoff |")
    w("")

    w("## Inputs")
    w("")
    w("| Symbol | Meaning | Value | Source |")
    w("|---|---|---|---|")
    w(f"| δ | null bar | {DELTA} (provisional) | §11 |")
    w(f"| r | test share | {TEST_SHARE} | §4 split |")
    w(f"| ℓ | looped-trace share | {len(looped)}/{len(trace_rows)} = {loop_share:.4f} | `c1_traces.csv` |")
    w(f"| o (GSM8K) | onsets per non-looped trace | {sum(len(onsets[k]) for k in kept if k[0] == 'gsm8k')}/{n_gsm_kept} = {o_gsm:.4f} | `c1_onsets_paragraph.csv` |")
    w(f"| o (MATH500) | onsets per non-looped trace | {sum(len(onsets[k]) for k in kept if k[0] == 'math500')}/{n_math_kept} = {o_math:.4f} | `c1_onsets_paragraph.csv` |")
    w(f"| cap | §4 cap on N | {CAP} (**not set in the pre-registration**; stand-in per the handoff) | §4 `[DECIDE]` |")
    w("")

    w("## 1. Minimum events m (§11)")
    w("")
    w("> m = the smallest number of test-split onsets per band such that the")
    w("> half-width of the 98.75% interval on the lift is ≤ δ, using the")
    w("> Hanley–McNeil standard error of an AUROC of 0.70 on m positives and m")
    w("> matched negatives ... combined with an assumed correlation of 0.5")
    w("> between them (so SE_lift = SE of one AUROC).")
    w("")
    w(f"- interval {INTERVAL:.4f} two-sided → z = **{z:.4f}**")
    w("- assumed AUROC 0.70, fixed before any probe exists")
    w("")
    w("| δ | m | SE at m | half-width z·SE |")
    w("|---|---|---|---|")
    for d in sorted({DELTA, *DELTA_SECONDARY}):
        mm = m_main if d == DELTA else m_alt[d]
        se = hanley_mcneil_se(0.70, mm, mm)
        star = " (primary)" if d == DELTA else ""
        w(f"| {d}{star} | **{mm}** | {se:.5f} | {z * se:.5f} |")
    w("")

    w("## 2. Eligibility shares e_b (§4)")
    w("")
    w("> e_b: the share of onsets whose previous onset (or the reasoning start,")
    w("> for a first onset) lies more than the band's upper bound tokens back.")
    w("")
    for ds in ("gsm8k", "math500"):
        w(f"**{ds}** ({sum(len(onsets[k]) for k in kept if k[0] == ds)} onsets over "
          f"{n_gsm_kept if ds == 'gsm8k' else n_math_kept} non-looped traces)")
        w("")
        w("| band | upper bound | onsets with gap > bound | e_b |")
        w("|---|---|---|---|")
        for name, _lo, hi in BANDS:
            share, n, total = elig[ds][hi]
            w(f"| {name} | {hi} | {n} / {total} | **{share:.4f}** |")
        w("")

    w("## 3. Trace count N (§4)")
    w("")
    w("> N = the smallest integer such that N × r × (1 − ℓ) × o × e_b ≥ m for")
    w("> each of bands 2–5, 6–10 and 11–20. Band 21–40 is reported whatever its")
    w("> count.")
    w("")
    w(f"With m = {m_main}, r = {TEST_SHARE}, ℓ = {loop_share:.4f}, o = {o_gsm:.4f} (GSM8K):")
    w("")
    w("| band | e_b | onsets per trace in band | N | required by §4 |")
    w("|---|---|---|---|---|")
    for name, _lo, hi in BANDS:
        e = elig["gsm8k"][hi][0]
        per = TEST_SHARE * (1 - loop_share) * o_gsm * e
        req = "yes" if name in REQUIRED_BANDS else "no (reported only)"
        w(f"| {name} | {e:.4f} | {per:.4f} | {n_gsm[name]} | {req} |")
    w("")
    w(f"**Binding band: {binding_band} → N = {n_required} GSM8K traces.**")
    w("")
    w(f"Switch rule: N {'>' if switched else '≤'} cap ({CAP}), so the MATH500 switch "
      f"{'**applies**' if switched else 'does not apply'}.")
    w("")
    w(f"For the record, the same computation on MATH500 (o = {o_math:.4f}) gives "
      f"N = {n_math[binding_math]} at binding band {binding_math}.")
    w("")

    w("## 4. Split (§4)")
    w("")
    w("> train / validation / test = 50 / 15 / 35, unless N × (1 − ℓ) × o × 0.15")
    w("> < 100 projected validation onsets, in which case 45 / 20 / 35.")
    w("")
    w(f"- N × (1 − ℓ) × o × 0.15 = {final_n} × {1 - loop_share:.4f} × {final_o:.4f} × "
      f"{VALIDATION_SHARE} = **{val_onsets:.1f}** projected validation onsets")
    w(f"- {val_onsets:.1f} {'≥' if val_onsets >= 100 else '<'} 100 → **{split}**")
    w("")

    w("## 5. Positive positions by token kind (§6)")
    w("")
    w("Tokens at distances 2–40 before each paragraph-rule onset in the "
      f"{len(kept)} non-looped pilot traces, excluding positions inside any "
      "marker span (onset or merged) and positions before the reasoning start.")
    w("")
    w(f"- positions skipped as before the reasoning start: {skipped_before_reasoning}")
    w(f"- positions skipped as inside a marker span: {skipped_in_marker}")
    w("")
    w("| kind | " + " | ".join(name for name, _, _ in BANDS) + " | all bands |")
    w("|---|" + "---|" * (len(BANDS) + 1))
    for kind in TOKEN_KINDS:
        cells = " | ".join(str(positives[name].get(kind, 0)) for name, _, _ in BANDS)
        w(f"| `{kind}` | {cells} | **{total_positives.get(kind, 0)}** |")
    totals = " | ".join(str(sum(positives[name].values())) for name, _, _ in BANDS)
    w(f"| **total** | {totals} | **{sum(total_positives.values())}** |")
    w("")

    w("## 6. Quiet zone (§6)")
    w("")
    w("> 100 tokens, unless for any token kind the pilot count of negative")
    w("> positions (no marker in the next 100 tokens) is smaller than the pilot")
    w(f"> count of positive positions of that kind in the largest band ({largest_band});")
    w("> then 60.")
    w("")
    w(f"| kind | negatives at 100 | positives in {largest_band} | negatives < positives? |")
    w("|---|---|---|---|")
    for kind in TOKEN_KINDS:
        n100 = neg100.get(kind, 0)
        pos = positives[largest_band].get(kind, 0)
        flag = "**yes**" if (pos > 0 and n100 < pos) else "no"
        w(f"| `{kind}` | {n100} | {pos} | {flag} |")
    w("")
    w(f"Kinds failing the test: {failing_kinds if failing_kinds else 'none'} → "
      f"**quiet zone = {quiet_zone} tokens**.")
    if neg60 is not None:
        w("")
        w("Negatives at 60 tokens, for the record: "
          + ", ".join(f"`{k}` {neg60.get(k, 0)}" for k in TOKEN_KINDS))
    w("")

    w("## 7. Matching kinds (§6)")
    w("")
    w("> A kind stands on its own if at least 20 pilot positive positions")
    w("> (across all bands) fall in it; rarer kinds merge into `other`.")
    w("")
    w("| kind | positives, all bands | ≥ 20? | outcome |")
    w("|---|---|---|---|")
    for kind in TOKEN_KINDS:
        n = total_positives.get(kind, 0)
        if n >= KIND_THRESHOLD:
            outcome = "stands alone"
        elif n > 0:
            outcome = "merges into `other`"
        else:
            outcome = "unused (no positives)"
        w(f"| `{kind}` | {n} | {'yes' if n >= KIND_THRESHOLD else 'no'} | {outcome} |")
    w("")
    w(f"**Final kind list: {', '.join(final_kinds)}.** "
      f"Merged into `other`: {', '.join(merged_kinds) if merged_kinds else 'none'}.")
    w("")

    w("## 8. Generation time projection")
    w("")
    w(f"- N = {final_n} traces")
    w(f"- mean new tokens per non-looped {final_ds} pilot trace = {mean_tokens_ds:.1f}")
    w(f"- at {GENERATION_RATE:.0f} tokens/second → **{hours:.1f} hours** "
      f"({final_n * mean_tokens_ds / GENERATION_RATE / 60:.0f} minutes)")
    w("")
    w(f"Using the mean over all {len(kept)} non-looped pilot traces "
      f"({mean_tokens_all:.1f} tokens) instead gives {hours_all:.1f} hours.")
    w("")

    (RESULTS / "count_calculations.md").write_text("\n".join(doc) + "\n")
    print("\n".join(doc[:28]))
    print(f"\nwrote {RESULTS / 'count_calculations.md'}")


if __name__ == "__main__":
    main()
