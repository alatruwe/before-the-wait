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
    ASSUMED_AUC,
    ASSUMED_RHO,
    BANDS,
    CAP,
    INTERVAL,
    MARGIN,
    REQUIRED_BANDS,
    ROUND_TO,
    apply_margin,
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
from before_the_wait.generate import MODEL_ID, MODEL_REVISION  # noqa: E402
from before_the_wait.label import TOKEN_KINDS, token_kind  # noqa: E402

RESULTS = REPO_ROOT / "results"
TRACE_DIR = REPO_ROOT / "traces" / "pilot"

DELTA = 0.05
DELTA_SECONDARY = (0.03, 0.08)
TEST_SHARE = 0.35
VALIDATION_SHARE = 0.15
GENERATION_RATE = 63.0  # tokens/second, from the handoff
KIND_THRESHOLD = 20  # §6: a kind stands alone at >= 20 pilot positive positions
MATCH_RATIO = 2  # §6 as revised 2026-10-19: two negatives per positive
BYTES_PER_POSITION = 28 * 1536 * 2  # layers 1-28, width 1536, fp16: ~86 KB

# The 2026-10-17 run (commit fed3fa6), for the change table. That run used
# AUROC 0.70 and rho 0.5 for m, gaps from the previous onset only, positives
# without the no-hit-in-between rule, and 1:1 matching.
BEFORE = {
    "m": 338,
    "n_rule": 157,
    "n": 190,
    "binding": "21-40",
    "positives": {"2-5": 1151, "6-10": 1429, "11-20": 2867, "21-40": 5695},
    "eligible_gsm8k": {"2-5": 114, "6-10": 110, "11-20": 106, "21-40": 96},
    "quiet_zone": 100,
}


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
            hit_idx = [int(h["token_index"]) for h in all_hits[k]]
            for o in onsets[k]:
                gaps.append(gap_to_previous_hit(int(o["token_index"]), hit_idx, t0))
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
    n_rule_gsm = n_gsm[binding_band]
    n_final_gsm = apply_margin(n_rule_gsm)

    n_math = per_band_n(o_math, elig["math500"])
    binding_math = max(REQUIRED_BANDS, key=lambda b: n_math[b])
    n_rule_math = n_math[binding_math]

    switched = n_final_gsm > CAP
    n_rule = n_rule_math if switched else n_rule_gsm
    final_n = min(apply_margin(n_rule_math), CAP) if switched else n_final_gsm
    final_o = o_math if switched else o_gsm
    final_ds = "math500" if switched else "gsm8k"
    binding = binding_math if switched else binding_band

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
    skipped: Counter[str] = Counter()
    gate_positives_by_ds: Counter[str] = Counter()
    for k in kept:
        tr = traces[k]
        ids = tr["output_token_ids"]
        t0, _t1 = tr["reasoning_token_span"]
        spans = marker_spans[k]
        hit_idx = [int(h["token_index"]) for h in all_hits[k]]
        for o in onsets[k]:
            t = int(o["token_index"])
            if classify_band_position(t - 1, t, hit_idx, spans, t0) is None:
                gate_positives_by_ds[tr["dataset"]] += 1
            for name, lo, hi in BANDS:
                for d in range(lo, hi + 1):
                    pos = t - d
                    reason = classify_band_position(pos, t, hit_idx, spans, t0)
                    if reason is not None:
                        skipped[reason] += 1
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
                     and neg100.get(k, 0) < MATCH_RATIO * positives[largest_band].get(k, 0)]
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
    # Looped traces are generated too (and mostly run to the cap), so the mean
    # over every pilot trace of the dataset is the fuller projection.
    ds_rows = [r for k, r in trace_rows.items() if k[0] == final_ds]
    mean_tokens_every = sum(int(r["new_tokens"]) for r in ds_rows) / len(ds_rows)
    hours_every = final_n * mean_tokens_every / GENERATION_RATE / 3600

    # ---------- 9. activation disk ----------
    n_ds_kept = n_math_kept if switched else n_gsm_kept
    kept_traces_at_n = final_n * (1 - loop_share)
    band_pos_per_trace = {name: sum(positives_by_ds[(final_ds, name)].values()) / n_ds_kept
                          for name, _, _ in BANDS}
    gate_pos_per_trace = gate_positives_by_ds[final_ds] / n_ds_kept
    proj_band_pos = {name: v * kept_traces_at_n for name, v in band_pos_per_trace.items()}
    proj_gate_pos = gate_pos_per_trace * kept_traces_at_n
    proj_pos = sum(proj_band_pos.values()) + proj_gate_pos
    proj_neg = MATCH_RATIO * proj_pos
    proj_bytes = (proj_pos + proj_neg) * BYTES_PER_POSITION

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
    w(f"| Trace count from the rule, N_rule | **{n_rule}** | §4 (all four bands) |")
    w(f"| **Final trace count N** | **{final_n}** {final_ds} traces | §4 / D12 "
      f"(×{MARGIN}, up to the nearest {ROUND_TO}) |")
    w(f"| Binding band | {binding} | §4 |")
    w(f"| Split | **{split}** | §4 |")
    w(f"| Quiet zone | **{quiet_zone} tokens** | §6 |")
    w(f"| Matching kinds | **{', '.join(final_kinds)}** | §6 |")
    w(f"| Projected generation | **{hours_every:.1f} h** | handoff |")
    w(f"| Projected activation disk | **{proj_bytes / 1e9:.1f} GB** | handoff |")
    w("")

    w("## Changes from the 2026-10-17 run")
    w("")
    w("Recomputed 2026-10-19 under the revised positive rule (no lexicon hit of")
    w("any kind strictly between a position and its onset; e_b measured from the")
    w(f"previous hit of any kind), m assumptions (AUROC {ASSUMED_AUC}, correlation")
    w(f"{ASSUMED_RHO}) and 1:{MATCH_RATIO} matching. Before = commit `fed3fa6`.")
    w("")
    w("| Quantity | 2026-10-17 | 2026-10-19 |")
    w("|---|---|---|")
    w(f"| m (δ = {DELTA}) | {BEFORE['m']} | **{m_main}** |")
    w(f"| N_rule | {BEFORE['n_rule']} | **{n_rule}** |")
    w(f"| N | {BEFORE['n']} | **{final_n}** |")
    w(f"| Binding band | {BEFORE['binding']} | **{binding}** |")
    for name, _lo, hi in BANDS:
        w(f"| Pilot positive positions, {name} | {BEFORE['positives'][name]} | "
          f"**{sum(positives[name].values())}** |")
    for name, _lo, hi in BANDS:
        w(f"| GSM8K onsets with gap > {hi} (e_b numerator), {name} | "
          f"{BEFORE['eligible_gsm8k'][name]} | **{elig['gsm8k'][hi][1]}** |")
    w(f"| Quiet zone | {BEFORE['quiet_zone']} | **{quiet_zone}** |")
    w("")

    w("## Inputs")
    w("")
    w("| Symbol | Meaning | Value | Source |")
    w("|---|---|---|---|")
    w(f"| δ | null bar | {DELTA} (provisional) | §11 |")
    w(f"| A | assumed AUROC for m | {ASSUMED_AUC} | §11 |")
    w(f"| ρ | assumed probe–rival correlation for m | {ASSUMED_RHO} | §11 |")
    w(f"| matching | negatives per positive | 1:{MATCH_RATIO} | §6 |")
    w(f"| r | test share | {TEST_SHARE} | §4 split |")
    w(f"| ℓ | looped-trace share | {len(looped)}/{len(trace_rows)} = {loop_share:.4f} | `c1_traces.csv` |")
    w(f"| o (GSM8K) | onsets per non-looped trace | {sum(len(onsets[k]) for k in kept if k[0] == 'gsm8k')}/{n_gsm_kept} = {o_gsm:.4f} | `c1_onsets_paragraph.csv` |")
    w(f"| o (MATH500) | onsets per non-looped trace | {sum(len(onsets[k]) for k in kept if k[0] == 'math500')}/{n_math_kept} = {o_math:.4f} | `c1_onsets_paragraph.csv` |")
    w(f"| margin | D12 inflation on N_rule | ×{MARGIN}, rounded up to the nearest {ROUND_TO} | §4 / D12 |")
    w(f"| cap | §4 cap on N | {CAP} | §4 / D12 |")
    w("")

    w("## 1. Minimum events m (§11)")
    w("")
    w("> m = the smallest number of test-split onsets per band such that the")
    w("> half-width of the 98.75% interval on the lift is ≤ δ, using the")
    w(f"> Hanley–McNeil standard error of an AUROC of {ASSUMED_AUC} on m positives and m")
    w(f"> matched negatives for each of the two AUROCs, combined with an assumed")
    w(f"> correlation of {ASSUMED_RHO} between them.")
    w("")
    w(f"- interval {INTERVAL:.4f} two-sided → z = **{z:.4f}**")
    w(f"- assumed AUROC {ASSUMED_AUC} and correlation {ASSUMED_RHO}, stated before any probe exists")
    w(f"- SE_lift = √(2(1 − ρ)) × SE_AUROC = √{2 * (1 - ASSUMED_RHO):.1f} × SE_AUROC")
    w("")
    w("| δ | m | SE_AUROC at m | SE_lift at m | half-width z·SE_lift |")
    w("|---|---|---|---|---|")
    for d in sorted({DELTA, *DELTA_SECONDARY}):
        mm = m_main if d == DELTA else m_alt[d]
        se = hanley_mcneil_se(ASSUMED_AUC, mm, mm)
        sel = lift_se(ASSUMED_AUC, mm, mm)
        star = " (primary)" if d == DELTA else ""
        w(f"| {d}{star} | **{mm}** | {se:.5f} | {sel:.5f} | {z * sel:.5f} |")
    w("")

    w("## 2. Eligibility shares e_b (§4)")
    w("")
    w("> e_b: the share of onsets whose previous lexicon hit of any kind, onset or")
    w("> merged (or the reasoning start, for a first hit), lies more than the")
    w("> band's upper bound tokens back.")
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
    w("> each of the four bands ... The final count is that N inflated by 20%")
    w("> and rounded up to the nearest ten, because o is estimated from 13")
    w("> pilot traces. Capped at 1,000. (D12)")
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
    w(f"**Binding band: {binding_band} → N_rule = {n_rule_gsm} GSM8K traces.**")
    w("")
    w(f"D12 margin: ceil({MARGIN} × {n_rule_gsm}) = {__import__('math').ceil(MARGIN * n_rule_gsm)}, "
      f"rounded up to the nearest {ROUND_TO} → **N = {n_final_gsm}**.")
    w("")
    w(f"Switch rule: N {'>' if switched else '≤'} cap ({CAP}), so the MATH500 switch "
      f"{'**applies**' if switched else 'does not apply'}.")
    w("")
    w(f"For the record, the same computation on MATH500 (o = {o_math:.4f}) gives "
      f"N_rule = {n_rule_math} at binding band {binding_math}, N = {apply_margin(n_rule_math)}.")
    w("")
    w("Projected test-split onsets per band at the final N, against m:")
    w("")
    w("| band | projected test onsets at N | ≥ m? |")
    w("|---|---|---|")
    for name, _lo, hi in BANDS:
        e = elig[final_ds][hi][0]
        proj = final_n * TEST_SHARE * (1 - loop_share) * final_o * e
        w(f"| {name} | {proj:.1f} | {'yes' if proj >= m_main else '**no**'} |")
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
      f"{len(kept)} non-looped pilot traces, excluding positions before the "
      "reasoning start, positions inside any marker span (onset or merged), and "
      "positions with a lexicon hit of any kind strictly between them and the "
      "onset (so each position belongs only to its nearest onset).")
    w("")
    w(f"- positions skipped as before the reasoning start: {skipped['before_reasoning']}")
    w(f"- positions skipped as inside a marker span: {skipped['in_marker']}")
    w(f"- positions skipped as having a hit in between: {skipped['hit_between']}")
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
    w("> positions (no marker in the next 100 tokens) is smaller than twice the")
    w(f"> pilot count of positive positions of that kind in band {largest_band}")
    w("> (1:2 matching); then 60.")
    w("")
    w(f"| kind | negatives at 100 | positives in {largest_band} | {MATCH_RATIO}× positives "
      f"| negatives < {MATCH_RATIO}× positives? |")
    w("|---|---|---|---|---|")
    for kind in TOKEN_KINDS:
        n100 = neg100.get(kind, 0)
        pos = positives[largest_band].get(kind, 0)
        flag = "**yes**" if (pos > 0 and n100 < MATCH_RATIO * pos) else "no"
        w(f"| `{kind}` | {n100} | {pos} | {MATCH_RATIO * pos} | {flag} |")
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
    w(f"- mean new tokens per {final_ds} pilot trace, looped included = {mean_tokens_every:.1f}")
    w(f"- at {GENERATION_RATE:.0f} tokens/second → **{hours_every:.1f} hours** "
      f"({final_n * mean_tokens_every / GENERATION_RATE / 60:.0f} minutes)")
    w("")
    w(f"Looped traces are generated before they are excluded, so the mean includes them. "
      f"Over non-looped {final_ds} traces only ({mean_tokens_ds:.1f} tokens) it is "
      f"{hours:.1f} hours; over all {len(kept)} non-looped pilot traces "
      f"({mean_tokens_all:.1f} tokens), {hours_all:.1f} hours.")
    w("")

    w("## 9. Activation disk projection")
    w("")
    w(f"Hidden states at layers 1–28, width 1,536, fp16: 28 × 1,536 × 2 = "
      f"{BYTES_PER_POSITION:,} bytes (~86 KB) per position. Positives projected from "
      f"the {n_ds_kept} non-looped {final_ds} pilot traces under the revised rule, "
      f"scaled to N × (1 − ℓ) = {kept_traces_at_n:.1f} kept traces.")
    w("")
    w("| class | per kept pilot trace | projected at N |")
    w("|---|---|---|")
    w(f"| gate positives (d = 1) | {gate_pos_per_trace:.2f} | {proj_gate_pos:,.0f} |")
    for name, _lo, _hi in BANDS:
        w(f"| band {name} positives | {band_pos_per_trace[name]:.2f} | {proj_band_pos[name]:,.0f} |")
    w(f"| **all positives** | | **{proj_pos:,.0f}** |")
    w(f"| negatives, 1:{MATCH_RATIO} | | **{proj_neg:,.0f}** |")
    w(f"| **total positions** | | **{proj_pos + proj_neg:,.0f}** |")
    w("")
    w(f"→ **{proj_bytes / 1e9:.1f} GB** ({proj_bytes / 2**30:.1f} GiB). The negatives "
      f"figure is an upper bound: it gives every band and the gate its own "
      f"{MATCH_RATIO} negatives per positive. §6 allows reuse across bands (never "
      f"within one), so the real number of distinct negatives is lower.")
    w("")

    (RESULTS / "count_calculations.md").write_text("\n".join(doc) + "\n")
    print("\n".join(doc[:28]))
    print(f"\nwrote {RESULTS / 'count_calculations.md'}")


if __name__ == "__main__":
    main()
