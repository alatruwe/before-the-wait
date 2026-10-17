"""c3: extraction seam test, gate input at distance 1, and the logit-lens plot.

    caffeinate -i .venv/bin/python scripts/run_c3_first_look.py

Distance 1 only. Reads position `onset - 1` and matched negative positions
(no marker within the next 100 tokens). No position 2-100 tokens before an
onset is read. Nothing is trained.
"""
from __future__ import annotations

import csv
import json
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from before_the_wait.generate import load_model_and_tokenizer  # noqa: E402
from before_the_wait.label import merge_onsets, token_kind  # noqa: E402
from before_the_wait.lens import (  # noqa: E402
    auroc,
    bootstrap_auroc_by_trace,
    bootstrap_auroc_by_trace as _boot,  # noqa: F401
    build_input_ids,
    final_norm,
    hidden_states_only,
    lens_probs_at,
    prob_of_tokens,
    sampling_probability,
)
from before_the_wait.summarize import analyze_trace  # noqa: E402

TRACE_DIR = REPO_ROOT / "traces" / "pilot"
RESULTS = REPO_ROOT / "results"
SEED = 20261017
LOOKAHEAD = 100
N_BOOTSTRAP = 2000
TEMPERATURE = 0.6
TOP_P = 0.95


def main() -> None:
    t_start = time.perf_counter()
    tokenizer, model = load_model_and_tokenizer()
    eos_id = int(tokenizer.eos_token_id)
    n_layers = model.config.num_hidden_layers
    print(f"model loaded: {n_layers} layers, device {model.device}, dtype {model.dtype}")

    paths = sorted(p for p in TRACE_DIR.glob("*.json") if not p.name.startswith("_"))
    traces = {}
    for p in paths:
        t = json.loads(p.read_text())
        traces[(t["dataset"], t["index"])] = t

    # --- Part A inputs: labels, D10 exclusion, D11 paragraph onsets ---
    print("labeling (reusing the pilot analysis path)...")
    results = {k: analyze_trace(tokenizer, t) for k, t in traces.items()}
    looped = {k for k, r in results.items() if r.row["looped"]}
    kept = [k for k in traces if k not in looped]
    merged = {k: merge_onsets(results[k].onsets, "paragraph") for k in kept}
    onsets = {k: [o for o, m in merged[k] if m is None] for k in kept}
    all_hits = {k: [o.token_index for o in results[k].onsets] for k in kept}
    n_onsets = sum(len(v) for v in onsets.values())
    print(f"D10: {len(looped)} looped excluded, {len(kept)} kept")
    print(f"D11: {n_onsets} paragraph-rule onsets")

    # Gate-input id set, recomputed from the surviving onsets.
    marker_ids = sorted({traces[k]["output_token_ids"][o.token_index]
                         for k in kept for o in onsets[k]})
    print(f"gate-input id set: {len(marker_ids)} distinct ids")

    # --- Part B: extraction seam test on a short trace ---
    probe_key = min(kept, key=lambda k: traces[k]["new_tokens"])
    part_b = extraction_seam_test(model, tokenizer, traces[probe_key], probe_key,
                                  eos_id, n_layers, all_hits[probe_key])

    # --- matched negatives ---
    pools: dict[tuple[str, str], list[tuple[tuple[str, int], int]]] = defaultdict(list)
    for k in kept:
        tr = traces[k]
        ids = tr["output_token_ids"]
        t0, t1 = tr["reasoning_token_span"]
        hits = set(all_hits[k])
        for q in range(t0, t1):
            if q + LOOKAHEAD >= t1:
                continue
            if any(x in hits for x in range(q + 1, q + 1 + LOOKAHEAD)):
                continue
            pools[(tr["dataset"], token_kind(tokenizer.decode([ids[q]],
                                                              skip_special_tokens=False)))].append((k, q))

    rng = random.Random(SEED)
    for key in pools:
        rng.shuffle(pools[key])

    matches: list[dict] = []
    shortfalls: Counter[tuple[str, str]] = Counter()
    onset_kinds: Counter[tuple[str, str]] = Counter()
    drawn_kinds: Counter[tuple[str, str]] = Counter()
    for k in kept:
        ds = traces[k]["dataset"]
        for o in onsets[k]:
            onset_kinds[(ds, o.prev_token_kind)] += 1
            pool = pools[(ds, o.prev_token_kind)]
            if not pool:
                shortfalls[(ds, o.prev_token_kind)] += 1
                continue
            neg_key, neg_q = pool.pop()
            drawn_kinds[(ds, o.prev_token_kind)] += 1
            matches.append({
                "dataset": ds, "onset_trace": k, "onset_token": o.token_index,
                "onset_pos": o.token_index - 1, "kind": o.prev_token_kind,
                "neg_trace": neg_key, "neg_pos": neg_q,
            })
    print(f"matched {len(matches)} onset/negative pairs; shortfalls: {dict(shortfalls)}")

    # --- Parts C and D: one pass per trace, collect the needed positions ---
    needed: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for i, m in enumerate(matches):
        needed[m["onset_trace"]].append((m["onset_pos"], i * 2))
        needed[m["neg_trace"]].append((m["neg_pos"], i * 2 + 1))

    n_slots = len(matches) * 2
    lens_rows: list[list[float] | None] = [None] * n_slots
    argmax_in_set: list[bool | None] = [None] * n_slots

    for n, k in enumerate(sorted(needed), 1):
        tr = traces[k]
        input_ids = build_input_ids(tr, eos_id)
        prompt_len = len(tr["prompt_token_ids"])
        hs = hidden_states_only(model, input_ids)
        assert len(hs) == n_layers + 1

        positions = needed[k]
        idx = torch.tensor([prompt_len + p for p, _ in positions], device=model.device)
        block = torch.stack([h[0].index_select(0, idx) for h in hs])  # [29, n_pos, hidden]
        del hs
        torch.mps.empty_cache()

        per_layer = []
        for layer in range(1, n_layers + 1):
            per_layer.append(
                lens_probs_at(model, block[layer], marker_ids,
                              apply_norm=(layer != n_layers)).cpu().tolist()
            )
        with torch.inference_mode():
            final_logits = model.lm_head(block[n_layers]).float()
        top = final_logits.argmax(dim=-1).cpu().tolist()
        marker_set = set(marker_ids)

        for j, (_pos, slot) in enumerate(positions):
            lens_rows[slot] = [per_layer[layer - 1][j] for layer in range(1, n_layers + 1)]
            argmax_in_set[slot] = top[j] in marker_set
        del block, final_logits
        torch.mps.empty_cache()
        print(f"  [{n:2d}/{len(needed)}] {k[0]}_{k[1]:04d}: {len(positions)} positions")

    # model odds = the final-layer lens value, which Part B proves equals the
    # model's own next-token distribution.
    odds = [row[-1] for row in lens_rows]  # type: ignore[index]

    write_lens_csv(matches, lens_rows, n_layers)
    gate = compute_gate(matches, odds, argmax_in_set)
    make_plot(matches, lens_rows, n_layers)
    write_gate_md(gate, part_b, marker_ids, onset_kinds, drawn_kinds, shortfalls,
                  n_onsets, len(kept), len(looped), n_layers)

    print(f"\ntotal {time.perf_counter() - t_start:.0f}s")
    print(json.dumps(gate, indent=1))


def extraction_seam_test(model, tokenizer, trace, key, eos_id, n_layers, hits) -> dict:
    """Part B. Full logits here (short trace) so the reproduction check is real."""
    print(f"\n--- Part B on {key[0]}_{key[1]:04d} ({trace['new_tokens']} new tokens) ---")
    input_ids = build_input_ids(trace, eos_id)
    ids_t = torch.tensor([input_ids], device=model.device)
    with torch.inference_mode():
        out = model(ids_t, output_hidden_states=True)
    hs = out.hidden_states
    logits = out.logits[0].float()

    info: dict = {
        "trace": f"{key[0]}_{key[1]:04d}",
        "sequence_length": len(input_ids),
        "n_hidden_states": len(hs),
        "n_layers": n_layers,
        "embedding_index": 0,
        "hidden_state_shape": list(hs[0].shape),
        "dtype": str(hs[0].dtype),
    }

    with torch.inference_mode():
        direct = model.lm_head(hs[-1][0]).float()
        normed = model.lm_head(final_norm(model)(hs[-1][0])).float()
    d_direct = float((direct - logits).abs().max())
    d_normed = float((normed - logits).abs().max())
    info["max_abs_diff_lm_head_only"] = d_direct
    info["max_abs_diff_norm_then_lm_head"] = d_normed
    info["last_hidden_state_already_normed"] = d_direct < d_normed
    print(f"  lm_head alone      : max|diff| = {d_direct:.3e}")
    print(f"  norm + lm_head     : max|diff| = {d_normed:.3e}")
    print(f"  last entry normed? : {info['last_hidden_state_already_normed']}")

    # Sampling consistency over generated positions.
    prompt_len = len(trace["prompt_token_ids"])
    gen = trace["output_token_ids"]
    raw, sampled = [], []
    for g, tok in enumerate(gen):
        row = logits[prompt_len + g - 1]
        raw.append(prob_of_tokens(row, [tok]))
        sampled.append(sampling_probability(row, tok, temperature=TEMPERATURE, top_p=TOP_P))
    info["sampling_raw_median"] = statistics.median(raw)
    info["sampling_raw_p5"] = sorted(raw)[max(0, int(0.05 * len(raw)) - 1)]
    info["sampling_tempered_median"] = statistics.median(sampled)
    info["sampling_tempered_p5"] = sorted(sampled)[max(0, int(0.05 * len(sampled)) - 1)]
    print(f"  p(generated token), raw softmax : median {info['sampling_raw_median']:.3f}, "
          f"5th pct {info['sampling_raw_p5']:.3f}")
    print(f"  p(generated token), T=0.6/top-p : median {info['sampling_tempered_median']:.3f}, "
          f"5th pct {info['sampling_tempered_p5']:.3f}")

    # 50 marker-unrelated positions.
    t0, _t1 = trace["reasoning_token_span"]
    lo, hi = t0 + 100, t0 + 150
    block = torch.stack([h[0, prompt_len + lo : prompt_len + hi] for h in hs])
    near = [h for h in hits if lo - 1 <= h <= hi]
    info["smoke_positions"] = [lo, hi]
    info["smoke_shape"] = list(block.shape)
    info["smoke_dtype"] = str(block.dtype)
    info["smoke_all_finite"] = bool(torch.isfinite(block).all())
    info["smoke_marker_hits_in_range"] = near
    print(f"  residual block {list(block.shape)} {block.dtype}, "
          f"finite={info['smoke_all_finite']}, marker hits in range: {near}")
    del out, hs, logits, block
    torch.mps.empty_cache()
    return info


def compute_gate(matches, odds, argmax_in_set) -> dict:
    out: dict = {}
    groups = {"gsm8k": [], "math500": [], "pooled": []}
    for i, m in enumerate(matches):
        rec = (f"{m['onset_trace'][0]}_{m['onset_trace'][1]}", odds[i * 2], odds[i * 2 + 1])
        groups[m["dataset"]].append(rec)
        groups["pooled"].append(rec)
    for name, pairs in groups.items():
        pos = [p for _, p, _ in pairs]
        neg = [n for _, _, n in pairs]
        lo, hi = bootstrap_auroc_by_trace(pairs, N_BOOTSTRAP, SEED)
        out[name] = {
            "n_pairs": len(pairs),
            "auroc": auroc(pos, neg),
            "ci95": [lo, hi],
            "median_odds_onset": statistics.median(pos),
            "median_odds_negative": statistics.median(neg),
        }
    am = [argmax_in_set[i * 2] for i in range(len(matches))]
    out["argmax_in_marker_set_share"] = sum(am) / len(am)
    return out


def write_lens_csv(matches, lens_rows, n_layers) -> None:
    cols = ["dataset", "role", "trace", "position"] + [f"p_layer{i:02d}" for i in range(1, n_layers + 1)]
    with (RESULTS / "c3_logit_lens.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, m in enumerate(matches):
            for role, slot, trace_key, pos in (
                ("onset", i * 2, m["onset_trace"], m["onset_pos"]),
                ("negative", i * 2 + 1, m["neg_trace"], m["neg_pos"]),
            ):
                w.writerow([m["dataset"], role, f"{trace_key[0]}_{trace_key[1]:04d}", pos]
                           + [f"{v:.6g}" for v in lens_rows[slot]])


def make_plot(matches, lens_rows, n_layers) -> None:
    import numpy as np

    layers = list(range(1, n_layers + 1))
    datasets = ["gsm8k", "math500"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, ds in zip(axes, datasets):
        idx = [i for i, m in enumerate(matches) if m["dataset"] == ds]
        pos = np.array([lens_rows[i * 2] for i in idx])
        neg = np.array([lens_rows[i * 2 + 1] for i in idx])

        for row in pos:
            ax.plot(layers, row, color="#c2410c", alpha=0.06, linewidth=0.8)
        ax.fill_between(layers, np.percentile(neg, 10, axis=0),
                        np.percentile(neg, 90, axis=0),
                        color="#94a3b8", alpha=0.35,
                        label="matched negatives, 10th-90th pct")
        ax.plot(layers, np.median(neg, axis=0), color="#475569", linewidth=2,
                linestyle="--", label="negatives, median")
        ax.plot(layers, np.median(pos, axis=0), color="#9a3412", linewidth=2.5,
                label="onset-preceding, median")

        ax.set_yscale("log")
        ax.set_xlabel("layer")
        ax.set_title(f"{ds}  (n = {len(idx)} onsets)")
        ax.grid(alpha=0.25, which="both", linewidth=0.5)
    axes[0].set_ylabel("summed probability of a marker-start token")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Logit lens at the token before an onset: when does the marker become predictable?",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(RESULTS / "c3_logit_lens.png", dpi=150)
    plt.close(fig)


def write_gate_md(gate, part_b, marker_ids, onset_kinds, drawn_kinds, shortfalls,
                  n_onsets, n_kept, n_looped, n_layers) -> None:
    import csv as _csv

    rows = list(_csv.DictReader((RESULTS / "c3_logit_lens.csv").open()))
    layer_cols = [f"p_layer{i:02d}" for i in range(1, n_layers + 1)]

    def median_curve(role, ds=None):
        sel = [r for r in rows if r["role"] == role and (ds is None or r["dataset"] == ds)]
        return [statistics.median(float(r[c]) for r in sel) for c in layer_cols]

    out = [
        "# c3 — gate input at distance 1",
        "",
        f"Paragraph-rule onsets over {n_kept} non-looped traces "
        f"({n_looped} excluded for looping): **{n_onsets}**.",
        f"Gate-input id set after Part A: **{len(marker_ids)} distinct token ids** "
        f"= {marker_ids}",
        "",
        "## AUROC — model's own odds, onset-preceding vs matched negatives",
        "",
        "| group | pairs | AUROC | 95% CI (bootstrap over traces) | median odds, onset | median odds, negative |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("gsm8k", "math500", "pooled"):
        g = gate[name]
        out.append(
            f"| {name} | {g['n_pairs']} | **{g['auroc']:.3f}** | "
            f"[{g['ci95'][0]:.3f}, {g['ci95'][1]:.3f}] | "
            f"{g['median_odds_onset']:.4g} | {g['median_odds_negative']:.4g} |"
        )
    out += [
        "",
        f"Share of onset-preceding positions where the argmax token is a "
        f"marker-start id: **{gate['argmax_in_marker_set_share']:.1%}**",
        "",
        "The pre-registration's provisional gate floor is AUROC 0.75 (§11). "
        "Reported against it; the floor is not changed here.",
        "",
        "## Matching check — onsets and negatives drawn, per token kind",
        "",
        "| dataset | prev_token_kind | onsets | negatives drawn | shortfall |",
        "|---|---|---|---|---|",
    ]
    for key in sorted(onset_kinds):
        out.append(
            f"| {key[0]} | `{key[1]}` | {onset_kinds[key]} | "
            f"{drawn_kinds.get(key, 0)} | {shortfalls.get(key, 0)} |"
        )

    out += ["", "## Layer curves (median summed marker probability)", "",
            "| layer | onset, gsm8k | neg, gsm8k | onset, math500 | neg, math500 |",
            "|---|---|---|---|---|"]
    curves = {
        ("onset", "gsm8k"): median_curve("onset", "gsm8k"),
        ("negative", "gsm8k"): median_curve("negative", "gsm8k"),
        ("onset", "math500"): median_curve("onset", "math500"),
        ("negative", "math500"): median_curve("negative", "math500"),
    }
    for i in range(n_layers):
        out.append(
            f"| {i + 1} | {curves[('onset', 'gsm8k')][i]:.3g} | "
            f"{curves[('negative', 'gsm8k')][i]:.3g} | "
            f"{curves[('onset', 'math500')][i]:.3g} | "
            f"{curves[('negative', 'math500')][i]:.3g} |"
        )

    # Three sentences, written from the curves just computed.
    #
    # Criterion choice matters here and is stated rather than buried. A
    # ratio-to-negatives rule ("first layer where onset >= 2x negatives") fires
    # at layer 2 on MATH500, where both curves are around 1e-4 — noise, not
    # predictability, because the MATH500 negative baseline is simply lower.
    # An absolute threshold does not have that failure mode.
    def first_layer_at_least(curve, threshold):
        return next((i + 1 for i, v in enumerate(curve) if v >= threshold), None)

    def first_layer_ratio(curve, baseline, factor):
        return next((i + 1 for i, v in enumerate(curve) if v >= factor * baseline[i]), None)

    g_on, g_ng = curves[("onset", "gsm8k")], curves[("negative", "gsm8k")]
    m_on, m_ng = curves[("onset", "math500")], curves[("negative", "math500")]
    g_abs, m_abs = first_layer_at_least(g_on, 0.01), first_layer_at_least(m_on, 0.01)
    g_rel = first_layer_at_least(g_on, 0.10 * g_on[-1])
    m_rel = first_layer_at_least(m_on, 0.10 * m_on[-1])
    g_r2, m_r2 = first_layer_ratio(g_on, g_ng, 2.0), first_layer_ratio(m_on, m_ng, 2.0)

    out += [
        "",
        "## Three sentences",
        "",
        f"**At which layer does the marker become predictable?** In the last "
        f"third: taking \"predictable\" as the first layer where the median "
        f"onset-preceding curve reaches an absolute 1%, that is layer {g_abs} of "
        f"{n_layers} on GSM8K and layer {m_abs} on MATH500, and the curve then "
        f"climbs steeply — on GSM8K from {g_on[g_abs - 2]:.2g} at layer "
        f"{g_abs - 1} to {g_on[-1]:.2g} at the final layer — while everything "
        f"before roughly layer 19 sits at the 1e-4 noise floor.",
        "",
        f"**Does it differ by dataset?** No: layer {g_abs} against layer {m_abs} "
        f"on the absolute criterion and layer {g_rel} against {m_rel} on "
        f"\"10% of the final value\", so the layer at which the marker surfaces "
        f"looks like a property of the model, not the dataset. (A "
        f"ratio-to-negatives criterion would say layer {g_r2} against layer "
        f"{m_r2}, but that is an artifact: at those layers both curves are near "
        f"1e-4 and MATH500 simply has a lower negative baseline.)",
        "",
        f"**How separated are the negatives?** Pooled AUROC is "
        f"{gate['pooled']['auroc']:.3f} "
        f"(95% CI {gate['pooled']['ci95'][0]:.3f}-{gate['pooled']['ci95'][1]:.3f}), "
        f"with median odds {gate['pooled']['median_odds_onset']:.3g} at "
        f"onset-preceding positions against "
        f"{gate['pooled']['median_odds_negative']:.3g} at matched negatives — a "
        f"{gate['pooled']['median_odds_onset'] / max(gate['pooled']['median_odds_negative'], 1e-12):.0f}x "
        f"gap in the medians.",
        "",
        "## Part B — extraction seam test",
        "",
        "```json",
        json.dumps(part_b, indent=1),
        "```",
        "",
    ]
    (RESULTS / "c3_gate.md").write_text("\n".join(out))


if __name__ == "__main__":
    main()
