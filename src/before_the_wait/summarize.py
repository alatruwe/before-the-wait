"""Build the three c1 results files from saved traces.

Reads traces only — never the model. Text and counts, nothing else.
"""
from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

from before_the_wait.align import prefix_lengths
from before_the_wait.label import (
    LOOP_REPEATS,
    LOOP_WINDOW,
    TOKEN_KINDS,
    HardCheckFailure,
    Onset,
    is_lexicon_gram,
    is_looped,
    label_trace,
    negative_pool,
    paragraph_initial_ngrams,
)
from before_the_wait.lexicon import _LEXICON

MARKER_COLUMNS = [m.lower() for m in _LEXICON]

TRACE_COLUMNS = [
    "dataset", "index", "finish_reason", "new_tokens", "reasoning_tokens",
    "reasoning_complete", "tokens_per_second", "correct", "looped",
    "marker_count", "markers_per_1k_reasoning_tokens",
] + MARKER_COLUMNS

ONSET_COLUMNS = [
    "dataset", "index", "token_index", "marker", "char_start",
    "prev_token_text", "prev_token_kind", "paragraph_initial",
    "tokens_since_prev_hit", "tokens_to_next_hit", "same_paragraph_as_prev_hit",
    # Not in the handoff's list: recorded because the review's proposed
    # hard-check fix turns on how common mid-token onsets are.
    "onset_mid_token",
]


@dataclass
class TraceResult:
    row: dict[str, Any]
    onsets: list[Onset]
    failures: list[HardCheckFailure]
    neg100: Counter[str]
    neg100_skipped: int
    neg60: Counter[str]
    neg60_skipped: int
    first_hit_distance: int | None
    onset_token_ids: Counter[int]
    para_unigrams: Counter[str]
    para_bigrams: Counter[str]
    mismatch: dict[str, Any] | None


def analyze_trace(tokenizer: Any, trace: dict[str, Any]) -> TraceResult:
    ids: list[int] = trace["output_token_ids"]
    output_text: str = trace["output_text"]
    c0, c1 = trace["reasoning_char_span"]
    t0, t1 = trace["reasoning_token_span"]

    prefix_len = prefix_lengths(tokenizer, ids)
    onsets, failures = label_trace(
        tokenizer,
        dataset=trace["dataset"],
        index=trace["index"],
        ids=ids,
        output_text=output_text,
        prefix_len=prefix_len,
        reasoning_char_span=(c0, c1),
        reasoning_token_span=(t0, t1),
    )

    reasoning_tokens = max(0, t1 - t0)
    reasoning_text = output_text[c0:c1]
    marker_counts = Counter(o.marker for o in onsets)
    onset_idx = {o.token_index for o in onsets}

    neg100, skip100 = negative_pool(tokenizer, ids, (t0, t1), onset_idx, 100)
    neg60, skip60 = negative_pool(tokenizer, ids, (t0, t1), onset_idx, 60)

    row: dict[str, Any] = {
        "dataset": trace["dataset"],
        "index": trace["index"],
        "finish_reason": trace["finish_reason"],
        "new_tokens": trace["new_tokens"],
        "reasoning_tokens": reasoning_tokens,
        "reasoning_complete": trace["reasoning_complete"],
        "tokens_per_second": round(trace["tokens_per_second"], 2),
        "correct": trace["correct"],
        "looped": is_looped(ids[t0:t1]),
        "marker_count": len(onsets),
        "markers_per_1k_reasoning_tokens": (
            round(1000 * len(onsets) / reasoning_tokens, 2) if reasoning_tokens else 0.0
        ),
    }
    for m in MARKER_COLUMNS:
        row[m] = marker_counts.get(m, 0)

    mismatch = None
    if not trace["correct"]:
        mismatch = {
            "dataset": trace["dataset"],
            "index": trace["index"],
            "gold": trace["gold_answer"],
            "boxed": trace["boxed_answer"],
        }

    return TraceResult(
        row=row,
        onsets=onsets,
        failures=failures,
        neg100=neg100,
        neg100_skipped=skip100,
        neg60=neg60,
        neg60_skipped=skip60,
        first_hit_distance=(onsets[0].token_index - t0) if onsets else None,
        onset_token_ids=Counter(ids[o.token_index] for o in onsets),
        para_unigrams=paragraph_initial_ngrams(reasoning_text, 1),
        para_bigrams=paragraph_initial_ngrams(reasoning_text, 2),
        mismatch=mismatch,
    )


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "n/a"


def _share_table(counter: Counter[str], total: int) -> list[str]:
    lines = ["| value | count | share |", "|---|---|---|"]
    for key, n in counter.most_common():
        lines.append(f"| `{key}` | {n} | {_pct(n, total)} |")
    return lines


def _describe(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"mean {statistics.mean(values):.2f}, median {statistics.median(values):.2f}, "
        f"min {min(values):.2f}, max {max(values):.2f}"
    )


PARAGRAPH_ONSET_COLUMNS = ONSET_COLUMNS + ["merged_into"]


def build_addendum(
    kept: list[TraceResult],
    looped: list[TraceResult],
    merged: dict[tuple[str, int], list[tuple[Onset, int | None]]],
    window_counts: dict[int, int],
) -> str:
    """Addendum: D10 exclusions and D11 hits -> onsets, per dataset and pooled."""
    out = [
        "",
        "---",
        "",
        "# Addendum — D10 and D11 applied (2026-10-17)",
        "",
        "D10: looping traces excluded from all onset-level outputs. D11: onsets",
        "by the paragraph rule — the first hit in a paragraph is the onset, later",
        "hits in that paragraph merge into it.",
        "",
        "## D10 exclusions",
        "",
        f"- traces excluded for looping: **{len(looped)}** "
        f"({', '.join(f'{r.row['dataset']}_{r.row['index']}' for r in looped)})",
        f"- traces kept: **{len(kept)}**",
        "",
    ]

    groups: list[tuple[str, list[TraceResult]]] = []
    by_ds: dict[str, list[TraceResult]] = {}
    for r in kept:
        by_ds.setdefault(r.row["dataset"], []).append(r)
    groups += sorted(by_ds.items())
    groups.append(("pooled", kept))

    out += ["## D11 — hits to onsets (paragraph rule)", "",
            "| group | traces | hits | onsets | merged | onsets per trace |",
            "|---|---|---|---|---|---|"]
    for name, group in groups:
        keys = [(r.row["dataset"], r.row["index"]) for r in group]
        pairs = [p for k in keys for p in merged[k]]
        hits = len(pairs)
        onsets = sum(1 for _, m in pairs if m is None)
        out.append(
            f"| {name} | {len(group)} | {hits} | **{onsets}** | {hits - onsets} | "
            f"{onsets / len(group):.1f} |"
        )
    out.append("")

    out += ["## Per-marker shares of onsets (paragraph rule)", ""]
    for name, group in groups:
        keys = [(r.row["dataset"], r.row["index"]) for r in group]
        onsets = [o for k in keys for o, m in merged[k] if m is None]
        counts = Counter(o.marker for o in onsets)
        out += [f"**{name}** ({len(onsets)} onsets)", ""]
        out += _share_table(counts, len(onsets))
        out.append("")

    out += [
        "## Window rule (parked — counted once, for the record)",
        "",
        "Not used anywhere in this experiment. A hit is an onset unless the",
        "previous hit in the same trace, merged or not, is within N tokens",
        "(difference <= N), per the pre-registration's fixed wording.",
        "",
        "| N | onsets, non-looped traces |",
        "|---|---|",
    ]
    for n in sorted(window_counts):
        out.append(f"| {n} | {window_counts[n]} |")
    out.append("")
    return "\n".join(out)


def build_summary(
    results: list[TraceResult], by_dataset: dict[str, list[TraceResult]], meta: dict[str, Any]
) -> str:
    out: list[str] = ["# c1 pilot summary", ""]
    out.append(f"Model `{meta['model_id']}` @ `{meta['model_revision']}`, fp16, mps.")
    out.append(
        f"Generated {meta['n_traces']} traces "
        f"({', '.join(f'{k} {len(v)}' for k, v in by_dataset.items())}), "
        f"temperature 0.6 / top-p 0.95, max 4096 new tokens, one sample per problem."
    )
    out.append("")
    out.append(f"Generation-prompt suffix: `{meta['prompt_suffix']}`")
    out.append(
        f"`<think>` token ids {meta['think_ids']}, `</think>` token ids {meta['close_ids']}."
    )
    out.append(f"Hard check: {meta['hard_check_total']} hits checked, "
               f"{meta['hard_check_failures']} failures.")
    out.append("")

    groups: list[tuple[str, list[TraceResult]]] = list(by_dataset.items()) + [("pooled", results)]

    for name, group in groups:
        out.append(f"## {name}")
        out.append("")
        onsets = [o for r in group for o in r.onsets]
        n = len(group)

        # 1. Traces
        truncated = sum(1 for r in group if r.row["finish_reason"] == "length")
        incomplete = sum(1 for r in group if not r.row["reasoning_complete"])
        looped = sum(1 for r in group if r.row["looped"])
        correct = sum(1 for r in group if r.row["correct"])
        out += [
            "### 1. Traces",
            "",
            f"- n = {n}",
            f"- truncated (`length`): {truncated} ({_pct(truncated, n)})",
            f"- reasoning incomplete (no `</think>`): {incomplete} ({_pct(incomplete, n)})",
            f"- looped: {looped} ({_pct(looped, n)})",
            f"- accuracy: {correct}/{n} ({_pct(correct, n)})",
            "",
        ]

        # 2. Density
        per_trace = [float(r.row["marker_count"]) for r in group]
        per_1k = [float(r.row["markers_per_1k_reasoning_tokens"]) for r in group]
        reasoning_total = sum(r.row["reasoning_tokens"] for r in group)
        out += [
            "### 2. Density",
            "",
            f"- hits per trace: {_describe(per_trace)}",
            f"- hits per 1k reasoning tokens (per trace): {_describe(per_1k)}",
            f"- pooled: {len(onsets)} hits over {reasoning_total} reasoning tokens = "
            f"{1000 * len(onsets) / reasoning_total:.2f} per 1k"
            if reasoning_total
            else "- pooled: no reasoning tokens",
            "",
        ]

        # 3. Per-marker counts
        marker_counts = Counter(o.marker for o in onsets)
        out += ["### 3. Per-marker counts", ""]
        out += _share_table(marker_counts, len(onsets))
        wait_n = marker_counts.get("wait", 0)
        out += ["", f"**Share of hits that are `wait`: {_pct(wait_n, len(onsets))}** "
                    f"({wait_n}/{len(onsets)})", ""]

        # 4. Paragraph-initial and prev_token_kind
        para = sum(1 for o in onsets if o.paragraph_initial)
        kinds = Counter(o.prev_token_kind for o in onsets)
        mid = sum(1 for o in onsets if o.onset_mid_token)
        out += [
            "### 4. Position",
            "",
            f"- paragraph-initial: {para}/{len(onsets)} ({_pct(para, len(onsets))})",
            f"- mid-token onsets: {mid}/{len(onsets)} ({_pct(mid, len(onsets))})",
            "",
            "`prev_token_kind`:",
            "",
        ]
        out += _share_table(kinds, len(onsets))
        out.append("")

        # 5. Gaps
        gaps = [o.tokens_since_prev_hit for o in onsets if o.tokens_since_prev_hit is not None]
        firsts = [r.first_hit_distance for r in group if r.first_hit_distance is not None]
        out += ["### 5. Gaps", "", f"Hits with a previous hit in the same trace: {len(gaps)}", ""]
        out += ["| threshold | share of gaps >= | share of first-hit distances >= |", "|---|---|---|"]
        for thr in (5, 10, 20, 40):
            g = sum(1 for x in gaps if x >= thr)
            f = sum(1 for x in firsts if x >= thr)
            out.append(f"| {thr} | {_pct(g, len(gaps))} | {_pct(f, len(firsts))} |")
        out += ["", f"First-hit distance from reasoning start: {_describe([float(x) for x in firsts])}", ""]

        # 6. Clustering
        clustered = sum(1 for o in onsets if o.same_paragraph_as_prev_hit)
        eligible = sum(1 for o in onsets if o.same_paragraph_as_prev_hit is not None)
        out += [
            "### 6. Clustering",
            "",
            f"- hits with another hit earlier in the same paragraph: {clustered}/{eligible} "
            f"({_pct(clustered, eligible)})",
            "",
        ]

        # 7. Negatives pool
        neg100: Counter[str] = Counter()
        neg60: Counter[str] = Counter()
        skip100 = skip60 = 0
        for r in group:
            neg100 += r.neg100
            neg60 += r.neg60
            skip100 += r.neg100_skipped
            skip60 += r.neg60_skipped
        out += [
            "### 7. Negatives pool",
            "",
            f"- positions with no hit in the next 100 tokens: **{sum(neg100.values())}** "
            f"(excluded for insufficient lookahead: {skip100})",
            f"- positions with no hit in the next 60 tokens: **{sum(neg60.values())}** "
            f"(excluded for insufficient lookahead: {skip60})",
            "",
            "By token kind at that position:",
            "",
            "| kind | no hit in next 100 | no hit in next 60 |",
            "|---|---|---|",
        ]
        for kind in TOKEN_KINDS:
            out.append(f"| `{kind}` | {neg100.get(kind, 0)} | {neg60.get(kind, 0)} |")
        out.append("")

        # 8. Gate input
        ids_counter: Counter[int] = Counter()
        for r in group:
            ids_counter += r.onset_token_ids
        out += [
            "### 8. Gate input — generated token ids that begin a marker",
            "",
            f"{len(ids_counter)} distinct token ids.",
            "",
            "| token id | count |",
            "|---|---|",
        ]
        for tid, c in ids_counter.most_common():
            out.append(f"| {tid} | {c} |")
        out.append("")

        # 9. Lexicon candidates
        uni: Counter[str] = Counter()
        bi: Counter[str] = Counter()
        for r in group:
            uni += r.para_unigrams
            bi += r.para_bigrams
        uni_miss = [(g, c) for g, c in uni.most_common() if not is_lexicon_gram(g)][:30]
        bi_miss = [(g, c) for g, c in bi.most_common() if not is_lexicon_gram(g)][:30]
        out += [
            "### 9. Paragraph openers the lexicon does not match",
            "",
            "Listed only — nothing is added to the lexicon.",
            "",
            "| rank | word | n | bigram | n |",
            "|---|---|---|---|---|",
        ]
        for i in range(max(len(uni_miss), len(bi_miss))):
            w, wc = uni_miss[i] if i < len(uni_miss) else ("", "")
            b, bc = bi_miss[i] if i < len(bi_miss) else ("", "")
            out.append(f"| {i + 1} | `{w}` | {wc} | `{b}` | {bc} |")
        out.append("")

        # 10. Speed
        speeds = [float(r.row["tokens_per_second"]) for r in group]
        out += ["### 10. Speed", "", f"- tokens/second: {_describe(speeds)}", ""]

    return "\n".join(out)
