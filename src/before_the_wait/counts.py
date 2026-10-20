"""Count calculations for the pre-registration's §4, §6 and §11 rules.

Pure arithmetic over the pilot counts. No model, no activations, no probe —
the only model artifact touched is the tokenizer, to decode individual tokens
for their kind and to measure how many tokens a marker's text spans.
"""
from __future__ import annotations

import math

# Two-sided 98.75% interval (95% with a 4-band correction, §10).
INTERVAL = 0.9875

BANDS: list[tuple[str, int, int]] = [
    ("2-5", 2, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-40", 21, 40),
]
# §4 as amended by D12 (2026-10-17): all four bands must be satisfied. It
# previously required only the first three, with 21-40 reported whatever its
# count — which left that band unable to return a null.
REQUIRED_BANDS = ("2-5", "6-10", "11-20", "21-40")

# D12: the rule's N is inflated by this margin and rounded up to the nearest
# ROUND_TO, because o is estimated from only 13 pilot traces.
MARGIN = 1.2
ROUND_TO = 10
CAP = 1000


def z_for_interval(interval: float = INTERVAL) -> float:
    """z for a two-sided interval of the given coverage."""
    from scipy.stats import norm

    return float(norm.ppf(1 - (1 - interval) / 2))


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Standard error of an AUROC, Hanley & McNeil (1982).

    Q1 = A / (2 - A) is the probability that two randomly chosen positives
    both rank above a negative; Q2 = 2A^2 / (1 + A) the mirror case.
    """
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    numerator = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc**2)
        + (n_neg - 1) * (q2 - auc**2)
    )
    return math.sqrt(numerator / (n_pos * n_neg))


# §11 as revised 2026-10-19: the assumed AUROC and correlation behind m. They
# were 0.70 and 0.5 (which made SE_lift equal one AUROC's SE).
ASSUMED_AUC = 0.60
ASSUMED_RHO = 0.30


def lift_se(auc: float, n_pos: int, n_neg: int, rho: float = ASSUMED_RHO) -> float:
    """SE of the difference of two AUROCs with equal variance and correlation rho.

    Var(A-B) = Var(A) + Var(B) - 2*rho*SD(A)*SD(B) = 2*(1-rho)*Var(A).
    """
    return math.sqrt(2 * (1 - rho)) * hanley_mcneil_se(auc, n_pos, n_neg)


def minimum_events(delta: float, auc: float = ASSUMED_AUC, rho: float = ASSUMED_RHO,
                   interval: float = INTERVAL, max_m: int = 1_000_000) -> int:
    """§11: smallest m whose interval half-width on the lift is <= delta.

    Uses the Hanley-McNeil SE on m positives and m negatives for each AUROC.
    """
    z = z_for_interval(interval)
    for m in range(2, max_m):
        if z * lift_se(auc, m, m, rho) <= delta:
            return m
    raise ValueError(f"no m below {max_m} reaches delta={delta}")


def trace_count(m: int, r: float, loop_share: float, onsets_per_trace: float,
                eligibility: float) -> int:
    """§4: smallest integer N with N * r * (1-l) * o * e_b >= m."""
    per_trace = r * (1 - loop_share) * onsets_per_trace * eligibility
    if per_trace <= 0:
        raise ValueError("non-positive onsets per trace")
    return math.ceil(m / per_trace)


def apply_margin(n_rule: int, margin: float = MARGIN, round_to: int = ROUND_TO) -> int:
    """§4/D12: ceil(margin * N_rule), then round up to the nearest `round_to`.

    Both steps round up, so an exact multiple is left alone: 150 -> 180, not
    190.
    """
    inflated = math.ceil(n_rule * margin)
    return math.ceil(inflated / round_to) * round_to


def projected_validation_onsets(n: int, loop_share: float, onsets_per_trace: float,
                                validation_share: float = 0.15) -> float:
    """§4 split rule: N * (1-l) * o * 0.15."""
    return n * (1 - loop_share) * onsets_per_trace * validation_share


def gap_to_previous_hit(onset: int, hit_indices: list[int], reasoning_start: int) -> int:
    """§4 e_b: tokens from the previous lexicon hit of any kind to the onset.

    A merged hit counts as much as an onset (§5/§6 band-positive rule,
    2026-10-19): no position before it can be a positive for a later onset.
    The reasoning start stands in when there is no earlier hit.
    """
    prev = max((h for h in hit_indices if h < onset), default=reasoning_start)
    return onset - prev


def classify_band_position(pos: int, onset: int, hit_indices: list[int],
                           spans: list[tuple[int, int]], reasoning_start: int) -> str | None:
    """Why `pos` is not a band positive for `onset`, or None if it is one.

    §5/§6 as revised 2026-10-19: a position is a band positive only if no
    lexicon hit of any kind, onset or merged, lies strictly between it and its
    onset. That also makes the onset the nearest one, so each position belongs
    to at most one onset. Checked in order: before the reasoning start, inside a
    marker span, a hit in between.
    """
    if pos < reasoning_start:
        return "before_reasoning"
    if any(s <= pos < e for s, e in spans):
        return "in_marker"
    if any(pos < h < onset for h in hit_indices):
        return "hit_between"
    return None


def band_positives(onset: int, lo: int, hi: int, hit_indices: list[int],
                   spans: list[tuple[int, int]], reasoning_start: int) -> list[int]:
    """Positions at distances lo..hi before `onset` that are band positives."""
    return [onset - d for d in range(lo, hi + 1)
            if classify_band_position(onset - d, onset, hit_indices, spans,
                                      reasoning_start) is None]


def marker_token_span(tokenizer, ids: list[int], token_index: int,
                      marker: str, max_tokens: int = 12) -> tuple[int, int]:
    """Tokens [start, end) whose decoded text covers the marker's characters.

    Decodes forward from the onset token until the decoded text (after
    stripping the leading whitespace the hard check also strips) is at least
    as long as the marker.
    """
    for k in range(1, max_tokens + 1):
        if token_index + k > len(ids):
            break
        text = tokenizer.decode(ids[token_index : token_index + k],
                                skip_special_tokens=False).lstrip()
        if len(text) >= len(marker):
            return token_index, token_index + k
    return token_index, min(token_index + 1, len(ids))
