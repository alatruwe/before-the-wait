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
# §4 requires N to satisfy these three; 21-40 is reported whatever its count.
REQUIRED_BANDS = ("2-5", "6-10", "11-20")


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


def minimum_events(delta: float, auc: float = 0.70, interval: float = INTERVAL,
                   max_m: int = 1_000_000) -> int:
    """§11: smallest m whose interval half-width on the lift is <= delta.

    SE_lift = SE of one AUROC, which is what an assumed correlation of 0.5
    between the probe's and the rival's AUROC gives:
    Var(A-B) = Var(A) + Var(B) - 2*rho*SD(A)*SD(B) = Var(A) when the two have
    equal variance and rho = 0.5.
    """
    z = z_for_interval(interval)
    for m in range(2, max_m):
        if z * hanley_mcneil_se(auc, m, m) <= delta:
            return m
    raise ValueError(f"no m below {max_m} reaches delta={delta}")


def trace_count(m: int, r: float, loop_share: float, onsets_per_trace: float,
                eligibility: float) -> int:
    """§4: smallest integer N with N * r * (1-l) * o * e_b >= m."""
    per_trace = r * (1 - loop_share) * onsets_per_trace * eligibility
    if per_trace <= 0:
        raise ValueError("non-positive onsets per trace")
    return math.ceil(m / per_trace)


def projected_validation_onsets(n: int, loop_share: float, onsets_per_trace: float,
                                validation_share: float = 0.15) -> float:
    """§4 split rule: N * (1-l) * o * 0.15."""
    return n * (1 - loop_share) * onsets_per_trace * validation_share


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
