"""Teacher-forced extraction and the logit lens, at distance 1 only.

Pre-freeze rule (CLAUDE.md): this module reads the model's own next-token
probabilities and each layer's residual through the model's own final norm and
unembedding, at the token immediately before an onset and at matched negative
positions. It trains nothing. There is no probe, no classifier, no fitted
parameter of any kind here — `lm_head` and the final norm are the model's own
weights, used unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")

import torch  # noqa: E402


def build_input_ids(trace: dict[str, Any], eos_id: int) -> list[int]:
    """Prompt + generated ids, restoring the EOS that generation stripped."""
    ids = list(trace["prompt_token_ids"]) + list(trace["output_token_ids"])
    if trace["eos_token_stripped"]:
        ids = ids + [eos_id]
    return ids


def hidden_states_only(model: Any, input_ids: list[int]) -> tuple[torch.Tensor, ...]:
    """Forward pass through the base model, skipping the lm_head.

    Calling `model(...)` would materialize logits for every position — about
    1.3 GB for a 4k-token trace at this vocabulary size. We only need a handful
    of positions, and Part B establishes that lm_head over the last residual
    reproduces those logits exactly, so we apply lm_head ourselves afterwards.
    """
    ids = torch.tensor([input_ids], device=model.device)
    with torch.inference_mode():
        out = model.model(ids, output_hidden_states=True)
    return out.hidden_states


def final_norm(model: Any) -> Any:
    return model.model.norm


def lens_probs_at(
    model: Any,
    residuals: torch.Tensor,
    marker_ids: list[int],
    *,
    apply_norm: bool,
) -> torch.Tensor:
    """Summed probability over `marker_ids`, for a [n_pos, hidden] residual block.

    `apply_norm` is False for a residual that already has the model's final
    norm applied (the last hidden_states entry on this architecture).
    """
    h = residuals
    if apply_norm:
        h = final_norm(model)(h)
    with torch.inference_mode():
        logits = model.lm_head(h).float()
        probs = torch.softmax(logits, dim=-1)
        return probs[:, marker_ids].sum(dim=-1)


def prob_of_tokens(logits_row: torch.Tensor, token_ids: list[int]) -> float:
    probs = torch.softmax(logits_row.float(), dim=-1)
    return float(probs[token_ids].sum())


def sampling_probability(
    logits_row: torch.Tensor, token_id: int, *, temperature: float, top_p: float
) -> float:
    """Probability of `token_id` under temperature + top-p, as generation used.

    Returns 0.0 if the token falls outside the nucleus.
    """
    scaled = logits_row.float() / temperature
    probs = torch.softmax(scaled, dim=-1)
    order = torch.argsort(probs, descending=True)
    sorted_probs = probs[order]
    cumulative = torch.cumsum(sorted_probs, dim=0)
    # Keep everything up to and including the first index that reaches top_p.
    keep = cumulative < top_p
    keep[0] = True
    cutoff = int(keep.sum())
    kept_ids = order[:cutoff]
    if token_id not in set(kept_ids.tolist()):
        return 0.0
    renormalized = sorted_probs[:cutoff] / sorted_probs[:cutoff].sum()
    position = (kept_ids == token_id).nonzero()[0, 0]
    return float(renormalized[position])


def auroc(positive: list[float], negative: list[float]) -> float:
    """Rank-based AUROC. Not a fitted model — a summary of two score sets."""
    from sklearn.metrics import roc_auc_score

    y = [1] * len(positive) + [0] * len(negative)
    return float(roc_auc_score(y, positive + negative))


def bootstrap_auroc_by_trace(
    pairs: list[tuple[str, float, float]], n_resamples: int, seed: int
) -> tuple[float, float]:
    """95% interval for AUROC, resampling whole traces with replacement.

    `pairs` is (trace_key, onset_score, matched_negative_score). Resampling at
    the trace level rather than the position level respects that positions
    within a trace are not independent.
    """
    import random

    import numpy as np

    by_trace: dict[str, list[tuple[float, float]]] = {}
    for key, pos, neg in pairs:
        by_trace.setdefault(key, []).append((pos, neg))
    keys = list(by_trace)

    rng = random.Random(seed)
    scores: list[float] = []
    for _ in range(n_resamples):
        drawn = [keys[rng.randrange(len(keys))] for _ in keys]
        p: list[float] = []
        n: list[float] = []
        for k in drawn:
            for a, b in by_trace[k]:
                p.append(a)
                n.append(b)
        if p and n:
            scores.append(auroc(p, n))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)
