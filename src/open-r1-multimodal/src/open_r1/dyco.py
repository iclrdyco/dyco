"""Token-level DyCo-RL statistics and advantage reweighting.

All normalizations are per response. These helpers do not mix examples or
rollout groups. Token position zero is the first generated token (including
its predicting attention step), not the first token of the input prompt.
"""

import math

import torch


def _check_matrix(value, name):
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, tokens].")


def minmax_attention_mass(mass, token_mask):
    """Normalize one modality over valid generated positions in each response.

    A constant mass has no within-response contrast and is mapped to zero.
    Padding is excluded from both extrema and the returned scores.
    """
    _check_matrix(mass, "mass")
    if mass.shape != token_mask.shape:
        raise ValueError("mass and token_mask must have identical shapes.")
    mass = mass.float()
    token_mask = token_mask.bool()
    if not torch.isfinite(mass[token_mask]).all() or (mass[token_mask] < 0).any():
        raise ValueError("Valid attention masses must be finite and non-negative.")
    if mass.shape[1] == 0:
        return mass
    minimum = mass.masked_fill(~token_mask, float("inf")).amin(dim=1, keepdim=True)
    maximum = mass.masked_fill(~token_mask, float("-inf")).amax(dim=1, keepdim=True)
    has_tokens = token_mask.any(dim=1, keepdim=True)
    minimum = torch.where(has_tokens, minimum, torch.zeros_like(minimum))
    maximum = torch.where(has_tokens, maximum, torch.zeros_like(maximum))
    span = maximum - minimum
    denominator = torch.where(span > 0, span, torch.ones_like(span))
    normalized = (mass - minimum) / denominator
    return torch.where(token_mask & (span > 0), normalized, torch.zeros_like(normalized))


def alignment_metadata(visual_mass, text_mass, visual_fr, text_fr, token_mask, tau=0.05):
    """Assign symmetric-margin roles and select sequence-normalized mass.

    All inputs have shape [batch, generated_tokens]. The first generated
    position has no preceding generated-token pair and is kept neutral.
    Role IDs are +1 (visual), -1 (text), and 0 (neutral or padding).
    """
    if not math.isfinite(tau) or tau < 0:
        raise ValueError("dyco_tau must be finite and non-negative.")
    _check_matrix(token_mask, "token_mask")
    for value in (visual_mass, text_mass, visual_fr, text_fr):
        if value.shape != token_mask.shape:
            raise ValueError("DyCo statistics must match the generated-token mask exactly.")
    token_mask = token_mask.bool()
    for distance in (visual_fr, text_fr):
        if not torch.isfinite(distance[token_mask]).all():
            raise ValueError("Fisher--Rao distances must be finite.")
    visual_score = minmax_attention_mass(visual_mass, token_mask)
    text_score = minmax_attention_mass(text_mass, token_mask)
    difference = visual_fr.float() - text_fr.float()
    visual = (difference > tau) & token_mask
    text = (difference < -tau) & token_mask
    if token_mask.shape[1]:
        visual[:, 0] = False
        text[:, 0] = False
    roles = visual.long() - text.long()
    return {
        "alignment_scores": visual_score * visual + text_score * text,
        "role_mask": visual | text,
        "role_ids": roles,
        "token_mask": token_mask,
    }


def role_weights(alignment_scores, role_mask, token_mask=None):
    """Masked softmax with mean weight one over each non-neutral token set.

    The empty set yields zero weights. Neutral and padding positions never
    enter the softmax denominator, including neutral positions with score 0.
    """
    _check_matrix(alignment_scores, "alignment_scores")
    if alignment_scores.shape != role_mask.shape:
        raise ValueError("alignment_scores and role_mask must have identical shapes.")
    active = role_mask.bool()
    if token_mask is not None:
        if token_mask.shape != active.shape:
            raise ValueError("token_mask must match alignment_scores.")
        active = active & token_mask.bool()
    scores = alignment_scores.float()
    if not torch.isfinite(scores[active]).all():
        raise ValueError("Active alignment scores must be finite.")
    if scores.shape[1] == 0:
        return scores
    count = active.sum(dim=1, keepdim=True)
    maximum = scores.masked_fill(~active, float("-inf")).amax(dim=1, keepdim=True)
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    exponentials = torch.exp(scores.masked_fill(~active, float("-inf")) - maximum)
    return count * exponentials / exponentials.sum(dim=1, keepdim=True).clamp_min(1e-12)


def reweight_token_advantages(
    advantages, alignment_scores, role_mask, token_mask=None, alpha=0.2, centered=False
):
    """Apply the paper's signed token-level factor; never modify the reward.

    Positive advantages are increased and negative advantages attenuated more
    strongly at higher-scoring active tokens. The optional centered ablation
    substitutes (w - 1) for w on those tokens. Neutral advantages are unchanged;
    padding returns zero. The factor is applied exactly as written, without
    clipping. Large alpha can reverse negative-advantage signs; the default
    alpha=0.2 and scores in [0, 1] keep factors positive because w < exp(1).
    """
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("dyco_alpha must be finite and non-negative.")
    weights = role_weights(alignment_scores, role_mask, token_mask)
    if advantages.ndim == 1 and advantages.shape[0] == weights.shape[0]:
        base = advantages[:, None].expand_as(weights)
    elif advantages.shape == weights.shape:
        base = advantages
    else:
        raise ValueError("advantages must have shape [batch] or [batch, tokens].")
    active = role_mask.bool()
    if token_mask is not None:
        active = active & token_mask.bool()
    perturbation = weights - 1.0 if centered else weights
    factor = 1.0 + alpha * base.sign() * perturbation
    result = torch.where(active, factor * base, base)
    if token_mask is not None:
        result = result.masked_fill(~token_mask.bool(), 0.0)
    return result
