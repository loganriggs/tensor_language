"""Exact 2x2 decomposition for simultaneous causal command interventions."""

from __future__ import annotations


CELLS = ("00", "10", "01", "11")


class JointCompositionError(ValueError):
    pass


def decompose(torch, cells):
    if set(cells) != set(CELLS):
        raise JointCompositionError("exact cells 00,10,01,11 are required")
    values = {name: torch.as_tensor(value).float() for name, value in cells.items()}
    shape = values["00"].shape
    if any(value.shape != shape for value in values.values()):
        raise JointCompositionError("all cells must have identical shape")
    temporal = values["10"] - values["00"]
    iswas = values["01"] - values["00"]
    interaction = values["11"] - values["10"] - values["01"] + values["00"]
    additive = values["00"] + temporal + iswas
    closure = additive + interaction - values["11"]
    both_effect = values["11"] - values["00"]
    scale = both_effect.square().mean().sqrt().clamp_min(1e-12)
    cosine_denominator = ((additive - values["00"]).norm() * both_effect.norm()).clamp_min(1e-12)
    return {
        "components": {"baseline": values["00"], "temporal": temporal, "iswas": iswas,
                       "interaction": interaction},
        "additive_prediction": additive,
        "closure_max_abs_error": float(closure.abs().max()),
        "interaction_rms_over_both_rms": float(interaction.square().mean().sqrt() / scale),
        "additive_cosine_to_both_effect": float(((additive - values["00"]) * both_effect).sum()
                                                 / cosine_denominator),
        "temporal_shapley": temporal + interaction / 2,
        "iswas_shapley": iswas + interaction / 2,
    }


def serializable(result):
    def convert(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value
    return convert(result)
