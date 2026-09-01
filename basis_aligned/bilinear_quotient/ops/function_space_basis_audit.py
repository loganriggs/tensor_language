"""FUNCTION-SPACE BASIS AUDIT (rung 280; CPU-only receipt analysis).

CONVENTION: every d-vector is signed per-position CE added above the real model;
lower aggregate CE is better. This analysis is registered after rungs 275-278
landed but BEFORE rung 279's clean `cev_purevalue.pt` exists.

Known evidence motivating, but not scored by, this audit:
  - band pair cosine(d_ct96, d_t120) = 0.9840;
  - contained value config raw cosine with band = 0.8660, but its residual
    d_v96-d_ct96 has cosine -0.017 with band;
  - contained m16-KO residual has cosine 0.041 with band and 0.053 with the
    value residual.

NEW REGISTERED PREDICTIONS, adjudicated by the not-yet-landed clean pure-value
arm (value-r96 with exact patterns):
  (a) CLEAN REPLICATION: cosine(d_purevalue, d_v96-d_ct96) >= 0.90 AND
      abs(cosine(d_purevalue, d_ct96)) < 0.30.
  (b) VECTOR ADDITIVITY: ||d_v96-d_ct96-d_purevalue|| / ||d_purevalue||
      <= 0.35.
  (c) LOW-RANK FAMILY ALGEBRA: after replacing the post-hoc value residual by
      the clean arm, max absolute off-diagonal cosine among
      {band-average, pure-value, m16-KO residual} <= 0.30; the band pair stays
      >= 0.95 after removing base-difficulty quantile means.

NULL: the contained-config subtraction does not transport to the clean arm
(cosine < 0.70 or relative error > 0.60), or damage families share a generic
direction (off-diagonal > 0.50). Passing is IDENTIFICATION of a small signed
damage algebra, not ADOPTION of a repair; a legal activation-derived repair
still needs held-out predictive, certificate, price, and intervention gates.

PRICE: zero deployed values/ops; CPU receipt analysis only. Self-reviewed.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "function_space_basis_audit_results.json"
NAMES = ("ct96", "t120", "v96", "kom16", "purevalue")


def cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(x, y, dim=0))


def difficulty_residual(x: torch.Tensor, base: torch.Tensor, bins: int = 100) -> torch.Tensor:
    """Remove a flexible nuisance mean as a function of native token difficulty."""
    out = x.clone()
    for idx in base.argsort().tensor_split(bins):
        out[idx] -= out[idx].mean()
    return out


def row_cosines(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Document-local cosine after removing each document's scalar mean."""
    x = x.reshape(-1, 256)
    y = y.reshape(-1, 256)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    return (x * y).sum(dim=1) / (x.norm(dim=1) * y.norm(dim=1)).clamp_min(1e-15)


def quantiles(x: torch.Tensor) -> list[float]:
    q = torch.tensor([0.025, 0.25, 0.5, 0.75, 0.975], dtype=x.dtype)
    return [round(float(v), 6) for v in torch.quantile(x, q)]


def main() -> None:
    required = [ROOT / "census_state_diverse.pt"] + [ROOT / f"cev_{n}.pt" for n in NAMES]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"WAIT: missing registered receipts: {missing}")

    state = torch.load(required[0], map_location="cpu", weights_only=False)
    base = state["basev"].reshape(-1).double()
    damage = {
        name: torch.load(ROOT / f"cev_{name}.pt", map_location="cpu", weights_only=False)
        .reshape(-1)
        .double()
        - base
        for name in NAMES
    }
    if len({x.numel() for x in damage.values()} | {base.numel()}) != 1:
        raise SystemExit("INSTRUMENT FAIL: receipt vector shapes differ")

    band_a = damage["ct96"]
    band_b = damage["t120"]
    band = 0.5 * (band_a + band_b)
    value_residual = damage["v96"] - band_a
    ko_residual = damage["kom16"] - band_a
    pure_value = damage["purevalue"]

    pure_value_residual_cos = cosine(pure_value, value_residual)
    pure_band_cos = cosine(pure_value, band)
    additivity_error = float((value_residual - pure_value).norm() / pure_value.norm())

    family = {"band": band, "pure_value": pure_value, "m16_ko_residual": ko_residual}
    family_names = list(family)
    gram = torch.empty((len(family_names), len(family_names)), dtype=torch.float64)
    for i, left in enumerate(family_names):
        for j, right in enumerate(family_names):
            gram[i, j] = cosine(family[left], family[right])
    offdiag = gram - torch.eye(len(family_names), dtype=gram.dtype)
    max_family_offdiag = float(offdiag.abs().max())

    q_band_a = difficulty_residual(band_a, base)
    q_band_b = difficulty_residual(band_b, base)
    difficulty_residual_band_cos = cosine(q_band_a, q_band_b)

    doc_band = row_cosines(band_a, band_b)
    doc_value = row_cosines(band_a, pure_value)
    doc_advantage = doc_band - doc_value
    gen = torch.Generator().manual_seed(280)
    bootstrap_idx = torch.randint(0, doc_advantage.numel(), (5000, doc_advantage.numel()), generator=gen)
    bootstrap_means = doc_advantage[bootstrap_idx].mean(dim=1)

    pred_a = pure_value_residual_cos >= 0.90 and abs(pure_band_cos) < 0.30
    pred_b = additivity_error <= 0.35
    pred_c = max_family_offdiag <= 0.30 and difficulty_residual_band_cos >= 0.95
    null_triggered = pure_value_residual_cos < 0.70 or additivity_error > 0.60 or max_family_offdiag > 0.50

    result = {
        "convention": "signed per-position CE added above real; lower aggregate is better",
        "registered_before_purevalue_landed": True,
        "n_positions": base.numel(),
        "damage_means": {k: round(float(v.mean()), 7) for k, v in damage.items()},
        "screen_band_pair_cosine": round(cosine(band_a, band_b), 7),
        "pure_value_vs_contained_value_residual_cosine": round(pure_value_residual_cos, 7),
        "pure_value_vs_band_cosine": round(pure_band_cos, 7),
        "vector_additivity_relative_error": round(additivity_error, 7),
        "family_names": family_names,
        "family_cosine_gram": [[round(float(x), 7) for x in row] for row in gram],
        "max_abs_family_offdiagonal": round(max_family_offdiag, 7),
        "difficulty_residualized_band_cosine": round(difficulty_residual_band_cos, 7),
        "document_band_cosine_quantiles": quantiles(doc_band),
        "document_pure_value_cosine_quantiles": quantiles(doc_value),
        "document_band_advantage_quantiles": quantiles(doc_advantage),
        "document_band_advantage_positive_fraction": round(float((doc_advantage > 0).double().mean()), 7),
        "document_band_advantage_bootstrap_mean_ci95": [
            round(float(x), 7)
            for x in torch.quantile(
                bootstrap_means,
                torch.tensor([0.025, 0.5, 0.975], dtype=bootstrap_means.dtype),
            )
        ],
        "pred_a_clean_replication": bool(pred_a),
        "pred_b_vector_additivity": bool(pred_b),
        "pred_c_low_rank_family_algebra": bool(pred_c),
        "null_triggered": bool(null_triggered),
        "decision_level": "identification screen; adoption requires a legal held-out repair",
    }
    # Post-outcome correction, reported separately from the registered scores above.  The full-rank
    # path control landed only after this audit was registered and exposed a shared instrument vector.
    # It must not silently rewrite the preregistered verdict, but it does change the scientific object.
    path_file = ROOT / "cev_pathfull.pt"
    if path_file.exists():
        path_damage = torch.load(path_file, map_location="cpu", weights_only=False).reshape(-1).double() - base
        corrected_band_a = band_a - path_damage
        corrected_band_b = band_b - path_damage
        corrected_band = 0.5 * (corrected_band_a + corrected_band_b)
        corrected_pure = pure_value - path_damage
        corrected_family = {
            "band_after_path": corrected_band,
            "pure_value_after_path": corrected_pure,
            "m16_ko_residual": ko_residual,
        }
        corrected_names = list(corrected_family)
        corrected_gram = torch.empty((3, 3), dtype=torch.float64)
        for i, left in enumerate(corrected_names):
            for j, right in enumerate(corrected_names):
                corrected_gram[i, j] = cosine(corrected_family[left], corrected_family[right])
        corrected_offdiag = corrected_gram - torch.eye(3, dtype=torch.float64)
        result["posthoc_path_control_addendum"] = {
            "status": "exploratory recalibration; path control landed after preregistration",
            "path_damage_mean": round(float(path_damage.mean()), 7),
            "path_cosine_with_ct96": round(cosine(path_damage, band_a), 7),
            "path_cosine_with_t120": round(cosine(path_damage, band_b), 7),
            "corrected_band_means": {
                "ct96_minus_path": round(float(corrected_band_a.mean()), 7),
                "t120_minus_path": round(float(corrected_band_b.mean()), 7),
            },
            "corrected_band_pair_cosine": round(cosine(corrected_band_a, corrected_band_b), 7),
            "pure_value_vs_contained_value_residual_cosine": round(
                cosine(corrected_pure, value_residual), 7
            ),
            "vector_additivity_relative_error": round(
                float((value_residual - corrected_pure).norm() / corrected_pure.norm()), 7
            ),
            "pure_value_vs_corrected_band_cosine": round(cosine(corrected_pure, corrected_band), 7),
            "family_names": corrected_names,
            "family_cosine_gram": [
                [round(float(x), 7) for x in row] for row in corrected_gram
            ],
            "max_abs_family_offdiagonal": round(float(corrected_offdiag.abs().max()), 7),
            "interpretation": (
                "the common 0.055 mode is path-dominated; after subtraction, damage families are "
                "approximately orthogonal but ct96/t120 no longer support a rank-one floor"
            ),
        }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
