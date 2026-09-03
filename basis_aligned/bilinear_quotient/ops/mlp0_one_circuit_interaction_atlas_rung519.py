#!/usr/bin/env python3
"""RUNG519 -- exact MLP0 interaction partners for one circuit.

pred_a: exact/live 49-term decomposition, whole-drop replay, and planted recovery
pred_b: 1--8 semantic terms are target-specific on both discovery halves
pred_c: at least one frozen term confirms on new documents and all 62 circuits
pred_d: the finite subset response law transfers and recovers the parent effect
pred_e: joint removal selectively changes the target without off-target task damage

BQGATE: EXPERIMENT
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
OPS = ROOT / "basis_aligned/bilinear_quotient/ops"
PREREG = POLY / "MLP0_ONE_CIRCUIT_INTERACTION_ATLAS_RUNG519_PREREGISTRATION.md"
PREREG_SHA256 = "ce351753c1a2d5cabb8f6bddf21b103e9c841d7be9003eece49d10246b68603b"
R518_RESULT = ROOT / "basis_aligned/bilinear_quotient/mlp0_head_relation_circuit_quotient_rung518_results.json"
R518_BUNDLE = ROOT / "basis_aligned/bilinear_quotient/mlp0_head_relation_circuit_quotient_rung518_bundle.pt"
R518_SOURCE = OPS / "mlp0_head_relation_circuit_quotient_rung518.py"
R518_PREREG = POLY / "MLP0_HEAD_RELATION_CIRCUIT_QUOTIENT_RUNG518_PREREGISTRATION.md"
CIRCUIT_INDEX = ROOT / "basis_aligned/bilinear_quotient/CIRCUITS_INDEX.md"
HASHES = {
    R518_RESULT: "52e4d3677713a8cfa8ec2064e071a19dbb6534d71764338f7f26ecef3ea3f623",
    R518_BUNDLE: "fe9851946cdc8248cf9ea151d768589f886a1e41576c56748148ff6d24565329",
    R518_SOURCE: "6294a208fdd0a4facdb93929305296bacbbcc2dc83e59ce376697cc67cd71b65",
    R518_PREREG: "54ee23d84dcb515917b563690aef1c6c8e0a53909cabda59088825404ad7e382",
    CIRCUIT_INDEX: "e3e510bbf549c851efcd818169650f0e28b3866a22ae4a8d856fd66de87e87a0",
}
TARGET_CIRCUIT = "r.2.0.2"
SELECTED_ATOM_NAME = "H4.DISTANT_SAME"
SELECTED_ATOM = 4 * 5 + 3
TARGET_EFFECTS = (0.012520589941432902, 0.003909140586171755,
                  0.01718100727561911, 0.004190039411971824)
CONTROL_SEEDS = tuple(range(519100, 519116))
DISCOVERY = (500, 748, 624)
CONFIRMATION = (752, 1000, 876)
N_ATOMS = 45
N_NORMALIZED_SOURCES = 47
N_BILINEAR_TERMS = 47
N_TERMS = 49
NUMERICAL_SOURCE = 46
NORMALIZATION_TERM = 47
DEPLOYMENT_ROUNDING_TERM = 48


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(value.float(), weight.float())


def term_names(atom_names: tuple[str, ...]) -> tuple[str, ...]:
    if len(atom_names) != N_ATOMS or atom_names[SELECTED_ATOM] != SELECTED_ATOM_NAME:
        raise ValueError("rung518 atom vocabulary changed")
    partners = ("TOKEN",) + atom_names + ("NUMERICAL",)
    names = []
    selected_source = 1 + SELECTED_ATOM
    for index, partner in enumerate(partners):
        names.append("SELF" if index == selected_source else f"WITH::{partner}")
    names.extend(("NORMALIZATION", "DEPLOYMENT_ROUNDING"))
    return tuple(names)


def normalized_sources(token_base: torch.Tensor, atoms: torch.Tensor,
                       normalized: torch.Tensor) -> torch.Tensor:
    """Return TOKEN, 45 atoms, and one exact numerical closing source."""
    if atoms.shape[0] != N_ATOMS or token_base.shape != normalized.shape \
            or atoms.shape[1:] != normalized.shape:
        raise ValueError("normalized-source shapes changed")
    raw = torch.cat((token_base.float().unsqueeze(0), atoms.float()), 0)
    raw_sum = raw.sum(0)
    gain = (normalized.float() * raw_sum).sum(-1, keepdim=True) \
        / raw_sum.square().sum(-1, keepdim=True).clamp_min(1e-30)
    semantic = raw * gain.unsqueeze(0)
    numerical = normalized.float() - semantic.sum(0)
    return torch.cat((semantic, numerical.unsqueeze(0)), 0)


def _float_mlp(mlp, state: torch.Tensor) -> torch.Tensor:
    hidden = _linear(state, mlp.Left.weight) * _linear(state, mlp.Right.weight)
    return _linear(hidden, mlp.Down.weight) + mlp.Down_bias.float()


def interaction_terms(mlp, sources: torch.Tensor, normalized_drop: torch.Tensor,
                      deployed_full: torch.Tensor,
                      deployed_drop: torch.Tensor) -> dict:
    """Split one source's deployed removal difference into 47+2 exact terms."""
    if sources.shape[0] != N_NORMALIZED_SOURCES:
        raise ValueError("expected TOKEN + 45 atoms + NUMERICAL")
    selected = 1 + SELECTED_ATOM
    z_full = sources.sum(0)
    z_fixed_drop = z_full - sources[selected]
    left = _linear(sources, mlp.Left.weight)
    right = _linear(sources, mlp.Right.weight)
    outputs = []
    for partner in range(N_NORMALIZED_SOURCES):
        hidden = left[selected] * right[partner]
        if partner != selected:
            hidden = hidden + left[partner] * right[selected]
        outputs.append(_linear(hidden, mlp.Down.weight))
    fixed_full = _float_mlp(mlp, z_full)
    fixed_drop = _float_mlp(mlp, z_fixed_drop)
    renormalized_drop = _float_mlp(mlp, normalized_drop)
    outputs.append(fixed_drop - renormalized_drop)
    outputs.append(
        (deployed_full.float() - deployed_drop.float())
        - (fixed_full - renormalized_drop))
    terms = torch.stack(outputs)
    target = deployed_full.float() - deployed_drop.float()
    denominator = target.double().square().sum().clamp_min(1e-30)
    fixed_target = fixed_full - fixed_drop
    return {
        "terms": terms,
        "target": target,
        "normalized_source_relative_squared": float(
            (z_full.double() - sources.double().sum(0)).square().sum()
            / z_full.double().square().sum().clamp_min(1e-30)),
        "fixed_gain_relative_squared": float(
            (terms[:N_BILINEAR_TERMS].sum(0).double() - fixed_target.double())
            .square().sum() / fixed_target.double().square().sum().clamp_min(1e-30)),
        "deployed_relative_squared": float(
            (terms.sum(0).double() - target.double()).square().sum() / denominator),
    }


def select_atom_from_r518(bundle: dict) -> dict:
    names = tuple(bundle["collections"]["discovery"]["arms"])
    atom_names = tuple(name.removeprefix("SINGLE::")
                       for name in names[2:2 + N_ATOMS])
    responses = bundle["discovery_responses"]
    tags = tuple(bundle["discovery_tags"])
    target = tags.index(TARGET_CIRCUIT)
    scored = []
    for atom, name in enumerate(atom_names):
        values = tuple(float(responses[half]["circuit"][atom, background, target])
                       for half in ("half0", "half1") for background in (0, 1))
        same_sign = min(values) > 0 or max(values) < 0
        score = min(abs(value) for value in values) if same_sign else float("-inf")
        scored.append((score, atom, name, values))
    winner = max(scored)
    return {"atom": winner[1], "name": winner[2], "effects": winner[3],
            "minimum_absolute_effect": winner[0]}


def _rank_and_ratio(vector: torch.Tensor, target: int) -> tuple[int, float]:
    absolute = vector.double().abs()
    rank = 1 + int((absolute > absolute[target]).sum())
    ratio = float(absolute[target] / absolute.median().clamp_min(1e-30))
    return rank, ratio


def discover_terms(effects: torch.Tensor, whole: torch.Tensor, target: int,
                   rank_limit: int = 4) -> list[dict]:
    """Apply the frozen two-half target-recovery and specificity rules."""
    if effects.ndim != 3 or effects.shape[0] != N_TERMS or effects.shape[1] != 2:
        raise ValueError("term effects must be [49,2,circuit]")
    if whole.shape != effects.shape[1:]:
        raise ValueError("whole-atom effects do not match term effects")
    candidates = []
    for term in range(N_BILINEAR_TERMS):
        if term == NUMERICAL_SOURCE:
            continue
        recoveries = effects[term, :, target] / whole[:, target].clamp_min(1e-30)
        ranks, ratios = zip(*(_rank_and_ratio(effects[term, half], target)
                              for half in range(2)))
        positive = bool((recoveries >= .15).all())
        stable = positive and float(recoveries.max() / recoveries.min()) <= 2
        holds = stable and max(ranks) <= rank_limit and min(ratios) >= 2
        if holds:
            candidates.append({
                "term": term, "recoveries": recoveries.tolist(),
                "target_ranks": list(ranks), "target_to_median": list(ratios),
            })
    return candidates


def permutation_control_counts(effects: torch.Tensor, whole: torch.Tensor,
                               target: int) -> list[int]:
    dimensions = effects.shape[-1]
    counts = []
    for seed in CONTROL_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        keys = torch.rand(N_TERMS, dimensions, generator=generator)
        order = keys.argsort(-1)[:, None].expand(-1, 2, -1)
        shuffled = torch.gather(effects, -1, order)
        counts.append(len(discover_terms(shuffled, whole, target)))
    return counts


def mobius(table: torch.Tensor) -> torch.Tensor:
    """Boolean-lattice inversion along the subset axis."""
    values = table.clone().double()
    subsets = values.shape[0]
    if subsets < 1 or subsets & (subsets - 1):
        raise ValueError("subset table length must be a power of two")
    bits = subsets.bit_length() - 1
    for bit in range(bits):
        for mask in range(subsets):
            if mask & (1 << bit):
                values[mask] -= values[mask ^ (1 << bit)]
    return values


def planted_problem(seed: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    generator = torch.Generator().manual_seed(seed)
    effects = torch.randn(N_TERMS, 2, 32, generator=generator, dtype=torch.float64) * .002
    whole = torch.ones(2, 32, dtype=torch.float64)
    planted = [0, 5, 17]
    for offset, term in enumerate(planted):
        effects[term, :, 0] = .25 + .03 * offset
        effects[term, :, 1:] *= .25
    effects[NUMERICAL_SOURCE, :, 0] = .5
    effects[NORMALIZATION_TERM, :, 0] = .5
    effects[DEPLOYMENT_ROUNDING_TERM, :, 0] = .5
    return effects, whole, planted


def planted_suite() -> dict:
    cases = []
    for seed in range(51900, 51908):
        effects, whole, expected = planted_problem(seed)
        found = [row["term"] for row in discover_terms(effects, whole, 0)]
        coefficients = torch.zeros(8, dtype=torch.float64)
        coefficients[1], coefficients[2], coefficients[4] = .2, -.1, .3
        coefficients[3], coefficients[7] = .4, -.25
        table = torch.zeros_like(coefficients)
        for mask in range(8):
            table[mask] = sum(coefficients[sub]
                              for sub in range(8) if sub & ~mask == 0)
        exact_mobius = bool(torch.allclose(mobius(table), coefficients, atol=1e-12))
        controls_zero = permutation_control_counts(effects, whole, 0) == [0] * 16
        cases.append({"seed": seed, "expected": expected, "found": found,
                      "candidate_exact": found == expected,
                      "mobius_exact": exact_mobius,
                      "controls_zero": controls_zero})
    return {"cases": cases, "all_eight_exact": all(
        row["candidate_exact"] and row["mobius_exact"] and row["controls_zero"]
        for row in cases)}


def validate_inputs() -> dict:
    if sha256(PREREG) != PREREG_SHA256:
        raise RuntimeError("rung519 preregistration changed after source freeze")
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen dependency hash mismatch: {path}")
    result = json.loads(R518_RESULT.read_text())
    if not (result.get("pred_a_exact_live_45_piece_instrument") is True
            and result.get("pred_b_small_circuit_defined_relation") is False
            and result.get("strong_null") is True
            and result.get("next_step")
            == "leave_head_relation_basis_for_one_circuit_exact_interaction_atlas"):
        raise RuntimeError("rung518 route changed")
    bundle = torch.load(R518_BUNDLE, map_location="cpu", weights_only=False)
    selected = select_atom_from_r518(bundle)
    if selected["atom"] != SELECTED_ATOM or selected["name"] != SELECTED_ATOM_NAME \
            or any(abs(left - right) > 1e-12
                   for left, right in zip(selected["effects"], TARGET_EFFECTS)):
        raise RuntimeError("frozen rung518 atom selection changed")
    return {"selected": selected, "term_names": term_names(tuple(result["atom_names"]))}


def dry_run() -> dict:
    validated = validate_inputs()
    planted = planted_suite()
    if not planted["all_eight_exact"]:
        raise RuntimeError("rung519 planted recovery failed")
    return {
        "status": "dry_run_passed", "rung": 519,
        "model_loaded": False, "model_outcomes_opened": False,
        "target_circuit": TARGET_CIRCUIT,
        "selected_atom": validated["selected"],
        "normalized_sources": N_NORMALIZED_SOURCES,
        "bilinear_terms": N_BILINEAR_TERMS, "all_terms": N_TERMS,
        'pred_a_exact_live_interaction_instrument': None,
        'pred_b_small_circuit_specific_bilinear_support': None,
        'pred_c_heldout_term_identification': None,
        'pred_d_predictable_finite_composition': None,
        'pred_e_selective_target_circuit_manipulation': None,
        "planted_recovery": planted,
    }


def scientific_main() -> None:
    validate_inputs()
    pred_a = False
    pred_b = False
    pred_c = False
    pred_d = False
    pred_e = False
    _ = (pred_a, pred_b, pred_c, pred_d, pred_e)
    raise RuntimeError("rung519 model path is fail-closed pending deployed term collector")


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in os.sys.argv:
        print(json.dumps(dry_run(), indent=2, sort_keys=True))
        return
    scientific_main()


if __name__ == "__main__":
    main()
