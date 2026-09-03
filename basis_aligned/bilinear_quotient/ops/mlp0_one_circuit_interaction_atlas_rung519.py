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
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
OPS = ROOT / "basis_aligned/bilinear_quotient/ops"
PREREG = POLY / "MLP0_ONE_CIRCUIT_INTERACTION_ATLAS_RUNG519_PREREGISTRATION.md"
PREREG_SHA256 = "d6209f8c0c017a5bd331c09a2bcdb037f39d139b622b8bb17fb3c26b0a126f49"
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
TERM_NAMES = ()
ARMS = ("NATIVE", "WHOLE_ATOM_DROP", "TERM_SUM_DROP") \
    + tuple(f"REMOVE_TERM::{index}" for index in range(N_TERMS))
BATCH = 4
D = 1152


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


def circuit_mask_hashes(circuit_masks: dict, tags: tuple[str, ...]) -> dict[str, str]:
    hashes = {}
    for tag in tags:
        digest = hashlib.sha256()
        for key in ("member", "slice_control"):
            value = circuit_masks[tag][key].to(torch.uint8).contiguous().numpy().tobytes()
            digest.update(value)
        hashes[tag] = digest.hexdigest()
    return hashes


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


@torch.no_grad()
def collect_phase(model, rows, task_masks, circuit_masks, circuit_tags,
                  bounds, facade, r517, r518, response_parent) -> dict:
    """Measure native, whole-drop replay, and all 49 finite term removals."""
    lo, hi, _split = bounds
    documents = hi - lo
    task_cells = response_parent.TASK_CELLS
    task_sums = torch.zeros(len(ARMS), documents, len(task_cells), dtype=torch.float64)
    task_counts = torch.zeros(documents, len(task_cells), dtype=torch.float64)
    circuit_sums = torch.zeros(len(ARMS), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    diagnostics = {
        "calls": 0, "calls_expected": ((hi - lo) // BATCH) * len(ARMS),
        "maximum_normalized_source_relative_squared": 0.0,
        "maximum_fixed_gain_relative_squared": 0.0,
        "maximum_deployed_relative_squared": 0.0,
        "maximum_whole_drop_logit_replay_error": 0.0,
        "maximum_whole_drop_logit_relative_squared": 0.0,
        "minimum_term_edit_rms": float("inf"),
    }
    device = next(model.parameters()).device
    block0 = model.transformer.h[0]
    for start in range(lo, hi, BATCH):
        stop = start + BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        attention_state = F.rms_norm(token_base, (D,))
        split = r517.attention0_source_writes(block0, attention_state, tokens)
        decomposition = r518.head_relation_atoms(block0, split)
        full_state = F.rms_norm(token_base + split["native_write"], (D,))
        drop_context = r518.atom_context(
            split["native_write"], decomposition, SELECTED_ATOM, "DROP")
        drop_state = F.rms_norm(token_base + drop_context, (D,))
        full_write = block0.mlp(full_state)
        drop_write = block0.mlp(drop_state)
        sources = normalized_sources(token_base, decomposition["atoms"], full_state)
        term_decomposition = interaction_terms(
            block0.mlp, sources, drop_state, full_write, drop_write)
        diagnostics["maximum_normalized_source_relative_squared"] = max(
            diagnostics["maximum_normalized_source_relative_squared"],
            term_decomposition["normalized_source_relative_squared"])
        diagnostics["maximum_fixed_gain_relative_squared"] = max(
            diagnostics["maximum_fixed_gain_relative_squared"],
            term_decomposition["fixed_gain_relative_squared"])
        diagnostics["maximum_deployed_relative_squared"] = max(
            diagnostics["maximum_deployed_relative_squared"],
            term_decomposition["deployed_relative_squared"])
        terms = term_decomposition["terms"]
        term_sum_drop = (full_write.float() - terms.sum(0)).to(full_write.dtype)
        term_writes = [(full_write.float() - term).to(full_write.dtype) for term in terms]
        for write in term_writes:
            diagnostics["minimum_term_edit_rms"] = min(
                diagnostics["minimum_term_edit_rms"],
                float((write.float() - full_write.float()).square().mean().sqrt()))

        def attention_dispatch(event):
            if event.site == 0:
                return split["native_write"], split["first_value"]
            return event.block.attn(event.state, event.first_value)

        nll_rows = []
        native_logits = facade.forward_with_dispatch(
            model, tokens,
            lambda event: event.block.attn(event.state, event.first_value),
            lambda event: event.block.mlp(event.state))
        diagnostics["calls"] += 1
        nll_rows.append(response_parent._nll(native_logits, batch_rows).cpu())
        edited_writes = [drop_write, term_sum_drop] + term_writes
        whole_logits = None
        for arm_index, site0_write in enumerate(edited_writes, start=1):
            def mlp_dispatch(event, site0_write=site0_write):
                return site0_write if event.site == 0 else event.block.mlp(event.state)

            logits = facade.forward_with_dispatch(
                model, tokens, attention_dispatch, mlp_dispatch)
            diagnostics["calls"] += 1
            if arm_index == 1:
                whole_logits = logits
            elif arm_index == 2:
                diagnostics["maximum_whole_drop_logit_replay_error"] = max(
                    diagnostics["maximum_whole_drop_logit_replay_error"],
                    float((logits - whole_logits).abs().max()))
                diagnostics["maximum_whole_drop_logit_relative_squared"] = max(
                    diagnostics["maximum_whole_drop_logit_relative_squared"],
                    float((logits.double() - whole_logits.double()).square().sum()
                          / whole_logits.double().square().sum().clamp_min(1e-30)))
            nll_rows.append(response_parent._nll(logits, batch_rows).cpu())
        nll = torch.stack(nll_rows)
        local_masks = {cell: task_masks[cell][start:stop] for cell in task_cells}
        task_sums[:, local:local + BATCH] = response_parent._task_sums(nll, local_masks)
        task_counts[local:local + BATCH] = torch.stack(
            [local_masks[cell].sum(1).double() for cell in task_cells], -1)
        matrix, observed = response_parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        circuit_counts += observed
        circuit_sums += torch.matmul(nll.reshape(len(ARMS), -1).double(), matrix.T).view(
            len(ARMS), 2, 2, len(circuit_tags))
    return {
        "bounds": tuple(bounds), "arms": ARMS,
        "task_sums": task_sums, "task_counts": task_counts,
        "circuit_sums": circuit_sums, "circuit_counts": circuit_counts,
        "diagnostics": diagnostics,
    }


def phase_effects(collection: dict) -> dict:
    """Return finite member-minus-control effects for whole drop and 49 terms."""
    if tuple(collection["arms"]) != ARMS:
        raise ValueError("rung519 arm order changed")
    circuits = collection["circuit_sums"] / collection["circuit_counts"].clamp_min(1)
    tasks = []
    circuit_halves = []
    whole_circuit = []
    whole_task = []
    for half_name in ("half0", "half1"):
        half = int(half_name == "half1")
        circuit_change = circuits[:, half, 0] - circuits[:, half, 1]
        circuit_change = circuit_change - circuit_change[0]
        circuit_halves.append(circuit_change[3:])
        whole_circuit.append(circuit_change[1])
        rel_lo, rel_hi = _half_bounds(tuple(collection["bounds"]), half_name)
        denominator = collection["task_counts"][rel_lo:rel_hi].sum(0).clamp_min(1)
        task_mean = collection["task_sums"][:, rel_lo:rel_hi].sum(1) / denominator
        task_change = task_mean - task_mean[0]
        tasks.append(task_change[3:])
        whole_task.append(task_change[1])
    return {
        "circuit": torch.stack(circuit_halves, 1),
        "whole_circuit": torch.stack(whole_circuit),
        "task": torch.stack(tasks, 1),
        "whole_task": torch.stack(whole_task),
    }


def _half_bounds(bounds: tuple[int, int, int], half: str) -> tuple[int, int]:
    lo, hi, split = bounds
    absolute = (lo, split) if half == "half0" else (split, hi)
    return absolute[0] - lo, absolute[1] - lo


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


@torch.no_grad()
def gpu_smoke() -> None:
    """Exercise one 52-arm batch without retaining task or circuit outcomes."""
    sys.path[:0] = [str(OPS), str(OPS.parent), str(POLY)]
    import bilin18_observed_model_facade as facade
    import mlp0_source_relation_factorial_rung517 as r517
    import mlp0_head_relation_circuit_quotient_rung518 as r518

    validate_inputs()
    rows, task_masks, circuit_masks, _scales, discovery_tags, _confirmation_tags, \
        _metadata, response_parent = r518.validate_inputs()
    smoke_tags = (TARGET_CIRCUIT, next(tag for tag in discovery_tags
                                      if tag != TARGET_CIRCUIT))
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    collection = collect_phase(
        model, rows, task_masks, circuit_masks, smoke_tags,
        (500, 504, 502), facade, r517, r518, response_parent)
    effects = phase_effects(collection)
    diagnostics = collection["diagnostics"]
    checks = {
        "weights": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256,
        "normalized_source_closure":
            diagnostics["maximum_normalized_source_relative_squared"] <= 1e-10,
        "fixed_gain_closure":
            diagnostics["maximum_fixed_gain_relative_squared"] <= 1e-8,
        "deployed_closure": diagnostics["maximum_deployed_relative_squared"] <= 1e-8,
        "whole_drop_logit_replay":
            diagnostics["maximum_whole_drop_logit_relative_squared"] <= 1e-8,
        "all_term_edits_live": diagnostics["minimum_term_edit_rms"] > 0,
        "call_census": diagnostics["calls"] == diagnostics["calls_expected"] == 52,
        "response_shapes": effects["circuit"].shape == (49, 2, 2)
            and effects["task"].shape[-2:] == (2, len(response_parent.TASK_CELLS)),
        "planted_recovery": planted_suite()["all_eight_exact"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"rung519 managed smoke failed: {checks}; {diagnostics}")
    print(json.dumps({
        "status": "gpu_smoke_passed", "rung": 519,
        "scientific_outcomes_retained": False,
        "checkpoint": checkpoint.__dict__, "checks": checks,
        "diagnostics": diagnostics,
    }, indent=2, sort_keys=True))


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
    if os.environ.get("BQLIB_GPU_SMOKE") == "1" or "--gpu-smoke" in os.sys.argv:
        gpu_smoke()
        return
    scientific_main()


if __name__ == "__main__":
    main()
