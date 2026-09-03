#!/usr/bin/env python3
"""RUNG516 -- circuit coordinates that stably force consumer-term splits.

# BQGATE: EXPERIMENT
# pred_a: exact zero-pair route and rung515 screen replay
# pred_b: circuits reject a material population that is task-compatible
# pred_c: eight half0-selected circuit witnesses transfer to half1 and beat controls
# pred_d: independently selected half0/half1 witness identities are stable

This is a conditional, CPU-only analysis.  The rung515 receipt and bundle hashes are
filled by a no-outcome preflight after the managed rung lands.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops", POLY, ROOT.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import attention11_mlp11_finite_downstream_term_quotient_rung515 as r515


PREREG = POLY / "MLP10_CONSUMER_CIRCUIT_SEPARATION_COVER_RUNG516_PREREGISTRATION.md"
R515_RESULT = ROOT / "attention11_mlp11_finite_downstream_term_quotient_rung515_results.json"
R515_BUNDLE = ROOT / "attention11_mlp11_finite_downstream_term_quotient_rung515_bundle.pt"
OUT = ROOT / "mlp10_consumer_circuit_separation_cover_rung516_results.json"

PREREG_SHA256 = "580cd5c3ca072e34dcf34e19aadb05b754bff89cdfe6968631d1235d8bb47b9b"
# Filled only after rung515 is terminal; analysis cannot run while either value is pending.
R515_RESULT_SHA256 = "0e33c6e39bd329a90dbc5cf178b335801f7b9ba69bad35a39a40d93d58f95599"
R515_BUNDLE_SHA256 = "294a1489db2c90466b6a90261981805f96be885ef8904213418ebada44b6b6b7"

CONTROL_SEEDS = tuple(range(51600, 51616))
PLANTED_SEEDS = tuple(range(51680, 51688))
COVER_SIZES = (1, 2, 4, 8, 16, 32)
ENERGY_FRACTION = .10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def predicted_metrics(left: torch.Tensor, right: torch.Tensor,
                      beta: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """All pairwise metrics; independent reimplementation of rung515."""
    dot = left @ right.T
    left_norm = torch.linalg.vector_norm(left, dim=1)
    right_norm = torch.linalg.vector_norm(right, dim=1)
    cosine = dot / (left_norm[:, None] * right_norm[None, :]).clamp_min(1e-30)
    cosine = cosine * beta.sign()
    forward = (
        left_norm[:, None].square() + beta.square() * right_norm[None, :].square()
        - 2 * beta * dot
    ).clamp_min(0).sqrt() / left_norm[:, None].clamp_min(1e-30)
    inverse = torch.where(beta.abs() > 1e-30, beta.reciprocal(),
                          torch.full_like(beta, math.inf))
    backward = (
        right_norm[None, :].square() + inverse.square() * left_norm[:, None].square()
        - 2 * inverse * dot
    ).clamp_min(0).sqrt() / right_norm[None, :].clamp_min(1e-30)
    return cosine, forward, backward


def build_pair_table(matrices: dict) -> dict:
    """Reconstruct the full registered pair screen and retain task-compatible rows."""
    rows = {key: [] for key in (
        "name", "beta", "left0", "right0", "left1", "right1",
        "task_compatible", "full0_split", "full1_split", "quality")}
    material_nodes = set()
    candidate_names = []
    block_top = []

    for subset in range(r515.N_SUBSETS):
        for relation, (left_action, right_action) in enumerate(r515.r513.RELATION_ACTIONS):
            for site, names in r515.SITE_TERMS.items():
                offset = r515.SITE_OFFSETS[site]
                indices = slice(offset, offset + len(names))
                c0_left = matrices["half0"]["circuit"][subset, left_action, indices]
                c0_right = matrices["half0"]["circuit"][subset, right_action, indices]
                dot = c0_left @ c0_right.T
                beta = dot / c0_right.square().sum(-1)[None, :].clamp_min(1e-30)
                safe_beta = torch.where(beta.abs() > 1e-30, beta, torch.ones_like(beta))

                metrics = {}
                for window in ("half0", "half1"):
                    for kind in ("circuit", "task"):
                        left = matrices[window][kind][subset, left_action, indices]
                        right = matrices[window][kind][subset, right_action, indices]
                        metrics[(window, kind)] = predicted_metrics(left, right, safe_beta)

                pooled_c = matrices["pooled"]["circuit"][subset, :, indices]
                pooled_t = matrices["pooled"]["task"][subset, :, indices]
                left_c_rms = pooled_c[left_action].square().mean(-1).sqrt()
                right_c_rms = pooled_c[right_action].square().mean(-1).sqrt()
                left_t_norm = torch.linalg.vector_norm(pooled_t[left_action], dim=-1)
                right_t_norm = torch.linalg.vector_norm(pooled_t[right_action], dim=-1)
                for action, c_rms, t_norm in (
                    (left_action, left_c_rms, left_t_norm),
                    (right_action, right_c_rms, right_t_norm),
                ):
                    for term in ((c_rms >= .0005) & (t_norm >= .00025)).nonzero().flatten().tolist():
                        material_nodes.add((subset, action, site, term))

                material = (
                    (left_c_rms[:, None] >= .0005) & (right_c_rms[None, :] >= .0005)
                    & (left_t_norm[:, None] >= .00025) & (right_t_norm[None, :] >= .00025))
                scale = (beta.abs() >= .25) & (beta.abs() <= 4)
                c0_cos, c0_f, c0_b = metrics[("half0", "circuit")]
                c1_cos, c1_f, c1_b = metrics[("half1", "circuit")]
                t0_cos, t0_f, t0_b = metrics[("half0", "task")]
                t1_cos, t1_f, t1_b = metrics[("half1", "task")]
                task_ok = (
                    material & scale
                    & (t0_cos >= .70) & (t0_f <= .65) & (t0_b <= .65)
                    & (t1_cos >= .70) & (t1_f <= .65) & (t1_b <= .65))
                circuit0_ok = (c0_cos >= .90) & (c0_f <= .35) & (c0_b <= .35)
                circuit1_ok = (c1_cos >= .80) & (c1_f <= .50) & (c1_b <= .50)
                candidate = task_ok & circuit0_ok & circuit1_ok
                quality = torch.stack((
                    c0_cos - .90, .35 - c0_f, .35 - c0_b,
                    c1_cos - .80, .50 - c1_f, .50 - c1_b,
                    t0_cos - .70, .65 - t0_f, .65 - t0_b,
                    t1_cos - .70, .65 - t1_f, .65 - t1_b,
                )).amin(0)

                values, flat = torch.topk(
                    quality.flatten().nan_to_num(nan=-math.inf), min(3, quality.numel()))
                for value, flat_index in zip(values.tolist(), flat.tolist()):
                    lt, rt = divmod(flat_index, len(names))
                    block_top.append({
                        "name": r515.pair_name(subset, relation, site, lt, rt),
                        "quality_margin": value,
                        "passes": bool(candidate[lt, rt]),
                    })

                c0l = matrices["half0"]["circuit"][subset, left_action, indices]
                c0r = matrices["half0"]["circuit"][subset, right_action, indices]
                c1l = matrices["half1"]["circuit"][subset, left_action, indices]
                c1r = matrices["half1"]["circuit"][subset, right_action, indices]
                for lt in range(len(names)):
                    for rt in range(len(names)):
                        name = r515.pair_name(subset, relation, site, lt, rt)
                        rows["name"].append(name)
                        rows["beta"].append(beta[lt, rt])
                        rows["left0"].append(c0l[lt])
                        rows["right0"].append(c0r[rt])
                        rows["left1"].append(c1l[lt])
                        rows["right1"].append(c1r[rt])
                        rows["task_compatible"].append(task_ok[lt, rt])
                        rows["full0_split"].append(~circuit0_ok[lt, rt])
                        rows["full1_split"].append(~circuit1_ok[lt, rt])
                        rows["quality"].append(quality[lt, rt])
                        if bool(candidate[lt, rt]):
                            candidate_names.append(name)

    for key in ("beta", "task_compatible", "full0_split", "full1_split", "quality"):
        rows[key] = torch.stack(rows[key])
    for key in ("left0", "right0", "left1", "right1"):
        rows[key] = torch.stack(rows[key])
    block_top.sort(key=lambda row: row["quality_margin"], reverse=True)
    rows["candidate_names"] = candidate_names
    rows["top_screens"] = block_top[:20]
    rows["material_nodes"] = len(material_nodes)
    return rows


def split_mask(table: dict, selected: list[int], window: int) -> torch.Tensor:
    if not selected:
        return torch.zeros_like(table["task_compatible"], dtype=torch.bool)
    left = table[f"left{window}"][:, selected]
    right = table[f"right{window}"][:, selected]
    beta = table["beta"]
    full_left = table[f"left{window}"].square().sum(-1).clamp_min(1e-30)
    full_right = table[f"right{window}"].square().sum(-1).clamp_min(1e-30)
    left_energy = left.square().sum(-1)
    right_energy = right.square().sum(-1)
    signal = ((left_energy / full_left >= ENERGY_FRACTION)
              & (right_energy / full_right >= ENERGY_FRACTION))
    dot = (left * right).sum(-1)
    cosine = beta.sign() * dot / (left_energy * right_energy).sqrt().clamp_min(1e-30)
    forward = ((left - beta[:, None] * right).square().sum(-1)
               / left_energy.clamp_min(1e-30)).sqrt()
    inverse = torch.where(beta.abs() > 1e-30, beta.reciprocal(),
                          torch.full_like(beta, math.inf))
    backward = ((right - inverse[:, None] * left).square().sum(-1)
                / right_energy.clamp_min(1e-30)).sqrt()
    cos_bar, residual_bar = ((.90, .35) if window == 0 else (.80, .50))
    return signal & ((cosine < cos_bar) | (forward > residual_bar) | (backward > residual_bar))


def greedy_order(table: dict, target: torch.Tensor, window: int) -> tuple[list[int], list[int]]:
    selected, counts = [], []
    for _ in range(32):
        best_coordinate, best_count = None, -1
        for coordinate in range(32):
            if coordinate in selected:
                continue
            count = int((split_mask(table, selected + [coordinate], window) & target).sum())
            if count > best_count:
                best_coordinate, best_count = coordinate, count
        selected.append(best_coordinate)
        counts.append(best_count)
    return selected, counts


def coverage_curve(table: dict, order: list[int], target: torch.Tensor,
                   window: int) -> dict[str, float]:
    denominator = int(target.sum())
    return {
        str(size): (float((split_mask(table, order[:size], window) & target).sum()) / denominator
                    if denominator else 0.0)
        for size in COVER_SIZES
    }


def witness_analysis(table: dict, tags: list[str], *, control_seeds=CONTROL_SEEDS) -> dict:
    task = table["task_compatible"].bool()
    target0 = task & table["full0_split"].bool()
    target1 = task & table["full1_split"].bool()
    order0, marginal0 = greedy_order(table, target0, 0)
    order1, marginal1 = greedy_order(table, target1, 1)
    curve0 = coverage_curve(table, order0, target0, 0)
    # Prospective transfer denominator stays the half0-defined target population.
    transfer1 = coverage_curve(table, order0, target0, 1)
    independent1 = coverage_curve(table, order1, target1, 1)
    controls = []
    for seed in control_seeds:
        order = torch.randperm(32, generator=torch.Generator().manual_seed(seed)).tolist()
        controls.append({
            "seed": seed,
            "order": order,
            "half0": coverage_curve(table, order, target0, 0),
            "half1_transfer": coverage_curve(table, order, target0, 1),
        })
    top0, top1 = set(order0[:8]), set(order1[:8])
    jaccard = len(top0 & top1) / len(top0 | top1) if top0 | top1 else 0.0

    # Which single coordinates contribute most to fixed-beta half0 disagreement?
    left, right, beta = table["left0"], table["right0"], table["beta"]
    forward = (left - beta[:, None] * right).square() / left.square().sum(-1)[:, None].clamp_min(1e-30)
    inverse = torch.where(beta.abs() > 1e-30, beta.reciprocal(),
                          torch.full_like(beta, math.inf))
    backward = (right - inverse[:, None] * left).square() / right.square().sum(-1)[:, None].clamp_min(1e-30)
    contribution = torch.maximum(forward, backward)
    dominant = contribution.argmax(-1)
    dominant_counts = []
    for coordinate, tag in enumerate(tags):
        dominant_counts.append({
            "coordinate": coordinate, "circuit_tag": tag,
            "half0_target_pairs_dominated": int(((dominant == coordinate) & target0).sum()),
        })
    dominant_counts.sort(key=lambda row: (-row["half0_target_pairs_dominated"], row["coordinate"]))

    return {
        "task_compatible_pairs": int(task.sum()),
        "full32_half0_circuit_rejected_pairs": int(target0.sum()),
        "full32_half1_circuit_rejected_pairs": int(target1.sum()),
        "half0_rejected_fraction_of_task_compatible": (
            float(target0.sum()) / int(task.sum()) if int(task.sum()) else 0.0),
        "half0_selected_order": [tags[index] for index in order0],
        "half0_selected_indices": order0,
        "half0_greedy_cumulative_counts": marginal0,
        "half0_coverage": curve0,
        "half1_transfer_coverage_of_half0_target": transfer1,
        "half1_selected_order": [tags[index] for index in order1],
        "half1_selected_indices": order1,
        "half1_greedy_cumulative_counts": marginal1,
        "half1_independent_coverage": independent1,
        "top8_jaccard": jaccard,
        "controls": controls,
        "maximum_control_half1_transfer_at_8": max(
            (row["half1_transfer"]["8"] for row in controls), default=0.0),
        "dominant_half0_residual_coordinates": dominant_counts,
    }


def planted_case(seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    planted = torch.randperm(32, generator=generator)[:8].sort().values.tolist()
    pairs = 512
    assigned = torch.tensor([planted[index % 8] for index in range(pairs)])
    beta = torch.full((pairs,), 1.5, dtype=torch.float64)
    right0 = .002 * torch.randn(pairs, 32, generator=generator, dtype=torch.float64)
    right1 = .002 * torch.randn(pairs, 32, generator=generator, dtype=torch.float64)
    for row, coordinate in enumerate(assigned.tolist()):
        right0[row, coordinate] = 1.0
        right1[row, coordinate] = .95
    left0 = beta[:, None] * right0
    left1 = beta[:, None] * right1
    left0[torch.arange(pairs), assigned] += 2.0
    left1[torch.arange(pairs), assigned] += 1.9
    table = {
        "beta": beta, "left0": left0, "right0": right0,
        "left1": left1, "right1": right1,
        "task_compatible": torch.ones(pairs, dtype=torch.bool),
        "full0_split": torch.ones(pairs, dtype=torch.bool),
        "full1_split": torch.ones(pairs, dtype=torch.bool),
    }
    analysis = witness_analysis(table, [f"c{index}" for index in range(32)])
    recovered = set(analysis["half0_selected_indices"][:8])
    control_max = analysis["maximum_control_half1_transfer_at_8"]
    control_gap = float(analysis["half1_transfer_coverage_of_half0_target"]["8"]) - control_max
    holds = bool(
        recovered == set(planted)
        and float(analysis["half0_coverage"]["8"]) >= .75
        and float(analysis["half1_transfer_coverage_of_half0_target"]["8"]) >= .65
        and control_gap >= .10)
    return {
        "seed": seed, "planted": planted,
        "recovered": analysis["half0_selected_indices"][:8],
        "half0_coverage_at_8": analysis["half0_coverage"]["8"],
        "half1_coverage_at_8": analysis["half1_transfer_coverage_of_half0_target"]["8"],
        "maximum_control_half1_at_8": control_max,
        "holds": holds,
    }


def planted_suite() -> dict:
    cases = [planted_case(seed) for seed in PLANTED_SEEDS]
    return {"cases": cases, "all_exact_set_recoveries_and_bars_hold": all(c["holds"] for c in cases)}


def validate_route() -> tuple[dict, dict]:
    if "PENDING" in R515_RESULT_SHA256 or "PENDING" in R515_BUNDLE_SHA256:
        raise RuntimeError("rung515 hashes have not been installed by the no-outcome preflight")
    hashes = {PREREG: PREREG_SHA256, R515_RESULT: R515_RESULT_SHA256,
              R515_BUNDLE: R515_BUNDLE_SHA256}
    for path, expected in hashes.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(R515_RESULT.read_text())
    if not (
        result.get("pred_a_exact_live_identifiable_finite_downstream_instrument") is True
        and result.get("pred_b_small_downstream_relation_beats_controls") is False
        and result.get("analysis", {}).get("discovery_summary", {}).get("candidate_count") == 0
        and result.get("next_step")
        == "leave_mlp10_consumer_descent_for_task_defined_state_transition_or_new_gap"
    ):
        raise RuntimeError("rung515 conditional zero-pair route changed")
    bundle = torch.load(R515_BUNDLE, map_location="cpu", weights_only=False)
    if set(bundle.get("collections", {})) != {"discovery"}:
        raise RuntimeError("rung515 B-false bundle unexpectedly opened later data")
    return result, bundle


def main() -> None:
    started = time.time()
    planted = planted_suite()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        if sha256(PREREG) != PREREG_SHA256:
            raise RuntimeError("rung516 preregistration hash changed")
        print(json.dumps({
            "status": "dry_run_passed", "rung": 516,
            "model_loaded": False, "outcomes_opened": False,
            "planted": planted,
            "rung515_hashes_pending": "PENDING" in R515_RESULT_SHA256,
        }, indent=2, sort_keys=True))
        return

    r515_result, bundle = validate_route()
    collection = bundle["collections"]["discovery"]
    matrices = r515.response_matrices(collection)
    table = build_pair_table(matrices)
    replay_top = r515_result["analysis"]["discovery_summary"]["top_screens"]
    margin_error = max(
        (abs(left["quality_margin"] - right["quality_margin"])
         for left, right in zip(table["top_screens"], replay_top)), default=0.0)
    names_exact = [row["name"] for row in table["top_screens"]] \
        == [row["name"] for row in replay_top]
    pred_a = bool(
        planted["all_exact_set_recoveries_and_bars_hold"]
        and len(table["name"]) == r515.PAIR_COUNT == 17460
        and table["material_nodes"]
        == r515_result["analysis"]["discovery_summary"]["material_nodes"]
        and len(table["candidate_names"])
        == r515_result["analysis"]["discovery_summary"]["candidate_count"] == 0
        and names_exact and margin_error <= 1e-10)

    tags = list(collection["circuit_tags"])
    analysis = witness_analysis(table, tags)
    task_count = analysis["task_compatible_pairs"]
    rejected_fraction = analysis["half0_rejected_fraction_of_task_compatible"]
    pred_b = bool(pred_a and task_count >= 64 and rejected_fraction >= .50)
    half0_at8 = analysis["half0_coverage"]["8"]
    half1_at8 = analysis["half1_transfer_coverage_of_half0_target"]["8"]
    control_max = analysis["maximum_control_half1_transfer_at_8"]
    control_gap = half1_at8 - control_max
    pred_c = bool(
        pred_b and half0_at8 >= .75 and half1_at8 >= .65
        and control_gap >= .10)
    pred_d = bool(pred_c and analysis["top8_jaccard"] >= .50)
    strong_null = not (pred_a and pred_b and pred_c and pred_d)
    if not pred_a:
        next_step = "repair_cpu_replay_or_planted_instrument_only"
    elif not pred_b:
        next_step = "leave_consumer_descent_task_or_materiality_already_explains_split"
    elif not pred_c or not pred_d:
        next_step = "leave_exact_term_vocabulary_circuit_separation_is_diffuse_or_unstable"
    else:
        next_step = "use_named_witnesses_in_task_defined_downstream_state_transition_preregistration"

    result = {
        "status": "complete", "rung": 516,
        "claim_level": "document_stable_discovery_circuit_separation_cover_only",
        "source_hashes": {
            str(PREREG): sha256(PREREG), str(R515_RESULT): sha256(R515_RESULT),
            str(R515_BUNDLE): sha256(R515_BUNDLE),
        },
        "source_code_sha256": sha256(Path(__file__)),
        "route": {
            "rung515_pred_a": r515_result[
                "pred_a_exact_live_identifiable_finite_downstream_instrument"],
            "rung515_candidate_count": r515_result["analysis"]["discovery_summary"][
                "candidate_count"],
            "confirmation_circuit_families_opened": False,
        },
        "planted_recovery": planted,
        "replay": {
            "pairs": len(table["name"]), "material_nodes": table["material_nodes"],
            "candidate_count": len(table["candidate_names"]),
            "top20_names_exact": names_exact,
            "top20_max_quality_margin_abs_error": margin_error,
        },
        "analysis": analysis,
        "pred_a_exact_zero_pair_route_and_screen_replay": pred_a,
        "pred_b_circuits_add_real_split": pred_b,
        "pred_c_compact_document_stable_witness_set": pred_c,
        "pred_d_witness_identities_stable": pred_d,
        "strong_null": strong_null,
        "execution_price": {
            "full_model_forwards": 0, "backwards": 0,
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
            "pair_rows": len(table["name"]), "greedy_coordinate_evaluations": 1056,
            "control_prefixes": len(CONTROL_SEEDS) * len(COVER_SIZES),
        },
        "claim_limits": [
            "same_32_circuit_identities_across_document_halves_only",
            "thirty_confirmation_circuit_families_unopened",
            "no_state_count_or_minimal_quotient_theorem",
            "no_executable_component_or_substitution_claim",
        ],
        "runtime_s": time.time() - started,
        "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 516,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c, "pred_d": pred_d,
        "strong_null": strong_null,
        "task_compatible_pairs": task_count,
        "half0_rejected_fraction": rejected_fraction,
        "half0_top8_coverage": half0_at8,
        "half1_top8_transfer": half1_at8,
        "control_max": control_max,
        "control_gap": control_gap,
        "top8_jaccard": analysis["top8_jaccard"],
        "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
