#!/usr/bin/env python3
"""RUNG483 -- operational MLP0 branch quotient at attention1 and MLP1.

Measure exact forward-mode responses and complete physical removals of the
rung401 T/C/I/S branches at three immediate consumer outputs.  The result asks
whether T and I are one shared, two distinct, or consumer-specific variables.
No rank, sparse support, or CE objective is used.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp0_branch_circuit_response_rung481 as parent


PREREG = POLY / "MLP0_IMMEDIATE_CONSUMER_QUOTIENT_RUNG483_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/mlp0_branch_circuit_response_rung481.py"
PARENT_RESULT = ROOT / "mlp0_branch_circuit_response_rung481_results.json"
OUT = ROOT / "mlp0_immediate_consumer_quotient_rung483_results.json"
HASHES = {
    PREREG: "210136132fc3ab256812acc05545891181eb4a7e1c361b2f664e274deb82f143",
    PARENT_SOURCE: "ef08017a30ceb0c9e4481198fc1d58c5b0bf8cd37707d2223c42db9eb04f1f44",
    PARENT_RESULT: "2af2e9d934d85223cb01cb731ad2bcbe54b8b90cbf21f6ba6753cc1347e84573",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
}
BRANCHES = parent.BRANCHES
PAIRS = tuple(itertools.combinations(range(len(BRANCHES)), 2))
PAIR_NAMES = tuple(f"{BRANCHES[a]}x{BRANCHES[b]}" for a, b in PAIRS)
CONSUMERS = ("attention1", "mlp1_direct", "mlp1_total")
DISCOVERY_RANGE = (0, 500)
DISCOVERY_SPLIT = 250
VALIDATION_RANGE = (500, 1000)
VALIDATION_SPLIT = 750
BATCH = 4
D = 1152
TOKENS = 256
EPSILON = 0.125
POSITION_SHIFTS = tuple(((31 * index) % 255) + 1 for index in range(16))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cosine_from_sums(dot: float, left2: float, right2: float) -> float:
    return float(dot / max(math.sqrt(max(left2, 0.0) * max(right2, 0.0)), 1e-30))


def _scaled_error_from_sums(dot: float, predictor2: float, target2: float) -> float:
    alpha = dot / max(predictor2, 1e-30)
    residual2 = target2 - 2.0 * alpha * dot + alpha * alpha * predictor2
    return math.sqrt(max(residual2, 0.0) / max(target2, 1e-30))


def _held_scale_error(fit_gram: torch.Tensor, held_gram: torch.Tensor,
                      predictor: int, target: int) -> tuple[float, float]:
    alpha = float(fit_gram[predictor, target]
                  / fit_gram[predictor, predictor].clamp_min(1e-30))
    target2 = float(held_gram[target, target])
    residual2 = float(
        held_gram[target, target]
        - 2.0 * alpha * held_gram[predictor, target]
        + alpha * alpha * held_gram[predictor, predictor])
    return alpha, math.sqrt(max(residual2, 0.0) / max(target2, 1e-30))


def _gram_cosines(gram: torch.Tensor) -> torch.Tensor:
    diagonal = torch.diagonal(gram, dim1=-2, dim2=-1).clamp_min(1e-30)
    denominator = torch.sqrt(diagonal[..., :, None] * diagonal[..., None, :])
    return gram / denominator


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    if receipt.get("rung") != 481 \
            or receipt.get("pred_a_exact_lawful_instrument") is not True \
            or receipt.get("strong_null") is not True \
            or receipt.get("next_step") != "consumer_specific_attention1_mlp1_jacobians" \
            or receipt.get("validation_licensed_and_opened") is not False:
        raise RuntimeError("rung481 did not license the immediate-consumer route")
    rows, _masks, _discovery_tags, _validation_tags, fit_rows, metadata = \
        parent.validate_inputs()
    if rows.ndim != 2 or len(rows) != 1000 or rows.shape[1] != TOKENS + 1:
        raise RuntimeError("rung483 row authority changed")
    return rows, fit_rows, metadata


@torch.no_grad()
def _full_native_capture(model, tokens):
    captures = {}
    calls = {"attention": 0, "mlp": 0}

    def attention(event):
        calls["attention"] += 1
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 1:
            captures["attention1"] = write.detach().clone()
        return write, first_value

    def mlp(event):
        calls["mlp"] += 1
        write = event.block.mlp(event.state)
        if event.site == 1:
            captures["mlp1_total"] = write.detach().clone()
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    if set(captures) != {"attention1", "mlp1_total"}:
        raise RuntimeError("native consumer capture failed")
    del logits
    return captures, calls


@torch.no_grad()
def _native_prefix(model, tokens, reference):
    x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
    block0, block1 = model.transformer.h[0], model.transformer.h[1]
    before_a0 = block0.lambdas[0] * x0 + block0.lambdas[1] * x0
    a0, first_value = block0.attn(F.rms_norm(before_a0, (D,)), None)
    before_m0 = before_a0 + a0
    z0 = F.rms_norm(before_m0, (D,))
    m0 = block0.mlp(z0)
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    down = block0.mlp.Down.weight.detach().float()
    retained, branches, collinearity = parent._exact_components(
        before_a0, a0, z0, reference, left, right, down)
    analytical = retained + sum(branches.values(), start=torch.zeros_like(retained))
    direct = parent.branch_parent._T(z0, z0, left, right, down)
    deployed = m0.float() - block0.mlp.Down_bias.detach().float()
    branches = {name: branches[name].to(m0.dtype) for name in BRANCHES}
    after_m0 = before_m0 + m0
    before_a1 = block1.lambdas[0] * after_m0 + block1.lambdas[1] * x0
    a1, _ = block1.attn(F.rms_norm(before_a1, (D,)), first_value)
    m1 = block1.mlp(F.rms_norm(before_a1 + a1, (D,)))
    return {
        "x0": x0.detach(), "before_m0": before_m0.detach(),
        "m0": m0.detach(), "first_value": first_value.detach(),
        "a1": a1.detach(), "m1": m1.detach(), "branches": branches,
        "analytical_num": float((analytical.double() - direct.double()).square().sum()),
        "analytical_den": float(direct.double().square().sum()),
        "deployed_num": float((analytical.double() - deployed.double()).square().sum()),
        "deployed_den": float(deployed.double().square().sum()),
        "collinearity_max": float(collinearity.max()),
    }


def _consumer_outputs(model, cache, removed):
    """Outputs after subtracting ``removed`` from native MLP0."""
    block1 = model.transformer.h[1]
    after_m0 = cache["before_m0"] + cache["m0"] - removed
    before_a1 = block1.lambdas[0] * after_m0 + block1.lambdas[1] * cache["x0"]
    state1 = F.rms_norm(before_a1, (D,))
    attention1, _ = block1.attn(state1, cache["first_value"])
    direct_mlp1 = block1.mlp(F.rms_norm(before_a1 + cache["a1"], (D,)))
    total_mlp1 = block1.mlp(F.rms_norm(before_a1 + attention1, (D,)))
    return attention1, direct_mlp1, total_mlp1


def _directional_response(model, cache, branch):
    alpha = torch.zeros((), dtype=branch.dtype, device=branch.device)
    tangent_alpha = torch.ones_like(alpha)

    def function(value):
        return _consumer_outputs(model, cache, value * branch)

    primal, tangent = torch.autograd.functional.jvp(
        function, alpha, tangent_alpha, create_graph=False, strict=True)
    return tuple(value.detach() for value in primal), tuple(value.detach() for value in tangent)


def _empty_stats():
    return {
        "tangent_gram": torch.zeros(2, 3, 4, 4, dtype=torch.float64),
        "exact_gram": torch.zeros(2, 3, 4, 4, dtype=torch.float64),
        "raw_branch_gram": torch.zeros(2, 4, 4, dtype=torch.float64),
        "finite_checks": torch.zeros(2, 3, 4, 3, dtype=torch.float64),
        "physical_checks": torch.zeros(2, 3, 4, 3, dtype=torch.float64),
        "shuffle_tangent_dot": torch.zeros(2, 3, 16, dtype=torch.float64),
        "shuffle_exact_dot": torch.zeros(2, 3, 16, dtype=torch.float64),
        "pair_interaction_square": torch.zeros(2, 3, 6, dtype=torch.float64),
        "pair_interaction_mean_sum": torch.zeros(2, 3, 6, D, dtype=torch.float64),
        "singleton_mean_sum": torch.zeros(2, 3, 4, D, dtype=torch.float64),
        "position_counts": torch.zeros(2, dtype=torch.float64),
    }


def _accumulate_gram(target, values):
    matrix = torch.stack(values).double().reshape(len(values), -1)
    target += (matrix @ matrix.T).cpu()


def _accumulate_phase(stats, half, branches, tangents, finite, exact, interactions):
    _accumulate_gram(stats["raw_branch_gram"][half], branches)
    for consumer in range(len(CONSUMERS)):
        tangent_values = [tangents[b][consumer] for b in range(len(BRANCHES))]
        exact_values = [exact[b][consumer] for b in range(len(BRANCHES))]
        _accumulate_gram(stats["tangent_gram"][half, consumer], tangent_values)
        _accumulate_gram(stats["exact_gram"][half, consumer], exact_values)
        for branch in range(len(BRANCHES)):
            t = tangent_values[branch].double()
            f = finite[branch][consumer].double()
            q = exact_values[branch].double()
            stats["finite_checks"][half, consumer, branch] += torch.tensor(
                [float((t * f).sum()), float(t.square().sum()), float(f.square().sum())],
                dtype=torch.float64)
            stats["physical_checks"][half, consumer, branch] += torch.tensor(
                [float((t * q).sum()), float(t.square().sum()), float(q.square().sum())],
                dtype=torch.float64)
            stats["singleton_mean_sum"][half, consumer, branch] += \
                q.sum(dim=(0, 1)).cpu()
        t = tangent_values[BRANCHES.index("T")].double()
        i = tangent_values[BRANCHES.index("I")].double()
        tq = exact_values[BRANCHES.index("T")].double()
        iq = exact_values[BRANCHES.index("I")].double()
        for control, shift in enumerate(POSITION_SHIFTS):
            stats["shuffle_tangent_dot"][half, consumer, control] += float(
                (t * torch.roll(i, shifts=shift, dims=1)).sum())
            stats["shuffle_exact_dot"][half, consumer, control] += float(
                (tq * torch.roll(iq, shifts=shift, dims=1)).sum())
        for pair in range(len(PAIRS)):
            value = interactions[pair][consumer].double()
            stats["pair_interaction_square"][half, consumer, pair] += float(
                value.square().sum())
            stats["pair_interaction_mean_sum"][half, consumer, pair] += \
                value.sum(dim=(0, 1)).cpu()
    stats["position_counts"][half] += branches[0].shape[0] * branches[0].shape[1]


def _subset_nested(values, condition):
    return [tuple(component[condition] for component in branch) for branch in values]


def collect_phase(model, rows, reference, start_doc, stop_doc, split):
    stats = _empty_stats()
    audit = {
        "full_native_forwards": 0, "full_native_attention_calls": 0,
        "full_native_mlp_calls": 0, "native_prefixes": 0,
        "directional_derivatives": 0, "finite_difference_prefixes": 0,
        "singleton_removal_prefixes": 0, "pair_removal_prefixes": 0,
    }
    errors = {
        "prefix_attention1_num": 0.0, "prefix_attention1_den": 0.0,
        "prefix_mlp1_num": 0.0, "prefix_mlp1_den": 0.0,
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
        "collinearity_max": 0.0, "branch_max_abs": {name: 0.0 for name in BRANCHES},
    }
    device = next(model.parameters()).device
    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        tokens = rows[start:stop, :-1].to(device)
        native_capture, native_calls = _full_native_capture(model, tokens)
        audit["full_native_forwards"] += 1
        audit["full_native_attention_calls"] += native_calls["attention"]
        audit["full_native_mlp_calls"] += native_calls["mlp"]
        cache = _native_prefix(model, tokens, reference)
        audit["native_prefixes"] += 1
        for key, observed in (("attention1", cache["a1"]), ("mlp1_total", cache["m1"])):
            difference = observed.float() - native_capture[key].float()
            errors[f"prefix_{'attention1' if key == 'attention1' else 'mlp1'}_num"] += \
                float(difference.double().square().sum())
            errors[f"prefix_{'attention1' if key == 'attention1' else 'mlp1'}_den"] += \
                float(native_capture[key].double().square().sum())
        errors["analytical_num"] += cache["analytical_num"]
        errors["analytical_den"] += cache["analytical_den"]
        errors["deployed_num"] += cache["deployed_num"]
        errors["deployed_den"] += cache["deployed_den"]
        errors["collinearity_max"] = max(errors["collinearity_max"], cache["collinearity_max"])
        branch_values = [cache["branches"][name] for name in BRANCHES]
        for name, value in zip(BRANCHES, branch_values):
            errors["branch_max_abs"][name] = max(
                errors["branch_max_abs"][name], float(value.abs().max()))

        base_outputs = (cache["a1"], cache["m1"], cache["m1"])
        tangents, finite, exact = [], [], []
        for branch in branch_values:
            primal, tangent = _directional_response(model, cache, branch)
            audit["directional_derivatives"] += 1
            if any(not bool(torch.isfinite(value).all()) for value in (*primal, *tangent)):
                raise RuntimeError("directional derivative produced a nonfinite value")
            plus = _consumer_outputs(model, cache, EPSILON * branch)
            minus = _consumer_outputs(model, cache, -EPSILON * branch)
            audit["finite_difference_prefixes"] += 2
            finite.append(tuple(
                ((a.float() - b.float()) / (2.0 * EPSILON)).detach()
                for a, b in zip(plus, minus)))
            tangents.append(tangent)
            removed = _consumer_outputs(model, cache, branch)
            audit["singleton_removal_prefixes"] += 1
            exact.append(tuple(
                (a.float() - b.float()).detach() for a, b in zip(removed, base_outputs)))

        interactions = []
        for left, right in PAIRS:
            joint = _consumer_outputs(model, cache, branch_values[left] + branch_values[right])
            audit["pair_removal_prefixes"] += 1
            interactions.append(tuple(
                (joint[c].float() - base_outputs[c].float()
                 - exact[left][c] - exact[right][c]).detach()
                for c in range(len(CONSUMERS))))

        global_rows = torch.arange(start, stop, device=device)
        for half, condition in enumerate((global_rows < split, global_rows >= split)):
            if not condition.any():
                continue
            phase_branches = [value[condition] for value in branch_values]
            phase_tangents = _subset_nested(tangents, condition)
            phase_finite = _subset_nested(finite, condition)
            phase_exact = _subset_nested(exact, condition)
            phase_interactions = _subset_nested(interactions, condition)
            _accumulate_phase(
                stats, half, phase_branches, phase_tangents, phase_finite,
                phase_exact, phase_interactions)
        del native_capture, cache, tangents, finite, exact, interactions

    batches = math.ceil((stop_doc - start_doc) / BATCH)
    expected = {
        "full_native_forwards": batches,
        "full_native_attention_calls": 18 * batches,
        "full_native_mlp_calls": 18 * batches,
        "native_prefixes": batches,
        "directional_derivatives": 4 * batches,
        "finite_difference_prefixes": 8 * batches,
        "singleton_removal_prefixes": 4 * batches,
        "pair_removal_prefixes": 6 * batches,
    }
    instrument = {
        "prefix_attention1_relative_squared": errors["prefix_attention1_num"]
        / max(errors["prefix_attention1_den"], 1e-30),
        "prefix_mlp1_relative_squared": errors["prefix_mlp1_num"]
        / max(errors["prefix_mlp1_den"], 1e-30),
        "analytical_branch_identity_relative_squared": errors["analytical_num"]
        / max(errors["analytical_den"], 1e-30),
        "deployed_branch_identity_relative_squared": errors["deployed_num"]
        / max(errors["deployed_den"], 1e-30),
        "normalization_noncollinearity_max_relative_squared": errors["collinearity_max"],
        "branch_deployed_max_abs": errors["branch_max_abs"],
        "calls": audit, "expected_calls": expected, "calls_exact": audit == expected,
    }
    return stats, instrument


def _check_reports(checks):
    reports = {}
    holds = True
    for half in range(2):
        for consumer, consumer_name in enumerate(CONSUMERS):
            for branch, branch_name in enumerate(BRANCHES):
                dot, predictor2, target2 = map(float, checks[half, consumer, branch])
                cosine = _cosine_from_sums(dot, predictor2, target2)
                error = _scaled_error_from_sums(dot, predictor2, target2)
                reports[f"half{half}:{consumer_name}:{branch_name}"] = {
                    "cosine": cosine, "best_scalar_adjusted_relative_error": error,
                }
                holds &= cosine >= .98 and error <= .20
    return reports, bool(holds)


def _physical_reports(checks):
    reports = {}
    holds = True
    for half in range(2):
        for consumer, consumer_name in enumerate(CONSUMERS):
            for branch_name in ("T", "I"):
                branch = BRANCHES.index(branch_name)
                dot, predictor2, target2 = map(float, checks[half, consumer, branch])
                cosine = _cosine_from_sums(dot, predictor2, target2)
                error = _scaled_error_from_sums(dot, predictor2, target2)
                reports[f"half{half}:{consumer_name}:{branch_name}"] = {
                    "cosine": cosine, "best_scalar_adjusted_relative_error": error,
                }
                holds &= cosine >= .75 and error <= .60
    return reports, bool(holds)


def analyze_phase(stats):
    tangent_gram = stats["tangent_gram"]
    exact_gram = stats["exact_gram"]
    tangent_cos = _gram_cosines(tangent_gram)
    exact_cos = _gram_cosines(exact_gram)
    raw_cos = _gram_cosines(stats["raw_branch_gram"])
    t_index, i_index = BRANCHES.index("T"), BRANCHES.index("I")
    shuffle_tangent, shuffle_exact = {}, {}
    for half in range(2):
        for consumer, name in enumerate(CONSUMERS):
            tangent_den = math.sqrt(float(
                tangent_gram[half, consumer, t_index, t_index]
                * tangent_gram[half, consumer, i_index, i_index]))
            exact_den = math.sqrt(float(
                exact_gram[half, consumer, t_index, t_index]
                * exact_gram[half, consumer, i_index, i_index]))
            shuffle_tangent[f"half{half}:{name}"] = [
                float(value / max(tangent_den, 1e-30))
                for value in stats["shuffle_tangent_dot"][half, consumer]]
            shuffle_exact[f"half{half}:{name}"] = [
                float(value / max(exact_den, 1e-30))
                for value in stats["shuffle_exact_dot"][half, consumer]]

    scales = {"tangent": {}, "exact": {}}
    for consumer, name in enumerate(CONSUMERS):
        for kind, gram in (("tangent", tangent_gram), ("exact", exact_gram)):
            alpha, error = _held_scale_error(
                gram[0, consumer], gram[1, consumer], t_index, i_index)
            scales[kind][name] = {"alpha_fit_half0": alpha, "relative_error_half1": error}

    shared = True
    split = True
    shared_consumers, split_consumers = [], []
    for consumer, name in enumerate(CONSUMERS):
        consumer_shared = True
        consumer_split = True
        for half in range(2):
            tc = float(tangent_cos[half, consumer, t_index, i_index])
            qc = float(exact_cos[half, consumer, t_index, i_index])
            tangent_q95 = float(torch.quantile(
                torch.tensor(shuffle_tangent[f"half{half}:{name}"], dtype=torch.float64),
                .95, interpolation="higher"))
            exact_q95 = float(torch.quantile(
                torch.tensor(shuffle_exact[f"half{half}:{name}"], dtype=torch.float64),
                .95, interpolation="higher"))
            shared &= tc >= .90 and qc >= .80 \
                and tc >= tangent_q95 + .15 and qc >= exact_q95 + .15
            split &= abs(tc) <= .65 and abs(qc) <= .65
            consumer_shared &= tc >= .85 and qc >= .75
            consumer_split &= abs(tc) <= .55 and abs(qc) <= .55
        if consumer_shared:
            shared_consumers.append(name)
        if consumer_split:
            split_consumers.append(name)
        shared &= scales["tangent"][name]["relative_error_half1"] <= .35 \
            and scales["exact"][name]["relative_error_half1"] <= .35
    consumer_specific = bool(
        shared_consumers and split_consumers
        and any(left != right for left in shared_consumers for right in split_consumers))

    finite_reports, finite_holds = _check_reports(stats["finite_checks"])
    physical_reports, physical_holds = _physical_reports(stats["physical_checks"])
    pair_reports = {}
    stable_pairs = []
    for pair, name in enumerate(PAIR_NAMES):
        left, right = PAIRS[pair]
        consumer_reports = {}
        pair_stable_any_consumer = False
        for consumer, consumer_name in enumerate(CONSUMERS):
            ratios = []
            signs = []
            for half in range(2):
                interaction2 = float(stats["pair_interaction_square"][half, consumer, pair])
                singleton2 = min(
                    float(exact_gram[half, consumer, left, left]),
                    float(exact_gram[half, consumer, right, right]))
                ratios.append(math.sqrt(interaction2 / max(singleton2, 1e-30)))
                mean = stats["pair_interaction_mean_sum"][half, consumer, pair]
                signs.append([
                    math.copysign(1.0, float(torch.dot(
                        mean, stats["singleton_mean_sum"][half, consumer, branch])) or 1.0)
                    for branch in (left, right)
                ])
            mean_cosine = float(F.cosine_similarity(
                stats["pair_interaction_mean_sum"][0, consumer, pair],
                stats["pair_interaction_mean_sum"][1, consumer, pair], dim=0, eps=1e-30))
            ratio_stability = max(ratios) / max(min(ratios), 1e-30)
            stable = bool(min(ratios) >= .20 and ratio_stability <= 2.0
                          and mean_cosine >= .50)
            pair_stable_any_consumer |= stable
            consumer_reports[consumer_name] = {
                "relative_interaction_norm_by_half": ratios,
                "half_ratio_factor": ratio_stability,
                "mean_output_cosine_between_halves": mean_cosine,
                "interaction_dot_singleton_signs_by_half": signs,
                "stable_and_material": stable,
            }
        if pair_stable_any_consumer:
            stable_pairs.append(name)
        pair_reports[name] = consumer_reports

    relations = {
        "shared": bool(shared), "split": bool(split),
        "consumer_specific": consumer_specific,
    }
    return {
        "tangent_gram": tangent_gram.tolist(), "exact_removal_gram": exact_gram.tolist(),
        "raw_branch_gram": stats["raw_branch_gram"].tolist(),
        "tangent_cosines": tangent_cos.tolist(), "exact_removal_cosines": exact_cos.tolist(),
        "raw_branch_cosines": raw_cos.tolist(),
        "position_shuffle_tangent_cosines": shuffle_tangent,
        "position_shuffle_exact_cosines": shuffle_exact,
        "half0_to_half1_T_to_I_scales": scales,
        "finite_difference_reports": finite_reports,
        "tangent_to_physical_reports": physical_reports,
        "shared_consumers": shared_consumers, "split_consumers": split_consumers,
        "relations": relations, "exactly_one_relation": sum(relations.values()) == 1,
        "pair_reports": pair_reports, "stable_material_pairs": stable_pairs,
        "pred_b_tangent_predicts_physical_T_I": physical_holds,
        "pred_c_shared": bool(shared), "pred_c_split": bool(split),
        "pred_c_consumer_specific": consumer_specific,
        "pred_d_stable_material_I_pair": any("I" in name.split("x") for name in stable_pairs),
        "finite_difference_holds": finite_holds,
    }


def _instrument_valid(instrument, analysis, *, discovery):
    del discovery
    positions_ok = instrument["position_counts"] == [250 * TOKENS, 250 * TOKENS]
    return bool(
        instrument["prefix_attention1_relative_squared"] <= 1e-12
        and instrument["prefix_mlp1_relative_squared"] <= 1e-12
        and instrument["analytical_branch_identity_relative_squared"] <= 1e-8
        and instrument["deployed_branch_identity_relative_squared"] <= 1e-5
        and instrument["calls_exact"] and positions_ok
        and all(value > 0 for value in instrument["branch_deployed_max_abs"].values())
        and analysis["finite_difference_holds"])


def _pair_validation(discovery, validation):
    reports = []
    for pair in discovery["stable_material_pairs"]:
        left, right = PAIRS[PAIR_NAMES.index(pair)]
        for consumer in CONSUMERS:
            before = discovery["pair_reports"][pair][consumer]
            if not before["stable_and_material"]:
                continue
            after = validation["pair_reports"][pair][consumer]
            signs_match = after["interaction_dot_singleton_signs_by_half"] \
                == before["interaction_dot_singleton_signs_by_half"]
            reports.append({
                "pair": pair, "consumer": consumer,
                "validation_material": after["stable_and_material"],
                "singleton_signs_match": signs_match,
                "holds": bool(after["stable_and_material"] and signs_match),
            })
    return reports, all(row["holds"] for row in reports)


def _serial_stats(stats):
    return {key: value.tolist() for key, value in stats.items()}


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert BRANCHES == ("T", "C", "I", "S") and len(PAIRS) == 6
        assert len(set(POSITION_SHIFTS)) == 16 and min(POSITION_SHIFTS) > 0
        print(json.dumps({
            "status": "dry_run_passed", "rung": 483, "model_loaded": False,
            "discovery_outcomes_opened": False, "validation_outcomes_opened": False,
            "final_or_sealed_opened": False,
            "discovery_batches": 500 // BATCH,
            "registered_predictions": ["pred_a", "pred_b", "pred_c_shared",
                                       "pred_c_split", "pred_c_consumer_specific",
                                       "pred_d", "pred_e"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung483 output namespace already exists")
    rows, fit_rows, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    reference = parent.branch_parent._reference_moments(
        model, fit_rows, torch.device("cuda"))
    discovery_stats, discovery_instrument = collect_phase(
        model, rows, reference, *DISCOVERY_RANGE, DISCOVERY_SPLIT)
    discovery_instrument["position_counts"] = discovery_stats["position_counts"].tolist()
    discovery = analyze_phase(discovery_stats)
    pred_a = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256 and _instrument_valid(
        discovery_instrument, discovery, discovery=True))
    pred_b = discovery["pred_b_tangent_predicts_physical_T_I"]
    c_values = {
        "shared": discovery["pred_c_shared"],
        "split": discovery["pred_c_split"],
        "consumer_specific": discovery["pred_c_consumer_specific"],
    }
    selected_relation = next((key for key, value in c_values.items() if value), None) \
        if sum(c_values.values()) == 1 else None
    pred_d = discovery["pred_d_stable_material_I_pair"]
    validation_licensed = bool(pred_a and pred_b and selected_relation is not None)
    validation_stats = validation_instrument = validation = None
    validation_pair_reports = []
    pred_e = False
    if validation_licensed:
        validation_stats, validation_instrument = collect_phase(
            model, rows, reference, *VALIDATION_RANGE, VALIDATION_SPLIT)
        validation_instrument["position_counts"] = validation_stats["position_counts"].tolist()
        validation = analyze_phase(validation_stats)
        pairs, pairs_hold = _pair_validation(discovery, validation)
        validation_pair_reports = pairs
        pred_e = bool(
            _instrument_valid(validation_instrument, validation, discovery=False)
            and validation["pred_b_tangent_predicts_physical_T_I"]
            and validation[f"pred_c_{selected_relation}"]
            and pairs_hold)
    strong_null = bool(not pred_a or not pred_b or selected_relation is None)
    result = {
        "status": "complete", "rung": 483,
        "claim_level": "immediate_consumer_operational_quotient_identification_screen",
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "consumers": {
            "attention1": "recomputed attention1 write",
            "mlp1_direct": "MLP1 write with attention1 write restored native",
            "mlp1_total": "MLP1 write with attention1 recomputed",
        },
        "branches": list(BRANCHES), "pairs": list(PAIR_NAMES),
        "epsilon": EPSILON, "position_shuffle_offsets": list(POSITION_SHIFTS),
        "discovery": {
            "documents": list(DISCOVERY_RANGE), "split": DISCOVERY_SPLIT,
            "instrument": discovery_instrument, "statistics": _serial_stats(discovery_stats),
            "analysis": discovery,
        },
        "validation": None if validation is None else {
            "documents": list(VALIDATION_RANGE), "split": VALIDATION_SPLIT,
            "instrument": validation_instrument, "statistics": _serial_stats(validation_stats),
            "analysis": validation, "selected_relation": selected_relation,
            "pair_checks": validation_pair_reports,
        },
        "validation_licensed_and_opened": validation_licensed,
        "selected_relation": selected_relation,
        'pred_a_exact_lawful_derivative_instrument': pred_a,
        'pred_b_tangent_predicts_physical_T_I': pred_b,
        'pred_c_exactly_one_operational_relation': selected_relation is not None,
        "c_shared_holds": c_values["shared"], "c_split_holds": c_values["split"],
        "c_consumer_specific_holds": c_values["consumer_specific"],
        'pred_d_stable_material_I_pair': pred_d,
        'pred_e_heldout_documents': pred_e,
        "strong_null": strong_null,
        "final_or_sealed_opened": False,
        "execution_price": {
            "discovery_full_model_forwards": discovery_instrument["calls"]["full_native_forwards"],
            "validation_full_model_forwards": 0 if validation_instrument is None else
            validation_instrument["calls"]["full_native_forwards"],
            "discovery_short_prefix_evaluations": sum(
                discovery_instrument["calls"][key] for key in (
                    "native_prefixes", "directional_derivatives",
                    "finite_difference_prefixes", "singleton_removal_prefixes",
                    "pair_removal_prefixes")),
            "validation_short_prefix_evaluations": 0 if validation_instrument is None else sum(
                validation_instrument["calls"][key] for key in (
                    "native_prefixes", "directional_derivatives",
                    "finite_difference_prefixes", "singleton_removal_prefixes",
                    "pair_removal_prefixes")),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            f"{selected_relation}_T_I_task_conditioned_selective_intervention"
            if pred_e else "task_conditioned_reader_functionals_or_finite_interchange"
            if strong_null else "heldout_relation_failed_no_claim"),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 483,
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "c_relations": c_values, "selected_relation": selected_relation,
        "strong_null": strong_null, "validation_opened": validation_licensed,
        "next_step": result["next_step"], "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
