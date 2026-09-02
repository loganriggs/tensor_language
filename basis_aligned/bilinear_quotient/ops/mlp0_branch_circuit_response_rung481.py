#!/usr/bin/env python3
"""RUNG481 -- exact MLP0 T/C/I/S effects on downstream circuit tags.

Conditional on rung480's valid scientific strong null. All 16 physical branch
subsets and a native baseline are evaluated in one process. Discovery tests
opposing T/I hypotheses: downstream-distinct versus one shared downstream
variable.

pred_a: exact branch identity, in-run native replay, hashes, masks, and counts.
pred_b: T and I have stable circuit-selective responses above controls.
pred_c_split: downstream use distinguishes T from I.
pred_c_shared: downstream use treats T and I as one scalar-related variable.
pred_d: at least one stable material pair interaction involving I.
pred_e: the selected T/I relation and material interactions validate on held-out
        circuit families and documents (opened only after A+B+exactly-one-C).
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
import equality_product_circuit_response_graph_rung477b as circuit_parent
import mlp0_centered_context_anova_factorial as branch_parent
import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent


PREREG = POLY / "MLP0_BRANCH_CIRCUIT_RESPONSE_RUNG481_PREREGISTRATION.md"
R401_SOURCE = ROOT / "ops/mlp0_centered_context_anova_exact_residual.py"
R401_RESULT = ROOT / "mlp0_centered_context_anova_exact_residual_results.json"
R400_SOURCE = ROOT / "ops/mlp0_centered_context_anova_factorial.py"
R477B_SOURCE = ROOT / "ops/equality_product_circuit_response_graph_rung477b.py"
R477B_RESULT = ROOT / "equality_product_circuit_response_graph_rung477b_results.json"
R480_SOURCE = ROOT / "ops/attention0_downstream_canonical_block_rung480.py"
R480_RESULT = ROOT / "attention0_downstream_canonical_block_rung480_results.json"
ROWS_RECEIPT = ROOT / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
OUT = ROOT / "mlp0_branch_circuit_response_rung481_results.json"
HASHES = {
    PREREG: "379ce0ccf327fb49307686fbd47c3d315b104e25ab3e121683016318bc3e3979",
    R401_SOURCE: "7a64736ec5cb4b6fb4088b896c8638138ad2c93c8ceb6c577d7d5782b32cd032",
    R401_RESULT: "6650b97c9f5b53714d29f999eff6653bdbc9273c9238e4c10ce607d8d5728277",
    R400_SOURCE: "1495ec13abf80bbd3d0bf33db8c0457e1bc5eab7421bcb1b96a780278d808322",
    R477B_SOURCE: "ebf9c91e0a823cd263ec997ff185822323d41aadb5f53cdee031bfc8c908cd6b",
    R477B_RESULT: "38349612eb9ca8cf480afe63a1c9cad8c258948ed64383680f42dcf7876a2191",
    R480_SOURCE: "616aa6e103011598fac8ea710b023f7c1cbaf59d96115d17cf04ec14f508b577",
}
BRANCHES = ("T", "C", "I", "S")
PAIRS = tuple(itertools.combinations(range(len(BRANCHES)), 2))
PAIR_NAMES = tuple(f"{BRANCHES[left]}x{BRANCHES[right]}" for left, right in PAIRS)
MASK_TYPES = ("member", "slice_control")
PERMUTATION_SEEDS = tuple(range(2026090281, 2026090297))
DISCOVERY_RANGE = (0, 500)
DISCOVERY_SPLIT = 250
VALIDATION_RANGE = (500, 1000)
VALIDATION_SPLIT = 750
BATCH = 4
D = 1152
TOKENS = 256
FULL_ARM = (1 << len(BRANCHES)) - 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cosine(left, right) -> float:
    left = torch.as_tensor(left, dtype=torch.float64).reshape(-1)
    right = torch.as_tensor(right, dtype=torch.float64).reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return float(torch.dot(left, right) / denominator.clamp_min(1e-30))


def _center(value):
    value = torch.as_tensor(value, dtype=torch.float64)
    return value - value.mean(-1, keepdim=True)


def _relative_squared(difference, reference) -> float:
    difference = torch.as_tensor(difference, dtype=torch.float64)
    reference = torch.as_tensor(reference, dtype=torch.float64)
    return float(difference.square().sum() / reference.square().sum().clamp_min(1e-30))


def _quantile95(values) -> float:
    return float(torch.quantile(
        torch.as_tensor(values, dtype=torch.float64), .95, interpolation="higher"))


def _arm_name(mask: int) -> str:
    names = [name for index, name in enumerate(BRANCHES) if mask & (1 << index)]
    return "+".join(names) if names else "EMPTY"


def _shapley(performance):
    """performance shape [...,16]; return [...,4]."""
    performance = torch.as_tensor(performance, dtype=torch.float64)
    result = torch.zeros(*performance.shape[:-1], len(BRANCHES), dtype=torch.float64)
    n = len(BRANCHES)
    for branch in range(n):
        others = tuple(index for index in range(n) if index != branch)
        for size in range(len(others) + 1):
            coefficient = (math.factorial(size) * math.factorial(n - size - 1)
                           / math.factorial(n))
            for chosen in itertools.combinations(others, size):
                mask = sum(1 << index for index in chosen)
                result[..., branch] += coefficient * (
                    performance[..., mask | (1 << branch)] - performance[..., mask])
    return result


def _pair_interactions(performance):
    """Average second difference over settings of the other two branches."""
    performance = torch.as_tensor(performance, dtype=torch.float64)
    result = torch.zeros(*performance.shape[:-1], len(PAIRS), dtype=torch.float64)
    for pair_index, (left, right) in enumerate(PAIRS):
        others = tuple(index for index in range(len(BRANCHES))
                       if index not in (left, right))
        values = []
        for size in range(len(others) + 1):
            for chosen in itertools.combinations(others, size):
                mask = sum(1 << index for index in chosen)
                values.append(
                    performance[..., mask | (1 << left) | (1 << right)]
                    - performance[..., mask | (1 << left)]
                    - performance[..., mask | (1 << right)]
                    + performance[..., mask])
        result[..., pair_index] = torch.stack(values).mean(0)
    return result


def _mobius(performance):
    performance = torch.as_tensor(performance, dtype=torch.float64)
    output = torch.zeros_like(performance)
    for mask in range(1 << len(BRANCHES)):
        members = tuple(index for index in range(len(BRANCHES)) if mask & (1 << index))
        for size in range(len(members) + 1):
            for chosen in itertools.combinations(members, size):
                child = sum(1 << index for index in chosen)
                output[..., mask] += (-1.0) ** (len(members) - size) * performance[..., child]
    return output


def _exact_components(token_base, attention_write, normalized, reference, left, right, down):
    retained, branches, _g, _gain, collinearity = branch_parent._components(
        token_base, attention_write, normalized, reference, left, right, down)
    z = normalized.float()
    raw = token_base.float() + attention_write.float()
    denominator = raw.square().sum(-1, keepdim=True).clamp_min(1e-30)
    scale = (z * raw).sum(-1, keepdim=True) / denominator
    scaled_raw = scale * raw
    residual = z - scaled_raw
    retained = retained + (
        branch_parent._T(scaled_raw, residual, left, right, down)
        + branch_parent._T(residual, scaled_raw, left, right, down)
        + branch_parent._T(residual, residual, left, right, down))
    return retained, branches, collinearity


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    r401 = json.loads(R401_RESULT.read_text())
    if r401.get("rung") != 401 or any(r401.get(key) is not True for key in (
        "pred_a_exact_residual_identity_and_live_census",
        "pred_b_all_physical_arms_reproduce_rung400",
        "pred_c_inherited_context_outcome_holds_without_bar_change",
        "pred_d_fit_select_sign_and_order_transport",
    )) or r401.get("null_exact_repair_or_context_stability_fails") is not False:
        raise RuntimeError("rung401 exact branch authority changed")
    if not R480_RESULT.is_file():
        raise RuntimeError("conditional route not satisfied: rung480 result is absent")
    r480 = json.loads(R480_RESULT.read_text())
    if r480.get("pred_a_exact_lawful_instrument") is not True \
            or r480.get("strong_null") is not True:
        raise RuntimeError("conditional route not satisfied: rung480 did not have A+scientific null")
    rows, _positive, circuit_masks, _scale, discovery_tags, validation_tags, metadata, _ = \
        circuit_parent.validate_inputs()
    if len(rows) != 1000 or len(discovery_tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("circuit-row or tag authority changed")
    if set(circuit_masks) != set(discovery_tags) | set(validation_tags):
        raise RuntimeError("62-circuit partition changed")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    if len(fit_rows) != 96:
        raise RuntimeError("rung401 reference rows changed")
    return rows, circuit_masks, discovery_tags, validation_tags, fit_rows, {
        **metadata,
        "rung401_result_sha256": sha256(R401_RESULT),
        "rung477b_result_sha256": sha256(R477B_RESULT),
        "rung480_result_sha256": sha256(R480_RESULT),
        "permutation_seeds": list(PERMUTATION_SEEDS),
    }


def _native_forward(model, tokens):
    audit = {"attention": 0, "mlp": 0}

    def attention(event):
        audit["attention"] += 1
        return event.block.attn(event.state, event.first_value)

    def mlp(event):
        audit["mlp"] += 1
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    return logits, audit


def _arm_forward(model, tokens, token_base, reference, mask, cache, diagnostics):
    audit = {"attention": 0, "site0": 0, "other_mlp": 0}
    block0 = model.transformer.h[0]
    left = block0.mlp.Left.weight.detach().float()
    right = block0.mlp.Right.weight.detach().float()
    down = block0.mlp.Down.weight.detach().float()

    def attention(event):
        audit["attention"] += 1
        return event.block.attn(event.state, event.first_value)

    def mlp(event):
        if event.site != 0:
            audit["other_mlp"] += 1
            return event.block.mlp(event.state)
        audit["site0"] += 1
        native = event.block.mlp(event.state)
        if not cache:
            retained, branches, collinearity = _exact_components(
                token_base, event.attention_write, event.state, reference, left, right, down)
            analytical = retained + sum(branches.values(), start=torch.zeros_like(retained))
            direct = branch_parent._T(event.state, event.state, left, right, down)
            deployed = native.float() - event.block.mlp.Down_bias.detach().float()
            diagnostics["analytical_num"] += float(
                (analytical.double() - direct.double()).square().sum())
            diagnostics["analytical_den"] += float(direct.double().square().sum())
            diagnostics["deployed_num"] += float(
                (analytical.double() - deployed.double()).square().sum())
            diagnostics["deployed_den"] += float(deployed.double().square().sum())
            diagnostics["collinearity_max"] = max(
                diagnostics["collinearity_max"], float(collinearity.max()))
            for name in BRANCHES:
                diagnostics["branch_squared_norm"][name] += float(
                    branches[name].double().square().sum())
                diagnostics["branch_deployed_max_abs"][name] = max(
                    diagnostics["branch_deployed_max_abs"][name],
                    float(branches[name].to(native.dtype).abs().max()))
            cache.update({
                "state": event.state.detach().clone(),
                "attention": event.attention_write.detach().clone(),
                "native": native.detach().clone(),
                "branches": {name: value.detach().clone() for name, value in branches.items()},
            })
        else:
            diagnostics["state_replay_max_abs"] = max(
                diagnostics["state_replay_max_abs"],
                float((event.state - cache["state"]).abs().max()),
                float((event.attention_write - cache["attention"]).abs().max()),
                float((native - cache["native"]).abs().max()),
            )
        if mask == FULL_ARM:
            return native
        omitted = sum(
            (cache["branches"][name] for index, name in enumerate(BRANCHES)
             if not mask & (1 << index)),
            start=torch.zeros_like(cache["branches"]["T"]),
        )
        return native - omitted.to(native.dtype)

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True)
    return logits, audit


def _batch_selections(circuit_masks, tags, start, stop, split):
    selections = []
    rows = torch.arange(start, stop)
    for half, condition in enumerate((rows < split, rows >= split)):
        for mask_index, mask_type in enumerate(MASK_TYPES):
            for circuit_index, tag in enumerate(tags):
                selected = circuit_masks[tag][mask_type].view(1000, TOKENS)[start:stop].clone()
                selected &= condition[:, None]
                if selected.any():
                    selections.append((half, mask_index, circuit_index, selected))
    return selections


@torch.no_grad()
def collect_phase(model, rows, circuit_masks, tags, reference, start_doc, stop_doc, split):
    sums = torch.zeros(2, len(MASK_TYPES), len(tags), 16, dtype=torch.float64)
    counts = torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64)
    pooled_sums = torch.zeros(2, 16, dtype=torch.float64)
    pooled_counts = torch.zeros(2, dtype=torch.float64)
    diagnostics = {
        "analytical_num": 0.0, "analytical_den": 0.0,
        "deployed_num": 0.0, "deployed_den": 0.0,
        "collinearity_max": 0.0, "state_replay_max_abs": 0.0,
        "native_full_logits_num": 0.0, "native_full_logits_den": 0.0,
        "native_full_ce_max_abs": 0.0,
        "branch_squared_norm": {name: 0.0 for name in BRANCHES},
        "branch_deployed_max_abs": {name: 0.0 for name in BRANCHES},
    }
    calls = {
        "native_forwards": 0, "native_attention": 0, "native_mlp": 0,
        "arm_forwards": 0, "arm_attention": 0, "arm_site0": 0, "arm_other_mlp": 0,
    }
    device = next(model.parameters()).device
    for start in range(start_doc, stop_doc, BATCH):
        stop = min(start + BATCH, stop_doc)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        targets = batch_rows[:, 1:].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        block0 = model.transformer.h[0]
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        native_logits, native_audit = _native_forward(model, tokens)
        native_nll = F.cross_entropy(
            native_logits.reshape(-1, native_logits.shape[-1]), targets.reshape(-1),
            reduction="none").view(len(batch_rows), -1)
        calls["native_forwards"] += 1
        calls["native_attention"] += native_audit["attention"]
        calls["native_mlp"] += native_audit["mlp"]
        selections = _batch_selections(circuit_masks, tags, start, stop, split)
        for half, mask_index, circuit_index, selected in selections:
            counts[half, mask_index, circuit_index] += int(selected.sum())
        cache = {}
        for mask in range(16):
            logits, audit = _arm_forward(
                model, tokens, token_base, reference, mask, cache, diagnostics)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                reduction="none").view(len(batch_rows), -1)
            for half, mask_index, circuit_index, selected_cpu in selections:
                selected = selected_cpu.to(device)
                sums[half, mask_index, circuit_index, mask] += float(nll[selected].sum())
            global_rows = torch.arange(start, stop, device=device)
            for half, row_condition in enumerate((global_rows < split, global_rows >= split)):
                if row_condition.any():
                    pooled_sums[half, mask] += float(nll[row_condition].sum())
                    if mask == 0:
                        pooled_counts[half] += int(row_condition.sum()) * nll.shape[1]
            calls["arm_forwards"] += 1
            calls["arm_attention"] += audit["attention"]
            calls["arm_site0"] += audit["site0"]
            calls["arm_other_mlp"] += audit["other_mlp"]
            if mask == FULL_ARM:
                difference = logits.float() - native_logits.float()
                diagnostics["native_full_logits_num"] += float(difference.double().square().sum())
                diagnostics["native_full_logits_den"] += float(native_logits.double().square().sum())
                diagnostics["native_full_ce_max_abs"] = max(
                    diagnostics["native_full_ce_max_abs"],
                    float((nll - native_nll).abs().max()))
        del native_logits, native_nll, cache
    batches = math.ceil((stop_doc - start_doc) / BATCH)
    expected_calls = {
        "native_forwards": batches,
        "native_attention": 18 * batches,
        "native_mlp": 18 * batches,
        "arm_forwards": 16 * batches,
        "arm_attention": 16 * 18 * batches,
        "arm_site0": 16 * batches,
        "arm_other_mlp": 16 * 17 * batches,
    }
    instrument = {
        "analytical_identity_relative_squared": diagnostics["analytical_num"]
        / max(diagnostics["analytical_den"], 1e-30),
        "deployed_identity_relative_squared": diagnostics["deployed_num"]
        / max(diagnostics["deployed_den"], 1e-30),
        "native_full_logits_relative_squared": diagnostics["native_full_logits_num"]
        / max(diagnostics["native_full_logits_den"], 1e-30),
        "native_full_ce_max_abs": diagnostics["native_full_ce_max_abs"],
        "state_and_attention_replay_max_abs": diagnostics["state_replay_max_abs"],
        "normalization_noncollinearity_max_relative_squared": diagnostics["collinearity_max"],
        "branch_squared_norm": diagnostics["branch_squared_norm"],
        "branch_deployed_max_abs": diagnostics["branch_deployed_max_abs"],
        "calls": calls, "expected_calls": expected_calls,
        "calls_exact": calls == expected_calls,
    }
    return {
        "ce_sums": sums, "counts": counts,
        "pooled_ce_sums": pooled_sums, "pooled_counts": pooled_counts,
        "instrument": instrument,
    }


def _profile_report(profile, member, control, difficulty, seed_offset=0):
    profile = torch.as_tensor(profile, dtype=torch.float64)
    member = torch.as_tensor(member, dtype=torch.float64)
    control = torch.as_tensor(control, dtype=torch.float64)
    centered = _center(profile)
    difficulty = _center(difficulty)
    beta = float(torch.dot(centered[0], difficulty[0])
                 / difficulty[0].square().sum().clamp_min(1e-30))
    residual = centered - beta * difficulty
    raw_controls, residual_controls = [], []
    for seed in PERMUTATION_SEEDS:
        generator = torch.Generator().manual_seed(seed + seed_offset)
        permutation = torch.randperm(profile.shape[1], generator=generator)
        raw_controls.append(_cosine(centered[0], centered[1, permutation]))
        residual_controls.append(_cosine(residual[0], residual[1, permutation]))
    raw_cosine = _cosine(centered[0], centered[1])
    residual_cosine = _cosine(residual[0], residual[1])
    member_control_ratios = [
        float(torch.linalg.vector_norm(member[half])
              / torch.linalg.vector_norm(control[half]).clamp_min(1e-30))
        for half in range(2)
    ]
    return {
        "profile": profile.tolist(),
        "centered_profile": centered.tolist(),
        "difficulty_beta_fit_half0": beta,
        "difficulty_residual_profile": residual.tolist(),
        "raw_cross_half_cosine": raw_cosine,
        "difficulty_residual_cross_half_cosine": residual_cosine,
        "raw_permutation_cosines": raw_controls,
        "residual_permutation_cosines": residual_controls,
        "raw_permutation_95pct": _quantile95(raw_controls),
        "residual_permutation_95pct": _quantile95(residual_controls),
        "member_control_norm_ratios": member_control_ratios,
    }


def analyze_phase(collection):
    sums = collection["ce_sums"].double()
    counts = collection["counts"].double().clamp_min(1)
    means = sums / counts[..., None]
    performance = -means
    shapley = _shapley(performance)
    interactions = _pair_interactions(performance)
    mobius = _mobius(performance)
    difficulty = means[:, 0, :, FULL_ARM] - means[:, 1, :, FULL_ARM]
    branch_profiles = shapley[:, 0] - shapley[:, 1]
    pair_profiles = interactions[:, 0] - interactions[:, 1]

    branches = {}
    for branch_index, name in enumerate(BRANCHES):
        report = _profile_report(
            branch_profiles[:, :, branch_index],
            shapley[:, 0, :, branch_index], shapley[:, 1, :, branch_index],
            difficulty, seed_offset=100 * branch_index)
        report["median_absolute_selective_effect_by_half"] = [
            float(branch_profiles[half, :, branch_index].abs().median()) for half in range(2)]
        branches[name] = report

    top_two = []
    for half in range(2):
        order = sorted(
            BRANCHES,
            key=lambda name: branches[name]["median_absolute_selective_effect_by_half"][half],
            reverse=True)
        top_two.append(order[:2])

    t_raw = branch_profiles[:, :, BRANCHES.index("T")]
    i_raw = branch_profiles[:, :, BRANCHES.index("I")]
    t_center, i_center = _center(t_raw), _center(i_raw)
    ti_cosines = [_cosine(t_center[half], i_center[half]) for half in range(2)]
    opposite = (t_raw * i_raw < 0).all(0)
    opposite_count = int(opposite.sum())
    opposite_counts_by_floor = {}
    for floor in (0.0, 1e-4, 1e-3, 1e-2):
        material = torch.minimum(t_raw.abs(), i_raw.abs()).min(0).values > floor
        opposite_counts_by_floor[f"{floor:.0e}"] = int((opposite & material).sum())
    alpha = float(torch.dot(t_center[0], i_center[0])
                  / t_center[0].square().sum().clamp_min(1e-30))
    shared_error = float(torch.linalg.vector_norm(i_center[1] - alpha * t_center[1])
                         / torch.linalg.vector_norm(i_center[1]).clamp_min(1e-30))
    shared_sign_mismatches = int((i_raw[1] * (alpha * t_raw[1]) < 0).sum())
    pred_split = bool(
        max(abs(value) for value in ti_cosines) <= .70 and opposite_count >= 8)
    pred_shared = bool(
        min(ti_cosines) >= .90 and shared_error <= .35 and shared_sign_mismatches <= 3)

    pairs = {}
    stable_material_pairs = []
    for pair_index, name in enumerate(PAIR_NAMES):
        left, right = PAIRS[pair_index]
        report = _profile_report(
            pair_profiles[:, :, pair_index],
            interactions[:, 0, :, pair_index], interactions[:, 1, :, pair_index],
            difficulty, seed_offset=1000 + 100 * pair_index)
        relative = []
        for half in range(2):
            numerator = torch.linalg.vector_norm(pair_profiles[half, :, pair_index])
            denominator = min(
                torch.linalg.vector_norm(branch_profiles[half, :, left]),
                torch.linalg.vector_norm(branch_profiles[half, :, right]))
            relative.append(float(numerator / denominator.clamp_min(1e-30)))
        report["relative_to_smaller_singleton_norm_by_half"] = relative
        report["pooled_selective_median"] = float(pair_profiles[:, :, pair_index].median())
        material = bool(
            min(relative) >= .20
            and report["raw_cross_half_cosine"] >= .60
            and report["difficulty_residual_cross_half_cosine"] >= .60
            and report["raw_cross_half_cosine"] >= report["raw_permutation_95pct"] + .15
            and report["difficulty_residual_cross_half_cosine"]
            >= report["residual_permutation_95pct"] + .15)
        report["stable_and_material"] = material
        if material:
            stable_material_pairs.append(name)
        pairs[name] = report

    pred_b = True
    for name in ("T", "I"):
        report = branches[name]
        pred_b &= (
            report["raw_cross_half_cosine"] >= .70
            and report["difficulty_residual_cross_half_cosine"] >= .70
            and report["raw_cross_half_cosine"] >= report["raw_permutation_95pct"] + .15
            and report["difficulty_residual_cross_half_cosine"]
            >= report["residual_permutation_95pct"] + .15
            and min(report["member_control_norm_ratios"]) >= 1.25)
    pred_b = bool(pred_b and all(set(names) == {"T", "I"} for names in top_two))
    pred_d = any("I" in name.split("x") for name in stable_material_pairs)
    pooled_ce = collection["pooled_ce_sums"].double() \
        / collection["pooled_counts"].double()[:, None].clamp_min(1)
    return {
        "mean_ce": means.tolist(),
        "pooled_ce_by_half_and_arm": {
            f"half{half}": {_arm_name(mask): float(pooled_ce[half, mask])
                            for mask in range(16)} for half in range(2)},
        "shapley_by_half_mask_circuit_branch": shapley.tolist(),
        "pair_interactions_by_half_mask_circuit_pair": interactions.tolist(),
        "mobius_by_half_mask_circuit_arm": mobius.tolist(),
        "full_model_difficulty_profile": difficulty.tolist(),
        "branch_reports": branches, "pair_reports": pairs,
        "top_two_branches_by_half": top_two,
        "ti_relation": {
            "cosine_by_half": ti_cosines,
            "opposite_sign_count_both_halves": opposite_count,
            "opposite_sign_counts_by_minimum_absolute_effect_floor": opposite_counts_by_floor,
            "shared_alpha_fit_half0": alpha,
            "shared_scale_error_half1": shared_error,
            "shared_sign_mismatches_half1": shared_sign_mismatches,
        },
        "stable_material_pairs": stable_material_pairs,
        "pred_b_stable_circuit_selective_T_and_I": pred_b,
        "pred_c_split": pred_split, "pred_c_shared": pred_shared,
        "pred_d_stable_material_I_interaction": bool(pred_d),
    }


def _instrument_valid(collection, *, require_discovery_support):
    instrument = collection["instrument"]
    support = collection["counts"]
    support_ok = bool((support > 0).all())
    if require_discovery_support:
        support_ok = support_ok and int(support[:, 0].min()) >= 39 \
            and int(support[:, 1].min()) >= 439
    return bool(
        instrument["analytical_identity_relative_squared"] <= 1e-8
        and instrument["deployed_identity_relative_squared"] <= 1e-5
        and instrument["native_full_logits_relative_squared"] <= 1e-12
        and instrument["native_full_ce_max_abs"] <= 1e-7
        and instrument["state_and_attention_replay_max_abs"] == 0.0
        and instrument["calls_exact"] and support_ok
        and all(value > 0 for value in instrument["branch_squared_norm"].values())
        and all(value > 0 for value in instrument["branch_deployed_max_abs"].values()))


def _validation_holds(discovery, validation, selected_relation):
    if not validation["pred_b_stable_circuit_selective_T_and_I"]:
        return False, []
    relation_holds = validation[f"pred_c_{selected_relation}"]
    pair_checks = []
    for name in discovery["stable_material_pairs"]:
        before = discovery["pair_reports"][name]
        after = validation["pair_reports"][name]
        before_sign = math.copysign(1.0, before["pooled_selective_median"] or 1.0)
        after_sign = math.copysign(1.0, after["pooled_selective_median"] or 1.0)
        pair_checks.append({
            "pair": name,
            "validation_cross_half_cosine": after["raw_cross_half_cosine"],
            "median_sign_matches_discovery": before_sign == after_sign,
            "holds": after["raw_cross_half_cosine"] >= .50 and before_sign == after_sign,
        })
    return bool(relation_holds and all(row["holds"] for row in pair_checks)), pair_checks


def _serial_collection(collection):
    return {
        "ce_sums": collection["ce_sums"].tolist(),
        "counts": collection["counts"].tolist(),
        "pooled_ce_sums": collection["pooled_ce_sums"].tolist(),
        "pooled_counts": collection["pooled_counts"].tolist(),
        "instrument": collection["instrument"],
    }


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(BRANCHES) == 4 and len(PAIRS) == 6 and len(PERMUTATION_SEEDS) == 16
        assert FULL_ARM == 15 and DISCOVERY_SPLIT == 250 and VALIDATION_SPLIT == 750
        print(json.dumps({
            "status": "dry_run_passed", "rung": 481, "model_loaded": False,
            "discovery_outcomes_opened": False, "validation_outcomes_opened": False,
            "final_or_sealed_opened": False,
            "discovery_forwards": 500 // BATCH * 17,
            "conditional_validation_forwards": 500 // BATCH * 17,
            "predictions": ["pred_a", "pred_b", "pred_c_split", "pred_c_shared",
                            "pred_d", "pred_e"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung481 output namespace already exists")
    rows, circuit_masks, discovery_tags, validation_tags, fit_rows, metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    reference = branch_parent._reference_moments(model, fit_rows, torch.device("cuda"))
    discovery_collection = collect_phase(
        model, rows, circuit_masks, discovery_tags, reference,
        DISCOVERY_RANGE[0], DISCOVERY_RANGE[1], DISCOVERY_SPLIT)
    discovery = analyze_phase(discovery_collection)
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and _instrument_valid(discovery_collection, require_discovery_support=True))
    pred_b = discovery["pred_b_stable_circuit_selective_T_and_I"]
    pred_c_split = discovery["pred_c_split"]
    pred_c_shared = discovery["pred_c_shared"]
    pred_d = discovery["pred_d_stable_material_I_interaction"]
    validation_licensed = bool(pred_a and pred_b and (pred_c_split != pred_c_shared))
    validation_collection = validation = None
    validation_pair_checks = []
    pred_e = False
    selected_relation = None
    if validation_licensed:
        selected_relation = "split" if pred_c_split else "shared"
        validation_collection = collect_phase(
            model, rows, circuit_masks, validation_tags, reference,
            VALIDATION_RANGE[0], VALIDATION_RANGE[1], VALIDATION_SPLIT)
        validation = analyze_phase(validation_collection)
        validation_science, validation_pair_checks = _validation_holds(
            discovery, validation, selected_relation)
        pred_e = bool(
            _instrument_valid(validation_collection, require_discovery_support=False)
            and validation_science)
    strong_null = bool(not pred_a or not pred_b or not (pred_c_split != pred_c_shared))
    result = {
        "status": "complete", "rung": 481,
        "claim_level": "exact_branch_by_circuit_grouping_screen_not_compression",
        "conditional_route_satisfied": True,
        "source_hashes": {str(path): sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branch_definitions": {
            "T": "fixed-gain token main effect", "C": "fixed-gain context main effect",
            "I": "fixed-gain token-by-context bilinear interaction",
            "S": "example-specific normalization-gain effect",
        },
        "discovery": {
            "documents": list(DISCOVERY_RANGE), "split": DISCOVERY_SPLIT,
            "tags": discovery_tags, "collection": _serial_collection(discovery_collection),
            "analysis": discovery,
        },
        "validation": None if validation is None else {
            "documents": list(VALIDATION_RANGE), "split": VALIDATION_SPLIT,
            "tags": validation_tags, "collection": _serial_collection(validation_collection),
            "analysis": validation, "selected_relation": selected_relation,
            "material_pair_checks": validation_pair_checks,
        },
        "validation_licensed_and_opened": validation_licensed,
        'pred_a_exact_lawful_instrument': pred_a,
        'pred_b_stable_circuit_selective_T_and_I': pred_b,
        'pred_c_exactly_one_T_I_relation': pred_c_split != pred_c_shared,
        "c_split_holds": pred_c_split, "c_shared_holds": pred_c_shared,
        'pred_d_stable_material_I_interaction': pred_d,
        'pred_e_heldout_circuits_and_documents': pred_e,
        "strong_null": strong_null,
        "final_or_sealed_opened": False,
        "execution_price": {
            "discovery_forwards": discovery_collection["instrument"]["calls"]["native_forwards"]
            + discovery_collection["instrument"]["calls"]["arm_forwards"],
            "validation_forwards": 0 if validation_collection is None else
            validation_collection["instrument"]["calls"]["native_forwards"]
            + validation_collection["instrument"]["calls"]["arm_forwards"],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            f"{selected_relation}_T_I_downstream_response_decomposition"
            if pred_e else
            "consumer_specific_attention1_mlp1_jacobians" if strong_null else
            "heldout_relation_failed_no_claim"
        ),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": 481,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "validation_opened": validation_licensed,
        "selected_relation": selected_relation, "next_step": result["next_step"],
        "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
