#!/usr/bin/env python3
"""RUNG511 -- exact L/R/LR score-change branches and finite factorial effects."""

# BQGATE: EXPERIMENT
# pred_a: exact three-branch/factorial instrument is live
# pred_b: a fixed same-subset cross-action relation passes discovery
# pred_c: a frozen relation predicts new documents and circuit families
# pred_d: a relation passes bidirectional physical branch substitution
# pred_e: a two-branch relation composes predictably and is selective

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
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp10_observable_predictive_state_quotient_rung510 as r510


PREREG = POLY / "MLP10_SCORE_CHANGE_THREE_BRANCH_FACTORIAL_RUNG511_PREREGISTRATION.md"
R510_RESULT = ROOT / "mlp10_observable_predictive_state_quotient_rung510_results.json"
R510_BUNDLE = ROOT / "mlp10_observable_predictive_state_quotient_rung510_bundle.pt"
R510_SOURCE = ROOT / "ops/mlp10_observable_predictive_state_quotient_rung510.py"
OUT = ROOT / "mlp10_score_change_three_branch_factorial_rung511_results.json"
BUNDLE = ROOT / "mlp10_score_change_three_branch_factorial_rung511_bundle.pt"

HASHES = {
    PREREG: "95a296478a5adc21ef0ef9bf8a1762ddd86e8f1312258733ce1de2eb2d9b4cd4",
    R510_RESULT: "16d100e7b92152fc70939b000934699882605c30c513c570f6c519b80f943177",
    R510_BUNDLE: "a8832624c94e3e9aa491d26290e55a14f94aa103eb7cddc3df3a0e1b34c3eed7",
    R510_SOURCE: "7901aa5d9c7c39bf5666e0f081bfe08047f23c73eec08b12508c601def7b967a",
}

DISCOVERY = (500, 748, 624)
CONFIRMATION = (752, 1000, 876)
BRANCH_NAMES = ("L", "R", "LR")
SUBSET_MASKS = (1, 2, 4, 3, 5, 6, 7)
SUBSET_NAMES = tuple(
    "+".join(BRANCH_NAMES[index] for index in range(3) if mask & (1 << index))
    for mask in SUBSET_MASKS)
ARMS = ("intact",) + SUBSET_NAMES
N_ACTIONS = len(r510.r509.parent.SOURCES)
N_SUBSETS = len(SUBSET_MASKS)
N_NODES = N_ACTIONS * N_SUBSETS
NODE_NAMES = tuple(
    f"{source}::{subset}"
    for source in r510.r509.parent.SOURCES for subset in SUBSET_NAMES)
CONTROL_SEEDS = tuple(range(51100, 51116))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def node_parts(node: int) -> tuple[int, int]:
    if not 0 <= node < N_NODES:
        raise ValueError("rung511 node index changed")
    return divmod(node, N_SUBSETS)


def _relative_squared(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    return float((actual.float() - predicted.float()).square().sum()
                 / actual.float().square().sum().clamp_min(1e-30))


def _captured_forward(model, tokens, scales, **kwargs):
    """Use the audited explicit forward while observing MLP10's exact input once."""
    observed = []

    def capture_input(_module, arguments):
        if len(arguments) != 1:
            raise RuntimeError("MLP10 hook argument count changed")
        observed.append(arguments[0].detach())

    mlp = model.transformer.h[r510.r509.parent.TARGET].mlp
    handle = mlp.register_forward_pre_hook(capture_input)
    try:
        logits, capture, diagnostics, audit = r510.r509.parent._forward(
            model, tokens, scales, capture_mlp10=True, **kwargs)
    finally:
        handle.remove()
    if len(observed) != 1:
        raise RuntimeError(f"MLP10 input hook fired {len(observed)} times")
    capture = dict(capture)
    capture["z"] = observed[0]
    return logits, capture, diagnostics, audit


@torch.no_grad()
def deployed_branches(mlp, absent: dict, current: dict) -> tuple[tuple[torch.Tensor, ...], dict]:
    """Return exact deployed Möbius branches L, R, LR in float32 output space."""
    z0, za = absent["z"], current["z"]
    l0, r0 = mlp.Left(z0), mlp.Right(z0)
    la, ra = mlp.Left(za), mlp.Right(za)
    f00 = mlp.Down(l0 * r0)
    f10 = mlp.Down(la * r0)
    f01 = mlp.Down(l0 * ra)
    f11 = mlp.Down(la * ra)
    absent_replay = f00 + mlp.Down_bias
    current_replay = f11 + mlp.Down_bias
    total = current["deployed_write"].float() - absent["deployed_write"].float()
    left = f10.float() - f00.float()
    right = f01.float() - f00.float()
    joint = total - left - right

    l0f = F.linear(z0.float(), mlp.Left.weight.float())
    r0f = F.linear(z0.float(), mlp.Right.weight.float())
    laf = F.linear(za.float(), mlp.Left.weight.float())
    raf = F.linear(za.float(), mlp.Right.weight.float())
    dl, dr = laf - l0f, raf - r0f
    ideal = (
        F.linear(dl * r0f, mlp.Down.weight.float()),
        F.linear(l0f * dr, mlp.Down.weight.float()),
        F.linear(dl * dr, mlp.Down.weight.float()),
    )
    branches = (left, right, joint)
    diagnostics = {
        "absent_corner_replay_max_abs": float(
            (absent_replay - absent["deployed_write"]).float().abs().max()),
        "current_corner_replay_max_abs": float(
            (current_replay - current["deployed_write"]).float().abs().max()),
        "deployed_branch_sum_relative_squared": _relative_squared(total, sum(branches)),
        "float32_ideal_branch_relative_squared": [
            _relative_squared(ideal[index], branches[index]) for index in range(3)],
        "float32_ideal_sum_relative_squared": _relative_squared(total, sum(ideal)),
        "branch_rms": [float(value.square().mean().sqrt()) for value in branches],
        "total_rms": float(total.square().mean().sqrt()),
    }
    return branches, diagnostics


def subset_output(branches: tuple[torch.Tensor, ...], subset_index: int) -> torch.Tensor:
    mask = SUBSET_MASKS[subset_index]
    selected = [branches[index] for index in range(3) if mask & (1 << index)]
    if not selected:
        raise RuntimeError("empty branch subset")
    return sum(selected)


def _empty_diagnostics() -> dict:
    row = r510.r509._empty_diagnostics()
    row.update({
        "hooks": 0, "hooks_expected": 0, "hooks_exact": False,
        "four_corner_replays": 0, "four_corner_replays_expected": 0,
        "four_corner_replays_exact": False,
        "absent_corner_replay_max_abs": 0.0,
        "current_corner_replay_max_abs": 0.0,
        "deployed_branch_sum_relative_squared": 0.0,
        "float32_ideal_branch_relative_squared": [0.0, 0.0, 0.0],
        "float32_ideal_sum_relative_squared": 0.0,
        "subset_patches": 0, "subset_patches_expected": 0,
        "subset_patches_exact": False,
    })
    return row


def _update_branch_diagnostics(total: dict, current: dict) -> None:
    for key in (
        "absent_corner_replay_max_abs", "current_corner_replay_max_abs",
        "deployed_branch_sum_relative_squared", "float32_ideal_sum_relative_squared",
    ):
        total[key] = max(total[key], current[key])
    total["float32_ideal_branch_relative_squared"] = [
        max(total["float32_ideal_branch_relative_squared"][index], current["float32_ideal_branch_relative_squared"][index])
        for index in range(3)]


@torch.no_grad()
def collect_factorial(model, rows, task_masks, circuit_masks, circuit_tags,
                      scales, bounds):
    lo, hi, _split = bounds
    documents = hi - lo
    task = torch.zeros(N_ACTIONS, len(ARMS), documents,
                       len(r510.r509.parent.TASK_CELLS), dtype=torch.float64)
    counts = torch.zeros(documents, len(r510.r509.parent.TASK_CELLS), dtype=torch.float64)
    base_task = torch.zeros_like(counts)
    circuit_sums = torch.zeros(
        N_ACTIONS, len(ARMS), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    diagnostics = _empty_diagnostics()
    device = next(model.parameters()).device
    mlp = model.transformer.h[r510.r509.parent.TARGET].mlp
    for start in range(lo, hi, r510.r509.parent.BATCH):
        stop = start + r510.r509.parent.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop]
                 for cell in r510.r509.parent.TASK_CELLS}
        direct_logits, _, direct_diag, _ = r510.r509.parent._forward(
            model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        r510.r509._update_diagnostics(diagnostics, direct_diag)
        absent_logits, absent, absent_diag, _ = _captured_forward(
            model, tokens, scales, action="P", absent=True)
        diagnostics["calls"]["analytical"] += 1
        diagnostics["hooks"] += 1
        r510.r509._update_diagnostics(diagnostics, absent_diag)
        base_task[local:local + r510.r509.parent.BATCH] = r510.r509.parent._task_sums(
            r510.r509.parent._nll(direct_logits, batch_rows).detach().cpu().unsqueeze(0), masks)[0]
        action_nll = []
        for action_index, source in enumerate(r510.r509.parent.SOURCES):
            logits, current, current_diag, _ = _captured_forward(
                model, tokens, scales, action=source)
            diagnostics["calls"]["analytical"] += 1
            diagnostics["hooks"] += 1
            r510.r509._update_diagnostics(diagnostics, current_diag)
            r510.r509.parent._score_delta_closure(diagnostics, current, absent)
            if source == "N":
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"],
                    float((logits.float() - direct_logits.float()).abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    _relative_squared(direct_logits, logits))
            branches, branch_diag = deployed_branches(mlp, absent, current)
            diagnostics["four_corner_replays"] += 1
            _update_branch_diagnostics(diagnostics, branch_diag)
            nll_rows = [r510.r509.parent._nll(logits, batch_rows).detach().cpu()]
            for subset_index in range(N_SUBSETS):
                delta = subset_output(branches, subset_index)
                replacement = current["deployed_write"].float() - delta
                edited_logits, _captures, patch_diag, patch_audit = \
                    r510.r509.parent.score_parent.run_forward(
                        model, tokens, action=source, scales=scales,
                        patch_writes={"m10": replacement.to(current["deployed_write"].dtype)})
                diagnostics["calls"]["analytical"] += 1
                diagnostics["subset_patches"] += patch_audit["patches"]
                edit_rms = patch_diag["patch_rms_max"]
                diagnostics["zero_term_edits"] += int(edit_rms <= 0)
                if edit_rms > 0:
                    diagnostics["minimum_nonzero_term_edit_rms"] = min(
                        diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
                nll_rows.append(
                    r510.r509.parent._nll(edited_logits, batch_rows).detach().cpu())
            nll_stack = torch.stack(nll_rows)
            task[action_index, :, local:local + r510.r509.parent.BATCH] = \
                r510.r509.parent._task_sums(nll_stack, masks)
            action_nll.append(nll_stack)
        counts[local:local + r510.r509.parent.BATCH] = torch.stack(
            [masks[cell].sum(1).double() for cell in r510.r509.parent.TASK_CELLS], -1)
        matrix, observed = r510.r509.parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        circuit_counts += observed
        for action_index, nll_stack in enumerate(action_nll):
            circuit_sums[action_index] += torch.matmul(
                nll_stack.view(len(ARMS), -1).double(), matrix.T,
            ).view(len(ARMS), 2, 2, len(circuit_tags))
    batches = documents // r510.r509.parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches, "analytical": batches * (1 + N_ACTIONS * (1 + N_SUBSETS))}
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["hooks_expected"] = batches * (1 + N_ACTIONS)
    diagnostics["hooks_exact"] = diagnostics["hooks"] == diagnostics["hooks_expected"]
    diagnostics["four_corner_replays_expected"] = batches * N_ACTIONS
    diagnostics["four_corner_replays_exact"] = (
        diagnostics["four_corner_replays"] == diagnostics["four_corner_replays_expected"])
    diagnostics["subset_patches_expected"] = batches * N_ACTIONS * N_SUBSETS
    diagnostics["subset_patches_exact"] = (
        diagnostics["subset_patches"] == diagnostics["subset_patches_expected"])
    diagnostics["patches"] = diagnostics["subset_patches"]
    diagnostics["patches_expected"] = diagnostics["subset_patches_expected"]
    diagnostics["patches_exact"] = diagnostics["subset_patches_exact"]
    return {
        "bounds": bounds, "arms": ARMS, "task": task, "task_counts": counts,
        "base_task": base_task, "source_task": task[:, 0],
        "circuit_tags": tuple(circuit_tags), "circuit_sums": circuit_sums,
        "circuit_counts": circuit_counts, "diagnostics": diagnostics,
    }


def _instrument(collection: dict) -> bool:
    d = collection["diagnostics"]
    return bool(
        d["calls_exact"] and d["hooks_exact"] and d["four_corner_replays_exact"]
        and d["subset_patches_exact"] and d["zero_term_edits"] == 0
        and d["factor_reconstruction_max"] <= 1e-10
        and d["raw_source_relative_squared"] <= r510.r509.parent.DEPLOYED_BF16_BAR
        and d["normalized_closure_relative_squared"] <= 1e-12
        and d["normalized_numerical_rms_ratio"] <= .02
        and d["float32_mlp10_closure"] <= 1e-10
        and d["deployed_mlp10_relative_squared"] <= r510.r509.parent.DEPLOYED_BF16_BAR
        and d["score_delta_float32_closure"] <= 1e-10
        and math.isfinite(d["score_delta_predeployment_relative_squared"])
        and d["score_delta_deployed_closure_relative_squared"] <= 1e-12
        and d["minimum_nonzero_score_edit_rms"] > 0
        and d["minimum_nonzero_term_edit_rms"] > 0
        and d["native_replay_logit_max_abs"] == 0.0
        and d["native_replay_relative_squared"] <= 1e-12
        and d["absent_corner_replay_max_abs"] == 0.0
        and d["current_corner_replay_max_abs"] == 0.0
        and d["deployed_branch_sum_relative_squared"] <= 1e-12)


def _task_vector(collection: dict, node: int, window: str) -> torch.Tensor:
    action, subset = node_parts(node)
    source = r510.r509.parent.SOURCES[action]
    return r510.r509.parent.finite_vector(
        collection, SUBSET_NAMES[subset], collection, source, window).double()


def _circuit_vector(collection: dict, node: int, window: str) -> torch.Tensor:
    action, subset = node_parts(node)
    source = r510.r509.parent.SOURCES[action]
    return r510.r509._circuit_fingerprint(
        collection, SUBSET_NAMES[subset], source, window).double()


def response_matrices(collection: dict) -> dict[str, dict[str, torch.Tensor]]:
    return {
        window: {
            "task": torch.stack([_task_vector(collection, node, window)
                                 for node in range(N_NODES)]),
            "circuit": torch.stack([_circuit_vector(collection, node, window)
                                    for node in range(N_NODES)]),
        }
        for window in ("half0", "half1", "pooled")}


def discover_relations(matrices: dict) -> tuple[list[dict], dict]:
    passing, checks = [], {}
    tested = 0
    for subset in range(N_SUBSETS):
        nodes = [action * N_SUBSETS + subset for action in range(N_ACTIONS)]
        for left, right in itertools.combinations(nodes, 2):
            tested += 1
            c0_left = matrices["half0"]["circuit"][left]
            c0_right = matrices["half0"]["circuit"][right]
            beta = float(torch.dot(c0_left, c0_right)
                         / torch.dot(c0_right, c0_right).clamp_min(1e-30))
            metrics = r510._pair_metrics(matrices, left, right, beta)
            key = f"{NODE_NAMES[left]} <-> {NODE_NAMES[right]}"
            checks[key] = metrics
            if metrics["holds"]:
                passing.append({
                    "left_node": left, "right_node": right,
                    "left_name": NODE_NAMES[left], "right_name": NODE_NAMES[right],
                    "subset_index": subset, "subset_name": SUBSET_NAMES[subset],
                    "beta_left_from_right": beta,
                })
    if tested != 42:
        raise RuntimeError(f"same-subset relation count changed: {tested}")
    return passing, {"relations_tested": tested, "candidate_count": len(passing),
                     "material_nodes": int(sum(
                         matrices["pooled"]["circuit"][node].square().mean().sqrt() >= .0005
                         and torch.linalg.vector_norm(matrices["pooled"]["task"][node]) >= .00025
                         for node in range(N_NODES))), "checks": checks}


def permutation_control_counts(matrices: dict) -> list[int]:
    counts = []
    dimensions = matrices["half0"]["circuit"].shape[1]
    for seed in CONTROL_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        keys = torch.rand(N_NODES, dimensions, generator=generator)
        order = keys.argsort(dim=1)
        control = {
            window: {
                "task": matrices[window]["task"],
                "circuit": torch.gather(matrices[window]["circuit"], 1, order),
            }
            for window in ("half0", "half1", "pooled")}
        candidates, _summary = discover_relations(control)
        counts.append(len(candidates))
    return counts


def confirm_relations(matrices: dict, candidates: list[dict]) -> tuple[list[dict], dict]:
    passing, checks = [], {}
    for candidate in candidates:
        metrics = r510._pair_metrics(
            matrices, candidate["left_node"], candidate["right_node"],
            candidate["beta_left_from_right"])
        holds = bool(metrics["material"] and metrics["scale_holds"])
        for window in ("half0", "half1", "pooled"):
            c = metrics["windows"][window]["circuit"]
            t = metrics["windows"][window]["task"]
            window_holds = bool(
                min(c["left_from_right_cosine"], c["right_from_left_cosine"]) >= .75
                and max(c["left_from_right_relative_residual"],
                        c["right_from_left_relative_residual"]) <= .55
                and min(t["left_from_right_cosine"], t["right_from_left_cosine"]) >= .70
                and max(t["left_from_right_relative_residual"],
                        t["right_from_left_relative_residual"]) <= .65)
            metrics["windows"][window]["confirmation_holds"] = window_holds
            holds &= window_holds
        metrics["holds"] = bool(holds)
        key = f"{candidate['left_name']} <-> {candidate['right_name']}"
        checks[key] = metrics
        if holds:
            passing.append(candidate)
    return passing, checks


def _effect(collection: dict, action: int, subset: int, window: str,
            kind: str) -> torch.Tensor:
    node = action * N_SUBSETS + subset
    return (_task_vector if kind == "task" else _circuit_vector)(collection, node, window)


def mobius_effect(collection: dict, action: int, mask: int,
                  window: str, kind: str) -> torch.Tensor:
    total = None
    submask = mask
    while submask:
        subset = SUBSET_MASKS.index(submask)
        sign = -1 if ((mask.bit_count() - submask.bit_count()) % 2) else 1
        value = sign * _effect(collection, action, subset, window, kind)
        total = value if total is None else total + value
        submask = (submask - 1) & mask
    if total is None:
        raise RuntimeError("empty Möbius effect")
    return total


def _composition_rule(collection: dict, candidate: dict) -> dict:
    subset = candidate["subset_index"]
    mask = SUBSET_MASKS[subset]
    if mask.bit_count() < 2 or mask == 7:
        return {"kind": "ineligible_subset", "holds": False, "discovery": {}}
    actions = [node_parts(candidate[key])[0] for key in ("left_node", "right_node")]
    beta = candidate["beta_left_from_right"]
    report = {}
    additive = True
    stable = True
    material_terms = 0
    for window in ("half0", "half1", "pooled"):
        window_rows = []
        for higher in SUBSET_MASKS:
            if higher.bit_count() < 2 or higher & mask != higher:
                continue
            entries = []
            for action in actions:
                interaction_c = mobius_effect(collection, action, higher, window, "circuit")
                interaction_t = mobius_effect(collection, action, higher, window, "task")
                joint_c = _effect(collection, action, subset, window, "circuit")
                joint_t = _effect(collection, action, subset, window, "task")
                circuit_ratio = float(torch.linalg.vector_norm(interaction_c)
                                      / torch.linalg.vector_norm(joint_c).clamp_min(1e-30))
                task_ratio = float(torch.linalg.vector_norm(interaction_t)
                                   / torch.linalg.vector_norm(joint_t).clamp_min(1e-30))
                entries.append((interaction_c, interaction_t, circuit_ratio, task_ratio))
                additive &= circuit_ratio <= .25 and task_ratio <= .25
            left_c, left_t, _, _ = entries[0]
            right_c, right_t, _, _ = entries[1]
            material = bool(
                min(float(left_c.square().mean().sqrt()),
                    float(right_c.square().mean().sqrt())) >= .0005
                and min(float(torch.linalg.vector_norm(left_t)),
                        float(torch.linalg.vector_norm(right_t))) >= .00025)
            if material:
                material_terms += int(window == "pooled")
                for actual, donor in ((left_c, right_c), (left_t, right_t)):
                    cosine = float(r510._safe_cosine_rows(actual[None], (beta * donor)[None])[0])
                    residual = float(r510._residual_rows(actual[None], (beta * donor)[None])[0])
                    if window != "pooled":
                        stable &= cosine >= .70 and residual <= .65
            window_rows.append({
                "interaction": "+".join(BRANCH_NAMES[i] for i in range(3) if higher & (1 << i)),
                "circuit_ratios_to_joint": [entries[0][2], entries[1][2]],
                "task_ratios_to_joint": [entries[0][3], entries[1][3]],
                "material": material,
            })
        report[window] = window_rows
    if additive:
        kind = "additive"
    elif material_terms > 0 and stable:
        kind = "interaction_stable"
    else:
        kind = "none"
    return {"kind": kind, "holds": kind != "none", "discovery": report,
            "material_higher_order_terms": material_terms}


def _composition_confirmation(collection: dict, candidate: dict, rule: dict) -> dict:
    fresh = _composition_rule(collection, candidate)
    if rule["kind"] == "additive":
        holds = fresh["kind"] == "additive"
    elif rule["kind"] == "interaction_stable":
        holds = fresh["kind"] == "interaction_stable"
    else:
        holds = False
    return {"frozen_kind": rule["kind"], "fresh": fresh, "holds": bool(holds)}


def _selective(collection: dict, candidate: dict) -> dict:
    subset = candidate["subset_index"]
    rows, holds = {}, True
    for node_key in ("left_node", "right_node"):
        action, _ = node_parts(candidate[node_key])
        source = r510.r509.parent.SOURCES[action]
        source_rows = {}
        for window in ("half0", "half1"):
            all_copy, off_target = r510.r509.parent.finite_all_off(
                collection, SUBSET_NAMES[subset], collection, source, window)
            cell_holds = bool(abs(all_copy) >= .002
                              and abs(all_copy) >= 3 * abs(off_target))
            source_rows[window] = {"all_copy_nat": all_copy,
                                   "off_target_nat": off_target,
                                   "holds": cell_holds}
            holds &= cell_holds
        rows[source] = source_rows
    return {"sources": rows, "holds": bool(holds)}


@torch.no_grad()
def collect_substitutions(model, rows, task_masks, circuit_masks, circuit_tags,
                          scales, bounds, candidates):
    lo, hi, _split = bounds
    documents = hi - lo
    directions = []
    for edge_index, candidate in enumerate(candidates):
        directions.extend((
            {"edge": edge_index, "target": candidate["left_node"],
             "donor": candidate["right_node"], "scale": candidate["beta_left_from_right"],
             "side": "left_from_right"},
            {"edge": edge_index, "target": candidate["right_node"],
             "donor": candidate["left_node"], "scale": 1.0 / candidate["beta_left_from_right"],
             "side": "right_from_left"},
        ))
    task = torch.zeros(len(directions), documents, len(r510.r509.parent.TASK_CELLS),
                       dtype=torch.float64)
    counts = torch.zeros(documents, len(r510.r509.parent.TASK_CELLS), dtype=torch.float64)
    circuit_sums = torch.zeros(len(directions), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    diagnostics = _empty_diagnostics()
    diagnostics["substitution_patches"] = 0
    diagnostics["substitution_patches_expected"] = 0
    diagnostics["substitution_patches_exact"] = False
    device = next(model.parameters()).device
    mlp = model.transformer.h[r510.r509.parent.TARGET].mlp
    for start in range(lo, hi, r510.r509.parent.BATCH):
        stop = start + r510.r509.parent.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop]
                 for cell in r510.r509.parent.TASK_CELLS}
        direct_logits, _, direct_diag, _ = r510.r509.parent._forward(
            model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        r510.r509._update_diagnostics(diagnostics, direct_diag)
        _absent_logits, absent, absent_diag, _ = _captured_forward(
            model, tokens, scales, action="P", absent=True)
        diagnostics["calls"]["analytical"] += 1
        diagnostics["hooks"] += 1
        r510.r509._update_diagnostics(diagnostics, absent_diag)
        captures, branch_sets = [], []
        for source in r510.r509.parent.SOURCES:
            logits, current, current_diag, _ = _captured_forward(
                model, tokens, scales, action=source)
            diagnostics["calls"]["analytical"] += 1
            diagnostics["hooks"] += 1
            r510.r509._update_diagnostics(diagnostics, current_diag)
            r510.r509.parent._score_delta_closure(diagnostics, current, absent)
            if source == "N":
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"],
                    float((logits.float() - direct_logits.float()).abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    _relative_squared(direct_logits, logits))
            branches, branch_diag = deployed_branches(mlp, absent, current)
            diagnostics["four_corner_replays"] += 1
            _update_branch_diagnostics(diagnostics, branch_diag)
            captures.append(current)
            branch_sets.append(branches)
        nll_rows = []
        for direction in directions:
            target_action, target_subset = node_parts(direction["target"])
            donor_action, donor_subset = node_parts(direction["donor"])
            if target_subset != donor_subset:
                raise RuntimeError("physical relation changed subset")
            source = r510.r509.parent.SOURCES[target_action]
            delta = direction["scale"] * subset_output(branch_sets[donor_action], donor_subset)
            replacement = captures[target_action]["deployed_write"].float() - delta
            logits, _captures, patch_diag, patch_audit = \
                r510.r509.parent.score_parent.run_forward(
                    model, tokens, action=source, scales=scales,
                    patch_writes={"m10": replacement.to(captures[target_action]["deployed_write"].dtype)})
            diagnostics["calls"]["analytical"] += 1
            diagnostics["substitution_patches"] += patch_audit["patches"]
            edit_rms = patch_diag["patch_rms_max"]
            diagnostics["zero_term_edits"] += int(edit_rms <= 0)
            if edit_rms > 0:
                diagnostics["minimum_nonzero_term_edit_rms"] = min(
                    diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
            nll_rows.append(r510.r509.parent._nll(logits, batch_rows).detach().cpu())
        nll_stack = torch.stack(nll_rows)
        task[:, local:local + r510.r509.parent.BATCH] = r510.r509.parent._task_sums(
            nll_stack, masks)
        counts[local:local + r510.r509.parent.BATCH] = torch.stack(
            [masks[cell].sum(1).double() for cell in r510.r509.parent.TASK_CELLS], -1)
        matrix, observed = r510.r509.parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        circuit_counts += observed
        circuit_sums += torch.matmul(
            nll_stack.view(len(directions), -1).double(), matrix.T,
        ).view(len(directions), 2, 2, len(circuit_tags))
    batches = documents // r510.r509.parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches,
        "analytical": batches * (1 + N_ACTIONS + len(directions)),
    }
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["hooks_expected"] = batches * (1 + N_ACTIONS)
    diagnostics["hooks_exact"] = diagnostics["hooks"] == diagnostics["hooks_expected"]
    diagnostics["four_corner_replays_expected"] = batches * N_ACTIONS
    diagnostics["four_corner_replays_exact"] = (
        diagnostics["four_corner_replays"] == diagnostics["four_corner_replays_expected"])
    diagnostics["substitution_patches_expected"] = batches * len(directions)
    diagnostics["substitution_patches_exact"] = (
        diagnostics["substitution_patches"] == diagnostics["substitution_patches_expected"])
    return {"bounds": bounds, "directions": directions, "task": task,
            "task_counts": counts, "circuit_tags": tuple(circuit_tags),
            "circuit_sums": circuit_sums, "circuit_counts": circuit_counts,
            "diagnostics": diagnostics}


def _substitution_instrument(collection: dict) -> bool:
    d = collection["diagnostics"]
    return bool(
        d["calls_exact"] and d["hooks_exact"] and d["four_corner_replays_exact"]
        and d["substitution_patches_exact"] and d["zero_term_edits"] == 0
        and d["factor_reconstruction_max"] <= 1e-10
        and d["raw_source_relative_squared"] <= r510.r509.parent.DEPLOYED_BF16_BAR
        and d["normalized_closure_relative_squared"] <= 1e-12
        and d["normalized_numerical_rms_ratio"] <= .02
        and d["float32_mlp10_closure"] <= 1e-10
        and d["deployed_mlp10_relative_squared"] <= r510.r509.parent.DEPLOYED_BF16_BAR
        and d["score_delta_deployed_closure_relative_squared"] <= 1e-12
        and d["minimum_nonzero_score_edit_rms"] > 0
        and d["minimum_nonzero_term_edit_rms"] > 0
        and d["native_replay_logit_max_abs"] == 0.0
        and d["native_replay_relative_squared"] <= 1e-12
        and d["absent_corner_replay_max_abs"] == 0.0
        and d["current_corner_replay_max_abs"] == 0.0
        and d["deployed_branch_sum_relative_squared"] <= 1e-12)


def _native_response(exact: dict, node: int, window: str) -> dict[str, torch.Tensor]:
    return {"task": _task_vector(exact, node, window),
            "circuit": _circuit_vector(exact, node, window)}


def _substituted_response(substitutions: dict, exact: dict, direction: int,
                          window: str) -> dict[str, torch.Tensor]:
    target_node = substitutions["directions"][direction]["target"]
    action, _subset = node_parts(target_node)
    if window == "pooled":
        lo, hi = 0, substitutions["task"].shape[1]
        circuit_sum = substitutions["circuit_sums"][direction].sum(0)
        intact_sum = exact["circuit_sums"][action, 0].sum(0)
        circuit_counts = substitutions["circuit_counts"].sum(0)
    else:
        half = {"half0": 0, "half1": 1}[window]
        bounds = substitutions["bounds"]
        absolute = ((bounds[0], bounds[2]), (bounds[2], bounds[1]))[half]
        lo, hi = absolute[0] - bounds[0], absolute[1] - bounds[0]
        circuit_sum = substitutions["circuit_sums"][direction, half]
        intact_sum = exact["circuit_sums"][action, 0, half]
        circuit_counts = substitutions["circuit_counts"][half]
    task_numerator = (substitutions["task"][direction, lo:hi]
                      - exact["task"][action, 0, lo:hi]).sum(0)
    task_denominator = substitutions["task_counts"][lo:hi].sum(0).clamp_min(1)
    full_task = task_numerator / task_denominator
    indices = [r510.r509.parent.TASK_CELLS.index(cell)
               for cell in r510.r509.parent.GRAD_CELLS[:4]]
    circuit = ((circuit_sum - intact_sum) / circuit_counts.clamp_min(1))[0] \
        - ((circuit_sum - intact_sum) / circuit_counts.clamp_min(1))[1]
    return {"task": full_task[indices], "circuit": circuit}


def score_substitutions(substitutions: dict, exact: dict,
                        candidates: list[dict]) -> tuple[list[dict], dict]:
    passing, checks = [], {}
    for edge_index, candidate in enumerate(candidates):
        row = {"directions": {}, "holds": True}
        for local_direction, side in enumerate(("left_from_right", "right_from_left")):
            direction = 2 * edge_index + local_direction
            target = substitutions["directions"][direction]["target"]
            side_row = {"windows": {}, "holds": True}
            for window in ("half0", "half1", "pooled"):
                native = _native_response(exact, target, window)
                observed = _substituted_response(substitutions, exact, direction, window)
                values = {}
                holds = True
                for kind in ("circuit", "task"):
                    cosine = float(r510._safe_cosine_rows(
                        native[kind][None], observed[kind][None])[0])
                    residual = float(r510._residual_rows(
                        native[kind][None], observed[kind][None])[0])
                    values[f"{kind}_cosine"] = cosine
                    values[f"{kind}_relative_residual"] = residual
                    holds &= cosine >= (.75 if kind == "circuit" else .70)
                    holds &= residual <= (.55 if kind == "circuit" else .65)
                values["holds"] = bool(holds)
                side_row["windows"][window] = values
                side_row["holds"] &= holds
            row["directions"][side] = side_row
            row["holds"] &= side_row["holds"]
        key = f"{candidate['left_name']} <-> {candidate['right_name']}"
        checks[key] = row
        if row["holds"]:
            passing.append(candidate)
    return passing, checks


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(R510_RESULT.read_text())
    if not (
        result.get("pred_a_exact_live_singleton_and_substitution_instrument") is True
        and result.get("pred_b_one_to_sixteen_discovery_equivalence_pairs") is False
        and result.get("analysis", {}).get("discovery_summary", {}).get("candidate_count") == 0
        and result.get("execution_price", {}).get("full_forwards") == 63116
        and result.get("next_step")
        == "registered_multi_term_signed_combinations_without_pair_ranking"
    ):
        raise RuntimeError("rung510 zero-pair route changed")
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        r510.validate_inputs()
    if len(discovery_tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("62-circuit partition changed")
    return rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, {
        **metadata, "rung510_result_sha256": sha256(R510_RESULT),
        "rung510_bundle_sha256": sha256(R510_BUNDLE),
        "branches": list(BRANCH_NAMES), "subsets": list(SUBSET_NAMES),
        "documents": {"discovery": list(DISCOVERY), "confirmation": list(CONFIRMATION)},
        "circuits": {"discovery": list(discovery_tags), "confirmation": list(validation_tags)},
    }


def dry_run() -> None:
    generator = torch.Generator(device="cpu").manual_seed(511)
    circuit = .002 * torch.randn(N_NODES, 32, generator=generator, dtype=torch.float64)
    task = .002 * torch.randn(N_NODES, 4, generator=generator, dtype=torch.float64)
    # Same subset, actions N/P: an intentionally planted signed relation.
    circuit[N_SUBSETS] = -2 * circuit[0]
    task[N_SUBSETS] = -2 * task[0]
    matrices = {window: {"circuit": circuit.clone(), "task": task.clone()}
                for window in ("half0", "half1", "pooled")}
    candidates, summary = discover_relations(matrices)
    assert summary["relations_tested"] == 42
    assert any(row["left_node"] == 0 and row["right_node"] == N_SUBSETS
               for row in candidates)
    assert 2 * 2108 + 372 + 124 * 42 == 9796
    fake = {"arms": ARMS}
    values = {mask: torch.tensor([float(mask)]) for mask in SUBSET_MASKS}
    assert sum(((-1) ** (3 - sub.bit_count())) * values[sub]
               for sub in SUBSET_MASKS)[0].item() == 0.0
    print(json.dumps({
        "status": "dry_run_passed", "rung": 511, "model_loaded": False,
        "outcomes_opened": False, "branches": list(BRANCH_NAMES),
        "subsets": list(SUBSET_NAMES), "relations_tested": 42,
        "maximum_conditional_forwards": 9796,
    }, indent=2, sort_keys=True))


def _bundle_collection(collection: dict) -> dict:
    return {key: value for key, value in collection.items() if key != "diagnostics"}


def _gpu_smoke() -> None:
    rows, task_masks, circuit_masks, scales, discovery_tags, _validation_tags, _metadata = \
        validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    bounds = (500, 504, 502)
    factorial = collect_factorial(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, bounds)
    candidate = {
        "left_node": 0, "right_node": N_SUBSETS,
        "left_name": NODE_NAMES[0], "right_name": NODE_NAMES[N_SUBSETS],
        "subset_index": 0, "subset_name": SUBSET_NAMES[0],
        "beta_left_from_right": 1.0,
    }
    substitutions = collect_substitutions(
        model, rows, task_masks, circuit_masks, discovery_tags,
        scales, bounds, [candidate])
    checks = {
        "weights": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256,
        "factorial": _instrument(factorial),
        "substitutions": _substitution_instrument(substitutions),
        "all_twenty_eight_subset_patches": factorial["diagnostics"]["subset_patches"] == 28,
        "both_substitution_patches": substitutions["diagnostics"]["substitution_patches"] == 2,
    }
    passed = all(checks.values())
    print(json.dumps({
        "status": "smoke_passed" if passed else "smoke_failed", "rung": 511,
        "scientific_outcomes_retained": False, "checks": checks,
        "factorial_diagnostics": factorial["diagnostics"],
        "substitution_diagnostics": substitutions["diagnostics"],
        "full_forwards": sum(factorial["diagnostics"]["calls"].values())
        + sum(substitutions["diagnostics"]["calls"].values()),
        "backwards": 0,
    }, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(
            f"rung511 CUDA smoke failed: "
            f"{sorted(name for name, value in checks.items() if not value)}")


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        dry_run()
        return
    if os.environ.get("BQLIB_GPU_SMOKE") == "1" or "--gpu-smoke" in sys.argv:
        _gpu_smoke()
        return
    started = time.time()
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung511 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    collections = {}
    collections["factorial_discovery"] = collect_factorial(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, DISCOVERY)
    discovery_calibration = r510.r509.parent._calibration(
        collections["factorial_discovery"]["base_task"],
        collections["factorial_discovery"]["source_task"],
        collections["factorial_discovery"]["task_counts"], DISCOVERY)
    discovery_calibration_ok = r510.r509.parent.state_parent.calibration_holds(
        discovery_calibration)
    discovery_matrices = response_matrices(collections["factorial_discovery"])
    candidates, discovery_summary = discover_relations(discovery_matrices)
    control_counts = permutation_control_counts(discovery_matrices)

    confirmed, confirmation_checks = [], {}
    physical_pairs, physical_checks = [], {}
    composition_rules, composition_checks, selective_checks = {}, {}, {}
    confirmation_calibration, confirmation_calibration_ok = {}, False
    if discovery_calibration_ok and candidates:
        collections["factorial_confirmation"] = collect_factorial(
            model, rows, task_masks, circuit_masks, validation_tags, scales, CONFIRMATION)
        confirmation_calibration = r510.r509.parent._calibration(
            collections["factorial_confirmation"]["base_task"],
            collections["factorial_confirmation"]["source_task"],
            collections["factorial_confirmation"]["task_counts"], CONFIRMATION)
        confirmation_calibration_ok = r510.r509.parent.state_parent.calibration_holds(
            confirmation_calibration)
        confirmation_matrices = response_matrices(collections["factorial_confirmation"])
        confirmed, confirmation_checks = confirm_relations(
            confirmation_matrices, candidates)
        for candidate in confirmed:
            key = f"{candidate['left_name']} <-> {candidate['right_name']}"
            composition_rules[key] = _composition_rule(
                collections["factorial_discovery"], candidate)
            composition_checks[key] = _composition_confirmation(
                collections["factorial_confirmation"], candidate, composition_rules[key])
            selective_checks[key] = _selective(
                collections["factorial_confirmation"], candidate)
    if confirmation_calibration_ok and confirmed:
        collections["substitutions"] = collect_substitutions(
            model, rows, task_masks, circuit_masks, validation_tags,
            scales, CONFIRMATION, confirmed)
        physical_pairs, physical_checks = score_substitutions(
            collections["substitutions"], collections["factorial_confirmation"], confirmed)

    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and discovery_calibration_ok
        and _instrument(collections["factorial_discovery"])
        and ("factorial_confirmation" not in collections
             or (confirmation_calibration_ok and _instrument(collections["factorial_confirmation"])))
        and ("substitutions" not in collections
             or _substitution_instrument(collections["substitutions"])))
    pred_b = bool(pred_a and candidates)
    pred_c = bool(pred_b and confirmation_calibration_ok and confirmed)
    pred_d = bool(pred_c and physical_pairs)
    physical_keys = {
        f"{row['left_name']} <-> {row['right_name']}" for row in physical_pairs}
    pred_e_pairs = [row for row in physical_pairs
                    if SUBSET_MASKS[row["subset_index"]].bit_count() == 2
                    and composition_checks[
                        f"{row['left_name']} <-> {row['right_name']}"]["holds"]
                    and selective_checks[
                        f"{row['left_name']} <-> {row['right_name']}"]["holds"]]
    pred_e = bool(pred_d and pred_e_pairs)
    strong_null = not (pred_a and pred_b and pred_c and pred_d and pred_e)
    if not pred_a:
        next_step = "repair_three_branch_factorial_instrument_only"
    elif not pred_b:
        next_step = "localize_exact_branches_at_first_downstream_consumer_including_mlp11_question_interface"
    elif not pred_c:
        next_step = "consumer_specific_nonlinear_branch_readout_test"
    elif not pred_d:
        next_step = "localize_first_consumer_that_separates_response_similar_branch_combinations"
    elif not pred_e:
        next_step = "retain_singleton_portability_without_distributed_mlp_split_claim"
    else:
        next_step = "validate_branch_program_on_ood_code_then_resolve_stable_branches_by_earlier_writes"

    bundle_payload = {
        "schema": "rung511_three_branch_factorial_v1",
        "collections": {name: _bundle_collection(collection)
                        for name, collection in collections.items()},
        "discovery_candidates": candidates, "confirmed_pairs": confirmed,
        "physical_pairs": physical_pairs,
        "raw_tokens_logits_hidden_states_or_weights_included": False,
    }
    torch.save(bundle_payload, BUNDLE)
    result = {
        "status": "complete", "rung": 511,
        "claim_level": "exact_three_branch_factorial_until_heldout_physical_portability_passes",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "branches": list(BRANCH_NAMES), "subsets": list(SUBSET_NAMES),
        "calibration": {"discovery": discovery_calibration,
                        "confirmation": confirmation_calibration},
        "calibration_holds": {"discovery": discovery_calibration_ok,
                              "confirmation": confirmation_calibration_ok},
        "diagnostics": {name: collection["diagnostics"]
                        for name, collection in collections.items()},
        "analysis": {
            "discovery_summary": discovery_summary,
            "discovery_candidates": candidates,
            "permutation_control_candidate_counts": control_counts,
            "confirmation_checks": confirmation_checks,
            "confirmed_pairs": confirmed,
            "composition_rules": composition_rules,
            "composition_checks": composition_checks,
            "selective_checks": selective_checks,
            "physical_checks": physical_checks,
            "physical_pairs": physical_pairs,
            "distributed_pairs": pred_e_pairs,
        },
        'pred_a_exact_live_three_branch_factorial_instrument': pred_a,
        'pred_b_fixed_same_subset_cross_action_discovery': pred_b,
        'pred_c_heldout_documents_and_circuit_families': pred_c,
        'pred_d_bidirectional_physical_branch_substitution': pred_d,
        'pred_e_two_branch_predictable_selective_computation': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "full_forwards": sum(sum(collection["diagnostics"]["calls"].values())
                                 for collection in collections.values()),
            "backwards": 0, "fixed_relations_tested": 42,
            "discovery_candidates": len(candidates), "confirmed_pairs": len(confirmed),
            "physical_pairs": len(physical_pairs), "maximum_conditional_forwards": 9796,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started, "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 511,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "discovery_candidates": len(candidates), "confirmed_pairs": len(confirmed),
        "physical_pairs": len(physical_pairs), "distributed_pairs": len(pred_e_pairs),
        "execution_price": result["execution_price"], "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
