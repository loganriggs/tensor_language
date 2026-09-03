#!/usr/bin/env python3
"""RUNG518 -- circuit-defined quotient of MLP0 head-by-source pieces.

pred_a: the exact/live 45-piece intervention and planted-recovery instrument passes
pred_b: 1--16 pairs pass two-half task+circuit discovery and permutation control
pred_c: a frozen pair predicts the other 30 circuit families and documents
pred_d: a confirmed pair passes bidirectional physical replacement
pred_e: a physical component crosses heads while splitting one native head

BQGATE: EXPERIMENT
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
PREREG = POLY / "MLP0_HEAD_RELATION_CIRCUIT_QUOTIENT_RUNG518_PREREGISTRATION.md"
PREREG_SHA256 = "54ee23d84dcb515917b563690aef1c6c8e0a53909cabda59088825404ad7e382"
R517_RESULT = ROOT / "basis_aligned/bilinear_quotient/mlp0_source_relation_factorial_rung517_results.json"
R517_SOURCE = ROOT / "basis_aligned/bilinear_quotient/ops/mlp0_source_relation_factorial_rung517.py"
R517_PREREG = POLY / "MLP0_SOURCE_RELATION_FACTORIAL_RUNG517_PREREGISTRATION.md"
R510_SOURCE = ROOT / "basis_aligned/bilinear_quotient/ops/mlp10_observable_predictive_state_quotient_rung510.py"
OUT = ROOT / "basis_aligned/bilinear_quotient/mlp0_head_relation_circuit_quotient_rung518_results.json"
BUNDLE = ROOT / "basis_aligned/bilinear_quotient/mlp0_head_relation_circuit_quotient_rung518_bundle.pt"
HASHES = {
    R517_RESULT: "c8405a36cab0e8b50d91e3f525bf5a5106a95d2c42447ce9b83ab29378fd8307",
    R517_SOURCE: "5d9acfa5798e9d391e6507d5d7136ec498e4f1b42893a372c47c07c7be6bae97",
    R517_PREREG: "a0ff4160af15b57c549c3998e24010d7f60f14d34b2de811fd5f5a5824bde56c",
    R510_SOURCE: "7901aa5d9c7c39bf5666e0f081bfe08047f23c73eec08b12508c601def7b967a",
}
GROUPS = ("SELF", "PREVIOUS", "NEAR", "DISTANT_SAME", "DISTANT_OTHER")
N_HEADS = 9
N_ATOMS = N_HEADS * len(GROUPS)
ATOM_NAMES = tuple(f"H{head}.{group}" for head in range(N_HEADS) for group in GROUPS)
PLANTED_PAIRS = ((0, 7), (3, 14), (8, 31), (20, 44))
PLANTED_SEEDS = tuple(range(51800, 51808))
CONTROL_SEEDS = tuple(range(518100, 518116))
HEAD_DIM = 128
D = 1152
BATCH = 4
TOKENS = 256
DISCOVERY = (500, 748, 624)
CONFIRMATION = (752, 1000, 876)
ARMS = ("EMPTY", "FULL") + tuple(f"SINGLE::{name}" for name in ATOM_NAMES) \
    + tuple(f"DROP::{name}" for name in ATOM_NAMES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atom_index(head: int, group: int) -> int:
    if not 0 <= head < N_HEADS or not 0 <= group < len(GROUPS):
        raise ValueError("head or relation index is outside the frozen vocabulary")
    return head * len(GROUPS) + group


def atom_parts(atom: int) -> tuple[int, int]:
    if not 0 <= atom < N_ATOMS:
        raise ValueError("atom index is outside the frozen vocabulary")
    return divmod(atom, len(GROUPS))


@torch.no_grad()
def head_relation_atoms(block, split: dict) -> dict:
    """Construct the fixed 9x5 source pieces from a rung517 attention split."""
    pattern = split["pattern"]
    value = split["value"]
    masks = split["partition_masks"]
    native = split["native_write"]
    if pattern.ndim != 4 or pattern.shape[1] != N_HEADS:
        raise ValueError("attention pattern is not [batch,9,query,source]")
    if value.shape[2:] != (N_HEADS, HEAD_DIM):
        raise ValueError("attention values do not use the frozen 9x128 head shape")
    if masks.shape[0] != len(GROUPS):
        raise ValueError("source relation vocabulary changed")
    projection = block.attn.c_proj.weight
    if projection.shape != (D, D):
        raise ValueError("attention0 output projection shape changed")
    atoms = []
    for head in range(N_HEADS):
        weight = projection[:, head * HEAD_DIM:(head + 1) * HEAD_DIM]
        for group in range(len(GROUPS)):
            selected = pattern[:, head] * masks[group].to(pattern.dtype)
            head_output = torch.einsum("bqk,bkd->bqd", selected, value[:, :, head])
            atoms.append(F.linear(head_output, weight).float())
    atoms = torch.stack(atoms)
    semantic_sum = atoms.sum(0)
    remainder = native.float() - semantic_sum
    denominator = native.double().square().sum().clamp_min(1e-30)
    return {
        "atoms": atoms,
        "remainder": remainder,
        "relative_squared_closure": float(
            (semantic_sum.double() + remainder.double() - native.double())
            .square().sum() / denominator),
        "remainder_relative_energy": float(
            remainder.double().square().sum() / denominator),
    }


def atom_context(native: torch.Tensor, decomposition: dict, atom: int,
                 background: str) -> torch.Tensor:
    if not 0 <= atom < N_ATOMS:
        raise ValueError("atom index is outside the frozen vocabulary")
    if background == "SINGLE":
        context = decomposition["remainder"] + decomposition["atoms"][atom]
    elif background == "DROP":
        context = native.float() - decomposition["atoms"][atom]
    else:
        raise ValueError("background must be SINGLE or DROP")
    return context.to(native.dtype)


def _half_bounds(bounds: tuple[int, int, int], half: str) -> tuple[int, int]:
    lo, hi, split = bounds
    absolute = (lo, split) if half == "half0" else (split, hi)
    return absolute[0] - lo, absolute[1] - lo


def half_supports_positive(counts: torch.Tensor,
                           bounds: tuple[int, int, int]) -> bool:
    """Require every coordinate to have pooled support in each registered half."""
    if counts.ndim != 2 or counts.shape[0] != bounds[1] - bounds[0]:
        raise ValueError("support table does not match the registered document bounds")
    return all(bool((counts[slice(*_half_bounds(bounds, half))].sum(0) > 0).all())
               for half in ("half0", "half1"))


def half_support_totals(counts: torch.Tensor,
                        bounds: tuple[int, int, int]) -> dict[str, list[int]]:
    """Report the pooled coordinate supports used by the validity gate."""
    return {
        half: [int(value) for value in
               counts[slice(*_half_bounds(bounds, half))].sum(0).tolist()]
        for half in ("half0", "half1")
    }


def response_matrices(collection: dict, task_indices: tuple[int, ...]) -> dict:
    """Convert accumulated NLL sums into [atom,background,coordinate] benefits."""
    if tuple(collection["arms"]) != ARMS:
        raise ValueError("rung518 arm order changed")
    outputs = {}
    for half in ("half0", "half1"):
        lo, hi = _half_bounds(tuple(collection["bounds"]), half)
        task_den = collection["task_counts"][lo:hi].sum(0).clamp_min(1)
        task_mean = collection["task_sums"][:, lo:hi].sum(1) / task_den
        circuit_mean = collection["circuit_sums"][:, int(half == "half1")] \
            / collection["circuit_counts"][int(half == "half1")].clamp_min(1)
        task = torch.zeros(N_ATOMS, 2, len(task_indices), dtype=torch.float64)
        circuit = torch.zeros(
            N_ATOMS, 2, circuit_mean.shape[-1], dtype=torch.float64)
        for atom in range(N_ATOMS):
            single, drop = 2 + atom, 2 + N_ATOMS + atom
            task[atom, 0] = (task_mean[0] - task_mean[single])[list(task_indices)]
            task[atom, 1] = (task_mean[drop] - task_mean[1])[list(task_indices)]
            single_effect = circuit_mean[0] - circuit_mean[single]
            drop_effect = circuit_mean[drop] - circuit_mean[1]
            circuit[atom, 0] = single_effect[0] - single_effect[1]
            circuit[atom, 1] = drop_effect[0] - drop_effect[1]
        outputs[half] = {"task": task, "circuit": circuit}
    return outputs


@torch.no_grad()
def collect_phase(model, rows, task_masks, circuit_masks, circuit_tags, bounds,
                  facade, r517, response_parent) -> dict:
    """Collect all 92 finite atom arms plus one independent native replay."""
    lo, hi, _split = bounds
    documents = hi - lo
    task_cells = response_parent.TASK_CELLS
    task_sums = torch.zeros(len(ARMS), documents, len(task_cells), dtype=torch.float64)
    task_counts = torch.zeros(documents, len(task_cells), dtype=torch.float64)
    circuit_sums = torch.zeros(len(ARMS), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    minimum_single = float("inf")
    minimum_drop = float("inf")
    max_atom_closure = 0.0
    max_native_replay = 0.0
    calls = 0
    device = next(model.parameters()).device
    block0 = model.transformer.h[0]
    for start in range(lo, hi, BATCH):
        stop = start + BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        state = F.rms_norm(token_base, (D,))
        split = r517.attention0_source_writes(block0, state, tokens)
        decomposition = head_relation_atoms(block0, split)
        max_atom_closure = max(max_atom_closure, decomposition["relative_squared_closure"])
        contexts = [decomposition["remainder"].to(split["native_write"].dtype),
                    split["native_write"]]
        contexts.extend(atom_context(split["native_write"], decomposition, atom, "SINGLE")
                        for atom in range(N_ATOMS))
        contexts.extend(atom_context(split["native_write"], decomposition, atom, "DROP")
                        for atom in range(N_ATOMS))
        site0_writes = [block0.mlp(F.rms_norm(token_base + context, (D,)))
                        for context in contexts]
        empty_write, full_write = site0_writes[:2]
        for atom in range(N_ATOMS):
            minimum_single = min(
                minimum_single,
                float((site0_writes[2 + atom].float() - empty_write.float())
                      .square().mean().sqrt()))
            minimum_drop = min(
                minimum_drop,
                float((site0_writes[2 + N_ATOMS + atom].float() - full_write.float())
                      .square().mean().sqrt()))
        nll_rows = []
        full_logits = None
        for arm_index, site0_write in enumerate(site0_writes):
            def attention_dispatch(event):
                if event.site == 0:
                    return split["native_write"], split["first_value"]
                return event.block.attn(event.state, event.first_value)

            def mlp_dispatch(event, site0_write=site0_write):
                return site0_write if event.site == 0 else event.block.mlp(event.state)

            logits = facade.forward_with_dispatch(
                model, tokens, attention_dispatch, mlp_dispatch)
            if arm_index == 1:
                full_logits = logits
            nll_rows.append(response_parent._nll(logits, batch_rows).cpu())
            calls += 1
        native_logits = facade.forward_with_dispatch(
            model, tokens,
            lambda event: event.block.attn(event.state, event.first_value),
            lambda event: event.block.mlp(event.state))
        calls += 1
        if full_logits is None:
            raise RuntimeError("FULL arm was not evaluated")
        max_native_replay = max(
            max_native_replay, float((native_logits - full_logits).abs().max()))
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
        "bounds": tuple(bounds), "arms": ARMS, "task_sums": task_sums,
        "task_counts": task_counts, "circuit_sums": circuit_sums,
        "circuit_counts": circuit_counts,
        "diagnostics": {
            "calls": calls, "calls_expected": ((hi - lo) // BATCH) * 93,
            "maximum_atom_closure": max_atom_closure,
            "maximum_native_replay_error": max_native_replay,
            "minimum_single_edit_rms": minimum_single,
            "minimum_drop_edit_rms": minimum_drop,
        },
    }


def validate_inputs():
    if sha256(PREREG) != PREREG_SHA256:
        raise RuntimeError("rung518 preregistration changed after source freeze")
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen dependency hash mismatch: {path}")
    parent = json.loads(R517_RESULT.read_text())
    if not (
        parent.get("pred_a_exact_live_instrument") is True
        and parent.get("pred_b_prose_localization") is False
        and parent.get("pred_c_structured_text_widening") is False
        and parent.get("pred_d_split_stable_source_roles") is True
        and parent.get("pred_e_downstream_specificity_screen") is False
        and parent.get("strong_null") is True
        and parent.get("next_step") == "retain_diagnostic_only_and_choose_new_program_gap"
    ):
        raise RuntimeError("rung517 route changed")
    sys.path.insert(0, str(ROOT / "basis_aligned/bilinear_quotient/ops"))
    sys.path.insert(0, str(ROOT / "basis_aligned/bilinear_quotient"))
    sys.path.insert(0, str(POLY))
    import mlp10_observable_predictive_state_quotient_rung510 as r510

    rows, task_masks, circuit_masks, scales, discovery_tags, confirmation_tags, metadata = \
        r510.validate_inputs()
    if rows.shape != (1000, 257):
        raise RuntimeError(f"rung510 row shape changed: {tuple(rows.shape)}")
    if len(discovery_tags) != 32 or len(confirmation_tags) != 30:
        raise RuntimeError("the frozen 32/30 circuit partition changed")
    return rows, task_masks, circuit_masks, scales, discovery_tags, confirmation_tags, {
        **metadata,
        "rung517_result_sha256": sha256(R517_RESULT),
        "rung517_source_sha256": sha256(R517_SOURCE),
        "rung517_preregistration_sha256": sha256(R517_PREREG),
        "rung510_source_sha256": sha256(R510_SOURCE),
    }, r510.r509.parent


@torch.no_grad()
def gpu_smoke() -> None:
    """Exercise one full 45-piece batch while retaining no scientific response."""
    sys.path.insert(0, str(ROOT / "basis_aligned/bilinear_quotient/ops"))
    sys.path.insert(0, str(POLY))
    import bilin18_observed_model_facade as facade
    import mlp0_source_relation_factorial_rung517 as r517

    rows, task_masks, circuit_masks, _scales, discovery_tags, _confirmation_tags, \
        _metadata, response_parent = validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    collection = collect_phase(
        model, rows, task_masks, circuit_masks, discovery_tags[:2],
        (500, 504, 502), facade, r517, response_parent)
    task_indices = tuple(response_parent.TASK_CELLS.index(cell)
                         for cell in response_parent.GRAD_CELLS[:4])
    matrices = response_matrices(collection, task_indices)
    diagnostics = collection["diagnostics"]
    checks = {
        "weights": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256,
        "45_piece_closure": diagnostics["maximum_atom_closure"] <= 1e-8,
        "native_replay": diagnostics["maximum_native_replay_error"] == 0,
        "all_single_edits_live": diagnostics["minimum_single_edit_rms"] > 0,
        "all_drop_edits_live": diagnostics["minimum_drop_edit_rms"] > 0,
        "call_census": diagnostics["calls"] == diagnostics["calls_expected"] == 93,
        "response_shapes": all(
            matrices[half]["circuit"].shape == (45, 2, 2)
            and matrices[half]["task"].shape == (45, 2, 4)
            for half in ("half0", "half1")),
        "planted_recovery": planted_suite()["all_eight_exact"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"rung518 managed smoke failed: {checks}")
    print(json.dumps({
        "status": "gpu_smoke_passed", "rung": 518,
        "scientific_outcomes_retained": False,
        "checkpoint": checkpoint.__dict__, "checks": checks,
        "diagnostics": diagnostics,
    }, indent=2, sort_keys=True))


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    return float((left @ right) / denominator.clamp_min(1e-30))


def _relative_residual(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    return float((actual - predicted).norm() / actual.norm().clamp_min(1e-30))


def _score_pair(responses: dict, left: int, right: int, beta: float,
                circuit_cosine: float, circuit_residual: float) -> dict:
    safe_beta = beta if abs(beta) > 1e-30 else 1.0
    row = {"beta_left_from_right": beta, "halves": {}}
    material = True
    holds = .25 <= abs(beta) <= 4
    for half in ("half0", "half1"):
        row["halves"][half] = {}
        for background in range(2):
            entry = {}
            for kind in ("circuit", "task"):
                lvec = responses[half][kind][left, background].double()
                rvec = responses[half][kind][right, background].double()
                predicted_left = beta * rvec
                predicted_right = lvec / safe_beta
                signed_cosine = _cosine(lvec, rvec) * (1 if beta >= 0 else -1)
                entry[kind] = {
                    "signed_cosine": signed_cosine,
                    "left_from_right_relative_residual": _relative_residual(
                        lvec, predicted_left),
                    "right_from_left_relative_residual": _relative_residual(
                        rvec, predicted_right),
                }
                if kind == "circuit":
                    material &= min(float(lvec.square().mean().sqrt()),
                                    float(rvec.square().mean().sqrt())) >= .0005
                    holds &= (signed_cosine >= circuit_cosine
                              and max(entry[kind]["left_from_right_relative_residual"],
                                      entry[kind]["right_from_left_relative_residual"])
                              <= circuit_residual)
                else:
                    material &= min(float(lvec.norm()), float(rvec.norm())) >= .00025
                    holds &= (signed_cosine >= .70
                              and max(entry[kind]["left_from_right_relative_residual"],
                                      entry[kind]["right_from_left_relative_residual"]) <= .65)
            row["halves"][half][str(background)] = entry
    row["material"] = bool(material)
    row["holds"] = bool(material and holds)
    return row


def pair_metrics(responses: dict, left: int, right: int) -> dict:
    """Fit half0 circuit scale and score both backgrounds without pooling them away."""
    fit_left = responses["half0"]["circuit"][left].reshape(-1).double()
    fit_right = responses["half0"]["circuit"][right].reshape(-1).double()
    beta = float((fit_left @ fit_right) / fit_right.square().sum().clamp_min(1e-30))
    return _score_pair(responses, left, right, beta, .85, .50)


def discover_pairs(responses: dict) -> list[dict]:
    candidates = []
    for left in range(N_ATOMS):
        for right in range(left + 1, N_ATOMS):
            metrics = pair_metrics(responses, left, right)
            if metrics["holds"]:
                candidates.append({
                    "left": left, "right": right,
                    "left_name": ATOM_NAMES[left], "right_name": ATOM_NAMES[right],
                    **metrics,
                })
    return candidates


def permutation_control_counts(responses: dict) -> list[int]:
    counts = []
    dimensions = responses["half0"]["circuit"].shape[-1]
    for seed in CONTROL_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        keys = torch.rand(N_ATOMS, dimensions, generator=generator)
        order = keys.argsort(-1)[:, None].expand(-1, 2, -1)
        control = {
            half: {
                "task": responses[half]["task"],
                "circuit": torch.gather(responses[half]["circuit"], -1, order),
            }
            for half in ("half0", "half1")
        }
        counts.append(len(discover_pairs(control)))
    return counts


def confirmation_pairs(responses: dict, candidates: list[dict]) -> tuple[list[dict], dict]:
    passing, checks = [], {}
    for candidate in candidates:
        metrics = _score_pair(
            responses, candidate["left"], candidate["right"],
            candidate["beta_left_from_right"], .75, .55)
        key = f"{candidate['left_name']} <-> {candidate['right_name']}"
        checks[key] = metrics
        if metrics["holds"]:
            passing.append(candidate)
    return passing, checks


@torch.no_grad()
def collect_substitutions(model, rows, task_masks, circuit_masks, circuit_tags, bounds,
                          candidates, facade, r517, response_parent) -> dict:
    directions = []
    for pair_index, candidate in enumerate(candidates):
        beta = candidate["beta_left_from_right"]
        directions.extend((
            {"pair": pair_index, "target": candidate["left"],
             "donor": candidate["right"], "scale": beta,
             "name": f"{candidate['left_name']} <- {beta:.9g}*{candidate['right_name']}"},
            {"pair": pair_index, "target": candidate["right"],
             "donor": candidate["left"], "scale": 1 / beta,
             "name": f"{candidate['right_name']} <- {1 / beta:.9g}*{candidate['left_name']}"},
        ))
    lo, hi, _split = bounds
    documents = hi - lo
    task_cells = response_parent.TASK_CELLS
    task_sums = torch.zeros(len(directions), documents, len(task_cells), dtype=torch.float64)
    task_counts = torch.zeros(documents, len(task_cells), dtype=torch.float64)
    circuit_sums = torch.zeros(len(directions), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    minimum_context_edit = float("inf")
    minimum_mlp0_edit = float("inf")
    calls = 0
    device = next(model.parameters()).device
    block0 = model.transformer.h[0]
    for start in range(lo, hi, BATCH):
        stop = start + BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        state = F.rms_norm(token_base, (D,))
        split = r517.attention0_source_writes(block0, state, tokens)
        decomposition = head_relation_atoms(block0, split)
        full_write = block0.mlp(F.rms_norm(token_base + split["native_write"], (D,)))
        nll_rows = []
        for direction in directions:
            target = decomposition["atoms"][direction["target"]]
            donor = decomposition["atoms"][direction["donor"]]
            context = (split["native_write"].float() - target
                       + direction["scale"] * donor).to(split["native_write"].dtype)
            site0_write = block0.mlp(F.rms_norm(token_base + context, (D,)))
            minimum_context_edit = min(
                minimum_context_edit,
                float((context.float() - split["native_write"].float())
                      .square().mean().sqrt()))
            minimum_mlp0_edit = min(
                minimum_mlp0_edit,
                float((site0_write.float() - full_write.float()).square().mean().sqrt()))

            def attention_dispatch(event):
                if event.site == 0:
                    return split["native_write"], split["first_value"]
                return event.block.attn(event.state, event.first_value)

            def mlp_dispatch(event, site0_write=site0_write):
                return site0_write if event.site == 0 else event.block.mlp(event.state)

            logits = facade.forward_with_dispatch(
                model, tokens, attention_dispatch, mlp_dispatch)
            nll_rows.append(response_parent._nll(logits, batch_rows).cpu())
            calls += 1
        nll = torch.stack(nll_rows)
        local_masks = {cell: task_masks[cell][start:stop] for cell in task_cells}
        task_sums[:, local:local + BATCH] = response_parent._task_sums(nll, local_masks)
        task_counts[local:local + BATCH] = torch.stack(
            [local_masks[cell].sum(1).double() for cell in task_cells], -1)
        matrix, observed = response_parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        circuit_counts += observed
        circuit_sums += torch.matmul(nll.reshape(len(directions), -1).double(), matrix.T).view(
            len(directions), 2, 2, len(circuit_tags))
    return {
        "bounds": tuple(bounds), "directions": directions,
        "task_sums": task_sums, "task_counts": task_counts,
        "circuit_sums": circuit_sums, "circuit_counts": circuit_counts,
        "diagnostics": {
            "calls": calls,
            "calls_expected": ((hi - lo) // BATCH) * len(directions),
            "minimum_context_edit_rms": minimum_context_edit,
            "minimum_mlp0_edit_rms": minimum_mlp0_edit,
        },
    }


def score_substitutions(substitutions: dict, confirmation: dict,
                        task_indices: tuple[int, ...],
                        off_target_index: int) -> tuple[list[dict], dict]:
    passing_pairs = []
    checks = {}
    direction_pass = []
    for direction_index, direction in enumerate(substitutions["directions"]):
        target = direction["target"]
        row = {"name": direction["name"], "halves": {}}
        holds = True
        for half in ("half0", "half1"):
            lo, hi = _half_bounds(tuple(confirmation["bounds"]), half)
            half_index = int(half == "half1")
            task_den = confirmation["task_counts"][lo:hi].sum(0).clamp_min(1)
            full_task = confirmation["task_sums"][1, lo:hi].sum(0) / task_den
            drop_index = 2 + N_ATOMS + target
            drop_task = confirmation["task_sums"][drop_index, lo:hi].sum(0) / task_den
            sub_task = substitutions["task_sums"][direction_index, lo:hi].sum(0) / task_den
            circuit_den = confirmation["circuit_counts"][half_index].clamp_min(1)
            full_circuit = confirmation["circuit_sums"][1, half_index] / circuit_den
            drop_circuit = confirmation["circuit_sums"][drop_index, half_index] / circuit_den
            sub_circuit = substitutions["circuit_sums"][direction_index, half_index] / circuit_den
            target_task = (drop_task - full_task)[list(task_indices)]
            recovered_task = (drop_task - sub_task)[list(task_indices)]
            target_circuit_raw = drop_circuit - full_circuit
            recovered_circuit_raw = drop_circuit - sub_circuit
            target_circuit = target_circuit_raw[0] - target_circuit_raw[1]
            recovered_circuit = recovered_circuit_raw[0] - recovered_circuit_raw[1]
            target_vector = torch.cat((target_task, target_circuit)).double()
            recovered_vector = torch.cat((recovered_task, recovered_circuit)).double()
            projection = float(
                (recovered_vector @ target_vector)
                / target_vector.square().sum().clamp_min(1e-30))
            cosine = _cosine(recovered_vector, target_vector)
            off_target_damage = float(sub_task[off_target_index] - full_task[off_target_index])
            half_holds = bool(
                projection >= .70 and cosine >= .80 and abs(off_target_damage) <= .002)
            row["halves"][half] = {
                "recovery_projection": projection, "cosine": cosine,
                "off_target_delta_ce_nat": off_target_damage, "holds": half_holds,
            }
            holds &= half_holds
        row["holds"] = bool(holds)
        checks[direction["name"]] = row
        direction_pass.append(bool(holds))
    for pair_index in range(len(substitutions["directions"]) // 2):
        if direction_pass[2 * pair_index] and direction_pass[2 * pair_index + 1]:
            passing_pairs.append({"pair": pair_index, "directions": [
                substitutions["directions"][2 * pair_index]["name"],
                substitutions["directions"][2 * pair_index + 1]["name"],
            ]})
    return passing_pairs, checks


def native_boundary_evidence(discovery: dict, candidates: list[dict],
                             physical_pairs: list[dict]) -> list[dict]:
    evidence = []
    for physical in physical_pairs:
        candidate = candidates[physical["pair"]]
        left, right = candidate["left"], candidate["right"]
        left_head, _ = atom_parts(left)
        right_head, _ = atom_parts(right)
        if left_head == right_head:
            continue
        witnesses = []
        for component_atom in (left, right):
            head, _group = atom_parts(component_atom)
            for other in range(head * len(GROUPS), (head + 1) * len(GROUPS)):
                if other == component_atom:
                    continue
                material = all(
                    float(discovery[half]["circuit"][atom, background]
                          .double().square().mean().sqrt()) >= .0005
                    for half in ("half0", "half1")
                    for background in range(2)
                    for atom in (component_atom, other))
                task_material = all(
                    float(discovery[half]["task"][atom, background].double().norm()) >= .00025
                    for half in ("half0", "half1")
                    for background in range(2)
                    for atom in (component_atom, other))
                if not material or not task_material:
                    continue
                half_checks = []
                for half in ("half0", "half1"):
                    background_checks = []
                    for background in range(2):
                        component = discovery[half]["circuit"][component_atom, background]
                        alternative = discovery[half]["circuit"][other, background]
                        background_checks.append(bool(
                            abs(_cosine(component.double(), alternative.double())) <= .50
                            and int(component.abs().argmax()) != int(alternative.abs().argmax())))
                    half_checks.append(any(background_checks))
                if all(half_checks):
                    witnesses.append({
                        "component_atom": ATOM_NAMES[component_atom],
                        "same_head_split_atom": ATOM_NAMES[other],
                    })
        if witnesses:
            evidence.append({
                "pair": f"{candidate['left_name']} <-> {candidate['right_name']}",
                "crosses_heads": True, "split_witnesses": witnesses,
            })
    return evidence


def planted_problem(seed: int) -> tuple[dict, set[tuple[int, int]]]:
    generator = torch.Generator().manual_seed(seed)
    responses = {}
    for half in ("half0", "half1"):
        responses[half] = {
            "circuit": .01 * torch.randn(N_ATOMS, 2, 32, generator=generator,
                                          dtype=torch.float64),
            "task": .01 * torch.randn(N_ATOMS, 2, 4, generator=generator,
                                       dtype=torch.float64),
        }
    betas = (0.5, -0.75, 1.5, -2.0)
    for (left, right), beta in zip(PLANTED_PAIRS, betas):
        for half in responses.values():
            for kind in ("circuit", "task"):
                half[kind][left] = beta * half[kind][right]
    return responses, set(PLANTED_PAIRS)


def planted_suite() -> dict:
    cases = []
    all_exact = True
    for seed in PLANTED_SEEDS:
        responses, expected = planted_problem(seed)
        found = {(row["left"], row["right"]) for row in discover_pairs(responses)}
        exact = found == expected
        all_exact &= exact
        cases.append({"seed": seed, "expected": sorted(expected), "found": sorted(found),
                      "exact": exact})
    return {"cases": cases, "all_eight_exact": bool(all_exact)}


def dry_run() -> dict:
    if sha256(PREREG) != PREREG_SHA256:
        raise RuntimeError("rung518 preregistration changed after source freeze")
    planted = planted_suite()
    if not planted["all_eight_exact"]:
        raise RuntimeError("rung518 planted pair detector failed")
    return {
        "status": "dry_run_passed", "rung": 518,
        "model_loaded": False, "model_outcomes_opened": False,
        "heads": N_HEADS, "relations": list(GROUPS), "atoms": N_ATOMS,
        "unordered_pairs": N_ATOMS * (N_ATOMS - 1) // 2,
        "planted_recovery": planted,
        "preregistration_sha256": sha256(PREREG),
    }


def scientific_main() -> None:
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung518 output namespace already exists")
    started = time.time()
    sys.path.insert(0, str(ROOT / "basis_aligned/bilinear_quotient/ops"))
    sys.path.insert(0, str(POLY))
    import bilin18_observed_model_facade as facade
    import mlp0_source_relation_factorial_rung517 as r517

    rows, task_masks, circuit_masks, _scales, discovery_tags, confirmation_tags, \
        metadata, response_parent = validate_inputs()
    task_indices = tuple(response_parent.TASK_CELLS.index(cell)
                         for cell in response_parent.GRAD_CELLS[:4])
    off_target_index = response_parent.TASK_CELLS.index("off_target")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    collections = {}
    collections["discovery"] = collect_phase(
        model, rows, task_masks, circuit_masks, discovery_tags, DISCOVERY,
        facade, r517, response_parent)
    discovery_responses = response_matrices(collections["discovery"], task_indices)
    candidates = discover_pairs(discovery_responses)
    control_counts = permutation_control_counts(discovery_responses)
    control_q95 = float(torch.quantile(
        torch.tensor(control_counts, dtype=torch.float64), .95, interpolation="higher"))
    diagnostic = collections["discovery"]["diagnostics"]
    task_support_totals = half_support_totals(
        collections["discovery"]["task_counts"], DISCOVERY)
    planted = planted_suite()
    pred_a = bool(
        planted["all_eight_exact"]
        and diagnostic["maximum_atom_closure"] <= 1e-8
        and diagnostic["maximum_native_replay_error"] == 0
        and diagnostic["minimum_single_edit_rms"] > 0
        and diagnostic["minimum_drop_edit_rms"] > 0
        and diagnostic["calls"] == diagnostic["calls_expected"] == 5766
        and half_supports_positive(
            collections["discovery"]["task_counts"], DISCOVERY)
        and bool((collections["discovery"]["circuit_counts"] > 0).all()))
    pred_b = bool(pred_a and 1 <= len(candidates) <= 16 and len(candidates) > control_q95)

    confirmation_checks = {}
    confirmed = []
    if pred_b:
        collections["confirmation"] = collect_phase(
            model, rows, task_masks, circuit_masks, confirmation_tags, CONFIRMATION,
            facade, r517, response_parent)
        confirmation_responses = response_matrices(
            collections["confirmation"], task_indices)
        confirmed, confirmation_checks = confirmation_pairs(
            confirmation_responses, candidates)
    pred_c = bool(confirmed)

    substitution_checks = {}
    physical_pairs = []
    if pred_c:
        collections["substitutions"] = collect_substitutions(
            model, rows, task_masks, circuit_masks, confirmation_tags,
            CONFIRMATION, confirmed, facade, r517, response_parent)
        physical_pairs, substitution_checks = score_substitutions(
            collections["substitutions"], collections["confirmation"],
            task_indices, off_target_index)
    pred_d = bool(physical_pairs)
    boundary_evidence = native_boundary_evidence(
        discovery_responses, confirmed, physical_pairs) if pred_d else []
    pred_e = bool(boundary_evidence)
    strong_null = not (pred_a and pred_b and pred_c and pred_d and pred_e)
    if not pred_a:
        next_step = "repair_exact_or_response_instrument_only"
    elif not pred_b:
        next_step = "leave_head_relation_basis_for_one_circuit_exact_interaction_atlas"
    elif not pred_c:
        next_step = "retain_in_sample_screen_only_and_leave_head_relation_basis"
    elif not pred_d:
        next_step = "retain_predictive_relation_without_circuit_equivalence_claim"
    elif not pred_e:
        next_step = "retain_physical_pair_without_native_boundary_improvement_claim"
    else:
        next_step = "build_and_compose_executable_cross_head_component"

    torch.save({
        "collections": collections,
        "discovery_responses": discovery_responses,
        "task_indices": task_indices,
        "discovery_tags": discovery_tags,
        "confirmation_tags": confirmation_tags,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 518,
        "claim_level": "operational_head_relation_quotient_not_compression_or_adoption",
        "checkpoint": checkpoint.__dict__,
        "source_hashes": {
            "preregistration": sha256(PREREG),
            "source": sha256(Path(sys.argv[0]).resolve()),
            "bundle": sha256(BUNDLE),
            **{path.name: digest for path, digest in HASHES.items()},
        },
        "row_sha256": hashlib.sha256(rows.contiguous().numpy().tobytes()).hexdigest(),
        "metadata": metadata,
        "atom_names": list(ATOM_NAMES), "atoms": N_ATOMS,
        "unordered_pairs_tested": N_ATOMS * (N_ATOMS - 1) // 2,
        "discovery_candidates": candidates,
        "discovery_candidate_count": len(candidates),
        "permutation_control_counts": control_counts,
        "permutation_control_q95_higher": control_q95,
        "confirmation_checks": confirmation_checks,
        "confirmed_pair_names": [
            f"{row['left_name']} <-> {row['right_name']}" for row in confirmed],
        "physical_substitution_checks": substitution_checks,
        "physical_pairs": physical_pairs,
        "native_boundary_evidence": boundary_evidence,
        'pred_a_exact_live_45_piece_instrument': pred_a,
        'pred_b_small_circuit_defined_relation': pred_b,
        'pred_c_heldout_circuit_prediction': pred_c,
        'pred_d_bidirectional_physical_interchange': pred_d,
        'pred_e_native_boundary_changing_unit': pred_e,
        "strong_null": strong_null, "next_step": next_step,
        "diagnostics": {
            **{key: collection["diagnostics"] for key, collection in collections.items()},
            "discovery_task_support_by_half": task_support_totals,
        },
        "execution_price": {
            "full_model_forwards": sum(
                collection["diagnostics"]["calls"] for collection in collections.values()),
            "backwards": 0, "deployed_parameters_added": 0,
            "deployed_parameters_saved": 0,
        },
        "claim_limits": [
            "fixed_45_atom_vocabulary_not_a_complete_semantic_basis",
            "no_rank_sae_reconstruction_or_compression_claim",
            "confirmation_and_substitution_are_gate_conditional",
            "full_pass_would_be_identification_not_adoption",
        ],
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": "complete", "rung": 518,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "discovery_candidates": len(candidates), "control_q95": control_q95,
        "confirmed": len(confirmed), "physical_pairs": len(physical_pairs),
        "boundary_components": len(boundary_evidence),
        "next_step": next_step, "runtime_s": result["runtime_s"],
    }, indent=2, sort_keys=True))


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(dry_run(), indent=2, sort_keys=True))
        return
    if os.environ.get("BQLIB_GPU_SMOKE") == "1" or "--gpu-smoke" in sys.argv:
        gpu_smoke()
        return
    scientific_main()


if __name__ == "__main__":
    main()
