#!/usr/bin/env python3
"""RUNG510 -- observable pairwise downstream equivalence inside MLP10."""

# BQGATE: EXPERIMENT
# pred_a: exact singleton and physical-substitution instruments are live
# pred_b: one to sixteen observable pairs pass discovery without ranking
# pred_c: at least one frozen pair predicts new documents and circuit families
# pred_d: at least one pair passes bidirectional physical term substitution
# pred_e: at least one physical pair uses different exact source terms

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

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import mlp10_coupled_causal_dictionary_rung509 as r509


PREREG = POLY / "MLP10_OBSERVABLE_PREDICTIVE_STATE_QUOTIENT_RUNG510_PREREGISTRATION.md"
PREFLIGHT_ADDENDUM = (
    POLY / "MLP10_OBSERVABLE_PREDICTIVE_STATE_QUOTIENT_RUNG510_PREFLIGHT_ADDENDUM.md")
R509_SOURCE = ROOT / "ops/mlp10_coupled_causal_dictionary_rung509.py"
R509_IDENTIFIABILITY_RESULT = (
    ROOT / "mlp10_coupled_causal_dictionary_rung509_identifiability_results.json")
R509_REPAIR = POLY / "MLP10_COUPLED_CAUSAL_DICTIONARY_RUNG509_IDENTIFIABILITY_REPAIR.md"
OUT = ROOT / "mlp10_observable_predictive_state_quotient_rung510_results.json"
BUNDLE = ROOT / "mlp10_observable_predictive_state_quotient_rung510_bundle.pt"
HASHES = {
    PREREG: "e344760333af378ea5604c211c259a27d9ff030b60bad8054ca962d465f46055",
    PREFLIGHT_ADDENDUM: "8e239ef80e02274b84f5cde1bc046e0e0656cdffb66450a1c50da76968bbc279",
    R509_SOURCE: "f346b78ab47006c68d522d153d441603e627e60233e6cba3dd703e7225ef6ec3",
    R509_IDENTIFIABILITY_RESULT: (
        "7a10e97a41328b97008d1e1b81a70de77977bdd2fb615dd701878ee9d26a3d1a"),
    R509_REPAIR: "381988395edd4d54c1d08ba99bef336ed0ca708fc48497dc479887e0d647f5bf",
}

DISCOVERY = (500, 748, 624)
CONFIRMATION = (752, 1000, 876)
MAX_CANDIDATES = 16
CONTROL_SEEDS = tuple(range(51000, 51016))
N_ACTIONS = len(r509.parent.SOURCES)
N_TERMS = len(r509.parent.PAIR_NAMES)
N_NODES = N_ACTIONS * N_TERMS
NODE_NAMES = tuple(
    f"{source}::{term}"
    for source in r509.parent.SOURCES for term in r509.parent.PAIR_NAMES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def node_parts(node: int) -> tuple[int, int]:
    if not 0 <= node < N_NODES:
        raise ValueError("node index changed")
    return divmod(node, N_TERMS)


def _task_matrix(collection: dict, window: str) -> torch.Tensor:
    rows = []
    for source in r509.parent.SOURCES:
        for term in r509.parent.PAIR_NAMES:
            rows.append(r509.parent.finite_vector(
                collection, term, collection, source, window).double())
    result = torch.stack(rows)
    if result.shape != (N_NODES, 4):
        raise RuntimeError("task response shape changed")
    return result


def _circuit_matrix(collection: dict, window: str) -> torch.Tensor:
    rows = []
    for source in r509.parent.SOURCES:
        for term in r509.parent.PAIR_NAMES:
            rows.append(r509._circuit_fingerprint(
                collection, term, source, window).double())
    result = torch.stack(rows)
    if result.shape != (N_NODES, len(collection["circuit_tags"])):
        raise RuntimeError("circuit response shape changed")
    return result


def response_matrices(collection: dict) -> dict[str, dict[str, torch.Tensor]]:
    return {
        window: {
            "task": _task_matrix(collection, window),
            "circuit": _circuit_matrix(collection, window),
        }
        for window in ("half0", "half1", "pooled")
    }


def _safe_cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left * right).sum(-1)
    denominator = torch.linalg.vector_norm(left, dim=-1) \
        * torch.linalg.vector_norm(right, dim=-1)
    return numerator / denominator.clamp_min(1e-30)


def _residual_rows(actual: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(actual - predicted, dim=-1) \
        / torch.linalg.vector_norm(actual, dim=-1).clamp_min(1e-30)


def _pair_metrics(matrices: dict, left: int, right: int, beta: float) -> dict:
    row = {"beta_left_from_right": beta, "windows": {}}
    inverse = 1.0 / beta
    holds = True
    for window in ("half0", "half1", "pooled"):
        entry = {}
        for kind in ("circuit", "task"):
            left_vector = matrices[window][kind][left]
            right_vector = matrices[window][kind][right]
            forward = beta * right_vector
            backward = inverse * left_vector
            entry[kind] = {
                "left_from_right_cosine": float(_safe_cosine_rows(
                    left_vector[None], forward[None])[0]),
                "left_from_right_relative_residual": float(_residual_rows(
                    left_vector[None], forward[None])[0]),
                "right_from_left_cosine": float(_safe_cosine_rows(
                    right_vector[None], backward[None])[0]),
                "right_from_left_relative_residual": float(_residual_rows(
                    right_vector[None], backward[None])[0]),
            }
        row["windows"][window] = entry
    circuit_rms = [
        float(matrices["pooled"]["circuit"][node].square().mean().sqrt())
        for node in (left, right)]
    task_norm = [
        float(torch.linalg.vector_norm(matrices["pooled"]["task"][node]))
        for node in (left, right)]
    row["circuit_rms_nat"] = circuit_rms
    row["task_norm_nat"] = task_norm
    row["material"] = bool(min(circuit_rms) >= .0005 and min(task_norm) >= .00025)
    row["scale_holds"] = bool(.25 <= abs(beta) <= 4)
    for window in ("half0", "half1"):
        c = row["windows"][window]["circuit"]
        t = row["windows"][window]["task"]
        c_cos = .90 if window == "half0" else .80
        c_res = .35 if window == "half0" else .50
        window_holds = bool(
            min(c["left_from_right_cosine"], c["right_from_left_cosine"]) >= c_cos
            and max(c["left_from_right_relative_residual"],
                    c["right_from_left_relative_residual"]) <= c_res
            and min(t["left_from_right_cosine"], t["right_from_left_cosine"]) >= .70
            and max(t["left_from_right_relative_residual"],
                    t["right_from_left_relative_residual"]) <= .65)
        row["windows"][window]["holds"] = window_holds
        holds &= window_holds
    row["holds"] = bool(row["material"] and row["scale_holds"] and holds)
    return row


def discover_pairs(matrices: dict) -> tuple[list[dict], dict]:
    """Vectorized all-pairs detector; returns every passing upper-triangle pair."""
    c0 = matrices["half0"]["circuit"].double()
    c1 = matrices["half1"]["circuit"].double()
    t0 = matrices["half0"]["task"].double()
    t1 = matrices["half1"]["task"].double()
    cp = matrices["pooled"]["circuit"].double()
    tp = matrices["pooled"]["task"].double()
    dot = c0 @ c0.T
    norm2 = c0.square().sum(-1)
    beta = dot / norm2[None, :].clamp_min(1e-30)
    safe_beta = torch.where(beta.abs() > 1e-30, beta, torch.ones_like(beta))

    def predicted_metrics(values: torch.Tensor):
        raw_dot = values @ values.T
        norms = torch.linalg.vector_norm(values, dim=1)
        raw_cos = raw_dot / (norms[:, None] * norms[None, :]).clamp_min(1e-30)
        signed_cos = raw_cos * safe_beta.sign()
        forward_sq = (
            values.square().sum(-1)[:, None]
            + beta.square() * values.square().sum(-1)[None, :]
            - 2 * beta * raw_dot).clamp_min(0)
        forward = forward_sq.sqrt() / norms[:, None].clamp_min(1e-30)
        inverse = safe_beta.reciprocal()
        backward_sq = (
            values.square().sum(-1)[None, :]
            + inverse.square() * values.square().sum(-1)[:, None]
            - 2 * inverse * raw_dot).clamp_min(0)
        backward = backward_sq.sqrt() / norms[None, :].clamp_min(1e-30)
        return signed_cos, forward, backward

    c0_cos, c0_forward, c0_backward = predicted_metrics(c0)
    c1_cos, c1_forward, c1_backward = predicted_metrics(c1)
    t0_cos, t0_forward, t0_backward = predicted_metrics(t0)
    t1_cos, t1_forward, t1_backward = predicted_metrics(t1)
    circuit_rms = cp.square().mean(-1).sqrt()
    task_norm = torch.linalg.vector_norm(tp, dim=-1)
    material = ((circuit_rms[:, None] >= .0005) & (circuit_rms[None, :] >= .0005)
                & (task_norm[:, None] >= .00025) & (task_norm[None, :] >= .00025))
    mask = (
        material & (beta.abs() >= .25) & (beta.abs() <= 4)
        & (c0_cos >= .90) & (c0_forward <= .35) & (c0_backward <= .35)
        & (c1_cos >= .80) & (c1_forward <= .50) & (c1_backward <= .50)
        & (t0_cos >= .70) & (t0_forward <= .65) & (t0_backward <= .65)
        & (t1_cos >= .70) & (t1_forward <= .65) & (t1_backward <= .65))
    mask = torch.triu(mask, diagonal=1)
    indices = torch.nonzero(mask, as_tuple=False)
    candidates = []
    for left, right in indices.tolist():
        metrics = _pair_metrics(matrices, left, right, float(beta[left, right]))
        if not metrics["holds"]:
            raise RuntimeError("vectorized and scalar pair detectors disagree")
        candidates.append({
            "left_node": left, "right_node": right,
            "left_name": NODE_NAMES[left], "right_name": NODE_NAMES[right],
            **metrics,
        })
    summary = {
        "nodes": N_NODES,
        "unordered_pairs_tested": N_NODES * (N_NODES - 1) // 2,
        "candidate_count": len(candidates),
        "small_relation": bool(1 <= len(candidates) <= MAX_CANDIDATES),
        "material_nodes": int(((circuit_rms >= .0005) & (task_norm >= .00025)).sum()),
    }
    return candidates, summary


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
            for window in ("half0", "half1", "pooled")
        }
        _pairs, summary = discover_pairs(control)
        counts.append(summary["candidate_count"])
    return counts


def confirmation_pairs(matrices: dict, candidates: list[dict]) -> tuple[list[dict], dict]:
    passing, checks = [], {}
    for candidate in candidates:
        left, right = candidate["left_node"], candidate["right_node"]
        beta = candidate["beta_left_from_right"]
        metrics = _pair_metrics(matrices, left, right, beta)
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


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    identifiability = json.loads(R509_IDENTIFIABILITY_RESULT.read_text())
    if not (
        identifiability.get("holds") is False
        and identifiability.get("model_loaded") is False
        and identifiability.get("model_outcomes_opened") is False
        and identifiability.get("registered_failure_route")
        == "downstream_predictive_state_quotient_without_latent_dictionary"
        and not r509.OUT.exists()
        and not r509.BUNDLE.exists()
    ):
        raise RuntimeError("rung509 failure route changed or model outcome was opened")
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        r509.parent.validate_inputs()
    if len(discovery_tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("62-circuit discovery/confirmation partition changed")
    return rows, task_masks, circuit_masks, scales, list(discovery_tags), \
        list(validation_tags), {
            **metadata,
            "rung509_identifiability_sha256": sha256(R509_IDENTIFIABILITY_RESULT),
            "rung509_model_outcome_absent": True,
            "documents": {"discovery": list(DISCOVERY), "unused": [748, 752],
                          "confirmation": list(CONFIRMATION)},
            "circuits": {"discovery": list(discovery_tags),
                         "confirmation": list(validation_tags)},
        }


def _physical_diagnostics() -> dict:
    diagnostics = r509._empty_diagnostics()
    diagnostics.update({"substitution_patches": 0, "substitution_patches_expected": 0,
                        "substitution_patches_exact": False})
    return diagnostics


@torch.no_grad()
def collect_substitutions(model, rows, task_masks, circuit_masks, circuit_tags,
                          scales, bounds, candidates):
    lo, hi, _split = bounds
    documents = hi - lo
    directions = []
    for edge_index, candidate in enumerate(candidates):
        directions.extend((
            {"edge": edge_index, "target": candidate["left_node"],
             "donor": candidate["right_node"],
             "scale": candidate["beta_left_from_right"], "side": "left_from_right"},
            {"edge": edge_index, "target": candidate["right_node"],
             "donor": candidate["left_node"],
             "scale": 1.0 / candidate["beta_left_from_right"], "side": "right_from_left"},
        ))
    task = torch.zeros(len(directions), documents, len(r509.parent.TASK_CELLS),
                       dtype=torch.float64)
    counts = torch.zeros(documents, len(r509.parent.TASK_CELLS), dtype=torch.float64)
    circuit_sums = torch.zeros(len(directions), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    diagnostics = _physical_diagnostics()
    device = next(model.parameters()).device
    mlp = model.transformer.h[r509.parent.TARGET].mlp
    needed_nodes = sorted({row[key] for row in directions for key in ("target", "donor")})
    for start in range(lo, hi, r509.parent.BATCH):
        stop = start + r509.parent.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in r509.parent.TASK_CELLS}
        direct_logits, _, direct_diag, _ = r509.parent._forward(
            model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        r509._update_diagnostics(diagnostics, direct_diag)
        _absent_logits, absent, absent_diag, _ = r509.parent._forward(
            model, tokens, scales, action="P", absent=True, capture_mlp10=True)
        diagnostics["calls"]["analytical"] += 1
        r509._update_diagnostics(diagnostics, absent_diag)
        captures = []
        for action, source in enumerate(r509.parent.SOURCES):
            logits, capture, diag, _ = r509.parent._forward(
                model, tokens, scales, action=source, capture_mlp10=True)
            diagnostics["calls"]["analytical"] += 1
            r509._update_diagnostics(diagnostics, diag)
            r509.parent._score_delta_closure(diagnostics, capture, absent)
            if source == "N":
                difference = logits.detach().float() - direct_logits.float()
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"],
                    float(difference.abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    float(difference.square().sum())
                    / max(float(direct_logits.float().square().sum()), 1e-30))
            captures.append(capture)
        absent_outputs = {}
        donor_deltas = {}
        for node in needed_nodes:
            action, term = node_parts(node)
            if term not in absent_outputs:
                absent_outputs[term] = r509.parent._pair_output(mlp, absent["factors"], term)
            donor_deltas[node] = (
                r509.parent._pair_output(mlp, captures[action]["factors"], term)
                - absent_outputs[term])
        nll_rows = []
        for direction in directions:
            target_action, _target_term = node_parts(direction["target"])
            source = r509.parent.SOURCES[target_action]
            delta = direction["scale"] * donor_deltas[direction["donor"]]
            replacement = captures[target_action]["deployed_write"] \
                - delta.to(captures[target_action]["deployed_write"].dtype)
            logits, _captures, patch_diag, patch_audit = r509.parent.score_parent.run_forward(
                model, tokens, action=source, scales=scales,
                patch_writes={"m10": replacement})
            diagnostics["calls"]["analytical"] += 1
            diagnostics["substitution_patches"] += patch_audit["patches"]
            edit_rms = patch_diag["patch_rms_max"]
            diagnostics["zero_term_edits"] += int(edit_rms <= 0)
            if edit_rms > 0:
                diagnostics["minimum_nonzero_term_edit_rms"] = min(
                    diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
            nll_rows.append(r509.parent._nll(logits, batch_rows).detach().cpu())
        nll_stack = torch.stack(nll_rows)
        task[:, local:local + r509.parent.BATCH] = r509.parent._task_sums(
            nll_stack, masks)
        counts[local:local + r509.parent.BATCH] = torch.stack(
            [masks[cell].sum(1).double() for cell in r509.parent.TASK_CELLS], -1)
        matrix, observed = r509.parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        circuit_counts += observed
        circuit_sums += torch.matmul(
            nll_stack.view(len(directions), -1).double(), matrix.T,
        ).view(len(directions), 2, 2, len(circuit_tags))
    batches = documents // r509.parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches,
        "analytical": batches * (1 + N_ACTIONS + len(directions)),
    }
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["substitution_patches_expected"] = batches * len(directions)
    diagnostics["substitution_patches_exact"] = (
        diagnostics["substitution_patches"]
        == diagnostics["substitution_patches_expected"])
    return {
        "bounds": bounds, "directions": directions, "task": task,
        "task_counts": counts, "circuit_tags": tuple(circuit_tags),
        "circuit_sums": circuit_sums, "circuit_counts": circuit_counts,
        "diagnostics": diagnostics,
    }


def _substitution_instrument(collection: dict) -> bool:
    d = collection["diagnostics"]
    return bool(
        d["calls_exact"] and d["substitution_patches_exact"]
        and d["zero_term_edits"] == 0
        and d["factor_reconstruction_max"] <= 1e-10
        and d["raw_source_relative_squared"] <= r509.parent.DEPLOYED_BF16_BAR
        and d["normalized_closure_relative_squared"] <= 1e-12
        and d["normalized_numerical_rms_ratio"] <= .02
        and d["float32_mlp10_closure"] <= 1e-10
        and d["deployed_mlp10_relative_squared"] <= r509.parent.DEPLOYED_BF16_BAR
        and d["score_delta_float32_closure"] <= 1e-10
        and math.isfinite(d["score_delta_predeployment_relative_squared"])
        and d["score_delta_deployed_closure_relative_squared"] <= 1e-12
        and d["minimum_nonzero_score_edit_rms"] > 0
        and d["minimum_nonzero_term_edit_rms"] > 0
        and d["native_replay_logit_max_abs"] == 0.0
        and d["native_replay_relative_squared"] <= 1e-12)


def _native_node_response(exact: dict, node: int, window: str) -> dict[str, torch.Tensor]:
    action, term = node_parts(node)
    source = r509.parent.SOURCES[action]
    name = r509.parent.PAIR_NAMES[term]
    return {
        "task": r509.parent.finite_vector(exact, name, exact, source, window).double(),
        "circuit": r509._circuit_fingerprint(exact, name, source, window).double(),
    }


def _substituted_response(substitutions: dict, exact: dict, direction: int,
                          window: str) -> dict[str, torch.Tensor]:
    target_node = substitutions["directions"][direction]["target"]
    action, _term = node_parts(target_node)
    intact_index = exact["arms"].index("intact")
    if window == "pooled":
        lo, hi = 0, substitutions["task"].shape[1]
        circuit_sum = substitutions["circuit_sums"][direction].sum(0)
        intact_sum = exact["circuit_sums"][action, intact_index].sum(0)
        circuit_counts = substitutions["circuit_counts"].sum(0)
    else:
        half = {"half0": 0, "half1": 1}[window]
        bounds = substitutions["bounds"]
        absolute = ((bounds[0], bounds[2]), (bounds[2], bounds[1]))[half]
        lo, hi = absolute[0] - bounds[0], absolute[1] - bounds[0]
        circuit_sum = substitutions["circuit_sums"][direction, half]
        intact_sum = exact["circuit_sums"][action, intact_index, half]
        circuit_counts = substitutions["circuit_counts"][half]
    task_numerator = (
        substitutions["task"][direction, lo:hi]
        - exact["task"][action, intact_index, lo:hi]).sum(0)
    task_denominator = substitutions["task_counts"][lo:hi].sum(0).clamp_min(1)
    task_full = task_numerator / task_denominator
    indices = [r509.parent.TASK_CELLS.index(cell) for cell in r509.parent.GRAD_CELLS[:4]]
    circuit = ((circuit_sum - intact_sum) / circuit_counts.clamp_min(1))[0] \
        - ((circuit_sum - intact_sum) / circuit_counts.clamp_min(1))[1]
    return {"task": task_full[indices], "circuit": circuit}


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
                native = _native_node_response(exact, target, window)
                observed = _substituted_response(substitutions, exact, direction, window)
                circuit_cos = float(_safe_cosine_rows(
                    native["circuit"][None], observed["circuit"][None])[0])
                circuit_residual = float(_residual_rows(
                    native["circuit"][None], observed["circuit"][None])[0])
                task_cos = float(_safe_cosine_rows(
                    native["task"][None], observed["task"][None])[0])
                task_residual = float(_residual_rows(
                    native["task"][None], observed["task"][None])[0])
                holds = bool(circuit_cos >= .75 and circuit_residual <= .55
                             and task_cos >= .70 and task_residual <= .65)
                side_row["windows"][window] = {
                    "circuit_cosine": circuit_cos,
                    "circuit_relative_residual": circuit_residual,
                    "task_cosine": task_cos,
                    "task_relative_residual": task_residual,
                    "holds": holds,
                }
                side_row["holds"] &= holds
            row["directions"][side] = side_row
            row["holds"] &= side_row["holds"]
        key = f"{candidate['left_name']} <-> {candidate['right_name']}"
        checks[key] = row
        if row["holds"]:
            passing.append(candidate)
    return passing, checks


def quotient_groups(passing: list[dict]) -> list[dict]:
    adjacency = {}
    edge_map = {}
    for edge in passing:
        left, right = edge["left_node"], edge["right_node"]
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
        edge_map[(left, right)] = edge["beta_left_from_right"]
    groups, visited = [], set()
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack, component = [start], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency.get(node, ()))
        visited |= component
        nodes = sorted(component)
        complete = all((min(left, right), max(left, right)) in edge_map
                       for left, right in itertools.combinations(nodes, 2))
        root = nodes[0]
        scales = {root: 1.0}
        for node in nodes[1:]:
            beta = edge_map[(root, node)]
            scales[node] = 1.0 / beta
        errors = []
        if complete:
            for left, right in itertools.combinations(nodes, 2):
                observed = edge_map[(left, right)]
                expected = scales[left] / scales[right]
                errors.append(abs(observed - expected) / max(abs(expected), 1e-30))
        cycle_consistent = bool(complete and (not errors or max(errors) <= .25))
        if cycle_consistent:
            groups.append({
                "nodes": nodes, "node_names": [NODE_NAMES[node] for node in nodes],
                "scales_relative_to_root": {NODE_NAMES[node]: scales[node] for node in nodes},
                "maximum_scale_cycle_relative_error": max(errors, default=0.0),
                "complete_graph": True, "cycle_consistent": True,
            })
    return groups


def _bundle_collection(collection):
    return {key: value for key, value in collection.items() if key != "diagnostics"}


def dry_run() -> None:
    generator = torch.Generator(device="cpu").manual_seed(510)
    base_circuit = torch.randn(N_NODES, 32, generator=generator, dtype=torch.float64)
    base_task = torch.randn(N_NODES, 4, generator=generator, dtype=torch.float64)
    base_circuit *= .002
    base_task *= .002
    base_circuit[1] = -2 * base_circuit[0]
    base_task[1] = -2 * base_task[0]
    matrices = {
        window: {"circuit": base_circuit.clone(), "task": base_task.clone()}
        for window in ("half0", "half1", "pooled")}
    candidates, summary = discover_pairs(matrices)
    assert summary["unordered_pairs_tested"] == 511566
    assert any(edge["left_node"] == 0 and edge["right_node"] == 1
               for edge in candidates)
    assert 2 * 63116 + 372 + 124 * 16 == 128588
    print(json.dumps({
        "status": "dry_run_passed", "rung": 510, "model_loaded": False,
        "outcomes_opened": False, "nodes": N_NODES,
        "unordered_pairs_tested": summary["unordered_pairs_tested"],
        "maximum_conditional_forwards": 128588,
    }, indent=2, sort_keys=True))


def _gpu_smoke() -> None:
    rows, task_masks, circuit_masks, scales, discovery_tags, _validation_tags, _metadata = \
        validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    bounds = (500, 504, 502)
    exact = r509.collect_exact(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, bounds)
    candidates = [{
        "left_node": 0, "right_node": N_TERMS + 1,
        "left_name": NODE_NAMES[0], "right_name": NODE_NAMES[N_TERMS + 1],
        "beta_left_from_right": 1.0,
    }]
    substitutions = collect_substitutions(
        model, rows, task_masks, circuit_masks, discovery_tags,
        scales, bounds, candidates)
    checks = {
        "weights": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256,
        "exact": r509._instrument(exact, exact=True),
        "substitutions": _substitution_instrument(substitutions),
        "all_exact_patches": exact["diagnostics"]["patches"] == 1012,
        "both_substitution_patches": (
            substitutions["diagnostics"]["substitution_patches"] == 2),
    }
    passed = all(checks.values())
    print(json.dumps({
        "status": "smoke_passed" if passed else "smoke_failed", "rung": 510,
        "scientific_outcomes_retained": False, "checks": checks,
        "exact_diagnostics": exact["diagnostics"],
        "substitution_diagnostics": substitutions["diagnostics"],
        "full_forwards": sum(exact["diagnostics"]["calls"].values())
        + sum(substitutions["diagnostics"]["calls"].values()),
        "backwards": 0,
    }, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(
            f"rung510 CUDA smoke failed: "
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
        raise RuntimeError("rung510 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    collections = {}
    collections["exact_discovery"] = r509.collect_exact(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, DISCOVERY)
    discovery_calibration = r509.parent._calibration(
        collections["exact_discovery"]["base_task"],
        collections["exact_discovery"]["source_task"],
        collections["exact_discovery"]["task_counts"], DISCOVERY)
    discovery_calibration_ok = r509.parent.state_parent.calibration_holds(
        discovery_calibration)
    discovery_matrices = response_matrices(collections["exact_discovery"])
    candidates, discovery_summary = discover_pairs(discovery_matrices)
    control_counts = permutation_control_counts(discovery_matrices)
    pred_b_pre = bool(discovery_summary["small_relation"])

    confirmation_checks, confirmed = {}, []
    physical_checks, physical_pairs, groups = {}, [], []
    confirmation_calibration, confirmation_calibration_ok = {}, False
    if discovery_calibration_ok and pred_b_pre:
        collections["exact_confirmation"] = r509.collect_exact(
            model, rows, task_masks, circuit_masks, validation_tags, scales, CONFIRMATION)
        confirmation_calibration = r509.parent._calibration(
            collections["exact_confirmation"]["base_task"],
            collections["exact_confirmation"]["source_task"],
            collections["exact_confirmation"]["task_counts"], CONFIRMATION)
        confirmation_calibration_ok = r509.parent.state_parent.calibration_holds(
            confirmation_calibration)
        confirmation_matrices = response_matrices(collections["exact_confirmation"])
        confirmed, confirmation_checks = confirmation_pairs(
            confirmation_matrices, candidates)
    if confirmation_calibration_ok and confirmed:
        collections["substitutions"] = collect_substitutions(
            model, rows, task_masks, circuit_masks, validation_tags,
            scales, CONFIRMATION, confirmed)
        physical_pairs, physical_checks = score_substitutions(
            collections["substitutions"], collections["exact_confirmation"], confirmed)
        groups = quotient_groups(physical_pairs)

    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and discovery_calibration_ok
        and ("exact_confirmation" not in collections or confirmation_calibration_ok)
        and r509._instrument(collections["exact_discovery"], exact=True)
        and all(r509._instrument(collection, exact=True)
                for name, collection in collections.items()
                if name == "exact_confirmation")
        and all(_substitution_instrument(collection)
                for name, collection in collections.items()
                if name == "substitutions"))
    pred_b = bool(pred_a and pred_b_pre)
    pred_c = bool(pred_b and confirmation_calibration_ok and confirmed)
    pred_d = bool(pred_c and physical_pairs)
    pred_e = bool(pred_d and any(
        node_parts(edge["left_node"])[1] != node_parts(edge["right_node"])[1]
        for edge in physical_pairs))
    strong_null = not (pred_a and pred_b and pred_c and pred_d and pred_e)
    if not pred_a:
        next_step = "repair_observable_equivalence_instrument_only"
    elif not pred_b and discovery_summary["candidate_count"] == 0:
        next_step = "registered_multi_term_signed_combinations_without_pair_ranking"
    elif not pred_b:
        next_step = "add_independent_downstream_tasks_without_selecting_sixteen"
    elif not pred_c:
        next_step = "consumer_specific_nonlinear_readout_test"
    elif not pred_d:
        next_step = "localize_first_downstream_consumer_that_separates_response_similar_pairs"
    elif not pred_e:
        next_step = "retain_same_term_action_portability_without_new_grouping_claim"
    else:
        next_step = "validate_quotient_pairs_on_ood_code_then_price_executable_replacement"

    bundle_payload = {
        "schema": "rung510_observable_predictive_state_quotient_v1",
        "collections": {name: _bundle_collection(collection)
                        for name, collection in collections.items()},
        "discovery_candidates": candidates, "confirmed_pairs": confirmed,
        "physical_pairs": physical_pairs,
        "raw_tokens_logits_hidden_states_or_weights_included": False,
    }
    torch.save(bundle_payload, BUNDLE)
    result = {
        "status": "complete", "rung": 510,
        "claim_level": "observable_pair_until_bidirectional_physical_substitution_passes",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "node_count": N_NODES,
        "node_names": list(NODE_NAMES),
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
            "physical_checks": physical_checks,
            "physical_pairs": physical_pairs,
            "quotient_groups": groups,
        },
        'pred_a_exact_live_singleton_and_substitution_instrument': pred_a,
        'pred_b_one_to_sixteen_discovery_equivalence_pairs': pred_b,
        'pred_c_heldout_documents_and_circuit_families': pred_c,
        'pred_d_bidirectional_physical_substitution': pred_d,
        'pred_e_different_exact_terms_share_downstream_variable': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "full_forwards": sum(sum(collection["diagnostics"]["calls"].values())
                                 for collection in collections.values()),
            "backwards": 0,
            "cpu_pair_comparisons": N_NODES * (N_NODES - 1) // 2,
            "discovery_candidates": len(candidates),
            "confirmed_pairs": len(confirmed),
            "physical_pairs": len(physical_pairs),
            "maximum_conditional_forwards": 128588,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started, "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 510,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "discovery_candidates": len(candidates), "confirmed_pairs": len(confirmed),
        "physical_pairs": len(physical_pairs), "quotient_groups": len(groups),
        "execution_price": result["execution_price"], "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
