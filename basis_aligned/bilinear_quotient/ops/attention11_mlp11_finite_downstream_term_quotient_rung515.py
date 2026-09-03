#!/usr/bin/env python3
"""RUNG515 -- finite nonlinear downstream quotient of exact consumer terms."""

# BQGATE: EXPERIMENT
# pred_a: exact finite-removal, control, planted-recovery, and substitution instrument is live
# pred_b: one to sixteen cross-action exact-term pairs pass discovery and beat controls
# pred_c: at least one frozen pair predicts untouched documents and thirty held-out circuits
# pred_d: at least one pair passes bidirectional same-site physical term substitution
# pred_e: a physical pair changes term identity or reuses one mapping across branch subsets

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
for path in (ROOT, ROOT / "ops", POLY, ROOT.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import attention11_mlp11_constrained_multi_term_programs_rung514 as r514
import mlp10_observable_predictive_state_quotient_rung510 as r510


r513, r512, r511, parent = r514.r513, r514.r512, r514.r511, r514.parent
PREREG = POLY / "ATTENTION11_MLP11_FINITE_DOWNSTREAM_TERM_QUOTIENT_RUNG515_PREREGISTRATION.md"
R514_RESULT = ROOT / "attention11_mlp11_constrained_multi_term_programs_rung514_results.json"
R514_BUNDLE = ROOT / "attention11_mlp11_constrained_multi_term_programs_rung514_bundle.pt"
R514_SOURCE = ROOT / "ops/attention11_mlp11_constrained_multi_term_programs_rung514.py"
R514_PREREG = POLY / "ATTENTION11_MLP11_CONSTRAINED_MULTI_TERM_PROGRAMS_RUNG514_PREREGISTRATION.md"
R514_ADDENDUM = POLY / "ATTENTION11_MLP11_CONSTRAINED_MULTI_TERM_PROGRAMS_RUNG514_PREFLIGHT_ADDENDUM.md"
R510_PREREG = POLY / "MLP10_OBSERVABLE_PREDICTIVE_STATE_QUOTIENT_RUNG510_PREREGISTRATION.md"
R510_RESULT = ROOT / "mlp10_observable_predictive_state_quotient_rung510_results.json"
R510_SOURCE = ROOT / "ops/mlp10_observable_predictive_state_quotient_rung510.py"
OUT = ROOT / "attention11_mlp11_finite_downstream_term_quotient_rung515_results.json"
BUNDLE = ROOT / "attention11_mlp11_finite_downstream_term_quotient_rung515_bundle.pt"

HASHES = {
    PREREG: "cb041ae0c0595aa2e5bbaab64923d686ae59fe96586a5def23bb0c080e16dfcf",
    R514_RESULT: "864f7834bd15f8dda591a5aa8e925b4af6b757cdaa3cce54f1e02e56271c00ec",
    R514_BUNDLE: "6e4d1037ef64563001907da1af6ec2ffa4e4ccc581c59a26f02ce9801d82b7b1",
    R514_SOURCE: "4248ce6e14789a6d0ee0d907626d4ae3b884c06ae5bbf82520d9ec0e62dcd28a",
    R514_PREREG: "602e167697e1eda8099ee8e52037cb3bf844f793722bba6da463b89cb0fd7957",
    R514_ADDENDUM: "30e3635ecc31ffc764b41d65edad426671fac3bf1651ac04317983b32cf3f0c7",
    R510_PREREG: "e344760333af378ea5604c211c259a27d9ff030b60bad8054ca962d465f46055",
    R510_RESULT: "16d100e7b92152fc70939b000934699882605c30c513c570f6c519b80f943177",
    R510_SOURCE: "7901aa5d9c7c39bf5666e0f081bfe08047f23c73eec08b12508c601def7b967a",
}

DISCOVERY = (500, 748, 624)
CONFIRMATION = (752, 1000, 876)
MAX_CANDIDATES = 16
CONTROL_SEEDS = tuple(range(51510, 51526))
PLANTED_SEEDS = tuple(range(51500, 51508))
SITE_NAMES = ("a11", "m11")
SITE_TERMS = {"a11": r513.ATTENTION_TERMS, "m11": r513.MLP_TERMS}
SITE_OFFSETS = {"a11": 0, "m11": len(r513.ATTENTION_TERMS)}
N_SUBSETS = len(r513.SELECTED_SUBSETS)
N_ACTIONS = len(parent.SOURCES)
N_TERMS = len(r513.TERM_NAMES)
PAIR_COUNT = N_SUBSETS * len(r513.RELATION_ACTIONS) * sum(
    len(names) ** 2 for names in SITE_TERMS.values())
COPY_INDICES = tuple(parent.TASK_CELLS.index(cell) for cell in r512.COPY_CELLS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_name(subset: int, relation: int, site: str,
              left_term: int, right_term: int) -> str:
    left_action, right_action = r513.RELATION_ACTIONS[relation]
    subset_name = r511.SUBSET_NAMES[r513.SELECTED_SUBSETS[subset]]
    return (
        f"{subset_name} @ {parent.SOURCES[left_action]}::{SITE_TERMS[site][left_term]}"
        f" <-> {parent.SOURCES[right_action]}::{SITE_TERMS[site][right_term]}"
    )


def _window_document_slice(collection: dict, window: str) -> slice:
    lo, hi, split = collection["bounds"]
    if window == "half0":
        return slice(0, split - lo)
    if window == "half1":
        return slice(split - lo, hi - lo)
    if window == "pooled":
        return slice(0, hi - lo)
    raise ValueError(f"unknown window {window}")


def _task_effect(collection: dict, subset: int, action: int,
                 term: int, window: str) -> torch.Tensor:
    selected = _window_document_slice(collection, window)
    numerator = (
        collection["removal_task"][subset, action, term, selected]
        - collection["intact_task"][action, selected]
    ).sum(0)
    denominator = collection["task_counts"][selected].sum(0).clamp_min(1)
    return (numerator / denominator)[list(COPY_INDICES)].double()


def _circuit_effect(collection: dict, subset: int, action: int,
                    term: int, window: str) -> torch.Tensor:
    if window == "pooled":
        removed = collection["removal_circuit_sums"][subset, action, term].sum(0)
        intact = collection["intact_circuit_sums"][action].sum(0)
        counts = collection["circuit_counts"].sum(0)
    else:
        half = {"half0": 0, "half1": 1}[window]
        removed = collection["removal_circuit_sums"][subset, action, term, half]
        intact = collection["intact_circuit_sums"][action, half]
        counts = collection["circuit_counts"][half]
    effect = (removed - intact) / counts.clamp_min(1)
    return (effect[0] - effect[1]).double()


def response_matrices(collection: dict) -> dict:
    matrices = {}
    for window in ("half0", "half1", "pooled"):
        task = torch.empty(N_SUBSETS, N_ACTIONS, N_TERMS, len(COPY_INDICES),
                           dtype=torch.float64)
        circuit = torch.empty(N_SUBSETS, N_ACTIONS, N_TERMS,
                              len(collection["circuit_tags"]), dtype=torch.float64)
        for subset in range(N_SUBSETS):
            for action in range(N_ACTIONS):
                for term in range(N_TERMS):
                    task[subset, action, term] = _task_effect(
                        collection, subset, action, term, window)
                    circuit[subset, action, term] = _circuit_effect(
                        collection, subset, action, term, window)
        matrices[window] = {"task": task, "circuit": circuit}
    return matrices


def _predicted_metrics(left: torch.Tensor, right: torch.Tensor,
                       beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _single_pair_metrics(matrices: dict, candidate: dict,
                         *, confirmation: bool = False) -> dict:
    subset, relation, site = (
        candidate["subset"], candidate["relation"], candidate["site"])
    left_action, right_action = r513.RELATION_ACTIONS[relation]
    offset = SITE_OFFSETS[site]
    left_term = offset + candidate["left_term"]
    right_term = offset + candidate["right_term"]
    beta = candidate["beta_left_from_right"]
    row = {"beta_left_from_right": beta, "windows": {}}
    holds = True
    for window in ("half0", "half1", "pooled"):
        entry = {}
        for kind in ("circuit", "task"):
            left = matrices[window][kind][subset, left_action, left_term]
            right = matrices[window][kind][subset, right_action, right_term]
            predicted = beta * right
            reciprocal = left / beta
            entry[kind] = {
                "cosine": float(r510._safe_cosine_rows(left[None], predicted[None])[0]),
                "left_from_right_relative_residual": float(
                    r510._residual_rows(left[None], predicted[None])[0]),
                "right_from_left_relative_residual": float(
                    r510._residual_rows(right[None], reciprocal[None])[0]),
            }
        row["windows"][window] = entry
    circuit_rms = []
    task_norm = []
    for action, term in ((left_action, left_term), (right_action, right_term)):
        circuit_rms.append(float(
            matrices["pooled"]["circuit"][subset, action, term].square().mean().sqrt()))
        task_norm.append(float(torch.linalg.vector_norm(
            matrices["pooled"]["task"][subset, action, term])))
    row["circuit_rms_nat"] = circuit_rms
    row["task_norm_nat"] = task_norm
    row["material"] = bool(min(circuit_rms) >= .0005 and min(task_norm) >= .00025)
    row["scale_holds"] = bool(.25 <= abs(beta) <= 4)
    for window in ("half0", "half1") if not confirmation else ("half0", "half1", "pooled"):
        c, t = row["windows"][window]["circuit"], row["windows"][window]["task"]
        c_cos = .75 if confirmation else (.90 if window == "half0" else .80)
        c_res = .55 if confirmation else (.35 if window == "half0" else .50)
        window_holds = bool(
            c["cosine"] >= c_cos
            and max(c["left_from_right_relative_residual"],
                    c["right_from_left_relative_residual"]) <= c_res
            and t["cosine"] >= .70
            and max(t["left_from_right_relative_residual"],
                    t["right_from_left_relative_residual"]) <= .65)
        row["windows"][window]["holds"] = window_holds
        holds &= window_holds
    row["holds"] = bool(row["material"] and row["scale_holds"] and holds)
    return row


def discover_pairs(matrices: dict, *, retain: bool = True) -> tuple[list[dict], dict]:
    candidates, top = [], []
    material_nodes = set()
    for subset in range(N_SUBSETS):
        for relation, (left_action, right_action) in enumerate(r513.RELATION_ACTIONS):
            for site, names in SITE_TERMS.items():
                offset = SITE_OFFSETS[site]
                indices = slice(offset, offset + len(names))
                c0_left = matrices["half0"]["circuit"][subset, left_action, indices]
                c0_right = matrices["half0"]["circuit"][subset, right_action, indices]
                dot = c0_left @ c0_right.T
                beta = dot / c0_right.square().sum(-1)[None, :].clamp_min(1e-30)
                valid_beta = torch.where(beta.abs() > 1e-30, beta, torch.ones_like(beta))
                metrics = {}
                for window in ("half0", "half1"):
                    for kind in ("circuit", "task"):
                        left = matrices[window][kind][subset, left_action, indices]
                        right = matrices[window][kind][subset, right_action, indices]
                        metrics[(window, kind)] = _predicted_metrics(left, right, valid_beta)
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
                c0_cos, c0_f, c0_b = metrics[("half0", "circuit")]
                c1_cos, c1_f, c1_b = metrics[("half1", "circuit")]
                t0_cos, t0_f, t0_b = metrics[("half0", "task")]
                t1_cos, t1_f, t1_b = metrics[("half1", "task")]
                mask = (
                    material & (beta.abs() >= .25) & (beta.abs() <= 4)
                    & (c0_cos >= .90) & (c0_f <= .35) & (c0_b <= .35)
                    & (c1_cos >= .80) & (c1_f <= .50) & (c1_b <= .50)
                    & (t0_cos >= .70) & (t0_f <= .65) & (t0_b <= .65)
                    & (t1_cos >= .70) & (t1_f <= .65) & (t1_b <= .65))
                quality = torch.stack((
                    c0_cos - .90, .35 - c0_f, .35 - c0_b,
                    c1_cos - .80, .50 - c1_f, .50 - c1_b,
                    t0_cos - .70, .65 - t0_f, .65 - t0_b,
                    t1_cos - .70, .65 - t1_f, .65 - t1_b,
                )).amin(0)
                for left_term, right_term in mask.nonzero(as_tuple=False).tolist():
                    candidate = {
                        "subset": subset, "subset_index": r513.SELECTED_SUBSETS[subset],
                        "subset_name": r511.SUBSET_NAMES[r513.SELECTED_SUBSETS[subset]],
                        "relation": relation, "relation_name": r513.RELATION_NAMES[relation],
                        "site": site, "left_term": left_term, "right_term": right_term,
                        "left_term_name": names[left_term], "right_term_name": names[right_term],
                        "beta_left_from_right": float(beta[left_term, right_term]),
                    }
                    if retain:
                        candidate.update(_single_pair_metrics(matrices, candidate))
                    candidates.append(candidate)
                take = min(3, quality.numel())
                values, flat = torch.topk(quality.flatten().nan_to_num(nan=-math.inf), take)
                for value, index in zip(values.tolist(), flat.tolist()):
                    left_term, right_term = divmod(index, len(names))
                    top.append({
                        "name": pair_name(subset, relation, site, left_term, right_term),
                        "quality_margin": value,
                        "passes": bool(mask[left_term, right_term]),
                    })
    top.sort(key=lambda row: row["quality_margin"], reverse=True)
    return candidates, {
        "pairs_tested": PAIR_COUNT,
        "candidate_count": len(candidates),
        "small_relation": bool(1 <= len(candidates) <= MAX_CANDIDATES),
        "material_nodes": len(material_nodes),
        "top_screens": top[:20],
    }


def permutation_control_counts(matrices: dict) -> list[int]:
    counts = []
    dimensions = matrices["half0"]["circuit"].shape[-1]
    for seed in CONTROL_SEEDS:
        control = {
            window: {"task": values["task"], "circuit": values["circuit"].clone()}
            for window, values in matrices.items()
        }
        for action in range(1, N_ACTIONS):
            generator = torch.Generator().manual_seed(seed * 10 + action)
            order = torch.randperm(dimensions, generator=generator)
            for window in control:
                control[window]["circuit"][:, action] = \
                    control[window]["circuit"][:, action, :, order]
        _pairs, summary = discover_pairs(control, retain=False)
        counts.append(summary["candidate_count"])
    return counts


def planted_recovery_suite() -> dict:
    cases = []
    for case_index, seed in enumerate(PLANTED_SEEDS):
        generator = torch.Generator().manual_seed(seed)
        matrices = {}
        for window in ("half0", "half1", "pooled"):
            matrices[window] = {
                "circuit": .002 * torch.randn(
                    N_SUBSETS, N_ACTIONS, N_TERMS, 32,
                    generator=generator, dtype=torch.float64),
                "task": .002 * torch.randn(
                    N_SUBSETS, N_ACTIONS, N_TERMS, 4,
                    generator=generator, dtype=torch.float64),
            }
        subset = case_index % N_SUBSETS
        relation = case_index % len(r513.RELATION_ACTIONS)
        site = SITE_NAMES[case_index % 2]
        width, offset = len(SITE_TERMS[site]), SITE_OFFSETS[site]
        left_term = (3 * case_index + 1) % width
        right_term = (5 * case_index + 2) % width
        left_action, right_action = r513.RELATION_ACTIONS[relation]
        beta = (-1.5 if case_index % 2 else 1.75)
        for window in matrices:
            for kind in ("circuit", "task"):
                donor = matrices[window][kind][subset, right_action, offset + right_term]
                matrices[window][kind][subset, left_action, offset + left_term] = beta * donor
        pairs, _summary = discover_pairs(matrices, retain=False)
        expected = (subset, relation, site, left_term, right_term)
        observed = [(p["subset"], p["relation"], p["site"],
                     p["left_term"], p["right_term"]) for p in pairs]
        cases.append({
            "seed": seed, "expected": expected, "observed": observed,
            "holds": observed == [expected],
        })
    return {"cases": cases, "all_exact_unique_recoveries": all(c["holds"] for c in cases)}


def _empty_diagnostics() -> dict:
    row = r513._empty_diagnostics()
    row.update({
        "branch_patches": 0, "branch_patches_expected": 0,
        "branch_patches_exact": False,
        "term_removal_patches": 0, "term_removal_patches_expected": 0,
        "term_removal_patches_exact": False,
        "maximum_term_patch_capture_error": 0.0,
    })
    return row


def _empty_collection(documents: int, circuit_count: int, bounds: tuple) -> dict:
    return {
        "bounds": bounds,
        "intact_task": torch.zeros(
            N_ACTIONS, documents, len(parent.TASK_CELLS), dtype=torch.float64),
        "removal_task": torch.zeros(
            N_SUBSETS, N_ACTIONS, N_TERMS, documents, len(parent.TASK_CELLS),
            dtype=torch.float64),
        "task_counts": torch.zeros(
            documents, len(parent.TASK_CELLS), dtype=torch.float64),
        "base_task": torch.zeros(
            documents, len(parent.TASK_CELLS), dtype=torch.float64),
        "intact_circuit_sums": torch.zeros(
            N_ACTIONS, 2, 2, circuit_count, dtype=torch.float64),
        "removal_circuit_sums": torch.zeros(
            N_SUBSETS, N_ACTIONS, N_TERMS, 2, 2, circuit_count,
            dtype=torch.float64),
        "circuit_counts": torch.zeros(2, 2, circuit_count, dtype=torch.float64),
        "source_gram": {
            window: torch.zeros(
                r513.N_LOCAL_NODES, r513.N_LOCAL_NODES, dtype=torch.float64)
            for window in ("half0", "half1", "pooled")
        },
        "diagnostics": _empty_diagnostics(),
    }


@torch.no_grad()
def collect_removals(model, rows, task_masks, circuit_masks, circuit_tags,
                     scales, bounds):
    lo, hi, split = bounds
    documents = hi - lo
    data = _empty_collection(documents, len(circuit_tags), bounds)
    data["circuit_tags"] = tuple(circuit_tags)
    diagnostics = data["diagnostics"]
    device = next(model.parameters()).device
    mlp10 = model.transformer.h[parent.TARGET].mlp

    for start in range(lo, hi, parent.BATCH):
        stop, local = start + parent.BATCH, start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in parent.TASK_CELLS}
        copy_mask = masks["all_positive"].to(device)
        half = "half0" if start < split else "half1"

        direct_logits, _, direct_diag, _ = parent._forward(
            model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        r511.r510.r509._update_diagnostics(diagnostics, direct_diag)
        absent_logits, absent, absent_diag, _ = r511._captured_forward(
            model, tokens, scales, action="P", absent=True)
        diagnostics["calls"]["analytical"] += 1
        diagnostics["hooks"] += 1
        r511.r510.r509._update_diagnostics(diagnostics, absent_diag)
        data["base_task"][local:local + len(batch_rows)] = parent._task_sums(
            parent._nll(absent_logits, batch_rows).detach().cpu().unsqueeze(0), masks)[0]

        intact_nll, removal_nll, source_vectors = [], [], [None] * r513.N_LOCAL_NODES
        for action, source in enumerate(parent.SOURCES):
            current_result, current_consumer = r513.factor_consumer_call(
                model, lambda source=source: r511._captured_forward(
                    model, tokens, scales, action=source))
            logits, current, current_diag, _ = current_result
            diagnostics["calls"]["analytical"] += 1
            diagnostics["hooks"] += 1
            diagnostics["factor_consumer_captures"] += 1
            r511.r510.r509._update_diagnostics(diagnostics, current_diag)
            parent._score_delta_closure(diagnostics, current, absent)
            if source == "N":
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"],
                    float((logits.float() - direct_logits.float()).abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    r511._relative_squared(direct_logits, logits))
            branches, branch_diag = r511.deployed_branches(mlp10, absent, current)
            diagnostics["four_corner_replays"] += 1
            r511._update_branch_diagnostics(diagnostics, branch_diag)
            current_nll = parent._nll(logits, batch_rows).detach().cpu()
            intact_nll.append(current_nll)
            data["intact_task"][action, local:local + len(batch_rows)] = \
                parent._task_sums(current_nll.unsqueeze(0), masks)[0]

            action_removals = []
            for selected_subset, subset_index in enumerate(r513.SELECTED_SUBSETS):
                delta10 = r511.subset_output(branches, subset_index)
                replacement10 = current["deployed_write"].float() - delta10
                removed_result, removed_consumer = r513.factor_consumer_call(
                    model, lambda source=source, replacement10=replacement10: (
                        parent.score_parent.run_forward(
                            model, tokens, action=source, scales=scales,
                            patch_writes={
                                "m10": replacement10.to(current["deployed_write"].dtype)
                            })))
                _removed_logits, _removed_captures, patch_diag, patch_audit = removed_result
                diagnostics["calls"]["analytical"] += 1
                diagnostics["factor_consumer_captures"] += 1
                diagnostics["branch_patches"] += patch_audit["patches"]
                branch_rms = patch_diag["patch_rms_max"]
                diagnostics["zero_term_edits"] += int(branch_rms <= 0)
                if branch_rms > 0:
                    diagnostics["minimum_nonzero_term_edit_rms"] = min(
                        diagnostics["minimum_nonzero_term_edit_rms"], branch_rms)
                terms, term_diag = r513.exact_terms(
                    model, removed_consumer, current_consumer)
                diagnostics["attention_corner_evaluations"] += 32
                diagnostics["mlp_corner_evaluations"] += 4
                r513._update_exact_diagnostics(diagnostics, term_diag)
                node = r513.local_node(action, selected_subset)
                source_vectors[node] = delta10[copy_mask].reshape(-1).float().cpu()

                term_nll = []
                for term, term_tensor in enumerate(terms):
                    site = "a11" if term < SITE_OFFSETS["m11"] else "m11"
                    requested = current_consumer[site].float() - term_tensor.float()
                    patched_logits, captures, term_patch_diag, term_patch_audit = \
                        parent.score_parent.run_forward(
                            model, tokens, action=source, scales=scales,
                            patch_writes={site: requested.to(current_consumer[site].dtype)},
                            capture_keys=(site,))
                    diagnostics["calls"]["analytical"] += 1
                    diagnostics["term_removal_patches"] += term_patch_audit["patches"]
                    diagnostics["maximum_term_patch_capture_error"] = max(
                        diagnostics["maximum_term_patch_capture_error"],
                        float((captures[site] - requested.to(captures[site].dtype))
                              .float().abs().max()))
                    edit_rms = term_patch_diag["patch_rms_max"]
                    diagnostics["zero_term_edits"] += int(edit_rms <= 0)
                    if edit_rms > 0:
                        diagnostics["minimum_nonzero_term_edit_rms"] = min(
                            diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
                    term_nll.append(parent._nll(patched_logits, batch_rows).detach().cpu())
                term_nll = torch.stack(term_nll)
                action_removals.append(term_nll)
                data["removal_task"][selected_subset, action, :,
                    local:local + len(batch_rows)] = parent._task_sums(term_nll, masks)
            removal_nll.append(torch.stack(action_removals))

        source_values = torch.stack(source_vectors).float()
        gram = (source_values @ source_values.T).double()
        data["source_gram"][half] += gram
        data["source_gram"]["pooled"] += gram
        data["task_counts"][local:local + len(batch_rows)] = torch.stack(
            [masks[cell].sum(1).double() for cell in parent.TASK_CELLS], -1)
        matrix, observed = parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        data["circuit_counts"] += observed
        intact_stack = torch.stack(intact_nll)
        data["intact_circuit_sums"] += torch.matmul(
            intact_stack.view(N_ACTIONS, -1).double(), matrix.T,
        ).view(N_ACTIONS, 2, 2, len(circuit_tags))
        removal_stack = torch.stack(removal_nll)
        data["removal_circuit_sums"] += torch.matmul(
            removal_stack.view(N_ACTIONS * N_SUBSETS * N_TERMS, -1).double(), matrix.T,
        ).view(N_ACTIONS, N_SUBSETS, N_TERMS, 2, 2, len(circuit_tags)).permute(
            1, 0, 2, 3, 4, 5)

    batches = documents // parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches,
        "analytical": batches * (1 + N_ACTIONS + N_ACTIONS * N_SUBSETS * (1 + N_TERMS)),
    }
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["hooks_expected"] = batches * (1 + N_ACTIONS)
    diagnostics["hooks_exact"] = diagnostics["hooks"] == diagnostics["hooks_expected"]
    diagnostics["four_corner_replays_expected"] = batches * N_ACTIONS
    diagnostics["four_corner_replays_exact"] = (
        diagnostics["four_corner_replays"] == diagnostics["four_corner_replays_expected"])
    diagnostics["factor_consumer_captures_expected"] = batches * N_ACTIONS * (1 + N_SUBSETS)
    diagnostics["factor_consumer_captures_exact"] = (
        diagnostics["factor_consumer_captures"]
        == diagnostics["factor_consumer_captures_expected"])
    diagnostics["attention_corner_evaluations_expected"] = batches * N_ACTIONS * N_SUBSETS * 32
    diagnostics["attention_corner_evaluations_exact"] = (
        diagnostics["attention_corner_evaluations"]
        == diagnostics["attention_corner_evaluations_expected"])
    diagnostics["mlp_corner_evaluations_expected"] = batches * N_ACTIONS * N_SUBSETS * 4
    diagnostics["mlp_corner_evaluations_exact"] = (
        diagnostics["mlp_corner_evaluations"]
        == diagnostics["mlp_corner_evaluations_expected"])
    diagnostics["branch_patches_expected"] = batches * N_ACTIONS * N_SUBSETS
    diagnostics["branch_patches_exact"] = (
        diagnostics["branch_patches"] == diagnostics["branch_patches_expected"])
    diagnostics["term_removal_patches_expected"] = batches * N_ACTIONS * N_SUBSETS * N_TERMS
    diagnostics["term_removal_patches_exact"] = (
        diagnostics["term_removal_patches"] == diagnostics["term_removal_patches_expected"])
    diagnostics["patches"] = diagnostics["branch_patches"] + diagnostics["term_removal_patches"]
    diagnostics["patches_expected"] = (
        diagnostics["branch_patches_expected"] + diagnostics["term_removal_patches_expected"])
    diagnostics["patches_exact"] = diagnostics["branch_patches_exact"] \
        and diagnostics["term_removal_patches_exact"]
    return data


def removal_instrument(collection: dict) -> bool:
    d = collection["diagnostics"]
    return bool(
        d["calls_exact"] and d["hooks_exact"] and d["four_corner_replays_exact"]
        and d["factor_consumer_captures_exact"]
        and d["attention_corner_evaluations_exact"] and d["mlp_corner_evaluations_exact"]
        and d["patches_exact"] and d["zero_term_edits"] == 0
        and d["minimum_nonzero_term_edit_rms"] > 0
        and d["maximum_term_patch_capture_error"] == 0.0
        and d["removed_attention_corner_replay_max_abs"] == 0.0
        and d["intact_attention_corner_replay_max_abs"] == 0.0
        and d["attention_mobius_relative_squared"] <= 1e-10
        and d["attention_numerical_remainder_rms_ratio"] <= .01
        and d["mlp_deployed_branch_sum_relative_squared"] <= 1e-12
        and d["native_replay_logit_max_abs"] == 0.0
        and d["native_replay_relative_squared"] <= 1e-12)


def source_relation_reproduction(collection: dict) -> tuple[list[str], dict]:
    return r513.reproduce_source_relations({"statistics": {
        "source_gram": collection["source_gram"]}})


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(R514_RESULT.read_text())
    if not (
        result.get("pred_a_exact_live_identifiable_joint_program_instrument") is True
        and result.get("pred_b_constrained_multi_term_discovery") is False
        and result.get("analysis", {}).get("discovery_summary", {})
            .get("real", {}).get("counts", {}).get("accepted") == 0
        and result.get("next_step")
        == "preregister_task_conditioned_nonlinear_reader_of_exact_terms_with_heldout_circuit_outcomes"
    ):
        raise RuntimeError("rung514 registered failure route changed")
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        r514.validate_inputs()
    if len(discovery_tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("62-circuit discovery/confirmation partition changed")
    return rows, task_masks, circuit_masks, scales, list(discovery_tags), \
        list(validation_tags), {
            **metadata,
            "rung514_result_sha256": sha256(R514_RESULT),
            "rung514_bundle_sha256": sha256(R514_BUNDLE),
            "documents": {"discovery": list(DISCOVERY), "unused": [748, 752],
                          "confirmation": list(CONFIRMATION)},
            "circuits": {"discovery": list(discovery_tags),
                         "confirmation": list(validation_tags)},
            "nodes": N_SUBSETS * N_ACTIONS * N_TERMS,
            "fixed_pairs": PAIR_COUNT,
        }


def confirm_pairs(matrices: dict, candidates: list[dict]) -> tuple[list[dict], dict]:
    passing, checks = [], {}
    for candidate in candidates:
        metrics = _single_pair_metrics(matrices, candidate, confirmation=True)
        name = pair_name(candidate["subset"], candidate["relation"], candidate["site"],
                         candidate["left_term"], candidate["right_term"])
        checks[name] = metrics
        if metrics["holds"]:
            passing.append(candidate)
    return passing, checks


def _empty_substitution(documents: int, circuit_count: int,
                        bounds: tuple, candidates: list[dict]) -> dict:
    directions = []
    for candidate_index, candidate in enumerate(candidates):
        left_action, right_action = r513.RELATION_ACTIONS[candidate["relation"]]
        directions.extend((
            {"candidate": candidate_index, "target_action": left_action,
             "donor_action": right_action, "scale": candidate["beta_left_from_right"],
             "side": "left_from_right"},
            {"candidate": candidate_index, "target_action": right_action,
             "donor_action": left_action, "scale": 1.0 / candidate["beta_left_from_right"],
             "side": "right_from_left"},
        ))
    diagnostics = _empty_diagnostics()
    diagnostics.update({
        "substitution_patches": 0, "substitution_patches_expected": 0,
        "substitution_patches_exact": False,
    })
    return {
        "bounds": bounds, "directions": directions,
        "task": torch.zeros(
            len(directions), documents, len(parent.TASK_CELLS), dtype=torch.float64),
        "task_counts": torch.zeros(
            documents, len(parent.TASK_CELLS), dtype=torch.float64),
        "circuit_sums": torch.zeros(
            len(directions), 2, 2, circuit_count, dtype=torch.float64),
        "circuit_counts": torch.zeros(2, 2, circuit_count, dtype=torch.float64),
        "diagnostics": diagnostics,
    }


@torch.no_grad()
def collect_substitutions(model, rows, task_masks, circuit_masks, circuit_tags,
                          scales, bounds, candidates):
    lo, hi, _split = bounds
    data = _empty_substitution(hi - lo, len(circuit_tags), bounds, candidates)
    diagnostics = data["diagnostics"]
    device = next(model.parameters()).device
    mlp10 = model.transformer.h[parent.TARGET].mlp
    needed_terms = {}
    for candidate in candidates:
        left_action, right_action = r513.RELATION_ACTIONS[candidate["relation"]]
        offset = SITE_OFFSETS[candidate["site"]]
        needed_terms.setdefault((left_action, candidate["subset"]), set()).add(
            offset + candidate["left_term"])
        needed_terms.setdefault((right_action, candidate["subset"]), set()).add(
            offset + candidate["right_term"])
    for start in range(lo, hi, parent.BATCH):
        stop, local = start + parent.BATCH, start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in parent.TASK_CELLS}
        direct_logits, _, direct_diag, _ = parent._forward(
            model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        r511.r510.r509._update_diagnostics(diagnostics, direct_diag)
        _absent_logits, absent, absent_diag, _ = r511._captured_forward(
            model, tokens, scales, action="P", absent=True)
        diagnostics["calls"]["analytical"] += 1
        diagnostics["hooks"] += 1
        r511.r510.r509._update_diagnostics(diagnostics, absent_diag)
        current_consumers, exact = {}, {}
        for action, source in enumerate(parent.SOURCES):
            current_result, current_consumer = r513.factor_consumer_call(
                model, lambda source=source: r511._captured_forward(
                    model, tokens, scales, action=source))
            logits, current, current_diag, _ = current_result
            diagnostics["calls"]["analytical"] += 1
            diagnostics["hooks"] += 1
            diagnostics["factor_consumer_captures"] += 1
            r511.r510.r509._update_diagnostics(diagnostics, current_diag)
            parent._score_delta_closure(diagnostics, current, absent)
            if source == "N":
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"],
                    float((logits.float() - direct_logits.float()).abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    r511._relative_squared(direct_logits, logits))
            branches, branch_diag = r511.deployed_branches(mlp10, absent, current)
            diagnostics["four_corner_replays"] += 1
            r511._update_branch_diagnostics(diagnostics, branch_diag)
            current_consumers[action] = current_consumer
            for selected_subset, subset_index in enumerate(r513.SELECTED_SUBSETS):
                delta10 = r511.subset_output(branches, subset_index)
                replacement10 = current["deployed_write"].float() - delta10
                removed_result, removed_consumer = r513.factor_consumer_call(
                    model, lambda source=source, replacement10=replacement10: (
                        parent.score_parent.run_forward(
                            model, tokens, action=source, scales=scales,
                            patch_writes={
                                "m10": replacement10.to(current["deployed_write"].dtype)
                            })))
                _logits, _captures, patch_diag, patch_audit = removed_result
                diagnostics["calls"]["analytical"] += 1
                diagnostics["factor_consumer_captures"] += 1
                diagnostics["branch_patches"] += patch_audit["patches"]
                branch_rms = patch_diag["patch_rms_max"]
                diagnostics["zero_term_edits"] += int(branch_rms <= 0)
                if branch_rms > 0:
                    diagnostics["minimum_nonzero_term_edit_rms"] = min(
                        diagnostics["minimum_nonzero_term_edit_rms"], branch_rms)
                terms, term_diag = r513.exact_terms(
                    model, removed_consumer, current_consumer)
                diagnostics["attention_corner_evaluations"] += 32
                diagnostics["mlp_corner_evaluations"] += 4
                r513._update_exact_diagnostics(diagnostics, term_diag)
                key = (action, selected_subset)
                exact[key] = {term: terms[term] for term in needed_terms.get(key, ())}
                del terms

        nll_rows = []
        for direction in data["directions"]:
            candidate = candidates[direction["candidate"]]
            target_action = direction["target_action"]
            donor_action = direction["donor_action"]
            site = candidate["site"]
            offset = SITE_OFFSETS[site]
            donor_local_term = (candidate["right_term"]
                                if direction["side"] == "left_from_right"
                                else candidate["left_term"])
            donor = exact[(donor_action, candidate["subset"])][offset + donor_local_term]
            current_write = current_consumers[target_action][site]
            requested = current_write.float() - direction["scale"] * donor.float()
            logits, captures, patch_diag, patch_audit = parent.score_parent.run_forward(
                model, tokens, action=parent.SOURCES[target_action], scales=scales,
                patch_writes={site: requested.to(current_write.dtype)}, capture_keys=(site,))
            diagnostics["calls"]["analytical"] += 1
            diagnostics["substitution_patches"] += patch_audit["patches"]
            diagnostics["maximum_term_patch_capture_error"] = max(
                diagnostics["maximum_term_patch_capture_error"],
                float((captures[site] - requested.to(captures[site].dtype))
                      .float().abs().max()))
            edit_rms = patch_diag["patch_rms_max"]
            diagnostics["zero_term_edits"] += int(edit_rms <= 0)
            if edit_rms > 0:
                diagnostics["minimum_nonzero_term_edit_rms"] = min(
                    diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
            nll_rows.append(parent._nll(logits, batch_rows).detach().cpu())
        nll_stack = torch.stack(nll_rows)
        data["task"][:, local:local + len(batch_rows)] = parent._task_sums(nll_stack, masks)
        data["task_counts"][local:local + len(batch_rows)] = torch.stack(
            [masks[cell].sum(1).double() for cell in parent.TASK_CELLS], -1)
        matrix, observed = parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        data["circuit_counts"] += observed
        data["circuit_sums"] += torch.matmul(
            nll_stack.view(len(data["directions"]), -1).double(), matrix.T,
        ).view(len(data["directions"]), 2, 2, len(circuit_tags))

    batches = (hi - lo) // parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches,
        "analytical": batches * (1 + N_ACTIONS + N_ACTIONS * N_SUBSETS
                                  + len(data["directions"])),
    }
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["hooks_expected"] = batches * (1 + N_ACTIONS)
    diagnostics["hooks_exact"] = diagnostics["hooks"] == diagnostics["hooks_expected"]
    diagnostics["four_corner_replays_expected"] = batches * N_ACTIONS
    diagnostics["four_corner_replays_exact"] = (
        diagnostics["four_corner_replays"] == diagnostics["four_corner_replays_expected"])
    diagnostics["factor_consumer_captures_expected"] = batches * N_ACTIONS * (1 + N_SUBSETS)
    diagnostics["factor_consumer_captures_exact"] = (
        diagnostics["factor_consumer_captures"]
        == diagnostics["factor_consumer_captures_expected"])
    diagnostics["attention_corner_evaluations_expected"] = batches * N_ACTIONS * N_SUBSETS * 32
    diagnostics["attention_corner_evaluations_exact"] = (
        diagnostics["attention_corner_evaluations"]
        == diagnostics["attention_corner_evaluations_expected"])
    diagnostics["mlp_corner_evaluations_expected"] = batches * N_ACTIONS * N_SUBSETS * 4
    diagnostics["mlp_corner_evaluations_exact"] = (
        diagnostics["mlp_corner_evaluations"]
        == diagnostics["mlp_corner_evaluations_expected"])
    diagnostics["branch_patches_expected"] = batches * N_ACTIONS * N_SUBSETS
    diagnostics["branch_patches_exact"] = (
        diagnostics["branch_patches"] == diagnostics["branch_patches_expected"])
    diagnostics["substitution_patches_expected"] = batches * len(data["directions"])
    diagnostics["substitution_patches_exact"] = (
        diagnostics["substitution_patches"] == diagnostics["substitution_patches_expected"])
    diagnostics["patches"] = diagnostics["branch_patches"] + diagnostics["substitution_patches"]
    diagnostics["patches_expected"] = (
        diagnostics["branch_patches_expected"] + diagnostics["substitution_patches_expected"])
    diagnostics["patches_exact"] = diagnostics["branch_patches_exact"] \
        and diagnostics["substitution_patches_exact"]
    return data


def substitution_instrument(collection: dict) -> bool:
    d = collection["diagnostics"]
    return bool(
        d["calls_exact"] and d["hooks_exact"] and d["four_corner_replays_exact"]
        and d["factor_consumer_captures_exact"]
        and d["attention_corner_evaluations_exact"] and d["mlp_corner_evaluations_exact"]
        and d["patches_exact"] and d["zero_term_edits"] == 0
        and d["minimum_nonzero_term_edit_rms"] > 0
        and d["maximum_term_patch_capture_error"] == 0.0
        and d["removed_attention_corner_replay_max_abs"] == 0.0
        and d["intact_attention_corner_replay_max_abs"] == 0.0
        and d["attention_numerical_remainder_rms_ratio"] <= .01
        and d["mlp_deployed_branch_sum_relative_squared"] <= 1e-12
        and d["native_replay_logit_max_abs"] == 0.0
        and d["native_replay_relative_squared"] <= 1e-12)


def _substitution_task_effect(substitutions: dict, exact: dict,
                              direction: int, window: str) -> torch.Tensor:
    candidate_index = substitutions["directions"][direction]["candidate"]
    target_action = substitutions["directions"][direction]["target_action"]
    selected = _window_document_slice(substitutions, window)
    numerator = (
        substitutions["task"][direction, selected]
        - exact["intact_task"][target_action, selected]
    ).sum(0)
    denominator = substitutions["task_counts"][selected].sum(0).clamp_min(1)
    _ = candidate_index
    return (numerator / denominator)[list(COPY_INDICES)].double()


def _substitution_circuit_effect(substitutions: dict, exact: dict,
                                 direction: int, window: str) -> torch.Tensor:
    target_action = substitutions["directions"][direction]["target_action"]
    if window == "pooled":
        changed = substitutions["circuit_sums"][direction].sum(0)
        intact = exact["intact_circuit_sums"][target_action].sum(0)
        counts = substitutions["circuit_counts"].sum(0)
    else:
        half = {"half0": 0, "half1": 1}[window]
        changed = substitutions["circuit_sums"][direction, half]
        intact = exact["intact_circuit_sums"][target_action, half]
        counts = substitutions["circuit_counts"][half]
    effect = (changed - intact) / counts.clamp_min(1)
    return (effect[0] - effect[1]).double()


def _off_target_difference(substitutions: dict, exact: dict, candidate: dict,
                           direction: int, window: str) -> float:
    target_action = substitutions["directions"][direction]["target_action"]
    target_term = (candidate["left_term"]
                   if substitutions["directions"][direction]["side"] == "left_from_right"
                   else candidate["right_term"])
    global_term = SITE_OFFSETS[candidate["site"]] + target_term
    selected = _window_document_slice(substitutions, window)
    index = parent.TASK_CELLS.index("off_target")
    numerator = (
        substitutions["task"][direction, selected, index]
        - exact["removal_task"][candidate["subset"], target_action,
                                 global_term, selected, index]
    ).sum()
    denominator = substitutions["task_counts"][selected, index].sum().clamp_min(1)
    return float(numerator / denominator)


def score_substitutions(substitutions: dict, exact: dict,
                        candidates: list[dict]) -> tuple[list[dict], dict]:
    passing, checks = [], {}
    for candidate_index, candidate in enumerate(candidates):
        row = {"directions": {}, "holds": True}
        for local_direction, side in enumerate(("left_from_right", "right_from_left")):
            direction = 2 * candidate_index + local_direction
            target_action = substitutions["directions"][direction]["target_action"]
            target_local_term = (candidate["left_term"] if local_direction == 0
                                 else candidate["right_term"])
            target_term = SITE_OFFSETS[candidate["site"]] + target_local_term
            side_row = {"windows": {}, "holds": True}
            for window in ("half0", "half1", "pooled"):
                native_task = _task_effect(
                    exact, candidate["subset"], target_action, target_term, window)
                native_circuit = _circuit_effect(
                    exact, candidate["subset"], target_action, target_term, window)
                substituted_task = _substitution_task_effect(
                    substitutions, exact, direction, window)
                substituted_circuit = _substitution_circuit_effect(
                    substitutions, exact, direction, window)
                circuit_cos = float(r510._safe_cosine_rows(
                    native_circuit[None], substituted_circuit[None])[0])
                circuit_res = float(r510._residual_rows(
                    native_circuit[None], substituted_circuit[None])[0])
                task_cos = float(r510._safe_cosine_rows(
                    native_task[None], substituted_task[None])[0])
                task_res = float(r510._residual_rows(
                    native_task[None], substituted_task[None])[0])
                off_target = _off_target_difference(
                    substitutions, exact, candidate, direction, window)
                holds = bool(
                    circuit_cos >= .75 and circuit_res <= .55
                    and task_cos >= .70 and task_res <= .65
                    and abs(off_target) <= .002)
                side_row["windows"][window] = {
                    "circuit_cosine": circuit_cos,
                    "circuit_relative_residual": circuit_res,
                    "task_cosine": task_cos,
                    "task_relative_residual": task_res,
                    "off_target_ce_difference_nat": off_target,
                    "holds": holds,
                }
                if window in ("half0", "half1"):
                    side_row["holds"] &= holds
            row["directions"][side] = side_row
            row["holds"] &= side_row["holds"]
        name = pair_name(candidate["subset"], candidate["relation"], candidate["site"],
                         candidate["left_term"], candidate["right_term"])
        checks[name] = row
        if row["holds"]:
            passing.append(candidate)
    return passing, checks


def quotient_groups(passing: list[dict]) -> list[dict]:
    groups = []
    for (subset, site), rows in itertools.groupby(
        sorted(passing, key=lambda x: (x["subset"], x["site"])),
        key=lambda x: (x["subset"], x["site"]),
    ):
        edges = list(rows)
        adjacency = {}
        beta = {}
        for edge in edges:
            left_action, right_action = r513.RELATION_ACTIONS[edge["relation"]]
            left = (left_action, edge["left_term"])
            right = (right_action, edge["right_term"])
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
            beta[frozenset((left, right))] = edge["beta_left_from_right"]
        visited = set()
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
            complete = all(frozenset((a, b)) in beta for a, b in itertools.combinations(nodes, 2))
            # Cycle pricing is only meaningful for a complete graph; the registered action graph
            # usually yields edges rather than a complete four-node component.
            if complete:
                groups.append({
                    "subset": subset, "site": site,
                    "nodes": [{"action": parent.SOURCES[a], "term": SITE_TERMS[site][t]}
                              for a, t in nodes],
                    "complete_graph": True, "cycle_consistent": len(nodes) <= 2,
                })
    return groups


def _bundle_collection(collection: dict) -> dict:
    return {key: value for key, value in collection.items() if key != "diagnostics"}


def dry_run() -> None:
    planted = planted_recovery_suite()
    assert planted["all_exact_unique_recoveries"]
    assert N_SUBSETS * N_ACTIONS * N_TERMS == 816
    assert PAIR_COUNT == 17460
    assert len(CONTROL_SEEDS) == 16
    assert 2 * 52452 + 1860 + 124 * 16 == 108748
    print(json.dumps({
        "status": "dry_run_passed", "rung": 515,
        "model_loaded": False, "outcomes_opened": False,
        "nodes": 816, "fixed_pairs": PAIR_COUNT,
        "planted_cases": len(planted["cases"]),
        "all_planted_pairs_uniquely_recovered": True,
        "maximum_conditional_forwards": 108748,
    }, indent=2, sort_keys=True))


@torch.no_grad()
def gpu_smoke() -> None:
    planted = planted_recovery_suite()
    rows, task_masks, circuit_masks, scales, discovery_tags, _validation_tags, _metadata = \
        validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    bounds = (500, 504, 502)
    exact = collect_removals(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, bounds)
    calibration = parent._calibration(
        exact["base_task"], exact["intact_task"], exact["task_counts"], bounds)
    candidate = {
        "subset": 0, "subset_index": r513.SELECTED_SUBSETS[0],
        "subset_name": r511.SUBSET_NAMES[r513.SELECTED_SUBSETS[0]],
        "relation": 0, "relation_name": r513.RELATION_NAMES[0],
        "site": "a11", "left_term": 0, "right_term": 1,
        "left_term_name": r513.ATTENTION_TERMS[0],
        "right_term_name": r513.ATTENTION_TERMS[1],
        "beta_left_from_right": 1.0,
    }
    substitutions = collect_substitutions(
        model, rows, task_masks, circuit_masks, discovery_tags,
        scales, bounds, [candidate])
    checks = {
        "weights": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256,
        "planted_recovery": planted["all_exact_unique_recoveries"],
        "removals": removal_instrument(exact),
        "substitutions": substitution_instrument(substitutions),
        "native_calibration_semantics": (
            .9 <= calibration["pooled"]["N"]["recovery_vs_native"] <= 1.1),
        "all_816_term_removals": exact["diagnostics"]["term_removal_patches"] == 816,
        "all_24_branch_patches": exact["diagnostics"]["branch_patches"] == 24,
        "both_substitution_patches": (
            substitutions["diagnostics"]["substitution_patches"] == 2),
    }
    passed = all(checks.values())
    print(json.dumps({
        "status": "smoke_passed" if passed else "smoke_failed", "rung": 515,
        "scientific_outcomes_retained": False, "checks": checks,
        "planted_recovery": planted,
        "removal_diagnostics": exact["diagnostics"],
        "substitution_diagnostics": substitutions["diagnostics"],
        "smoke_calibration": calibration,
        "full_forwards": sum(exact["diagnostics"]["calls"].values())
                         + sum(substitutions["diagnostics"]["calls"].values()),
        "backwards": 0,
    }, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(
            f"rung515 smoke failed: "
            f"{sorted(name for name, value in checks.items() if not value)}")


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        dry_run()
        return
    if os.environ.get("BQLIB_GPU_SMOKE") == "1" or "--gpu-smoke" in sys.argv:
        gpu_smoke()
        return
    started = time.time()
    planted = planted_recovery_suite()
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung515 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    collections = {}
    collections["discovery"] = collect_removals(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, DISCOVERY)
    discovery_calibration = parent._calibration(
        collections["discovery"]["base_task"], collections["discovery"]["intact_task"],
        collections["discovery"]["task_counts"], DISCOVERY)
    discovery_calibration_ok = parent.state_parent.calibration_holds(discovery_calibration)
    source_relations, source_checks = source_relation_reproduction(collections["discovery"])
    source_relations_ok = source_relations == list(r513.SOURCE_RELATION_NAMES)
    discovery_instrument = removal_instrument(collections["discovery"])
    discovery_matrices = response_matrices(collections["discovery"])
    candidates, discovery_summary = discover_pairs(discovery_matrices)
    control_counts = permutation_control_counts(discovery_matrices)
    control_max = max(control_counts, default=0)
    discovery_count_ok = bool(
        1 <= len(candidates) <= MAX_CANDIDATES and len(candidates) > control_max)

    confirmed, confirmation_checks = [], {}
    confirmation_calibration, confirmation_calibration_ok = {}, False
    confirmation_instrument = False
    if (planted["all_exact_unique_recoveries"] and discovery_calibration_ok
            and source_relations_ok and discovery_instrument and discovery_count_ok):
        collections["confirmation"] = collect_removals(
            model, rows, task_masks, circuit_masks, validation_tags, scales, CONFIRMATION)
        confirmation_calibration = parent._calibration(
            collections["confirmation"]["base_task"],
            collections["confirmation"]["intact_task"],
            collections["confirmation"]["task_counts"], CONFIRMATION)
        confirmation_calibration_ok = parent.state_parent.calibration_holds(
            confirmation_calibration)
        confirmation_instrument = removal_instrument(collections["confirmation"])
        confirmation_matrices = response_matrices(collections["confirmation"])
        confirmed, confirmation_checks = confirm_pairs(
            confirmation_matrices, candidates)

    physical_pairs, physical_checks, groups = [], {}, []
    physical_instrument_ok = False
    if confirmation_calibration_ok and confirmation_instrument and confirmed:
        collections["substitutions"] = collect_substitutions(
            model, rows, task_masks, circuit_masks, validation_tags,
            scales, CONFIRMATION, confirmed)
        physical_instrument_ok = substitution_instrument(collections["substitutions"])
        physical_pairs, physical_checks = score_substitutions(
            collections["substitutions"], collections["confirmation"], confirmed)
        groups = quotient_groups(physical_pairs)

    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and planted["all_exact_unique_recoveries"]
        and discovery_calibration_ok and source_relations_ok and discovery_instrument
        and ("confirmation" not in collections
             or (confirmation_calibration_ok and confirmation_instrument))
        and ("substitutions" not in collections or physical_instrument_ok))
    pred_b = bool(pred_a and discovery_count_ok)
    pred_c = bool(pred_b and confirmation_calibration_ok and confirmed)
    pred_d = bool(pred_c and physical_pairs)
    mapping_counts = {}
    for edge in physical_pairs:
        mapping = (edge["site"], edge["left_term"], edge["right_term"], edge["relation"])
        mapping_counts[mapping] = mapping_counts.get(mapping, 0) + 1
    pred_e = bool(pred_d and (
        any(edge["left_term"] != edge["right_term"] for edge in physical_pairs)
        or any(count >= 2 for count in mapping_counts.values())))
    strong_null = not (pred_a and pred_b and pred_c and pred_d)

    if not pred_a:
        next_step = "repair_exact_finite_downstream_term_instrument_only"
    elif not pred_b and len(candidates) == 0:
        next_step = "leave_mlp10_consumer_descent_for_task_defined_state_transition_or_new_gap"
    elif not pred_b:
        next_step = "strengthen_independent_circuit_observations_without_best_k_selection"
    elif not pred_c:
        next_step = "identify_heldout_circuit_outcomes_that_break_discovery_pairs"
    elif not pred_d:
        next_step = "localize_first_later_site_that_separates_finite_removal_pairs"
    elif not pred_e:
        next_step = "validate_same_term_action_portability_on_ood_code"
    else:
        next_step = "validate_nontrivial_term_quotient_on_ood_code_then_price_interface"

    bundle_payload = {
        "schema": "rung515_finite_downstream_term_quotient_v1",
        "collections": {name: _bundle_collection(collection)
                        for name, collection in collections.items()},
        "discovery_candidates": candidates,
        "confirmed_pairs": confirmed,
        "physical_pairs": physical_pairs,
        "raw_tokens_logits_hidden_states_or_weights_included": False,
    }
    torch.save(bundle_payload, BUNDLE)
    full_forwards = sum(
        sum(collection["diagnostics"]["calls"].values())
        for collection in collections.values())
    result = {
        "status": "complete", "rung": 515,
        "claim_level": "finite_downstream_pair_until_bidirectional_substitution_passes",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "planted_recovery": planted,
        "calibration": {"discovery": discovery_calibration,
                        "confirmation": confirmation_calibration},
        "calibration_holds": {"discovery": discovery_calibration_ok,
                              "confirmation": confirmation_calibration_ok},
        "source_relation_reproduction": {
            "expected": list(r513.SOURCE_RELATION_NAMES),
            "observed": source_relations, "holds": source_relations_ok,
            "checks": source_checks,
        },
        "diagnostics": {name: collection["diagnostics"]
                        for name, collection in collections.items()},
        "analysis": {
            "discovery_summary": discovery_summary,
            "discovery_candidates": candidates,
            "permutation_control_candidate_counts": control_counts,
            "maximum_control_candidate_count": control_max,
            "confirmation_checks": confirmation_checks,
            "confirmed_pairs": confirmed,
            "physical_checks": physical_checks,
            "physical_pairs": physical_pairs,
            "quotient_groups": groups,
        },
        'pred_a_exact_live_identifiable_finite_downstream_instrument': pred_a,
        'pred_b_small_downstream_relation_beats_controls': pred_b,
        'pred_c_heldout_documents_and_thirty_circuits': pred_c,
        'pred_d_bidirectional_same_site_physical_substitution': pred_d,
        'pred_e_cross_term_grouping_or_branch_reuse': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {
            "path": str(BUNDLE), "sha256": sha256(BUNDLE),
            "bytes": BUNDLE.stat().st_size,
        },
        "execution_price": {
            "full_forwards": full_forwards, "backwards": 0,
            "finite_term_nodes": N_SUBSETS * N_ACTIONS * N_TERMS,
            "fixed_pair_comparisons": PAIR_COUNT,
            "discovery_candidates": len(candidates),
            "confirmed_pairs": len(confirmed), "physical_pairs": len(physical_pairs),
            "maximum_conditional_forwards": 108748,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started, "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 515,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "discovery_candidates": len(candidates),
        "maximum_control_candidate_count": control_max,
        "confirmed_pairs": len(confirmed), "physical_pairs": len(physical_pairs),
        "quotient_groups": len(groups), "execution_price": result["execution_price"],
        "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
