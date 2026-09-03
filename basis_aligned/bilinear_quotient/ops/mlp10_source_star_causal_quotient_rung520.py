#!/usr/bin/env python3
"""RUNG520 -- exact MLP10 source-star causal quotient."""

# BQGATE: EXPERIMENT
# pred_a: exact live 22-source-star intervention instrument
# pred_b: one to sixteen discovery relations beat circuit permutations
# pred_c: at least one relation predicts held-out documents and circuits
# pred_d: at least one relation passes bidirectional physical substitution
# pred_e: at least one physical relation joins different named sources

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
import mlp10_exact_source_pair_causal_split_rung507 as r507
import mlp10_coupled_causal_dictionary_rung509 as r509


PREREG = POLY / "MLP10_SOURCE_STAR_CAUSAL_QUOTIENT_RUNG520_PREREGISTRATION.md"
ADDENDUM = POLY / "MLP10_SOURCE_STAR_CAUSAL_QUOTIENT_RUNG520_PREFLIGHT_ADDENDUM.md"
R519_RESULT = ROOT / "mlp0_one_circuit_interaction_atlas_rung519_results.json"
R519_SOURCE = ROOT / "ops/mlp0_one_circuit_interaction_atlas_rung519.py"
R519_PREREG = POLY / "MLP0_ONE_CIRCUIT_INTERACTION_ATLAS_RUNG519_PREREGISTRATION.md"
R510_RESULT = ROOT / "mlp10_observable_predictive_state_quotient_rung510_results.json"
R510_BUNDLE = ROOT / "mlp10_observable_predictive_state_quotient_rung510_bundle.pt"
R510_SOURCE = ROOT / "ops/mlp10_observable_predictive_state_quotient_rung510.py"
R510_PREREG = POLY / "MLP10_OBSERVABLE_PREDICTIVE_STATE_QUOTIENT_RUNG510_PREREGISTRATION.md"
R507_RESULT = ROOT / "mlp10_exact_source_pair_causal_split_rung507_results.json"
R507_BUNDLE = ROOT / "mlp10_exact_source_pair_causal_split_rung507_bundle.pt"
R507_SOURCE = ROOT / "ops/mlp10_exact_source_pair_causal_split_rung507.py"
R507_PREREG = POLY / "MLP10_EXACT_SOURCE_PAIR_CAUSAL_SPLIT_RUNG507_PREREGISTRATION.md"
OUT = ROOT / "mlp10_source_star_causal_quotient_rung520_results.json"
BUNDLE = ROOT / "mlp10_source_star_causal_quotient_rung520_bundle.pt"

HASHES = {
    PREREG: "25b1adb8d22bd8111d15b66bac1f802ab668a80ff8db52b5a07ae6ec2039b0fa",
    ADDENDUM: "64aecacd346737404bb0455a0be76dcf814ad824bc5d2545278e7f16396cd4c1",
    R519_RESULT: "3eb5188fa65a746a987d4bee851aaed46b08d7ba905b596dd091d01bd29386f6",
    R519_SOURCE: "0f06c7a41ad4f308a647422ad8aa0e545d90a1fcc9a4a41b6bbeabb1fbd6ec0a",
    R519_PREREG: "cce9e3b25a0633e94bb32d89e2aa6e3f587c88949be9b1c78f0072ca19f14d55",
    R510_RESULT: "16d100e7b92152fc70939b000934699882605c30c513c570f6c519b80f943177",
    R510_BUNDLE: "a8832624c94e3e9aa491d26290e55a14f94aa103eb7cddc3df3a0e1b34c3eed7",
    R510_SOURCE: "7901aa5d9c7c39bf5666e0f081bfe08047f23c73eec08b12508c601def7b967a",
    R510_PREREG: "e344760333af378ea5604c211c259a27d9ff030b60bad8054ca962d465f46055",
    R507_RESULT: "f3ce5669bb86e5e4a36e4fa44a2c2ff488bc3806ab86380ad359c0c6310fe57c",
    R507_BUNDLE: "bc72fcd9e1b7be5be3219ffd1284d8aa23c9c89778ca8a3e02faf8d0ba889dcd",
    R507_SOURCE: "4bb6fbf9a12cbdae05162cff86abb84d31c834dfa2f7a1d92d75f5092d2e8035",
    R507_PREREG: "4bfd001804fde4ab0852172c5fe5242fb523258f1e60cd9aa14c26a94428a8e9",
}

DISCOVERY = (500, 748, 624)
CONFIRMATION = (752, 1000, 876)
CONTROL_SEEDS = tuple(range(520100, 520116))
MAX_CANDIDATES = 16
N_ACTIONS = len(r507.SOURCES)
N_STARS = len(r507.NAMED_SOURCES)
N_NODES = N_ACTIONS * N_STARS
STAR_INDICES = tuple(
    tuple(index for index, pair in enumerate(r507.SOURCE_PAIRS) if source in pair)
    for source in range(N_STARS))
NODE_NAMES = tuple(
    f"{action}::{source}"
    for action in r507.SOURCES for source in r507.NAMED_SOURCES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def node_parts(node: int) -> tuple[int, int]:
    if not 0 <= node < N_NODES:
        raise ValueError("node index changed")
    return divmod(node, N_STARS)


def source_kind(source_index: int) -> str:
    name = r507.NAMED_SOURCES[source_index]
    return "embedding" if name == "E" else ("attention" if name.startswith("A") else "mlp")


def circuit_mask_hashes(circuit_masks: dict, tags: tuple[str, ...]) -> dict[str, str]:
    hashes = {}
    for tag in tags:
        digest = hashlib.sha256()
        for key in ("member", "slice_control"):
            digest.update(circuit_masks[tag][key].to(torch.uint8).contiguous().numpy().tobytes())
        hashes[tag] = digest.hexdigest()
    return hashes


def deduplicate_circuit_tags(circuit_masks: dict,
                             tags: tuple[str, ...]) -> tuple[tuple[str, ...], dict]:
    hashes = circuit_mask_hashes(circuit_masks, tags)
    first, kept, duplicates = {}, [], {}
    for tag in tags:
        digest = hashes[tag]
        if digest in first:
            duplicates[tag] = first[digest]
        else:
            first[digest] = tag
            kept.append(tag)
    return tuple(kept), {"hashes": hashes, "duplicates": duplicates,
                         "input_count": len(tags), "retained_count": len(kept)}


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    r519 = json.loads(R519_RESULT.read_text())
    r510 = json.loads(R510_RESULT.read_text())
    r507_result = json.loads(R507_RESULT.read_text())
    if not (r519.get("pred_a_exact_live_interaction_instrument") is True
            and r519.get("pred_b_small_circuit_specific_bilinear_support") is False
            and r519.get("strong_null") is True):
        raise RuntimeError("rung519 route changed")
    if not (r510.get("pred_a_exact_live_singleton_and_substitution_instrument") is True
            and r510.get("pred_b_one_to_sixteen_discovery_equivalence_pairs") is False
            and r510.get("strong_null") is True):
        raise RuntimeError("rung510 route changed")
    if not (r507_result.get("pred_a_exact_live_decomposition_and_intervention_instrument") is True
            and r507_result.get("strong_null") is True):
        raise RuntimeError("rung507 route changed")
    rows, task_masks, circuit_masks, scales, discovery_tags, confirmation_tags, metadata = \
        r507.validate_inputs()
    discovery_tags, discovery_identity = deduplicate_circuit_tags(
        circuit_masks, tuple(discovery_tags))
    confirmation_tags, confirmation_identity = deduplicate_circuit_tags(
        circuit_masks, tuple(confirmation_tags))
    if len(discovery_tags) != 32 or len(confirmation_tags) != 30:
        raise RuntimeError("frozen circuit partition changed after exact-mask deduplication")
    if len(STAR_INDICES) != 22 or any(len(indices) != 22 for indices in STAR_INDICES):
        raise RuntimeError("source-star indexing changed")
    if any(len(set(indices)) != 22 for indices in STAR_INDICES):
        raise RuntimeError("source-star contains duplicate terms")
    return rows, task_masks, circuit_masks, scales, discovery_tags, confirmation_tags, {
        **metadata,
        "documents": {"discovery": list(DISCOVERY), "unused": [748, 752],
                      "confirmation": list(CONFIRMATION)},
        "circuits": {"discovery": list(discovery_tags),
                     "confirmation": list(confirmation_tags)},
        "circuit_mask_identity": {"discovery": discovery_identity,
                                  "confirmation": confirmation_identity},
        "star_indices": {r507.NAMED_SOURCES[i]: list(indices)
                         for i, indices in enumerate(STAR_INDICES)},
    }


def _star_hidden(factors: dict, source_index: int) -> torch.Tensor:
    total = torch.zeros_like(factors["left"][:, :, 0])
    for term_index in STAR_INDICES[source_index]:
        total = total + r507._pair_hidden(factors, term_index)
    return total


def _independent_star_hidden(factors: dict, source_index: int) -> torch.Tensor:
    left = factors["left"]
    right = factors["right"]
    left_sum = left.sum(2)
    right_sum = right.sum(2)
    left_source = left[:, :, source_index]
    right_source = right[:, :, source_index]
    return left_source * right_sum + (left_sum - left_source) * right_source


def _star_output(mlp, factors: dict, source_index: int) -> torch.Tensor:
    return r507._linear(_star_hidden(factors, source_index), mlp.Down.weight.float())


def _empty_diagnostics() -> dict:
    row = r509._empty_diagnostics()
    row.update({
        "star_index_count_min": min(map(len, STAR_INDICES)),
        "star_index_count_max": max(map(len, STAR_INDICES)),
        "star_hidden_closure_relative_squared": 0.0,
        "star_output_closure_relative_squared": 0.0,
        "star_patches": 0, "star_patches_expected": 0,
        "star_patches_exact": False,
    })
    return row


def _audit_star_closure(diagnostics: dict, mlp, factors: dict) -> None:
    for source_index in range(N_STARS):
        summed = _star_hidden(factors, source_index)
        independent = _independent_star_hidden(factors, source_index)
        diagnostics["star_hidden_closure_relative_squared"] = max(
            diagnostics["star_hidden_closure_relative_squared"],
            r507._relative_squared(summed, independent))
        summed_output = r507._linear(summed, mlp.Down.weight.float())
        independent_output = r507._linear(independent, mlp.Down.weight.float())
        diagnostics["star_output_closure_relative_squared"] = max(
            diagnostics["star_output_closure_relative_squared"],
            r507._relative_squared(summed_output, independent_output))


@torch.no_grad()
def collect_stars(model, rows, task_masks, circuit_masks, circuit_tags,
                  scales, bounds):
    lo, hi, _split = bounds
    documents = hi - lo
    arms = ("intact",) + tuple(r507.NAMED_SOURCES)
    task = torch.zeros(N_ACTIONS, len(arms), documents, len(r507.TASK_CELLS),
                       dtype=torch.float64)
    counts = torch.zeros(documents, len(r507.TASK_CELLS), dtype=torch.float64)
    base_task = torch.zeros_like(counts)
    circuit_sums = torch.zeros(
        N_ACTIONS, len(arms), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    diagnostics = _empty_diagnostics()
    device = next(model.parameters()).device
    mlp = model.transformer.h[r507.TARGET].mlp
    for start in range(lo, hi, r507.BATCH):
        stop = start + r507.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in r507.TASK_CELLS}
        direct_logits, _, direct_diag, _ = r507._forward(
            model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        r509._update_diagnostics(diagnostics, direct_diag)
        absent_logits, absent, absent_diag, _ = r507._forward(
            model, tokens, scales, action="P", absent=True, capture_mlp10=True)
        diagnostics["calls"]["analytical"] += 1
        r509._update_diagnostics(diagnostics, absent_diag)
        _audit_star_closure(diagnostics, mlp, absent["factors"])
        base_task[local:local + r507.BATCH] = r507._task_sums(
            r507._nll(absent_logits, batch_rows).detach().cpu()[None], masks)[0]
        counts[local:local + r507.BATCH] = torch.stack(
            [masks[cell].sum(1).double() for cell in r507.TASK_CELLS], -1)
        absent_stars = tuple(_star_output(mlp, absent["factors"], s)
                             for s in range(N_STARS))
        absent_partition = r507._sum_unordered_pair_hidden(absent["factors"])
        diagnostics["exact_term_partition_relative_squared"] = max(
            diagnostics["exact_term_partition_relative_squared"],
            r507._relative_squared(
                absent_partition,
                absent["factors"]["left"].sum(2) * absent["factors"]["right"].sum(2)))
        nll_rows = []
        for action in r507.SOURCES:
            logits, current, current_diag, _ = r507._forward(
                model, tokens, scales, action=action, capture_mlp10=True)
            diagnostics["calls"]["analytical"] += 1
            r509._update_diagnostics(diagnostics, current_diag)
            r507._score_delta_closure(diagnostics, current, absent)
            _audit_star_closure(diagnostics, mlp, current["factors"])
            if action == "N":
                difference = logits.detach().float() - direct_logits.float()
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"], float(difference.abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    float(difference.square().sum())
                    / max(float(direct_logits.float().square().sum()), 1e-30))
            current_partition = r507._sum_unordered_pair_hidden(current["factors"])
            diagnostics["exact_term_partition_relative_squared"] = max(
                diagnostics["exact_term_partition_relative_squared"],
                r507._relative_squared(
                    current_partition,
                    current["factors"]["left"].sum(2)
                    * current["factors"]["right"].sum(2)))
            source_nll = [r507._nll(logits, batch_rows).detach().cpu()]
            for source_index in range(N_STARS):
                delta = _star_output(mlp, current["factors"], source_index) \
                    - absent_stars[source_index]
                replacement = current["deployed_write"] \
                    - delta.to(current["deployed_write"].dtype)
                patched_logits, _capture, patch_diag, patch_audit = \
                    r507.score_parent.run_forward(
                        model, tokens, action=action, scales=scales,
                        patch_writes={"m10": replacement})
                diagnostics["calls"]["analytical"] += 1
                diagnostics["star_patches"] += patch_audit["patches"]
                edit_rms = patch_diag["patch_rms_max"]
                diagnostics["zero_term_edits"] += int(edit_rms <= 0)
                if edit_rms > 0:
                    diagnostics["minimum_nonzero_term_edit_rms"] = min(
                        diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
                source_nll.append(r507._nll(patched_logits, batch_rows).detach().cpu())
            nll_rows.extend(source_nll)
        nll_stack = torch.stack(nll_rows).view(
            N_ACTIONS, len(arms), r507.BATCH, r507.TOKENS)
        task[:, :, local:local + r507.BATCH] = r507._task_sums(
            nll_stack.view(-1, r507.BATCH, r507.TOKENS), masks).view(
                N_ACTIONS, len(arms), r507.BATCH, len(r507.TASK_CELLS))
        if circuit_tags:
            matrix, observed = r507.state_parent._circuit_mask_matrix(
                circuit_masks, circuit_tags, start, stop, bounds)
            circuit_counts += observed
            circuit_sums += torch.matmul(
                nll_stack.view(N_ACTIONS * len(arms), -1).double(), matrix.T,
            ).view(N_ACTIONS, len(arms), 2, 2, len(circuit_tags))
    batches = documents // r507.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches,
        "analytical": batches * (1 + N_ACTIONS * len(arms)),
    }
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["star_patches_expected"] = batches * N_ACTIONS * N_STARS
    diagnostics["star_patches_exact"] = (
        diagnostics["star_patches"] == diagnostics["star_patches_expected"])
    return {
        "bounds": bounds, "arms": arms, "task": task, "task_counts": counts,
        "base_task": base_task, "source_task": task[:, 0],
        "circuit_tags": tuple(circuit_tags), "circuit_sums": circuit_sums,
        "circuit_counts": circuit_counts, "diagnostics": diagnostics,
    }


def _support(collection: dict) -> dict:
    lo, hi, split = collection["bounds"]
    middle = split - lo
    task_half = torch.stack((collection["task_counts"][:middle].sum(0),
                             collection["task_counts"][middle:].sum(0)))
    circuit = collection["circuit_counts"]
    return {
        "task_half_minimum": float(task_half.min()),
        "circuit_member_control_half_minimum": float(circuit.min()) if circuit.numel() else math.inf,
        "holds": bool((task_half > 0).all() and (not circuit.numel() or (circuit > 0).all())),
    }


def _instrument(collection: dict, *, require_support=True) -> bool:
    d = collection["diagnostics"]
    support = _support(collection)
    return bool(
        d["calls_exact"] and d["star_patches_exact"] and d["zero_term_edits"] == 0
        and d["factor_reconstruction_max"] <= 1e-10
        and d["raw_source_relative_squared"] <= r507.DEPLOYED_BF16_BAR
        and d["normalized_closure_relative_squared"] <= 1e-12
        and d["normalized_numerical_rms_ratio"] <= .02
        and d["float32_mlp10_closure"] <= 1e-10
        and d["deployed_mlp10_relative_squared"] <= r507.DEPLOYED_BF16_BAR
        and d["score_delta_float32_closure"] <= 1e-10
        and math.isfinite(d["score_delta_predeployment_relative_squared"])
        and d["score_delta_deployed_closure_relative_squared"] <= 1e-12
        and d["minimum_nonzero_score_edit_rms"] > 0
        and d["minimum_nonzero_term_edit_rms"] > 0
        and d["exact_term_partition_relative_squared"] <= 1e-10
        and d["star_hidden_closure_relative_squared"] <= 1e-8
        and d["star_output_closure_relative_squared"] <= 1e-8
        and d["native_replay_logit_max_abs"] == 0.0
        and d["native_replay_relative_squared"] <= 1e-12
        and (support["holds"] or not require_support))


def _circuit_fingerprint(collection: dict, arm: str, action: str,
                         window: str) -> torch.Tensor:
    action_index = r507.SOURCES.index(action)
    arm_index = collection["arms"].index(arm)
    intact_index = collection["arms"].index("intact")
    if window == "pooled":
        target = collection["circuit_sums"][action_index, arm_index].sum(0)
        intact = collection["circuit_sums"][action_index, intact_index].sum(0)
        counts = collection["circuit_counts"].sum(0)
    else:
        half = {"half0": 0, "half1": 1}[window]
        target = collection["circuit_sums"][action_index, arm_index, half]
        intact = collection["circuit_sums"][action_index, intact_index, half]
        counts = collection["circuit_counts"][half]
    effects = (target - intact) / counts.clamp_min(1)
    return effects[0] - effects[1]


def response_matrices(collection: dict) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    for window in ("half0", "half1", "pooled"):
        task, circuit = [], []
        for action in r507.SOURCES:
            for source in r507.NAMED_SOURCES:
                task.append(r507.finite_vector(
                    collection, source, collection, action, window).double())
                circuit.append(_circuit_fingerprint(
                    collection, source, action, window).double())
        result[window] = {"task": torch.stack(task), "circuit": torch.stack(circuit)}
        if result[window]["task"].shape != (N_NODES, 4):
            raise RuntimeError("task response shape changed")
        if result[window]["circuit"].shape != (N_NODES, len(collection["circuit_tags"])):
            raise RuntimeError("circuit response shape changed")
    return result


def _safe_cosine_rows(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(-1) / (
        torch.linalg.vector_norm(left, dim=-1)
        * torch.linalg.vector_norm(right, dim=-1)).clamp_min(1e-30)


def _residual_rows(actual: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(actual - predicted, dim=-1) \
        / torch.linalg.vector_norm(actual, dim=-1).clamp_min(1e-30)


def _pair_metrics(matrices: dict, left: int, right: int, beta: float,
                  confirmation=False) -> dict:
    inverse = 1.0 / beta
    row = {"beta_left_from_right": beta, "windows": {}}
    holds = True
    for window in ("half0", "half1", "pooled"):
        entry = {}
        for kind in ("circuit", "task"):
            left_vector = matrices[window][kind][left]
            right_vector = matrices[window][kind][right]
            entry[kind] = {
                "left_from_right_cosine": float(_safe_cosine_rows(
                    left_vector[None], (beta * right_vector)[None])[0]),
                "left_from_right_relative_residual": float(_residual_rows(
                    left_vector[None], (beta * right_vector)[None])[0]),
                "right_from_left_cosine": float(_safe_cosine_rows(
                    right_vector[None], (inverse * left_vector)[None])[0]),
                "right_from_left_relative_residual": float(_residual_rows(
                    right_vector[None], (inverse * left_vector)[None])[0]),
            }
        row["windows"][window] = entry
    circuit_rms = [float(matrices["pooled"]["circuit"][n].square().mean().sqrt())
                   for n in (left, right)]
    task_norm = [float(torch.linalg.vector_norm(matrices["pooled"]["task"][n]))
                 for n in (left, right)]
    row["circuit_rms_nat"] = circuit_rms
    row["task_norm_nat"] = task_norm
    row["material"] = bool(min(circuit_rms) >= .0005 and min(task_norm) >= .00025)
    row["scale_holds"] = bool(.25 <= abs(beta) <= 4)
    tested_windows = ("half0", "half1", "pooled") if confirmation else ("half0", "half1")
    for window in tested_windows:
        c = row["windows"][window]["circuit"]
        t = row["windows"][window]["task"]
        c_cos = .75 if confirmation else (.90 if window == "half0" else .80)
        c_res = .55 if confirmation else (.35 if window == "half0" else .50)
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
    c0, c1 = matrices["half0"]["circuit"].double(), matrices["half1"]["circuit"].double()
    t0, t1 = matrices["half0"]["task"].double(), matrices["half1"]["task"].double()
    cp, tp = matrices["pooled"]["circuit"].double(), matrices["pooled"]["task"].double()
    dot = c0 @ c0.T
    norm2 = c0.square().sum(-1)
    beta = dot / norm2[None, :].clamp_min(1e-30)
    safe_beta = torch.where(beta.abs() > 1e-30, beta, torch.ones_like(beta))

    def predicted_metrics(values):
        raw_dot = values @ values.T
        norms = torch.linalg.vector_norm(values, dim=1)
        cosine = raw_dot / (norms[:, None] * norms[None, :]).clamp_min(1e-30)
        signed_cosine = cosine * safe_beta.sign()
        forward = (values.square().sum(-1)[:, None]
                   + beta.square() * values.square().sum(-1)[None, :]
                   - 2 * beta * raw_dot).clamp_min(0).sqrt() / norms[:, None].clamp_min(1e-30)
        inverse = safe_beta.reciprocal()
        backward = (values.square().sum(-1)[None, :]
                    + inverse.square() * values.square().sum(-1)[:, None]
                    - 2 * inverse * raw_dot).clamp_min(0).sqrt() / norms[None, :].clamp_min(1e-30)
        return signed_cosine, forward, backward

    c0_cos, c0_f, c0_b = predicted_metrics(c0)
    c1_cos, c1_f, c1_b = predicted_metrics(c1)
    t0_cos, t0_f, t0_b = predicted_metrics(t0)
    t1_cos, t1_f, t1_b = predicted_metrics(t1)
    circuit_rms = cp.square().mean(-1).sqrt()
    task_norm = torch.linalg.vector_norm(tp, dim=-1)
    material = ((circuit_rms[:, None] >= .0005) & (circuit_rms[None, :] >= .0005)
                & (task_norm[:, None] >= .00025) & (task_norm[None, :] >= .00025))
    mask = (material & (beta.abs() >= .25) & (beta.abs() <= 4)
            & (c0_cos >= .90) & (c0_f <= .35) & (c0_b <= .35)
            & (c1_cos >= .80) & (c1_f <= .50) & (c1_b <= .50)
            & (t0_cos >= .70) & (t0_f <= .65) & (t0_b <= .65)
            & (t1_cos >= .70) & (t1_f <= .65) & (t1_b <= .65))
    indices = torch.nonzero(torch.triu(mask, diagonal=1), as_tuple=False)
    candidates = []
    for left, right in indices.tolist():
        metrics = _pair_metrics(matrices, left, right, float(beta[left, right]))
        if not metrics["holds"]:
            raise RuntimeError("vectorized and scalar pair detectors disagree")
        left_action, left_source = node_parts(left)
        right_action, right_source = node_parts(right)
        candidates.append({
            "left_node": left, "right_node": right,
            "left_name": NODE_NAMES[left], "right_name": NODE_NAMES[right],
            "same_source": left_source == right_source,
            "same_action": left_action == right_action,
            "cross_source": left_source != right_source,
            "cross_kind": source_kind(left_source) != source_kind(right_source),
            **metrics,
        })
    return candidates, {
        "nodes": N_NODES,
        "unordered_pairs_tested": N_NODES * (N_NODES - 1) // 2,
        "candidate_count": len(candidates),
        "small_relation": bool(1 <= len(candidates) <= MAX_CANDIDATES),
        "material_nodes": int(((circuit_rms >= .0005) & (task_norm >= .00025)).sum()),
        "action_portability_pairs": sum(row["same_source"] and not row["same_action"]
                                        for row in candidates),
        "cross_source_pairs": sum(row["cross_source"] for row in candidates),
        "cross_kind_pairs": sum(row["cross_kind"] for row in candidates),
    }


def permutation_control_counts(matrices: dict) -> list[int]:
    counts = []
    dimensions = matrices["half0"]["circuit"].shape[1]
    for seed in CONTROL_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.rand(N_NODES, dimensions, generator=generator).argsort(dim=1)
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
        metrics = _pair_metrics(
            matrices, candidate["left_node"], candidate["right_node"],
            candidate["beta_left_from_right"], confirmation=True)
        key = f"{candidate['left_name']} <-> {candidate['right_name']}"
        checks[key] = metrics
        if metrics["holds"]:
            passing.append(candidate)
    return passing, checks


def _comparison(actual: torch.Tensor, prediction: torch.Tensor) -> dict:
    return {
        "cosine": float(_safe_cosine_rows(actual[None], prediction[None])[0]),
        "relative_residual": float(_residual_rows(actual[None], prediction[None])[0]),
        "actual_norm_nat": float(torch.linalg.vector_norm(actual)),
        "prediction_norm_nat": float(torch.linalg.vector_norm(prediction)),
    }


def multiple_mediator_discrepancy(matrices: dict, circuit_tags: tuple[str, ...]) -> dict:
    payload = torch.load(R510_BUNDLE, map_location="cpu", weights_only=False)
    exact = payload["collections"]["exact_discovery"]
    if list(exact["circuit_tags"]) != list(circuit_tags):
        raise RuntimeError("rung510 circuit coordinates changed")
    rows = []
    for node in range(N_NODES):
        action_index, source_index = node_parts(node)
        action = r507.SOURCES[action_index]
        window_rows = {}
        for window in ("half0", "half1", "pooled"):
            summed_task = sum((r507.finite_vector(
                exact, r507.PAIR_NAMES[term], exact, action, window).double()
                for term in STAR_INDICES[source_index]), torch.zeros(4, dtype=torch.float64))
            summed_circuit = sum((r509._circuit_fingerprint(
                exact, r507.PAIR_NAMES[term], action, window).double()
                for term in STAR_INDICES[source_index]),
                torch.zeros(len(circuit_tags), dtype=torch.float64))
            window_rows[window] = {
                "task": _comparison(matrices[window]["task"][node], summed_task),
                "circuit": _comparison(matrices[window]["circuit"][node], summed_circuit),
            }
        rows.append({"node": node, "node_name": NODE_NAMES[node], "windows": window_rows})
    pooled_task_residuals = [row["windows"]["pooled"]["task"]["relative_residual"]
                             for row in rows]
    pooled_circuit_residuals = [row["windows"]["pooled"]["circuit"]["relative_residual"]
                                for row in rows]
    return {
        "definition": "actual joint star-removal CE effect versus sum of 22 singleton-removal CE effects",
        "selector": False, "rows": rows,
        "summary": {
            "nodes": len(rows),
            "median_pooled_task_relative_residual": float(torch.tensor(
                pooled_task_residuals, dtype=torch.float64).median()),
            "maximum_pooled_task_relative_residual": max(pooled_task_residuals),
            "median_pooled_circuit_relative_residual": float(torch.tensor(
                pooled_circuit_residuals, dtype=torch.float64).median()),
            "maximum_pooled_circuit_relative_residual": max(pooled_circuit_residuals),
        },
    }


def _physical_diagnostics() -> dict:
    row = _empty_diagnostics()
    row.update({"substitution_patches": 0, "substitution_patches_expected": 0,
                "substitution_patches_exact": False})
    return row


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
    task = torch.zeros(len(directions), documents, len(r507.TASK_CELLS), dtype=torch.float64)
    counts = torch.zeros(documents, len(r507.TASK_CELLS), dtype=torch.float64)
    circuit_sums = torch.zeros(len(directions), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    diagnostics = _physical_diagnostics()
    device = next(model.parameters()).device
    mlp = model.transformer.h[r507.TARGET].mlp
    needed = sorted({direction[key] for direction in directions for key in ("target", "donor")})
    for start in range(lo, hi, r507.BATCH):
        stop = start + r507.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in r507.TASK_CELLS}
        direct_logits, _, direct_diag, _ = r507._forward(model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        r509._update_diagnostics(diagnostics, direct_diag)
        _absent_logits, absent, absent_diag, _ = r507._forward(
            model, tokens, scales, action="P", absent=True, capture_mlp10=True)
        diagnostics["calls"]["analytical"] += 1
        r509._update_diagnostics(diagnostics, absent_diag)
        _audit_star_closure(diagnostics, mlp, absent["factors"])
        captures = []
        for action in r507.SOURCES:
            logits, capture, diag, _ = r507._forward(
                model, tokens, scales, action=action, capture_mlp10=True)
            diagnostics["calls"]["analytical"] += 1
            r509._update_diagnostics(diagnostics, diag)
            r507._score_delta_closure(diagnostics, capture, absent)
            _audit_star_closure(diagnostics, mlp, capture["factors"])
            if action == "N":
                difference = logits.detach().float() - direct_logits.float()
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"], float(difference.abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    float(difference.square().sum())
                    / max(float(direct_logits.float().square().sum()), 1e-30))
            captures.append(capture)
        absent_stars = {source: _star_output(mlp, absent["factors"], source)
                        for source in {node_parts(node)[1] for node in needed}}
        donor_deltas = {}
        for node in needed:
            action_index, source_index = node_parts(node)
            donor_deltas[node] = _star_output(
                mlp, captures[action_index]["factors"], source_index) - absent_stars[source_index]
        nll_rows = []
        for direction in directions:
            target_action, _target_source = node_parts(direction["target"])
            action = r507.SOURCES[target_action]
            delta = direction["scale"] * donor_deltas[direction["donor"]]
            replacement = captures[target_action]["deployed_write"] \
                - delta.to(captures[target_action]["deployed_write"].dtype)
            logits, _capture, patch_diag, patch_audit = r507.score_parent.run_forward(
                model, tokens, action=action, scales=scales,
                patch_writes={"m10": replacement})
            diagnostics["calls"]["analytical"] += 1
            diagnostics["substitution_patches"] += patch_audit["patches"]
            edit_rms = patch_diag["patch_rms_max"]
            diagnostics["zero_term_edits"] += int(edit_rms <= 0)
            if edit_rms > 0:
                diagnostics["minimum_nonzero_term_edit_rms"] = min(
                    diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
            nll_rows.append(r507._nll(logits, batch_rows).detach().cpu())
        nll_stack = torch.stack(nll_rows)
        task[:, local:local + r507.BATCH] = r507._task_sums(nll_stack, masks)
        counts[local:local + r507.BATCH] = torch.stack(
            [masks[cell].sum(1).double() for cell in r507.TASK_CELLS], -1)
        matrix, observed = r507.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        circuit_counts += observed
        circuit_sums += torch.matmul(
            nll_stack.view(len(directions), -1).double(), matrix.T,
        ).view(len(directions), 2, 2, len(circuit_tags))
    batches = documents // r507.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches,
        "analytical": batches * (1 + N_ACTIONS + len(directions)),
    }
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["substitution_patches_expected"] = batches * len(directions)
    diagnostics["substitution_patches_exact"] = (
        diagnostics["substitution_patches"] == diagnostics["substitution_patches_expected"])
    return {
        "bounds": bounds, "directions": directions, "task": task,
        "task_counts": counts, "circuit_tags": tuple(circuit_tags),
        "circuit_sums": circuit_sums, "circuit_counts": circuit_counts,
        "diagnostics": diagnostics,
    }


def _substitution_instrument(collection: dict) -> bool:
    d = collection["diagnostics"]
    return bool(
        d["calls_exact"] and d["substitution_patches_exact"] and d["zero_term_edits"] == 0
        and d["factor_reconstruction_max"] <= 1e-10
        and d["raw_source_relative_squared"] <= r507.DEPLOYED_BF16_BAR
        and d["normalized_closure_relative_squared"] <= 1e-12
        and d["normalized_numerical_rms_ratio"] <= .02
        and d["float32_mlp10_closure"] <= 1e-10
        and d["deployed_mlp10_relative_squared"] <= r507.DEPLOYED_BF16_BAR
        and d["score_delta_float32_closure"] <= 1e-10
        and math.isfinite(d["score_delta_predeployment_relative_squared"])
        and d["score_delta_deployed_closure_relative_squared"] <= 1e-12
        and d["minimum_nonzero_score_edit_rms"] > 0
        and d["minimum_nonzero_term_edit_rms"] > 0
        and d["star_hidden_closure_relative_squared"] <= 1e-8
        and d["star_output_closure_relative_squared"] <= 1e-8
        and d["native_replay_logit_max_abs"] == 0.0
        and d["native_replay_relative_squared"] <= 1e-12)


def _native_node_response(exact: dict, node: int, window: str) -> dict:
    action_index, source_index = node_parts(node)
    action, source = r507.SOURCES[action_index], r507.NAMED_SOURCES[source_index]
    return {
        "task": r507.finite_vector(exact, source, exact, action, window).double(),
        "circuit": _circuit_fingerprint(exact, source, action, window).double(),
    }


def _substituted_response(substitutions: dict, exact: dict, direction: int,
                          window: str) -> dict:
    target_node = substitutions["directions"][direction]["target"]
    action_index, _source_index = node_parts(target_node)
    intact_index = exact["arms"].index("intact")
    if window == "pooled":
        lo, hi = 0, substitutions["task"].shape[1]
        circuit_sum = substitutions["circuit_sums"][direction].sum(0)
        intact_sum = exact["circuit_sums"][action_index, intact_index].sum(0)
        circuit_counts = substitutions["circuit_counts"].sum(0)
    else:
        half = {"half0": 0, "half1": 1}[window]
        bounds = substitutions["bounds"]
        absolute = ((bounds[0], bounds[2]), (bounds[2], bounds[1]))[half]
        lo, hi = absolute[0] - bounds[0], absolute[1] - bounds[0]
        circuit_sum = substitutions["circuit_sums"][direction, half]
        intact_sum = exact["circuit_sums"][action_index, intact_index, half]
        circuit_counts = substitutions["circuit_counts"][half]
    full_task = (substitutions["task"][direction, lo:hi]
                 - exact["task"][action_index, intact_index, lo:hi]).sum(0) \
        / substitutions["task_counts"][lo:hi].sum(0).clamp_min(1)
    task_indices = [r507.TASK_CELLS.index(cell) for cell in r507.GRAD_CELLS[:4]]
    circuit_effect = (circuit_sum - intact_sum) / circuit_counts.clamp_min(1)
    return {"task": full_task[task_indices],
            "off_target": float(full_task[r507.TASK_CELLS.index("off_target")]),
            "circuit": circuit_effect[0] - circuit_effect[1]}


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
                circuit = _comparison(native["circuit"], observed["circuit"])
                task = _comparison(native["task"], observed["task"])
                off_target_holds = bool(window == "pooled" or abs(observed["off_target"]) <= .002)
                holds = bool(circuit["cosine"] >= .75
                             and circuit["relative_residual"] <= .55
                             and task["cosine"] >= .70
                             and task["relative_residual"] <= .65
                             and off_target_holds)
                side_row["windows"][window] = {
                    "circuit": circuit, "task": task,
                    "off_target_effect_nat": observed["off_target"],
                    "off_target_holds": off_target_holds, "holds": holds,
                }
                side_row["holds"] &= holds
            row["directions"][side] = side_row
            row["holds"] &= side_row["holds"]
        checks[f"{candidate['left_name']} <-> {candidate['right_name']}"] = row
        if row["holds"]:
            passing.append(candidate)
    return passing, checks


def quotient_groups(passing: list[dict]) -> list[dict]:
    adjacency, edge_map = {}, {}
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
        complete = all((left, right) in edge_map
                       for left, right in itertools.combinations(nodes, 2))
        root = nodes[0]
        scales = {root: 1.0}
        for node in nodes[1:]:
            scales[node] = 1.0 / edge_map[(root, node)]
        errors = []
        if complete:
            for left, right in itertools.combinations(nodes, 2):
                expected = scales[left] / scales[right]
                observed = edge_map[(left, right)]
                errors.append(abs(observed - expected) / max(abs(expected), 1e-30))
        if complete and (not errors or max(errors) <= .25):
            groups.append({
                "nodes": nodes, "node_names": [NODE_NAMES[node] for node in nodes],
                "scales_relative_to_root": {NODE_NAMES[node]: scales[node] for node in nodes},
                "maximum_scale_cycle_relative_error": max(errors, default=0.0),
            })
    return groups


def planted_suite() -> dict:
    cases = []
    for case in range(8):
        generator = torch.Generator(device="cpu").manual_seed(5200 + case)
        circuit = torch.zeros(N_NODES, 32, dtype=torch.float64)
        task = torch.zeros(N_NODES, 4, dtype=torch.float64)
        left, right = 2 * case, 2 * case + 1
        circuit[left] = .002 * torch.randn(32, generator=generator)
        task[left] = .002 * torch.randn(4, generator=generator)
        circuit[right] = -2 * circuit[left]
        task[right] = -2 * task[left]
        matrices = {window: {"circuit": circuit.clone(), "task": task.clone()}
                    for window in ("half0", "half1", "pooled")}
        pairs, summary = discover_pairs(matrices)
        controls = permutation_control_counts(matrices)
        recovered = any(row["left_node"] == left and row["right_node"] == right
                        for row in pairs)
        holds = bool(recovered and summary["candidate_count"] == 1 and max(controls) == 0)
        cases.append({"case": case, "planted_pair": [left, right],
                      "candidate_count": summary["candidate_count"],
                      "control_counts": controls, "holds": holds})
    return {"cases": cases, "holds": all(row["holds"] for row in cases)}


def _bundle_collection(collection):
    return {key: value for key, value in collection.items() if key != "diagnostics"}


def dry_run() -> None:
    planted = planted_suite()
    assert N_STARS == 22 and N_NODES == 88
    assert all(len(indices) == 22 for indices in STAR_INDICES)
    assert N_NODES * (N_NODES - 1) // 2 == 3828
    assert planted["holds"]
    assert 2 * 5828 + 372 + 124 * 16 == 14012
    print(json.dumps({
        "status": "dry_run_passed", "rung": 520,
        "model_loaded": False, "outcomes_opened": False,
        "sources": N_STARS, "nodes": N_NODES,
        "unordered_pairs_tested": 3828,
        "maximum_conditional_forwards": 14012,
        "planted_suite": planted,
    }, indent=2, sort_keys=True))


def _gpu_smoke() -> None:
    rows, task_masks, circuit_masks, scales, discovery_tags, confirmation_tags, _metadata = \
        validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    windows = (
        ("discovery_half0", (500, 504, 502), discovery_tags[:1]),
        ("discovery_half1", (624, 628, 626), discovery_tags[:1]),
        ("confirmation_half0", (752, 756, 754), confirmation_tags[:1]),
        ("confirmation_half1", (876, 880, 878), confirmation_tags[:1]),
    )
    collections, checks = {}, {"weights": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256}
    for name, bounds, tags in windows:
        collections[name] = collect_stars(
            model, rows, task_masks, circuit_masks, tags, scales, bounds)
        checks[name] = _instrument(collections[name], require_support=False)
    passed = all(checks.values())
    print(json.dumps({
        "status": "smoke_passed" if passed else "smoke_failed", "rung": 520,
        "scientific_outcomes_retained": False, "checks": checks,
        "diagnostics": {name: row["diagnostics"] for name, row in collections.items()},
        "full_forwards": sum(sum(row["diagnostics"]["calls"].values())
                             for row in collections.values()),
        "backwards": 0,
    }, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(f"rung520 CUDA smoke failed: "
                           f"{sorted(name for name, value in checks.items() if not value)}")


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        dry_run()
        return
    if os.environ.get("BQLIB_GPU_SMOKE") == "1" or "--gpu-smoke" in sys.argv:
        _gpu_smoke()
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung520 output namespace already exists")
    started = time.time()
    rows, task_masks, circuit_masks, scales, discovery_tags, confirmation_tags, metadata = \
        validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    collections = {}
    collections["discovery"] = collect_stars(
        model, rows, task_masks, circuit_masks, discovery_tags, scales, DISCOVERY)
    discovery_calibration = r507._calibration(
        collections["discovery"]["base_task"], collections["discovery"]["source_task"],
        collections["discovery"]["task_counts"], DISCOVERY)
    discovery_calibration_ok = r507.state_parent.calibration_holds(discovery_calibration)
    discovery_matrices = response_matrices(collections["discovery"])
    candidates, discovery_summary = discover_pairs(discovery_matrices)
    control_counts = permutation_control_counts(discovery_matrices)
    control_q95 = float(torch.quantile(
        torch.tensor(control_counts, dtype=torch.float64), .95, interpolation="higher"))
    pred_b_pre = bool(discovery_summary["small_relation"]
                      and discovery_summary["candidate_count"] > control_q95)
    discrepancy = multiple_mediator_discrepancy(discovery_matrices, discovery_tags)

    confirmation_calibration, confirmation_calibration_ok = {}, False
    confirmation_checks, confirmed = {}, []
    physical_checks, physical_pairs, groups = {}, [], []
    if discovery_calibration_ok and pred_b_pre:
        collections["confirmation"] = collect_stars(
            model, rows, task_masks, circuit_masks, confirmation_tags, scales, CONFIRMATION)
        confirmation_calibration = r507._calibration(
            collections["confirmation"]["base_task"],
            collections["confirmation"]["source_task"],
            collections["confirmation"]["task_counts"], CONFIRMATION)
        confirmation_calibration_ok = r507.state_parent.calibration_holds(
            confirmation_calibration)
        confirmation_matrices = response_matrices(collections["confirmation"])
        confirmed, confirmation_checks = confirmation_pairs(
            confirmation_matrices, candidates)
    if confirmation_calibration_ok and confirmed:
        collections["substitutions"] = collect_substitutions(
            model, rows, task_masks, circuit_masks, confirmation_tags,
            scales, CONFIRMATION, confirmed)
        physical_pairs, physical_checks = score_substitutions(
            collections["substitutions"], collections["confirmation"], confirmed)
        groups = quotient_groups(physical_pairs)

    planted = planted_suite()
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and discovery_calibration_ok and planted["holds"]
        and _instrument(collections["discovery"])
        and ("confirmation" not in collections
             or (confirmation_calibration_ok and _instrument(collections["confirmation"])))
        and ("substitutions" not in collections
             or _substitution_instrument(collections["substitutions"])))
    pred_b = bool(pred_a and pred_b_pre)
    pred_c = bool(pred_b and confirmation_calibration_ok and confirmed)
    pred_d = bool(pred_c and physical_pairs)
    pred_e = bool(pred_d and any(
        node_parts(edge["left_node"])[1] != node_parts(edge["right_node"])[1]
        for edge in physical_pairs))
    strong_null = not (pred_a and pred_b and pred_c and pred_d and pred_e)
    if not pred_a:
        next_step = "repair_exact_source_star_instrument_only"
    elif not pred_b:
        next_step = "task_defined_finite_state_transition_spanning_multiple_downstream_sites"
    elif not pred_c:
        next_step = "retain_discovery_screen_only_and_test_consumer_conditioning"
    elif not pred_d:
        next_step = "localize_first_consumer_that_separates_response_similar_source_stars"
    elif not pred_e:
        next_step = "retain_action_portability_without_cross_source_grouping_claim"
    else:
        next_step = "test_joint_installation_composition_and_literal_program_price"

    torch.save({
        "schema": "rung520_mlp10_source_star_causal_quotient_v1",
        "collections": {name: _bundle_collection(row) for name, row in collections.items()},
        "discovery_candidates": candidates, "confirmed_pairs": confirmed,
        "physical_pairs": physical_pairs,
        "raw_tokens_logits_hidden_states_or_weights_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": 520,
        "claim_level": "source_star_response_pair_until_bidirectional_physical_substitution",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "star_definition": {
            "named_sources": list(r507.NAMED_SOURCES),
            "star_term_indices": [list(indices) for indices in STAR_INDICES],
            "terms_per_star": 22, "stars_overlap_on_cross_terms": True,
            "nodes": list(NODE_NAMES),
        },
        "calibration": {"discovery": discovery_calibration,
                        "confirmation": confirmation_calibration},
        "calibration_holds": {"discovery": discovery_calibration_ok,
                              "confirmation": confirmation_calibration_ok},
        "support": {name: _support(row) for name, row in collections.items()
                    if name in ("discovery", "confirmation")},
        "diagnostics": {name: row["diagnostics"] for name, row in collections.items()},
        "analysis": {
            "discovery_summary": discovery_summary,
            "discovery_candidates": candidates,
            "permutation_control_candidate_counts": control_counts,
            "permutation_control_q95_higher": control_q95,
            "multiple_mediator_discrepancy": discrepancy,
            "confirmation_checks": confirmation_checks,
            "confirmed_pairs": confirmed,
            "physical_checks": physical_checks,
            "physical_pairs": physical_pairs,
            "quotient_groups": groups,
            "physical_cross_source_pairs": sum(
                node_parts(row["left_node"])[1] != node_parts(row["right_node"])[1]
                for row in physical_pairs),
            "physical_cross_kind_pairs": sum(
                source_kind(node_parts(row["left_node"])[1])
                != source_kind(node_parts(row["right_node"])[1])
                for row in physical_pairs),
        },
        "planted_suite": planted,
        'pred_a_exact_live_source_star_instrument': pred_a,
        'pred_b_one_to_sixteen_pairs_beat_permutation_control': pred_b,
        'pred_c_heldout_documents_and_circuits': pred_c,
        'pred_d_bidirectional_physical_substitution': pred_d,
        'pred_e_cross_native_source_boundary': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "full_forwards": sum(sum(row["diagnostics"]["calls"].values())
                                 for row in collections.values()),
            "backwards": 0,
            "cpu_pair_comparisons": 3828,
            "discovery_candidates": len(candidates),
            "confirmed_pairs": len(confirmed),
            "physical_pairs": len(physical_pairs),
            "maximum_conditional_forwards": 14012,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started, "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 520,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "discovery_candidates": len(candidates), "control_q95": control_q95,
        "confirmed_pairs": len(confirmed), "physical_pairs": len(physical_pairs),
        "quotient_groups": len(groups), "execution_price": result["execution_price"],
        "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
