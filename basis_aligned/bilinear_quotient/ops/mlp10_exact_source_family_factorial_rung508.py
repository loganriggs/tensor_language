#!/usr/bin/env python3
"""RUNG508 -- finite exact MLP10 family terms and joint-removal prediction."""

# BQGATE: EXPERIMENT
# pred_a: exact live finite source-family instrument
# pred_b: two to eight finite discovery family terms without ranking
# pred_c: at least two family terms confirm on held-out documents
# pred_d: a discovery-frozen joint-removal rule predicts confirmation
# pred_e: a confirmed family term contains the equality-attention family

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
import mlp10_exact_source_pair_causal_split_rung507 as parent


PREREG = POLY / "MLP10_EXACT_SOURCE_FAMILY_FACTORIAL_RUNG508_PREREGISTRATION.md"
ADDENDUM = POLY / "MLP10_EXACT_SOURCE_FAMILY_FACTORIAL_RUNG508_PREFLIGHT_ADDENDUM.md"
PARENT_SOURCE = ROOT / "ops/mlp10_exact_source_pair_causal_split_rung507.py"
PARENT_RESULT = ROOT / "mlp10_exact_source_pair_causal_split_rung507_results.json"
PARENT_BUNDLE = ROOT / "mlp10_exact_source_pair_causal_split_rung507_bundle.pt"
OUT = ROOT / "mlp10_exact_source_family_factorial_rung508_results.json"
BUNDLE = ROOT / "mlp10_exact_source_family_factorial_rung508_bundle.pt"
HASHES = {
    PREREG: "eb2a8cbf3c2aaf97b31f35179bb207361e1285d183404480c46fad9fe7a48af6",
    ADDENDUM: "c4c6cec497f68e8dc7c9b1e5d2a7537040a0d51f42b648cf7ada9d7f54158914",
    PARENT_SOURCE: "4bb6fbf9a12cbdae05162cff86abb84d31c834dfa2f7a1d92d75f5092d2e8035",
    PARENT_RESULT: "f3ce5669bb86e5e4a36e4fa44a2c2ff488bc3806ab86380ad359c0c6310fe57c",
    PARENT_BUNDLE: "bc72fcd9e1b7be5be3219ffd1284d8aa23c9c89778ca8a3e02faf8d0ba889dcd",
}

FAMILIES = {
    "E": ("E",),
    "A_pre": tuple(f"A{i}" for i in range(5)),
    "A_eq": tuple(f"A{i}" for i in range(5, 9)),
    "A_post": ("A9", "A10"),
    "M_pre": tuple(f"M{i}" for i in range(5)),
    "M_post": tuple(f"M{i}" for i in range(5, 10)),
}
FAMILY_NAMES = tuple(FAMILIES)
FAMILY_SOURCE_INDICES = tuple(
    tuple(parent.NAMED_SOURCES.index(source) for source in FAMILIES[family])
    for family in FAMILY_NAMES
)
FAMILY_PAIRS = tuple(itertools.combinations_with_replacement(range(len(FAMILY_NAMES)), 2))
GROUP_NAMES = tuple(f"{FAMILY_NAMES[left]}x{FAMILY_NAMES[right]}"
                    for left, right in FAMILY_PAIRS)
SOURCE_TO_FAMILY = {
    parent.NAMED_SOURCES.index(source): family_index
    for family_index, family in enumerate(FAMILY_NAMES)
    for source in FAMILIES[family]
}
GROUP_SPECS = tuple(tuple(
    pair_index for pair_index, (left_source, right_source) in enumerate(parent.SOURCE_PAIRS)
    if tuple(sorted((SOURCE_TO_FAMILY[left_source], SOURCE_TO_FAMILY[right_source])))
    == family_pair
) for family_pair in FAMILY_PAIRS)
FULL_NAMED = "FULL_NAMED"
DISCOVERY = (500, 748, 624)
CONFIRMATION = (752, 1000, 876)
MAX_TERMS = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _group_relationship(left_name, right_name):
    left = set(FAMILY_PAIRS[GROUP_NAMES.index(left_name)])
    right = set(FAMILY_PAIRS[GROUP_NAMES.index(right_name)])
    shared = sorted(left & right)
    return {
        "left_term": left_name, "right_term": right_name,
        "shared_families": [FAMILY_NAMES[index] for index in shared],
        "same_left_family": FAMILY_PAIRS[GROUP_NAMES.index(left_name)][0]
        == FAMILY_PAIRS[GROUP_NAMES.index(right_name)][0],
        "same_right_family": FAMILY_PAIRS[GROUP_NAMES.index(left_name)][1]
        == FAMILY_PAIRS[GROUP_NAMES.index(right_name)][1],
    }


def _family_outputs(mlp, factors):
    left = [factors["left"][:, :, indices].sum(2) for indices in FAMILY_SOURCE_INDICES]
    right = [factors["right"][:, :, indices].sum(2) for indices in FAMILY_SOURCE_INDICES]
    hidden = []
    for first, second in FAMILY_PAIRS:
        value = left[first] * right[second]
        if first != second:
            value = value + left[second] * right[first]
        hidden.append(value)
    hidden = torch.stack(hidden, dim=2)
    outputs = parent._linear(hidden, mlp.Down.weight.float())
    return outputs, hidden.sum(2)


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    if not (
        result.get("pred_a_exact_live_decomposition_and_intervention_instrument") is True
        and result.get("pred_b_sparse_source_stable_gradient_screen") is True
        and result.get("pred_c_at_least_two_terms_finitely_confirm") is False
        and result.get("strong_null") is True
        and result.get("next_step")
        == "replace_gradient_screen_with_registered_input_source_family_factorial"
        and "validation" not in result.get("diagnostics", {})
    ):
        raise RuntimeError("rung507 route or unopened-validation identity changed")
    rows, task_masks, circuit_masks, scales, _discovery_tags, validation_tags, metadata = \
        parent.validate_inputs()
    covered_sources = [source for family in FAMILIES.values() for source in family]
    covered_pairs = [index for spec in GROUP_SPECS for index in spec]
    if sorted(covered_sources) != sorted(parent.NAMED_SOURCES) \
            or len(set(covered_sources)) != len(parent.NAMED_SOURCES):
        raise RuntimeError("source families do not partition named sources")
    if len(GROUP_NAMES) != 21 or sorted(covered_pairs) != list(range(253)) \
            or len(set(covered_pairs)) != 253:
        raise RuntimeError("family terms do not partition exact pair terms")
    support = {
        phase: {
            f"{left}:{right}": {
                cell: int(task_masks[cell][left:right].sum())
                for cell in parent.TASK_CELLS
            }
            for left, right in ((bounds[0], bounds[2]), (bounds[2], bounds[1]))
        }
        for phase, bounds in (("discovery", DISCOVERY), ("confirmation", CONFIRMATION))
    }
    if min(value for phase in support.values() for half in phase.values()
           for value in half.values()) <= 0:
        raise RuntimeError("task support changed")
    return rows, task_masks, circuit_masks, scales, list(validation_tags), {
        **metadata, "rung507_result_sha256": sha256(PARENT_RESULT),
        "rung507_bundle_sha256": sha256(PARENT_BUNDLE),
        "families": FAMILIES, "family_terms": list(GROUP_NAMES),
        "documents": {"discovery": list(DISCOVERY), "unused": [748, 752],
                      "confirmation": list(CONFIRMATION)},
        "task_support": support,
    }


def _collection_diagnostics():
    diagnostics = parent._empty_diagnostics()
    diagnostics.update({
        "family_partition_relative_squared": 0.0,
        "family_score_delta_relative_squared": 0.0,
        "patches": 0, "patches_expected": 0, "patches_exact": False,
    })
    return diagnostics


def _update_parent_diagnostics(total, row):
    parent._update_diagnostics(total, row)


def collect_singletons(model, rows, task_masks, circuit_masks, circuit_tags, scales, bounds):
    lo, hi, _split = bounds
    documents = hi - lo
    arms = ("intact", FULL_NAMED) + GROUP_NAMES
    task = torch.zeros(len(parent.SOURCES), len(arms), documents,
                       len(parent.TASK_CELLS), dtype=torch.float64)
    counts = torch.zeros(documents, len(parent.TASK_CELLS), dtype=torch.float64)
    base_task = torch.zeros_like(counts)
    circuit_sums = torch.zeros(
        len(parent.SOURCES), len(arms), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    diagnostics = _collection_diagnostics()
    device = next(model.parameters()).device
    mlp = model.transformer.h[parent.TARGET].mlp
    for start in range(lo, hi, parent.BATCH):
        stop = start + parent.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in parent.TASK_CELLS}
        direct_logits, _, direct_diag, _ = parent._forward(model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        _update_parent_diagnostics(diagnostics, direct_diag)
        absent_logits, absent, absent_diag, _ = parent._forward(
            model, tokens, scales, action="P", absent=True, capture_mlp10=True)
        diagnostics["calls"]["analytical"] += 1
        _update_parent_diagnostics(diagnostics, absent_diag)
        base_task[local:local + parent.BATCH] = parent._task_sums(
            parent._nll(absent_logits, batch_rows).detach().cpu()[None], masks)[0]
        counts[local:local + parent.BATCH] = torch.stack(
            [masks[cell].sum(1).double() for cell in parent.TASK_CELLS], -1)
        nll_rows = []
        for source in parent.SOURCES:
            logits, current, current_diag, _ = parent._forward(
                model, tokens, scales, action=source, capture_mlp10=True)
            diagnostics["calls"]["analytical"] += 1
            _update_parent_diagnostics(diagnostics, current_diag)
            parent._score_delta_closure(diagnostics, current, absent)
            if source == "N":
                difference = logits.detach().float() - direct_logits.float()
                diagnostics["native_replay_logit_max_abs"] = max(
                    diagnostics["native_replay_logit_max_abs"], float(difference.abs().max()))
                diagnostics["native_replay_relative_squared"] = max(
                    diagnostics["native_replay_relative_squared"],
                    float(difference.square().sum())
                    / max(float(direct_logits.float().square().sum()), 1e-30))
            current_groups, current_sum = _family_outputs(mlp, current["factors"])
            absent_groups, absent_sum = _family_outputs(mlp, absent["factors"])
            diagnostics["family_partition_relative_squared"] = max(
                diagnostics["family_partition_relative_squared"],
                parent._relative_squared(current_sum,
                                         current["factors"]["left"].sum(2)
                                         * current["factors"]["right"].sum(2)),
                parent._relative_squared(absent_sum,
                                         absent["factors"]["left"].sum(2)
                                         * absent["factors"]["right"].sum(2)))
            group_deltas = current_groups - absent_groups
            summed_delta = group_deltas.sum(2)
            semantic_delta = current["factors"]["semantic_output"] \
                - absent["factors"]["semantic_output"]
            diagnostics["family_score_delta_relative_squared"] = max(
                diagnostics["family_score_delta_relative_squared"],
                parent._relative_squared(summed_delta, semantic_delta))
            source_nll = [parent._nll(logits, batch_rows).detach().cpu()]
            patch_deltas = (summed_delta,) + tuple(group_deltas[:, :, index]
                                                      for index in range(len(GROUP_NAMES)))
            for delta in patch_deltas:
                replacement = current["deployed_write"] - delta.to(
                    current["deployed_write"].dtype)
                patched_logits, _captures, patch_diag, patch_audit = \
                    parent.score_parent.run_forward(
                        model, tokens, action=source, scales=scales,
                        patch_writes={"m10": replacement})
                diagnostics["calls"]["analytical"] += 1
                diagnostics["patches"] += patch_audit["patches"]
                edit_rms = patch_diag["patch_rms_max"]
                diagnostics["zero_term_edits"] += int(edit_rms <= 0)
                if edit_rms > 0:
                    diagnostics["minimum_nonzero_term_edit_rms"] = min(
                        diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
                source_nll.append(parent._nll(patched_logits, batch_rows).detach().cpu())
            nll_rows.extend(source_nll)
        nll_stack = torch.stack(nll_rows).view(
            len(parent.SOURCES), len(arms), parent.BATCH, parent.TOKENS)
        task[:, :, local:local + parent.BATCH] = parent._task_sums(
            nll_stack.view(-1, parent.BATCH, parent.TOKENS), masks).view(
                len(parent.SOURCES), len(arms), parent.BATCH, len(parent.TASK_CELLS))
        matrix, observed = parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        circuit_counts += observed
        circuit_sums += torch.matmul(
            nll_stack.view(len(parent.SOURCES) * len(arms), -1).double(), matrix.T,
        ).view(len(parent.SOURCES), len(arms), 2, 2, len(circuit_tags))
    batches = documents // parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches, "analytical": batches * (1 + len(parent.SOURCES) * len(arms))}
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["patches_expected"] = batches * len(parent.SOURCES) * (len(arms) - 1)
    diagnostics["patches_exact"] = diagnostics["patches"] == diagnostics["patches_expected"]
    return {
        "bounds": bounds, "arms": arms, "task": task, "task_counts": counts,
        "base_task": base_task, "source_task": task[:, 0],
        "circuit_tags": tuple(circuit_tags), "circuit_sums": circuit_sums,
        "circuit_counts": circuit_counts, "diagnostics": diagnostics,
    }


def collect_pairs(model, rows, task_masks, scales, bounds, pair_names):
    lo, hi, _split = bounds
    documents = hi - lo
    task = torch.zeros(len(parent.SOURCES), len(pair_names), documents,
                       len(parent.TASK_CELLS), dtype=torch.float64)
    counts = torch.zeros(documents, len(parent.TASK_CELLS), dtype=torch.float64)
    diagnostics = _collection_diagnostics()
    device = next(model.parameters()).device
    mlp = model.transformer.h[parent.TARGET].mlp
    pair_indices = tuple((GROUP_NAMES.index(left), GROUP_NAMES.index(right))
                         for left, right in pair_names)
    arms = tuple(f"{left}+{right}" for left, right in pair_names)
    for start in range(lo, hi, parent.BATCH):
        stop = start + parent.BATCH
        local = start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in parent.TASK_CELLS}
        _absent_logits, absent, absent_diag, _ = parent._forward(
            model, tokens, scales, action="P", absent=True, capture_mlp10=True)
        diagnostics["calls"]["analytical"] += 1
        _update_parent_diagnostics(diagnostics, absent_diag)
        counts[local:local + parent.BATCH] = torch.stack(
            [masks[cell].sum(1).double() for cell in parent.TASK_CELLS], -1)
        nll_rows = []
        for source in parent.SOURCES:
            _logits, current, current_diag, _ = parent._forward(
                model, tokens, scales, action=source, capture_mlp10=True)
            diagnostics["calls"]["analytical"] += 1
            _update_parent_diagnostics(diagnostics, current_diag)
            parent._score_delta_closure(diagnostics, current, absent)
            current_groups, current_sum = _family_outputs(mlp, current["factors"])
            absent_groups, absent_sum = _family_outputs(mlp, absent["factors"])
            diagnostics["family_partition_relative_squared"] = max(
                diagnostics["family_partition_relative_squared"],
                parent._relative_squared(current_sum,
                                         current["factors"]["left"].sum(2)
                                         * current["factors"]["right"].sum(2)),
                parent._relative_squared(absent_sum,
                                         absent["factors"]["left"].sum(2)
                                         * absent["factors"]["right"].sum(2)))
            group_deltas = current_groups - absent_groups
            diagnostics["family_score_delta_relative_squared"] = max(
                diagnostics["family_score_delta_relative_squared"],
                parent._relative_squared(
                    group_deltas.sum(2), current["factors"]["semantic_output"]
                    - absent["factors"]["semantic_output"]))
            for left_index, right_index in pair_indices:
                delta = group_deltas[:, :, left_index] + group_deltas[:, :, right_index]
                replacement = current["deployed_write"] - delta.to(
                    current["deployed_write"].dtype)
                patched_logits, _captures, patch_diag, patch_audit = \
                    parent.score_parent.run_forward(
                        model, tokens, action=source, scales=scales,
                        patch_writes={"m10": replacement})
                diagnostics["calls"]["analytical"] += 1
                diagnostics["patches"] += patch_audit["patches"]
                edit_rms = patch_diag["patch_rms_max"]
                diagnostics["zero_term_edits"] += int(edit_rms <= 0)
                if edit_rms > 0:
                    diagnostics["minimum_nonzero_term_edit_rms"] = min(
                        diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
                nll_rows.append(parent._nll(patched_logits, batch_rows).detach().cpu())
        nll_stack = torch.stack(nll_rows).view(
            len(parent.SOURCES), len(arms), parent.BATCH, parent.TOKENS)
        task[:, :, local:local + parent.BATCH] = parent._task_sums(
            nll_stack.view(-1, parent.BATCH, parent.TOKENS), masks).view(
                len(parent.SOURCES), len(arms), parent.BATCH, len(parent.TASK_CELLS))
    batches = documents // parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": 0,
        "analytical": batches * (1 + len(parent.SOURCES) * (1 + len(arms))),
    }
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["patches_expected"] = batches * len(parent.SOURCES) * len(arms)
    diagnostics["patches_exact"] = diagnostics["patches"] == diagnostics["patches_expected"]
    return {"bounds": bounds, "arms": arms, "task": task, "task_counts": counts,
            "diagnostics": diagnostics}


def finite_vector(collection, arm, intact, source, window="pooled", context_only=True):
    return parent.finite_vector(collection, arm, intact, source, window, context_only)


def finite_all_off(collection, arm, intact, source, window="pooled"):
    return parent.finite_all_off(collection, arm, intact, source, window)


def _term_checks(discovery):
    checks, selected = {}, []
    for term in GROUP_NAMES:
        row = {"sources": {}, "source_comparisons": {}}
        holds = True
        for source in parent.SOURCES:
            vector = finite_vector(discovery, term, discovery, source)
            full = finite_vector(discovery, FULL_NAMED, discovery, source)
            repeat = parent._comparison(
                finite_vector(discovery, term, discovery, source, "half0"),
                finite_vector(discovery, term, discovery, source, "half1"))
            projection = abs(float(torch.dot(vector, full)
                                   / torch.dot(full, full).clamp_min(1e-30)))
            all_copy, off_target = finite_all_off(discovery, term, discovery, source)
            source_holds = bool(
                float(torch.linalg.vector_norm(vector)) >= .00025
                and projection >= .05 and repeat["cosine"] >= .50
                and repeat["norm_ratio"] <= 3 and abs(all_copy) >= .00025
                and abs(all_copy) >= 2 * abs(off_target))
            row["sources"][source] = {
                "task_vector_nat": vector.tolist(),
                "task_norm_nat": float(torch.linalg.vector_norm(vector)),
                "projection_fraction_on_full_named": projection,
                "repeat": repeat, "all_copy_effect_nat": all_copy,
                "off_target_effect_nat": off_target, "holds": source_holds,
            }
            holds &= source_holds
        native = finite_vector(discovery, term, discovery, "N")
        for source in parent.SOURCES[1:]:
            metric = parent._comparison(
                native, finite_vector(discovery, term, discovery, source))
            metric["holds"] = bool(metric["cosine"] >= .70 and metric["norm_ratio"] <= 3)
            row["source_comparisons"][f"N:{source}"] = metric
            holds &= metric["holds"]
        row["holds"] = bool(holds)
        checks[term] = row
        if holds:
            selected.append(term)
    return selected, checks


def _confirm(discovery, confirmation, terms):
    checks, confirmed = {}, []
    for term in terms:
        row = {"sources": {}, "source_comparisons": {}}
        holds = True
        for source in parent.SOURCES:
            before = finite_vector(discovery, term, discovery, source)
            vector = finite_vector(confirmation, term, confirmation, source)
            transfer = parent._comparison(before, vector)
            repeat = parent._comparison(
                finite_vector(confirmation, term, confirmation, source, "half0"),
                finite_vector(confirmation, term, confirmation, source, "half1"))
            all_copy, off_target = finite_all_off(confirmation, term, confirmation, source)
            source_holds = bool(
                float(torch.linalg.vector_norm(vector)) >= .00025
                and transfer["cosine"] >= .60 and transfer["norm_ratio"] <= 3
                and repeat["cosine"] >= .50 and repeat["norm_ratio"] <= 3
                and abs(all_copy) >= .00025 and abs(all_copy) >= 2 * abs(off_target))
            row["sources"][source] = {
                "task_vector_nat": vector.tolist(), "discovery_transfer": transfer,
                "repeat": repeat, "all_copy_effect_nat": all_copy,
                "off_target_effect_nat": off_target, "holds": source_holds,
            }
            holds &= source_holds
        native = finite_vector(confirmation, term, confirmation, "N")
        for source in parent.SOURCES[1:]:
            metric = parent._comparison(
                native, finite_vector(confirmation, term, confirmation, source))
            metric["holds"] = bool(metric["cosine"] >= .65 and metric["norm_ratio"] <= 3)
            row["source_comparisons"][f"N:{source}"] = metric
            holds &= metric["holds"]
        row["holds"] = bool(holds)
        checks[term] = row
        if holds:
            confirmed.append(term)
    return confirmed, checks


def _score_pair(discovery, confirmation, discovery_pairs, confirmation_pairs,
                left, right, rule):
    name = f"{left}+{right}"
    row = {"sources": {}, "same_output": True}
    holds = bool(rule["identified"])
    for source in parent.SOURCES:
        left_vector = finite_vector(confirmation, left, confirmation, source)
        right_vector = finite_vector(confirmation, right, confirmation, source)
        joint = finite_vector(confirmation_pairs, name, confirmation, source)
        predicted = parent.predict_composition(rule, left_vector, right_vector)
        prediction_cosine = parent.state_parent.cosine(predicted, joint)
        prediction_residual = parent._relative_residual(joint, predicted)
        half_prediction_cosines = []
        for window in ("half0", "half1"):
            half_left = finite_vector(confirmation, left, confirmation, source, window)
            half_right = finite_vector(confirmation, right, confirmation, source, window)
            half_joint = finite_vector(
                confirmation_pairs, name, confirmation, source, window)
            half_prediction_cosines.append(parent.state_parent.cosine(
                parent.predict_composition(rule, half_left, half_right), half_joint))
        all_copy, off_target = finite_all_off(
            confirmation_pairs, name, confirmation, source)
        discovery_same = parent.state_parent.cosine(
            finite_vector(discovery, left, discovery, source),
            finite_vector(discovery, right, discovery, source)) >= .80
        confirmation_same = parent.state_parent.cosine(left_vector, right_vector) >= .80
        source_holds = bool(
            prediction_cosine >= .70 and prediction_residual <= .65
            and min(half_prediction_cosines) > 0 and abs(all_copy) >= .00025
            and abs(all_copy) >= 2 * abs(off_target))
        row["sources"][source] = {
            "prediction_cosine": prediction_cosine,
            "prediction_relative_residual": prediction_residual,
            "half_prediction_cosines": half_prediction_cosines,
            "discovery_singleton_same_output": discovery_same,
            "confirmation_singleton_same_output": confirmation_same,
            "all_copy_effect_nat": all_copy, "off_target_effect_nat": off_target,
            "holds": source_holds,
        }
        holds &= source_holds
        row["same_output"] &= discovery_same and confirmation_same
    row["holds"] = bool(holds)
    return row


def _instrument(collection, *, singleton):
    d = collection["diagnostics"]
    base = bool(
        d["calls_exact"] and d["patches_exact"] and d["zero_term_edits"] == 0
        and d["factor_reconstruction_max"] <= 1e-10
        and d["raw_source_relative_squared"] <= parent.DEPLOYED_BF16_BAR
        and d["normalized_closure_relative_squared"] <= 1e-12
        and d["normalized_numerical_rms_ratio"] <= .02
        and d["float32_mlp10_closure"] <= 1e-10
        and d["deployed_mlp10_relative_squared"] <= parent.DEPLOYED_BF16_BAR
        and d["score_delta_float32_closure"] <= 1e-10
        and math.isfinite(d["score_delta_predeployment_relative_squared"])
        and d["score_delta_deployed_closure_relative_squared"] <= 1e-12
        and d["family_partition_relative_squared"] <= 1e-10
        and d["family_score_delta_relative_squared"] <= 1e-10
        and d["minimum_nonzero_score_edit_rms"] > 0
        and d["minimum_nonzero_term_edit_rms"] > 0)
    if singleton:
        return bool(base and d["native_replay_logit_max_abs"] == 0.0
                    and d["native_replay_relative_squared"] <= 1e-12)
    return base


def _bundle_collection(collection):
    return {key: collection[key] for key in collection if key != "diagnostics"}


def _gpu_smoke():
    """Exercise one managed batch while retaining no family task-effect values."""
    rows, task_masks, circuit_masks, scales, circuit_tags, _metadata = validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bounds = (500, 504, 502)
    singletons = collect_singletons(
        model, rows, task_masks, circuit_masks, circuit_tags[:2], scales, bounds)
    pairs = collect_pairs(
        model, rows, task_masks, scales, bounds,
        (("A_eqxA_eq", "A_eqxM_post"),))
    single_ok = _instrument(singletons, singleton=True)
    pair_ok = _instrument(pairs, singleton=False)
    checks = {
        "weights": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256,
        "singletons": single_ok,
        "joint": pair_ok,
        "singletons_all_21_plus_full": singletons["diagnostics"]["patches"] == 88,
        "one_joint_per_source": pairs["diagnostics"]["patches"] == 4,
    }
    passed = all(checks.values())
    print(json.dumps({
        "status": "smoke_passed" if passed else "smoke_failed", "rung": 508,
        "scientific_outcomes_retained": False, "checks": checks,
        "singleton_diagnostics": singletons["diagnostics"],
        "pair_diagnostics": pairs["diagnostics"],
        "full_forwards": sum(sum(collection["diagnostics"]["calls"].values())
                             for collection in (singletons, pairs)),
        "backwards": 0,
    }, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(
            f"rung508 CUDA smoke failed: "
            f"{sorted(name for name, value in checks.items() if not value)}")


def main():
    started = time.time()
    rows, task_masks, circuit_masks, scales, circuit_tags, metadata = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        assert len(FAMILY_NAMES) == 6 and len(GROUP_NAMES) == 21
        assert sorted(index for spec in GROUP_SPECS for index in spec) == list(range(253))
        assert 12276 + 496 * math.comb(8, 2) == 26164
        print(json.dumps({
            "status": "dry_run_passed", "rung": 508, "model_loaded": False,
            "outcomes_opened": False, "families": FAMILIES,
            "family_term_count": len(GROUP_NAMES),
            "maximum_conditional_forwards": 26164,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung508 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    collections = {}
    collections["discovery"] = collect_singletons(
        model, rows, task_masks, circuit_masks, circuit_tags, scales, DISCOVERY)
    discovery_calibration = parent._calibration(
        collections["discovery"]["base_task"], collections["discovery"]["source_task"],
        collections["discovery"]["task_counts"], DISCOVERY)
    discovery_calibration_ok = parent.state_parent.calibration_holds(discovery_calibration)
    selected, discovery_checks = _term_checks(collections["discovery"])
    discovery_identifying = 2 <= len(selected) <= MAX_TERMS

    confirmed, confirmation_checks = [], {}
    confirmation_calibration, confirmation_calibration_ok = {}, False
    rules, composition_checks, predictable_pairs = {}, {}, []
    same_output_pairs, relationships = [], {}
    if discovery_calibration_ok and discovery_identifying:
        collections["confirmation"] = collect_singletons(
            model, rows, task_masks, circuit_masks, circuit_tags, scales, CONFIRMATION)
        confirmation_calibration = parent._calibration(
            collections["confirmation"]["base_task"],
            collections["confirmation"]["source_task"],
            collections["confirmation"]["task_counts"], CONFIRMATION)
        confirmation_calibration_ok = parent.state_parent.calibration_holds(
            confirmation_calibration)
        confirmed, confirmation_checks = _confirm(
            collections["discovery"], collections["confirmation"], selected)
        if confirmation_calibration_ok and 2 <= len(confirmed) <= MAX_TERMS:
            term_pairs = tuple(itertools.combinations(confirmed, 2))
            collections["pair_discovery"] = collect_pairs(
                model, rows, task_masks, scales, DISCOVERY, term_pairs)
            collections["pair_confirmation"] = collect_pairs(
                model, rows, task_masks, scales, CONFIRMATION, term_pairs)
            for left, right in term_pairs:
                name = f"{left}+{right}"
                rules[name] = parent.fit_composition(
                    collections["discovery"], collections["pair_discovery"], left, right)
                relationships[name] = _group_relationship(left, right)
                if rules[name]["identified"]:
                    composition_checks[name] = _score_pair(
                        collections["discovery"], collections["confirmation"],
                        collections["pair_discovery"], collections["pair_confirmation"],
                        left, right, rules[name])
                    if composition_checks[name]["holds"]:
                        predictable_pairs.append(name)
                    if composition_checks[name]["holds"] \
                            and composition_checks[name]["same_output"]:
                        same_output_pairs.append(name)

    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and _instrument(collections["discovery"], singleton=True)
        and all(_instrument(collection, singleton=name == "confirmation")
                for name, collection in collections.items() if name != "discovery"))
    pred_b = bool(pred_a and discovery_calibration_ok and discovery_identifying)
    pred_c = bool(pred_b and confirmation_calibration_ok
                  and 2 <= len(confirmed) <= MAX_TERMS)
    pred_d = bool(pred_c and predictable_pairs)
    pred_e = bool(pred_c and any("A_eq" in term.split("x") for term in confirmed))
    strong_null = not (pred_a and pred_b and pred_c and pred_d and pred_e)

    if not pred_a:
        next_step = "repair_family_algebra_or_intervention_only"
    elif not discovery_calibration_ok or not confirmation_calibration_ok and "confirmation" in collections:
        next_step = "stop_family_assay_preserve_score_gauge"
    elif not discovery_identifying and len(selected) < 2:
        next_step = "coupled_left_right_output_dictionary_with_finite_prediction"
    elif not discovery_identifying:
        next_step = "add_independent_task_outcomes_without_best_eight"
    elif not pred_c:
        next_step = "preserve_architecture_family_split_as_corpus_specific_screen"
    elif not pred_d:
        next_step = "model_higher_order_suffix_state_dependence"
    elif not pred_e:
        next_step = "audit_normalization_mediated_route_before_semantic_label"
    else:
        next_step = "refine_confirmed_families_and_build_executable_mlp10_candidate"

    bundle_payload = {
        "schema": "rung508_exact_mlp10_source_family_finite_stats_v1",
        "collections": {name: _bundle_collection(collection)
                        for name, collection in collections.items()},
        "raw_tokens_logits_hidden_states_or_weights_included": False,
        "confirmation_opened": "confirmation" in collections,
        "pair_outcomes_opened": "pair_confirmation" in collections,
    }
    torch.save(bundle_payload, BUNDLE)
    result = {
        "status": "complete", "rung": 508,
        "claim_level": "finite_exact_source_family_split_not_executable_replacement",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "families": FAMILIES,
        "family_terms": list(GROUP_NAMES), "score_sources": list(parent.SOURCES),
        "calibration": {"discovery": discovery_calibration,
                        "confirmation": confirmation_calibration},
        "diagnostics": {name: collection["diagnostics"]
                        for name, collection in collections.items()},
        "analysis": {
            "discovery_terms": selected, "discovery_identifying": discovery_identifying,
            "discovery_checks": discovery_checks,
            "confirmed_terms": confirmed, "confirmation_checks": confirmation_checks,
            "composition_rules": rules, "composition_checks": composition_checks,
            "predictable_composition_pairs": predictable_pairs,
            "same_output_pairs": same_output_pairs,
            "input_sharing_relationships": relationships,
        },
        'pred_a_exact_live_finite_source_family_instrument': pred_a,
        'pred_b_sparse_finite_family_split': pred_b,
        'pred_c_at_least_two_family_terms_confirm': pred_c,
        'pred_d_pair_composition_predicts_confirmation': pred_d,
        'pred_e_equality_attention_family_participates': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "full_forwards": sum(sum(collection["diagnostics"]["calls"].values())
                                 for collection in collections.values()),
            "backwards": 0, "selected_count_k": len(selected),
            "confirmed_count_q": len(confirmed),
            "confirmed_pair_count": math.comb(len(confirmed), 2),
            "maximum_conditional_forwards": 26164,
            "fitted_vectors": 0,
            "fitted_scalars": sum(rule["kind"] == "one_scalar_interaction"
                                  for rule in rules.values()),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started, "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 508,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "discovery_terms": selected, "confirmed_terms": confirmed,
        "predictable_pairs": predictable_pairs, "same_output_pairs": same_output_pairs,
        "execution_price": result["execution_price"], "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
