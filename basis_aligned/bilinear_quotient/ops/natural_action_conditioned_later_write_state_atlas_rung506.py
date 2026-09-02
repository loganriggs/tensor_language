#!/usr/bin/env python3
"""RUNG506 -- finite downstream-effect grouping of all 19 later writes."""

# BQGATE: EXPERIMENT
# pred_a: exact live conditional finite-intervention instrument
# pred_b: four score actions remain calibrated on new documents
# pred_c: at least one no-ranking whole-write edge confirms
# pred_d: at least one edge validates on held-out circuit families and documents
# pred_e: at least one validated edge composes and removes selectively

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_product_circuit_response_graph_rung477 as circuit_parent
import equality_score_gauged_downstream_program_rung505 as parent


PREREG = POLY / "NATURAL_ACTION_CONDITIONED_LATER_WRITE_STATE_ATLAS_RUNG506_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/equality_score_gauged_downstream_program_rung505.py"
PARENT_RESULT = ROOT / "equality_score_gauged_downstream_program_rung505_results.json"
CIRCUIT_SOURCE = ROOT / "ops/equality_product_circuit_response_graph_rung477.py"
CENSUS = ROOT / "census_state_diverse.pt"
BATTERY = ROOT / "circuits/BATTERY.json"
CLAUSE_AUDIT = ROOT / "rung505_clause_audit_results.json"
CROSS_CORPUS_AUDIT = ROOT / "rung505_cross_corpus_component_audit_results.json"
OUT = ROOT / "natural_action_conditioned_later_write_state_atlas_rung506_results.json"
BUNDLE = ROOT / "natural_action_conditioned_later_write_state_atlas_rung506_bundle.pt"
SMOKE_OUT = ROOT / "natural_action_conditioned_later_write_state_atlas_rung506_gpu_smoke_results.json"
HASHES = {
    PREREG: "4870c6fc89ecc6412e94cdb9ba32c6bbb8148543a4b11a650ef1e53bf5fc898a",
    PARENT_SOURCE: "0c5f6679ec40cb02bd6af1e28b0b41ca2ad7967fd4b6c9d73a4f388153f3e4de",
    PARENT_RESULT: "3720a2feb24fc5ec4554d858a00a576a1fcd44f0e789d2b728e66483d7d8d1a1",
    CIRCUIT_SOURCE: "7c76b115977ab102884c2233e31a284e3d509ca2b3ed9291cefe1d47562aa770",
    CENSUS: "c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b",
    BATTERY: "86d7ac72eeb95f9ec80a3e92ef65e28c0df66a36b9291d2d1d2d01f7bb6c5030",
    CLAUSE_AUDIT: "4a2b7b343f7bc3a2667bfe3b082e22c69ae2cc098c12a8f34c27d45eb337de16",
    CROSS_CORPUS_AUDIT: "10bb665b742c7c3a2642fc6841c19a886a68f1e1f7483d37805c312be939dcfd",
}

SOURCES = parent.SOURCES
SITES = parent.ALL_LATER_SITES
CELLS = parent.CELLS
TASK_CONTEXT_INDICES = tuple(CELLS.index(name) for name in parent.CONTEXT_CELLS)
MASK_TYPES = ("member", "slice_control")
BATCH = 4
TOKENS = 256
DISCOVERY = (0, 248, 124)
CONFIRMATION = (248, 496, 372)
VALIDATION = (500, 1000, 750)
DISCOVERY_ROOTS = circuit_parent.DISCOVERY_ROOTS
VALIDATION_ROOTS = circuit_parent.VALIDATION_ROOTS
MAX_EDGES = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def edge_name(left: str, right: str) -> str:
    if SITES.index(left) >= SITES.index(right):
        raise ValueError("edge site order changed")
    return f"{left}+{right}"


def parse_edge(name: str) -> tuple[str, str]:
    left, right = name.split("+")
    if edge_name(left, right) != name:
        raise ValueError("malformed edge name")
    return left, right


def _root(tag: str) -> int:
    return int(tag.split(".")[1])


def _support_report(circuit_masks, tags, bounds):
    lo, hi, split = bounds
    report = {}
    for left, right in ((lo, split), (split, hi)):
        key = f"{left}:{right}"
        report[key] = {
            kind: {
                tag: int(circuit_masks[tag][kind].view(1000, TOKENS)[left:right].sum())
                for tag in tags
            }
            for kind in MASK_TYPES
        }
    return report


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(PARENT_RESULT.read_text())
    if not (
        receipt.get("pred_a_exact_live_intervention_instrument") is True
        and receipt.get("pred_b_score_actions_calibrated_in_patch_harness") is True
        and receipt.get("pred_c_fixed_program_transfers_code_to_natural") is False
        and receipt.get("pred_d_program_invariant_across_sign_gauge") is False
        and receipt.get("pred_e_correct_gauge_orientation_specific") is True
        and receipt.get("next_step") == "abandon_fixed_five_site_program_as_code_specific"
    ):
        raise RuntimeError("rung505 result route changed")
    clause = json.loads(CLAUSE_AUDIT.read_text())
    if clause.get("new_model_outcomes_opened") is not False or any(
        row["task_sign_pattern_holds"] for row in clause["source_clause_checks"].values()
    ):
        raise RuntimeError("rung505 clause correction changed")

    rows, task_masks, scales, parent_metadata = parent.validate_inputs()
    c_rows, _positive, circuit_masks, _scale, discovery_tags, validation_tags, circuit_metadata = (
        circuit_parent.validate_inputs()
    )
    if not torch.equal(rows, c_rows):
        raise RuntimeError("task and circuit row authorities differ")
    if tuple(tag for tag in circuit_masks if _root(tag) in DISCOVERY_ROOTS) != tuple(discovery_tags):
        raise RuntimeError("discovery circuit order changed")
    if tuple(tag for tag in circuit_masks if _root(tag) in VALIDATION_ROOTS) != tuple(validation_tags):
        raise RuntimeError("validation circuit order changed")
    if len(discovery_tags) != 32 or len(validation_tags) != 30:
        raise RuntimeError("62-circuit partition changed")
    supports = {
        "discovery": _support_report(circuit_masks, discovery_tags, DISCOVERY),
        "confirmation": _support_report(circuit_masks, discovery_tags, CONFIRMATION),
        "validation": _support_report(circuit_masks, validation_tags, VALIDATION),
    }
    observed_minima = {
        phase: {
            kind: [min(part[kind].values()) for part in report.values()]
            for kind in MASK_TYPES
        }
        for phase, report in supports.items()
    }
    expected_minima = {
        "discovery": {"member": [27, 27], "slice_control": [213, 193]},
        "confirmation": {"member": [16, 23], "slice_control": [208, 223]},
        "validation": {"member": [32, 46], "slice_control": [343, 372]},
    }
    if observed_minima != expected_minima:
        raise RuntimeError(f"precomputed circuit support changed: {observed_minima}")
    task_support = {
        phase: {
            f"{left}:{right}": {cell: int(task_masks[cell][left:right].sum()) for cell in CELLS}
            for left, right in ((bounds[0], bounds[2]), (bounds[2], bounds[1]))
        }
        for phase, bounds in (
            ("discovery", DISCOVERY), ("confirmation", CONFIRMATION),
            ("validation", VALIDATION),
        )
    }
    if min(value for phase in task_support.values() for part in phase.values()
           for value in part.values()) <= 0:
        raise RuntimeError("task support changed")
    return rows, task_masks, circuit_masks, scales, list(discovery_tags), list(validation_tags), {
        "parent": parent_metadata,
        "circuit": circuit_metadata,
        "documents": {
            "discovery": list(DISCOVERY), "confirmation": list(CONFIRMATION),
            "unused": [496, 500], "validation": list(VALIDATION),
        },
        "circuit_support": supports,
        "circuit_support_minima": observed_minima,
        "task_support": task_support,
    }


def _nll(logits: torch.Tensor, batch_rows: torch.Tensor) -> torch.Tensor:
    targets = batch_rows[:, 1:].to(logits.device)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view(len(batch_rows), -1).float().cpu()


def _phase_slices(bounds):
    lo, hi, split = bounds
    return ((lo, split), (split, hi))


def _empty_collection(bounds, tags, patch_sets, include_instrument):
    lo, hi, _split = bounds
    documents = hi - lo
    arms = tuple("intact" if not sites else edge_name(*sites) if len(sites) == 2 else sites[0]
                 for sites in patch_sets)
    if len(set(arms)) != len(arms):
        raise ValueError("duplicate patch arm")
    return {
        "bounds": tuple(bounds),
        "tags": tuple(tags),
        "arms": arms,
        "patch_sets": tuple(tuple(sites) for sites in patch_sets),
        "base_task": torch.zeros(documents, len(CELLS), dtype=torch.float64),
        "source_task": torch.zeros(
            len(SOURCES), len(arms), documents, len(CELLS), dtype=torch.float64),
        "task_counts": torch.zeros(documents, len(CELLS), dtype=torch.float64),
        "source_circuit_sums": torch.zeros(
            len(SOURCES), len(arms), 2, len(MASK_TYPES), len(tags), dtype=torch.float64),
        "circuit_counts": torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64),
        "diagnostics": {
            "include_instrument": include_instrument,
            "calls": {"native": 0, "analytical": 0},
            "captures": 0,
            "patches": 0,
            "native_replay_logit_max_abs": 0.0,
            "native_replay_relative_squared": 0.0,
            "factor_reconstruction_max": 0.0,
            "minimum_nonzero_score_edit_rms": math.inf,
            "patch_rms_max": {
                source: {arm: 0.0 for arm in arms if arm != "intact"} for source in SOURCES
            },
        },
    }


def _task_sums(nll_stack, local_task_masks):
    mask = torch.stack([local_task_masks[cell].double() for cell in CELLS], dim=-1)
    return torch.einsum("abp,bpc->abc", nll_stack.double(), mask)


def _circuit_mask_matrix(circuit_masks, tags, start, stop, bounds):
    rows = torch.arange(start, stop)
    vectors = []
    counts = torch.zeros(2, len(MASK_TYPES), len(tags), dtype=torch.float64)
    for half, (lo, hi) in enumerate(_phase_slices(bounds)):
        in_half = ((rows >= lo) & (rows < hi))[:, None]
        for kind_index, kind in enumerate(MASK_TYPES):
            for tag_index, tag in enumerate(tags):
                selected = circuit_masks[tag][kind].view(1000, TOKENS)[start:stop] & in_half
                counts[half, kind_index, tag_index] += int(selected.sum())
                vectors.append(selected.flatten().double())
    return torch.stack(vectors), counts


@torch.no_grad()
def collect_phase(
    model,
    rows,
    task_masks,
    circuit_masks,
    tags,
    scales,
    bounds,
    patch_sets,
    *,
    include_instrument,
):
    data = _empty_collection(bounds, tags, patch_sets, include_instrument)
    diagnostics = data["diagnostics"]
    device = next(model.parameters()).device
    lo_doc, hi_doc, _split = bounds
    for start in range(lo_doc, hi_doc, BATCH):
        stop = min(start + BATCH, hi_doc)
        local = start - lo_doc
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        local_task_masks = {cell: task_masks[cell][start:stop] for cell in CELLS}

        if include_instrument:
            native_logits, _, _, _audit = parent.run_forward(model, tokens, direct=True)
            replay_logits, _, replay_diag, _audit = parent.run_forward(model, tokens, action="N")
            diagnostics["calls"]["native"] += 1
            diagnostics["calls"]["analytical"] += 1
            difference = replay_logits.float() - native_logits.float()
            diagnostics["native_replay_logit_max_abs"] = max(
                diagnostics["native_replay_logit_max_abs"], float(difference.abs().max()))
            diagnostics["native_replay_relative_squared"] = max(
                diagnostics["native_replay_relative_squared"],
                float(difference.square().sum()) /
                max(float(native_logits.float().square().sum()), 1e-30),
            )
            diagnostics["factor_reconstruction_max"] = max(
                diagnostics["factor_reconstruction_max"], replay_diag["factor_reconstruction_max"])

        base_logits, absent_writes, base_diag, base_audit = parent.run_forward(
            model, tokens, action="P", absent=True, scales=scales,
            capture_keys=SITES,
        )
        diagnostics["calls"]["analytical"] += 1
        diagnostics["captures"] += base_audit["captures"]
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], base_diag["factor_reconstruction_max"])
        if base_diag["late_edit_rms"] > 0:
            diagnostics["minimum_nonzero_score_edit_rms"] = min(
                diagnostics["minimum_nonzero_score_edit_rms"], base_diag["late_edit_rms"])
        base_nll = _nll(base_logits, batch_rows)
        data["base_task"][local:local + len(batch_rows)] = _task_sums(
            base_nll[None], local_task_masks)[0]
        observed_task = torch.stack(
            [local_task_masks[cell].sum(1).double() for cell in CELLS], dim=-1)
        data["task_counts"][local:local + len(batch_rows)] = observed_task
        circuit_matrix, circuit_counts = _circuit_mask_matrix(
            circuit_masks, tags, start, stop, bounds)
        data["circuit_counts"] += circuit_counts

        nll_rows = []
        for source in SOURCES:
            for arm, sites in zip(data["arms"], data["patch_sets"]):
                patches = {site: absent_writes[site] for site in sites}
                logits, _, diag, audit = parent.run_forward(
                    model, tokens, action=source, scales=scales, patch_writes=patches,
                )
                diagnostics["calls"]["analytical"] += 1
                diagnostics["patches"] += audit["patches"]
                diagnostics["factor_reconstruction_max"] = max(
                    diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
                if diag["late_edit_rms"] > 0:
                    diagnostics["minimum_nonzero_score_edit_rms"] = min(
                        diagnostics["minimum_nonzero_score_edit_rms"], diag["late_edit_rms"])
                if sites:
                    diagnostics["patch_rms_max"][source][arm] = max(
                        diagnostics["patch_rms_max"][source][arm], diag["patch_rms_max"])
                nll_rows.append(_nll(logits, batch_rows))
        nll_stack = torch.stack(nll_rows).view(
            len(SOURCES), len(data["arms"]), len(batch_rows), TOKENS)
        task_sums = _task_sums(
            nll_stack.view(-1, len(batch_rows), TOKENS), local_task_masks,
        ).view(len(SOURCES), len(data["arms"]), len(batch_rows), len(CELLS))
        data["source_task"][:, :, local:local + len(batch_rows)] = task_sums
        flattened = nll_stack.view(len(SOURCES) * len(data["arms"]), -1).double()
        circuit_sums = torch.matmul(flattened, circuit_matrix.T).view(
            len(SOURCES), len(data["arms"]), 2, len(MASK_TYPES), len(tags))
        data["source_circuit_sums"] += circuit_sums

    batches = math.ceil((hi_doc - lo_doc) / BATCH)
    expected = {
        "native": batches if include_instrument else 0,
        "analytical": batches * ((2 if include_instrument else 0) - (1 if include_instrument else 0)
                                  + 1 + len(SOURCES) * len(data["arms"])),
    }
    # The analytical term above is replay (when instrumented), score-absent capture, and all arms.
    expected["analytical"] = batches * (
        (1 if include_instrument else 0) + 1 + len(SOURCES) * len(data["arms"]))
    expected_captures = batches * len(SITES)
    expected_patches = batches * len(SOURCES) * sum(len(sites) for sites in data["patch_sets"])
    diagnostics["calls_expected"] = expected
    diagnostics["calls_exact"] = diagnostics["calls"] == expected
    diagnostics["captures_expected"] = expected_captures
    diagnostics["captures_exact"] = diagnostics["captures"] == expected_captures
    diagnostics["patches_expected"] = expected_patches
    diagnostics["patches_exact"] = diagnostics["patches"] == expected_patches
    return data


def cosine(left, right):
    left = torch.as_tensor(left, dtype=torch.float64)
    right = torch.as_tensor(right, dtype=torch.float64)
    denominator = float(torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right))
    return float(torch.dot(left, right)) / max(denominator, 1e-30)


def norm_ratio(left, right):
    left_norm = float(torch.linalg.vector_norm(torch.as_tensor(left, dtype=torch.float64)))
    right_norm = float(torch.linalg.vector_norm(torch.as_tensor(right, dtype=torch.float64)))
    return max(left_norm, right_norm) / max(min(left_norm, right_norm), 1e-30)


def rms(vector):
    vector = torch.as_tensor(vector, dtype=torch.float64)
    return float(torch.sqrt(torch.mean(vector.square())))


def comparison(left, right):
    left = torch.as_tensor(left, dtype=torch.float64)
    right = torch.as_tensor(right, dtype=torch.float64)
    return {
        "cosine": cosine(left, right),
        "norm_ratio": norm_ratio(left, right),
        "left_rms_nat": rms(left),
        "right_rms_nat": rms(right),
    }


def _arm_index(collection, arm):
    return collection["arms"].index(arm)


def _window_index(window):
    return {"half0": 0, "half1": 1}[window]


def task_vector(target, arm, intact, source, window, *, context_only=True):
    source_index = SOURCES.index(source)
    arm_index = _arm_index(target, arm)
    intact_index = _arm_index(intact, "intact")
    if window == "pooled":
        lo, hi = 0, target["source_task"].shape[2]
    else:
        bounds = target["bounds"]
        absolute = _phase_slices(bounds)[_window_index(window)]
        lo, hi = absolute[0] - bounds[0], absolute[1] - bounds[0]
    numerator = (
        target["source_task"][source_index, arm_index, lo:hi]
        - intact["source_task"][source_index, intact_index, lo:hi]
    ).sum(0)
    denominator = target["task_counts"][lo:hi].sum(0).clamp_min(1)
    vector = numerator / denominator
    return vector[list(TASK_CONTEXT_INDICES)] if context_only else vector


def circuit_fingerprint(target, arm, intact, source, window):
    source_index = SOURCES.index(source)
    arm_index = _arm_index(target, arm)
    intact_index = _arm_index(intact, "intact")
    if window == "pooled":
        target_sums = target["source_circuit_sums"][source_index, arm_index].sum(0)
        intact_sums = intact["source_circuit_sums"][source_index, intact_index].sum(0)
        counts = target["circuit_counts"].sum(0)
    else:
        half = _window_index(window)
        target_sums = target["source_circuit_sums"][source_index, arm_index, half]
        intact_sums = intact["source_circuit_sums"][source_index, intact_index, half]
        counts = target["circuit_counts"][half]
    effects = (target_sums - intact_sums) / counts.clamp_min(1)
    return effects[0] - effects[1]


def calibration(collection):
    reports = {}
    intact_index = _arm_index(collection, "intact")
    bounds = collection["bounds"]
    windows = {
        "half0": (0, bounds[2] - bounds[0]),
        "half1": (bounds[2] - bounds[0], bounds[1] - bounds[0]),
        "pooled": (0, bounds[1] - bounds[0]),
    }
    all_index = CELLS.index("all_positive")
    off_index = CELLS.index("off_target")
    for window, (lo, hi) in windows.items():
        reference_rows = collection["base_task"][lo:hi, all_index] - collection[
            "source_task"][SOURCES.index("N"), intact_index, lo:hi, all_index]
        reference_counts = collection["task_counts"][lo:hi, all_index]
        valid_reference = reference_counts > 0
        reference_doc_effect = reference_rows[valid_reference] / reference_counts[valid_reference]
        reference_mean = float(reference_rows.sum() / reference_counts.sum().clamp_min(1))
        reports[window] = {}
        for source in SOURCES:
            source_index = SOURCES.index(source)
            source_rows = collection["base_task"][lo:hi, all_index] - collection[
                "source_task"][source_index, intact_index, lo:hi, all_index]
            source_doc_effect = source_rows[valid_reference] / reference_counts[valid_reference]
            source_mean = float(source_rows.sum() / reference_counts.sum().clamp_min(1))
            off_delta = (
                collection["source_task"][source_index, intact_index, lo:hi, off_index]
                - collection["source_task"][SOURCES.index("N"), intact_index, lo:hi, off_index]
            ).sum() / collection["task_counts"][lo:hi, off_index].sum().clamp_min(1)
            reports[window][source] = {
                "all_copy_effect_nat": source_mean,
                "recovery_vs_native": source_mean / (
                    reference_mean if abs(reference_mean) > 1e-30 else 1e-30),
                "per_document_cosine_vs_native": cosine(source_doc_effect, reference_doc_effect),
                "off_target_minus_native_nat": float(off_delta),
            }
    return reports


def calibration_holds(reports):
    return all(
        .65 <= reports[window][source]["recovery_vs_native"] <= 1.40
        and reports[window][source]["per_document_cosine_vs_native"] >= .85
        and abs(reports[window][source]["off_target_minus_native_nat"]) <= .01
        for window in ("half0", "half1", "pooled")
        for source in SOURCES[1:]
    )


def discover_edges(collection):
    site_checks = {}
    eligible = []
    for site in SITES:
        checks = {"sources": {}, "source_comparisons": {}}
        holds = True
        for source in SOURCES:
            pooled = circuit_fingerprint(collection, site, collection, source, "pooled")
            repeat = comparison(
                circuit_fingerprint(collection, site, collection, source, "half0"),
                circuit_fingerprint(collection, site, collection, source, "half1"),
            )
            row = {
                "pooled_rms_nat": rms(pooled), "repeat": repeat,
                "holds": bool(rms(pooled) >= .0005 and repeat["cosine"] >= .50
                              and repeat["norm_ratio"] <= 3),
            }
            checks["sources"][source] = row
            holds &= row["holds"]
        for source in SOURCES[1:]:
            row = comparison(
                circuit_fingerprint(collection, site, collection, "N", "pooled"),
                circuit_fingerprint(collection, site, collection, source, "pooled"),
            )
            row["holds"] = bool(row["cosine"] >= .70 and row["norm_ratio"] <= 3)
            checks["source_comparisons"][f"N:{source}"] = row
            holds &= row["holds"]
        checks["eligible"] = bool(holds)
        site_checks[site] = checks
        if holds:
            eligible.append(site)

    pair_checks = {}
    edges = []
    for left, right in itertools.combinations(eligible, 2):
        name = edge_name(left, right)
        checks = {"sources": {}}
        holds = True
        for source in SOURCES:
            pooled = comparison(
                circuit_fingerprint(collection, left, collection, source, "pooled"),
                circuit_fingerprint(collection, right, collection, source, "pooled"),
            )
            repeats = [comparison(
                circuit_fingerprint(collection, left, collection, source, window),
                circuit_fingerprint(collection, right, collection, source, window),
            ) for window in ("half0", "half1")]
            task = comparison(
                task_vector(collection, left, collection, source, "pooled"),
                task_vector(collection, right, collection, source, "pooled"),
            )
            source_holds = bool(
                pooled["cosine"] >= .85 and pooled["norm_ratio"] <= 3
                and all(row["cosine"] >= .60 for row in repeats)
                and task["cosine"] >= .60)
            checks["sources"][source] = {
                "pooled_circuit": pooled, "repeat_circuit": repeats,
                "task": task, "holds": source_holds,
            }
            holds &= source_holds
        checks["holds"] = bool(holds)
        pair_checks[name] = checks
        if holds:
            edges.append(name)
    return eligible, edges, {"sites": site_checks, "pairs": pair_checks}


def confirm_edges(collection, edges):
    checks = {}
    confirmed = []
    for name in edges:
        left, right = parse_edge(name)
        row = {"sources": {}, "source_invariance": {}}
        holds = True
        for source in SOURCES:
            pooled = comparison(
                circuit_fingerprint(collection, left, collection, source, "pooled"),
                circuit_fingerprint(collection, right, collection, source, "pooled"),
            )
            repeat_cosines = [cosine(
                circuit_fingerprint(collection, left, collection, source, window),
                circuit_fingerprint(collection, right, collection, source, window),
            ) for window in ("half0", "half1")]
            task = comparison(
                task_vector(collection, left, collection, source, "pooled"),
                task_vector(collection, right, collection, source, "pooled"),
            )
            source_holds = bool(
                pooled["cosine"] >= .75 and pooled["norm_ratio"] <= 3
                and min(repeat_cosines) >= .50 and task["cosine"] >= .50)
            row["sources"][source] = {
                "pooled_circuit": pooled, "repeat_cosines": repeat_cosines,
                "task": task, "holds": source_holds,
            }
            holds &= source_holds
        for site in (left, right):
            row["source_invariance"][site] = {}
            native = circuit_fingerprint(collection, site, collection, "N", "pooled")
            for source in SOURCES[1:]:
                metric = comparison(
                    native, circuit_fingerprint(collection, site, collection, source, "pooled"))
                metric["holds"] = bool(metric["cosine"] >= .60 and metric["norm_ratio"] <= 3)
                row["source_invariance"][site][f"N:{source}"] = metric
                holds &= metric["holds"]
        row["holds"] = bool(holds)
        checks[name] = row
        if holds:
            confirmed.append(name)
    return confirmed, checks


def validate_edges(collection, edges):
    checks = {}
    validated = []
    for name in edges:
        left, right = parse_edge(name)
        row = {"sources": {}}
        holds = True
        for source in SOURCES:
            pooled = comparison(
                circuit_fingerprint(collection, left, collection, source, "pooled"),
                circuit_fingerprint(collection, right, collection, source, "pooled"),
            )
            repeat_cosines = [cosine(
                circuit_fingerprint(collection, left, collection, source, window),
                circuit_fingerprint(collection, right, collection, source, window),
            ) for window in ("half0", "half1")]
            task = comparison(
                task_vector(collection, left, collection, source, "pooled"),
                task_vector(collection, right, collection, source, "pooled"),
            )
            source_holds = bool(
                pooled["cosine"] >= .70 and pooled["norm_ratio"] <= 3
                and min(repeat_cosines) > 0 and task["cosine"] >= .50)
            row["sources"][source] = {
                "pooled_circuit": pooled, "repeat_cosines": repeat_cosines,
                "task": task, "holds": source_holds,
            }
            holds &= source_holds
        row["holds"] = bool(holds)
        checks[name] = row
        if holds:
            validated.append(name)
    return validated, checks


def _relative_residual(actual, predicted):
    actual = torch.as_tensor(actual, dtype=torch.float64)
    predicted = torch.as_tensor(predicted, dtype=torch.float64)
    return float(torch.linalg.vector_norm(actual - predicted) /
                 torch.linalg.vector_norm(actual).clamp_min(1e-30))


def fit_composition_rule(singletons, pairs, name):
    left, right = parse_edge(name)
    lefts, rights, joints = [], [], []
    for source in SOURCES:
        lefts.append(circuit_fingerprint(singletons, left, singletons, source, "pooled"))
        rights.append(circuit_fingerprint(singletons, right, singletons, source, "pooled"))
        joints.append(circuit_fingerprint(pairs, name, singletons, source, "pooled"))
    left_vector = torch.cat(lefts)
    right_vector = torch.cat(rights)
    joint_vector = torch.cat(joints)
    summed = left_vector + right_vector
    interaction = joint_vector - summed
    joint_norm = torch.linalg.vector_norm(joint_vector).clamp_min(1e-30)
    interaction_ratio = float(torch.linalg.vector_norm(interaction) / joint_norm)
    left_residual = float(torch.linalg.vector_norm(joint_vector - left_vector) / joint_norm)
    right_residual = float(torch.linalg.vector_norm(joint_vector - right_vector) / joint_norm)
    row = {
        "interaction_over_joint": interaction_ratio,
        "left_redundancy_residual": left_residual,
        "right_redundancy_residual": right_residual,
    }
    if interaction_ratio <= .25:
        row.update({"kind": "additive", "beta": 0.0, "identified": True})
    elif left_residual <= .25:
        row.update({"kind": "left_redundant", "beta": None, "identified": True})
    elif right_residual <= .25:
        row.update({"kind": "right_redundant", "beta": None, "identified": True})
    else:
        beta = float(torch.dot(interaction, summed) / torch.dot(summed, summed).clamp_min(1e-30))
        scalar_residual = _relative_residual(interaction, beta * summed)
        identified = bool(abs(beta) >= .25 and -.8 <= beta <= 2 and scalar_residual <= .50)
        row.update({
            "kind": "one_scalar_interaction" if identified else "none",
            "beta": beta, "scalar_interaction_residual": scalar_residual,
            "identified": identified,
        })
    return row


def predict_joint(rule, left, right):
    if rule["kind"] == "additive":
        return left + right
    if rule["kind"] == "left_redundant":
        return left
    if rule["kind"] == "right_redundant":
        return right
    if rule["kind"] == "one_scalar_interaction":
        return (1 + rule["beta"]) * (left + right)
    raise ValueError("unidentified composition rule")


def score_composition(singletons, pairs, name, rule):
    row = {"sources": {}}
    holds = bool(rule["identified"])
    for source in SOURCES:
        left, right = parse_edge(name)
        left_circuit = circuit_fingerprint(singletons, left, singletons, source, "pooled")
        right_circuit = circuit_fingerprint(singletons, right, singletons, source, "pooled")
        joint_circuit = circuit_fingerprint(pairs, name, singletons, source, "pooled")
        predicted_circuit = predict_joint(rule, left_circuit, right_circuit)
        left_task = task_vector(singletons, left, singletons, source, "pooled")
        right_task = task_vector(singletons, right, singletons, source, "pooled")
        joint_task = task_vector(pairs, name, singletons, source, "pooled")
        predicted_task = predict_joint(rule, left_task, right_task)
        full_task = task_vector(
            pairs, name, singletons, source, "pooled", context_only=False)
        all_copy = abs(float(full_task[CELLS.index("all_positive")]))
        off_target = abs(float(full_task[CELLS.index("off_target")]))
        source_holds = bool(
            cosine(predicted_circuit, joint_circuit) >= .70
            and _relative_residual(joint_circuit, predicted_circuit) <= .65
            and cosine(predicted_task, joint_task) >= .60
            and _relative_residual(joint_task, predicted_task) <= .75
            and all_copy >= .002 and all_copy >= 3 * off_target)
        row["sources"][source] = {
            "circuit_prediction_cosine": cosine(predicted_circuit, joint_circuit),
            "circuit_prediction_relative_residual": _relative_residual(
                joint_circuit, predicted_circuit),
            "task_prediction_cosine": cosine(predicted_task, joint_task),
            "task_prediction_relative_residual": _relative_residual(joint_task, predicted_task),
            "all_copy_absolute_effect_nat": all_copy,
            "off_target_absolute_effect_nat": off_target,
            "selectivity_ratio": all_copy / max(off_target, 1e-30),
            "holds": source_holds,
        }
        holds &= source_holds
    row["holds"] = bool(holds)
    return row


def phase_instrument_holds(collection):
    diagnostics = collection["diagnostics"]
    replay_holds = (
        not diagnostics["include_instrument"]
        or (diagnostics["native_replay_relative_squared"] <= 1e-12
            and diagnostics["native_replay_logit_max_abs"] == 0.0)
    )
    return bool(
        replay_holds
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["minimum_nonzero_score_edit_rms"] > 0
        and diagnostics["calls_exact"] and diagnostics["captures_exact"]
        and diagnostics["patches_exact"]
        and torch.isfinite(collection["base_task"]).all()
        and torch.isfinite(collection["source_task"]).all()
        and torch.isfinite(collection["source_circuit_sums"]).all()
    )


def _serial_collection(collection):
    return {
        "bounds": list(collection["bounds"]), "tags": list(collection["tags"]),
        "arms": list(collection["arms"]),
        "patch_sets": [list(row) for row in collection["patch_sets"]],
        "task_counts": collection["task_counts"].tolist(),
        "circuit_counts": collection["circuit_counts"].tolist(),
        "diagnostics": collection["diagnostics"],
    }


def _bundle_collection(collection):
    return {
        key: collection[key] for key in (
            "bounds", "tags", "arms", "patch_sets", "base_task", "source_task",
            "task_counts", "source_circuit_sums", "circuit_counts",
        )
    }


def _gpu_smoke():
    """Exercise CUDA dispatch, capture, singleton, and pair patches without retaining outcomes."""
    if SMOKE_OUT.exists():
        raise RuntimeError("rung506 smoke namespace already exists")
    rows, task_masks, circuit_masks, scales, discovery_tags, _validation_tags, metadata = validate_inputs()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    collection = collect_phase(
        model, rows, task_masks, circuit_masks, discovery_tags[:2], scales,
        (0, 4, 2), ((), ("m8",), ("m8", "a9")), include_instrument=True)
    diagnostics = collection["diagnostics"]
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and phase_instrument_holds(collection))
    pred_b = bool(
        collection["source_task"].shape == (4, 3, 4, 6)
        and collection["source_circuit_sums"].shape == (4, 3, 2, 2, 2))
    pred_c = bool(
        diagnostics["calls"] == {"native": 1, "analytical": 14}
        and diagnostics["captures"] == 19 and diagnostics["patches"] == 12
        and all(diagnostics["patch_rms_max"][source]["m8"] > 0 for source in SOURCES)
        and all(diagnostics["patch_rms_max"][source]["m8+a9"] > 0 for source in SOURCES))
    dump({
        "status": "complete", "rung": "506_gpu_smoke",
        "claim_level": "operational_cuda_smoke_no_scientific_outcomes_retained",
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": {"documents_exercised": [0, 4], "circuit_tags": discovery_tags[:2],
                           "parent": metadata["parent"]},
        "diagnostics": diagnostics,
        "check_a_frozen_authorities_and_instrument_hold": pred_a,
        "check_b_expected_tensor_shapes_hold": pred_b,
        "check_c_singleton_and_pair_patches_are_live": pred_c,
        "strong_null": not (pred_a and pred_b and pred_c),
        "scientific_task_or_circuit_effects_retained": False,
        "next_step": "enqueue_full_rung506" if pred_a and pred_b and pred_c else "repair_instrument_only",
    }, SMOKE_OUT)
    print(json.dumps({
        "status": "complete", "rung": "506_gpu_smoke",
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "scientific_outcomes_retained": False,
    }, indent=2, sort_keys=True))


def main():
    started = time.time()
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = validate_inputs()
    singleton_patch_sets = ((),) + tuple((site,) for site in SITES)
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        assert len(SITES) == 19 and len(SOURCES) == 4
        assert len(tuple(itertools.combinations(SITES, 2))) == 171
        assert 20729 + 496 * 8 + 500 * 8 == 28697
        synthetic = torch.tensor([1.0, 2.0, 3.0])
        assert cosine(synthetic, synthetic) == 1.0
        print(json.dumps({
            "status": "dry_run_passed", "rung": 506, "model_loaded": False,
            "outcomes_opened": False, "discovery_documents": list(DISCOVERY),
            "confirmation_documents": list(CONFIRMATION),
            "validation_documents": list(VALIDATION),
            "maximum_conditional_forwards": 28697,
            "predictions": ["pred_a", "pred_b", "pred_c", "pred_d", "pred_e"],
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung506 output namespace already exists")

    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    collections = {}
    collections["discovery_singletons"] = collect_phase(
        model, rows, task_masks, circuit_masks, discovery_tags, scales,
        DISCOVERY, singleton_patch_sets, include_instrument=True)
    discovery_calibration = calibration(collections["discovery_singletons"])
    eligible_sites, discovery_edges, discovery_checks = discover_edges(
        collections["discovery_singletons"])
    discovery_identifying = 1 <= len(discovery_edges) <= MAX_EDGES
    pred_b_discovery = calibration_holds(discovery_calibration)

    confirmation_calibration = None
    confirmation_checks = {}
    confirmed_edges = []
    validation_calibration = None
    validation_checks = {}
    validated_edges = []
    composition_rules = {}
    confirmation_composition = {}
    validation_composition = {}
    compositional_edges = []

    if pred_b_discovery and discovery_identifying:
        discovery_pair_sets = tuple(parse_edge(name) for name in discovery_edges)
        collections["discovery_pairs"] = collect_phase(
            model, rows, task_masks, circuit_masks, discovery_tags, scales,
            DISCOVERY, discovery_pair_sets, include_instrument=False)
        confirmation_sets = singleton_patch_sets + discovery_pair_sets
        collections["confirmation"] = collect_phase(
            model, rows, task_masks, circuit_masks, discovery_tags, scales,
            CONFIRMATION, confirmation_sets, include_instrument=True)
        confirmation_calibration = calibration(collections["confirmation"])
        confirmed_edges, confirmation_checks = confirm_edges(
            collections["confirmation"], discovery_edges)
        for name in discovery_edges:
            composition_rules[name] = fit_composition_rule(
                collections["discovery_singletons"], collections["discovery_pairs"], name)
            if name in confirmed_edges and composition_rules[name]["identified"]:
                confirmation_composition[name] = score_composition(
                    collections["confirmation"], collections["confirmation"],
                    name, composition_rules[name])

        if calibration_holds(confirmation_calibration) and confirmed_edges:
            validation_sets = singleton_patch_sets + tuple(parse_edge(name) for name in confirmed_edges)
            collections["validation"] = collect_phase(
                model, rows, task_masks, circuit_masks, validation_tags, scales,
                VALIDATION, validation_sets, include_instrument=True)
            validation_calibration = calibration(collections["validation"])
            validated_edges, validation_checks = validate_edges(
                collections["validation"], confirmed_edges)
            for name in validated_edges:
                validation_composition[name] = score_composition(
                    collections["validation"], collections["validation"],
                    name, composition_rules[name])
                if (confirmation_composition.get(name, {}).get("holds") is True
                        and validation_composition[name]["holds"] is True):
                    compositional_edges.append(name)

    all_collections = list(collections.values())
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and all(phase_instrument_holds(collection) for collection in all_collections))
    pred_b = bool(
        pred_b_discovery
        and (confirmation_calibration is None or calibration_holds(confirmation_calibration))
        and (validation_calibration is None or calibration_holds(validation_calibration)))
    pred_c = bool(pred_a and pred_b and discovery_identifying and confirmed_edges)
    pred_d = bool(pred_c and validated_edges)
    pred_e = bool(pred_d and compositional_edges)
    strong_null = bool(not pred_a or not pred_b or not pred_c or not pred_d or not pred_e)

    if not pred_a:
        next_step = "repair_instrument_only"
    elif not pred_b:
        next_step = "stop_downstream_assay_preserve_validated_score_gauge"
    elif not discovery_identifying and not discovery_edges:
        next_step = "split_fixed_writes_into_exact_attention_or_bilinear_terms"
    elif not discovery_identifying:
        next_step = "enrich_downstream_actions_without_ranking_edges"
    elif not confirmed_edges:
        next_step = "whole_write_similarity_does_not_repeat_split_exact_internal_terms"
    elif not validated_edges:
        next_step = "change_downstream_coordinates_preserve_confirmation_screen"
    elif not compositional_edges:
        next_step = "preserve_observation_equivalence_but_reject_circuit_composition"
    else:
        next_step = "split_validated_whole_write_edges_and_build_executable_joint_replacement"

    bundle_payload = {
        "schema": "rung506_finite_later_write_state_atlas_sufficient_statistics_v1",
        "collections": {name: _bundle_collection(collection)
                        for name, collection in collections.items()},
        "raw_tokens_logits_hidden_states_or_weights_included": False,
        "validation_opened": "validation" in collections,
    }
    torch.save(bundle_payload, BUNDLE)
    result = {
        "status": "complete", "rung": 506,
        "claim_level": "finite_action_conditioned_whole_write_causal_state_identification_not_internal_split_or_adoption",
        "source_hashes": {str(path): expected for path, expected in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata,
        "sources": list(SOURCES), "sites": list(SITES),
        "discovery_tags": discovery_tags, "validation_tags": validation_tags,
        "collections": {name: _serial_collection(collection)
                        for name, collection in collections.items()},
        "calibration": {
            "discovery": discovery_calibration,
            "confirmation": confirmation_calibration,
            "validation": validation_calibration,
        },
        "analysis": {
            "eligible_sites": eligible_sites,
            "discovery_edges": discovery_edges,
            "discovery_identifying": discovery_identifying,
            "discovery_checks": discovery_checks,
            "confirmed_edges": confirmed_edges,
            "confirmation_checks": confirmation_checks,
            "validated_edges": validated_edges,
            "validation_checks": validation_checks,
            "composition_rules": composition_rules,
            "confirmation_composition": confirmation_composition,
            "validation_composition": validation_composition,
            "compositional_selective_edges": compositional_edges,
        },
        'pred_a_exact_live_conditional_instrument': pred_a,
        'pred_b_score_actions_recalibrate_new_documents': pred_b,
        'pred_c_at_least_one_whole_write_edge_confirms': pred_c,
        'pred_d_at_least_one_edge_validates_heldout_circuits_documents': pred_d,
        'pred_e_at_least_one_edge_composes_and_removes_selectively': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {
            "path": str(BUNDLE), "sha256": sha256(BUNDLE), "bytes": BUNDLE.stat().st_size,
        },
        "execution_price": {
            "full_forwards": sum(sum(c["diagnostics"]["calls"].values()) for c in all_collections),
            "backwards": 0,
            "discovery_edge_count_k": len(discovery_edges),
            "confirmed_edge_count_q": len(confirmed_edges),
            "maximum_conditional_forwards": 28697,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "fitted_scalars": sum(rule.get("kind") == "one_scalar_interaction"
                                  for rule in composition_rules.values()),
            "fitted_vectors": 0, "deployed_parameters_added": 0,
            "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started,
        "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 506,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "eligible_sites": eligible_sites, "discovery_edges": discovery_edges,
        "confirmed_edges": confirmed_edges, "validated_edges": validated_edges,
        "compositional_selective_edges": compositional_edges,
        "execution_price": result["execution_price"], "next_step": next_step,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
