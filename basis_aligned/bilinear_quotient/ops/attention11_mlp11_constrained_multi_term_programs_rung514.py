#!/usr/bin/env python3
"""RUNG514 -- fixed factor allocations and sparse signed consumer programs."""

# BQGATE: EXPERIMENT
# pred_a: exact joint-Gram, permutation, planted-recovery, calibration, and patch instrument is live
# pred_b: one to thirty-two fixed or sparse groups pass both discovery searches and familywise control
# pred_c: a discovery group predicts fresh documents with support and signs frozen
# pred_d: a confirmed group passes all six physical substitutions and matched removals
# pred_e: one causal program is reused by at least two MLP10 branch subsets

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
import attention11_mlp11_exact_factor_interactions_rung513 as r513


r512, r511, parent = r513.r512, r513.r511, r513.parent
PREREG = POLY / "ATTENTION11_MLP11_CONSTRAINED_MULTI_TERM_PROGRAMS_RUNG514_PREREGISTRATION.md"
PREFLIGHT_ADDENDUM = POLY / "ATTENTION11_MLP11_CONSTRAINED_MULTI_TERM_PROGRAMS_RUNG514_PREFLIGHT_ADDENDUM.md"
R513_RESULT = ROOT / "attention11_mlp11_exact_factor_interactions_rung513_results.json"
R513_BUNDLE = ROOT / "attention11_mlp11_exact_factor_interactions_rung513_bundle.pt"
R513_SOURCE = ROOT / "ops/attention11_mlp11_exact_factor_interactions_rung513.py"
R513_PREREG = POLY / "ATTENTION11_MLP11_EXACT_FACTOR_INTERACTIONS_RUNG513_PREREGISTRATION.md"
MISMATCH_PREREG = POLY / "MISMATCH_COVARIANCE_PROBE_PREREGISTRATION.md"
MISMATCH_SOURCE = ROOT / "ops/mismatch_covariance_probe.py"
MISMATCH_RESULT = ROOT / "mismatch_covariance_probe_results.json"
OUT = ROOT / "attention11_mlp11_constrained_multi_term_programs_rung514_results.json"
BUNDLE = ROOT / "attention11_mlp11_constrained_multi_term_programs_rung514_bundle.pt"

HASHES = {
    PREREG: "602e167697e1eda8099ee8e52037cb3bf844f793722bba6da463b89cb0fd7957",
    PREFLIGHT_ADDENDUM: "30e3635ecc31ffc764b41d65edad426671fac3bf1651ac04317983b32cf3f0c7",
    R513_RESULT: "043dd563baa5ffe5bda57c7774dc76e4727add3b4351699cb997ecfd563179d5",
    R513_BUNDLE: "06118d18594c4b167a3f3d46a2aa282969f6b061835f83a3b3d62b5ca72b8d8a",
    R513_SOURCE: "dda9c2636a99f76a2298e5cebccea1b1e8bd503c415f073ca93c984e8713fc98",
    R513_PREREG: "b895d1aefdac4c7deee0477c260a5e1ec087477925e841d0d2b8ebb4a02670aa",
    MISMATCH_PREREG: "164bc70dbed5098829e3efa47d9de24323d11c35a5e892014265eb2bc70f714b",
    MISMATCH_SOURCE: "6cc128ffb0721ffe9665a1429df6ae8e533d7a8bef995c4b7dd050fa23f9f7eb",
    MISMATCH_RESULT: "150b1448cb6df0218e6db9488921de15e29c35f98ea3fb647f5cde2833bb4d04",
}

DISCOVERY = r513.DISCOVERY
CONFIRMATION = r513.CONFIRMATION
DISCOVERY_WINDOWS = {
    "A_fit": (500, 560), "A_test": (560, 624),
    "B_fit": (624, 684), "B_test": (684, 748),
    "pooled": (500, 748),
}
CONFIRMATION_WINDOWS = {
    "half0": (752, 876), "half1": (876, 1000), "pooled": (752, 1000),
}
SEARCH_PAIRS = (("A_fit", "A_test"), ("B_fit", "B_test"))
CONTROL_SEEDS = tuple(range(51410, 51426))
PLANTED_SEEDS = tuple(range(51400, 51408))
MAX_CANDIDATES = 32
TASK_CELLS = parent.TASK_CELLS
COPY_CELLS = r512.COPY_CELLS
SITE_TERMS = {"a11": r513.ATTENTION_TERMS, "m11": r513.MLP_TERMS}
SITE_OFFSETS = {"a11": 0, "m11": len(r513.ATTENTION_TERMS)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sparse_programs(site: str) -> tuple[dict, ...]:
    names = SITE_TERMS[site]
    programs = []
    for size in (2, 3):
        for support in itertools.combinations(range(len(names)), size):
            for tail_signs in itertools.product((-1, 1), repeat=size - 1):
                signs = (1,) + tail_signs
                coefficient = torch.zeros(len(names), dtype=torch.float64)
                coefficient[list(support)] = torch.tensor(signs, dtype=torch.float64)
                programs.append({
                    "class": "sparse_signed", "site": site,
                    "name": "+".join(
                        ("" if sign > 0 else "-") + names[index]
                        for index, sign in zip(support, signs)),
                    "support": list(support), "signs": list(signs),
                    "coefficient": coefficient,
                })
    return tuple(programs)


def fixed_factor_programs(site: str) -> tuple[dict, ...]:
    if site == "a11":
        output = []
        for factor_index, factor_name in enumerate(r513.FACTOR_NAMES):
            coefficient = torch.zeros(31, dtype=torch.float64)
            for local_index, mask in enumerate(r513.ATTENTION_MASKS):
                if mask & (1 << factor_index):
                    coefficient[local_index] = 1.0 / mask.bit_count()
            output.append({
                "class": "fixed_shapley", "site": site,
                "name": f"A11 Shapley({factor_name})", "factor": factor_name,
                "support": coefficient.nonzero().flatten().tolist(),
                "signs": None, "coefficient": coefficient,
            })
        coefficient = sum(row["coefficient"] for row in output
                          if row["factor"] in ("Q", "Q2", "V"))
        output.append({
            "class": "fixed_mismatch_top3", "site": site,
            "name": "A11 Shapley(Q+Q2+V)", "factor": "Q+Q2+V",
            "support": coefficient.nonzero().flatten().tolist(),
            "signs": None, "coefficient": coefficient,
        })
        return tuple(output)
    if site == "m11":
        return (
            {"class": "fixed_shapley", "site": site, "name": "M11 Shapley(L)",
             "factor": "L", "support": [0, 2], "signs": None,
             "coefficient": torch.tensor([1., 0., .5], dtype=torch.float64)},
            {"class": "fixed_shapley", "site": site, "name": "M11 Shapley(R)",
             "factor": "R", "support": [1, 2], "signs": None,
             "coefficient": torch.tensor([0., 1., .5], dtype=torch.float64)},
        )
    raise ValueError(f"unknown site {site}")


PROGRAMS = {
    site: fixed_factor_programs(site) + canonical_sparse_programs(site)
    for site in ("a11", "m11")
}
PROGRAM_MATRICES = {
    site: torch.stack([program["coefficient"] for program in programs])
    for site, programs in PROGRAMS.items()
}


def _new_statistics(window_names, controls: bool) -> dict:
    real, shuffled = {}, {}
    for window in window_names:
        real[window] = {}
        for selected_subset in range(len(r513.SELECTED_SUBSETS)):
            real[window][selected_subset] = {}
            for site, names in SITE_TERMS.items():
                width = r511.N_ACTIONS * len(names)
                real[window][selected_subset][site] = {
                    "joint": torch.zeros(width, width, dtype=torch.float64),
                    "total": torch.zeros(r511.N_ACTIONS, r511.N_ACTIONS,
                                         dtype=torch.float64),
                }
    if controls:
        for seed in CONTROL_SEEDS:
            shuffled[seed] = {}
            for window in window_names:
                shuffled[seed][window] = {}
                for selected_subset in range(len(r513.SELECTED_SUBSETS)):
                    shuffled[seed][window][selected_subset] = {}
                    for site, names in SITE_TERMS.items():
                        width = r511.N_ACTIONS * len(names)
                        shuffled[seed][window][selected_subset][site] = torch.zeros(
                            width, width, dtype=torch.float64)
    return {
        "real": real, "controls": shuffled,
        "source": {window: torch.zeros(r513.N_LOCAL_NODES, r513.N_LOCAL_NODES,
                                       dtype=torch.float64)
                   for window in window_names},
    }


def _active_windows(start: int, window_bounds: dict) -> tuple[str, ...]:
    active = [name for name, (lo, hi) in window_bounds.items() if lo <= start < hi]
    if not active:
        raise RuntimeError(f"batch start {start} has no registered window")
    return tuple(active)


def _site_vectors(term_vectors, selected_subset: int, site: str) -> torch.Tensor:
    offset = SITE_OFFSETS[site]
    count = len(SITE_TERMS[site])
    rows = []
    for action in range(r511.N_ACTIONS):
        node = r513.local_node(action, selected_subset)
        rows.extend(term_vectors[node][offset:offset + count])
    return torch.stack(rows).float()


def _total_vectors(total_vectors, selected_subset: int, site: str) -> torch.Tensor:
    site_index = 0 if site == "a11" else 1
    return torch.stack([
        total_vectors[r513.local_node(action, selected_subset)][site_index]
        for action in range(r511.N_ACTIONS)
    ]).float()


def _permuted_site_vectors(values: torch.Tensor, term_count: int,
                            seed: int, start: int) -> torch.Tensor:
    reshaped = values.view(r511.N_ACTIONS, term_count, -1).clone()
    width = reshaped.shape[-1]
    for action in range(1, r511.N_ACTIONS):
        generator = torch.Generator(device=values.device).manual_seed(
            seed * 1_000_000 + start * 10 + action)
        order = torch.randperm(width, generator=generator, device=values.device)
        reshaped[action] = reshaped[action, :, order]
    return reshaped.view(r511.N_ACTIONS * term_count, width)


def update_joint_statistics(statistics: dict, term_vectors, total_vectors, source_vectors,
                            start: int, window_bounds: dict, controls: bool) -> None:
    windows = _active_windows(start, window_bounds)
    for selected_subset in range(len(r513.SELECTED_SUBSETS)):
        for site, names in SITE_TERMS.items():
            values = _site_vectors(term_vectors, selected_subset, site)
            totals = _total_vectors(total_vectors, selected_subset, site)
            joint = (values @ values.T).double().cpu()
            total = (totals @ totals.T).double().cpu()
            for window in windows:
                statistics["real"][window][selected_subset][site]["joint"] += joint
                statistics["real"][window][selected_subset][site]["total"] += total
            if controls:
                for seed in CONTROL_SEEDS:
                    permuted = _permuted_site_vectors(values, len(names), seed, start)
                    permuted_joint = (permuted @ permuted.T).double().cpu()
                    for window in windows:
                        statistics["controls"][seed][window][selected_subset][site] += permuted_joint
    source_values = torch.stack(source_vectors).float()
    source_gram = (source_values @ source_values.T).double().cpu()
    for window in windows:
        statistics["source"][window] += source_gram


def program_grams(joint: torch.Tensor, coefficients: torch.Tensor,
                  term_count: int) -> torch.Tensor:
    """Return one 4x4 response Gram per registered coefficient row."""
    output = torch.empty(len(coefficients), r511.N_ACTIONS, r511.N_ACTIONS,
                         dtype=torch.float64)
    for left in range(r511.N_ACTIONS):
        left_slice = slice(left * term_count, (left + 1) * term_count)
        for right in range(r511.N_ACTIONS):
            right_slice = slice(right * term_count, (right + 1) * term_count)
            block = joint[left_slice, right_slice]
            output[:, left, right] = torch.einsum(
                "pi,ij,pj->p", coefficients, block, coefficients)
    return output


def bank_program_grams(joint: torch.Tensor, programs: tuple[dict, ...],
                       term_count: int) -> torch.Tensor:
    """Sparse O(P*k^2) evaluation plus a tiny dense fixed-program block."""
    fixed_indices = [index for index, row in enumerate(programs)
                     if row["class"] != "sparse_signed"]
    sparse_indices = [index for index, row in enumerate(programs)
                      if row["class"] == "sparse_signed"]
    output = torch.empty(len(programs), r511.N_ACTIONS, r511.N_ACTIONS,
                         dtype=torch.float64)
    if fixed_indices:
        coefficients = torch.stack([programs[index]["coefficient"]
                                    for index in fixed_indices])
        output[fixed_indices] = program_grams(joint, coefficients, term_count)
    if sparse_indices:
        support = torch.tensor([
            programs[index]["support"] + [0] * (3 - len(programs[index]["support"]))
            for index in sparse_indices
        ], dtype=torch.long)
        values = torch.tensor([
            [float(programs[index]["coefficient"][position]) for position in programs[index]["support"]]
            + [0.0] * (3 - len(programs[index]["support"]))
            for index in sparse_indices
        ], dtype=torch.float64)
        for left in range(r511.N_ACTIONS):
            for right in range(r511.N_ACTIONS):
                block = joint[
                    left * term_count:(left + 1) * term_count,
                    right * term_count:(right + 1) * term_count,
                ]
                selected = block[support[:, :, None], support[:, None, :]]
                output[sparse_indices, left, right] = (
                    selected * values[:, :, None] * values[:, None, :]).sum((1, 2))
    return output


def search_pair_metrics(fit_program: torch.Tensor, test_program: torch.Tensor,
                        fit_total: torch.Tensor, test_total: torch.Tensor) -> dict:
    program_count = fit_program.shape[0]
    holds = torch.ones(program_count, dtype=torch.bool)
    eligible = torch.ones(program_count, dtype=torch.bool)
    margin = torch.full((program_count,), math.inf, dtype=torch.float64)
    rows = {}
    for relation_name, (left, right) in zip(r513.RELATION_NAMES, r513.RELATION_ACTIONS):
        beta = fit_program[:, left, right] / fit_program[:, right, right].clamp_min(1e-30)
        ll, rr, lr = (test_program[:, left, left], test_program[:, right, right],
                      test_program[:, left, right])
        cosine = torch.sign(beta) * lr / (ll.clamp_min(0) * rr.clamp_min(0)).sqrt().clamp_min(1e-30)
        inverse = 1.0 / beta
        forward = ((ll - 2 * beta * lr + beta.square() * rr).clamp_min(0)
                   / ll.clamp_min(1e-30)).sqrt()
        backward = ((rr - 2 * inverse * lr + inverse.square() * ll).clamp_min(0)
                    / rr.clamp_min(1e-30)).sqrt()
        fit_fraction = torch.stack([
            (fit_program[:, action, action].clamp_min(0)
             / fit_total[action, action].clamp_min(1e-30)).sqrt()
            for action in (left, right)
        ], 1).amin(1)
        test_fraction = torch.stack([
            (test_program[:, action, action].clamp_min(0)
             / test_total[action, action].clamp_min(1e-30)).sqrt()
            for action in (left, right)
        ], 1).amin(1)
        residual = torch.maximum(forward, backward)
        relation_eligible = ((beta.abs() >= .25) & (beta.abs() <= 4)
                             & (fit_fraction >= .10) & (test_fraction >= .10)
                             & torch.isfinite(beta) & torch.isfinite(residual))
        relation_holds = relation_eligible & (cosine >= .85) & (residual <= .55)
        eligible &= relation_eligible
        holds &= relation_holds
        margin = torch.minimum(margin, torch.minimum(cosine - .85, .55 - residual))
        rows[relation_name] = {
            "beta": beta, "cosine": cosine, "residual": residual,
            "fit_fraction": fit_fraction, "test_fraction": test_fraction,
            "holds": relation_holds,
        }
    return {"holds": holds, "eligible": eligible, "margin": margin, "relations": rows}


def _serial_relation(metrics: dict, index: int) -> dict:
    return {
        relation_name: {
            key: (bool(value[index]) if value.dtype == torch.bool else float(value[index]))
            for key, value in row.items()
        }
        for relation_name, row in metrics["relations"].items()
    }


def _program_descriptor(program: dict) -> dict:
    return {key: value for key, value in program.items() if key != "coefficient"}


def scan_banks(statistics: dict, *, control_seed: int | None = None,
               floor: float | None = None) -> dict:
    candidates, top, counts = [], [], {
        "programs": 0, "A_pass": 0, "B_pass": 0,
        "split_stable": 0, "above_familywise_floor": 0, "accepted": 0,
    }
    maximum_quality = -math.inf
    source = statistics["real"] if control_seed is None else statistics["controls"][control_seed]
    for selected_subset, subset_index in enumerate(r513.SELECTED_SUBSETS):
        subset_name = r511.SUBSET_NAMES[subset_index]
        for site, programs in PROGRAMS.items():
            term_count = len(SITE_TERMS[site])
            grams = {}
            for window in DISCOVERY_WINDOWS:
                joint = (source[window][selected_subset][site]["joint"]
                         if control_seed is None
                         else source[window][selected_subset][site])
                grams[window] = bank_program_grams(joint, programs, term_count)
            total = {
                window: statistics["real"][window][selected_subset][site]["total"]
                for window in DISCOVERY_WINDOWS
            }
            first = search_pair_metrics(
                grams["A_fit"], grams["A_test"], total["A_fit"], total["A_test"])
            second = search_pair_metrics(
                grams["B_fit"], grams["B_test"], total["B_fit"], total["B_test"])
            stable = first["holds"] & second["holds"]
            quality = torch.minimum(first["margin"], second["margin"])
            counts["programs"] += len(programs)
            counts["A_pass"] += int(first["holds"].sum())
            counts["B_pass"] += int(second["holds"].sum())
            counts["split_stable"] += int(stable.sum())
            familywise_eligible = first["eligible"] & second["eligible"]
            finite_quality = quality[familywise_eligible & torch.isfinite(quality)]
            if len(finite_quality):
                maximum_quality = max(maximum_quality, float(finite_quality.max()))
            if control_seed is not None:
                continue
            above = torch.ones_like(stable) if floor is None else quality >= floor
            accepted = stable & above
            counts["above_familywise_floor"] += int(above.sum())
            counts["accepted"] += int(accepted.sum())
            for index in accepted.nonzero().flatten().tolist():
                program = programs[index]
                candidates.append({
                    **_program_descriptor(program),
                    "coefficient": program["coefficient"].tolist(),
                    "selected_subset": selected_subset,
                    "subset_index": subset_index, "subset_name": subset_name,
                    "quality_margin": float(quality[index]),
                    "search_A": _serial_relation(first, index),
                    "search_B": _serial_relation(second, index),
                })
            take = min(5, len(quality))
            values, indices = torch.topk(quality.nan_to_num(nan=-math.inf), take)
            for value, index in zip(values.tolist(), indices.tolist()):
                top.append({
                    **_program_descriptor(programs[index]),
                    "selected_subset": selected_subset,
                    "subset_name": subset_name,
                    "quality_margin": value,
                    "split_stable": bool(stable[index]),
                })
    top.sort(key=lambda row: row["quality_margin"], reverse=True)
    return {
        "maximum_quality": maximum_quality, "counts": counts,
        "candidates": candidates, "top_screens": top[:20],
    }


def discover_programs(collection: dict) -> tuple[list[dict], dict]:
    controls = {
        seed: scan_banks(collection["statistics"], control_seed=seed)
        for seed in CONTROL_SEEDS
    }
    control_maxima = {seed: row["maximum_quality"] for seed, row in controls.items()}
    familywise_floor = max(control_maxima.values()) + .02
    real = scan_banks(collection["statistics"], floor=familywise_floor)
    return real["candidates"], {
        "fixed_groups": len(r513.SELECTED_SUBSETS) * sum(
            len(fixed_factor_programs(site)) for site in SITE_TERMS),
        "sparse_groups": len(r513.SELECTED_SUBSETS) * sum(
            len(canonical_sparse_programs(site)) for site in SITE_TERMS),
        "all_groups": real["counts"]["programs"],
        "familywise_control_maxima": control_maxima,
        "familywise_floor_with_margin": familywise_floor,
        "real": real,
    }


def source_relation_reproduction(collection: dict, discovery: bool) -> tuple[list[str], dict]:
    source = collection["statistics"]["source"]
    if discovery:
        grams = {
            "half0": source["A_fit"] + source["A_test"],
            "half1": source["B_fit"] + source["B_test"],
            "pooled": source["pooled"],
        }
    else:
        grams = {window: source[window] for window in ("half0", "half1", "pooled")}
    return r513.reproduce_source_relations({"statistics": {"source_gram": grams}})


def _toy_joint(seed: int, support: tuple[int, ...], signs: tuple[int, ...]) -> dict:
    term_count = len(r513.ATTENTION_TERMS)
    stats = _new_statistics(tuple(DISCOVERY_WINDOWS), controls=False)
    scales = (1.0, 1.15, .80, .95)
    for window_index, window in enumerate(DISCOVERY_WINDOWS):
        generator = torch.Generator().manual_seed(seed * 100 + window_index)
        dimension = 256
        response = torch.randn(r511.N_ACTIONS, term_count, dimension, generator=generator)
        shared = torch.randn(dimension, generator=generator)
        for action, scale in enumerate(scales):
            partial = torch.zeros(dimension)
            for position, term_index in enumerate(support[:-1]):
                response[action, term_index] = torch.randn(dimension, generator=generator)
                partial += signs[position] * response[action, term_index]
            last = support[-1]
            response[action, last] = (scale * shared - partial) / signs[-1]
        values = response.reshape(r511.N_ACTIONS * term_count, dimension).double()
        total_values = response.sum(1).double()
        row = stats["real"][window][0]["a11"]
        row["joint"] = values @ values.T
        row["total"] = total_values @ total_values.T
    return stats


def planted_recovery_suite() -> dict:
    programs = PROGRAMS["a11"]
    sparse_lookup = {
        (tuple(row["support"]), tuple(row["signs"])): index
        for index, row in enumerate(programs) if row["class"] == "sparse_signed"
    }
    rows, all_hold = [], True
    for offset, seed in enumerate(PLANTED_SEEDS):
        size = 2 + (offset % 2)
        generator = torch.Generator().manual_seed(seed)
        support = tuple(sorted(torch.randperm(31, generator=generator)[:size].tolist()))
        signs = (1,) + tuple(1 if int(value) else -1 for value in
                             torch.randint(0, 2, (size - 1,), generator=generator))
        expected = sparse_lookup[(support, signs)]
        stats = _toy_joint(seed, support, signs)
        grams = {
            window: bank_program_grams(
                stats["real"][window][0]["a11"]["joint"], programs, 31)
            for window in DISCOVERY_WINDOWS
        }
        totals = {window: stats["real"][window][0]["a11"]["total"]
                  for window in DISCOVERY_WINDOWS}
        first = search_pair_metrics(
            grams["A_fit"], grams["A_test"], totals["A_fit"], totals["A_test"])
        second = search_pair_metrics(
            grams["B_fit"], grams["B_test"], totals["B_fit"], totals["B_test"])
        recovered = (first["holds"] & second["holds"]).nonzero().flatten().tolist()
        holds = recovered == [expected]
        rows.append({
            "seed": seed, "support": list(support), "signs": list(signs),
            "expected_program_index": expected, "recovered_program_indices": recovered,
            "holds": holds,
        })
        all_hold &= holds
    return {"seeds": list(PLANTED_SEEDS), "cases": rows,
            "all_exact_unique_recoveries": bool(all_hold)}


def _single_program_gram(joint: torch.Tensor, coefficient: torch.Tensor) -> torch.Tensor:
    return program_grams(joint, coefficient.reshape(1, -1).double(), len(coefficient))[0]


def _fixed_beta_relation(program: torch.Tensor, total: torch.Tensor,
                         left: int, right: int, beta: float,
                         cosine_floor: float, residual_ceiling: float) -> dict:
    ll, rr, lr = float(program[left, left]), float(program[right, right]), float(program[left, right])
    cosine = math.copysign(1.0, beta) * lr / max(math.sqrt(max(ll * rr, 0.0)), 1e-30)
    inverse = 1.0 / beta
    forward = math.sqrt(max(ll - 2 * beta * lr + beta * beta * rr, 0.0) / max(ll, 1e-30))
    backward = math.sqrt(max(rr - 2 * inverse * lr + inverse * inverse * ll, 0.0) / max(rr, 1e-30))
    fractions = [math.sqrt(max(ll, 0.0) / max(float(total[left, left]), 1e-30)),
                 math.sqrt(max(rr, 0.0) / max(float(total[right, right]), 1e-30))]
    holds = bool(min(fractions) >= .10 and cosine >= cosine_floor
                 and max(forward, backward) <= residual_ceiling)
    return {
        "beta_left_from_right": beta, "cosine": cosine,
        "left_from_right_relative_residual": forward,
        "right_from_left_relative_residual": backward,
        "program_to_complete_response_rms": fractions,
        "material": min(fractions) >= .10, "holds": holds,
    }


def confirm_programs(discovery: dict, confirmation: dict,
                     candidates: list[dict]) -> tuple[list[dict], dict]:
    confirmed, checks = [], {}
    for candidate in candidates:
        coefficient = torch.tensor(candidate["coefficient"], dtype=torch.float64)
        selected_subset, site = candidate["selected_subset"], candidate["site"]
        discovery_row = discovery["statistics"]["real"]["pooled"][selected_subset][site]
        discovery_gram = _single_program_gram(discovery_row["joint"], coefficient)
        row = {"relations": {}, "holds": True}
        frozen_relations = {}
        for relation_name, (left, right) in zip(r513.RELATION_NAMES, r513.RELATION_ACTIONS):
            beta = float(discovery_gram[left, right]
                         / discovery_gram[right, right].clamp_min(1e-30))
            frozen_relations[relation_name] = {
                "left_action": left, "right_action": right,
                "beta_left_from_right": beta,
            }
            relation_row = {"windows": {}, "holds": True}
            for window in ("half0", "half1", "pooled"):
                confirmation_row = confirmation["statistics"]["real"][window][selected_subset][site]
                program = _single_program_gram(confirmation_row["joint"], coefficient)
                metrics = _fixed_beta_relation(
                    program, confirmation_row["total"], left, right, beta, .75, .65)
                relation_row["windows"][window] = metrics
                if window in ("half0", "half1"):
                    relation_row["holds"] &= metrics["holds"]
            row["relations"][relation_name] = relation_row
            row["holds"] &= relation_row["holds"]
        key = f"{candidate['subset_name']} @ {candidate['name']}"
        checks[key] = row
        if row["holds"]:
            confirmed.append({**candidate, "term_name": candidate["name"],
                              "relations": frozen_relations})
    return confirmed, checks


def collection_instrument(collection: dict, controls: bool) -> bool:
    diagnostics = collection["diagnostics"]
    statistics = collection["statistics"]
    return bool(
        r511._instrument(collection)
        and diagnostics["factor_consumer_captures_exact"]
        and diagnostics["attention_corner_evaluations_exact"]
        and diagnostics["mlp_corner_evaluations_exact"]
        and diagnostics["removed_attention_corner_replay_max_abs"] == 0.0
        and diagnostics["intact_attention_corner_replay_max_abs"] == 0.0
        and diagnostics["attention_numerical_remainder_rms_ratio"] <= .01
        and diagnostics["mlp_deployed_branch_sum_relative_squared"] <= 1e-12
        and diagnostics["joint_gram_symmetry_max_abs"] <= 1e-8
        and diagnostics["control_joint_grams_exact"]
        and bool(statistics["controls"]) == controls
        and all(torch.isfinite(row[site]["joint"]).all()
                and torch.isfinite(row[site]["total"]).all()
                for window in statistics["real"].values()
                for row in window.values() for site in SITE_TERMS)
    )


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(R513_RESULT.read_text())
    if not (result.get("pred_a_exact_live_factor_interaction_instrument") is True
            and result.get("pred_b_shared_factor_term_discovery") is False
            and result.get("analysis", {}).get("discovery_summary", {}).get("candidate_count") == 0
            and result.get("next_step")
            == "preregister_sparse_multi_term_mismatch_combinations_with_planted_identifiability_control"):
        raise RuntimeError("rung513 route changed")
    mismatch = json.loads(MISMATCH_RESULT.read_text())
    if not (mismatch.get("pred_a_exact_reproduction_of_513_shares") is True
            and mismatch.get("pred_b_mismatch_shape_is_gauge_covariant") is True
            and mismatch.get("pred_c_stable_dominant_factor_subspace") is True):
        raise RuntimeError("mismatch top-three addendum authority changed")
    rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, metadata = \
        r513.validate_inputs()
    return rows, task_masks, circuit_masks, scales, discovery_tags, validation_tags, {
        **metadata,
        "rung513_result_sha256": sha256(R513_RESULT),
        "rung513_bundle_sha256": sha256(R513_BUNDLE),
        "mismatch_result_sha256": sha256(MISMATCH_RESULT),
        "discovery_windows": DISCOVERY_WINDOWS,
        "confirmation_windows": CONFIRMATION_WINDOWS,
        "fixed_groups": 48, "sparse_groups": 113520, "all_groups": 113568,
    }


def _toy_physical_candidate() -> dict:
    program = fixed_factor_programs("a11")[-1]
    return {
        **_program_descriptor(program), "coefficient": program["coefficient"].tolist(),
        "term_name": program["name"], "selected_subset": 0,
        "subset_index": r513.SELECTED_SUBSETS[0],
        "subset_name": r511.SUBSET_NAMES[r513.SELECTED_SUBSETS[0]],
        "relations": {
            name: {"left_action": pair[0], "right_action": pair[1],
                   "beta_left_from_right": 1.0}
            for name, pair in zip(r513.RELATION_NAMES, r513.RELATION_ACTIONS)
        },
    }


def dry_run() -> None:
    planted = planted_recovery_suite()
    assert planted["all_exact_unique_recoveries"]
    assert len(PROGRAMS["a11"]) == 18916
    assert len(PROGRAMS["m11"]) == 12
    assert sum(len(rows) for rows in PROGRAMS.values()) * 6 == 113568
    attention_fixed = fixed_factor_programs("a11")
    torch.testing.assert_close(
        sum(row["coefficient"] for row in attention_fixed[:5]), torch.ones(31, dtype=torch.float64))
    mlp_fixed = fixed_factor_programs("m11")
    torch.testing.assert_close(
        sum(row["coefficient"] for row in mlp_fixed), torch.ones(3, dtype=torch.float64))
    candidate = _toy_physical_candidate()
    terms = tuple(torch.tensor([float(index)]) for index in range(34))
    expected = sum(terms[index] * candidate["coefficient"][index]
                   for index in range(31) if candidate["coefficient"][index] != 0)
    torch.testing.assert_close(program_tensor(terms, candidate), expected)
    assert 4216 + 1798 + 620 * 32 == 25854
    print(json.dumps({
        "status": "dry_run_passed", "rung": 514, "model_loaded": False,
        "outcomes_opened": False, "fixed_groups": 48, "sparse_groups": 113520,
        "all_groups": 113568, "planted_cases": 8,
        "all_planted_supports_uniquely_recovered": True,
        "maximum_conditional_forwards": 25854,
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
    collection = collect_joint(
        model, rows, task_masks, circuit_masks, discovery_tags, scales,
        bounds, DISCOVERY_WINDOWS, controls=True)
    calibration = parent._calibration(
        collection["base_task"], collection["source_task"], collection["task_counts"], bounds)
    native_recovery = calibration["pooled"]["N"]["recovery_vs_native"]
    candidate = _toy_physical_candidate()
    physical = collect_physical(
        model, rows, task_masks, circuit_masks, discovery_tags, scales,
        bounds, [candidate])
    checks = {
        "weights": checkpoint.weights_sha256 == facade.WEIGHTS_SHA256,
        "planted_recovery": planted["all_exact_unique_recoveries"],
        "collection": collection_instrument(collection, controls=True),
        "native_calibration_semantics": .9 <= native_recovery <= 1.1,
        "physical": r513.physical_instrument(physical),
        "all_192_control_grams": collection["diagnostics"]["control_joint_grams"] == 192,
        "all_28_branch_patches": collection["diagnostics"]["subset_patches"] == 28,
        "all_10_consumer_patches": physical["diagnostics"]["consumer_patches"] == 10,
    }
    passed = all(checks.values())
    print(json.dumps({
        "status": "smoke_passed" if passed else "smoke_failed", "rung": 514,
        "scientific_outcomes_retained": False, "checks": checks,
        "planted_recovery": planted,
        "collection_diagnostics": collection["diagnostics"],
        "physical_diagnostics": physical["diagnostics"],
        "smoke_calibration": calibration,
        "full_forwards": sum(collection["diagnostics"]["calls"].values())
                         + physical["diagnostics"]["calls"],
        "backwards": 0,
    }, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(f"rung514 smoke failed: "
                           f"{sorted(name for name, value in checks.items() if not value)}")


def _bundle_collection(collection: dict) -> dict:
    return {key: value for key, value in collection.items() if key != "diagnostics"}


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
        raise RuntimeError("rung514 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    collections = {}
    collections["discovery"] = collect_joint(
        model, rows, task_masks, circuit_masks, discovery_tags, scales,
        DISCOVERY, DISCOVERY_WINDOWS, controls=True)
    discovery_calibration = parent._calibration(
        collections["discovery"]["base_task"], collections["discovery"]["source_task"],
        collections["discovery"]["task_counts"], DISCOVERY)
    discovery_calibration_ok = parent.state_parent.calibration_holds(discovery_calibration)
    source_relations, source_checks = source_relation_reproduction(
        collections["discovery"], discovery=True)
    source_reproduced = source_relations == list(r513.SOURCE_RELATION_NAMES)
    discovery_instrument = collection_instrument(collections["discovery"], controls=True)
    candidates, discovery_summary = discover_programs(collections["discovery"])
    identifiable_count = 1 <= len(candidates) <= MAX_CANDIDATES

    confirmed, confirmation_checks = [], {}
    confirmation_calibration, confirmation_calibration_ok = {}, False
    if (planted["all_exact_unique_recoveries"] and discovery_calibration_ok
            and source_reproduced and discovery_instrument and identifiable_count):
        collections["confirmation"] = collect_joint(
            model, rows, task_masks, circuit_masks, validation_tags, scales,
            CONFIRMATION, CONFIRMATION_WINDOWS, controls=False)
        confirmation_calibration = parent._calibration(
            collections["confirmation"]["base_task"], collections["confirmation"]["source_task"],
            collections["confirmation"]["task_counts"], CONFIRMATION)
        confirmation_calibration_ok = parent.state_parent.calibration_holds(
            confirmation_calibration)
        confirmed, confirmation_checks = confirm_programs(
            collections["discovery"], collections["confirmation"], candidates)

    physical, physical_passing, physical_checks = None, [], {}
    if confirmation_calibration_ok and confirmed:
        physical = collect_physical(
            model, rows, task_masks, circuit_masks, validation_tags, scales,
            CONFIRMATION, confirmed)
        physical_passing, physical_checks = r513.score_physical(physical, confirmed)

    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and planted["all_exact_unique_recoveries"]
        and discovery_calibration_ok and source_reproduced and discovery_instrument
        and ("confirmation" not in collections or (
            confirmation_calibration_ok
            and collection_instrument(collections["confirmation"], controls=False)))
        and (physical is None or r513.physical_instrument(physical))
    )
    pred_b = bool(pred_a and identifiable_count)
    pred_c = bool(pred_b and confirmation_calibration_ok and confirmed)
    pred_d = bool(pred_c and physical_passing)
    reused = sorted({
        candidate["name"] for candidate in physical_passing
        if sum(other["name"] == candidate["name"]
               and other["subset_name"] != candidate["subset_name"]
               for other in physical_passing) > 0
    })
    pred_e = bool(pred_d and reused)
    strong_null = not (pred_a and pred_b and pred_c and pred_d)
    top3_candidates = [row for row in candidates if row["class"] == "fixed_mismatch_top3"]
    top3_confirmed = [row for row in confirmed if row["class"] == "fixed_mismatch_top3"]
    top3_physical = [row for row in physical_passing if row["class"] == "fixed_mismatch_top3"]
    if not pred_a:
        next_step = "repair_only_named_joint_gram_permutation_planted_or_patch_instrument"
    elif len(candidates) > MAX_CANDIDATES:
        next_step = "strengthen_downstream_observations_because_multi_term_basis_is_nonidentifiable"
    elif not pred_b:
        next_step = "preregister_task_conditioned_nonlinear_reader_of_exact_terms_with_heldout_circuit_outcomes"
    elif not pred_c:
        next_step = "diagnose_document_dependence_of_fixed_multi_term_relation"
    elif not pred_d:
        next_step = "split_at_first_downstream_reader_rejecting_multi_term_substitution"
    elif not pred_e:
        next_step = "validate_branch_specific_multi_term_circuit_on_fixed_ood_code"
    else:
        next_step = "validate_reused_multi_term_program_jointly_and_on_fixed_ood_code"

    bundle_payload = {
        "schema": "rung514_constrained_multi_term_programs_v1",
        "collections": {name: _bundle_collection(collection)
                        for name, collection in collections.items()},
        "candidates": candidates, "confirmed": confirmed,
        "physical": None if physical is None else {
            key: value for key, value in physical.items() if key != "diagnostics"},
        "physical_passing": physical_passing,
        "raw_tokens_logits_hidden_states_or_weights_included": False,
    }
    torch.save(bundle_payload, BUNDLE)
    result = {
        "status": "complete", "rung": 514,
        "claim_level": "constrained_multi_term_screen_until_heldout_physical_interchange_passes",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "planted_recovery": planted,
        "calibration": {"discovery": discovery_calibration,
                        "confirmation": confirmation_calibration},
        "calibration_holds": {"discovery": discovery_calibration_ok,
                              "confirmation": confirmation_calibration_ok},
        "source_relation_reproduction": {
            "expected": list(r513.SOURCE_RELATION_NAMES), "observed": source_relations,
            "holds": source_reproduced, "checks": source_checks,
        },
        "diagnostics": {name: collection["diagnostics"]
                        for name, collection in collections.items()},
        "physical_diagnostics": None if physical is None else physical["diagnostics"],
        "analysis": {
            "discovery_summary": discovery_summary,
            "discovery_candidates": candidates,
            "identifiable_candidate_count": identifiable_count,
            "confirmation_checks": confirmation_checks,
            "confirmed_programs": confirmed,
            "physical_checks": physical_checks,
            "physical_programs": physical_passing,
            "reused_program_names": reused,
            "outcome_conditioned_top3": {
                "discovery": top3_candidates, "confirmation": top3_confirmed,
                "physical": top3_physical,
            },
        },
        'pred_a_exact_live_identifiable_joint_program_instrument': pred_a,
        'pred_b_constrained_multi_term_discovery': pred_b,
        'pred_c_frozen_program_predicts_fresh_documents': pred_c,
        'pred_d_bidirectional_physical_program_substitution': pred_d,
        'pred_e_reused_program_across_branch_subsets': pred_e,
        "strong_null": strong_null,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "full_forwards": sum(sum(collection["diagnostics"]["calls"].values())
                                 for collection in collections.values())
                             + (0 if physical is None else physical["diagnostics"]["calls"]),
            "backwards": 0,
            "local_attention_corner_evaluations": sum(
                collection["diagnostics"]["attention_corner_evaluations"]
                for collection in collections.values())
                + (0 if physical is None else physical["diagnostics"]["attention_corner_evaluations"]),
            "local_mlp_corner_evaluations": sum(
                collection["diagnostics"]["mlp_corner_evaluations"]
                for collection in collections.values())
                + (0 if physical is None else physical["diagnostics"]["mlp_corner_evaluations"]),
            "fixed_groups": 48, "sparse_groups": 113520, "all_groups": 113568,
            "discovery_candidates": len(candidates), "confirmed_programs": len(confirmed),
            "physical_programs": len(physical_passing), "maximum_conditional_forwards": 25854,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        },
        "runtime_s": time.time() - started, "next_step": next_step,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 514,
        "pred_a": pred_a, "pred_b": pred_b, "pred_c": pred_c,
        "pred_d": pred_d, "pred_e": pred_e, "strong_null": strong_null,
        "source_relations_reproduced": source_reproduced,
        "discovery_candidates": len(candidates), "confirmed_programs": len(confirmed),
        "physical_programs": len(physical_passing), "reused_programs": reused,
        "top3_discovery": len(top3_candidates), "top3_confirmation": len(top3_confirmed),
        "top3_physical": len(top3_physical),
        "execution_price": result["execution_price"], "next_step": next_step,
    }, indent=2, sort_keys=True))


@torch.no_grad()
def collect_joint(model, rows, task_masks, circuit_masks, circuit_tags,
                  scales, bounds, window_bounds, controls: bool):
    lo, hi, _split = bounds
    documents = hi - lo
    task = torch.zeros(r511.N_ACTIONS, len(r511.ARMS), documents, len(TASK_CELLS),
                       dtype=torch.float64)
    counts = torch.zeros(documents, len(TASK_CELLS), dtype=torch.float64)
    base_task = torch.zeros_like(counts)
    circuit_sums = torch.zeros(
        r511.N_ACTIONS, len(r511.ARMS), 2, 2, len(circuit_tags), dtype=torch.float64)
    circuit_counts = torch.zeros(2, 2, len(circuit_tags), dtype=torch.float64)
    statistics = _new_statistics(tuple(window_bounds), controls)
    diagnostics = r513._empty_diagnostics()
    diagnostics.update({
        "joint_gram_windows": {name: 0 for name in window_bounds},
        "joint_gram_symmetry_max_abs": 0.0,
        "control_joint_grams": 0,
        "control_joint_grams_expected": 0,
        "control_joint_grams_exact": False,
    })
    device = next(model.parameters()).device
    mlp10 = model.transformer.h[parent.TARGET].mlp

    for start in range(lo, hi, parent.BATCH):
        stop, local = start + parent.BATCH, start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in TASK_CELLS}
        copy_mask = masks["all_positive"].to(device)
        active_windows = _active_windows(start, window_bounds)
        for window in active_windows:
            diagnostics["joint_gram_windows"][window] += 1

        direct_logits, _, direct_diag, _ = parent._forward(model, tokens, scales, direct=True)
        diagnostics["calls"]["direct"] += 1
        r511.r510.r509._update_diagnostics(diagnostics, direct_diag)
        absent_logits, absent, absent_diag, _ = r511._captured_forward(
            model, tokens, scales, action="P", absent=True)
        diagnostics["calls"]["analytical"] += 1
        diagnostics["hooks"] += 1
        r511.r510.r509._update_diagnostics(diagnostics, absent_diag)
        base_task[local:local + len(batch_rows)] = parent._task_sums(
            parent._nll(absent_logits, batch_rows).detach().cpu().unsqueeze(0), masks)[0]

        term_vectors = [[None] * len(r513.TERM_NAMES) for _ in range(r513.N_LOCAL_NODES)]
        total_vectors = [[None] * 2 for _ in range(r513.N_LOCAL_NODES)]
        source_vectors = [None] * r513.N_LOCAL_NODES
        action_nll = []
        for action_index, source in enumerate(parent.SOURCES):
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
            nll_rows = [parent._nll(logits, batch_rows).detach().cpu()]
            for subset_index in range(r511.N_SUBSETS):
                delta10 = r511.subset_output(branches, subset_index)
                replacement = current["deployed_write"].float() - delta10
                removed_result, removed_consumer = r513.factor_consumer_call(
                    model, lambda source=source, replacement=replacement: parent.score_parent.run_forward(
                        model, tokens, action=source, scales=scales,
                        patch_writes={"m10": replacement.to(current["deployed_write"].dtype)}))
                edited_logits, _captures, patch_diag, patch_audit = removed_result
                diagnostics["calls"]["analytical"] += 1
                diagnostics["factor_consumer_captures"] += 1
                diagnostics["subset_patches"] += patch_audit["patches"]
                edit_rms = patch_diag["patch_rms_max"]
                diagnostics["zero_term_edits"] += int(edit_rms <= 0)
                if edit_rms > 0:
                    diagnostics["minimum_nonzero_term_edit_rms"] = min(
                        diagnostics["minimum_nonzero_term_edit_rms"], edit_rms)
                nll_rows.append(parent._nll(edited_logits, batch_rows).detach().cpu())
                if subset_index not in r513.SELECTED_SUBSETS:
                    continue
                selected_subset = r513.SELECTED_SUBSETS.index(subset_index)
                node = r513.local_node(action_index, selected_subset)
                terms, term_diag = r513.exact_terms(model, removed_consumer, current_consumer)
                diagnostics["attention_corner_evaluations"] += 32
                diagnostics["mlp_corner_evaluations"] += 4
                r513._update_exact_diagnostics(diagnostics, term_diag)
                term_vectors[node] = [term[copy_mask].reshape(-1).float()
                                      for term in terms]
                total_vectors[node][0] = (
                    current_consumer["a11"].float() - removed_consumer["a11"].float()
                )[copy_mask].reshape(-1)
                total_vectors[node][1] = (
                    current_consumer["m11"].float() - removed_consumer["m11"].float()
                )[copy_mask].reshape(-1)
                source_vectors[node] = delta10[copy_mask].reshape(-1).float()
                del terms
            task[action_index, :, local:local + len(batch_rows)] = parent._task_sums(
                torch.stack(nll_rows), masks)
            action_nll.append(torch.stack(nll_rows))

        if any(value is None for rows_ in term_vectors for value in rows_) \
                or any(value is None for rows_ in total_vectors for value in rows_) \
                or any(value is None for value in source_vectors):
            raise RuntimeError("joint term response collection incomplete")
        update_joint_statistics(
            statistics, term_vectors, total_vectors, source_vectors,
            start, window_bounds, controls)
        if controls:
            diagnostics["control_joint_grams"] += (
                len(r513.SELECTED_SUBSETS) * len(SITE_TERMS) * len(CONTROL_SEEDS))
        counts[local:local + len(batch_rows)] = torch.stack(
            [masks[cell].sum(1).double() for cell in TASK_CELLS], -1)
        matrix, observed = parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        circuit_counts += observed
        for action_index, nll_stack in enumerate(action_nll):
            circuit_sums[action_index] += torch.matmul(
                nll_stack.view(len(r511.ARMS), -1).double(), matrix.T,
            ).view(len(r511.ARMS), 2, 2, len(circuit_tags))

    batches = documents // parent.BATCH
    diagnostics["calls_expected"] = {
        "direct": batches,
        "analytical": batches * (1 + r511.N_ACTIONS * (1 + r511.N_SUBSETS)),
    }
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["hooks_expected"] = batches * (1 + r511.N_ACTIONS)
    diagnostics["hooks_exact"] = diagnostics["hooks"] == diagnostics["hooks_expected"]
    diagnostics["four_corner_replays_expected"] = batches * r511.N_ACTIONS
    diagnostics["four_corner_replays_exact"] = (
        diagnostics["four_corner_replays"] == diagnostics["four_corner_replays_expected"])
    diagnostics["subset_patches_expected"] = batches * r511.N_ACTIONS * r511.N_SUBSETS
    diagnostics["subset_patches_exact"] = (
        diagnostics["subset_patches"] == diagnostics["subset_patches_expected"])
    diagnostics["patches"] = diagnostics["subset_patches"]
    diagnostics["patches_expected"] = diagnostics["subset_patches_expected"]
    diagnostics["patches_exact"] = diagnostics["subset_patches_exact"]
    diagnostics["factor_consumer_captures_expected"] = batches * r511.N_ACTIONS * (1 + r511.N_SUBSETS)
    diagnostics["factor_consumer_captures_exact"] = (
        diagnostics["factor_consumer_captures"]
        == diagnostics["factor_consumer_captures_expected"])
    diagnostics["attention_corner_evaluations_expected"] = batches * r513.N_LOCAL_NODES * 32
    diagnostics["attention_corner_evaluations_exact"] = (
        diagnostics["attention_corner_evaluations"]
        == diagnostics["attention_corner_evaluations_expected"])
    diagnostics["mlp_corner_evaluations_expected"] = batches * r513.N_LOCAL_NODES * 4
    diagnostics["mlp_corner_evaluations_exact"] = (
        diagnostics["mlp_corner_evaluations"] == diagnostics["mlp_corner_evaluations_expected"])
    diagnostics["control_joint_grams_expected"] = (
        batches * len(r513.SELECTED_SUBSETS) * len(SITE_TERMS) * len(CONTROL_SEEDS)
        if controls else 0)
    diagnostics["control_joint_grams_exact"] = (
        diagnostics["control_joint_grams"] == diagnostics["control_joint_grams_expected"])
    for window in window_bounds:
        for selected_subset in range(len(r513.SELECTED_SUBSETS)):
            for site in SITE_TERMS:
                joint = statistics["real"][window][selected_subset][site]["joint"]
                diagnostics["joint_gram_symmetry_max_abs"] = max(
                    diagnostics["joint_gram_symmetry_max_abs"],
                    float((joint - joint.T).abs().max()))
    return {
        "bounds": bounds, "window_bounds": window_bounds,
        "arms": r511.ARMS, "task": task, "task_counts": counts,
        "base_task": base_task, "source_task": task[:, 0],
        "circuit_tags": tuple(circuit_tags), "circuit_sums": circuit_sums,
        "circuit_counts": circuit_counts, "statistics": statistics,
        "diagnostics": diagnostics,
    }


def program_tensor(terms: tuple[torch.Tensor, ...], candidate: dict) -> torch.Tensor:
    offset = SITE_OFFSETS[candidate["site"]]
    coefficient = candidate["coefficient"]
    selected = [terms[offset + index].float() * float(value)
                for index, value in enumerate(coefficient) if value != 0]
    if not selected:
        raise RuntimeError("empty registered program")
    return sum(selected)


@torch.no_grad()
def collect_physical(model, rows, task_masks, circuit_masks, circuit_tags,
                     scales, bounds, candidates):
    lo, hi, _split = bounds
    data = r513._physical_empty(len(candidates), hi - lo, len(circuit_tags))
    data["bounds"] = bounds
    diagnostics = {
        "calls": 0, "calls_expected": 0, "calls_exact": False,
        "branch_patches": 0, "branch_patches_expected": 0,
        "consumer_patches": 0, "consumer_patches_expected": 0,
        "patches_exact": False, "zero_patch_edits": 0,
        "minimum_patch_rms": math.inf, "maximum_patch_capture_error": 0.0,
        "attention_corner_evaluations": 0, "mlp_corner_evaluations": 0,
    }
    device = next(model.parameters()).device
    mlp10 = model.transformer.h[parent.TARGET].mlp
    candidates_by_subset = {
        selected_subset: [index for index, row in enumerate(candidates)
                          if row["selected_subset"] == selected_subset]
        for selected_subset in range(len(r513.SELECTED_SUBSETS))
    }

    for start in range(lo, hi, parent.BATCH):
        stop, local = start + parent.BATCH, start - lo
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks = {cell: task_masks[cell][start:stop] for cell in TASK_CELLS}
        absent_result, _ = r513.factor_consumer_call(
            model, lambda: r511._captured_forward(model, tokens, scales, action="P", absent=True))
        _absent_logits, absent, _absent_diag, _ = absent_result
        diagnostics["calls"] += 1

        action_data = {}
        for action_index, source in enumerate(parent.SOURCES):
            current_result, current_consumer = r513.factor_consumer_call(
                model, lambda source=source: r511._captured_forward(
                    model, tokens, scales, action=source))
            logits, current, _diag, _audit = current_result
            diagnostics["calls"] += 1
            nll = parent._nll(logits, batch_rows).detach().cpu()
            data["intact_task"][action_index, local:local + len(batch_rows)] = parent._task_sums(
                nll.unsqueeze(0), masks)[0]
            branches, _ = r511.deployed_branches(mlp10, absent, current)
            action_data[action_index] = {"current": current_consumer, "programs": {}, "nll": nll}
            for selected_subset, subset_index in enumerate(r513.SELECTED_SUBSETS):
                delta10 = r511.subset_output(branches, subset_index)
                replacement = current["deployed_write"].float() - delta10
                removed_result, removed_consumer = r513.factor_consumer_call(
                    model, lambda source=source, replacement=replacement: parent.score_parent.run_forward(
                        model, tokens, action=source, scales=scales,
                        patch_writes={"m10": replacement.to(current["deployed_write"].dtype)}))
                _removed_logits, _captures, diag, audit = removed_result
                diagnostics["calls"] += 1
                diagnostics["branch_patches"] += audit["patches"]
                patch_rms = diag["patch_rms_max"]
                diagnostics["zero_patch_edits"] += int(patch_rms <= 0)
                diagnostics["minimum_patch_rms"] = min(
                    diagnostics["minimum_patch_rms"], patch_rms if patch_rms > 0 else math.inf)
                needed = candidates_by_subset[selected_subset]
                if needed:
                    terms, _ = r513.exact_terms(model, removed_consumer, current_consumer)
                    diagnostics["attention_corner_evaluations"] += 32
                    diagnostics["mlp_corner_evaluations"] += 4
                    for candidate_index in needed:
                        action_data[action_index]["programs"][candidate_index] = program_tensor(
                            terms, candidates[candidate_index])

        data["task_counts"][local:local + len(batch_rows)] = torch.stack(
            [masks[cell].sum(1).double() for cell in TASK_CELLS], -1)
        matrix, observed = parent.state_parent._circuit_mask_matrix(
            circuit_masks, circuit_tags, start, stop, bounds)
        data["circuit_counts"] += observed
        for action_index in range(r511.N_ACTIONS):
            nll = action_data[action_index]["nll"]
            data["intact_circuit_sums"][action_index] += torch.matmul(
                nll.reshape(1, -1).double(), matrix.T).view(2, 2, len(circuit_tags))

        for candidate_index, candidate in enumerate(candidates):
            site = candidate["site"]
            for action_index in range(r511.N_ACTIONS):
                target = action_data[action_index]
                term = target["programs"][candidate_index]
                replacement = target["current"][site].float() - term
                logits, captures, diag, audit = parent.score_parent.run_forward(
                    model, tokens, action=parent.SOURCES[action_index], scales=scales,
                    patch_writes={site: replacement.to(target["current"][site].dtype)},
                    capture_keys=(site,))
                diagnostics["calls"] += 1
                diagnostics["consumer_patches"] += audit["patches"]
                diagnostics["maximum_patch_capture_error"] = max(
                    diagnostics["maximum_patch_capture_error"],
                    float((captures[site] - replacement.to(captures[site].dtype)).float().abs().max()))
                patch_rms = diag["patch_rms_max"]
                diagnostics["zero_patch_edits"] += int(patch_rms <= 0)
                diagnostics["minimum_patch_rms"] = min(
                    diagnostics["minimum_patch_rms"], patch_rms if patch_rms > 0 else math.inf)
                nll = parent._nll(logits, batch_rows).detach().cpu()
                data["removal_task"][candidate_index, action_index,
                                     local:local + len(batch_rows)] = parent._task_sums(
                                         nll.unsqueeze(0), masks)[0]
                data["removal_circuit_sums"][candidate_index, action_index] += torch.matmul(
                    nll.reshape(1, -1).double(), matrix.T).view(2, 2, len(circuit_tags))

            direction_index = 0
            for relation_name in r513.RELATION_NAMES:
                relation = candidate["relations"][relation_name]
                for target_action, donor_action, beta in (
                    (relation["left_action"], relation["right_action"],
                     relation["beta_left_from_right"]),
                    (relation["right_action"], relation["left_action"],
                     1.0 / relation["beta_left_from_right"]),
                ):
                    target, donor = action_data[target_action], action_data[donor_action]
                    target_term = target["programs"][candidate_index]
                    donor_term = donor["programs"][candidate_index]
                    replacement = (target["current"][site].float() - target_term
                                   + beta * donor_term)
                    logits, captures, diag, audit = parent.score_parent.run_forward(
                        model, tokens, action=parent.SOURCES[target_action], scales=scales,
                        patch_writes={site: replacement.to(target["current"][site].dtype)},
                        capture_keys=(site,))
                    diagnostics["calls"] += 1
                    diagnostics["consumer_patches"] += audit["patches"]
                    diagnostics["maximum_patch_capture_error"] = max(
                        diagnostics["maximum_patch_capture_error"],
                        float((captures[site] - replacement.to(captures[site].dtype)).float().abs().max()))
                    patch_rms = diag["patch_rms_max"]
                    diagnostics["zero_patch_edits"] += int(patch_rms <= 0)
                    diagnostics["minimum_patch_rms"] = min(
                        diagnostics["minimum_patch_rms"], patch_rms if patch_rms > 0 else math.inf)
                    nll = parent._nll(logits, batch_rows).detach().cpu()
                    data["substitution_task"][candidate_index, direction_index,
                                               local:local + len(batch_rows)] = parent._task_sums(
                                                   nll.unsqueeze(0), masks)[0]
                    data["substitution_circuit_sums"][candidate_index, direction_index] += torch.matmul(
                        nll.reshape(1, -1).double(), matrix.T).view(2, 2, len(circuit_tags))
                    direction_index += 1

    batches = (hi - lo) // parent.BATCH
    diagnostics["calls_expected"] = batches * (
        1 + r511.N_ACTIONS + r511.N_ACTIONS * len(r513.SELECTED_SUBSETS)
        + 10 * len(candidates))
    diagnostics["calls_exact"] = diagnostics["calls"] == diagnostics["calls_expected"]
    diagnostics["branch_patches_expected"] = (
        batches * r511.N_ACTIONS * len(r513.SELECTED_SUBSETS))
    diagnostics["consumer_patches_expected"] = batches * 10 * len(candidates)
    diagnostics["patches_exact"] = bool(
        diagnostics["branch_patches"] == diagnostics["branch_patches_expected"]
        and diagnostics["consumer_patches"] == diagnostics["consumer_patches_expected"])
    data["diagnostics"] = diagnostics
    return data


if __name__ == "__main__":
    main()
