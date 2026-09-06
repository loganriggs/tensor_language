#!/usr/bin/env python3
"""Factor exact MLP4-to-H1/H4 bank mediation into routing and local V9 value."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_factorization pred_b_writer_and_bank_route_recur pred_c_local_value_is_dominant_path pred_d_interaction_is_secondary pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2 as rows_builder
import circuit_das_subspace as das
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_tense_auxiliary_is_was_mlp4_to_l9h1_h4_path_mediation_v1 as path
import run_tense_auxiliary_is_was_mlp4_contextual_source_factorial_v1 as writer


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.mlp4_h1h4_bank_routing_local_value_factorial_v1"
PATHS = {
    "path_result": ROOT / "circuits/followups/tense_auxiliary_is_was_mlp4_to_l9h1_h4_path_mediation_v1_result.json",
    "path_instrument": ROOT / "ops/run_tense_auxiliary_is_was_mlp4_to_l9h1_h4_path_mediation_v1.py",
    "cross_task_local_value": ROOT / "circuits/followups/aspectual_tense_h1h4_carrier_effective_value_branch_factorial_v1_result.json",
}
EXPECTED_PRIOR_SHA256 = "1e0031dc0c36448b0ba7145c822ceed5018dd3d22feba2c1930cc24760fb3f19"
EXPECTED = {
    "path_result": "6f5d01abec1debb41f67178a57db1ce79ad2a704e2ab094e2d8b1f055b3865d5",
    "path_instrument": "ba2d36ca8a2dc18de92f4ea43d25f7769d5bcb2cfa110290c355d6aff8717501",
    "cross_task_local_value": "9f3c04abe6bca5448d228d6fc71951b804e6a66a9a95834e59461a0d32bcf9b6",
}
FACTORS = ("routing_on_base_value", "local_v9_content_change", "routing_local_interaction")
HEADS = (1, 4)


class ExperimentError(RuntimeError):
    pass


def sha(pathname):
    return hashlib.sha256(pathname.read_bytes()).hexdigest()


def subsets():
    return tuple(subset for width in range(4) for subset in itertools.combinations(FACTORS, width))


def arm_id(subset):
    return "empty" if not subset else "+".join(subset)


def factorial_shapley(subset_values):
    result = {}
    for factor in FACTORS:
        total = 0.0
        for subset in subsets():
            if factor in subset:
                continue
            extended = tuple(item for item in FACTORS if item in set(subset) | {factor})
            total += math.factorial(len(subset)) * math.factorial(2 - len(subset)) / math.factorial(3) * (subset_values[extended] - subset_values[subset])
        result[factor] = total
    return result


class Backend(path.Backend):
    def capture_writer_raw(self, base_batch, donor_batch, base_capture, donor_capture, factors):
        raw = {}
        head_dim = self.model.config.n_embd // self.model.config.n_head

        def save(name):
            def hook(_module, _arguments, output):
                raw[name] = output.detach().clone().view(len(base_batch.row_ids), output.shape[1], self.model.config.n_head, head_dim)
            return hook

        handles = [self.model.transformer.h[0].attn.c_v.register_forward_hook(save("v1")), self.model.transformer.h[9].attn.c_v.register_forward_hook(save("v9"))]
        try:
            output, attention, tensor_error = self.capture_writer(base_batch, donor_batch, base_capture, donor_capture, factors)
        finally:
            for handle in handles:
                handle.remove()
        if set(raw) != {"v1", "v9"}:
            raise ExperimentError("raw value hook coverage failed")
        return output, attention, raw, tensor_error


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(filename) for name, filename in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    rows_by_bank = rows_builder.build_rows_by_bank()
    rows_builder.validate_rows_by_bank(rows_by_bank)
    rows = [row for row in rows_by_bank["is_was"] if row["transform_id"] in {"A1", "A2"}]
    authority = json.loads(PATHS["path_result"].read_text())
    if len(rows) != 32 or authority.get("terminal") != "screen" or len(subsets()) != 8:
        raise ExperimentError("population, path authority, or factorial changed")
    return rows


def factor_tensors(backend, base_batch, donor_batch, base_attention, hybrid_attention, base_raw, hybrid_raw):
    tensors, factor_error = {}, 0.0
    lamb = backend.model.transformer.h[9].attn.lamb.detach().float()
    v1_error = float((hybrid_raw["v1"].float() - base_raw["v1"].float()).abs().max())
    base_effective_error = float(((1.0 - lamb) * base_raw["v9"].float() + lamb * base_raw["v1"].float() - base_attention["value"].float()).abs().max())
    hybrid_effective_error = float(((1.0 - lamb) * hybrid_raw["v9"].float() + lamb * hybrid_raw["v1"].float() - hybrid_attention["value"].float()).abs().max())
    positions = writer.source_positions(base_batch, donor_batch)
    for index, (query, bank) in enumerate(zip(base_batch.semantic_positions, positions)):
        for head in HEADS:
            factors = {name: backend.torch.zeros_like(base_raw["v9"][index, 0, head], dtype=backend.torch.float32) for name in FACTORS}
            full_delta = backend.torch.zeros_like(factors[FACTORS[0]])
            for position in bank:
                p_base = base_attention["pattern"][index, head, query, position].float()
                p_hybrid = hybrid_attention["pattern"][index, head, query, position].float()
                delta_p = p_hybrid - p_base
                v9_base = base_raw["v9"][index, position, head].float()
                delta_v9 = hybrid_raw["v9"][index, position, head].float() - v9_base
                effective_base = base_attention["value"][index, position, head].float()
                factors[FACTORS[0]] += delta_p * effective_base
                factors[FACTORS[1]] += p_base * (1.0 - lamb) * delta_v9
                factors[FACTORS[2]] += delta_p * (1.0 - lamb) * delta_v9
                full_delta += p_hybrid * hybrid_attention["value"][index, position, head].float() - p_base * effective_base
            closure = sum(factors.values(), backend.torch.zeros_like(full_delta))
            factor_error = max(factor_error, float((closure - full_delta).abs().max()))
            tensors[(index, head)] = factors
    return tensors, {"layer0_v1_invariance_max_abs_error": v1_error, "base_effective_value_recombination_max_abs_error": base_effective_error, "hybrid_effective_value_recombination_max_abs_error": hybrid_effective_error, "bank_factor_closure_max_abs_error": factor_error, "observed_lambda9": float(lamb)}


def intervene(backend, base_batch, tensors, subset):
    head_dim = backend.model.config.n_embd // backend.model.config.n_head

    def patch_heads(_module, arguments):
        flattened = arguments[0]
        heads = flattened.view(len(base_batch.row_ids), flattened.shape[1], backend.model.config.n_head, head_dim).clone()
        for index, query in enumerate(base_batch.semantic_positions):
            for head in HEADS:
                delta = sum((tensors[(index, head)][factor] for factor in subset), backend.torch.zeros(head_dim, device=backend.device, dtype=backend.torch.float32))
                heads[index, query, head] = (heads[index, query, head].float() + delta).to(heads.dtype)
        return (heads.reshape_as(flattened),) + tuple(arguments[1:])

    handle = backend.model.transformer.h[9].attn.c_proj.register_forward_pre_hook(patch_heads)
    try:
        output, _capture = backend.manual_forward(base_batch)
    finally:
        handle.remove()
    return output


def summarize(values):
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("missing or nonfinite recovery")
    return {"count": len(values), "mean_recovery": statistics.fmean(values), "mean_absolute_recovery": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def main():
    rows = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "rows": 32, "factor_arms": 8, "intervention_records": 256, "model_forwards": 12, "example_evaluations": 384, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = Backend.load("cuda")
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_output, base_mlp_capture = backend.capture_bilinear(base_batch)
    donor_output, donor_mlp_capture = backend.capture_bilinear(donor_batch)
    empty_output, base_attention, base_raw, empty_tensor_error = backend.capture_writer_raw(base_batch, donor_batch, base_mlp_capture, donor_mlp_capture, ())
    writer_output, hybrid_attention, hybrid_raw, writer_tensor_error = backend.capture_writer_raw(base_batch, donor_batch, base_mlp_capture, donor_mlp_capture, path.FACTORS)
    forward_calls, evaluations = 4, 128
    native = {}
    for side, output in (("base", base_output), ("donor", donor_output)):
        for row, pair in zip(rows, output.answer_foil):
            native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(str(row["row_id"]), row["transform_id"], side, *producer._finite_pair(pair))
    empty_error = max(abs(a - b) for pair_a, pair_b in zip(base_output.answer_foil, empty_output.answer_foil) for a, b in zip(pair_a, pair_b))
    source_error = max(float(base_attention["reconstruction_max_abs"]), float(hybrid_attention["reconstruction_max_abs"]))
    tensors, instrument = factor_tensors(backend, base_batch, donor_batch, base_attention, hybrid_attention, base_raw, hybrid_raw)
    instrument.update({"empty_writer_hook_max_abs_logit_error": empty_error, "bilinear_tensor_reconstruction_max_abs_error": max(empty_tensor_error, writer_tensor_error), "attention_source_reconstruction_max_abs_error": source_error})
    capability_cells = []
    for family in ("A1", "A2"):
        for direction in ("present_to_past", "past_to_present"):
            cell_rows = [row for row in rows if row["transform_id"] == family and row["direction_id"] == direction]
            for side in ("base", "donor"):
                accuracy = sum(native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows) / len(cell_rows)
                capability_cells.append({"family": family, "direction": direction, "side": side, "count": len(cell_rows), "accuracy": accuracy, "threshold": 0.85, "passed": accuracy >= 0.85})
    records, summaries, values = [], {}, {}
    writer_recoveries = {"A1": [], "A2": []}
    for row, pair in zip(rows, writer_output.answer_foil):
        row_id, family = str(row["row_id"]), row["transform_id"]
        answer, foil = producer._finite_pair(pair)
        writer_recoveries[family].append(kernel.signed_pairwise_donor_recovery(-native[(row_id, "base")].margin, native[(row_id, "donor")].margin, -(answer - foil)))
    writer_summary = {family: summarize(writer_recoveries[family]) for family in ("A1", "A2")}
    for subset in subsets():
        output = intervene(backend, base_batch, tensors, subset)
        forward_calls += 1
        evaluations += len(rows)
        by_family = {"A1": [], "A2": []}
        for row, pair in zip(rows, output.answer_foil):
            row_id, family = str(row["row_id"]), row["transform_id"]
            answer, foil = producer._finite_pair(pair)
            recovery = kernel.signed_pairwise_donor_recovery(-native[(row_id, "base")].margin, native[(row_id, "donor")].margin, -(answer - foil))
            by_family[family].append(recovery)
            records.append({"arm": arm_id(subset), "family": family, "row_id": row_id, "recovery": recovery})
        summaries[arm_id(subset)] = {family: summarize(by_family[family]) for family in ("A1", "A2")}
        for family in ("A1", "A2"):
            values[(subset, family)] = summaries[arm_id(subset)][family]["mean_recovery"]
    shapley = {family: factorial_shapley({subset: values[(subset, family)] for subset in subsets()}) for family in ("A1", "A2")}
    local = (FACTORS[1],)
    additive = (FACTORS[0], FACTORS[1])
    local_retained = {family: values[(local, family)] / values[(FACTORS, family)] for family in ("A1", "A2")}
    additive_retained = {family: values[(additive, family)] / values[(FACTORS, family)] for family in ("A1", "A2")}
    writer_mean = statistics.fmean(writer_summary[family]["mean_recovery"] for family in ("A1", "A2"))
    full_mean = statistics.fmean(values[(FACTORS, family)] for family in ("A1", "A2"))
    pred_a = all(cell["passed"] for cell in capability_cells) and instrument["empty_writer_hook_max_abs_logit_error"] <= 1e-4 and instrument["bilinear_tensor_reconstruction_max_abs_error"] <= 2e-3 and max(value for key, value in instrument.items() if key.endswith("error") and key != "bilinear_tensor_reconstruction_max_abs_error") <= 1e-4 and len(records) == 256
    pred_b = abs(writer_mean - 0.31871042052323534) <= 0.02 and abs(full_mean - 0.15771071235856) <= 0.03 and all(values[(FACTORS, family)] > 0.0 and summaries[arm_id(FACTORS)][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_c = all(shapley[family][FACTORS[1]] > 0.0 and shapley[family][FACTORS[1]] > shapley[family][FACTORS[0]] and shapley[family][FACTORS[1]] > shapley[family][FACTORS[2]] and local_retained[family] >= 0.60 and summaries[FACTORS[1]][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = all(additive_retained[family] >= 0.80 and summaries[arm_id(additive)][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    price = {"model_forwards": forward_calls, "example_evaluations": evaluations, "rows": len(rows), "factor_arms": 8, "intervention_records": len(records), "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 12, "example_evaluations": 384, "rows": 32, "factor_arms": 8, "intervention_records": 256, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_authority_capability_exact_factorization": pred_a, "pred_b_writer_and_bank_route_recur": pred_b, "pred_c_local_value_is_dominant_path": pred_c, "pred_d_interaction_is_secondary": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e else "invalid")
    result = {"schema": "tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_result_v1", "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": EXPECTED, "capability_cells": capability_cells, "instrument": instrument, "writer_summary": writer_summary, "summaries": summaries, "factor_shapley": shapley, "local_value_retained_fraction": local_retained, "routing_plus_local_retained_fraction": additive_retained, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "mlp4_reaches_h1h4_bank_primarily_through_local_l9_value", "null": "local_value_dominance_or_interaction_prediction_misses", "invalid": "authority_capability_capture_closure_recurrence_coverage_or_price_invalid"}[terminal], "serial_seconds": time.perf_counter() - started, "next_action": "test the same internal edge computation on the has/had path" if terminal == "screen" else "retain the exact unsplit MLP4-to-bank path"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "writer_summary", "summaries", "factor_shapley", "local_value_retained_fraction", "routing_plus_local_retained_fraction", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
