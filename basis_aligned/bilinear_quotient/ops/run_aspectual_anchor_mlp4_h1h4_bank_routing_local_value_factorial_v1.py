#!/usr/bin/env python3
"""Test has/had reuse of the MLP4-to-H1/H4 routing/local-value computation."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_factorization pred_b_writer_and_bank_route_recur pred_c_local_value_factor_reuses pred_d_interaction_is_secondary pred_e_exact_zero_fit_price
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

import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_block4_contextual_source_writer_factorial_v1 as block4
import run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1 as path


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp4_h1h4_bank_routing_local_value_factorial_v1.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp4_h1h4_bank_routing_local_value_factorial_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp4_h1h4_bank_routing_local_value_factorial_v1"
PATHS = {
    "has_had_path_result": ROOT / "circuits/followups/aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1_result.json",
    "has_had_path_instrument": ROOT / "ops/run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1.py",
    "is_was_factor_screen": ROOT / "circuits/followups/tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1_result.json",
    "typed_shared_path": ROOT / "circuits/followups/aspectual_tense_typed_shared_contextual_path_v1_result.json",
}
EXPECTED_PRIOR_SHA256 = "3aabf6b2edfafe5b38d97859b7a8afa5a940dde2e044fd897f095e4548a1b5ad"
EXPECTED = {
    "has_had_path_result": "649cc961fd4203a9d7489344bbf169754081a288b5d575bcefcab2caf41da9ab",
    "has_had_path_instrument": "5449d45505267f2ebc92c9285b7a984fce773f834c2071e1d7e6d1a9f1949372",
    "is_was_factor_screen": "9a00cefe8986b1459c334c445a28208ae54911b81e5a42d78e1bc878777f07e4",
    "typed_shared_path": "afb17159330dc6abcf018d36313a7df2c78c6708b67feb8d2f2d9d2eee50faf0",
}
FACTORS = ("routing_on_base_value", "local_v9_content_change", "routing_local_interaction")
HEADS = (1, 4)


class ExperimentError(RuntimeError):
    pass


def sha(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()


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


class Backend(path.PathBackend):
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
    rows, spec = path.validate_static()
    parent = json.loads(PATHS["has_had_path_result"].read_text())
    is_was = json.loads(PATHS["is_was_factor_screen"].read_text())
    typed = json.loads(PATHS["typed_shared_path"].read_text())
    if len(rows) != 64 or parent.get("terminal") != "null" or not parent["predictions"]["pred_a_exact_path_instrument"] or is_was.get("terminal") != "screen" or typed.get("terminal") != "screen" or len(subsets()) != 8:
        raise ExperimentError("population or authority state changed")
    return rows, spec


def factor_tensors(backend, base_batch, donor_batch, base_attention, hybrid_attention, base_raw, hybrid_raw):
    tensors, factor_error = {}, 0.0
    lamb = backend.model.transformer.h[9].attn.lamb.detach().float()
    v1_error = float((hybrid_raw["v1"].float() - base_raw["v1"].float()).abs().max())
    base_effective_error = float(((1.0 - lamb) * base_raw["v9"].float() + lamb * base_raw["v1"].float() - base_attention["value"].float()).abs().max())
    hybrid_effective_error = float(((1.0 - lamb) * hybrid_raw["v9"].float() + lamb * hybrid_raw["v1"].float() - hybrid_attention["value"].float()).abs().max())
    positions = block4.source_positions(base_batch, donor_batch)
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
    rows, spec = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, "rows": 64, "family_batches": 2, "factor_arms": 8, "intervention_records": 512, "model_forwards": 24, "example_evaluations": 768, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = Backend.load("cuda")
    records, capability_cells = [], []
    summaries = {arm_id(subset): {} for subset in subsets()}
    writer_summary = {}
    instrument = {"layer0_v1_invariance_max_abs_error": 0.0, "base_effective_value_recombination_max_abs_error": 0.0, "hybrid_effective_value_recombination_max_abs_error": 0.0, "bank_factor_closure_max_abs_error": 0.0, "empty_writer_hook_max_abs_logit_error": 0.0, "bilinear_tensor_reconstruction_max_abs_error": 0.0, "attention_source_reconstruction_max_abs_error": 0.0, "observed_lambda9": None}
    forward_calls = evaluations = 0
    values = {}
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        if len(family_rows) != 32:
            raise ExperimentError("family batch size changed")
        base_batch = producer._batch(spec, family_rows, "base")
        donor_batch = producer._batch(spec, family_rows, "donor")
        base_output, base_mlp_capture = backend.capture_bilinear(base_batch)
        donor_output, donor_mlp_capture = backend.capture_bilinear(donor_batch)
        empty_output, base_attention, base_raw, empty_tensor_error = backend.capture_writer_raw(base_batch, donor_batch, base_mlp_capture, donor_mlp_capture, ())
        writer_output, hybrid_attention, hybrid_raw, writer_tensor_error = backend.capture_writer_raw(base_batch, donor_batch, base_mlp_capture, donor_mlp_capture, path.WRITER_FACTORS)
        forward_calls += 4
        evaluations += 4 * len(family_rows)
        native = {}
        for side, output in (("base", base_output), ("donor", donor_output)):
            for row, pair in zip(family_rows, output.answer_foil):
                native[(str(row.row_id), side)] = producer.NativeLogitEvidence(str(row.row_id), family, side, *producer._finite_pair(pair))
        for direction in ("present_to_past", "past_to_present"):
            cell_rows = [row for row in family_rows if row.direction_id == direction]
            for side in ("base", "donor"):
                accuracy = sum(native[(str(row.row_id), side)].margin > 0.0 for row in cell_rows) / len(cell_rows)
                capability_cells.append({"family": family, "direction": direction, "side": side, "count": len(cell_rows), "accuracy": accuracy, "threshold": 0.85, "passed": accuracy >= 0.85})
        checks = factor_tensors(backend, base_batch, donor_batch, base_attention, hybrid_attention, base_raw, hybrid_raw)
        tensors, factor_checks = checks
        for key, value in factor_checks.items():
            if key == "observed_lambda9":
                instrument[key] = value
            else:
                instrument[key] = max(float(instrument[key]), value)
        instrument["empty_writer_hook_max_abs_logit_error"] = max(float(instrument["empty_writer_hook_max_abs_logit_error"]), max(abs(a - b) for pair_a, pair_b in zip(base_output.answer_foil, empty_output.answer_foil) for a, b in zip(pair_a, pair_b)))
        instrument["bilinear_tensor_reconstruction_max_abs_error"] = max(float(instrument["bilinear_tensor_reconstruction_max_abs_error"]), empty_tensor_error, writer_tensor_error)
        instrument["attention_source_reconstruction_max_abs_error"] = max(float(instrument["attention_source_reconstruction_max_abs_error"]), float(base_attention["reconstruction_max_abs"]), float(hybrid_attention["reconstruction_max_abs"]))
        writer_values = []
        for row, pair in zip(family_rows, writer_output.answer_foil):
            answer, foil = producer._finite_pair(pair)
            writer_values.append(kernel.signed_pairwise_donor_recovery(-native[(str(row.row_id), "base")].margin, native[(str(row.row_id), "donor")].margin, -(answer - foil)))
        writer_summary[family] = summarize(writer_values)
        for subset in subsets():
            output = intervene(backend, base_batch, tensors, subset)
            forward_calls += 1
            evaluations += len(family_rows)
            arm_values = []
            for row, pair in zip(family_rows, output.answer_foil):
                answer, foil = producer._finite_pair(pair)
                recovery = kernel.signed_pairwise_donor_recovery(-native[(str(row.row_id), "base")].margin, native[(str(row.row_id), "donor")].margin, -(answer - foil))
                arm_values.append(recovery)
                records.append({"arm": arm_id(subset), "family": family, "row_id": str(row.row_id), "recovery": recovery})
            summaries[arm_id(subset)][family] = summarize(arm_values)
            values[(subset, family)] = summaries[arm_id(subset)][family]["mean_recovery"]
    shapley = {family: factorial_shapley({subset: values[(subset, family)] for subset in subsets()}) for family in ("A1", "A2")}
    local = (FACTORS[1],)
    additive = (FACTORS[0], FACTORS[1])
    local_retained = {family: values[(local, family)] / values[(FACTORS, family)] for family in ("A1", "A2")}
    additive_retained = {family: values[(additive, family)] / values[(FACTORS, family)] for family in ("A1", "A2")}
    writer_mean = statistics.fmean(writer_summary[family]["mean_recovery"] for family in ("A1", "A2"))
    full_mean = statistics.fmean(values[(FACTORS, family)] for family in ("A1", "A2"))
    pred_a = all(cell["passed"] for cell in capability_cells) and instrument["bilinear_tensor_reconstruction_max_abs_error"] <= 2e-3 and max(value for key, value in instrument.items() if key.endswith("error") and key != "bilinear_tensor_reconstruction_max_abs_error") <= 1e-4 and len(records) == 512
    pred_b = abs(writer_mean - 0.33379277118533013) <= 0.02 and abs(full_mean - 0.1327154237991511) <= 0.03 and all(values[(FACTORS, family)] > 0.0 and summaries[arm_id(FACTORS)][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_c = all(shapley[family][FACTORS[1]] > 0.0 and shapley[family][FACTORS[1]] > shapley[family][FACTORS[0]] and shapley[family][FACTORS[1]] > shapley[family][FACTORS[2]] and local_retained[family] >= 0.60 and summaries[FACTORS[1]][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = all(additive_retained[family] >= 0.80 and summaries[arm_id(additive)][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    price = {"model_forwards": forward_calls, "example_evaluations": evaluations, "rows": len(rows), "family_batches": 2, "factor_arms": 8, "intervention_records": len(records), "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    pred_e = price == {"model_forwards": 24, "example_evaluations": 768, "rows": 64, "family_batches": 2, "factor_arms": 8, "intervention_records": 512, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0}
    predictions = {"pred_a_authority_capability_exact_factorization": pred_a, "pred_b_writer_and_bank_route_recur": pred_b, "pred_c_local_value_factor_reuses": pred_c, "pred_d_interaction_is_secondary": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e else "invalid")
    result = {"schema": "aspectual_anchor_mlp4_h1h4_bank_routing_local_value_factorial_result_v1", "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "prior_art_sha256": EXPECTED_PRIOR_SHA256, "authority_sha256": EXPECTED, "scope_boundary": "Internal factor reuse only; the earlier total-mediation null remains immutable.", "capability_cells": capability_cells, "instrument": instrument, "writer_summary": writer_summary, "summaries": summaries, "factor_shapley": shapley, "local_value_retained_fraction": local_retained, "routing_plus_local_retained_fraction": additive_retained, "predictions": predictions, "price": price, "terminal": terminal, "reason": {"screen": "routing_local_value_factor_structure_reuses_across_tasks", "null": "internal_factor_reuse_or_interaction_prediction_misses", "invalid": "authority_capability_capture_closure_recurrence_coverage_or_price_invalid"}[terminal], "serial_seconds": time.perf_counter() - started, "next_action": "update the typed shared graph with the reused local-L9-value computation" if terminal == "screen" else "retain task-specific internal path computations"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "scope_boundary", "instrument", "writer_summary", "summaries", "factor_shapley", "local_value_retained_fraction", "routing_plus_local_retained_fraction", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
