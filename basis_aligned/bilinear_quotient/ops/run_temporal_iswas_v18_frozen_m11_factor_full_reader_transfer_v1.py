#!/usr/bin/env python3
"""Apply the frozen v17 M11 factor order to the full v18 reader delta."""

# BQGATE: EXPERIMENT pred_a_authority_formula_replay_hook_coverage_finiteness_and_exact_price pred_b_frozen_top32_factors_explain_full_M11_on_both_new_constructions pred_c_frozen_top128_factors_explain_most_full_M11_on_both_new_constructions pred_d_frozen_factor_program_remains_selective_on_P_and_C pred_e_no_U8_refit_reorder_or_v18_selection
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v18 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
import module_reader_loss_rescue as intervention
import run_temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1 as parent
import run_temporal_iswas_v18_frozen_shared_tensor_transfer_v1 as transfer


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_v1.json"
TRANSFER_RESULT = ROOT / "circuits/followups/temporal_iswas_v18_frozen_shared_tensor_transfer_v1_result.json"
TENSOR_RESULT = ROOT / "circuits/followups/temporal_iswas_v17_h3_m11_shared_basis_restricted_weight_tensor_v1_result.json"
CAPABILITY_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v18_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v18.py"
TRANSFER_RUNNER = ROOT / "ops/run_temporal_iswas_v18_frozen_shared_tensor_transfer_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v18_frozen_m11_factor_full_reader_transfer_v1"
RESULT_SCHEMA = "temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_result_v1"
EXPECTED = {
    "prior": "ad803dae9bd31e8ff0495e935aab515cec1173606fc52e2efc87072f998e600f",
    "transfer_result": "d26dca30ef08287587aca6c43cd5c1161c5f6d21ee0a4e9dcff22e68311276dc",
    "tensor_result": "f267b3ebbe151077f0aa44939e6f01e78fc8ae30e85d9b52e0858383ef893972",
    "capability_result": "e2f5a4368303a867646e6df7f67e3c64e98a3b3eb9a742ea7714ab413678020b",
    "builder": "44a51b3370bcf59f9cb5b17e5d63ac1cfdba3e883072112a49ec2f92ba78ae2a",
    "transfer_runner": "04cde7cd9929222835a6227dc14ffa4ccd9db6ef32b9efb4ce532acb791f9f65",
}
PANELS, TARGETS, CONTROLS = ("A1", "A2", "P", "C"), ("A1", "A2"), ("P", "C")
PREFIXES = (1, 2, 4, 8, 16, 32, 64, 128)
PRICE = {"checkpoint_loads": 1, "model_forwards": 56, "sequence_evaluations": 896,
         "scored_token_positions": 1792, "intervention_arms_per_panel": 11,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
BARS = {"replay": 1e-5, "formula": 1e-5, "top32_recovery": .70,
        "top32_cosine": .95, "top128_recovery": .80, "top128_cosine": .95,
        "output_relative_squared": 1e-8, "direction": .875, "control_ratio": .25}
PREDICTION_KEYS = (
    "pred_a_authority_formula_replay_hook_coverage_finiteness_and_exact_price",
    "pred_b_frozen_top32_factors_explain_full_M11_on_both_new_constructions",
    "pred_c_frozen_top128_factors_explain_most_full_M11_on_both_new_constructions",
    "pred_d_frozen_factor_program_remains_selective_on_P_and_C",
    "pred_e_no_U8_refit_reorder_or_v18_selection",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def capture_m11(forward, model):
    return intervention.capture_reader_outputs(forward, {"M11": model.transformer.h[11].mlp})


def calls_ok(calls):
    source_ok = all(set(calls[key].values()) == {1}
                    for key in ("native_source", "donor_source", "writer_source"))
    capture_ok = all(calls[key] == {"M11": {"reader": 1, "output": 1}}
                     for key in ("native_sites", "writer_sites"))
    replay_ok = (calls["writer_replay"]["readers"] == {}
        and calls["writer_replay"]["output"] == 0
        and set(calls["writer_replay"]["source"].values()) == {1})
    absent_ok = (calls["M11_absent"]["readers"] == {"M11": 1}
        and calls["M11_absent"]["output"] == 0
        and set(calls["M11_absent"]["source"].values()) == {1})
    output_ok = all(calls[key]["readers"] == {} and calls[key]["output"] == 1
        and set(calls[key]["source"].values()) == {1}
        for key in [f"top{count}" for count in PREFIXES] + ["all_hidden_formula"])
    return bool(source_ok and capture_ok and replay_ok and absent_ok and output_ok)


def panel_run(backend, rows, factor_order):
    torch, model = backend.torch, backend.model
    batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    positions = tuple(tuple(range(int(query) + 1)) for query in batch.semantic_positions)
    base_source, donor_source = transfer.suffix_position_rows(batch, donor_batch)
    calls = {}
    native_bundle, base_heads, calls["native_source"] = parent.capture_source_heads(
        lambda: capture_m11(lambda: parent.full_forward(backend, batch), model), model, batch)
    native, absent_readers, absent_outputs, calls["native_sites"] = native_bundle
    donor, donor_heads, calls["donor_source"] = parent.capture_source_heads(
        lambda: parent.full_forward(backend, donor_batch), model, donor_batch)
    writer_bundle, calls["writer_source"] = transfer.with_source_heads_aligned(
        lambda: capture_m11(lambda: parent.full_forward(backend, batch), model),
        model, batch, base_heads, donor_heads, base_source, donor_source)
    writer, present_readers, present_outputs, calls["writer_sites"] = writer_bundle
    outputs = {}
    outputs["writer_replay"], calls["writer_replay"] = transfer.run_tensor_arm(
        backend, batch, base_heads, donor_heads, base_source, donor_source, positions, {})
    outputs["M11_absent"], calls["M11_absent"] = transfer.run_tensor_arm(
        backend, batch, base_heads, donor_heads, base_source, donor_source, positions,
        {"M11": absent_readers["M11"]})

    mlp = model.transformer.h[11].mlp
    x0, x1 = absent_readers["M11"].float(), present_readers["M11"].float()
    hidden_delta = ((x1 @ mlp.Left.weight.detach().float().T)
                    * (x1 @ mlp.Right.weight.detach().float().T)
                    - (x0 @ mlp.Left.weight.detach().float().T)
                    * (x0 @ mlp.Right.weight.detach().float().T))
    down = mlp.Down.weight.detach().float()
    for count in PREFIXES:
        chosen = factor_order[:count]
        output_replacement = absent_outputs["M11"] + (
            hidden_delta[..., chosen] @ down[:, chosen].T).to(absent_outputs["M11"])
        label = f"top{count}"
        outputs[label], calls[label] = transfer.run_tensor_arm(
            backend, batch, base_heads, donor_heads, base_source, donor_source, positions,
            {}, output_replacement=output_replacement)
    full_delta = hidden_delta @ down.T
    all_output = absent_outputs["M11"] + full_delta.to(absent_outputs["M11"])
    outputs["all_hidden_formula"], calls["all_hidden_formula"] = transfer.run_tensor_arm(
        backend, batch, base_heads, donor_heads, base_source, donor_source, positions,
        {}, output_replacement=all_output)

    writer_margin = parent.toward_donor_margin(writer)
    margins = {label: parent.toward_donor_margin(output) for label, output in outputs.items()}
    reference = writer_margin - margins["M11_absent"]
    effects = {f"top{count}": margins[f"top{count}"] - margins["M11_absent"]
               for count in PREFIXES}
    reports = {label: parent.metrics(effect, reference) for label, effect in effects.items()}
    reports["native_accuracy"] = float(np.mean([a > f for a, f in native.answer_foil]))
    reports["donor_accuracy"] = float(np.mean([a > f for a, f in donor.answer_foil]))
    output_errors = []
    for row, row_positions in enumerate(positions):
        if row_positions:
            indices = list(row_positions)
            output_errors.append(float((all_output[row, indices].float()
                - present_outputs["M11"][row, indices].float()).abs().max()))
    instrument = {
        "writer_replay_max_abs_margin_error": float(np.max(np.abs(
            margins["writer_replay"] - writer_margin))),
        "all_hidden_output_max_abs_error": max(output_errors, default=0.0),
        "all_hidden_output_relative_squared_error": float(sum(
            (all_output[row, list(row_positions)].float()
             - present_outputs["M11"][row, list(row_positions)].float()).square().sum().item()
            for row, row_positions in enumerate(positions) if row_positions) / max(sum(
            present_outputs["M11"][row, list(row_positions)].float().square().sum().item()
            for row, row_positions in enumerate(positions) if row_positions), 1e-30)),
        "all_hidden_margin_max_abs_error": float(np.max(np.abs(
            margins["all_hidden_formula"] - writer_margin))),
        "calls": calls,
    }
    return reports, effects, instrument


def main():
    paths = {"prior": PRIOR, "transfer_result": TRANSFER_RESULT,
             "tensor_result": TENSOR_RESULT, "capability_result": CAPABILITY_RESULT,
             "builder": BUILDER, "transfer_runner": TRANSFER_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    transfer_result, tensor_result = json.loads(TRANSFER_RESULT.read_text()), json.loads(TENSOR_RESULT.read_text())
    capability = json.loads(CAPABILITY_RESULT.read_text())
    all_rows = fresh.build_rows()
    row_sets = {family: [row for row in all_rows if row["transform_id"] == family and (
        family in CONTROLS or row["row_id"] in set(capability["jointly_capable_row_ids"][family]))]
        for family in PANELS}
    authority_ok = bool(observed == EXPECTED
        and transfer_result.get("terminal") == "activation_specific_interface"
        and tensor_result.get("terminal") == "compact_shared_weight_tensor"
        and all(len(row_sets[family]) == 16 for family in PANELS))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "rows": {key: len(value) for key, value in row_sets.items()},
           "prefixes": PREFIXES, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("v18 full-reader factor authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")

    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend, torch = producer.Bilin18TorchBackend.load("cuda"), __import__("torch")
    stored_order = tuple(int(index) for index in tensor_result["M11_factor_order"])
    if len(stored_order) < max(PREFIXES) or len(set(stored_order[:max(PREFIXES)])) != max(PREFIXES):
        raise RuntimeError("frozen factor order changed")
    factor_order = torch.tensor(stored_order, dtype=torch.long, device=backend.device)
    reports, effects, instrument = {}, {}, {}
    with torch.no_grad():
        for family in PANELS:
            reports[family], panel_effects, instrument[family] = panel_run(
                backend, row_sets[family], factor_order)
            effects[family] = {key: value.tolist() for key, value in panel_effects.items()}
    target_rms = {label: transfer.rms(np.concatenate([
        np.asarray(effects[family][label], dtype=np.float64) for family in TARGETS]))
        for label in ("top32", "top128")}
    control_ratios = {family: {label: transfer.rms(effects[family][label])
        / max(target_rms[label], 1e-30) for label in target_rms} for family in CONTROLS}

    A = bool(authority_ok and all(calls_ok(record["calls"]) for record in instrument.values())
        and all(record["writer_replay_max_abs_margin_error"] <= BARS["replay"]
                and record["all_hidden_output_relative_squared_error"] <= BARS["output_relative_squared"]
                and record["all_hidden_margin_max_abs_error"] <= BARS["formula"]
                for record in instrument.values())
        and finite({"reports": reports, "effects": effects, "controls": control_ratios})
        and PRICE["model_forwards"] == len(PANELS) * (3 + PRICE["intervention_arms_per_panel"])
        and PRICE["model_forwards"] * 16 == PRICE["sequence_evaluations"]
        and PRICE["sequence_evaluations"] * 2 == PRICE["scored_token_positions"])
    B = all(reports[family]["top32"]["signed_recovery"] >= BARS["top32_recovery"]
        and reports[family]["top32"]["cosine"] >= BARS["top32_cosine"]
        and reports[family]["top32"]["direction_agreement"] >= BARS["direction"]
        for family in TARGETS)
    C = all(reports[family]["top128"]["signed_recovery"] >= BARS["top128_recovery"]
        and reports[family]["top128"]["cosine"] >= BARS["top128_cosine"]
        and reports[family]["top128"]["direction_agreement"] >= BARS["direction"]
        for family in TARGETS)
    D = all(ratio <= BARS["control_ratio"] for family in CONTROLS
            for ratio in control_ratios[family].values())
    E = True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A else "nonselective_factor_program" if not D else
        "compact_construction_general_factor_program" if B else
        "broad_construction_general_factor_program" if C else
        "factor_hierarchy_activation_specific")
    result = {"schema": RESULT_SCHEMA,
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "frozen_factor_prefixes": {str(count): stored_order[:count] for count in PREFIXES},
        "v18_fit_updates": 0, "v18_selected_parameters": 0,
        "instrument": instrument, "reports": reports, "effect_vectors": effects,
        "control_normalized_rms_ratios": control_ratios, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("instrument", "reports",
        "control_normalized_rms_ratios", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
