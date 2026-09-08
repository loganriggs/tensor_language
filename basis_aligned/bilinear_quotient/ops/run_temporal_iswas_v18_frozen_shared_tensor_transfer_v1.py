#!/usr/bin/env python3
"""No-refit v18 transfer of the literal v17 U8/H3/top-32 physical program."""

# BQGATE: EXPERIMENT pred_a_authority_frozen_program_capture_replay_roundtrip_and_exact_price pred_b_frozen_U8_reaches_and_causally_explains_both_readers_on_both_new_constructions pred_c_frozen_H3_remains_the_majority_A11_endpoint_on_both_new_constructions pred_d_frozen_top32_M11_factor_program_transfers_on_both_new_constructions pred_e_frozen_program_is_selective_on_answer_preserving_and_canonical_controls
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
import run_temporal_iswas_v17_a11_head_endpoint_ordered_cumulative_v1 as head_parent
import run_temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1 as parent
import run_temporal_iswas_v17_h3_m11_shared_basis_restricted_weight_tensor_v1 as tensor_parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v18_frozen_shared_tensor_transfer_v1.json"
TENSOR_RESULT = ROOT / "circuits/followups/temporal_iswas_v17_h3_m11_shared_basis_restricted_weight_tensor_v1_result.json"
HEAD_RESULT = ROOT / "circuits/followups/temporal_iswas_v17_a11_head_endpoint_ordered_cumulative_v1_result.json"
CAPABILITY_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v18_capability_v1_result.json"
CAPABILITY_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v18_capability_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v18.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v18_frozen_shared_tensor_transfer_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v18_frozen_shared_tensor_transfer_v1"
EXPECTED = {
    "prior": "1d28e68551fecebffbc25fe8a3b4e6cb07b531f31fb2ba8895c320f8930c0ea9",
    "tensor_result": "f267b3ebbe151077f0aa44939e6f01e78fc8ae30e85d9b52e0858383ef893972",
    "head_result": "7fa315b84f71f267c3ed4cea67231cbad9dd0ef2b5ad25a3b18133fc6a6e9c85",
    "capability_result": "e2f5a4368303a867646e6df7f67e3c64e98a3b3eb9a742ea7714ab413678020b",
    "capability_runner": "7410844c376c99b88aa8fe84f91b5afd501059e06de7948146d6b32181f8a063",
    "builder": "44a51b3370bcf59f9cb5b17e5d63ac1cfdba3e883072112a49ec2f92ba78ae2a",
}
PANELS = ("A1", "A2", "P", "C")
TARGETS, CONTROLS = ("A1", "A2"), ("P", "C")
RANK, H3, TOP = 8, 3, 32
PRICE = {"checkpoint_loads": 1, "model_forwards": 44, "sequence_evaluations": 704,
         "scored_token_positions": 1408, "intervention_arms_per_panel": 8,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
BARS = {"replay": 1e-5, "roundtrip": 0.0, "energy": .80,
        "U8_recovery": .80, "U8_cosine": .95, "H3_share": .50,
        "H3_cosine": .95, "H3_direction": .875, "top32_recovery": .70,
        "top32_cosine": .95, "control_ratio": .25}
PREDICTION_KEYS = (
    "pred_a_authority_frozen_program_capture_replay_roundtrip_and_exact_price",
    "pred_b_frozen_U8_reaches_and_causally_explains_both_readers_on_both_new_constructions",
    "pred_c_frozen_H3_remains_the_majority_A11_endpoint_on_both_new_constructions",
    "pred_d_frozen_top32_M11_factor_program_transfers_on_both_new_constructions",
    "pred_e_frozen_program_is_selective_on_answer_preserving_and_canonical_controls",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def selected_matrix(tensor, positions):
    torch = __import__("torch")
    chunks = [tensor[row, list(row_positions)].detach().float().cpu()
              for row, row_positions in enumerate(positions) if row_positions]
    return torch.cat(chunks, dim=0)


def energy_fraction(delta, positions, basis_cpu):
    matrix = selected_matrix(delta, positions)
    projected = (matrix @ basis_cpu) @ basis_cpu.T
    return float(projected.square().sum()) / max(float(matrix.square().sum()), 1e-30)


def rms(value):
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value)))


def panel_calls_ok(calls):
    source_ok = all(set(calls[key].values()) == {1}
                    for key in ("native_source", "donor_source", "writer_source"))
    capture_ok = (calls["native_a11"] == {"input": 1, "output": 1}
        and calls["writer_a11"] == {"input": 1, "output": 1}
        and all(record == {"reader": 1, "output": 1}
                for key in ("native_sites", "writer_sites")
                for record in calls[key].values()))
    tensor_ok = all(
        set(calls[key]["source"].values()) == {1}
        and (not calls[key]["readers"] or set(calls[key]["readers"].values()) == {1})
        and calls[key]["output"] == int(key == "M11_U8_top32")
        for key in ("writer_replay", "A11_absent", "A11_U8", "M11_absent",
                    "M11_U8", "M11_U8_top32"))
    head_ok = all(calls[key]["a11"] == 1 and set(calls[key]["source"].values()) == {1}
                  for key in ("H3_removed", "all_A11_heads_removed"))
    return bool(source_ok and capture_ok and tensor_ok and head_ok)


def capture_all(forward, model):
    return head_parent.capture_a11(
        lambda: tensor_parent.capture_sites(forward, model), model)


def suffix_position_rows(base_batch, donor_batch):
    base_rows, donor_rows = [], []
    for base_ids, donor_ids, base_query, donor_query in zip(
        base_batch.token_rows, donor_batch.token_rows,
        base_batch.semantic_positions, donor_batch.semantic_positions
    ):
        base_stop, donor_stop = int(base_query) + 1, int(donor_query) + 1
        common = 0
        while (common < base_stop and common < donor_stop
               and base_ids[base_stop - common - 1] == donor_ids[donor_stop - common - 1]):
            common += 1
        if common < 1:
            raise RuntimeError("paired rows lost their matched final suffix")
        base_positions = tuple(range(base_stop - common, base_stop))
        donor_positions = tuple(range(donor_stop - common, donor_stop))
        if [base_ids[index] for index in base_positions] != [
                donor_ids[index] for index in donor_positions]:
            raise RuntimeError("suffix alignment changed token identity")
        base_rows.append(base_positions)
        donor_rows.append(donor_positions)
    return tuple(base_rows), tuple(donor_rows)


def with_source_heads_aligned(forward, model, batch, base, donor,
                              base_positions, donor_positions):
    handles, calls = [], {layer: 0 for layer, _head in parent.SOURCE_HEADS}
    heads, width = model.config.n_head, model.config.n_embd // model.config.n_head
    for layer, head in parent.SOURCE_HEADS:
        def patch(_module, arguments, layer=layer, head=head):
            calls[layer] += 1
            value = arguments[0]
            changed = value.clone().view(len(batch.row_ids), value.shape[1], heads, width)
            for row, (base_row, donor_row) in enumerate(zip(base_positions, donor_positions)):
                if base_row:
                    changed[row, list(base_row), head] += (
                        donor[layer][row, list(donor_row), head]
                        - base[layer][row, list(base_row), head]
                    ).to(changed)
            return (changed.reshape_as(value),) + tuple(arguments[1:])
        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    try:
        output = forward()
    finally:
        for handle in handles:
            handle.remove()
    if set(calls.values()) != {1}:
        raise RuntimeError(f"aligned source-head patch coverage changed: {calls}")
    return output, calls


def run_tensor_arm(backend, batch, base_heads, donor_heads, base_source_positions,
                   donor_source_positions, positions, replacements, *, output_replacement=None):
    calls, output_calls, handles = {label: 0 for label in replacements}, 0, []
    targets = {"A11": backend.model.transformer.h[11].attn,
               "M11": backend.model.transformer.h[11].mlp}
    for label in replacements:
        def pre(_module, arguments, label=label):
            calls[label] += 1
            changed = intervention.replace_positions(arguments[0], replacements[label], positions)
            return (changed,) + tuple(arguments[1:])
        handles.append(targets[label].register_forward_pre_hook(pre))
    if output_replacement is not None:
        def post(_module, _arguments, output):
            nonlocal output_calls
            output_calls += 1
            return intervention.replace_positions(output, output_replacement, positions)
        handles.append(targets["M11"].register_forward_hook(post))
    try:
        output, source_calls = with_source_heads_aligned(
            lambda: parent.full_forward(backend, batch), backend.model, batch,
            base_heads, donor_heads, base_source_positions, donor_source_positions)
    finally:
        for handle in handles:
            handle.remove()
    if (calls and set(calls.values()) != {1}) or output_calls != int(output_replacement is not None):
        raise RuntimeError(f"aligned tensor arm coverage changed: {calls}/{output_calls}")
    return output, {"readers": calls, "output": output_calls, "source": source_calls}


def run_head_arm(backend, batch, base_heads, donor_heads, base_source_positions,
                 donor_source_positions, absent_a11, positions, removed):
    calls, saved_output = 0, {}
    def a11_pre(_module, arguments):
        nonlocal calls
        calls += 1
        changed = head_parent.replace_removed_heads(arguments[0], absent_a11, removed, positions)
        return (changed,) + tuple(arguments[1:])
    def a11_post(_module, _arguments, output):
        saved_output["value"] = output.detach().clone()
    pre_handle = backend.model.transformer.h[11].attn.c_proj.register_forward_pre_hook(a11_pre)
    post_handle = backend.model.transformer.h[11].attn.c_proj.register_forward_hook(a11_post)
    try:
        output, source_calls = with_source_heads_aligned(
            lambda: parent.full_forward(backend, batch), backend.model, batch,
            base_heads, donor_heads, base_source_positions, donor_source_positions)
    finally:
        pre_handle.remove(); post_handle.remove()
    if calls != 1 or set(saved_output) != {"value"}:
        raise RuntimeError("aligned A11 head arm coverage changed")
    return output, {"a11": calls, "source": source_calls}, saved_output["value"]


def panel_run(backend, rows, basis, basis_cpu, top):
    torch, model = backend.torch, backend.model
    batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    positions = tuple(tuple(range(int(query) + 1)) for query in batch.semantic_positions)
    base_source_positions, donor_source_positions = suffix_position_rows(batch, donor_batch)
    calls = {}

    native_bundle, base_heads, calls["native_source"] = parent.capture_source_heads(
        lambda: capture_all(lambda: parent.full_forward(backend, batch), model), model, batch)
    (native_sites, native_a11, calls["native_a11"]), = (native_bundle,)
    native, absent_readers, absent_outputs, calls["native_sites"] = native_sites
    donor, donor_heads, calls["donor_source"] = parent.capture_source_heads(
        lambda: parent.full_forward(backend, donor_batch), model, donor_batch)
    writer_bundle, calls["writer_source"] = with_source_heads_aligned(
        lambda: capture_all(lambda: parent.full_forward(backend, batch), model),
        model, batch, base_heads, donor_heads, base_source_positions, donor_source_positions)
    writer_sites, writer_a11, calls["writer_a11"] = writer_bundle
    writer, present_readers, present_outputs, calls["writer_sites"] = writer_sites

    projected = {label: tensor_parent.project_reader(
        absent_readers[label], present_readers[label], basis)
        for label in ("A11", "M11")}
    arm_specs = {
        "writer_replay": {},
        "A11_absent": {"A11": absent_readers["A11"]},
        "A11_U8": {"A11": projected["A11"]},
        "M11_absent": {"M11": absent_readers["M11"]},
        "M11_U8": {"M11": projected["M11"]},
    }
    outputs = {}
    for label, replacements in arm_specs.items():
        outputs[label], calls[label] = run_tensor_arm(
            backend, batch, base_heads, donor_heads, base_source_positions,
            donor_source_positions, positions, replacements)

    mlp = model.transformer.h[11].mlp
    x0, xu = absent_readers["M11"].float(), projected["M11"].float()
    hidden_delta = ((xu @ mlp.Left.weight.detach().float().T)
                    * (xu @ mlp.Right.weight.detach().float().T)
                    - (x0 @ mlp.Left.weight.detach().float().T)
                    * (x0 @ mlp.Right.weight.detach().float().T))
    top_delta = hidden_delta[..., top] @ mlp.Down.weight.detach().float()[:, top].T
    top_output = absent_outputs["M11"] + top_delta.to(absent_outputs["M11"])
    outputs["M11_U8_top32"], calls["M11_U8_top32"] = run_tensor_arm(
        backend, batch, base_heads, donor_heads, base_source_positions,
        donor_source_positions, positions,
        {"M11": projected["M11"]}, output_replacement=top_output)
    outputs["H3_removed"], calls["H3_removed"], _ = run_head_arm(
        backend, batch, base_heads, donor_heads, base_source_positions,
        donor_source_positions,
        native_a11["input"], positions, (H3,))
    outputs["all_A11_heads_removed"], calls["all_A11_heads_removed"], all_head_output = (
        run_head_arm(backend, batch, base_heads, donor_heads, base_source_positions,
                     donor_source_positions, native_a11["input"], positions,
                     head_parent.HEADS))

    native_margin, writer_margin = parent.toward_donor_margin(native), parent.toward_donor_margin(writer)
    margins = {label: parent.toward_donor_margin(output) for label, output in outputs.items()}
    effects = {
        "source": writer_margin - native_margin,
        "A11_full": writer_margin - margins["A11_absent"],
        "A11_U8": margins["A11_U8"] - margins["A11_absent"],
        "M11_full": writer_margin - margins["M11_absent"],
        "M11_U8": margins["M11_U8"] - margins["M11_absent"],
        "H3": writer_margin - margins["H3_removed"],
        "A11_all_heads": writer_margin - margins["all_A11_heads_removed"],
        "M11_top32": margins["M11_U8_top32"] - margins["M11_absent"],
    }
    reports = {
        "reachability": {label: energy_fraction(
            present_readers[label] - absent_readers[label], positions, basis_cpu)
            for label in ("A11", "M11")},
        "U8": {label: parent.metrics(effects[f"{label}_U8"], effects[f"{label}_full"])
               for label in ("A11", "M11")},
        "H3": parent.metrics(effects["H3"], effects["A11_all_heads"]),
        "top32": parent.metrics(effects["M11_top32"], effects["M11_U8"]),
        "native_accuracy": float(np.mean([answer > foil for answer, foil in native.answer_foil])),
        "donor_accuracy": float(np.mean([answer > foil for answer, foil in donor.answer_foil])),
    }
    output_errors = []
    for row, row_positions in enumerate(positions):
        if row_positions:
            indices = list(row_positions)
            output_errors.append(float((all_head_output[row, indices].float()
                - native_a11["output"][row, indices].float()).abs().max()))
    instrument = {
        "writer_replay_max_abs_margin_error": float(np.max(np.abs(
            margins["writer_replay"] - writer_margin))),
        "all_head_output_max_abs_error": max(output_errors, default=0.0),
        "calls": calls,
    }
    return reports, effects, instrument


def main():
    paths = {"prior": PRIOR, "tensor_result": TENSOR_RESULT, "head_result": HEAD_RESULT,
             "capability_result": CAPABILITY_RESULT, "capability_runner": CAPABILITY_RUNNER,
             "builder": BUILDER}
    observed = {name: sha(path) for name, path in paths.items()}
    tensor_result = json.loads(TENSOR_RESULT.read_text())
    head_result = json.loads(HEAD_RESULT.read_text())
    capability = json.loads(CAPABILITY_RESULT.read_text())
    all_rows = fresh.build_rows()
    row_sets = {
        family: [row for row in all_rows if row["transform_id"] == family and (
            family in CONTROLS or row["row_id"] in set(capability["jointly_capable_row_ids"][family]))]
        for family in PANELS
    }
    authority_ok = bool(observed == EXPECTED
        and tensor_result.get("terminal") == "compact_shared_weight_tensor"
        and head_result.get("terminal") == "h3_compact_A11_endpoint"
        and capability.get("terminal") == "screen"
        and capability.get("causal_outcomes_opened") is False
        and all(len(row_sets[family]) == 16 for family in PANELS))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "rows": {key: len(value) for key, value in row_sets.items()},
           "basis_rank": RANK, "head": H3, "top_factors": TOP, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok:
        raise RuntimeError("v18 frozen tensor transfer authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    basis_record = tensor_result["basis"]
    basis_cpu = torch.tensor(basis_record["values"], dtype=torch.float32).reshape(
        basis_record["shape"])
    basis_hash = hashlib.sha256(basis_cpu.contiguous().numpy().tobytes()).hexdigest()
    basis_roundtrip_error = float((basis_cpu - torch.tensor(
        basis_record["values"], dtype=torch.float32).reshape(basis_record["shape"])).abs().max())
    if basis_cpu.shape != (1152, RANK) or basis_hash != basis_record["sha256"]:
        raise RuntimeError("frozen v17 basis did not restore exactly")
    factor_order = tuple(int(index) for index in tensor_result["M11_factor_order"])
    if len(factor_order) < TOP or len(set(factor_order[:TOP])) != TOP:
        raise RuntimeError("frozen v17 factor order changed")
    top = torch.tensor(factor_order[:TOP], dtype=torch.long, device=backend.device)
    basis = basis_cpu.to(backend.device)

    reports, effect_vectors, instrument = {}, {}, {}
    with torch.no_grad():
        for family in PANELS:
            reports[family], effects, instrument[family] = panel_run(
                backend, row_sets[family], basis, basis_cpu, top)
            effect_vectors[family] = {key: value.tolist() for key, value in effects.items()}

    target_effects = {
        key: np.concatenate([np.asarray(effect_vectors[family][key], dtype=np.float64)
                             for family in TARGETS])
        for key in ("A11_U8", "M11_U8", "H3", "M11_top32")
    }
    control_ratios = {family: {
        key: rms(effect_vectors[family][key]) / max(rms(target_effects[key]), 1e-30)
        for key in target_effects} for family in CONTROLS}

    hook_ok = all(panel_calls_ok(record["calls"]) for record in instrument.values())
    A = bool(authority_ok and basis_roundtrip_error == BARS["roundtrip"]
        and basis_hash == basis_record["sha256"] and hook_ok
        and all(record["writer_replay_max_abs_margin_error"] <= BARS["replay"]
                for record in instrument.values())
        and finite({"reports": reports, "control_ratios": control_ratios,
                    "instrument": instrument, "effects": effect_vectors})
        and PRICE["model_forwards"] == len(PANELS) * (3 + PRICE["intervention_arms_per_panel"])
        and PRICE["model_forwards"] * 16 == PRICE["sequence_evaluations"]
        and PRICE["sequence_evaluations"] * 2 == PRICE["scored_token_positions"])
    B = all(reports[family]["reachability"][label] >= BARS["energy"]
        and reports[family]["U8"][label]["signed_recovery"] >= BARS["U8_recovery"]
        and reports[family]["U8"][label]["cosine"] >= BARS["U8_cosine"]
        for family in TARGETS for label in ("A11", "M11"))
    C = all(reports[family]["H3"]["signed_recovery"] >= BARS["H3_share"]
        and reports[family]["H3"]["cosine"] >= BARS["H3_cosine"]
        and reports[family]["H3"]["direction_agreement"] >= BARS["H3_direction"]
        for family in TARGETS)
    D = all(reports[family]["top32"]["signed_recovery"] >= BARS["top32_recovery"]
        and reports[family]["top32"]["cosine"] >= BARS["top32_cosine"]
        for family in TARGETS)
    E = all(ratio <= BARS["control_ratio"]
            for family in CONTROLS for ratio in control_ratios[family].values())
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A else
        "activation_specific_interface" if not B else
        "native_boundary_specific" if not C else
        "factor_activation_specific" if not D else
        "nonselective_shared_program" if not E else
        "construction_general_selective_shared_weight_program")
    result = {
        "schema": "temporal_iswas_v18_frozen_shared_tensor_transfer_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "frozen_program": {"basis_sha256": basis_hash, "basis_shape": list(basis_cpu.shape),
                           "H3": H3, "M11_top32": list(factor_order[:TOP]),
                           "v18_fit_updates": 0, "v18_selected_parameters": 0},
        "basis_roundtrip_max_abs_error": basis_roundtrip_error,
        "instrument": instrument, "reports": reports,
        "control_normalized_rms_ratios": control_ratios,
        "effect_vectors": effect_vectors, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "frozen_program", "instrument", "reports", "control_normalized_rms_ratios",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
