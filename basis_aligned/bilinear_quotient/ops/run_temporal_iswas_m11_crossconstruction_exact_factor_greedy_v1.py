#!/usr/bin/env python3
"""Exact fixed-coefficient greedy M11 factors across constructions."""

# BQGATE: EXPERIMENT pred_a_authority_exact_gram_greedy_formula_replays_hooks_and_price pred_b_crossconstruction_greedy_top128_is_compact_on_training_outputs pred_c_frozen_top128_transfers_to_v18_A2_output_and_causality pred_d_frozen_top256_recovers_most_v18_A2_causality pred_e_greedy_factor_program_is_selective_and_v19_remains_sealed
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
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17 as fresh17
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v18 as fresh18
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
import module_reader_loss_rescue as intervention
import run_temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1 as parent
import run_temporal_iswas_v18_frozen_shared_tensor_transfer_v1 as transfer


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_m11_crossconstruction_exact_factor_greedy_v1.json"
FACTOR_RESULT = ROOT / "circuits/followups/temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_v2_result.json"
TRANSFER_RESULT = ROOT / "circuits/followups/temporal_iswas_v18_frozen_shared_tensor_transfer_v1_result.json"
TENSOR_RESULT = ROOT / "circuits/followups/temporal_iswas_v17_h3_m11_shared_basis_restricted_weight_tensor_v1_result.json"
V19_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v19_capability_v1_result.json"
BUILDER17 = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17.py"
BUILDER18 = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v18.py"
OUT = ROOT / "circuits/followups/temporal_iswas_m11_crossconstruction_exact_factor_greedy_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.m11_crossconstruction_exact_factor_greedy_v1"
EXPECTED = {
    "prior": "17bdbb80d4117eb35f39ee9d626d3d895739450bf314a6e3df974768accbe65c",
    "factor_result": "f558c21f40bfb961e7abc3ecc84dd52e4153e3f7ec0e99a390594c487030a82d",
    "transfer_result": "d26dca30ef08287587aca6c43cd5c1161c5f6d21ee0a4e9dcff22e68311276dc",
    "tensor_result": "f267b3ebbe151077f0aa44939e6f01e78fc8ae30e85d9b52e0858383ef893972",
    "v19_capability": "28f8f134d78279dd82cd7cca94f76e9cf35815bd1a59430ad34601fd0294fdd4",
    "builder17": "7e59800324e5b6a4a9c5af564cbe8fb5cf642cfd6e702e1e97c32bc0cc54fe9e",
    "builder18": "44a51b3370bcf59f9cb5b17e5d63ac1cfdba3e883072112a49ec2f92ba78ae2a",
}
PREFIXES = (16, 32, 64, 128, 256)
PRICE = {"checkpoint_loads": 1, "model_forwards": 39, "sequence_evaluations": 624,
         "scored_token_positions": 1248, "intervention_arms_per_test_panel": 8,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0,
         "selected_native_factor_indices": 256}
BARS = {"replay": 1e-5, "formula": 1e-5, "output_rse": 1e-8,
        "gram_direct": 1e-5, "train_residual": .45, "heldout_residual": .45,
        "top128_recovery": .80, "top128_cosine": .95,
        "top256_recovery": .85, "top256_cosine": .95,
        "direction": .875, "control_ratio": .25}
PREDICTION_KEYS = (
    "pred_a_authority_exact_gram_greedy_formula_replays_hooks_and_price",
    "pred_b_crossconstruction_greedy_top128_is_compact_on_training_outputs",
    "pred_c_frozen_top128_transfers_to_v18_A2_output_and_causality",
    "pred_d_frozen_top256_recovers_most_v18_A2_causality",
    "pred_e_greedy_factor_program_is_selective_and_v19_remains_sealed",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def capture_m11(forward, model):
    return intervention.capture_reader_outputs(forward, {"M11": model.transformer.h[11].mlp})


def selected_matrix(tensor, positions):
    torch = __import__("torch")
    return torch.cat([tensor[row, list(row_positions)].detach().float()
                      for row, row_positions in enumerate(positions) if row_positions], dim=0)


def capture_delta(backend, rows):
    model = backend.model
    batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    positions = tuple(tuple(range(int(query) + 1)) for query in batch.semantic_positions)
    base_source, donor_source = transfer.suffix_position_rows(batch, donor_batch)
    calls = {}
    native_bundle, base_heads, calls["native_source"] = parent.capture_source_heads(
        lambda: capture_m11(lambda: parent.full_forward(backend, batch), model), model, batch)
    _native, absent_readers, absent_outputs, calls["native_sites"] = native_bundle
    _donor, donor_heads, calls["donor_source"] = parent.capture_source_heads(
        lambda: parent.full_forward(backend, donor_batch), model, donor_batch)
    writer_bundle, calls["writer_source"] = transfer.with_source_heads_aligned(
        lambda: capture_m11(lambda: parent.full_forward(backend, batch), model),
        model, batch, base_heads, donor_heads, base_source, donor_source)
    _writer, present_readers, present_outputs, calls["writer_sites"] = writer_bundle
    mlp = model.transformer.h[11].mlp
    x0, x1 = absent_readers["M11"].float(), present_readers["M11"].float()
    hidden_delta = ((x1 @ mlp.Left.weight.detach().float().T)
                    * (x1 @ mlp.Right.weight.detach().float().T)
                    - (x0 @ mlp.Left.weight.detach().float().T)
                    * (x0 @ mlp.Right.weight.detach().float().T))
    return (selected_matrix(hidden_delta, positions),
            selected_matrix(present_outputs["M11"] - absent_outputs["M11"], positions), calls)


def factor_gram(hidden, down):
    gram = (hidden.T @ hidden) * (down.T @ down)
    total = float(gram.sum())
    if not math.isfinite(total) or total <= 0: raise RuntimeError("factor Gram lost positive total")
    return gram / total


def exact_greedy(grams, count=256):
    torch = __import__("torch")
    pooled = sum(grams)
    residual_corr = pooled.sum(dim=1).clone()
    diagonal = torch.diagonal(pooled)
    available = torch.ones(pooled.shape[0], dtype=torch.bool, device=pooled.device)
    order = []
    for _step in range(count):
        gain = 2.0 * residual_corr - diagonal
        gain[~available] = -torch.inf
        chosen = int(torch.argmax(gain))
        order.append(chosen); available[chosen] = False
        residual_corr -= pooled[:, chosen]
    return tuple(order)


def residual_from_gram(gram, selected):
    torch = __import__("torch")
    mask = torch.ones(gram.shape[0], dtype=gram.dtype, device=gram.device)
    mask[list(selected)] = 0
    return math.sqrt(max(float(mask @ gram @ mask), 0.0))


def direct_residual(hidden, output, down, selected):
    approximation = hidden[:, list(selected)] @ down[:, list(selected)].T
    return float(__import__("torch").linalg.vector_norm(output - approximation)
                 / max(float(__import__("torch").linalg.vector_norm(output)), 1e-30))


def test_panel(backend, rows, order):
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
    prefix_outputs = {}
    for count in PREFIXES:
        chosen = order[:count]
        replacement = absent_outputs["M11"] + (
            hidden_delta[..., chosen] @ down[:, chosen].T).to(absent_outputs["M11"])
        label = f"top{count}"; prefix_outputs[count] = replacement
        outputs[label], calls[label] = transfer.run_tensor_arm(
            backend, batch, base_heads, donor_heads, base_source, donor_source, positions,
            {}, output_replacement=replacement)
    all_output = absent_outputs["M11"] + (hidden_delta @ down.T).to(absent_outputs["M11"])
    outputs["all_hidden_formula"], calls["all_hidden_formula"] = transfer.run_tensor_arm(
        backend, batch, base_heads, donor_heads, base_source, donor_source, positions,
        {}, output_replacement=all_output)
    writer_margin = parent.toward_donor_margin(writer)
    margins = {label: parent.toward_donor_margin(output) for label, output in outputs.items()}
    reference = writer_margin - margins["M11_absent"]
    effects = {f"top{count}": margins[f"top{count}"] - margins["M11_absent"]
               for count in PREFIXES}
    full_output = selected_matrix(present_outputs["M11"] - absent_outputs["M11"], positions)
    reports = {f"top{count}": {**parent.metrics(effects[f"top{count}"], reference),
        "output_relative_residual": float(torch.linalg.vector_norm(
            full_output - selected_matrix(prefix_outputs[count] - absent_outputs["M11"], positions))
            / max(float(torch.linalg.vector_norm(full_output)), 1e-30))} for count in PREFIXES}
    reports["native_accuracy"] = float(np.mean([a > f for a, f in native.answer_foil]))
    reports["donor_accuracy"] = float(np.mean([a > f for a, f in donor.answer_foil]))
    diff = selected_matrix(all_output - present_outputs["M11"], positions)
    instrument = {"writer_replay_max_abs_margin_error": float(np.max(np.abs(
        margins["writer_replay"] - writer_margin))),
        "all_hidden_margin_max_abs_error": float(np.max(np.abs(
            margins["all_hidden_formula"] - writer_margin))),
        "all_hidden_output_relative_squared_error": float(diff.square().sum()
            / max(float(selected_matrix(present_outputs["M11"], positions).square().sum()), 1e-30)),
        "calls": calls}
    return reports, effects, instrument


def calls_ok(calls, training=False):
    if not all(set(calls[key].values()) == {1}
               for key in ("native_source", "donor_source", "writer_source")): return False
    if not all(calls[key] == {"M11": {"reader": 1, "output": 1}}
               for key in ("native_sites", "writer_sites")): return False
    if training: return True
    if calls["M11_absent"]["readers"] != {"M11": 1}: return False
    return all(calls[key]["output"] == int(key not in ("writer_replay", "M11_absent"))
               and set(calls[key]["source"].values()) == {1}
               for key in ["writer_replay", "M11_absent"]
               + [f"top{count}" for count in PREFIXES] + ["all_hidden_formula"])


def main():
    paths = {"prior": PRIOR, "factor_result": FACTOR_RESULT,
             "transfer_result": TRANSFER_RESULT, "tensor_result": TENSOR_RESULT,
             "v19_capability": V19_CAPABILITY, "builder17": BUILDER17, "builder18": BUILDER18}
    observed = {name: sha(path) for name, path in paths.items()}
    factor_result, v19 = json.loads(FACTOR_RESULT.read_text()), json.loads(V19_CAPABILITY.read_text())
    rows17 = [row for row in fresh17.build_rows() if row["transform_id"] == "A2"]
    rows18 = {family: [row for row in fresh18.build_rows() if row["transform_id"] == family]
              for family in ("A1", "A2", "P", "C")}
    authority_ok = bool(observed == EXPECTED
        and factor_result.get("terminal") == "factor_hierarchy_activation_specific"
        and v19.get("terminal") == "screen" and v19.get("causal_outcomes_opened") is False
        and len(rows17) == 16 and all(len(rows18[key]) == 16 for key in rows18))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "selection_rows": {"v17_A2": len(rows17), "v18_A1": len(rows18["A1"])},
           "test_rows": {key: len(rows18[key]) for key in ("A2", "P", "C")},
           "prefixes": PREFIXES, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("cross-construction greedy authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend, torch = producer.Bilin18TorchBackend.load("cuda"), __import__("torch")
    down = backend.model.transformer.h[11].mlp.Down.weight.detach().float()
    training = {}
    with torch.no_grad():
        for label, rows in (("v17_A2", rows17), ("v18_A1", rows18["A1"])):
            hidden, output, calls = capture_delta(backend, rows)
            training[label] = {"hidden": hidden, "output": output, "calls": calls,
                               "gram": factor_gram(hidden, down)}
        order_tuple = exact_greedy([training[key]["gram"] for key in training], max(PREFIXES))
        order = torch.tensor(order_tuple, dtype=torch.long, device=backend.device)
        training_curves, gram_direct_error = {}, 0.0
        for label, record in training.items():
            training_curves[label] = {}
            for count in PREFIXES:
                gram_value = residual_from_gram(record["gram"], order_tuple[:count])
                direct_value = direct_residual(record["hidden"], record["output"], down,
                                               order_tuple[:count])
                gram_direct_error = max(gram_direct_error, abs(gram_value - direct_value))
                training_curves[label][str(count)] = {
                    "gram_relative_residual": gram_value,
                    "direct_relative_residual": direct_value}
        reports, effects, instruments = {}, {}, {}
        for family in ("A2", "P", "C"):
            reports[family], panel_effects, instruments[family] = test_panel(
                backend, rows18[family], order)
            effects[family] = {key: value.tolist() for key, value in panel_effects.items()}
    target_rms = {label: transfer.rms(effects["A2"][label]) for label in ("top128", "top256")}
    control_ratios = {family: {label: transfer.rms(effects[family][label])
        / max(target_rms[label], 1e-30) for label in target_rms} for family in ("P", "C")}
    A = bool(authority_ok and gram_direct_error <= BARS["gram_direct"]
        and all(calls_ok(training[label]["calls"], training=True) for label in training)
        and all(calls_ok(instruments[family]["calls"]) for family in instruments)
        and all(record["writer_replay_max_abs_margin_error"] <= BARS["replay"]
                and record["all_hidden_margin_max_abs_error"] <= BARS["formula"]
                and record["all_hidden_output_relative_squared_error"] <= BARS["output_rse"]
                for record in instruments.values())
        and finite({"training": training_curves, "reports": reports, "controls": control_ratios})
        and PRICE["model_forwards"] == 6 + 3 * (3 + PRICE["intervention_arms_per_test_panel"])
        and PRICE["sequence_evaluations"] == PRICE["model_forwards"] * 16)
    B = all(training_curves[label]["128"]["direct_relative_residual"] <= BARS["train_residual"]
            for label in training_curves)
    C = bool(reports["A2"]["top128"]["output_relative_residual"] <= BARS["heldout_residual"]
        and reports["A2"]["top128"]["signed_recovery"] >= BARS["top128_recovery"]
        and reports["A2"]["top128"]["cosine"] >= BARS["top128_cosine"]
        and reports["A2"]["top128"]["direction_agreement"] >= BARS["direction"])
    D = bool(reports["A2"]["top256"]["signed_recovery"] >= BARS["top256_recovery"]
        and reports["A2"]["top256"]["cosine"] >= BARS["top256_cosine"]
        and reports["A2"]["top256"]["direction_agreement"] >= BARS["direction"])
    E = all(ratio <= BARS["control_ratio"] for family in control_ratios
            for ratio in control_ratios[family].values())
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A else "nonselective_greedy" if not E else
        "compact_crossconstruction_factor_program" if C else
        "broad_crossconstruction_factor_program" if D else
        "training_compact_test_specific" if B else "diffuse_native_factor_support")
    result = {"schema": "temporal_iswas_m11_crossconstruction_exact_factor_greedy_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "greedy_order_first256": order_tuple, "training_residual_curves": training_curves,
        "gram_direct_residual_max_abs_error": gram_direct_error,
        "test_reports": reports, "test_effect_vectors": effects,
        "control_normalized_rms_ratios": control_ratios, "instrument": instruments,
        "v19_causal_outcomes_opened": False, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("training_residual_curves",
        "gram_direct_residual_max_abs_error", "test_reports", "control_normalized_rms_ratios",
        "instrument", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
