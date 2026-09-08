#!/usr/bin/env python3
"""Retrospective fixed-gain compiler for the task-tangent top-16 M11 core."""

# BQGATE: EXPERIMENT pred_a_authority_formula_replays_hooks_finiteness_and_exact_price pred_b_fixed_gain_top16_recovers_all_three_observed_target_panels pred_c_fixed_gain_is_construction_stable pred_d_fixed_gain_program_remains_selective_on_both_corpora pred_e_result_is_labeled_retrospective_and_cannot_promote_without_new_corpus
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
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v18 as fresh18
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v19 as fresh19
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
import run_temporal_iswas_m11_crossconstruction_exact_factor_greedy_v1 as executor
import run_temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1 as parent
import run_temporal_iswas_v18_frozen_shared_tensor_transfer_v1 as transfer


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_top16_fixed_gain_redundancy_compiler_v1.json"
TASK_RESULT = ROOT / "circuits/followups/temporal_iswas_m11_crossconstruction_task_tangent_factor_order_v1_result.json"
V19_RESULT = ROOT / "circuits/followups/temporal_iswas_v19_frozen_task_tangent_factor_confirmation_v1_result.json"
BUILDER18 = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v18.py"
BUILDER19 = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v19.py"
EXECUTOR = ROOT / "ops/run_temporal_iswas_m11_crossconstruction_exact_factor_greedy_v1.py"
TRANSFER = ROOT / "ops/run_temporal_iswas_v18_frozen_shared_tensor_transfer_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_top16_fixed_gain_redundancy_compiler_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.top16_fixed_gain_redundancy_compiler_v1"
EXPECTED = {
    "prior": "5d75d40d3aeabfcea71595249c91fcc110bd2f06d33003eb1c7425e3788986b4",
    "task_result": "e0af6cc965d2d6a560dbde548e191268c77fcac7a5817ad54ea2ccaefe7050b7",
    "v19_result": "876c66398bee6ca771534b63aba945231e55ae24b5b0250a4dd7115c2e64c127",
    "builder18": "44a51b3370bcf59f9cb5b17e5d63ac1cfdba3e883072112a49ec2f92ba78ae2a",
    "builder19": "67916a1c83a019e037fc14415932560ba98cc5dd46a623216b43fd36d267d8b6",
    "executor": "e83aa3a75819c68587f46d5b53f332eab71cb1800a294d448abc854a92dfefb7",
    "transfer": "04cde7cd9929222835a6227dc14ffa4ccd9db6ef32b9efb4ce532acb791f9f65",
}
GAIN, TOP = 1.25, 16
TARGETS = ("v18_A2", "v19_A1", "v19_A2")
CONTROLS = ("v18_P", "v18_C", "v19_P", "v19_C")
PRICE = {"checkpoint_loads": 1, "model_forwards": 56, "sequence_evaluations": 896,
         "scored_token_positions": 1792, "panels": 7, "forwards_per_panel": 8,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
BARS = {"replay": 1e-5, "formula": 1e-5, "output_rse": 1e-8,
        "recovery_low": .80, "recovery_high": 1.20, "cosine": .95,
        "direction": .875, "recovery_range": .30, "control_ratio": .25}
PREDICTION_KEYS = (
    "pred_a_authority_formula_replays_hooks_finiteness_and_exact_price",
    "pred_b_fixed_gain_top16_recovers_all_three_observed_target_panels",
    "pred_c_fixed_gain_is_construction_stable",
    "pred_d_fixed_gain_program_remains_selective_on_both_corpora",
    "pred_e_result_is_labeled_retrospective_and_cannot_promote_without_new_corpus",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def panel_run(backend, rows, chosen):
    torch, model = backend.torch, backend.model
    batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    positions = tuple(tuple(range(int(query) + 1)) for query in batch.semantic_positions)
    base_source, donor_source = transfer.suffix_position_rows(batch, donor_batch)
    calls = {}
    native_bundle, base_heads, calls["native_source"] = parent.capture_source_heads(
        lambda: executor.capture_m11(lambda: parent.full_forward(backend, batch), model), model, batch)
    native, absent_readers, absent_outputs, calls["native_sites"] = native_bundle
    donor, donor_heads, calls["donor_source"] = parent.capture_source_heads(
        lambda: parent.full_forward(backend, donor_batch), model, donor_batch)
    writer_bundle, calls["writer_source"] = transfer.with_source_heads_aligned(
        lambda: executor.capture_m11(lambda: parent.full_forward(backend, batch), model),
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
    selected_delta = hidden_delta[..., chosen] @ mlp.Down.weight.detach().float()[:, chosen].T
    for label, gain in (("top16_native", 1.0), ("top16_gain125", GAIN)):
        replacement = absent_outputs["M11"] + (gain * selected_delta).to(absent_outputs["M11"])
        outputs[label], calls[label] = transfer.run_tensor_arm(
            backend, batch, base_heads, donor_heads, base_source, donor_source, positions,
            {}, output_replacement=replacement)
    all_output = absent_outputs["M11"] + (hidden_delta @ mlp.Down.weight.detach().float().T).to(
        absent_outputs["M11"])
    outputs["all_hidden_formula"], calls["all_hidden_formula"] = transfer.run_tensor_arm(
        backend, batch, base_heads, donor_heads, base_source, donor_source, positions,
        {}, output_replacement=all_output)
    writer_margin = parent.toward_donor_margin(writer)
    margins = {label: parent.toward_donor_margin(output) for label, output in outputs.items()}
    reference = writer_margin - margins["M11_absent"]
    effects = {label: margins[label] - margins["M11_absent"]
               for label in ("top16_native", "top16_gain125")}
    reports = {label: parent.metrics(effect, reference) for label, effect in effects.items()}
    reports["native_accuracy"] = float(np.mean([a > f for a, f in native.answer_foil]))
    reports["donor_accuracy"] = float(np.mean([a > f for a, f in donor.answer_foil]))
    diff = executor.selected_matrix(all_output - present_outputs["M11"], positions)
    instrument = {"writer_replay_max_abs_margin_error": float(np.max(np.abs(
        margins["writer_replay"] - writer_margin))),
        "all_hidden_margin_max_abs_error": float(np.max(np.abs(
            margins["all_hidden_formula"] - writer_margin))),
        "all_hidden_output_relative_squared_error": float(diff.square().sum()
            / max(float(executor.selected_matrix(present_outputs["M11"], positions).square().sum()), 1e-30)),
        "calls": calls}
    return reports, effects, instrument


def calls_ok(calls):
    if not all(set(calls[key].values()) == {1}
               for key in ("native_source", "donor_source", "writer_source")): return False
    if not all(calls[key] == {"M11": {"reader": 1, "output": 1}}
               for key in ("native_sites", "writer_sites")): return False
    if calls["M11_absent"]["readers"] != {"M11": 1}: return False
    return all(calls[key]["output"] == int(key not in ("writer_replay", "M11_absent"))
        and set(calls[key]["source"].values()) == {1}
        for key in ("writer_replay", "M11_absent", "top16_native", "top16_gain125",
                    "all_hidden_formula"))


def main():
    paths = {"prior": PRIOR, "task_result": TASK_RESULT, "v19_result": V19_RESULT,
             "builder18": BUILDER18, "builder19": BUILDER19,
             "executor": EXECUTOR, "transfer": TRANSFER}
    observed = {name: sha(path) for name, path in paths.items()}
    task_result, v19_result = json.loads(TASK_RESULT.read_text()), json.loads(V19_RESULT.read_text())
    rows18, rows19 = fresh18.build_rows(), fresh19.build_rows()
    panels = {"v18_A2": [r for r in rows18 if r["transform_id"] == "A2"],
        "v18_P": [r for r in rows18 if r["transform_id"] == "P"],
        "v18_C": [r for r in rows18 if r["transform_id"] == "C"],
        "v19_A1": [r for r in rows19 if r["transform_id"] == "A1"],
        "v19_A2": [r for r in rows19 if r["transform_id"] == "A2"],
        "v19_P": [r for r in rows19 if r["transform_id"] == "P"],
        "v19_C": [r for r in rows19 if r["transform_id"] == "C"]}
    order_tuple = tuple(int(index) for index in task_result["stable_factor_order_first256"])
    authority_ok = bool(observed == EXPECTED
        and task_result.get("terminal") == "compact_task_read_factor_program"
        and v19_result.get("terminal") == "task_tangent_factor_confirmation_failed"
        and len(order_tuple) == len(set(order_tuple)) == 256
        and all(len(rows) == 16 for rows in panels.values()))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
        "panels": {key: len(value) for key, value in panels.items()},
        "gain": GAIN, "top": TOP, "price": PRICE, "evidence_status": "retrospective_calibration"}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("fixed-gain compiler authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    chosen = backend.torch.tensor(order_tuple[:TOP], dtype=backend.torch.long, device=backend.device)
    reports, effects, instruments = {}, {}, {}
    with backend.torch.no_grad():
        for label, rows in panels.items():
            reports[label], panel_effects, instruments[label] = panel_run(backend, rows, chosen)
            effects[label] = {key: value.tolist() for key, value in panel_effects.items()}
    target_rms = transfer.rms(np.concatenate([
        np.asarray(effects[label]["top16_gain125"], dtype=np.float64) for label in TARGETS]))
    control_ratios = {label: transfer.rms(effects[label]["top16_gain125"])
                      / max(target_rms, 1e-30) for label in CONTROLS}
    target_recovery = {label: reports[label]["top16_gain125"]["signed_recovery"]
                       for label in TARGETS}
    A = bool(authority_ok and all(calls_ok(value["calls"]) for value in instruments.values())
        and all(value["writer_replay_max_abs_margin_error"] <= BARS["replay"]
                and value["all_hidden_margin_max_abs_error"] <= BARS["formula"]
                and value["all_hidden_output_relative_squared_error"] <= BARS["output_rse"]
                for value in instruments.values())
        and finite({"reports": reports, "controls": control_ratios})
        and PRICE["model_forwards"] == PRICE["panels"] * PRICE["forwards_per_panel"]
        and PRICE["sequence_evaluations"] == PRICE["model_forwards"] * 16)
    B = all(BARS["recovery_low"] <= target_recovery[label] <= BARS["recovery_high"]
        and reports[label]["top16_gain125"]["cosine"] >= BARS["cosine"]
        and reports[label]["top16_gain125"]["direction_agreement"] >= BARS["direction"]
        for label in TARGETS)
    recovery_range = max(target_recovery.values()) - min(target_recovery.values())
    C = recovery_range <= BARS["recovery_range"]
    D = all(value <= BARS["control_ratio"] for value in control_ratios.values())
    E = True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A else "fixed_gain_nonselective" if not D else
        "fixed_gain_calibration_screen" if B and C else "scalar_redundancy_compiler_failed")
    result = {"schema": "temporal_iswas_top16_fixed_gain_redundancy_compiler_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "evidence_status": "retrospective_calibration_only_not_confirmation",
        "frozen_program": {"factor_indices": order_tuple[:TOP], "gain": GAIN},
        "reports": reports, "effect_vectors": effects, "target_recovery": target_recovery,
        "target_recovery_range": recovery_range, "control_normalized_rms_ratios": control_ratios,
        "instrument": instruments, "predictions": predictions, "terminal": terminal,
        "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("evidence_status", "frozen_program",
        "target_recovery", "target_recovery_range", "control_normalized_rms_ratios",
        "reports", "instrument", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
