#!/usr/bin/env python3
"""Sealed A11/M11 normalized-reader loss and complete-output rescue."""

# BQGATE: EXPERIMENT pred_a_authority_capability_source_replay_self_clamp_hook_coverage_finiteness_and_exact_price pred_b_A11_reader_loss_is_stable_and_complete_A11_output_rescues pred_c_M11_reader_loss_is_stable_and_complete_M11_output_rescues pred_d_joint_A11_M11_loss_physically_reproduces_positive_complementarity pred_e_matched_attention_and_mlp_controls_are_inert
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import module_reader_loss_rescue as intervention


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17.py"
CAPABILITY_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v17_capability_v1.py"
CAPABILITY_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v17_capability_v1_result.json"
BINDING = ROOT / "circuits/bindings/temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1.json"
EFFECT_RESULT = ROOT / "circuits/followups/temporal_iswas_identity_p7_live_module_effect_game_ood_v2_result.json"
GREEDY_RESULT = ROOT / "circuits/followups/temporal_iswas_l11h3_source_writer_weight_ordered_greedy_v1_result.json"
INTERVENTION = ROOT / "ops/module_reader_loss_rescue.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v17_a11_m11_reader_loss_output_rescue_v1"
EXPECTED = {
    "prior": "eac92572567b2d8e670dc2c02ec91af0439666f4901ab2aa2668f27f1af46923",
    "builder": "7e59800324e5b6a4a9c5af564cbe8fb5cf642cfd6e702e1e97c32bc0cc54fe9e",
    "capability_runner": "1ae08738956b6bb5a76ddff87c9e2a74c1d3f56b6f929e3c718caf5314936725",
    "effect_result": "171d60c3aaf2d35b6a400b9ac753c2c0d8936ee6d354ab91e3bcc9da4f813fc2",
    "greedy_result": "6a2ae6598ae95269f7d9e24e13196155342f690d331c78efed5115a8cf84bda2",
    "intervention": "18eac89dcf8964ad14ab4b90cfcbcd3695ec26056bdff075e620b1678958f857",
}
PRICE = {"checkpoint_loads": 1, "model_forwards": 17, "sequence_evaluations": 272,
         "scored_token_positions": 544, "intervention_arms": 14,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
BARS = {"replay": 1e-5, "A11_loss": .08, "M11_loss": .06, "rescue_fraction": .70,
        "rescued_cosine": .95, "rescued_recovery_error": .10,
        "joint_excess": .02, "control_loss": .03}
SOURCE_HEADS = ((7, 7), (9, 4))
SITES = ("A11", "M11", "A12_control", "M16_control")
ARM_SPECS = {
    "A11_reader_loss": (("A11",), ()),
    "A11_reader_loss_output_rescue": (("A11",), ("A11",)),
    "M11_reader_loss": (("M11",), ()),
    "M11_reader_loss_output_rescue": (("M11",), ("M11",)),
    "A11_M11_joint_reader_loss": (("A11", "M11"), ()),
    "A11_M11_joint_loss_A11_rescue": (("A11", "M11"), ("A11",)),
    "A11_M11_joint_loss_M11_rescue": (("A11", "M11"), ("M11",)),
    "A11_M11_joint_loss_both_rescue": (("A11", "M11"), ("A11", "M11")),
    "A12_control_reader_loss": (("A12_control",), ()),
    "A12_control_reader_loss_output_rescue": (("A12_control",), ("A12_control",)),
    "M16_control_reader_loss": (("M16_control",), ()),
    "M16_control_reader_loss_output_rescue": (("M16_control",), ("M16_control",)),
}
PREDICTION_KEYS = (
    "pred_a_authority_capability_source_replay_self_clamp_hook_coverage_finiteness_and_exact_price",
    "pred_b_A11_reader_loss_is_stable_and_complete_A11_output_rescues",
    "pred_c_M11_reader_loss_is_stable_and_complete_M11_output_rescues",
    "pred_d_joint_A11_M11_loss_physically_reproduces_positive_complementarity",
    "pred_e_matched_attention_and_mlp_controls_are_inert",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def site_modules(model):
    return {
        "A11": (model.transformer.h[11].attn, model.transformer.h[11].attn.c_proj),
        "M11": model.transformer.h[11].mlp,
        "A12_control": (model.transformer.h[12].attn, model.transformer.h[12].attn.c_proj),
        "M16_control": model.transformer.h[16].mlp,
    }


def postcue_rows(base_batch, donor_batch):
    rows = []
    for base_ids, donor_ids, query in zip(
        base_batch.token_rows, donor_batch.token_rows, base_batch.semantic_positions
    ):
        if len(base_ids) != len(donor_ids):
            raise RuntimeError("v17 A2 paired length changed")
        changed = [index for index, pair in enumerate(zip(base_ids[: query + 1], donor_ids[: query + 1]))
                   if pair[0] != pair[1]]
        if not changed or changed != list(range(changed[0], changed[-1] + 1)):
            raise RuntimeError("v17 A2 cue span changed")
        stop = changed[-1] + 1
        rows.append(tuple(range(stop, int(query) + 1)))
    return tuple(rows)


def capture_source_heads(forward, model, batch):
    saved, calls, handles = {}, {layer: 0 for layer, _head in SOURCE_HEADS}, []
    heads, width = model.config.n_head, model.config.n_embd // model.config.n_head
    for layer, _head in SOURCE_HEADS:
        def save(_module, arguments, layer=layer):
            calls[layer] += 1
            saved[layer] = arguments[0].detach().clone().view(
                len(batch.row_ids), arguments[0].shape[1], heads, width)
        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(save))
    try:
        output = forward()
    finally:
        for handle in handles:
            handle.remove()
    if set(saved) != {layer for layer, _head in SOURCE_HEADS} or set(calls.values()) != {1}:
        raise RuntimeError(f"source-head capture coverage changed: {calls}")
    return output, saved, calls


def with_source_heads(forward, model, batch, base, donor, positions):
    handles, calls = [], {layer: 0 for layer, _head in SOURCE_HEADS}
    heads, width = model.config.n_head, model.config.n_embd // model.config.n_head
    for layer, head in SOURCE_HEADS:
        def patch(_module, arguments, layer=layer, head=head):
            calls[layer] += 1
            value = arguments[0]
            changed = value.clone().view(len(batch.row_ids), value.shape[1], heads, width)
            for row, row_positions in enumerate(positions):
                if row_positions:
                    indices = list(row_positions)
                    changed[row, indices, head] += (
                        donor[layer][row, indices, head] - base[layer][row, indices, head]
                    ).to(changed)
            return (changed.reshape_as(value),) + tuple(arguments[1:])
        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    try:
        output = forward()
    finally:
        for handle in handles:
            handle.remove()
    if set(calls.values()) != {1}:
        raise RuntimeError(f"source-head patch coverage changed: {calls}")
    return output, calls


def capture_entry_and_modules(forward, model, modules):
    entry, calls = {}, 0
    def save_entry(_module, arguments):
        nonlocal calls
        calls += 1
        entry["x10"] = arguments[0].detach().clone()
    handle = model.transformer.h[10].register_forward_pre_hook(save_entry)
    try:
        output, readers, outputs, module_calls = intervention.capture_reader_outputs(
            forward, modules)
    finally:
        handle.remove()
    if calls != 1 or set(entry) != {"x10"}:
        raise RuntimeError("block-10 entry capture coverage changed")
    return output, entry["x10"], readers, outputs, {"entry": calls, "modules": module_calls}


def with_entry(forward, model, entry, positions):
    calls = 0
    def patch(_module, arguments):
        nonlocal calls
        calls += 1
        changed = intervention.replace_positions(arguments[0], entry, positions)
        return (changed,) + tuple(arguments[1:])
    handle = model.transformer.h[10].register_forward_pre_hook(patch)
    try:
        output = forward()
    finally:
        handle.remove()
    if calls != 1:
        raise RuntimeError("block-10 entry patch coverage changed")
    return output, calls


def run_arm(backend, batch, modules, entry, absent_readers, present_outputs,
            positions, losses, rescues):
    inner_calls = None
    def call():
        nonlocal inner_calls
        result, inner_calls = intervention.run_reader_loss_rescue(
            lambda: backend.native(batch, capture=False), modules,
            absent_readers, present_outputs, positions, losses=losses, rescues=rescues)
        return result
    output, entry_calls = with_entry(call, backend.model, entry, positions)
    return output, {"entry": entry_calls, "modules": inner_calls}


def toward_donor_margin(output):
    return np.asarray([float(foil) - float(answer) for answer, foil in output.answer_foil],
                      dtype=np.float64)


def metrics(value, reference):
    value, reference = np.asarray(value, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    rr = max(float(reference @ reference), 1e-30)
    vv, vr = float(value @ value), float(value @ reference)
    return {"signed_recovery": vr / rr,
            "cosine": vr / math.sqrt(max(vv * rr, 1e-30)),
            "relative_residual": float(np.linalg.norm(value - reference) / math.sqrt(rr)),
            "direction_agreement": float(np.mean(value * reference > 0)),
            "count": int(reference.size)}


def rescue_fraction(loss_value, rescue_value, reference):
    lost = reference - loss_value
    denominator = float(lost @ reference)
    if denominator <= 0:
        return 0.0
    return float((rescue_value - loss_value) @ reference / denominator)


def eligibility():
    if not BINDING.exists() or not CAPABILITY_RESULT.exists():
        return False, "awaiting_binding", None
    binding = json.loads(BINDING.read_text())
    capability = json.loads(CAPABILITY_RESULT.read_text())
    expected_binding = {
        "schema": "temporal_iswas_v17_a11_m11_reader_loss_output_rescue_binding_v1",
        "candidate_id": CANDIDATE_ID,
        "edge_prior_sha256": EXPECTED["prior"],
        "edge_runner_sha256": sha(SELF),
        "capability_result_sha256": sha(CAPABILITY_RESULT),
    }
    ok = bool(all(binding.get(key) == value for key, value in expected_binding.items())
        and capability.get("terminal") == "screen"
        and capability.get("causal_outcomes_opened") is False
        and all(capability.get("predictions", {}).values()))
    return ok, "eligible" if ok else "invalid_binding", binding


def main():
    files = {"prior": PRIOR, "builder": BUILDER, "capability_runner": CAPABILITY_RUNNER,
             "effect_result": EFFECT_RESULT, "greedy_result": GREEDY_RESULT,
             "intervention": INTERVENTION}
    observed = {name: sha(path) for name, path in files.items()}
    effect, greedy = json.loads(EFFECT_RESULT.read_text()), json.loads(GREEDY_RESULT.read_text())
    rows = [row for row in fresh.build_rows() if row["transform_id"] == "A2"]
    eligible, eligibility_status, binding = eligibility()
    authority_ok = bool(observed == EXPECTED and len(rows) == 16
        and effect.get("terminal") == "redundant_or_complementary_live_routes"
        and effect.get("stable_interaction_pairs") == ["A11+M11"]
        and greedy["selected_prefixes"]["iswas"]["heads"] == ["L07H07", "L09H04"])
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "eligibility": eligibility_status, "eligible": eligible, "rows": len(rows),
           "source_heads": SOURCE_HEADS, "sites": SITES, "arms": list(ARM_SPECS),
           "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok or not eligible:
        raise RuntimeError(f"sealed reader-loss/rescue authority unavailable: {eligibility_status}")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend, model = producer.Bilin18TorchBackend.load("cuda"), None
    model = backend.model
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    positions = tuple(tuple(range(int(query) + 1)) for query in base_batch.semantic_positions)
    source_positions = postcue_rows(base_batch, donor_batch)
    modules = site_modules(model)
    calls = {}
    with backend.torch.no_grad():
        def native_call():
            return capture_entry_and_modules(
                lambda: backend.native(base_batch, capture=False), model, modules)
        native_bundle, base_heads, calls["base_head_capture"] = capture_source_heads(
            native_call, model, base_batch)
        native, native_entry, absent_readers, _absent_outputs, calls["native_capture"] = native_bundle
        donor, donor_heads, calls["donor_head_capture"] = capture_source_heads(
            lambda: backend.native(donor_batch, capture=False), model, donor_batch)

        def writer_capture_call():
            return capture_entry_and_modules(
                lambda: backend.native(base_batch, capture=False), model, modules)
        writer_bundle, calls["source_patch"] = with_source_heads(
            writer_capture_call, model, base_batch, base_heads, donor_heads, source_positions)
        writer, writer_entry, _present_readers, present_outputs, calls["writer_capture"] = writer_bundle
        replay, calls["writer_replay"] = with_entry(
            lambda: backend.native(base_batch, capture=False), model, writer_entry, positions)
        self_clamp, calls["source_absent_self_clamp"] = run_arm(
            backend, base_batch, modules, native_entry, absent_readers, present_outputs,
            positions, ("A11", "M11"), ())
        arm_outputs = {}
        for arm, (losses, rescues) in ARM_SPECS.items():
            arm_outputs[arm], calls[arm] = run_arm(
                backend, base_batch, modules, writer_entry, absent_readers, present_outputs,
                positions, losses, rescues)

    native_margin, writer_margin = toward_donor_margin(native), toward_donor_margin(writer)
    replay_error = float(np.max(np.abs(toward_donor_margin(replay) - writer_margin)))
    self_error = float(np.max(np.abs(toward_donor_margin(self_clamp) - native_margin)))
    arm_margins = {arm: toward_donor_margin(output) for arm, output in arm_outputs.items()}
    reports, summaries = [], {}
    source_live = {}
    for phase in ("FIT", "HOLDOUT"):
        chosen = np.asarray([row["group_number"] < 8 if phase == "FIT" else row["group_number"] >= 8
                             for row in rows])
        reference = writer_margin[chosen] - native_margin[chosen]
        reference_squared_l2 = float(reference @ reference)
        source_live[phase] = reference_squared_l2 > 1e-12
        summaries[phase] = {"source_reference_squared_l2": reference_squared_l2,
                            "source_live": source_live[phase]}
        for arm, margin in arm_margins.items():
            value = margin[chosen] - native_margin[chosen]
            report = {"phase": phase, "arm": arm, **metrics(value, reference)}
            reports.append(report); summaries[phase][arm] = report
        for label, loss_arm, rescue_arm in (
            ("A11", "A11_reader_loss", "A11_reader_loss_output_rescue"),
            ("M11", "M11_reader_loss", "M11_reader_loss_output_rescue"),
            ("A12_control", "A12_control_reader_loss", "A12_control_reader_loss_output_rescue"),
            ("M16_control", "M16_control_reader_loss", "M16_control_reader_loss_output_rescue"),
        ):
            loss_value = arm_margins[loss_arm][chosen] - native_margin[chosen]
            rescue_value = arm_margins[rescue_arm][chosen] - native_margin[chosen]
            summaries[phase][label] = {
                "loss": 1.0 - summaries[phase][loss_arm]["signed_recovery"],
                "rescue_fraction": rescue_fraction(loss_value, rescue_value, reference),
                "rescued_cosine": summaries[phase][rescue_arm]["cosine"],
                "rescued_recovery_error": abs(1.0 - summaries[phase][rescue_arm]["signed_recovery"]),
            }
        summaries[phase]["joint_excess"] = (
            1.0 - summaries[phase]["A11_M11_joint_reader_loss"]["signed_recovery"]
            - summaries[phase]["A11"]["loss"] - summaries[phase]["M11"]["loss"])

    calls_ok = finite(calls)
    A = bool(authority_ok and eligible and replay_error <= BARS["replay"]
             and self_error <= BARS["replay"] and calls_ok and finite(summaries)
             and PRICE["model_forwards"] * len(rows) == PRICE["sequence_evaluations"])
    def site_pass(label, loss_bar):
        return all(summaries[phase][label]["loss"] >= loss_bar
            and summaries[phase][label]["rescue_fraction"] >= BARS["rescue_fraction"]
            and summaries[phase][label]["rescued_cosine"] >= BARS["rescued_cosine"]
            and summaries[phase][label]["rescued_recovery_error"] <= BARS["rescued_recovery_error"]
            for phase in ("FIT", "HOLDOUT"))
    B, C = site_pass("A11", BARS["A11_loss"]), site_pass("M11", BARS["M11_loss"])
    D = all(summaries[phase]["joint_excess"] >= BARS["joint_excess"]
        and summaries[phase]["A11_M11_joint_loss_both_rescue"]["cosine"] >= BARS["rescued_cosine"]
        and abs(1.0 - summaries[phase]["A11_M11_joint_loss_both_rescue"]["signed_recovery"])
            <= BARS["rescued_recovery_error"] for phase in ("FIT", "HOLDOUT"))
    E = all(abs(summaries[phase][label]["loss"]) <= BARS["control_loss"]
        and summaries[phase][label]["rescued_cosine"] >= BARS["rescued_cosine"]
        and summaries[phase][label]["rescued_recovery_error"] <= BARS["rescued_recovery_error"]
        for phase in ("FIT", "HOLDOUT") for label in ("A12_control", "M16_control"))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A else
        "source_writer_ood_null" if not all(source_live.values()) else
        "nonselective_background_route" if not E else
        "directed_A11_M11_complementary_interface" if B and C and D else
        "partial_directed_reader_interface" if B or C else "shapley_dependence_without_direct_edge")
    result = {"schema": "temporal_iswas_v17_a11_m11_reader_loss_output_rescue_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "binding": binding, "source_heads": SOURCE_HEADS, "sites": SITES,
        "source_live": source_live,
        "instrument": {"writer_replay_max_abs_margin_error": replay_error,
                       "source_absent_self_clamp_max_abs_margin_error": self_error,
                       "calls": calls},
        "reports": reports, "summaries": summaries, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("instrument", "summaries", "predictions",
                                                    "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
