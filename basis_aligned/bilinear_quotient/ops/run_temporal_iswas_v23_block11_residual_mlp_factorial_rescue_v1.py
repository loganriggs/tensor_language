#!/usr/bin/env python3
"""Exact block-11 residual-carrier versus MLP factorial rescue."""

# BQGATE: EXPERIMENT pred_a_authority_hooks_closures_self_patch_finiteness_and_exact_price pred_b_residual_carrier_is_the_dominant_selective_branch pred_c_m11_is_a_real_but_minor_branch pred_d_exact_joint_rescue_closes_the_block11_mediation
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1 as shared
import transport_boundary_capture as boundary

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_block11_residual_mlp_factorial_rescue_v1.json"
NECESSITY = ROOT / "circuits/followups/temporal_iswas_v23_four_head_necessity_occupied_mode_rescue_v1_result.json"
AUDIT = ROOT / "circuits/followups/temporal_iswas_v23_four_head_necessity_occupied_mode_rescue_v1_interpretation_audit.json"
V15 = ROOT / "circuits/followups/temporal_iswas_v15_residual_suffix_module_mediation_atlas_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23.py"
SHARED = ROOT / "ops/run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1.py"
BOUNDARY = ROOT / "ops/transport_boundary_capture.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_block11_residual_mlp_factorial_rescue_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v23_block11_residual_mlp_factorial_rescue_v1"
ROUTES = ("L8H1", "L9H1", "L9H4", "L11H3")
EXPECTED = {
    "prior": "564c59004f32c26632fbc54973ed86cdb9c8dfdd7f19207f3922c4ad57ce6e9e",
    "necessity": "284ff3e05dddd14a979c88c6aa9e662deac4721e678a077055021f919f7a943f",
    "audit": "12db9bababe983802bebacb2970e3247516cd2259dc7ea062cba8988e529efe2",
    "v15": "a8d990bc3aeec08193f8cd90ec64c6581e15927308c24fa6d6e72f879c248936",
    "builder": "a4830fd110b8cd854a5f28bfae776f697a15d4f791355990e02ea030fdca4c05",
    "shared": "9ab2a9edb60f4e3e4befebf11f55225659559a2a504075567218eab9bf903d06",
    "boundary": "d027438fbd9f65b336793cd628c8d55f41510adcf36266198b43449d36cdc8b9",
}
BARS = {"closure": 1e-4, "residual_projection": .65, "direction": .90,
        "control": .15, "mlp_min": .05, "mlp_max": .35,
        "dominance": .40, "joint_projection": .99, "joint_restore": 1e-4}
PRICE = {"checkpoint_loads": 1, "model_forwards_exact": 7,
         "sequence_evaluations_exact": 448, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_hooks_closures_self_patch_finiteness_and_exact_price",
    "pred_b_residual_carrier_is_the_dominant_selective_branch",
    "pred_c_m11_is_a_real_but_minor_branch",
    "pred_d_exact_joint_rescue_closes_the_block11_mediation",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def first_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def replace_first(output, value):
    if isinstance(output, tuple): return (value,) + tuple(output[1:])
    if isinstance(output, list): return [value] + list(output[1:])
    return value


def metrics(torch, value, reference, selected):
    x, y = value[selected].reshape(-1).double(), reference[selected].reshape(-1).double()
    xx, yy, xy = float(x @ x), float(y @ y), float(x @ y)
    return {"cosine": xy / math.sqrt(max(xx * yy, 1e-30)),
            "signed_projection": xy / max(yy, 1e-30),
            "relative_residual": math.sqrt(float((x-y) @ (x-y)) / max(yy, 1e-30)),
            "norm_ratio": math.sqrt(xx / max(yy, 1e-30)),
            "direction_fraction": float(((value[selected] * reference[selected]) > 0).float().mean())}


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(float(value))


def execute_capture(model, execute):
    saved, calls = {}, {"block11_output": 0}
    def capture(_module, _args, output):
        calls["block11_output"] += 1
        saved["block11_output"] = first_tensor(output).detach().clone()
    handle = model.transformer.h[11].register_forward_hook(capture)
    try:
        result, state, boundary_calls = boundary.execute_with_boundaries(
            model, execute, block_layers=(11,), raw_attention_layer=11, mlp_input_layer=11)
    finally:
        handle.remove()
    if calls["block11_output"] != 1: raise RuntimeError("block11 output capture count changed")
    return result, state, saved["block11_output"], {**calls, **boundary_calls}


def run_rescue(backend, batch, base_cache, correction):
    saved, calls = {}, {"block11_correction": 0}
    def correct(_module, _args, output):
        calls["block11_correction"] += 1
        tensor = first_tensor(output)
        if tensor.shape != correction.shape: raise RuntimeError("block11 correction shape changed")
        changed = tensor + correction.to(tensor)
        saved["block11_output"] = changed.detach().clone()
        return replace_first(output, changed)
    handle = backend.model.transformer.h[11].register_forward_hook(correct)
    try: logits, lengths, final, _ = shared.run_patch(backend, batch, base_cache, ROUTES, False)
    finally: handle.remove()
    if calls["block11_correction"] != 1: raise RuntimeError("block11 correction count changed")
    return logits, lengths, final, saved["block11_output"], calls


def main():
    paths = {"prior": PRIOR, "necessity": NECESSITY, "audit": AUDIT, "v15": V15,
             "builder": BUILDER, "shared": SHARED, "boundary": BOUNDARY}
    observed = {name: sha(path) for name, path in paths.items()}
    necessity = json.loads(NECESSITY.read_text()); audit = json.loads(AUDIT.read_text())
    rows = fresh.build_rows(); target_ids = set(
        json.loads((ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json").read_text())
        ["population"]["target_row_ids"])
    authority_ok = bool(observed == EXPECTED and len(rows) == 64 and
        sum(row["row_id"] in target_ids for row in rows) == 30 and
        necessity.get("predictions", {}).get("pred_b_four_head_union_is_selectively_necessary_in_reverse") is True and
        audit.get("terminal_label_correction", {}).get("correct_interpretation") ==
        "four_head_necessary_M11_is_minor_mediator_and_leading_mode_explains_most_of_M11_share")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "routes": ROUTES, "rows": len(rows), "target_rows": len(target_ids), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("residual/MLP factorial authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    with torch.no_grad():
        base_logits, base_lengths, _, base_cache, _, _ = shared.capture_native(backend, base_batch, False)
        donor_pack, donor_state, donor_output, donor_calls = execute_capture(
            backend.model, lambda: shared.capture_native(backend, donor_batch, False))
        donor_logits, donor_lengths, donor_final, donor_cache, _, _ = donor_pack
        self_logits, _, self_final, _ = shared.run_patch(backend, donor_batch, donor_cache, ROUTES, False)
        removed_pack, removed_state, removed_output, removed_calls = execute_capture(
            backend.model, lambda: shared.run_patch(backend, donor_batch, base_cache, ROUTES, False))
        removed_logits, removed_lengths, removed_final, _ = removed_pack
        donor_raw = boundary.pre_mlp_raw_state(backend.model, donor_state, layer=11)
        removed_raw = boundary.pre_mlp_raw_state(backend.model, removed_state, layer=11)
        delta_y = donor_raw - removed_raw
        delta_output = donor_output - removed_output
        delta_m = delta_output - delta_y
        y_logits, y_lengths, y_final, y_output, y_calls = run_rescue(
            backend, donor_batch, base_cache, delta_y)
        m_logits, m_lengths, m_final, m_output, m_calls = run_rescue(
            backend, donor_batch, base_cache, delta_m)
        joint_logits, joint_lengths, joint_final, joint_output, joint_calls = run_rescue(
            backend, donor_batch, base_cache, delta_y + delta_m)
        base_margin = shared.margin(torch, base_logits, base_lengths, rows, backend.device)
        donor_margin = shared.margin(torch, donor_logits, donor_lengths, rows, backend.device)
        removed_margin = shared.margin(torch, removed_logits, removed_lengths, rows, backend.device)
        y_margin = shared.margin(torch, y_logits, y_lengths, rows, backend.device)
        m_margin = shared.margin(torch, m_logits, m_lengths, rows, backend.device)
        joint_margin = shared.margin(torch, joint_logits, joint_lengths, rows, backend.device)
    closure = float((delta_output - delta_y - delta_m).abs().max())
    self_error = max(float((self_logits-donor_logits).abs().max()), float((self_final-donor_final).abs().max()))
    joint_restore = {"block11_output_max_abs": float((joint_output-donor_output).abs().max()),
                     "final_hidden_max_abs": float((joint_final-donor_final).abs().max()),
                     "logits_max_abs": float((joint_logits-donor_logits).abs().max())}
    target = torch.as_tensor([row["row_id"] in target_ids for row in rows], device=backend.device)
    panels = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
              for panel in ("P", "C")}
    halves = {name: target & torch.as_tensor([int(row["group_number"]) % 4 in residues for row in rows], device=backend.device)
              for name, residues in (("first", (0, 1)), ("second", (2, 3)))}
    native_delta = donor_margin-base_margin
    head_effect = donor_margin-removed_margin
    effects = {"residual": y_margin-removed_margin, "mlp": m_margin-removed_margin,
               "joint": joint_margin-removed_margin}
    target_rms = float(native_delta[target].square().mean().sqrt())
    reports = {name: metrics(torch, effect, head_effect, target) for name, effect in effects.items()}
    for name, effect in effects.items():
        reports[name]["controls"] = {panel: float(effect[mask].square().mean().sqrt()) / max(target_rms, 1e-30)
                                      for panel, mask in panels.items()}
    half_reports = {name: {half: metrics(torch, effect, head_effect, mask) for half, mask in halves.items()}
                    for name, effect in effects.items()}
    interaction = effects["joint"] - effects["residual"] - effects["mlp"]
    interaction_report = metrics(torch, interaction, head_effect, target)
    calls_ok = bool(all(value == 1 for value in (*donor_calls.values(), *removed_calls.values(),
        *y_calls.values(), *m_calls.values(), *joint_calls.values())))
    A = bool(authority_ok and calls_ok and closure <= BARS["closure"] and self_error <= BARS["closure"]
        and finite([reports, half_reports, interaction_report, joint_restore])
        and PRICE["model_forwards_exact"] == 7 and PRICE["sequence_evaluations_exact"] == 7*len(rows))
    B = bool(reports["residual"]["signed_projection"] >= BARS["residual_projection"]
        and reports["residual"]["direction_fraction"] >= BARS["direction"]
        and all(v <= BARS["control"] for v in reports["residual"]["controls"].values())
        and all(v["signed_projection"] > 0 for v in half_reports["residual"].values()))
    C = bool(BARS["mlp_min"] <= reports["mlp"]["signed_projection"] <= BARS["mlp_max"]
        and reports["residual"]["signed_projection"]-reports["mlp"]["signed_projection"] >= BARS["dominance"]
        and all(v["signed_projection"] > 0 for v in half_reports["mlp"].values()))
    D = bool(reports["joint"]["signed_projection"] >= BARS["joint_projection"]
        and max(joint_restore.values()) <= BARS["joint_restore"])
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A or not D else
                "residual_carrier_with_minor_M11_branch" if B and C else
                "residual_carrier_without_registered_minor_M11_branch" if B else
                "interacting_or_M11_dominant_block11_pair")
    result = {"schema": "temporal_iswas_v23_block11_residual_mlp_factorial_rescue_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": observed,
        "component_closure_max_abs": closure, "self_patch_max_abs": self_error,
        "joint_restore": joint_restore, "hook_calls": {"donor": donor_calls, "removed": removed_calls,
            "residual": y_calls, "mlp": m_calls, "joint": joint_calls},
        "reports": reports, "half_reports": half_reports, "downstream_interaction_report": interaction_report,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE,
        "model_forwards": 7, "selected_component_or_threshold_after_outcome": None}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("component_closure_max_abs", "self_patch_max_abs",
        "joint_restore", "reports", "downstream_interaction_report", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
