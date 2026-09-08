#!/usr/bin/env python3
"""Select exact M11 factors by a cross-construction downstream task tangent."""

# BQGATE: EXPERIMENT pred_a_authority_gradient_and_formula_instruments_hooks_finiteness_and_price pred_b_summed_native_factor_tangent_predicts_full_M11_effect_on_both_selection_constructions pred_c_stable_task_tangent_top64_transfers_to_v18_A2 pred_d_stable_task_tangent_top128_recovers_most_v18_A2 pred_e_task_tangent_factor_program_is_selective_and_v19_remains_sealed
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
import run_temporal_iswas_m11_crossconstruction_exact_factor_greedy_v1 as output_greedy
import run_temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1 as parent
import run_temporal_iswas_v18_frozen_shared_tensor_transfer_v1 as transfer


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_m11_crossconstruction_task_tangent_factor_order_v1.json"
GREEDY_RESULT = ROOT / "circuits/followups/temporal_iswas_m11_crossconstruction_exact_factor_greedy_v1_result.json"
FACTOR_RESULT = ROOT / "circuits/followups/temporal_iswas_v18_frozen_m11_factor_full_reader_transfer_v2_result.json"
V19_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v19_capability_v1_result.json"
GREEDY_RUNNER = ROOT / "ops/run_temporal_iswas_m11_crossconstruction_exact_factor_greedy_v1.py"
BUILDER17 = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17.py"
BUILDER18 = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v18.py"
OUT = ROOT / "circuits/followups/temporal_iswas_m11_crossconstruction_task_tangent_factor_order_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.m11_crossconstruction_task_tangent_factor_order_v1"
EXPECTED = {
    "prior": "d84fc16dd2448189bf73a969079a29e0c77cef334ba403909d3226b8cbd545f6",
    "greedy_result": "fa5db0f2539fd120b29dd1effa7114bc6eb0b4a2d841049ecbb1a4b5199d8907",
    "factor_result": "f558c21f40bfb961e7abc3ecc84dd52e4153e3f7ec0e99a390594c487030a82d",
    "v19_capability": "28f8f134d78279dd82cd7cca94f76e9cf35815bd1a59430ad34601fd0294fdd4",
    "greedy_runner": "e83aa3a75819c68587f46d5b53f332eab71cb1800a294d448abc854a92dfefb7",
    "builder17": "7e59800324e5b6a4a9c5af564cbe8fb5cf642cfd6e702e1e97c32bc0cc54fe9e",
    "builder18": "44a51b3370bcf59f9cb5b17e5d63ac1cfdba3e883072112a49ec2f92ba78ae2a",
}
PREFIXES = output_greedy.PREFIXES
PRICE = {"checkpoint_loads": 1, "model_forwards": 41, "sequence_evaluations": 656,
         "scored_token_positions": 1312, "transformer_backwards": 2,
         "model_updates": 0, "fit_parameters": 0, "selected_native_factor_indices": 256}
BARS = {"replay": 1e-5, "formula": 1e-5, "output_rse": 1e-8,
        "tangent_cosine": .90, "tangent_direction": .875,
        "top64_recovery": .75, "top64_cosine": .95,
        "top128_recovery": .85, "top128_cosine": .95,
        "direction": .875, "control_ratio": .25}
PREDICTION_KEYS = (
    "pred_a_authority_gradient_and_formula_instruments_hooks_finiteness_and_price",
    "pred_b_summed_native_factor_tangent_predicts_full_M11_effect_on_both_selection_constructions",
    "pred_c_stable_task_tangent_top64_transfers_to_v18_A2",
    "pred_d_stable_task_tangent_top128_recovers_most_v18_A2",
    "pred_e_task_tangent_factor_program_is_selective_and_v19_remains_sealed",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def full_logits(backend, batch):
    torch, F, model = backend.torch, backend.F, backend.model
    tokens, lengths = backend._tensor_batch(batch)
    x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(
        x, (model.config.n_embd,))) / 30.0)
    return logits, lengths


def margin_from_logits(backend, batch, logits, lengths):
    torch = backend.torch
    rows = torch.arange(len(lengths), device=backend.device)
    positions = torch.tensor([length - 1 for length in lengths], device=backend.device)
    answer = torch.tensor(batch.answer_ids, device=backend.device)
    foil = torch.tensor(batch.foil_ids, device=backend.device)
    return logits[rows, positions, foil] - logits[rows, positions, answer]


def selection_tangent(backend, rows):
    torch, model = backend.torch, backend.model
    batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    positions = tuple(tuple(range(int(query) + 1)) for query in batch.semantic_positions)
    base_source, donor_source = transfer.suffix_position_rows(batch, donor_batch)
    calls = {"leaf": 0}
    with torch.no_grad():
        native_bundle, base_heads, calls["native_source"] = parent.capture_source_heads(
            lambda: output_greedy.capture_m11(lambda: parent.full_forward(backend, batch), model),
            model, batch)
        _native, absent_readers, _absent_outputs, calls["native_sites"] = native_bundle
        _donor, donor_heads, calls["donor_source"] = parent.capture_source_heads(
            lambda: parent.full_forward(backend, donor_batch), model, donor_batch)

    saved = {}
    def m11_pre(_module, arguments):
        saved["reader"] = arguments[0].detach().clone()
    def m11_post(_module, _arguments, output):
        calls["leaf"] += 1
        leaf = output.detach().requires_grad_(True)
        saved["leaf"] = leaf
        return leaf
    pre = model.transformer.h[11].mlp.register_forward_pre_hook(m11_pre)
    post = model.transformer.h[11].mlp.register_forward_hook(m11_post)
    try:
        (logits, lengths), calls["writer_source"] = transfer.with_source_heads_aligned(
            lambda: full_logits(backend, batch), model, batch, base_heads, donor_heads,
            base_source, donor_source)
    finally:
        pre.remove(); post.remove()
    margin = margin_from_logits(backend, batch, logits, lengths)
    margin.sum().backward()
    if calls["leaf"] != 1 or saved["leaf"].grad is None:
        raise RuntimeError("M11 task-tangent boundary gradient was not live")
    gradient = saved["leaf"].grad.detach().float()
    present_reader = saved["reader"].float()
    absent_reader = absent_readers["M11"].float()
    mlp = model.transformer.h[11].mlp
    hidden_delta = ((present_reader @ mlp.Left.weight.detach().float().T)
                    * (present_reader @ mlp.Right.weight.detach().float().T)
                    - (absent_reader @ mlp.Left.weight.detach().float().T)
                    * (absent_reader @ mlp.Right.weight.detach().float().T))
    factor_tangent = hidden_delta * (gradient @ mlp.Down.weight.detach().float())
    unit_mean = torch.cat([factor_tangent[row, list(row_positions)]
                           for row, row_positions in enumerate(positions)], dim=0).mean(dim=0)
    tangent_rows = torch.stack([factor_tangent[row, list(row_positions)].sum()
                                for row, row_positions in enumerate(positions)])
    with torch.no_grad():
        absent, calls["M11_absent"] = transfer.run_tensor_arm(
            backend, batch, base_heads, donor_heads, base_source, donor_source, positions,
            {"M11": absent_reader})
    actual = margin.detach().float().cpu().numpy() - parent.toward_donor_margin(absent)
    report = parent.metrics(tangent_rows.detach().float().cpu().numpy(), actual)
    calls["native_sites"] = calls["native_sites"]
    return unit_mean.detach(), report, calls, float(gradient.abs().max())


def stable_order(means):
    torch = __import__("torch")
    normalized = [value / max(float(value.abs().sum()), 1e-30) for value in means]
    score = torch.stack(normalized).amin(dim=0)
    order = torch.argsort(score, descending=True, stable=True)
    return order, score


def selection_calls_ok(calls):
    return bool(calls["leaf"] == 1
        and all(set(calls[key].values()) == {1}
                for key in ("native_source", "donor_source", "writer_source"))
        and calls["native_sites"] == {"M11": {"reader": 1, "output": 1}}
        and calls["M11_absent"]["readers"] == {"M11": 1}
        and calls["M11_absent"]["output"] == 0
        and set(calls["M11_absent"]["source"].values()) == {1})


def main():
    paths = {"prior": PRIOR, "greedy_result": GREEDY_RESULT, "factor_result": FACTOR_RESULT,
             "v19_capability": V19_CAPABILITY, "greedy_runner": GREEDY_RUNNER,
             "builder17": BUILDER17, "builder18": BUILDER18}
    observed = {name: sha(path) for name, path in paths.items()}
    greedy = json.loads(GREEDY_RESULT.read_text()); v19 = json.loads(V19_CAPABILITY.read_text())
    rows17 = [row for row in fresh17.build_rows() if row["transform_id"] == "A2"]
    rows18 = {family: [row for row in fresh18.build_rows() if row["transform_id"] == family]
              for family in ("A1", "A2", "P", "C")}
    authority_ok = bool(observed == EXPECTED and greedy.get("terminal") == "nonselective_greedy"
        and v19.get("causal_outcomes_opened") is False and len(rows17) == 16
        and all(len(rows18[key]) == 16 for key in rows18))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "selection_rows": {"v17_A2": 16, "v18_A1": 16},
           "test_rows": {"A2": 16, "P": 16, "C": 16}, "prefixes": PREFIXES,
           "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("task-tangent factor authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters(): parameter.requires_grad_(False)
    means, selection_reports, selection_calls, gradient_max = [], {}, {}, {}
    for label, rows in (("v17_A2", rows17), ("v18_A1", rows18["A1"])):
        mean, selection_reports[label], selection_calls[label], gradient_max[label] = (
            selection_tangent(backend, rows))
        means.append(mean)
    order, scores = stable_order(means)
    reports, effects, instruments = {}, {}, {}
    with backend.torch.no_grad():
        for family in ("A2", "P", "C"):
            reports[family], panel_effects, instruments[family] = output_greedy.test_panel(
                backend, rows18[family], order)
            effects[family] = {key: value.tolist() for key, value in panel_effects.items()}
    target_rms = {label: transfer.rms(effects["A2"][label]) for label in ("top64", "top128")}
    control_ratios = {family: {label: transfer.rms(effects[family][label])
        / max(target_rms[label], 1e-30) for label in target_rms} for family in ("P", "C")}
    A = bool(authority_ok and all(selection_calls_ok(value) for value in selection_calls.values())
        and all(value > 0 and math.isfinite(value) for value in gradient_max.values())
        and all(output_greedy.calls_ok(value["calls"]) for value in instruments.values())
        and all(value["writer_replay_max_abs_margin_error"] <= BARS["replay"]
                and value["all_hidden_margin_max_abs_error"] <= BARS["formula"]
                and value["all_hidden_output_relative_squared_error"] <= BARS["output_rse"]
                for value in instruments.values())
        and finite({"selection": selection_reports, "reports": reports, "controls": control_ratios})
        and PRICE["model_forwards"] == 8 + 3 * 11
        and PRICE["sequence_evaluations"] == PRICE["model_forwards"] * 16)
    B = all(value["cosine"] >= BARS["tangent_cosine"]
        and value["direction_agreement"] >= BARS["tangent_direction"]
        for value in selection_reports.values())
    C = bool(reports["A2"]["top64"]["signed_recovery"] >= BARS["top64_recovery"]
        and reports["A2"]["top64"]["cosine"] >= BARS["top64_cosine"]
        and reports["A2"]["top64"]["direction_agreement"] >= BARS["direction"])
    D = bool(reports["A2"]["top128"]["signed_recovery"] >= BARS["top128_recovery"]
        and reports["A2"]["top128"]["cosine"] >= BARS["top128_cosine"]
        and reports["A2"]["top128"]["direction_agreement"] >= BARS["direction"])
    E = all(ratio <= BARS["control_ratio"] for family in control_ratios
            for ratio in control_ratios[family].values())
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid_instrument" if not A or not B else "nonselective_task_tangent" if not E else
        "compact_task_read_factor_program" if C else "broad_task_read_factor_program" if D else
        "construction_conditioned_task_covector")
    result = {"schema": "temporal_iswas_m11_crossconstruction_task_tangent_factor_order_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "stable_factor_order_first256": order[:256].detach().cpu().tolist(),
        "stable_scores_first256": scores[order[:256]].detach().cpu().tolist(),
        "selection_tangent_reports": selection_reports, "selection_gradient_max_abs": gradient_max,
        "selection_calls": selection_calls, "test_reports": reports,
        "test_effect_vectors": effects, "test_instrument": instruments,
        "control_normalized_rms_ratios": control_ratios, "v19_causal_outcomes_opened": False,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("selection_tangent_reports",
        "selection_gradient_max_abs", "test_reports", "control_normalized_rms_ratios",
        "test_instrument", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
