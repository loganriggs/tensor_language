#!/usr/bin/env python3
"""Extract the exact native-routing L11H3 value-source tensor."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_native_replay_capture_coverage_finiteness_and_exact_price pred_b_exact_bilinear_formula_equals_value_patch_head_delta pred_c_direct_head_interface_install_reproduces_downstream_effect pred_d_extracted_source_term_transfers_original_and_ood pred_e_extracted_interface_is_task_selective_and_causal
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import dual_command_head_module_factorial_contract as accounting
import run_temporal_iswas_l11h3_value_source_region_localization_v1 as loc


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_l11h3_native_routing_source_term_extraction_v1.json"
QK_RESULT = ROOT / "circuits/followups/temporal_iswas_l11h3_task_typed_source_qk_factorial_v2_result.json"
LOCALIZATION = ROOT / "circuits/followups/temporal_iswas_l11h3_value_source_region_localization_v1_result.json"
AUDIT = ROOT / "circuits/followups/temporal_iswas_l11h3_value_source_region_localization_v1_replay_audit_result.json"
ORIGINAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_v2.py"
OOD_BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_ood_v1.py"
ACCOUNTING = ROOT / "ops/dual_command_head_module_factorial_contract.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_l11h3_native_routing_source_term_extraction_v1_result.json"
EXPECTED = {
    "prior": "6ca52605e16761b8531ee769b01c3d45802fb132e224c2609583ede1d0660075",
    "qk_result": "a4aaaaf49604c61b33bac29bb728f558b2c3c6c2ccf9a79970047fb4651be556",
    "localization": "74d7e51e227c9799bb6e079edff1fb8b5c49e9e6725960958e0269e19d74882e",
    "audit": "04ae7f940a4677b71801cfc2d3d40f60911fa52007ece2561796f66c1f54372f",
    "original_builder": "72da11860ce5bf1c03ea126bc10fcb6c46dd2648169edb00a1e9e1dbeee8a0d0",
    "ood_builder": "6062f2b8dfaa54da71477b43cb9fc63650db389c5883c9e64125f0a200afeb5a",
    "accounting": "985b824c7d3d8b622dfb92e3522f107e15e5e095b1f40a59a46122ad9af68496",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
LOCKED = {"temporal": "bridge", "iswas": "postcue"}
PRICE = {"checkpoint_loads": 1, "model_forwards": 10, "sequence_evaluations": 1280,
         "scored_token_positions": 2560, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_pairing_native_replay_capture_coverage_finiteness_and_exact_price",
    "pred_b_exact_bilinear_formula_equals_value_patch_head_delta",
    "pred_c_direct_head_interface_install_reproduces_downstream_effect",
    "pred_d_extracted_source_term_transfers_original_and_ood",
    "pred_e_extracted_interface_is_task_selective_and_causal",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def patch_value(output, donor, pairs, position_rows, head=3):
    return loc.patch_value_tensor(output, donor, pairs, position_rows, head=head)


def add_head_delta(output, delta, head=3):
    if output.ndim != 3 or delta.shape != (*output.shape[:2], output.shape[-1] // 9):
        raise ValueError("direct head delta shape changed")
    width = output.shape[-1] // 9
    changed = output.clone()
    changed[..., head * width:(head + 1) * width] += delta.to(output.dtype)
    return changed


def _forward(backend, tokens, *, capture=False, donor_v=None, pairs=None,
             value_positions=None, direct_delta=None):
    torch, F, model = backend.torch, backend.F, backend.model
    factors = ("q", "k", "q2", "k2", "v")
    saved, handles = {}, []
    calls = {name: 0 for name in factors}
    proj_calls = 0

    def factor_hook(name):
        def hook(_module, _inputs, output):
            calls[name] += 1
            if capture:
                saved[name] = output.detach().clone()
            if name == "v" and donor_v is not None:
                return patch_value(output, donor_v, pairs, value_positions)
            return None
        return hook

    def projection_pre(_module, arguments):
        nonlocal proj_calls
        proj_calls += 1
        output = arguments[0]
        if capture:
            saved["head"] = output.detach().clone()
        if direct_delta is not None:
            output = add_head_delta(output, direct_delta)
            return (output,) + tuple(arguments[1:])
        return None

    attention = model.transformer.h[11].attn
    for name in factors:
        handles.append(getattr(attention, f"c_{name}").register_forward_hook(factor_hook(name)))
    handles.append(attention.c_proj.register_forward_pre_hook(projection_pre))
    try:
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, first = x, None
        for block in model.transformer.h:
            x, first = block(x, first, x0)
        logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)).float()
    finally:
        for handle in handles:
            handle.remove()
    if set(calls.values()) != {1} or proj_calls != 1:
        raise RuntimeError("L11 extraction hook coverage changed")
    if capture and set(saved) != {*factors, "head"}:
        raise RuntimeError("L11 extraction capture set changed")
    return logits, saved


def native_routing_delta(captures, attention, pairs, source_rows, torch, F, head=3):
    raw = {name: captures[name].view(captures[name].shape[0], captures[name].shape[1], 9, 128)
           for name in ("q", "k", "q2", "k2", "v")}
    cos, sin = attention.rotary(raw["q"])
    apply_rotary = sys.modules[type(attention).__module__].apply_rotary_emb
    q = apply_rotary(F.rms_norm(raw["q"], (128,)), cos, sin)
    k = apply_rotary(F.rms_norm(raw["k"], (128,)), cos, sin)
    q2 = apply_rotary(F.rms_norm(raw["q2"], (128,)), cos, sin)
    k2 = apply_rotary(F.rms_norm(raw["k2"], (128,)), cos, sin)
    score = torch.einsum("btd,bsd->bts", q[:, :, head], k[:, :, head]) / 128
    score2 = torch.einsum("btd,bsd->bts", q2[:, :, head], k2[:, :, head]) / 128
    pattern = score * score2
    causal = torch.tril(torch.ones(pattern.shape[1:], dtype=torch.bool, device=pattern.device))
    pattern = pattern.masked_fill(~causal, 0)
    delta_v = raw["v"][pairs, :, head] - raw["v"][:, :, head]
    source_mask = torch.zeros(delta_v.shape[:2], dtype=torch.bool, device=delta_v.device)
    for index, positions in enumerate(source_rows):
        source_mask[index, list(positions)] = True
    delta_v = delta_v * source_mask.unsqueeze(-1)
    delta_v = (1 - attention.lamb) * delta_v
    delta = torch.einsum("bts,bsd->btd", pattern, delta_v)
    return delta, pattern, source_mask


def selected_logits(logits, endpoints):
    values = []
    for index, (_row, _cell, endpoint) in enumerate(endpoints):
        for role in LOCKED:
            position = endpoint[f"{role}_position"] - 1
            positive, negative = ((loc.original._single(" will"), loc.original._single(" had"))
                if role == "temporal" else
                (loc.original._single(" is"), loc.original._single(" was")))
            values.extend((float(logits[index, position, positive]),
                           float(logits[index, position, negative])))
    return np.asarray(values, dtype=np.float64)


def run_population(backend, authority, population, localization):
    rows = authority.build_rows()
    endpoints, lookup = loc.parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch, F = backend.torch, backend.F
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints],
                          dtype=torch.long, device=backend.device)
    pairs = {role: loc.parent.pair_indices(endpoints, lookup, role) for role in LOCKED}
    queries = {role: [endpoint[f"{role}_position"] - 1
                      for _row, _cell, endpoint in endpoints] for role in LOCKED}
    regions = {role: [loc.source_regions(endpoint["ids"],
                endpoints[pairs[role][index]][2]["ids"], queries[role][index])
                for index, (_row, _cell, endpoint) in enumerate(endpoints)] for role in LOCKED}
    with torch.no_grad():
        native_logits, native_capture = _forward(backend, tokens, capture=True)
    native = {role: loc.margins(native_logits, endpoints, role) for role in LOCKED}
    records, tensors, replay_errors, direct_logit_errors = [], {}, [], []
    direct_effects, direct_collateral = {}, {}
    for role, source in LOCKED.items():
        source_rows = [item[source] for item in regions[role]]
        with torch.no_grad():
            patched_logits, patched_capture = _forward(backend, tokens, capture=True,
                donor_v=native_capture["v"], pairs=pairs[role], value_positions=source_rows)
            formula, pattern, source_mask = native_routing_delta(native_capture,
                backend.model.transformer.h[11].attn, pairs[role], source_rows, torch, F)
            direct_logits, _ = _forward(backend, tokens, direct_delta=formula)
        width = native_capture["head"].shape[-1] // 9
        sl = slice(3 * width, 4 * width)
        observed = patched_capture["head"][..., sl] - native_capture["head"][..., sl]
        difference = formula.float() - observed.float()
        formula_abs = float(difference.abs().max())
        formula_rel = float(difference.norm() / max(float(observed.float().norm()), 1e-30))
        direct_logit_error = float(np.max(np.abs(selected_logits(direct_logits, endpoints)
                                                - selected_logits(patched_logits, endpoints))))
        direct_logit_errors.append(direct_logit_error)
        value_effect = loc.margins(patched_logits, endpoints, role) - native[role]
        direct_effect = loc.margins(direct_logits, endpoints, role) - native[role]
        other = "iswas" if role == "temporal" else "temporal"
        value_other = loc.margins(patched_logits, endpoints, other) - native[other]
        direct_other = loc.margins(direct_logits, endpoints, other) - native[other]
        direct_effects[role], direct_collateral[role] = direct_effect, direct_other
        query_tensor = torch.stack([formula[index, query] for index, query in enumerate(queries[role])])
        tensors[role] = {"shape": list(query_tensor.shape), "sha256_fp32_le": tensor_sha(query_tensor),
            "complete_tensor_l2_norm": float(formula.float().norm()),
            "query_tensor": query_tensor.detach().float().cpu().tolist(),
            "query_positions": queries[role], "source_positions": [list(item) for item in source_rows],
            "row_ids": [row["row_id"] for row in rows],
            "pattern_selected_abs_max": float((pattern * source_mask[:, None, :]).abs().max())}
        for phase in ("FIT", "HOLDOUT"):
            for template in ("ALL",) + authority.TEMPLATES:
                selected = np.asarray([row["phase"] == phase and
                    (template == "ALL" or row["template_id"] == template)
                    for row, _cell, _endpoint in endpoints])
                report = accounting.effect_metrics(direct_effect[selected], value_effect[selected],
                                                    direct_other[selected])
                gold = native[role][pairs[role]] - native[role]
                value_gold = accounting.effect_metrics(value_effect[selected], gold[selected],
                                                        value_other[selected])
                old = next(item for item in localization["panels"][population]["reports"]
                    if item["role"] == role and item["phase"] == phase
                    and item["template_id"] == template and item["arm"] == source)
                replay_errors.extend((abs(value_gold["signed_recovery"]
                                          - old["command_gold_signed_recovery"]),
                                      abs(value_gold["non_target_to_target_gold_norm"]
                                          - old["command_gold_non_target_norm_ratio"])))
                records.append({"population": population, "role": role, "phase": phase,
                    "template_id": template, "formula_head_max_abs_error": formula_abs,
                    "formula_head_relative_l2_error": formula_rel,
                    "direct_selected_logit_max_abs_error": direct_logit_error, **report,
                    "direct_command_gold_non_target_norm_ratio":
                        accounting.effect_metrics(direct_effect[selected], gold[selected],
                                                  direct_other[selected])[
                            "non_target_to_target_gold_norm"]})
    causal_zero = max(abs(value) for value in direct_collateral["iswas"])
    return {"records": records, "tensors": tensors, "causal_zero": float(causal_zero),
        "command_gold_replay_max_abs_error": max(replay_errors),
        "direct_selected_logit_max_abs_error": max(direct_logit_errors),
        "capture_shapes": {name: list(value.shape) for name, value in native_capture.items()},
        "coverage_ok": all(set(item[LOCKED[role]]) <= set(item["full_prefix"])
                           for role in LOCKED for item in regions[role])}


def main():
    paths = {"prior": PRIOR, "qk_result": QK_RESULT, "localization": LOCALIZATION,
             "audit": AUDIT, "original_builder": ORIGINAL_BUILDER,
             "ood_builder": OOD_BUILDER, "accounting": ACCOUNTING, "producer": PRODUCER}
    observed = {name: sha(path) for name, path in paths.items()}
    qk_result, localization, audit = (json.loads(path.read_text())
        for path in (QK_RESULT, LOCALIZATION, AUDIT))
    authority_ok = bool(observed == EXPECTED
        and qk_result.get("terminal") == "stable_routing_invariant"
        and audit.get("terminal") == "analysis_target_mismatch_repaired"
        and audit.get("recovered_v1_outcome", {}).get("selections") == LOCKED)
    dry = {"candidate_id": json.loads(PRIOR.read_text())["candidate_id"], "dryrun": True,
           "authority_ok": authority_ok, "gpu_accessed": False, "model_loaded": False,
           "queue_touched": False, "locked_sources": LOCKED, "price": PRICE}
    if not authority_ok:
        raise RuntimeError(f"source-term authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = loc.producer.Bilin18TorchBackend.load("cuda")
    panels = {"original": run_population(backend, loc.original, "original", localization),
              "ood": run_population(backend, loc.ood, "ood", localization)}
    records = [row for panel in panels.values() for row in panel["records"]]
    replay_max = max(panel["command_gold_replay_max_abs_error"] for panel in panels.values())
    formula_abs = max(row["formula_head_max_abs_error"] for row in records)
    formula_rel = max(row["formula_head_relative_l2_error"] for row in records)
    direct_logit = max(row["direct_selected_logit_max_abs_error"] for row in records)
    counters = PRICE.copy()
    capture_ok = all(panel["coverage_ok"] and set(panel["capture_shapes"])
        == {"q", "k", "q2", "k2", "v", "head"}
        and all(shape[0] == 128 and shape[2] == 1152
                for name, shape in panel["capture_shapes"].items() if name != "head")
        for panel in panels.values())
    A = bool(authority_ok and replay_max <= 1e-5 and capture_ok and counters == PRICE
             and finite(panels))
    B = bool(formula_abs <= 1e-4 and formula_rel <= 1e-4)
    C = bool(direct_logit <= 1e-4 and all(row["signed_recovery"] >= .999
        and row["signed_recovery"] <= 1.001 and row["cosine"] >= .99999
        and row["direction_agreement"] == 1.0 and row["relative_residual"] <= .001
        for row in records))
    D = bool(B and C and set(panels) == {"original", "ood"})
    E = bool(all(panel["causal_zero"] <= 1e-5 for panel in panels.values())
        and all(row["direct_command_gold_non_target_norm_ratio"] <= .01 for row in records
                if row["template_id"] == "ALL"))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = "native_routing_source_term_extracted" if all(predictions.values()) else (
        "invalid" if not A else "native_routing_source_term_null")
    result = {"schema": "temporal_iswas_l11h3_native_routing_source_term_extraction_result_v1",
        "candidate_id": json.loads(PRIOR.read_text())["candidate_id"],
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "command_gold_replay_max_abs_error": replay_max,
        "formula_head_max_abs_error": formula_abs,
        "formula_head_relative_l2_error": formula_rel,
        "direct_selected_logit_max_abs_error": direct_logit,
        "panels": panels, "predictions": predictions, "terminal": terminal, "price": counters}
    atomic_create_json(OUT, result)
    print(json.dumps({"command_gold_replay_max_abs_error": replay_max,
        "formula_head_max_abs_error": formula_abs,
        "formula_head_relative_l2_error": formula_rel,
        "direct_selected_logit_max_abs_error": direct_logit,
        "predictions": predictions, "terminal": terminal, "price": counters}, sort_keys=True))


if __name__ == "__main__":
    main()
