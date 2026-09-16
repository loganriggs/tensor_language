#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_token_extraction_instrument pred_b_original_authority_prediction pred_c_fresh_authority_prediction
"""Verify the reusable token-input sparse graph on original and fresh authorities."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, os, signal, time
from pathlib import Path
import numpy as np

import circuit_fast_screen_candidate_subject_number_embedding_context_crossed_corrected_v1 as original_authority
import circuit_fast_screen_candidate_subject_number_rank1_fresh_confirmation as fresh_authority
import circuit_fast_screen_managed_runner as managed
import run_subject_number_l11h3_expanded_behavioral_graph_v2 as parent
import run_subject_number_l11h3_subject_value_upstream_mobius_v1 as base
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import subject_number_sparse_graph_token_extraction_v1 as component

RUNNER = Path(__file__).resolve(); ROOT = RUNNER.parents[3]; POLY = ROOT / "basis_aligned/polynomial_causal"
DECODER = base.DECODER
PARENT_RESULT = POLY / "SUBJECT_NUMBER_L11H3_EXPANDED_BEHAVIORAL_GRAPH_V2_RESULT.json"
PREREG = POLY / "SUBJECT_NUMBER_SPARSE_GRAPH_TOKEN_EXTRACTION_V1_PREREGISTRATION.md"
BINDING = POLY / "SUBJECT_NUMBER_SPARSE_GRAPH_TOKEN_EXTRACTION_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_SPARSE_GRAPH_TOKEN_EXTRACTION_V1_RESULT.json"
BARS = {"maximum_aggregation_error": 1e-10, "maximum_fresh_relative_l2": .15,
        "maximum_gauge_correction_relative_l2": 1e-5, "maximum_native_replay_error": 1e-5,
        "maximum_original_relative_l2": .10, "minimum_aligned_recovery": 0.,
        "minimum_fresh_cosine": .98, "minimum_native_accuracy": .75,
        "minimum_original_cosine": .99}
PRICE = {"component_calls": 4, "component_sequences": 160, "partial_forwards": 8,
         "partial_sequences": 320, "predictive_suffix_corners": 40, "target_suffix_corners": 4,
         "suffix_sequences": 1760, "native_forwards": 4, "native_sequences": 160,
         "external_activation_inputs": 0, "internally_derived_ports": 5, "edges": 9,
         "fits": 0, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_token_extraction_instrument": None,
    "pred_b_original_authority_prediction": None, "pred_c_fresh_authority_prediction": None}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"component": Path(component.__file__), "original_authority": Path(original_authority.__file__),
             "fresh_authority": Path(fresh_authority.__file__), "parent_runner": Path(parent.__file__),
             "parent_result": PARENT_RESULT, "decoder": DECODER, "preregistration": PREREG}
    if binding["files"] != {k: sha(v) for k, v in paths.items()} \
            or binding["original_authority_sha256"] != original_authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["fresh_authority_sha256"] != fresh_authority.EXPECTED_AUTHORITY_SHA256 \
            or binding["bars"] != BARS or binding["masks"] != list(component.MASKS) or binding["price"] != PRICE:
        raise ValueError("binding changed")
    decoder, parent_result = (json.loads(p.read_text()) for p in (DECODER, PARENT_RESULT))
    if decoder["terminal"] != "embedding_number_decoder_frozen" or parent_result["terminal"] != "expanded_sparse_behavioral_graph" \
            or parent_result["graph"]["selected_masks"][:9] != list(component.MASKS):
        raise ValueError("parent status or masks changed")
    return binding, decoder, parent_result, original_authority.build_rows(), fresh_authority.build_rows()

def original_panels(rows):
    return {"discovery": [i for i, r in enumerate(rows) if r["pair_index"] < 8 and r["template_id"] in ("near", "behind")],
        "context_ood": [i for i, r in enumerate(rows) if r["pair_index"] < 8 and r["template_id"] in ("under", "above")],
        "lexical_ood": [i for i, r in enumerate(rows) if r["pair_index"] >= 8 and r["template_id"] in ("near", "behind")],
        "joint_ood": [i for i, r in enumerate(rows) if r["pair_index"] >= 8 and r["template_id"] in ("under", "above")]}

def fresh_cells(rows):
    result = {}
    for direction in ("singular_to_plural", "plural_to_singular"):
        for template in ("behind_beside", "among_behind"):
            result[f"{direction}|{template}"] = [i for i, r in enumerate(rows)
                if r["direction_id"] == direction and r["template_id"] == template]
    return result

def metric(prediction, target):
    yn = np.linalg.norm(target); pn = np.linalg.norm(prediction)
    return {"rows": len(target), "target_rms": float(np.sqrt(np.mean(target ** 2))),
            "prediction_rms": float(np.sqrt(np.mean(prediction ** 2))),
            "relative_l2": float(np.linalg.norm(target - prediction) / max(yn, 1e-30)),
            "cosine": float(prediction @ target / max(pn * yn, 1e-30)),
            "aligned_recovery": float(prediction @ target / max(target @ target, 1e-30))}

def plan():
    _, decoder, parent_result, original_rows, fresh_rows = load_bound()
    return {"schema": "subject_number_sparse_graph_token_extraction_v1_plan", "model_loaded": False,
            "gpu_accessed": False, "queue_touched": False, "original_rows": len(original_rows), "fresh_rows": len(fresh_rows),
            "public_data_inputs": ["token_ids", "subject_positions", "answer_ids"], "external_activation_inputs": 0,
            "ports": list(component.PORTS), "masks": list(component.MASKS), "predictive_corners": list(component.PREDICTIVE_CORNERS),
            "parent_terminal": parent_result["terminal"], "bars": BARS, "price": PRICE,
            "binding_sha256": sha(BINDING), "decoder_axis_sha256": decoder["frozen_decoder"]["axis_float32_sha256"]}

def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, indent=2, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    signal.alarm(1200); binding, decoder, parent_result, original_rows, fresh_rows = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies(); torch.set_num_threads(2)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    started = time.perf_counter(); device = next(model.parameters()).device
    decoder_axis = torch.tensor(decoder["frozen_decoder"]["axis"], dtype=torch.float64, device=device)
    threshold = float(decoder["frozen_decoder"]["threshold"])
    counts = {"component_calls": 0, "component_sequences": 0, "partial_forwards": 0, "partial_sequences": 0,
              "predictive_suffix_corners": 0, "target_suffix_corners": 0, "suffix_sequences": 0,
              "native_forwards": 0, "native_sequences": 0, "external_activation_inputs": 0,
              "internally_derived_ports": len(component.PORTS), "edges": len(component.MASKS),
              "fits": 0, "backwards": 0, "parameter_updates": 0}
    max_replay = max_aggregation = max_gauge = 0.

    def run_batches(rows, batch_ids, endpoint_fn):
        nonlocal max_replay, max_aggregation, max_gauge
        predictions = np.zeros(len(rows), dtype=np.float64); targets = np.zeros(len(rows), dtype=np.float64)
        natives = np.zeros(len(rows), dtype=np.float64); atom_values = np.zeros((len(component.MASKS), len(rows)), dtype=np.float64)
        for ids in batch_ids:
            batch_rows = [rows[i] for i in ids]; endpoints = [endpoint_fn(r) for r in batch_rows]
            tokens = torch.tensor([e["token_ids"] for e in endpoints], dtype=torch.long, device=device)
            positions = torch.tensor([e["subject_position"] for e in endpoints], dtype=torch.long, device=device)
            answers = torch.tensor([[e["answer_id"], e["foil_id"]] for e in endpoints], dtype=torch.long, device=device)
            result = component.evaluate(model, tokens, positions, answers, decoder_axis, threshold, torch, F, include_target=True)
            native = component.native_margin(model, tokens, positions, answers, torch, F)
            max_replay = max(max_replay, float((native - result["base_margin"]).abs().max()))
            max_aggregation = max(max_aggregation, result["audit"]["aggregation_max_abs_error"])
            max_gauge = max(max_gauge, result["audit"]["gauge_correction_max_relative_l2"])
            predictions[ids] = result["prediction"].cpu().numpy(); targets[ids] = result["target"].cpu().numpy()
            natives[ids] = native.cpu().numpy(); atom_values[:, ids] = result["atoms"].cpu().numpy()
            n = len(ids); counts["component_calls"] += 1; counts["component_sequences"] += n
            counts["partial_forwards"] += result["audit"]["partial_forwards"]; counts["partial_sequences"] += 2 * n
            counts["predictive_suffix_corners"] += result["audit"]["predictive_corners"]; counts["target_suffix_corners"] += 1
            counts["suffix_sequences"] += result["audit"]["suffix_forwards"] * n
            counts["native_forwards"] += 1; counts["native_sequences"] += n
        return predictions, targets, natives, atom_values

    original_batches = [[i for i, r in enumerate(original_rows) if r["template_id"] in names]
                        for names in (("near", "behind"), ("under", "above"))]
    answer_ids = {"singular": original_authority.task14.ENCODING.encode(" is")[0],
                  "plural": original_authority.task14.ENCODING.encode(" are")[0]}
    def original_endpoint(row):
        return {"token_ids": row["token_ids"], "subject_position": row["subject_position"],
                "answer_id": row["native_answer_id"], "foil_id": answer_ids["plural" if row["number"] == "singular" else "singular"]}
    op, ot, on, oa = run_batches(original_rows, original_batches, original_endpoint)
    fresh_batches = [[i for i, r in enumerate(fresh_rows) if r["template_id"] == template]
                     for template in ("behind_beside", "among_behind")]
    def fresh_endpoint(row):
        e = row["endpoints"]["recipient"]
        return {"token_ids": e["ids"], "subject_position": row["subject_position"],
                "answer_id": e["answer_id"], "foil_id": e["foil_id"]}
    fp, ft, fn, fa = run_batches(fresh_rows, fresh_batches, fresh_endpoint)
    original_groups = original_panels(original_rows); fresh_groups = fresh_cells(fresh_rows)
    original_reports = {name: metric(op[ids], ot[ids]) for name, ids in original_groups.items()}
    fresh_reports = {name: metric(fp[ids], ft[ids]) for name, ids in fresh_groups.items()}
    original_capability = {name: float(np.mean(on[ids] > 0)) for name, ids in original_groups.items()}
    fresh_capability = {name: float(np.mean(fn[ids] > 0)) for name, ids in fresh_groups.items()}
    finite = bool(np.isfinite(np.asarray([max_replay, max_aggregation, max_gauge, *op, *ot, *fp, *ft, *oa.ravel(), *fa.ravel()])).all())
    pred_a = bool(finite and max_replay <= BARS["maximum_native_replay_error"]
        and max_aggregation <= BARS["maximum_aggregation_error"] and max_gauge <= BARS["maximum_gauge_correction_relative_l2"]
        and all(v >= BARS["minimum_native_accuracy"] for v in (*original_capability.values(), *fresh_capability.values()))
        and counts == PRICE and checkpoint.weights_sha256 == decoder["checkpoint_weights_sha256"])
    pred_b = bool(pred_a and all(p["relative_l2"] <= BARS["maximum_original_relative_l2"]
        and p["cosine"] >= BARS["minimum_original_cosine"] and p["aligned_recovery"] > BARS["minimum_aligned_recovery"]
        for p in original_reports.values()))
    pred_c = bool(pred_a and all(p["relative_l2"] <= BARS["maximum_fresh_relative_l2"]
        and p["cosine"] >= BARS["minimum_fresh_cosine"] and p["aligned_recovery"] > BARS["minimum_aligned_recovery"]
        for p in fresh_reports.values()))
    predictions = dict(zip(PREDICTION_REGISTRY, (pred_a, pred_b, pred_c)))
    terminal = "token_input_sparse_graph_extracted" if all(predictions.values()) else "valid_token_extraction_test" if pred_a else "invalid"
    result = {"schema": "subject_number_sparse_graph_token_extraction_v1_result", "terminal": terminal,
        "predictions": predictions, "instrument": {"finite": finite, "native_replay_max_abs_error": max_replay,
            "aggregation_max_abs_error": max_aggregation, "gauge_correction_max_relative_l2": max_gauge,
            "original_native_capability": original_capability, "fresh_native_capability": fresh_capability, "counts": counts},
        "component": {"public_data_inputs": ["token_ids", "subject_positions", "answer_ids"],
            "external_activation_inputs": 0, "internally_derived_ports": list(component.PORTS),
            "masks": list(component.MASKS), "terms": list(component.TERMS),
            "predictive_corners": list(component.PREDICTIVE_CORNERS), "coefficient_fits": 0},
        "original_authority": {"rows": len(original_rows), "reports": original_reports,
            "term_rms": np.sqrt(np.mean(oa ** 2, axis=1)).tolist()},
        "fresh_authority": {"rows": len(fresh_rows), "reports": fresh_reports,
            "term_rms": np.sqrt(np.mean(fa ** 2, axis=1)).tolist()},
        "bars": BARS, "price": PRICE, "original_authority_sha256": original_authority.canonical(original_rows),
        "fresh_authority_sha256": fresh_authority.canonical_sha256(fresh_rows), "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER), "component_sha256": sha(component.__file__),
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "wall_seconds": time.perf_counter() - started,
        "scope": "Reusable token-input sparse graph verified on original crossed and disjoint fresh authorities; no external activations or fits."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("terminal", "predictions", "instrument", "component", "original_authority", "fresh_authority")}, indent=2, sort_keys=True))
    assert pred_a

if __name__ == "__main__": main()
