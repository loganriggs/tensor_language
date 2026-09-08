#!/usr/bin/env python3
"""Projective entry-12 response-shape router from v15 to frozen v16."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_shape_symmetry_finiteness_and_exact_price pred_b_v15_leave_one_group_out_is_selective pred_c_v16_generalizes_under_both_oracle_folds pred_d_each_v16_target_is_covered pred_e_oracle_folds_agree_on_v16
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, os, time
from pathlib import Path

from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import entry12_response_shape_router_contract as shape
import run_temporal_iswas_v15_multidirection_contextual_gram_router_v16_v1 as base

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_response_shape_router_v16_v1.json"
GRAM_NULL = ROOT / "circuits/followups/temporal_iswas_v15_multidirection_contextual_gram_router_v16_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_response_shape_router_v16_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_response_shape_router_v16_v1"
EXPECTED = {
    "prior": "dfcef141a79616dfabcc2f112a090bdb64135dcd754ce463472e23798ed56bfd",
    "gram_null": "c5a8a707ddfd7ecea7a0f03a0ef8b028d03f1228faf7564d202b85586057b7a7",
    "shape": "c0b60fb80ca90f63dcc949f0af7f0ffb1e21d338b614fd487dbae6cd92587d3f",
    "base_runner": "2a6a245084826bd8242b326ec5c90022977cac4a7622e4d23a1eaa5653f12a05",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "v16_capability": "a1c2baf0bd9548e189ccc4ba4d11c4905f85af1ff7013d408c143f6f2a6434e3",
    "v15": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "v16": "5b1cb38cc62b5682c03505a5090e962d4773015efb44f8fad145515f5066b549",
}
FILES = {
    "prior": PRIOR, "gram_null": GRAM_NULL,
    "shape": ROOT / "ops/entry12_response_shape_router_contract.py",
    "base_runner": ROOT / "ops/run_temporal_iswas_v15_multidirection_contextual_gram_router_v16_v1.py",
    "oracles": base.ORACLES, "v16_capability": base.V16_CAPABILITY,
    "v15": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
    "v16": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16.py",
}
PRICE = {"native_capture_forwards": 4, "differentiable_transformer_forwards": 11,
         "transformer_backward_forwards": 0, "model_updates": 0,
         "example_evaluations": 656, "fit_parameters": 70}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def logo_report(torch, rows, features, labels):
    predicted = torch.empty_like(labels)
    groups = torch.tensor([int(row["group_number"]) for row in rows], device=labels.device)
    for held in range(16):
        train, test = groups != held, groups == held
        fitted = shape.fit(torch, features[train], labels[train])
        predicted[test] = shape.predict(torch, features[test], fitted)[0]
    return base.classify(torch, rows, labels, predicted)


def capture_features(backend, context, counters, bases_by_panel):
    _off_output, off_states = base.state_executor.execute(
        base.parent, backend, context, counters, {}, capture=True)
    off = off_states["entry12"]; experts = {}
    for panel in ("A1", "A2"):
        _output, states = base.state_executor.execute(
            base.parent, backend, context, counters, bases_by_panel[panel], capture=True)
        experts[panel] = states["entry12"]
    features = shape.features(backend.torch, off, experts, context["base_batch"].semantic_positions)
    reversed_experts = {name: off - (state - off) for name, state in experts.items()}
    reversed_features = shape.features(
        backend.torch, off, reversed_experts, context["base_batch"].semantic_positions)
    return features, float((features - reversed_features).abs().max())


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior, gram_null, oracles, capability = (
        json.loads(path.read_text()) for path in (PRIOR, GRAM_NULL, base.ORACLES, base.V16_CAPABILITY))
    authority = bool(observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
                     and gram_null.get("terminal") == "contextual_gram_in_distribution_null"
                     and oracles.get("terminal") == "construction_conditioned_coordinate"
                     and capability.get("terminal") == "manifest")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "feature_dimension": 14, "v16_C_excluded": True, "price": PRICE}
    if not authority: raise RuntimeError(f"authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)

    started = time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters(): parameter.requires_grad_(False)
    counters = {key: 0 for key in PRICE}; native = backend.native
    def counted(batch, *, capture):
        counters["native_capture_forwards"] += 1; counters["example_evaluations"] += len(batch.row_ids)
        return native(batch, capture=capture)
    backend.native = counted; torch = backend.torch
    rows15, rows16_all = base.v15.build_rows(), base.v16.build_rows()
    rows16 = [row for row in rows16_all if row["transform_id"] in ("A1", "A2", "P")]
    excluded16 = [row for row in rows16_all if row["transform_id"] == "C"]
    alignment15 = derive_full_sequence_alignment_contract(rows15, required_panels=("A1", "A2", "P", "C"))
    alignment16 = derive_full_sequence_alignment_contract(rows16, required_panels=("A1", "A2", "P"))

    bank15 = base.parent.capture_bank(backend, rows15, counters, factors=False)
    features15_parts, rows15_ordered, sign_errors = [], [], []
    for parity in (0, 1):
        context = base.parent.subset_context(bank15, base.parent.row_indices(rows15, parity=parity))
        bases = {panel: base.dependency.bases_for_evaluation(
            torch, backend.device, oracles, expert)[parity]
                 for expert, panel in base.EXPERTS.items()}
        values, error = capture_features(backend, context, counters, bases)
        features15_parts.append(values); sign_errors.append(error); rows15_ordered.extend(context["rows"])
    features15 = torch.cat(features15_parts)
    labels15 = base.finite_router.labels_for_rows(torch, rows15_ordered, device=backend.device)
    logo = logo_report(torch, rows15_ordered, features15, labels15)
    fitted = shape.fit(torch, features15, labels15)

    bank16 = base.parent.capture_bank(backend, rows16, counters, factors=False)
    context16 = base.parent.subset_context(bank16, range(len(rows16)))
    _off_output, off_states = base.state_executor.execute(
        base.parent, backend, context16, counters, {}, capture=True)
    off16 = off_states["entry12"]; fold_reports, fold_predictions = {}, []
    for fold in (0, 1):
        expert_states = {}
        for expert, panel in base.EXPERTS.items():
            bases = base.dependency.bases_for_evaluation(torch, backend.device, oracles, expert)[fold]
            _output, states = base.state_executor.execute(
                base.parent, backend, context16, counters, bases, capture=True)
            expert_states[panel] = states["entry12"]
        values = shape.features(torch, off16, expert_states, context16["base_batch"].semantic_positions)
        reversed_values = shape.features(torch, off16, {
            name: off16 - (state - off16) for name, state in expert_states.items()},
            context16["base_batch"].semantic_positions)
        sign_errors.append(float((values - reversed_values).abs().max()))
        predicted, scores = shape.predict(torch, values, fitted)
        truth = base.finite_router.labels_for_rows(torch, rows16, device=backend.device)
        fold_reports[str(fold)] = {**base.classify(torch, rows16, truth, predicted),
                                   "features": values.cpu().tolist(), "scores": scores.cpu().tolist()}
        fold_predictions.append(predicted)

    counters["fit_parameters"] = 70
    row_ids_disjoint = set(row["row_id"] for row in rows15).isdisjoint(row["row_id"] for row in rows16_all)
    c_excluded = all(len(row["base_ids"]) != len(row["donor_ids"]) for row in excluded16)
    sign_error = max(sign_errors)
    A = bool(authority and base.v15.validate_rows(rows15) == base.V15_ROWS_SHA256
             and base.v16.validate_rows(rows16_all) == base.V16_ROWS_SHA256
             and alignment15["panel_counts"] == {panel: 16 for panel in ("A1", "A2", "P", "C")}
             and alignment16["panel_counts"] == {panel: 16 for panel in ("A1", "A2", "P")}
             and len(rows15_ordered) == 64 and len(rows16) == 48 and c_excluded and row_ids_disjoint
             and features15.shape == (64, 14) and sign_error <= 1e-5
             and base.finite({"features15": features15.cpu().tolist(), "logo": logo,
                              "fold_reports": fold_reports}) and counters == PRICE)
    B = logo["macro_accuracy"] >= .95 and logo["control_predicted_target_count"] == 0
    C = all(fold_reports[str(fold)]["macro_accuracy"] >= .75
            and fold_reports[str(fold)]["control_predicted_target_count"] == 0 for fold in (0, 1))
    D = all(fold_reports[str(fold)]["confusion"][0][0] >= 12
            and fold_reports[str(fold)]["confusion"][1][1] >= 12 for fold in (0, 1))
    E = bool(torch.equal(fold_predictions[0], fold_predictions[1]))
    predictions = dict(zip((
        "pred_a_authority_alignment_shape_symmetry_finiteness_and_exact_price",
        "pred_b_v15_leave_one_group_out_is_selective",
        "pred_c_v16_generalizes_under_both_oracle_folds",
        "pred_d_each_v16_target_is_covered",
        "pred_e_oracle_folds_agree_on_v16"), map(bool, (A, B, C, D, E))))
    terminal = ("invalid" if not A else "response_shape_router_candidate" if all((B, C, D, E))
                else "response_shape_in_distribution_null" if not B else "response_shape_ood_null")
    result = {"schema": "temporal_iswas_v15_response_shape_router_v16_result_v1",
              "candidate_id": CANDIDATE_ID, "started_utc": datetime.now(timezone.utc).isoformat(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "alignment_contracts": {"v15": alignment15, "v16_without_c": alignment16},
              "v15_logo": logo, "v15_features": features15.cpu().tolist(),
              "v16_by_oracle_fold": fold_reports,
              "instrument": {"v15_v16_row_ids_disjoint": row_ids_disjoint,
                             "v16_c_explicitly_excluded": c_excluded,
                             "response_sign_invariance_max_abs_error": sign_error,
                             "oracle_fold_predictions_agree": E},
              "predictions": predictions, "terminal": terminal, "price": counters,
              "v16_scope": "OOD_TEXT_REUSE_NEW_INTERVENTION"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "v15_logo", "v16_by_oracle_fold", "instrument", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__": main()
