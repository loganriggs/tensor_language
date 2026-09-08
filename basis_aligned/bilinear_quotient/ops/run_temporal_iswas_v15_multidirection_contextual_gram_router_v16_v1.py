#!/usr/bin/env python3
"""Multi-direction v15 contextual Gram router evaluated on frozen v16."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_scope_finiteness_and_exact_price pred_b_v15_leave_one_group_out_is_selective pred_c_v16_generalizes_under_both_oracle_folds pred_d_each_v16_target_is_covered pred_e_oracle_folds_agree_on_v16
from __future__ import annotations

from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract
import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16 as v16
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import entry12_finite_router_contract as finite_router
import entry12_gram_router_contract as gram
import residual_state_mediation_executor as state_executor
import run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1 as dependency
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_multidirection_contextual_gram_router_v16_v1.json"
CROSSFIT_GRAM = ROOT / "circuits/followups/temporal_iswas_v15_entry12_crossfit_gram_router_v1_result.json"
EMBEDDING_NULL = ROOT / "circuits/followups/temporal_iswas_v15_tied_embedding_semantic_router_v16_v1_result.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
V16_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2_audit_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_multidirection_contextual_gram_router_v16_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_multidirection_contextual_gram_router_v16_v1"
EXPECTED = {
    "prior": "5345befc429044e06f23cda6c9be7323174bdd1d40a8e4f3f69ef9a756d313bc",
    "crossfit_gram": "8543717ad765068be8677f093d9bef6f204d6dc8a8d3ccca317985c4f9f2b7de",
    "embedding_null": "b331027d227a95deca30d8f7b31d5c6c8a5ec7d9457268df3de6aba0722aa1b8",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "v16_capability": "a1c2baf0bd9548e189ccc4ba4d11c4905f85af1ff7013d408c143f6f2a6434e3",
    "gram": "c3fdb42f84e96c057dbd20c56136100b27c784f041502365b13072104f15c266",
    "finite_router": "48007c4950bd9116601afbfe66e5e4a6d9bb7dbef65f90afc2dd405714809222",
    "state_executor": "ae3dc83c4a1954575778fa88744f964efc712e8bc483de078e366a90afee9720",
    "dependency": "4d435dfa6c6f29a34de2b5ba7679aafb3b622cdf0fbf2ea779c58bea982b1fe3",
    "parent": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "v15": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "v16": "5b1cb38cc62b5682c03505a5090e962d4773015efb44f8fad145515f5066b549",
}
FILES = {
    "prior": PRIOR, "crossfit_gram": CROSSFIT_GRAM, "embedding_null": EMBEDDING_NULL,
    "oracles": ORACLES, "v16_capability": V16_CAPABILITY,
    "gram": ROOT / "ops/entry12_gram_router_contract.py",
    "finite_router": ROOT / "ops/entry12_finite_router_contract.py",
    "state_executor": ROOT / "ops/residual_state_mediation_executor.py",
    "dependency": ROOT / "ops/run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.py",
    "parent": ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py",
    "v15": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
    "v16": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16.py",
}
V15_ROWS_SHA256 = "3f1d28abb658040493284b307cc27ba76f422dddb08ee9c53686c557d49f283c"
V16_ROWS_SHA256 = "4c5dfaee126c04ac7ea6ef5f53d6ad62a24806fa7133a0c773471a90e4d2e468"
EXPERTS = {"A1_oracle": "A1", "A2_oracle": "A2"}
PRICE = {"native_capture_forwards": 4, "differentiable_transformer_forwards": 11,
         "transformer_backward_forwards": 0, "model_updates": 0,
         "example_evaluations": 656, "fit_parameters": 30}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def classify(torch, rows, truth, predicted):
    report = finite_router.classification_report(torch, truth, predicted)
    report.update({"truth": truth.cpu().tolist(), "predictions": predicted.cpu().tolist(),
                   "row_ids": [row["row_id"] for row in rows]})
    return report


def logo_report(torch, rows, features, labels):
    predicted = torch.empty_like(labels)
    groups = torch.tensor([int(row["group_number"]) for row in rows], device=labels.device)
    for held in range(16):
        train = groups != held; test = ~train
        fitted = gram.fit(torch, features[train], labels[train])
        predicted[test] = gram.predict(torch, features[test], fitted)[0]
    return classify(torch, rows, labels, predicted)


def capture_features(backend, context, counters, bases_by_panel):
    _off_output, off_states = state_executor.execute(parent, backend, context, counters, {}, capture=True)
    off = off_states["entry12"]
    experts = {}
    for panel in ("A1", "A2"):
        _output, states = state_executor.execute(
            parent, backend, context, counters, bases_by_panel[panel], capture=True)
        experts[panel] = states["entry12"]
    return gram.gram_features(
        backend.torch, off, experts, context["base_batch"].semantic_positions)


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    gram_null, embedding_null, oracles, capability = (
        json.loads(path.read_text()) for path in (CROSSFIT_GRAM, EMBEDDING_NULL, ORACLES, V16_CAPABILITY))
    authority = bool(observed == EXPECTED
                     and gram_null.get("terminal") == "router_identification_failure"
                     and embedding_null.get("terminal") == "embedding_router_in_distribution_null"
                     and oracles.get("terminal") == "construction_conditioned_coordinate"
                     and capability.get("terminal") == "manifest")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "price": PRICE, "v16_C_excluded": True}
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
    backend.native = counted
    torch = backend.torch; rows15 = v15.build_rows(); rows16_all = v16.build_rows()
    rows16 = [row for row in rows16_all if row["transform_id"] in ("A1", "A2", "P")]
    excluded16 = [row for row in rows16_all if row["transform_id"] == "C"]
    alignment15 = derive_full_sequence_alignment_contract(rows15, required_panels=("A1", "A2", "P", "C"))
    alignment16 = derive_full_sequence_alignment_contract(rows16, required_panels=("A1", "A2", "P"))

    bank15 = parent.capture_bank(backend, rows15, counters, factors=False)
    features15_parts, rows15_ordered = [], []
    for parity in (0, 1):
        context = parent.subset_context(bank15, parent.row_indices(rows15, parity=parity))
        bases = {panel: dependency.bases_for_evaluation(
            torch, backend.device, oracles, expert)[parity] for expert, panel in EXPERTS.items()}
        features15_parts.append(capture_features(backend, context, counters, bases))
        rows15_ordered.extend(context["rows"])
    features15 = torch.cat(features15_parts)
    labels15 = finite_router.labels_for_rows(torch, rows15_ordered, device=backend.device)
    logo = logo_report(torch, rows15_ordered, features15, labels15)
    fitted = gram.fit(torch, features15, labels15)

    bank16 = parent.capture_bank(backend, rows16, counters, factors=False)
    context16 = parent.subset_context(bank16, range(len(rows16)))
    _off_output, off_states = state_executor.execute(parent, backend, context16, counters, {}, capture=True)
    off16 = off_states["entry12"]; fold_reports = {}; fold_predictions = []
    for fold in (0, 1):
        expert_states = {}
        for expert, panel in EXPERTS.items():
            bases = dependency.bases_for_evaluation(torch, backend.device, oracles, expert)[fold]
            _output, states = state_executor.execute(parent, backend, context16, counters, bases, capture=True)
            expert_states[panel] = states["entry12"]
        features16 = gram.gram_features(
            torch, off16, expert_states, context16["base_batch"].semantic_positions)
        predicted, scores = gram.predict(torch, features16, fitted)
        truth = finite_router.labels_for_rows(torch, rows16, device=backend.device)
        fold_reports[str(fold)] = {**classify(torch, rows16, truth, predicted),
                                   "features": features16.cpu().tolist(), "scores": scores.cpu().tolist()}
        fold_predictions.append(predicted)

    counters["fit_parameters"] = 30
    row_ids_disjoint = set(row["row_id"] for row in rows15).isdisjoint(row["row_id"] for row in rows16_all)
    c_excluded = all(len(row["base_ids"]) != len(row["donor_ids"]) for row in excluded16)
    A = bool(authority and v15.validate_rows(rows15) == V15_ROWS_SHA256
             and v16.validate_rows(rows16_all) == V16_ROWS_SHA256
             and alignment15["panel_counts"] == {panel: 16 for panel in ("A1", "A2", "P", "C")}
             and alignment16["panel_counts"] == {panel: 16 for panel in ("A1", "A2", "P")}
             and len(rows15_ordered) == 64 and len(rows16) == 48 and c_excluded and row_ids_disjoint
             and set(labels15.cpu().tolist()) == {0, 1, 2}
             and finite({"features15": features15.cpu().tolist(), "logo": logo,
                         "fold_reports": fold_reports}) and counters == PRICE)
    B = logo["macro_accuracy"] >= .95 and logo["control_predicted_target_count"] == 0
    C = all(fold_reports[str(fold)]["macro_accuracy"] >= .75
            and fold_reports[str(fold)]["control_predicted_target_count"] == 0 for fold in (0, 1))
    D = all(fold_reports[str(fold)]["confusion"][0][0] >= 12
            and fold_reports[str(fold)]["confusion"][1][1] >= 12 for fold in (0, 1))
    E = bool(torch.equal(fold_predictions[0], fold_predictions[1]))
    predictions = dict(zip((
        "pred_a_authority_alignment_scope_finiteness_and_exact_price",
        "pred_b_v15_leave_one_group_out_is_selective",
        "pred_c_v16_generalizes_under_both_oracle_folds",
        "pred_d_each_v16_target_is_covered",
        "pred_e_oracle_folds_agree_on_v16"), map(bool, (A, B, C, D, E))))
    terminal = ("invalid" if not A else "multidirection_contextual_gram_router_candidate"
                if all((B, C, D, E)) else "contextual_gram_in_distribution_null" if not B
                else "contextual_gram_ood_null")
    result = {"schema": "temporal_iswas_v15_multidirection_contextual_gram_router_v16_result_v1",
              "candidate_id": CANDIDATE_ID, "started_utc": datetime.now(timezone.utc).isoformat(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "alignment_contracts": {"v15": alignment15, "v16_without_c": alignment16},
              "v15_logo": logo, "v15_features": features15.cpu().tolist(),
              "v16_by_oracle_fold": fold_reports,
              "instrument": {"v15_v16_row_ids_disjoint": row_ids_disjoint,
                             "v16_c_explicitly_excluded": c_excluded,
                             "oracle_fold_predictions_agree": E},
              "predictions": predictions, "terminal": terminal, "price": counters,
              "v16_scope": "OOD_TEXT_REUSE_NEW_INTERVENTION"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "v15_logo", "v16_by_oracle_fold", "instrument", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__": main()
