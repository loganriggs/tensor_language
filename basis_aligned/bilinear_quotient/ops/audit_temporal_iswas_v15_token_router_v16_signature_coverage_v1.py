#!/usr/bin/env python3
"""Zero-model OOD support audit for the immutable v15 token-pair router."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_capability_alignment_and_v15_replay pred_b_fresh_target_signatures_are_unseen pred_c_default_off_certifies_zero_ood_transfer
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16 as v16
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_das_subspace
import unordered_token_pair_router_contract as router

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_token_router_v16_signature_coverage_audit_v1.json"
PARENT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_unordered_token_pair_router_v1_result.json"
V16_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2_audit_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_token_router_v16_signature_coverage_audit_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_token_router_v16_signature_coverage_audit_v1"
EXPECTED = {
    "prior": "055eb7f95c273a8c6e6a20636fa9981cc043bfb1c5bb0ed8c96ded15b9c2ca48",
    "parent": "582c19b22017fd0bb1070260b4737b8b046c54c3838a0a646a85254d77135be0",
    "v16_capability": "a1c2baf0bd9548e189ccc4ba4d11c4905f85af1ff7013d408c143f6f2a6434e3",
    "v15_builder": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "v16_builder": "5b1cb38cc62b5682c03505a5090e962d4773015efb44f8fad145515f5066b549",
    "router": "061f471d690c7549794af203e0d4ceefbbaa9f57fe747611388dcd676e85dbb9",
    "batch_helper": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
}
FILES = {
    "prior": PRIOR,
    "parent": PARENT,
    "v16_capability": V16_CAPABILITY,
    "v15_builder": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py",
    "v16_builder": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16.py",
    "router": ROOT / "ops/unordered_token_pair_router_contract.py",
    "batch_helper": ROOT / "ops/circuit_das_subspace.py",
}
PRICE = {"model_forwards": 0, "checkpoint_loads": 0, "example_evaluations": 128,
         "model_updates": 0, "fit_parameters": 0}
LABELS = {"A1": 0, "A2": 1, "P": 2, "C": 2}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rendered(mapping):
    return {repr(key): int(value) for key, value in mapping.items()}


def aligned(row):
    return (len(row["base_ids"]) == len(row["donor_ids"])
            and row["base_semantic_position"] == row["donor_semantic_position"])


def row_signature(row):
    return router.signature(row["base_ids"], row["donor_ids"], row["base_semantic_position"])


def confusion(rows, predictions):
    return [[sum(LABELS[row["transform_id"]] == actual and prediction == predicted
                 for row, prediction in zip(rows, predictions))
             for predicted in range(3)] for actual in range(3)]


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    authority = observed == EXPECTED
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "price": PRICE}
    if not authority:
        raise RuntimeError(f"authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)

    parent = json.loads(PARENT.read_text())
    capability = json.loads(V16_CAPABILITY.read_text())
    prior = json.loads(PRIOR.read_text())
    rows15, rows16 = v15.build_rows(), v16.build_rows()
    counts15 = Counter(row["transform_id"] for row in rows15)
    counts16 = Counter(row["transform_id"] for row in rows16)

    reconstructed = {}
    replay = True
    for held in (0, 1):
        train_rows = [row for row in rows15 if row["group_number"] % 2 == 1 - held]
        base = circuit_das_subspace._batch(None, train_rows, side="base")
        donor = circuit_das_subspace._batch(None, train_rows, side="donor")
        signatures = router.signatures(base, donor)
        labels = [LABELS[row["transform_id"]] for row in train_rows]
        fit = router.fit(signatures, labels)
        reconstructed[str(held)] = rendered(fit)
        replay = replay and reconstructed[str(held)] == parent["fits"][str(held)]["router"]["train_map"]

    auditable = [row for row in rows16 if row["transform_id"] in ("A1", "A2", "P")]
    excluded = [row for row in rows16 if row["transform_id"] == "C"]
    alignment = all(aligned(row) for row in auditable)
    c_nonapplicable = all(not aligned(row) for row in excluded)
    signatures16 = [row_signature(row) for row in auditable]
    target_indices = [index for index, row in enumerate(auditable)
                      if row["transform_id"] in ("A1", "A2")]

    folds = {}
    all_targets_unseen = True
    all_targets_off = True
    for held in (0, 1):
        training = [row for row in rows15 if row["group_number"] % 2 == 1 - held]
        fit = router.fit(
            router.signatures(circuit_das_subspace._batch(None, training, side="base"),
                              circuit_das_subspace._batch(None, training, side="donor")),
            [LABELS[row["transform_id"]] for row in training])
        predictions = list(router.predict(fit, signatures16))
        target_unseen = [signatures16[index] not in fit for index in target_indices]
        target_off = [predictions[index] == 2 for index in target_indices]
        all_targets_unseen = all_targets_unseen and all(target_unseen)
        all_targets_off = all_targets_off and all(target_off)
        folds[str(held)] = {
            "train_map": rendered(fit),
            "v16_signature_counts": {repr(item): signatures16.count(item) for item in sorted(set(signatures16), key=repr)},
            "confusion_a1_a2_p": confusion(auditable, predictions),
            "target_signature_coverage_count": sum(not flag for flag in target_unseen),
            "target_default_off_count": sum(target_off),
            "target_row_count": len(target_indices),
            "predicted_program_transfer_on_default_off_rows": 0.0,
        }

    pred_a = bool(
        authority and prior.get("candidate_id") == CANDIDATE_ID
        and parent.get("terminal") == "token_pair_router_candidate"
        and all(parent.get("predictions", {}).values())
        and capability.get("terminal") == "manifest"
        and all(capability.get("predictions", {}).values())
        and len(rows15) == len(rows16) == 64
        and counts15 == counts16 == Counter({"A1": 16, "A2": 16, "P": 16, "C": 16})
        and alignment and c_nonapplicable and replay)
    pred_b = bool(all_targets_unseen)
    pred_c = bool(all_targets_off and all(
        fold["target_signature_coverage_count"] == 0
        and fold["target_default_off_count"] == 32
        and fold["predicted_program_transfer_on_default_off_rows"] == 0.0
        for fold in folds.values()))
    predictions = {
        "pred_a_authority_capability_alignment_and_v15_replay": pred_a,
        "pred_b_fresh_target_signatures_are_unseen": pred_b,
        "pred_c_default_off_certifies_zero_ood_transfer": pred_c,
    }
    if not all(math.isfinite(value) for fold in folds.values()
               for key, value in fold.items() if key.endswith("transfer_on_default_off_rows")):
        pred_a = predictions["pred_a_authority_capability_alignment_and_v15_replay"] = False
    terminal = ("invalid" if not pred_a else "unseen_signature_default_off"
                if pred_b and pred_c else "covered_signature_requires_causal_replay"
                if not pred_b else "logical_consequence_failure")
    result = {
        "schema": "temporal_iswas_v15_token_router_v16_signature_coverage_audit_result_v1",
        "candidate_id": CANDIDATE_ID,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authority_sha256": EXPECTED,
        "row_counts": {"v15": dict(counts15), "v16": dict(counts16)},
        "instrument": {"a1_a2_p_aligned": alignment, "c_rows_explicitly_nonapplicable": c_nonapplicable,
                       "v15_parent_maps_replayed": replay, "reconstructed_maps": reconstructed},
        "folds": folds,
        "causal_outcomes_opened": False,
        "predictions": predictions,
        "terminal": terminal,
        "price": PRICE,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"predictions": predictions, "terminal": terminal,
                      "instrument": result["instrument"], "folds": folds, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
