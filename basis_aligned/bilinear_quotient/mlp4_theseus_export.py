#!/usr/bin/env python3
"""Export frozen MLP4 candidates without flattening incompatible MDL claims."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVENTORY = HERE / "mlp4_z4_candidate_inventory.json"
PROTOCOL = HERE / "mlp4_z4_validation_protocol.json"
RESULTS = HERE / "mlp4_z4_validation_results.json"
INVARIANTS = HERE / "mlp4_candidate_tensor_invariants.json"
OUTPUT = HERE / "mlp4_theseus_evidence.json"
LANES = ("held_out", "composite", "extraction", "removal", "ood")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def price(row):
    if row["family"] == "linear":
        return {"bits": row["quotient_bits"], "status": "quotient_generic_stratum",
                "eligible_for_unconditional_mdl": True,
                "caveat": "near-degenerate singular strata require another canonicalizer"}
    if row["family"] == "native_product":
        return {"bits": row["conditional_known_gauge_bits"],
                "status": "conditional_known_gauge",
                "eligible_for_unconditional_mdl": False,
                "caveat": "global partially symmetric CP equivalence is not quotiented"}
    return {"bits": row["canonical_program_bits"],
            "status": "canonical_seeded_program_nonminimal",
            "eligible_for_unconditional_mdl": False,
            "caveat": "portable executable length; no minimal-program claim"}


def build(results_path=RESULTS):
    inventory = json.loads(INVENTORY.read_text())
    protocol = json.loads(PROTOCOL.read_text())
    invariant_artifact = json.loads(INVARIANTS.read_text()) if INVARIANTS.exists() else None
    invariant_rows = {}
    if invariant_artifact is not None:
        if invariant_artifact["inventory_sha256"] != sha(INVENTORY):
            raise ValueError("tensor invariant inventory mismatch")
        invariant_rows = {row["candidate_id"]: row
                          for row in invariant_artifact["rows"]}
    results = json.loads(results_path.read_text()) if results_path.exists() else None
    points = {}
    if results is not None:
        if results.get("partial") is not False:
            raise ValueError("partial validation results cannot enter evidence export")
        if results["protocol_id"] != protocol["protocol_id"]:
            raise ValueError("validation protocol mismatch")
        points = {point["candidate_id"]: point for point in results["points"]}
        if set(points) != set(protocol["candidate_order"]):
            raise ValueError("validation must contain the complete frozen roster")
    candidates = []
    for row in inventory["candidates"]:
        candidate_id = row["candidate_id"]
        lanes = {lane: {"status": "unmeasured", "score": None}
                 for lane in LANES}
        if results is not None:
            point = points[candidate_id]
            if point["program_hash"] != row["canonical_bytes_hash"]:
                raise ValueError(f"program hash mismatch: {candidate_id}")
            lanes["held_out"] = {
                "status": "measured_prospective_validation",
                "score": point["fidelity"],
                "ce": point["ce"],
                "delta_ce": point["delta_ce"],
                "anchor": "fit_mean_mlp4_output",
            }
        structural = {"status": "not_applicable_affine"} \
            if row["family"] == "linear" else {"status": "not_audited"}
        if candidate_id in invariant_rows:
            measured = invariant_rows[candidate_id]
            structural = {"status": "measured_factorization_invariant",
                          "output_mode_rank": measured["rank"],
                          "output_mode_stable_rank": measured["stable_rank"],
                          "output_mode_entropy_rank": measured["entropy_rank"],
                          "energy_rank_90": measured["energy_rank_90"],
                          "energy_rank_95": measured["energy_rank_95"],
                          "energy_rank_99": measured["energy_rank_99"],
                          "top32_mode_energy_fraction": measured[
                              "top32_mode_energy_fraction"],
                          "not_behavioral_evidence": True,
                          "not_description_length": True}
        candidates.append({
            "candidate_id": candidate_id,
            "family": row["family"],
            "capacity": row["capacity"],
            "program_hash": row["canonical_bytes_hash"],
            "declared_inputs": ["blocks.4.mlp.rmsnorm_input"],
            "price": price(row),
            "structural_tensor": structural,
            "operational_lanes": lanes,
            "frontier_eligible": results is not None and all(
                lanes[lane]["status"] != "unmeasured" for lane in LANES),
        })
    return {
        "schema_version": 1,
        "evidence_id": "bilin18.blocks.4.mlp.z4-programs.v1",
        "model": "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd",
        "module": "blocks.4.mlp",
        "granularity": "module",
        "interface": {"inputs": ["blocks.4.mlp.rmsnorm_input"],
                      "output": "blocks.4.mlp.output", "positionwise": True},
        "inventory_sha256": sha(INVENTORY),
        "validation_protocol": protocol["protocol_id"],
        "validation_results_sha256": sha(results_path) if results is not None else None,
        "tensor_invariants_sha256": sha(INVARIANTS) if invariant_artifact is not None else None,
        "controls": {"identity": "retained_live_mlp4",
                     "zero_fidelity": "fit_mean_mlp4_output"},
        "score_definition": "1-(candidate_ce-live_ce)/(fit_mean_ce-live_ce)",
        "candidates": candidates,
        "coverage": {lane: sum(c["operational_lanes"][lane]["status"] != "unmeasured"
                               for c in candidates) for lane in LANES},
        "promotion_policy": "all five lanes required; price eligibility remains separate",
    }


def main():
    result = build()
    OUTPUT.write_text(json.dumps(result, indent=2)+"\n")
    print(f"wrote {OUTPUT.name}: {len(result['candidates'])} candidates, "
          f"coverage={result['coverage']}")


if __name__ == "__main__":
    main()
