#!/usr/bin/env python3
"""Register the rank-one compression of the subject-number write bank."""
from copy import deepcopy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]; BQ = REPO / "basis_aligned/bilinear_quotient"
sys.path.insert(0, str(BQ))
from circuit_registry_v2 import _atomic_json, _lock, circuit_path, design_key, execution_key, file_sha256, rebuild_registry_v2, validate_v2  # noqa: E402

TAG = "task.subject_verb.number_agreement"; OLD = "grammatical_subject_number.v23"; NEW = "grammatical_subject_number.v24"
EVENT = "task14.mlp6_7.direction_cardinality_rank1_compression.held.v1"
ARTIFACTS = {
    "task14_rank1_builder_v1": ("basis_aligned/polynomial_causal/build_subject_number_direction_cardinality_rank1_v1.py", "builder"),
    "task14_rank1_artifact_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json", "artifact"),
    "task14_rank1_prereg_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_PREREGISTRATION.md", "preregistration"),
    "task14_rank1_binding_v1": ("basis_aligned/polynomial_causal/SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_BINDING.json", "binding"),
    "task14_rank1_runner_v1": ("basis_aligned/bilinear_quotient/ops/run_subject_number_direction_cardinality_rank1_v1.py", "runner"),
    "task14_rank1_result_v1": ("basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_direction_cardinality_rank1_v1_result.json", "result"),
}


def artifact(path, kind): return {"path": path, "sha256": file_sha256(REPO / path), "kind": kind, "status": "frozen"}
def bind(record, event): event["design_key"] = design_key(record, event); event["execution_key"] = execution_key(record, event); return event


def main():
    result = json.loads((BQ / "circuits/fast_screens/subject_number_direction_cardinality_rank1_v1_result.json").read_text())
    assert result["terminal"] == "rank1_compressed_program" and all(result["score"]["predictions"].values())
    path = circuit_path(TAG); existing = json.loads(path.read_text())
    if any(e["event_id"] == EVENT for e in existing["evidence_events"]): validate_v2(existing); rebuild_registry_v2(); print("already registered"); return
    with _lock("registry"):
        record = json.loads(path.read_text())
        for artifact_id, spec in ARTIFACTS.items():
            value = artifact(*spec)
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value: raise ValueError(artifact_id)
            record["artifacts"][artifact_id] = value
        previous = next(c for c in record["claims"] if c["claim_id"] == OLD); claim = deepcopy(previous)
        claim.update({"claim_id": NEW, "revision": 24, "supersedes": OLD, "evidence_event_ids": [*previous["evidence_event_ids"], EVENT],
                      "next_missing": "The ten direction-by-cardinality L11H3 writes compress causally to one shared 1152D axis plus ten scalars, retaining 10.09% of write-bank storage and 25.07% of the charged interface with two direction vectors. This screen reuses the known third corpus. Freeze a genuinely new noun/template authority and confirm rank1 without refit; do not sweep rank, refit gain, repeat readers/programs, or quantize."})
        claim["causal_variable"]["operation"] = "select direction and four-factor background cardinality, then scale one shared rank-one L11H3 write axis"
        site = {"site_id": "L11H3.projected_write.direction_cardinality_rank1_program", "tensor_path": "one shared 1152D projected-write axis plus ten direction-by-cardinality scalars", "shape": [1152, 10], "intervention": "install alpha[direction,cardinality] times the frozen shared axis", "ceiling_event_ids": [EVENT]}
        claim["candidate_sites"].append(site); record["claims"].append(claim)
        s = result["score"]; st = result["plan"]["storage"]
        event = {"event_id": EVENT, "claim_id": NEW, "test_type": "compiled_equivalence", "stage": "complete", "verdict": "held", "failure_kind": None,
                 "family_ids": [], "site_id": site["site_id"], "split_plan_id": "task14_fresh_matched_natural_split_v1", "evaluation_role": "weights_only_rank1_causal_compression_screen",
                 "metrics": [
                     {"name": "rank1_energy", "estimate": result["plan"]["rank1_energy"], "ci95": None, "bar": "descriptive"},
                     {"name": "rank1_vs_original_cosine", "estimate": s["rank1_vs_original"]["cosine"], "ci95": None, "bar": ">=0.98"},
                     {"name": "rank1_vs_original_relative_l2", "estimate": s["rank1_vs_original"]["relative_l2_error"], "ci95": None, "bar": "<=0.20"},
                     {"name": "rank1_vs_native_cosine", "estimate": s["rank1_vs_native"]["cosine"], "ci95": None, "bar": ">=0.75"},
                     {"name": "write_storage_fraction", "estimate": st["write_storage_fraction"], "ci95": None, "bar": "<0.11"},
                     {"name": "total_interface_storage_fraction", "estimate": st["total_interface_storage_fraction"], "ci95": None, "bar": "<0.26"}],
                 "prereg_artifact_id": "task14_rank1_prereg_v1", "result_artifact_id": "task14_rank1_result_v1", "input_artifact_ids": list(ARTIFACTS),
                 "seed": None, "checkpoint_sha256": result["checkpoint_weights_sha256"], "supersedes_event_id": None, "replicates_event_id": None,
                 "sections": ["basis_aligned/polynomial_causal/SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_PREREGISTRATION.md"],
                 "notes": "Rank one was frozen from prototype coordinates only; no causal outcome, behavioral fit, gain, or rank sweep entered construction. Fresh-construction confirmation remains open."}
        record["evidence_events"].append(bind(record, event)); validate_v2(record); _atomic_json(path, record)
    rebuild_registry_v2(); final = json.loads(path.read_text()); validate_v2(final); print(json.dumps({"status": "registered", "claim_id": NEW, "event_id": EVENT}, indent=2))


if __name__ == "__main__": main()
