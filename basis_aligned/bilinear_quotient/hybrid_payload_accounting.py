#!/usr/bin/env python3
"""Scope-complete learned-constant accounting for the current partial program."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / "checkpoint_payload_inventory.json"
COVERAGE = HERE / "whole_model_coverage.json"
RETAINED_QK = HERE / "retained_qk_dtype_audit.json"
OUTPUT = HERE / "hybrid_payload_accounting.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    checkpoint = json.loads(CHECKPOINT.read_text())
    coverage = json.loads(COVERAGE.read_text())
    retained_qk = json.loads(RETAINED_QK.read_text())
    objects = {row["object_id"]: row for row in checkpoint["objects"]}
    if len(objects) != len(checkpoint["objects"]):
        raise ValueError("duplicate checkpoint object ownership")
    accounting = checkpoint["accounting"]
    if sum(row["element_count"] for row in objects.values()) != accounting["element_count"] \
            or sum(row["payload_bits"] for row in objects.values()) != \
            accounting["exact_tensor_payload_bits"]:
        raise ValueError("checkpoint object partition does not close")

    whole = coverage["whole_program_accounting_status"]
    partial = coverage["cross_module_programs"][0]
    structural = coverage["structural_scalar_schedule"]["codec"]
    if whole["accounted_disjoint_module_ids"] != ["mlp0", "mlp1", "mlp2", "mlp3"]:
        raise ValueError("unexpected candidate MLP scope")
    if partial["covered_object_count"] != 139:
        raise ValueError("unexpected decoded Q/K scope")

    mlp_replaced = [objects[f"mlp{layer}"] for layer in range(4)]
    mlp_retained = [objects[f"mlp{layer}"] for layer in range(4, 18)]
    attention = [objects[f"attn{layer}"] for layer in range(18)]
    qk_names = ("c_q.weight", "c_k.weight", "c_q2.weight", "c_k2.weight")
    vo_names = ("c_v.weight", "c_proj.weight")
    qk_tensors = [tensor for row in attention for tensor in row["tensors"]
                  if tensor["name"].endswith(qk_names)]
    vo_tensors = [tensor for row in attention for tensor in row["tensors"]
                  if tensor["name"].endswith(vo_names)]
    attn_scalars = [tensor for row in attention for tensor in row["tensors"]
                    if tensor["name"].endswith("attn.lamb")]
    schedule = objects["structural_residual_schedule"]
    if len(qk_tensors) != 72 or len(vo_tensors) != 36 or len(attn_scalars) != 18:
        raise ValueError("attention tensor ownership schema changed")

    retained_qk_elements = retained_qk["scope"]["scalar_count"]
    qk_elements = sum(row["element_count"] for row in qk_tensors)
    decoded_qk_elements = qk_elements-retained_qk_elements
    if decoded_qk_elements != 139*4*128*1152:
        raise ValueError("decoded Q/K scalar partition mismatch")
    structural_elements = (sum(row["element_count"] for row in attn_scalars)
                           + schedule["element_count"])
    if structural_elements != coverage["summary"]["structural_scalar_count"]:
        raise ValueError("structural scalar partition mismatch")

    charges = [
        {"charge_id": "candidate_mlp0_3", "kind": "candidate_codec",
         "covered_checkpoint_elements": sum(row["element_count"] for row in mlp_replaced),
         "bits": whole["accounted_module_candidate_bits"]},
        {"charge_id": "retained_mlp4_17", "kind": "literal_checkpoint_payload",
         "covered_checkpoint_elements": sum(row["element_count"] for row in mlp_retained),
         "bits": sum(row["payload_bits"] for row in mlp_retained)},
        {"charge_id": "candidate_attention_qk139", "kind": "candidate_codec",
         "covered_checkpoint_elements": decoded_qk_elements,
         "bits": partial["canonical_quotient_bits"]},
        {"charge_id": "retained_attention_qk23", "kind": "literal_checkpoint_slices",
         "covered_checkpoint_elements": retained_qk_elements,
         "bits": retained_qk["corrected_accounting"]["literal_checkpoint_slice_bits"]},
        {"charge_id": "retained_attention_vo", "kind": "literal_checkpoint_payload",
         "covered_checkpoint_elements": sum(row["element_count"] for row in vo_tensors),
         "bits": sum(row["payload_bits"] for row in vo_tensors)},
        {"charge_id": "structural_schedule", "kind": "exact_candidate_codec",
         "covered_checkpoint_elements": structural_elements,
         "bits": structural["selected_bits"]},
        {"charge_id": "retained_token_embedding_x0", "kind": "literal_checkpoint_payload",
         "covered_checkpoint_elements": objects["token_embedding_x0"]["element_count"],
         "bits": objects["token_embedding_x0"]["payload_bits"]},
        {"charge_id": "retained_unembedding", "kind": "literal_checkpoint_payload",
         "covered_checkpoint_elements": objects["unembedding"]["element_count"],
         "bits": objects["unembedding"]["payload_bits"]},
    ]
    covered = sum(row["covered_checkpoint_elements"] for row in charges)
    if covered != accounting["element_count"]:
        raise ValueError(f"hybrid scalar partition leaves {accounting['element_count']-covered}")
    total = sum(row["bits"] for row in charges)
    identity = accounting["exact_tensor_payload_bits"]
    result = {
        "schema_version": 1,
        "accounting_id": "bilin18.current-partial-program-plus-literal-remainder.v1",
        "sources": {"checkpoint_payload_inventory_sha256": sha(CHECKPOINT),
                    "whole_model_coverage_sha256": sha(COVERAGE),
                    "retained_qk_dtype_audit_sha256": sha(RETAINED_QK)},
        "scope": {"checkpoint_element_count": accounting["element_count"],
                  "covered_element_count": covered, "uncovered_element_count": 0,
                  "double_counted_element_count": 0,
                  "fixed_architecture_loader_schema": True},
        "charges": charges,
        "accounting": {"hybrid_learned_constant_payload_bits": total,
                       "identity_checkpoint_tensor_payload_bits": identity,
                       "payload_bits_removed": identity-total,
                       "hybrid_to_identity_payload_ratio": total/identity},
        "claims": {"scope_comparable_to_identity_tensor_payload": True,
                   "complete_learned_constant_payload_accounting": True,
                   "identity_replay": False,
                   "joint_operational_fidelity_certified": False,
                   "decoder_graph_schema_charged": False,
                   "complete_program_bound": False,
                   "quotient_price": False,
                   "minimal_description_length": False},
        "interpretation": "All checkpoint learned constants are replaced by a candidate stream or retained literally exactly once. Decoder, architecture, loader, tensor schema, and joint assembled fidelity remain outside this accounting."
    }
    return result


def main():
    result = build()
    OUTPUT.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["accounting"], indent=2))


if __name__ == "__main__":
    main()
