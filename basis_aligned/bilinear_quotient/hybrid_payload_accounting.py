#!/usr/bin/env python3
"""Scope-complete learned-constant accounting for the current partial program."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "hybrid_payload_source_snapshot.json"
OUTPUT = HERE / "hybrid_payload_accounting.json"


def build():
    source = json.loads(SOURCE.read_text())
    if source.get("schema_version") != 1 or len(source.get("source_sha256", {})) != 3:
        raise ValueError("unsupported or unbound source snapshot")
    identity = source["identity"]
    scopes = source["disjoint_checkpoint_scope"]
    if len({row["scope_id"] for row in scopes}) != len(scopes):
        raise ValueError("duplicate checkpoint scope")
    if sum(row["element_count"] for row in scopes) != identity["checkpoint_element_count"]:
        raise ValueError("source checkpoint scope does not close")
    charges = source["charges"]
    if len({row["charge_id"] for row in charges}) != len(charges):
        raise ValueError("duplicate hybrid charge")
    qk = {row["charge_id"]: row for row in charges}
    if qk["candidate_attention_qk139"]["covered_checkpoint_elements"] != 139*4*128*1152 \
            or qk["retained_attention_qk23"]["covered_checkpoint_elements"] != \
            23*4*128*1152:
        raise ValueError("Q/K head partition mismatch")
    covered = sum(row["covered_checkpoint_elements"] for row in charges)
    if covered != identity["checkpoint_element_count"]:
        raise ValueError(f"hybrid scalar partition leaves {identity['checkpoint_element_count']-covered}")
    total = sum(row["bits"] for row in charges)
    identity_bits = identity["checkpoint_tensor_payload_bits"]
    result = {
        "schema_version": 1,
        "accounting_id": "bilin18.current-partial-program-plus-literal-remainder.v1",
        "sources": source["source_sha256"],
        "source_snapshot": SOURCE.name,
        "scope": {"checkpoint_element_count": identity["checkpoint_element_count"],
                  "covered_element_count": covered, "uncovered_element_count": 0,
                  "double_counted_element_count": 0,
                  "fixed_architecture_loader_schema": True},
        "charges": charges,
        "accounting": {"hybrid_learned_constant_payload_bits": total,
                       "identity_checkpoint_tensor_payload_bits": identity_bits,
                       "payload_bits_removed": identity_bits-total,
                       "hybrid_to_identity_payload_ratio": total/identity_bits},
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
