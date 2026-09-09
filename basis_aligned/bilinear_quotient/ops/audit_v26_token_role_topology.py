#!/usr/bin/env python3
"""CPU-only interface audit; no model or capability/causal result access."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

import attention_source_factor_primitive as roles
import circuit_candidate_tense_auxiliary_is_was_structural_holdout_v26 as bank


def main():
    rows = bank.build_rows()
    width = max(len(row["base_ids"]) for row in rows)
    panels = defaultdict(list)
    for row in rows:
        masks = roles.token_role_partition(
            row["base_ids"], row["donor_ids"], row["base_semantic_position"],
            width, torch, device="cpu",
        )
        length = len(row["base_ids"])
        assert row["base_ids"][-1] == row["donor_ids"][-1]
        assert bool((masks[:, :length].sum(0) == 1).all())
        assert not bool(masks[:, length:].any())
        positions = {
            name: masks[index].nonzero().flatten().tolist()
            for index, name in enumerate(roles.SOURCE_GROUPS)
        }
        assert all(row["base_ids"][i] != row["donor_ids"][i]
                   for i in positions["changed"])
        assert all(row["base_ids"][i] == row["donor_ids"][i]
                   for name in ("unchanged_prefix", "matched_suffix") for i in positions[name])
        panels[row["construction_id"]].append({
            "row_id": row["row_id"], "length": length, "positions": positions,
            "changed_to_suffix_causal_pairs": sum(
                source <= destination for source in positions["changed"]
                for destination in positions["matched_suffix"]),
        })
    summaries = {}
    for name, records in sorted(panels.items()):
        summaries[name] = {
            "rows": len(records),
            "role_size_ranges": {
                role: [min(len(r["positions"][role]) for r in records),
                       max(len(r["positions"][role]) for r in records)]
                for role in roles.SOURCE_GROUPS
            },
            "all_changed_to_suffix_cells_nonempty": all(
                r["changed_to_suffix_causal_pairs"] > 0 for r in records),
        }
    print(json.dumps({
        "schema": "v26_token_role_topology_cpu_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows_sha256": bank.authority_sha256(),
        "source_sha256": {Path(m.__file__).name: hashlib.sha256(
            Path(m.__file__).read_bytes()).hexdigest() for m in (roles, bank)},
        "model_forwards": 0, "capability_or_causal_results_opened": False,
        "rows": len(rows), "padded_length": width,
        "partition_checks_passed": True, "panels": summaries,
        "row_topologies": dict(panels),
        "scope": "Token-equality roles transfer mechanically; semantic equivalence and causal transfer remain untested.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
