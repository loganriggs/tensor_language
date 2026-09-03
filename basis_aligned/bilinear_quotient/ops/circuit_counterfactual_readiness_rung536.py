#!/usr/bin/env python3
"""CPU-only audit of the 62 circuit records for DAS counterfactual readiness.

Circuit tags are tree-instance-local. A story attached to the old 212-row tree is
not evidence about a same-named mask in the diverse 1000-row tree.
"""

from __future__ import annotations

import collections
import hashlib
import json
import statistics
from itertools import combinations
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
BQ = ROOT / "bilinear_quotient"
CIRCUITS = BQ / "circuits"
BATTERY = CIRCUITS / "BATTERY.json"
REGISTRY = CIRCUITS / "registry.json"
STATE = BQ / "census_state_diverse.pt"
OUT = BQ / "circuit_counterfactual_readiness_rung536.json"
BATTERY_TREE = "diverse-1000row-v1"
DISCOVERY_ROOTS = {0, 2, 4, 6, 8, 18}
HELDOUT_ROOTS = {1, 3, 5, 7, 11, 13, 23}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_id(tag: str) -> int:
    return int(tag.split(".")[1])


def overlap_summary(masks: dict[str, set[int]], tags: list[str]) -> dict:
    intersections = []
    containments = []
    for left, right in combinations(tags, 2):
        a, b = masks[left], masks[right]
        overlap = len(a & b)
        intersections.append(overlap)
        if a <= b or b <= a:
            containments.append([left, right])
    return {
        "mask_count": len(tags),
        "pair_count": len(intersections),
        "nonzero_intersection_count": sum(value > 0 for value in intersections),
        "intersection_min": min(intersections),
        "intersection_median": statistics.median(intersections),
        "intersection_max": max(intersections),
        "exact_containment_count": len(containments),
        "exact_containment_pairs": containments,
    }


def main() -> None:
    battery = json.loads(BATTERY.read_text())
    registry = json.loads(REGISTRY.read_text())["circuits"]
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    state_by_tag = {leaf["tag"]: leaf for leaf in state["leaves"]}
    tags = list(battery["by_tag"])
    assert len(tags) == 62
    assert all(tag in state_by_tag for tag in tags)

    records = []
    input_hashes = {
        str(BATTERY.relative_to(ROOT)): sha256(BATTERY),
        str(REGISTRY.relative_to(ROOT)): sha256(REGISTRY),
        str(STATE.relative_to(ROOT)): sha256(STATE),
    }
    for tag in tags:
        circuit_path = CIRCUITS / registry[tag]["file"]
        circuit = json.loads(circuit_path.read_text())
        input_hashes[str(circuit_path.relative_to(ROOT))] = sha256(circuit_path)
        story = circuit.get("story") or {}
        mechanism_level = story.get("mechanism_level")
        has_program = story.get("program") is not None
        has_computational_code = (
            mechanism_level == "computational"
            and isinstance(story.get("mechanism"), dict)
            and bool(story["mechanism"].get("code"))
        )
        has_explicit_counterfactual_spec = any(
            key in circuit
            for key in (
                "counterfactual",
                "counterfactuals",
                "intervention_pairs",
                "counterfactual_dataset",
            )
        )
        card_tree = (circuit.get("tree") or {}).get("instance")
        tree_matches = card_tree == BATTERY_TREE
        if not tree_matches:
            readiness = "cross_tree_identity_unverified"
        elif has_computational_code:
            readiness = "computational_code_but_no_paired_counterfactual_dataset"
        elif mechanism_level == "surface":
            readiness = "surface_rule_hypothesis_only"
        elif has_program:
            readiness = "descriptive_program_without_validated_mechanism"
        else:
            readiness = "frozen_cluster_only"
        split = (
            "discovery"
            if root_id(tag) in DISCOVERY_ROOTS
            else "heldout"
            if root_id(tag) in HELDOUT_ROOTS
            else "unexpected"
        )
        assert split != "unexpected"
        records.append(
            {
                "tag": tag,
                "split": split,
                "card_tree": card_tree,
                "battery_tree": BATTERY_TREE,
                "card_tree_matches_battery_tree": tree_matches,
                "readiness": readiness,
                "mechanism_level": mechanism_level,
                "has_surface_program": has_program,
                "program_balanced_accuracy": story.get("program_bacc"),
                "program_null": story.get("program_null"),
                "has_computational_code": has_computational_code,
                "has_explicit_counterfactual_spec": has_explicit_counterfactual_spec,
                "headline": story.get("blind_name") or registry[tag].get("headline") or "",
                "battery_member_count": int(state_by_tag[tag]["member"].numel()),
            }
        )

    categories = collections.Counter(record["readiness"] for record in records)
    splits = collections.Counter(record["split"] for record in records)
    registry_story_categories = collections.Counter()
    for record in records:
        if record["has_computational_code"]:
            registry_story_categories["computational_code"] += 1
        elif record["mechanism_level"] == "surface":
            registry_story_categories["surface_rule"] += 1
        elif record["has_surface_program"]:
            registry_story_categories["descriptive_program"] += 1
        else:
            registry_story_categories["frozen_cluster_only"] += 1

    program_rows = [record for record in records if record["program_balanced_accuracy"] is not None]
    computational = [record["tag"] for record in records if record["has_computational_code"]]
    explicit = [record["tag"] for record in records if record["has_explicit_counterfactual_spec"]]
    tree_matches = [record["tag"] for record in records if record["card_tree_matches_battery_tree"]]
    tree_mismatches = [record["tag"] for record in records if not record["card_tree_matches_battery_tree"]]

    masks = {tag: set(map(int, state_by_tag[tag]["member"].tolist())) for tag in tags}
    all_overlap = overlap_summary(masks, tags)
    size_864_tags = [tag for tag in tags if len(masks[tag]) == 864]
    size_864_overlap = overlap_summary(masks, size_864_tags)

    result = {
        "rung": 536,
        "stage": "counterfactual_readiness_inventory",
        "status": "complete",
        "scope": {
            "battery_circuits": len(records),
            "discovery": splits["discovery"],
            "heldout": splits["heldout"],
        },
        "readiness_category_counts": dict(sorted(categories.items())),
        "registry_story_category_counts_before_tree_identity_gate": dict(
            sorted(registry_story_categories.items())
        ),
        "registry_card_computational_code_candidates": computational,
        "explicit_counterfactual_spec_candidates": explicit,
        "tree_identity": {
            "battery_tree": BATTERY_TREE,
            "matching_card_count": len(tree_matches),
            "mismatching_card_count": len(tree_mismatches),
            "matching_tags": tree_matches,
            "mismatching_tags": tree_mismatches,
            "warning": (
                "Circuit tags are tree-instance-local. A story from 212row-v1 cannot label a "
                "same-named diverse-1000row-v1 mask without an explicit membership/behavior remap."
            ),
        },
        "battery_mask_relationships": {
            "all_62": all_overlap,
            "size_864_masks": size_864_overlap,
            "warning": (
                "Every mask pair overlaps and some are exact containments. These masks are overlapping "
                "response regions, not mutually exclusive values of one causal variable."
            ),
        },
        "surface_program_diagnostics": {
            "records_with_program_before_tree_identity_gate": len(program_rows),
            "warning": (
                "These programs describe frozen census membership. They are neither counterfactual "
                "validity nor causal mechanism tests, and old-tree programs require remapping."
            ),
        },
        "decision": {
            "circuits_ready_for_weight_DAS_from_record_alone": 0,
            "binding_bottleneck": (
                "tree-matched circuit identity plus circuit-specific paired counterfactual construction "
                "and validation"
            ),
            "parallelizable_after_contract": True,
            "first_shared_requirement": (
                "at least two causally distinct counterfactual families for the same proposed variable, "
                "including answer-changing interchange and answer-preserving controls"
            ),
        },
        "records": records,
        "input_sha256": input_hashes,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "scope",
                    "readiness_category_counts",
                    "registry_story_category_counts_before_tree_identity_gate",
                    "registry_card_computational_code_candidates",
                    "explicit_counterfactual_spec_candidates",
                    "tree_identity",
                    "battery_mask_relationships",
                    "decision",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
