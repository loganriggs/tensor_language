#!/usr/bin/env python3
"""Outcome-blind manifest contract for the prospective R585 replacement.

This module reads only the frozen R578 row authority and the prospective R585
specification.  It has no model, CUDA, torch, queue, registry, or outcome-file
dependency.  The dependency-lock validator is deliberately supplied observed
hashes by its caller; it never discovers or opens R586/R587 artifacts itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT.parent / "polynomial_causal"
ROWS_PATH = ROOT / "induction_selector_payload_three_source_rows_rung578.json"
AMENDMENT_PATH = POLY / (
    "INDUCTION_SELECTOR_PAYLOAD_FROZEN_FACTOR_RUNG585_REPLACEMENT_AMENDMENT.md"
)
DRYRUN_PATH = ROOT / "induction_selector_payload_frozen_factor_rung585_manifest_dryrun.json"

ROWS_SHA256 = "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6"
AMENDMENT_SHA256 = "98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf"

SPLITS = ("FIT", "SELECT")
DIRECTIONS = ("base_to_donor", "donor_to_base")
ARMS = ("score", "payload", "joint")
CONDITIONS = ("s0p0", "s0p1", "s1p0", "s1p1")
BOOTSTRAPS = 2_000
BOOTSTRAP_NAMESPACE = "a8-r585-replacement-group-bootstrap-v1"
BATCH_SIZE = 32

F_SELECTOR = "two_valid_sources_selector_swap"
F_PAYLOAD = "payload_swap_match_preserved"
F_JOINT = "selector_payload_joint_answer_preserved"
F_MATCH = "match_break_payload_preserved"
F_NEUTRAL_SOURCE = "irrelevant_source_edit"
F_NEUTRAL_PAYLOAD = "irrelevant_payload_edit"
F_NUISANCE = "copy_relation_preserved_nuisance_change"
F_CONTRAST = "contrast_target_source_edit"

TARGET_FAMILIES = (F_SELECTOR, F_PAYLOAD, F_JOINT, F_MATCH)
CONTROL_FAMILIES = (F_NEUTRAL_SOURCE, F_NEUTRAL_PAYLOAD, F_NUISANCE)
INCLUDED_FAMILIES = TARGET_FAMILIES + CONTROL_FAMILIES

EXPECTED_VARIANTS = {
    F_SELECTOR: ("payload_assignment_0", "payload_assignment_1"),
    F_PAYLOAD: ("selector_0", "selector_1"),
    F_JOINT: ("payload_B", "payload_D"),
    F_MATCH: CONDITIONS,
    F_NEUTRAL_SOURCE: CONDITIONS,
    F_NEUTRAL_PAYLOAD: CONDITIONS,
    F_NUISANCE: tuple(f"{c}:filler_change" for c in CONDITIONS)
    + tuple(f"{c}:lag_extension" for c in CONDITIONS),
}

EXPECTED_SPLIT_COUNTS = {
    "FIT": {"rows": 1_872, "directions": 3_744, "endpoints": 1_728, "groups_per_cell": 72},
    "SELECT": {"rows": 936, "directions": 1_872, "endpoints": 864, "groups_per_cell": 36},
}

BOOTSTRAP_METRICS = (
    "denominator_mean",
    "numerator_mean",
    "donor_ce_mean",
    "single_score_harm_mean",
    "single_payload_harm_mean",
    "factorial_interaction_mean",
)

DEPENDENCY_LOCK_SCHEMA = "induction_selector_payload_frozen_factor_rung585_dependency_lock_v1"
DEPENDENCY_LOCK_KEYS = frozenset(
    {
        "schema",
        "r578_rows_sha256",
        "replacement_amendment_sha256",
        "r586_result_path",
        "r586_result_sha256",
        "r586_receipt_path",
        "r586_receipt_sha256",
        "r586_verdict",
        "r587_audit_path",
        "r587_audit_sha256",
        "r587_audit_verdict",
        "evaluated_splits",
        "forbidden_splits_opened",
    }
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sequence_id(ids: Sequence[int]) -> str:
    return sha256_bytes(b"r578-sequence-v1:" + canonical_json_bytes(list(ids)))


def _condition(row: Mapping[str, object], prefix: str) -> str:
    condition = f"s{row[prefix + '_selector']}p{row[prefix + '_payload_assignment']}"
    if condition not in CONDITIONS:
        raise ValueError(f"invalid {prefix} condition: {condition}")
    return condition


def _recipient_prefix(direction: str) -> str:
    if direction == "base_to_donor":
        return "base"
    if direction == "donor_to_base":
        return "donor"
    raise ValueError(f"unknown direction: {direction}")


def _donor_prefix(direction: str) -> str:
    return "donor" if _recipient_prefix(direction) == "base" else "base"


def _control_kind(row: Mapping[str, object]) -> str | None:
    family = row["family_id"]
    if family == F_NEUTRAL_SOURCE:
        return "neutral_source"
    if family == F_NEUTRAL_PAYLOAD:
        return "neutral_payload"
    if family == F_NUISANCE:
        variant = str(row["family_variant"])
        if variant.endswith(":filler_change"):
            return "filler"
        if variant.endswith(":lag_extension"):
            return "lag"
    return None


def _semantic_equality_count(structure: Mapping[str, object]) -> int:
    query = int(structure["query_id"])
    return sum(int(source) == query for source in structure["source_ids"])


def load_rows() -> list[dict]:
    """Load and verify only the R578 model-free row authority."""
    if sha256_file(ROWS_PATH) != ROWS_SHA256:
        raise RuntimeError("R578 row authority hash mismatch")
    if sha256_file(AMENDMENT_PATH) != AMENDMENT_SHA256:
        raise RuntimeError("R585 replacement amendment hash mismatch")
    document = json.loads(ROWS_PATH.read_text())
    rows = [row for row in document["rows"] if row["family_id"] in INCLUDED_FAMILIES]
    if any(row["family_id"] == F_CONTRAST for row in rows):
        raise AssertionError("contrast rows entered the R585 manifest")
    observed_variants = {
        family: tuple(sorted({str(row["family_variant"]) for row in rows if row["family_id"] == family}))
        for family in INCLUDED_FAMILIES
    }
    expected_variants = {family: tuple(sorted(variants)) for family, variants in EXPECTED_VARIANTS.items()}
    if observed_variants != expected_variants:
        raise RuntimeError("R578 included-family variant census changed")
    return rows


def build_authority_manifest(rows: Sequence[Mapping[str, object]] | None = None) -> dict[str, object]:
    """Return exact included rows, unique endpoints, and declared directions."""
    source_rows = list(load_rows() if rows is None else rows)
    row_records: list[dict[str, object]] = []
    endpoints: dict[tuple[str, str], dict[str, object]] = {}
    directions: list[dict[str, object]] = []

    for row in sorted(source_rows, key=lambda value: (value["split"], value["row_id"])):
        split = str(row["split"])
        if split not in SPLITS:
            continue
        if tuple(row["evaluation_directions"]) != DIRECTIONS:
            raise RuntimeError(f"unexpected directions for row {row['row_id']}")
        endpoint_ids: dict[str, str] = {}
        for prefix in ("base", "donor"):
            ids = tuple(int(token) for token in row[prefix + "_ids"])
            endpoint_id = _sequence_id(ids)
            endpoint_ids[prefix] = endpoint_id
            record = {
                "split": split,
                "endpoint_id": endpoint_id,
                "ids": list(ids),
                "answer_id": int(row[prefix + "_answer_id"]),
                "other_answer_id": int(row[prefix + "_other_answer_id"]),
                "condition": _condition(row, prefix),
            }
            key = (split, endpoint_id)
            if key in endpoints and endpoints[key] != record:
                raise RuntimeError(f"inconsistent endpoint metadata: {endpoint_id}")
            endpoints[key] = record

        row_records.append(
            {
                "split": split,
                "row_id": str(row["row_id"]),
                "group_id": str(row["group_id"]),
                "family": str(row["family_id"]),
                "variant": str(row["family_variant"]),
                "base_endpoint_id": endpoint_ids["base"],
                "donor_endpoint_id": endpoint_ids["donor"],
            }
        )
        for direction in DIRECTIONS:
            recipient = _recipient_prefix(direction)
            donor = _donor_prefix(direction)
            directions.append(
                {
                    "split": split,
                    "directed_id": f"{row['row_id']}:{direction}",
                    "row_id": str(row["row_id"]),
                    "group_id": str(row["group_id"]),
                    "family": str(row["family_id"]),
                    "variant": str(row["family_variant"]),
                    "direction": direction,
                    "recipient_condition": _condition(row, recipient),
                    "recipient_endpoint_id": endpoint_ids[recipient],
                    "donor_endpoint_id": endpoint_ids[donor],
                    "recipient_is_coherent": _semantic_equality_count(row[recipient + "_structure"]) == 1,
                    "donor_is_coherent": _semantic_equality_count(row[donor + "_structure"]) == 1,
                    "donor_coherence_sign": (
                        1 if _semantic_equality_count(row[donor + "_structure"]) == 1 else -1
                    ) if row["family_id"] == F_MATCH else None,
                    "answer_changes": bool(row["answer_changes"]),
                    "control_kind": _control_kind(row),
                }
            )

    endpoint_records = sorted(endpoints.values(), key=lambda value: (value["split"], value["endpoint_id"]))
    directions.sort(key=lambda value: (value["split"], value["directed_id"]))
    for split in SPLITS:
        expected = EXPECTED_SPLIT_COUNTS[split]
        split_rows = [value for value in row_records if value["split"] == split]
        split_endpoints = [value for value in endpoint_records if value["split"] == split]
        split_directions = [value for value in directions if value["split"] == split]
        actual = (len(split_rows), len(split_directions), len(split_endpoints))
        wanted = (expected["rows"], expected["directions"], expected["endpoints"])
        if actual != wanted:
            raise RuntimeError(f"{split} authority census mismatch: {actual} != {wanted}")
    return {"rows": row_records, "endpoints": endpoint_records, "directions": directions}


def _directed_cell_id(record: Mapping[str, object]) -> str:
    return "|".join(
        str(record[field])
        for field in ("split", "family", "variant", "recipient_condition", "direction")
    )


def build_cell_manifests(authority: Mapping[str, Sequence[Mapping[str, object]]] | None = None) -> dict[str, object]:
    """Build literal target/control/coverage/structural manifests."""
    authority = build_authority_manifest() if authority is None else authority
    buckets: dict[tuple[str, str, str, str, str], list[Mapping[str, object]]] = {}
    for record in authority["directions"]:
        key = tuple(str(record[field]) for field in ("split", "family", "variant", "recipient_condition", "direction"))
        buckets.setdefault(key, []).append(record)

    cells: list[dict[str, object]] = []
    for key, members in sorted(buckets.items()):
        split, family, variant, condition, direction = key
        group_ids = sorted(str(member["group_id"]) for member in members)
        if len(group_ids) != len(set(group_ids)):
            raise RuntimeError(f"duplicate semantic group in cell {key}")
        expected_groups = EXPECTED_SPLIT_COUNTS[split]["groups_per_cell"]
        if len(group_ids) != expected_groups:
            raise RuntimeError(f"wrong group count in cell {key}")
        role = "target" if family in TARGET_FAMILIES else "control"
        cells.append(
            {
                "cell_id": "|".join(key),
                "split": split,
                "family": family,
                "variant": variant,
                "recipient_condition": condition,
                "direction": direction,
                "role": role,
                "control_kind": members[0]["control_kind"],
                "group_ids": group_ids,
                "directed_ids": sorted(str(member["directed_id"]) for member in members),
            }
        )

    target_cells = [cell for cell in cells if cell["role"] == "target"]
    control_cells = [cell for cell in cells if cell["role"] == "control"]
    for split in SPLITS:
        if len([cell for cell in target_cells if cell["split"] == split]) != 20:
            raise RuntimeError(f"{split} target-cell census mismatch")
        if len([cell for cell in control_cells if cell["split"] == split]) != 32:
            raise RuntimeError(f"{split} control-cell census mismatch")

    structural_identities: list[dict[str, str]] = []
    for cell in cells:
        identities: tuple[tuple[str, str], ...] = ()
        if cell["family"] == F_SELECTOR:
            identities = (("payload", "replay"), ("joint", "score"))
        elif cell["control_kind"] == "lag":
            identities = (("payload", "replay"), ("joint", "score"))
        elif cell["family"] == F_MATCH and cell["direction"] == "base_to_donor":
            identities = (("joint", "score"),)
        elif cell["family"] == F_MATCH and cell["direction"] == "donor_to_base":
            identities = (("payload", "replay"),)
        for left, right in identities:
            structural_identities.append(
                {"cell_id": str(cell["cell_id"]), "left_arm": left, "right_arm": right}
            )

    control_arm_cells = [
        {**cell, "arm": arm}
        for cell in control_cells
        for arm in ARMS
    ]
    eligible_control_arm_cells = [
        cell
        for cell in control_arm_cells
        if not (cell["control_kind"] == "lag" and cell["arm"] == "payload")
    ]
    coverage_keys = [
        {"split": split, "arm": arm, "direction": direction, "recipient_condition": condition}
        for split in SPLITS
        for arm in ARMS
        for direction in DIRECTIONS
        for condition in CONDITIONS
    ]
    for split in SPLITS:
        if len([cell for cell in eligible_control_arm_cells if cell["split"] == split]) != 88:
            raise RuntimeError(f"{split} eligible-control-arm census mismatch")
        if len([key for key in coverage_keys if key["split"] == split]) != 24:
            raise RuntimeError(f"{split} coverage-key census mismatch")
        if len([identity for identity in structural_identities if identity["cell_id"].startswith(split + "|")]) != 32:
            raise RuntimeError(f"{split} structural-identity census mismatch")

    return {
        "target_cells": target_cells,
        "control_cells": control_cells,
        "coverage_keys": coverage_keys,
        "eligible_control_arm_cells": eligible_control_arm_cells,
        "structural_identities": structural_identities,
    }


def build_control_scale_lookup(manifests: Mapping[str, Sequence[Mapping[str, object]]] | None = None) -> list[dict[str, str]]:
    """Map every control arm cell to its unique, outcome-blind FIT target scale."""
    manifests = build_cell_manifests() if manifests is None else manifests
    target_cells = list(manifests["target_cells"])
    target_family = {"score": F_SELECTOR, "payload": F_PAYLOAD, "joint": F_SELECTOR}
    lookup: list[dict[str, str]] = []
    control_arm_cells = [
        {**control, "arm": arm}
        for control in manifests["control_cells"]
        for arm in ARMS
    ]
    for control in control_arm_cells:
        arm = str(control["arm"])
        matches = [
            target
            for target in target_cells
            if target["split"] == "FIT"
            and target["family"] == target_family[arm]
            and target["recipient_condition"] == control["recipient_condition"]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"non-unique target-scale mapping for {control['cell_id']}:{arm}")
        target = matches[0]
        lookup.append(
            {
                "split": str(control["split"]),
                "control_cell_id": str(control["cell_id"]),
                "arm": arm,
                "recipient_condition": str(control["recipient_condition"]),
                "fit_target_cell_id": str(target["cell_id"]),
                "fit_target_arm": arm,
            }
        )
    return lookup


def bootstrap_cell_id(cell: Mapping[str, object], arm: str, metric: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"invalid bootstrap arm: {arm}")
    if metric not in BOOTSTRAP_METRICS:
        raise ValueError(f"invalid bootstrap metric: {metric}")
    return "|".join(
        str(cell[field])
        for field in ("split", "family", "variant", "recipient_condition", "direction")
    ) + f"|{arm}|{metric}"


def expected_bootstrap_cells(manifests: Mapping[str, Sequence[Mapping[str, object]]] | None = None) -> list[dict[str, object]]:
    """Enumerate every and only applicable R585 bootstrap statistic cell."""
    manifests = build_cell_manifests() if manifests is None else manifests
    output: list[dict[str, object]] = []
    for cell in manifests["target_cells"]:
        family = cell["family"]
        specs: list[tuple[str, str]] = []
        if family == F_SELECTOR:
            specs = [(arm, metric) for arm in ("score", "joint") for metric in (
                "denominator_mean", "numerator_mean", "donor_ce_mean")]
        elif family == F_PAYLOAD:
            specs = [("score", metric) for metric in ("denominator_mean", "numerator_mean")]
            specs += [(arm, metric) for arm in ("payload", "joint") for metric in (
                "denominator_mean", "numerator_mean", "donor_ce_mean")]
        elif family == F_MATCH:
            specs = [(arm, metric) for arm in ("score", "joint") for metric in (
                "denominator_mean", "numerator_mean", "donor_ce_mean")]
            if cell["direction"] == "base_to_donor":
                specs += [("payload", metric) for metric in ("denominator_mean", "numerator_mean")]
        elif family == F_JOINT:
            specs = [
                ("score", "single_score_harm_mean"),
                ("payload", "single_payload_harm_mean"),
                ("joint", "factorial_interaction_mean"),
            ]
        else:
            raise RuntimeError(f"unexpected target family: {family}")
        for arm, metric in specs:
            output.append(
                {
                    "cell_id": bootstrap_cell_id(cell, arm, metric),
                    "group_ids": list(cell["group_ids"]),
                }
            )
    output.sort(key=lambda value: value["cell_id"])
    if len(output) != 248 or len({value["cell_id"] for value in output}) != 248:
        raise RuntimeError("bootstrap-cell census mismatch")
    if any(sum(value["cell_id"].startswith(split + "|") for value in output) != 124 for split in SPLITS):
        raise RuntimeError("per-split bootstrap-cell census mismatch")
    return output


def bootstrap_draw_index(cell_id: str, replicate: int, draw: int, group_count: int) -> int:
    """Return SHA-defined group index using first eight digest bytes, big-endian."""
    if replicate < 0 or draw < 0 or group_count <= 0 or group_count > 65_535:
        raise ValueError("invalid bootstrap index arguments")
    payload = f"{BOOTSTRAP_NAMESPACE}:{cell_id}:{replicate}:{draw}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % group_count


def bootstrap_draw_matrix(cell_id: str, group_ids: Iterable[str], replicates: int = BOOTSTRAPS) -> tuple[tuple[int, ...], ...]:
    groups = tuple(sorted(group_ids))
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("bootstrap group IDs must be nonempty and unique")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    return tuple(
        tuple(bootstrap_draw_index(cell_id, b, k, len(groups)) for k in range(len(groups)))
        for b in range(replicates)
    )


def big_endian_uint16_matrix_sha256(matrix: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    width: int | None = None
    for row in matrix:
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("ragged draw matrix")
        for value in row:
            if value < 0 or value > 65_535:
                raise ValueError("draw index does not fit uint16")
            digest.update(struct.pack(">H", value))
    if width is None:
        raise ValueError("empty draw matrix")
    return digest.hexdigest()


def phase_accounting(authority: Mapping[str, Sequence[Mapping[str, object]]] | None = None) -> dict[str, object]:
    authority = build_authority_manifest() if authority is None else authority
    phases: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        endpoints = sum(record["split"] == split for record in authority["endpoints"])
        directions = sum(record["split"] == split for record in authority["directions"])
        capture = math.ceil(endpoints / BATCH_SIZE)
        interventions_per_arm = math.ceil(directions / BATCH_SIZE)
        comparator = capture
        total = capture + len(ARMS) * interventions_per_arm + comparator
        phases[split] = {
            "unique_endpoints": endpoints,
            "directed_pairs": directions,
            "capture_replay_forwards": capture,
            "intervention_forwards_per_arm": interventions_per_arm,
            "intervention_forwards": len(ARMS) * interventions_per_arm,
            "native_comparator_forwards": comparator,
            "phase_max_forwards": total,
        }
    if phases["FIT"]["phase_max_forwards"] != 459 or phases["SELECT"]["phase_max_forwards"] != 231:
        raise RuntimeError("phase accounting changed")
    return {
        "batch_size": BATCH_SIZE,
        "arms": list(ARMS),
        "phases": phases,
        "total_max_forwards": phases["FIT"]["phase_max_forwards"] + phases["SELECT"]["phase_max_forwards"],
        "model_backwards": 0,
        "weight_updates": 0,
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def validate_dependency_lock(
    lock: Mapping[str, object], observed_artifact_sha256: Mapping[str, str]
) -> dict[str, object]:
    """Validate a caller-supplied dependency fixture without filesystem reads.

    Structurally valid held and null fixtures are both accepted.  Only the exact
    held verdict pair makes ``runnable`` true.  Artifact hashes are checked
    against the caller-supplied map, never by opening the declared paths.
    """
    missing = sorted(DEPENDENCY_LOCK_KEYS - set(lock))
    extra = sorted(set(lock) - DEPENDENCY_LOCK_KEYS)
    if missing or extra:
        raise ValueError(f"dependency-lock key mismatch: missing={missing}, extra={extra}")
    if lock["schema"] != DEPENDENCY_LOCK_SCHEMA:
        raise ValueError("dependency-lock schema mismatch")
    if lock["r578_rows_sha256"] != ROWS_SHA256:
        raise ValueError("dependency lock has wrong R578 authority")
    if lock["replacement_amendment_sha256"] != AMENDMENT_SHA256:
        raise ValueError("dependency lock has wrong replacement amendment")
    if lock["evaluated_splits"] != ["FIT", "SELECT"]:
        raise ValueError("dependency lock did not evaluate exactly FIT and SELECT")
    if lock["forbidden_splits_opened"] != []:
        raise ValueError("dependency lock opened a forbidden split")
    for prefix in ("r586_result", "r586_receipt", "r587_audit"):
        path = lock[prefix + "_path"]
        expected = lock[prefix + "_sha256"]
        if not isinstance(path, str) or not path:
            raise ValueError(f"invalid dependency path: {prefix}")
        if not _is_sha256(expected):
            raise ValueError(f"invalid dependency digest: {prefix}")
        if observed_artifact_sha256.get(path) != expected:
            raise ValueError(f"caller-supplied dependency digest mismatch: {prefix}")
    for key in ("r586_verdict", "r587_audit_verdict"):
        if not isinstance(lock[key], str) or not lock[key]:
            raise ValueError(f"invalid dependency verdict: {key}")
    reasons = []
    if lock["r586_verdict"] != "held_capability_screen":
        reasons.append("r586_not_held")
    if lock["r587_audit_verdict"] != "held_independent_audit":
        reasons.append("r587_audit_not_held")
    return {
        "schema_valid": True,
        "hashes_valid": True,
        "runnable": not reasons,
        "terminal": "dependency_held" if not reasons else "not_executed_upstream_dependency",
        "reasons": reasons,
    }


def build_planted_dependency_fixture(held: bool) -> tuple[dict[str, object], dict[str, str]]:
    """Create deterministic fake inputs for tests/dry-run; no live outcomes."""
    paths = {
        "r586_result": "fixture://r586/result.json",
        "r586_receipt": "fixture://r586/receipt.json",
        "r587_audit": "fixture://r587/audit.json",
    }
    hashes = {path: sha256_bytes(f"planted:{name}".encode()) for name, path in paths.items()}
    lock: dict[str, object] = {
        "schema": DEPENDENCY_LOCK_SCHEMA,
        "r578_rows_sha256": ROWS_SHA256,
        "replacement_amendment_sha256": AMENDMENT_SHA256,
        "r586_result_path": paths["r586_result"],
        "r586_result_sha256": hashes[paths["r586_result"]],
        "r586_receipt_path": paths["r586_receipt"],
        "r586_receipt_sha256": hashes[paths["r586_receipt"]],
        "r586_verdict": "held_capability_screen" if held else "scientific_null",
        "r587_audit_path": paths["r587_audit"],
        "r587_audit_sha256": hashes[paths["r587_audit"]],
        # A correctly audited scientific null still has a held audit envelope;
        # its non-runnable status comes from the R586 scientific verdict.
        "r587_audit_verdict": "held_independent_audit",
        "evaluated_splits": ["FIT", "SELECT"],
        "forbidden_splits_opened": [],
    }
    return lock, hashes


def build_dryrun() -> dict[str, object]:
    authority = build_authority_manifest()
    manifests = build_cell_manifests(authority)
    scale_lookup = build_control_scale_lookup(manifests)
    bootstrap_cells = expected_bootstrap_cells(manifests)
    held_lock, held_hashes = build_planted_dependency_fixture(True)
    null_lock, null_hashes = build_planted_dependency_fixture(False)
    sentinel = bootstrap_cells[0]
    sentinel_draws = bootstrap_draw_matrix(
        str(sentinel["cell_id"]), sentinel["group_ids"], replicates=BOOTSTRAPS
    )
    return {
        "schema": "induction_selector_payload_frozen_factor_rung585_manifest_dryrun_v1",
        "model_loaded": False,
        "outcomes_opened": [],
        "authorities": {"r578_rows": ROWS_SHA256, "replacement_amendment": AMENDMENT_SHA256},
        "authority_counts": {
            split: {
                "rows": sum(row["split"] == split for row in authority["rows"]),
                "endpoints": sum(row["split"] == split for row in authority["endpoints"]),
                "directions": sum(row["split"] == split for row in authority["directions"]),
            }
            for split in SPLITS
        },
        "manifest_counts": {
            split: {
                "target_cells": sum(cell["split"] == split for cell in manifests["target_cells"]),
                "control_cells": sum(cell["split"] == split for cell in manifests["control_cells"]),
                "coverage_keys": sum(key["split"] == split for key in manifests["coverage_keys"]),
                "eligible_control_arm_cells": sum(
                    cell["split"] == split for cell in manifests["eligible_control_arm_cells"]
                ),
                "structural_identities": sum(
                    identity["cell_id"].startswith(split + "|")
                    for identity in manifests["structural_identities"]
                ),
                "bootstrap_cells": sum(
                    cell["cell_id"].startswith(split + "|") for cell in bootstrap_cells
                ),
            }
            for split in SPLITS
        },
        "direction_manifest_sha256": sha256_bytes(canonical_json_bytes(authority["directions"])),
        "target_cell_manifest_sha256": sha256_bytes(canonical_json_bytes(manifests["target_cells"])),
        "control_cell_manifest_sha256": sha256_bytes(canonical_json_bytes(manifests["control_cells"])),
        "structural_identity_manifest_sha256": sha256_bytes(
            canonical_json_bytes(manifests["structural_identities"])
        ),
        "control_scale_lookup_count": len(scale_lookup),
        "control_scale_lookup_sha256": sha256_bytes(canonical_json_bytes(scale_lookup)),
        "bootstrap_cell_ids_sha256": sha256_bytes(
            canonical_json_bytes([cell["cell_id"] for cell in bootstrap_cells])
        ),
        "bootstrap_sentinel": {
            "cell_id": sentinel["cell_id"],
            "group_count": len(sentinel["group_ids"]),
            "replicates": BOOTSTRAPS,
            "big_endian_uint16_draw_sha256": big_endian_uint16_matrix_sha256(sentinel_draws),
        },
        "phase_accounting": phase_accounting(authority),
        "planted_dependency_checks": {
            "held": validate_dependency_lock(held_lock, held_hashes),
            "null": validate_dependency_lock(null_lock, null_hashes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=DRYRUN_PATH)
    args = parser.parse_args()
    if not args.dry_run:
        parser.error("only --dry-run is supported; model execution is intentionally absent")
    payload = build_dryrun()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
