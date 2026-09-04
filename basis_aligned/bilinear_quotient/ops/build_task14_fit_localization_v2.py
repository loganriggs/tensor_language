#!/usr/bin/env python3
"""Build and validate the CPU-only task14 FIT localization-v2 authorities.

This module reads only the frozen FIT prompt authority.  It has no model, checkpoint,
activation, CUDA, queue, result, or later-phase dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Iterable


SCHEMA_PARTITION = "task14_fit_localization_partition_v2"
SCHEMA_DONORS = "task14_fit_localization_donors_v2"
TASK_ID = "subject_verb.number_agreement"
PHASE = "FIT"
VERSION = 2

FROZEN_V1_COMMIT = "7986557ece6ee117cd40842fc02c9cf8d21149a5"
FROZEN_V1_PREREG_SHA256 = "6fb4b00080d9bf4b1eaec5953b2806b4a8c2fcc7323a2f938ce7f53192734e6e"
FROZEN_REVIEW_COMMIT = "52884a4691c3f388c4b0ba0c1327a39f1c0ef411"
FROZEN_REVIEW_SHA256 = "d4d7ac9b76d54eee73278a2af903c8c34472bcc27917b28c997475d50eab3da2"
FROZEN_AUTHORITY_FILE_SHA256 = "e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f"
FROZEN_FIT_ROWS_SHA256 = "3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1"
FROZEN_FULL_AUTHORITY_SHA256 = "1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1"
FROZEN_GENERATOR_SHA256 = "33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94"

PARTITION_SEED = "task14-fit-localization-v1|discovery-validation-pair-coherent"
ORIGINAL_DONOR_SEED = "task14-fit-localization-v1|donors"
COMPLETE_SUBJECT_SEED = "task14-fit-localization-v2|complete-subject-Q-donors"
DISCOVERY_GROUPS = (0, 1, 4, 6, 9, 10, 11, 15, 16, 17, 20, 22, 25, 26, 27, 31)
VALIDATION_GROUPS = (2, 3, 5, 7, 8, 12, 13, 14, 18, 19, 21, 23, 24, 28, 29, 30)
ORIGINAL_704_CORE_SHA256 = "25a1f09d5947301f573b223abfbcae1699555ddf809f2b137eabffcbe776f3dc"

HERE = Path(__file__).resolve().parent
AUTHORITY_PATH = HERE / "circuit_battery_task14_agreement_fit_authority.json"
PARTITION_PATH = HERE / "circuit_battery_task14_fit_localization_partition_v2.json"
DONORS_PATH = HERE / "circuit_battery_task14_fit_localization_donors_v2.json"

PARTITION_ORDER = ("DISCOVERY", "VALIDATION")
SOURCE_ORDER = ("v1_original_704", "v2_complete_subject_Q")
ORDINARY_FAMILIES = ("A1", "A2")


def canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return data + (b"\n" if newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        data = b""
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            data += chunk
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"file changed during read: {path}")
        return data
    finally:
        os.close(fd)


def load_frozen_authority(path: Path = AUTHORITY_PATH) -> dict[str, Any]:
    raw = safe_read_regular(path)
    if bytes_sha256(raw) != FROZEN_AUTHORITY_FILE_SHA256:
        raise ValueError("task14 FIT authority file hash mismatch")
    value = json.loads(raw)
    if set(value) != {
        "groups",
        "rows",
        "schema",
        "split",
        "split_records_sha256",
        "task14_authority_sha256",
        "task_id",
    }:
        raise ValueError("task14 FIT authority wrapper keys mismatch")
    if value["split"] != PHASE or value["task_id"] != TASK_ID or value["groups"] != 32:
        raise ValueError("task14 FIT authority wrapper metadata mismatch")
    if value["split_records_sha256"] != FROZEN_FIT_ROWS_SHA256:
        raise ValueError("task14 FIT logical-row hash mismatch")
    if value["task14_authority_sha256"] != FROZEN_FULL_AUTHORITY_SHA256:
        raise ValueError("task14 full-authority hash mismatch")
    rows = value["rows"]
    if len(rows) != 128 or len({r["row_id"] for r in rows}) != 128:
        raise ValueError("task14 FIT authority row census mismatch")
    panels: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row["split"] != PHASE or row["task_id"] != TASK_ID:
            raise ValueError("non-FIT or wrong-task row in FIT authority")
        panels.setdefault(row["group_number"], []).append(row)
    if set(panels) != set(range(32)):
        raise ValueError("task14 FIT group numbers mismatch")
    for group_number, panel in panels.items():
        if {r["transform_id"] for r in panel} != {"A1", "A2", "P", "C"}:
            raise ValueError(f"incomplete panel at group {group_number}")
        if len({r["group_id"] for r in panel}) != 1:
            raise ValueError(f"group-id mismatch at group {group_number}")
    return value


def _row_index(authority: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(r["group_number"], r["transform_id"]): r for r in authority["rows"]}


def build_partition(authority: dict[str, Any], *, _validate: bool = True) -> dict[str, Any]:
    idx = _row_index(authority)
    selected_base: set[int] = set()
    stratum_orders: dict[str, list[int]] = {}
    for parity, literal in ((0, "even_congruent_cycle"), (1, "odd_incongruent_cycle")):
        base_groups = [g for g in range(16) if g % 2 == parity]
        base_groups.sort(
            key=lambda g: hashlib.sha256(
                (
                    PARTITION_SEED
                    + "|"
                    + idx[(g, "A1")]["group_id"]
                    + "|"
                    + idx[(g + 16, "A1")]["group_id"]
                ).encode("utf-8")
            ).hexdigest()
        )
        stratum_orders[literal] = base_groups
        selected_base.update(base_groups[:4])
    discovery = tuple(sorted(selected_base | {g + 16 for g in selected_base}))
    validation = tuple(sorted(set(range(32)) - set(discovery)))
    if discovery != DISCOVERY_GROUPS or validation != VALIDATION_GROUPS:
        raise ValueError("pair-coherent partition drift")

    records = []
    for partition, groups in (("DISCOVERY", discovery), ("VALIDATION", validation)):
        for group_number in groups:
            mirror_base = group_number % 16
            records.append(
                {
                    "group_id": idx[(group_number, "A1")]["group_id"],
                    "group_number": group_number,
                    "mirror_base_group_number": mirror_base,
                    "mirror_group_numbers": [mirror_base, mirror_base + 16],
                    "partition": partition,
                    "stratum": "even_congruent_cycle" if mirror_base % 2 == 0 else "odd_incongruent_cycle",
                }
            )

    result = {
        "authority_file_sha256": FROZEN_AUTHORITY_FILE_SHA256,
        "fit_logical_rows_sha256": FROZEN_FIT_ROWS_SHA256,
        "full_logical_authority_sha256": FROZEN_FULL_AUTHORITY_SHA256,
        "generator_sha256": FROZEN_GENERATOR_SHA256,
        "partition_order": list(PARTITION_ORDER),
        "records": records,
        "records_sha256": canonical_sha256(records),
        "schema": SCHEMA_PARTITION,
        "seed_label": PARTITION_SEED,
        "selection_rule": {
            "indivisible_unit": "{g,g+16} for g=0..15",
            "strata": ["g even", "g odd"],
            "within_stratum_order": "ascending SHA256(seed_label|group_id(g)|group_id(g+16))",
            "discovery_units_per_stratum": 4,
            "validation_units_per_stratum": 4,
        },
        "source_review_commit": FROZEN_REVIEW_COMMIT,
        "source_review_sha256": FROZEN_REVIEW_SHA256,
        "source_v1_commit": FROZEN_V1_COMMIT,
        "source_v1_prereg_sha256": FROZEN_V1_PREREG_SHA256,
        "split": PHASE,
        "task_id": TASK_ID,
        "version": VERSION,
    }
    if _validate:
        validate_partition(result, authority)
    return result


def _endpoint(row: dict[str, Any], side: str) -> dict[str, Any]:
    if side not in {"base", "donor"}:
        raise ValueError("endpoint side must be base or donor")
    subject = row[f"{side}_subject_number"]
    if subject not in {"singular", "plural"}:
        raise ValueError("bad endpoint subject number")
    return {
        "attractor_plural": bool(row[f"{side}_attractor_plural"]),
        "endpoint_id": f"{row['row_id']}:{side}",
        "family": row["transform_id"],
        "group_id": row["group_id"],
        "group_number": row["group_number"],
        "head_pair": list(row["head_pair"]),
        "prompt_sha256": hashlib.sha256(row[f"{side}_text"].encode("utf-8")).hexdigest(),
        "row_id": row["row_id"],
        "side": side,
        "subject_state": 1 if subject == "plural" else -1,
    }


def _endpoints(authority: dict[str, Any]) -> list[dict[str, Any]]:
    return [_endpoint(row, side) for row in authority["rows"] for side in ("base", "donor")]


def _sha_rank(label: str, target_id: str, candidate_id: str) -> str:
    return hashlib.sha256((label + "|" + target_id + "|" + candidate_id).encode("utf-8")).hexdigest()


def _different_lexical_subject(target: dict[str, Any], donor: dict[str, Any]) -> bool:
    return target["head_pair"] != donor["head_pair"]


def _choose_one(
    candidates: Iterable[dict[str, Any]], *, label: str, target: dict[str, Any]
) -> dict[str, Any]:
    candidates = list(candidates)
    if not candidates:
        raise ValueError(f"empty donor pool for {label} / {target['endpoint_id']}")
    return min(candidates, key=lambda e: _sha_rank(label, target["endpoint_id"], e["endpoint_id"]))


def _original_704_core(
    authority: dict[str, Any], partition_by_group: dict[int, str]
) -> list[dict[str, str]]:
    rows = authority["rows"]
    all_endpoints = _endpoints(authority)
    records: list[dict[str, str]] = []
    for partition in PARTITION_ORDER:
        groups = {g for g, p in partition_by_group.items() if p == partition}
        endpoints = [e for e in all_endpoints if e["group_number"] in groups]
        for family in ORDINARY_FAMILIES:
            targets = sorted((e for e in endpoints if e["family"] == family), key=lambda e: e["endpoint_id"])
            for target in targets:
                other_side = "donor" if target["side"] == "base" else "base"
                records.append(
                    {
                        "partition": partition,
                        "arm": "answer_change",
                        "family": family,
                        "matching": "paired",
                        "target": target["endpoint_id"],
                        "donor": f"{target['row_id']}:{other_side}",
                    }
                )
                candidates = [
                    e
                    for e in endpoints
                    if e["family"] == family
                    and e["subject_state"] != target["subject_state"]
                    and e["attractor_plural"] == target["attractor_plural"]
                    and e["group_number"] != target["group_number"]
                ]
                candidates.sort(
                    key=lambda e: hashlib.sha256(
                        (
                            ORIGINAL_DONOR_SEED
                            + "|same|"
                            + partition
                            + "|"
                            + family
                            + "|"
                            + target["endpoint_id"]
                            + "|"
                            + e["endpoint_id"]
                        ).encode("utf-8")
                    ).hexdigest()
                )
                if len(candidates) < 2:
                    raise ValueError("original same-syntax donor pool too small")
                for number, donor in enumerate(candidates[:2], 1):
                    records.append(
                        {
                            "partition": partition,
                            "arm": "answer_change",
                            "family": family,
                            "matching": f"cross_noun_{number}",
                            "target": target["endpoint_id"],
                            "donor": donor["endpoint_id"],
                        }
                    )
                other_family = "A2" if family == "A1" else "A1"
                candidates = [
                    e
                    for e in endpoints
                    if e["family"] == other_family
                    and e["subject_state"] != target["subject_state"]
                    and e["attractor_plural"] == target["attractor_plural"]
                    and e["group_number"] != target["group_number"]
                ]
                donor = min(
                    candidates,
                    key=lambda e: hashlib.sha256(
                        (
                            ORIGINAL_DONOR_SEED
                            + "|syntax|"
                            + partition
                            + "|"
                            + family
                            + "|"
                            + target["endpoint_id"]
                            + "|"
                            + e["endpoint_id"]
                        ).encode("utf-8")
                    ).hexdigest(),
                )
                records.append(
                    {
                        "partition": partition,
                        "arm": "cross_syntax",
                        "family": family,
                        "matching": "cross_syntax_1",
                        "target": target["endpoint_id"],
                        "donor": donor["endpoint_id"],
                    }
                )
        for target in sorted((e for e in endpoints if e["family"] == "P"), key=lambda e: e["endpoint_id"]):
            candidates = [
                e
                for e in endpoints
                if e["family"] == "P"
                and e["subject_state"] != target["subject_state"]
                and e["attractor_plural"] == target["attractor_plural"]
                and e["group_number"] != target["group_number"]
            ]
            candidates.sort(
                key=lambda e: hashlib.sha256(
                    (
                        ORIGINAL_DONOR_SEED
                        + "|p|"
                        + partition
                        + "|"
                        + target["endpoint_id"]
                        + "|"
                        + e["endpoint_id"]
                    ).encode("utf-8")
                ).hexdigest()
            )
            if len(candidates) < 2:
                raise ValueError("original P donor pool too small")
            for number, donor in enumerate(candidates[:2], 1):
                records.append(
                    {
                        "partition": partition,
                        "arm": "P_positive_transfer",
                        "family": "P",
                        "matching": f"cross_noun_{number}",
                        "target": target["endpoint_id"],
                        "donor": donor["endpoint_id"],
                    }
                )
        for family in ("P", "C"):
            selected_rows = sorted(
                (r for r in rows if r["transform_id"] == family and r["group_number"] in groups),
                key=lambda r: r["row_id"],
            )
            for row in selected_rows:
                records.append(
                    {
                        "partition": partition,
                        "arm": f"{family}_zero_coordinate_control",
                        "family": family,
                        "matching": "paired",
                        "target": f"{row['row_id']}:base",
                        "donor": f"{row['row_id']}:donor",
                    }
                )
    records.sort(key=lambda r: (r["partition"], r["arm"], r["family"], r["target"], r["matching"]))
    envelope = {
        "schema": "task14_fit_localization_donors_v1",
        "authority_sha256": FROZEN_FIT_ROWS_SHA256,
        "seed_label": ORIGINAL_DONOR_SEED,
        "records": records,
    }
    if len(records) != 704 or canonical_sha256(envelope) != ORIGINAL_704_CORE_SHA256:
        raise ValueError("original 704-donor relation drift")
    return records


def _expanded_record(
    core: dict[str, str], endpoint_by_id: dict[str, dict[str, Any]], *, source_contract: str, q_only: bool
) -> dict[str, Any]:
    target = endpoint_by_id[core["target"]]
    donor = endpoint_by_id[core["donor"]]
    opposite = target["subject_state"] != donor["subject_state"]
    relation = "opposite_subject_toward_donor" if opposite else "same_subject_zero_projected_effect"
    identity = {
        "arm": core["arm"],
        "donor_endpoint_id": donor["endpoint_id"],
        "expected_relation": relation,
        "family": core["family"],
        "matching": core["matching"],
        "partition": core["partition"],
        "q_only": q_only,
        "source_contract": source_contract,
        "target_endpoint_id": target["endpoint_id"],
    }
    return {
        **identity,
        "record_id": canonical_sha256(identity),
    }


def _complete_subject_core(
    endpoints: list[dict[str, Any]], partition_by_group: dict[int, str]
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for partition in PARTITION_ORDER:
        eps = [e for e in endpoints if partition_by_group[e["group_number"]] == partition]
        c_endpoints = sorted((e for e in eps if e["family"] == "C"), key=lambda e: e["endpoint_id"])
        ordinary = [e for e in eps if e["family"] in ORDINARY_FAMILIES]
        for target in c_endpoints:
            for family in ORDINARY_FAMILIES:
                for donor_state, arm, relation_label in (
                    (-1, "C_to_ordinary_singular", "c_to_singular"),
                    (1, "C_to_ordinary_plural_control", "c_to_plural"),
                ):
                    candidates = [
                        e
                        for e in ordinary
                        if e["family"] == family
                        and e["subject_state"] == donor_state
                        and e["attractor_plural"] == target["attractor_plural"]
                        and e["group_number"] != target["group_number"]
                        and _different_lexical_subject(target, e)
                    ]
                    donor = _choose_one(
                        candidates,
                        label=f"{COMPLETE_SUBJECT_SEED}|{relation_label}|{partition}|{family}",
                        target=target,
                    )
                    records.append(
                        {
                            "partition": partition,
                            "arm": arm,
                            "family": "C",
                            "matching": f"donor_{family}_sha1",
                            "target": target["endpoint_id"],
                            "donor": donor["endpoint_id"],
                        }
                    )
        for target in sorted(ordinary, key=lambda e: e["endpoint_id"]):
            if target["subject_state"] == -1:
                arm, relation_label = "ordinary_singular_to_C", "singular_to_c"
            else:
                arm, relation_label = "ordinary_plural_to_C_control", "plural_to_c"
            candidates = [
                e
                for e in c_endpoints
                if e["attractor_plural"] == target["attractor_plural"]
                and e["group_number"] != target["group_number"]
                and _different_lexical_subject(target, e)
            ]
            donor = _choose_one(
                candidates,
                label=f"{COMPLETE_SUBJECT_SEED}|{relation_label}|{partition}|{target['family']}",
                target=target,
            )
            records.append(
                {
                    "partition": partition,
                    "arm": arm,
                    "family": target["family"],
                    "matching": "donor_C_sha1",
                    "target": target["endpoint_id"],
                    "donor": donor["endpoint_id"],
                }
            )
    if len(records) != 384:
        raise ValueError("complete-subject donor census drift")
    return records


def build_donors(
    authority: dict[str, Any], partition: dict[str, Any], *, _validate: bool = True
) -> dict[str, Any]:
    partition_by_group = {r["group_number"]: r["partition"] for r in partition["records"]}
    endpoints = _endpoints(authority)
    endpoint_by_id = {e["endpoint_id"]: e for e in endpoints}
    if len(endpoint_by_id) != 256:
        raise ValueError("endpoint identity collision")

    original_core = _original_704_core(authority, partition_by_group)
    expanded = [
        _expanded_record(
            record,
            endpoint_by_id,
            source_contract="v1_original_704",
            q_only=record["family"] == "C",
        )
        for record in original_core
    ]
    complete_core = _complete_subject_core(endpoints, partition_by_group)
    expanded.extend(
        _expanded_record(record, endpoint_by_id, source_contract="v2_complete_subject_Q", q_only=True)
        for record in complete_core
    )
    expanded.sort(
        key=lambda r: (
            PARTITION_ORDER.index(r["partition"]),
            SOURCE_ORDER.index(r["source_contract"]),
            r["arm"],
            r["family"],
            r["target_endpoint_id"],
            r["matching"],
            r["donor_endpoint_id"],
        )
    )
    for ordinal, record in enumerate(expanded):
        record["ordinal"] = ordinal

    partition_bytes = canonical_bytes(partition, newline=True)
    endpoint_records = sorted(endpoints, key=lambda e: e["endpoint_id"])
    result = {
        "authority_file_sha256": FROZEN_AUTHORITY_FILE_SHA256,
        "complete_subject_seed_label": COMPLETE_SUBJECT_SEED,
        "endpoints": endpoint_records,
        "endpoints_sha256": canonical_sha256(endpoint_records),
        "fit_logical_rows_sha256": FROZEN_FIT_ROWS_SHA256,
        "original_704_core_envelope": {
            "authority_sha256": FROZEN_FIT_ROWS_SHA256,
            "schema": "task14_fit_localization_donors_v1",
            "seed_label": ORIGINAL_DONOR_SEED,
        },
        "original_704_core_sha256": ORIGINAL_704_CORE_SHA256,
        "partition_artifact_sha256": bytes_sha256(partition_bytes),
        "partition_order": list(PARTITION_ORDER),
        "record_order": [
            "partition_order index",
            "source_contract_order index",
            "arm lexicographic",
            "family lexicographic",
            "target_endpoint_id lexicographic",
            "matching lexicographic",
            "donor_endpoint_id lexicographic",
        ],
        "records": expanded,
        "records_sha256": canonical_sha256(expanded),
        "schema": SCHEMA_DONORS,
        "source_contract_order": list(SOURCE_ORDER),
        "source_review_commit": FROZEN_REVIEW_COMMIT,
        "source_review_sha256": FROZEN_REVIEW_SHA256,
        "source_v1_commit": FROZEN_V1_COMMIT,
        "source_v1_prereg_sha256": FROZEN_V1_PREREG_SHA256,
        "split": PHASE,
        "task_id": TASK_ID,
        "version": VERSION,
    }
    if _validate:
        validate_donors(result, authority, partition)
    return result


def validate_partition(value: dict[str, Any], authority: dict[str, Any]) -> None:
    expected = build_partition(authority, _validate=False)
    if value != expected:
        raise ValueError("partition differs from exact deterministic authority")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != 32:
        raise ValueError("partition record census mismatch")
    if value.get("records_sha256") != canonical_sha256(records):
        raise ValueError("partition record digest mismatch")
    if [r["partition"] for r in records] != ["DISCOVERY"] * 16 + ["VALIDATION"] * 16:
        raise ValueError("partition record order mismatch")
    by_group = {r["group_number"]: r for r in records}
    if len(by_group) != 32:
        raise ValueError("duplicate partition group")
    if tuple(sorted(g for g, r in by_group.items() if r["partition"] == "DISCOVERY")) != DISCOVERY_GROUPS:
        raise ValueError("discovery membership mismatch")
    if tuple(sorted(g for g, r in by_group.items() if r["partition"] == "VALIDATION")) != VALIDATION_GROUPS:
        raise ValueError("validation membership mismatch")
    idx = _row_index(authority)
    for group_number, record in by_group.items():
        mirror = group_number % 16
        if record["group_id"] != idx[(group_number, "A1")]["group_id"]:
            raise ValueError("partition group-id mismatch")
        if record["mirror_base_group_number"] != mirror or record["mirror_group_numbers"] != [mirror, mirror + 16]:
            raise ValueError("partition mirror metadata mismatch")
        if by_group[mirror]["partition"] != by_group[mirror + 16]["partition"]:
            raise ValueError("mirror unit crosses partition")


def validate_donors(
    value: dict[str, Any], authority: dict[str, Any], partition: dict[str, Any]
) -> None:
    expected = build_donors(authority, partition, _validate=False)
    if value != expected:
        raise ValueError("donor manifest differs from exact deterministic authority")
    if value.get("schema") != SCHEMA_DONORS or value.get("task_id") != TASK_ID or value.get("split") != PHASE:
        raise ValueError("donor wrapper metadata mismatch")
    if value.get("partition_artifact_sha256") != bytes_sha256(canonical_bytes(partition, newline=True)):
        raise ValueError("donor partition binding mismatch")
    records = value.get("records")
    if not isinstance(records, list) or len(records) != 1088:
        raise ValueError("donor record census mismatch")
    if value.get("records_sha256") != canonical_sha256(records):
        raise ValueError("donor record digest mismatch")
    if [r["ordinal"] for r in records] != list(range(1088)):
        raise ValueError("donor ordinal/order mismatch")
    if len({r["record_id"] for r in records}) != 1088:
        raise ValueError("duplicate donor record id")
    expected_endpoints = sorted(_endpoints(authority), key=lambda e: e["endpoint_id"])
    if value.get("endpoints") != expected_endpoints:
        raise ValueError("donor endpoint authority mismatch")
    if value.get("endpoints_sha256") != canonical_sha256(expected_endpoints):
        raise ValueError("donor endpoint digest mismatch")
    endpoint_by_id = {e["endpoint_id"]: e for e in expected_endpoints}
    partition_by_group = {r["group_number"]: r["partition"] for r in partition["records"]}
    source_counts = {source: 0 for source in SOURCE_ORDER}
    arm_counts: dict[tuple[str, str], int] = {}
    for record in records:
        source_counts[record["source_contract"]] += 1
        arm_counts[(record["partition"], record["arm"])] = arm_counts.get((record["partition"], record["arm"]), 0) + 1
        target = endpoint_by_id.get(record["target_endpoint_id"])
        donor = endpoint_by_id.get(record["donor_endpoint_id"])
        if target is None or donor is None:
            raise ValueError("donor endpoint reference mismatch")
        if partition_by_group[target["group_number"]] != record["partition"] or partition_by_group[donor["group_number"]] != record["partition"]:
            raise ValueError("cross-partition donor record")
        if record["arm"] == "C_zero_coordinate_control":
            if target["attractor_plural"] == donor["attractor_plural"]:
                raise ValueError("paired C control must flip attractor number")
        elif target["attractor_plural"] != donor["attractor_plural"]:
            raise ValueError("donor attractor-number mismatch")
        expected_opposite = record["expected_relation"] == "opposite_subject_toward_donor"
        if expected_opposite != (target["subject_state"] != donor["subject_state"]):
            raise ValueError("donor subject relation mismatch")
        identity = {
            key: record[key]
            for key in (
                "arm",
                "donor_endpoint_id",
                "expected_relation",
                "family",
                "matching",
                "partition",
                "q_only",
                "source_contract",
                "target_endpoint_id",
            )
        }
        if record["record_id"] != canonical_sha256(identity):
            raise ValueError("donor record-id mismatch")
        if record["source_contract"] == "v2_complete_subject_Q":
            if not record["q_only"] or "C" not in {target["family"], donor["family"]}:
                raise ValueError("complete-subject relation is not Q-only C cross-construction")
            if target["family"] == donor["family"]:
                raise ValueError("complete-subject relation lacks construction change")
    if source_counts != {"v1_original_704": 704, "v2_complete_subject_Q": 384}:
        raise ValueError("donor source census mismatch")
    for partition_name in PARTITION_ORDER:
        expected = {
            "C_to_ordinary_singular": 64,
            "ordinary_singular_to_C": 32,
            "C_to_ordinary_plural_control": 64,
            "ordinary_plural_to_C_control": 32,
        }
        for arm, count in expected.items():
            if arm_counts.get((partition_name, arm)) != count:
                raise ValueError(f"complete-subject arm census mismatch: {partition_name}/{arm}")
        c_targets = {
            r["target_endpoint_id"]
            for r in records
            if r["partition"] == partition_name and r["arm"] == "C_to_ordinary_singular"
        }
        all_c = {
            e["endpoint_id"]
            for e in endpoint_by_id.values()
            if e["family"] == "C" and partition_by_group[e["group_number"]] == partition_name
        }
        if c_targets != all_c:
            raise ValueError("not every C endpoint has an affirmative singular donor")


def expected_artifacts(authority_path: Path = AUTHORITY_PATH) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = load_frozen_authority(authority_path)
    partition = build_partition(authority)
    donors = build_donors(authority, partition)
    return partition, donors


def check_artifacts(partition_path: Path = PARTITION_PATH, donors_path: Path = DONORS_PATH) -> None:
    expected_partition, expected_donors = expected_artifacts()
    if safe_read_regular(partition_path) != canonical_bytes(expected_partition, newline=True):
        raise ValueError("materialized partition bytes mismatch deterministic builder")
    if safe_read_regular(donors_path) != canonical_bytes(expected_donors, newline=True):
        raise ValueError("materialized donor bytes mismatch deterministic builder")


def write_artifacts(output_dir: Path) -> None:
    partition, donors = expected_artifacts()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        (PARTITION_PATH.name, partition),
        (DONORS_PATH.name, donors),
    ):
        path = output_dir / name
        with path.open("xb") as handle:
            handle.write(canonical_bytes(value, newline=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write-dir", type=Path)
    args = parser.parse_args()
    if args.check:
        check_artifacts()
        partition_raw = safe_read_regular(PARTITION_PATH)
        donor_raw = safe_read_regular(DONORS_PATH)
        print(
            json.dumps(
                {
                    "donor_artifact_sha256": bytes_sha256(donor_raw),
                    "donor_records": 1088,
                    "model_calls": 0,
                    "partition_artifact_sha256": bytes_sha256(partition_raw),
                    "partition_records": 32,
                    "status": "PASS",
                },
                sort_keys=True,
            )
        )
    else:
        write_artifacts(args.write_dir)


if __name__ == "__main__":
    main()
