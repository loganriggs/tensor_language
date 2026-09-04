#!/usr/bin/env python3
"""Build the token-bearing DISCOVERY-only endpoint shard for Task 14.

This CPU-only boundary is the only Program-A preparation step that parses the
full Task 14 authority.  Its output contains no prompt text and no VALIDATION
endpoint.  The production Program A should consume only the create-once shard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


OPS = Path(__file__).resolve().parent
SCHEMA = "task14_projector_discovery_endpoint_shard_v1"
TASK_ID = "subject_verb.number_agreement"
ANSWER_TOKEN_IDS = {" is": 318, " are": 389}

SOURCE_PATHS = {
    "authority": OPS / "circuit_battery_task14_agreement_fit_authority.json",
    "partition": OPS / "circuit_battery_task14_fit_localization_partition_v2.json",
    "donors": OPS / "circuit_battery_task14_fit_localization_donors_v2.json",
}
EXPECTED_SOURCE_SHA256 = {
    "authority": "e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f",
    "partition": "1f43b767fb39082d7872629d1a8b700e90e055c9529d9d319fe483f77d91fad3",
    "donors": "ff702f2936e2445a247c6fca3a55d177e80974b2a5e14fb6de0a5fe2761db50a",
}


class DiscoveryShardError(ValueError):
    """A source or derived shard violates the DISCOVERY-only contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_sources(
    source_paths: Mapping[str, Path], expected_sha256: Mapping[str, str]
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, str]]:
    if set(source_paths) != set(EXPECTED_SOURCE_SHA256) or set(expected_sha256) != set(
        EXPECTED_SOURCE_SHA256
    ):
        raise DiscoveryShardError("source path/hash names differ from the frozen contract")
    loaded: dict[str, dict[str, object]] = {}
    observed: dict[str, str] = {}
    for name in sorted(source_paths):
        path = Path(source_paths[name])
        digest = _sha256(path)
        if digest != expected_sha256[name]:
            raise DiscoveryShardError(
                f"immutable {name} hash changed: expected={expected_sha256[name]}, "
                f"observed={digest}"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        if type(value) is not dict:
            raise DiscoveryShardError(f"{name} source is not a JSON object")
        loaded[name] = value
        observed[name] = digest
    return loaded["authority"], loaded["partition"], loaded["donors"], observed


def _integer_list(value: object, *, name: str) -> list[int]:
    if not isinstance(value, list) or not value or any(
        type(item) is not int or item < 0 for item in value
    ):
        raise DiscoveryShardError(f"{name} must be a nonempty list of token IDs")
    return list(value)


def _side_endpoint(
    endpoint: Mapping[str, object], authority_by_row: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    endpoint_id = endpoint.get("endpoint_id")
    row_id = endpoint.get("row_id")
    side = endpoint.get("side")
    if not isinstance(endpoint_id, str) or not isinstance(row_id, str) or side not in {
        "base",
        "donor",
    }:
        raise DiscoveryShardError("donor endpoint identity is malformed")
    if endpoint_id != f"{row_id}:{side}" or row_id not in authority_by_row:
        raise DiscoveryShardError("donor endpoint does not map to one authority side")
    row = authority_by_row[row_id]
    if (
        endpoint.get("group_id") != row.get("group_id")
        or endpoint.get("group_number") != row.get("group_number")
        or endpoint.get("family") != row.get("transform_id")
    ):
        raise DiscoveryShardError("donor endpoint metadata differs from its authority row")

    ids = _integer_list(row.get(f"{side}_ids"), name=f"{endpoint_id} IDs")
    position = row.get(f"{side}_prediction_position")
    if type(position) is not int or position != len(ids) - 1:
        raise DiscoveryShardError("endpoint prediction position is not the final token")
    answer_text = row.get(f"{side}_answer")
    foil_text = row.get(f"{side}_foil")
    answer_id = row.get(f"{side}_answer_id")
    foil_id = ANSWER_TOKEN_IDS.get(foil_text) if isinstance(foil_text, str) else None
    if answer_id != ANSWER_TOKEN_IDS.get(answer_text) or type(foil_id) is not int:
        raise DiscoveryShardError("endpoint answer/foil token IDs changed")
    if answer_id == foil_id:
        raise DiscoveryShardError("endpoint answer and foil must differ")
    subject_state = endpoint.get("subject_state")
    subject_number = row.get(f"{side}_subject_number")
    if (subject_state, subject_number) not in {(-1, "singular"), (1, "plural")}:
        raise DiscoveryShardError("endpoint subject-state metadata changed")

    return {
        "endpoint_id": endpoint_id,
        "row_id": row_id,
        "side": side,
        "partition": "DISCOVERY",
        "group_id": endpoint["group_id"],
        "group_number": endpoint["group_number"],
        "family": endpoint["family"],
        "subject_state": subject_state,
        "ids": ids,
        "final_position": position,
        "answer_id": answer_id,
        "foil_id": foil_id,
    }


def validate_shard(
    payload: Mapping[str, object], *, discovery_groups: set[str], validation_groups: set[str]
) -> str:
    """Fail closed if a serialized endpoint escapes the DISCOVERY partition."""

    endpoints = payload.get("endpoints")
    if payload.get("schema") != SCHEMA or payload.get("task_id") != TASK_ID:
        raise DiscoveryShardError("shard identity changed")
    if not isinstance(endpoints, list) or len(endpoints) != 128:
        raise DiscoveryShardError("shard must contain exactly 128 endpoints")
    endpoint_ids: set[str] = set()
    observed_groups: set[str] = set()
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            raise DiscoveryShardError("shard endpoint is not an object")
        endpoint_id = endpoint.get("endpoint_id")
        group_id = endpoint.get("group_id")
        if not isinstance(endpoint_id, str) or endpoint_id in endpoint_ids:
            raise DiscoveryShardError("shard endpoint IDs are malformed or duplicated")
        if endpoint.get("partition") != "DISCOVERY":
            raise DiscoveryShardError("a non-DISCOVERY endpoint entered the shard")
        if group_id in validation_groups or group_id not in discovery_groups:
            raise DiscoveryShardError("a VALIDATION or unknown group entered the shard")
        if set(endpoint) != {
            "endpoint_id",
            "row_id",
            "side",
            "partition",
            "group_id",
            "group_number",
            "family",
            "subject_state",
            "ids",
            "final_position",
            "answer_id",
            "foil_id",
        }:
            raise DiscoveryShardError("shard endpoint fields changed")
        endpoint_ids.add(endpoint_id)
        observed_groups.add(str(group_id))
    if observed_groups != discovery_groups or len(observed_groups) != 16:
        raise DiscoveryShardError("shard does not cover exactly the DISCOVERY groups")
    return _canonical_sha256(endpoints)


def build_shard(
    *,
    source_paths: Mapping[str, Path] = SOURCE_PATHS,
    expected_sha256: Mapping[str, str] = EXPECTED_SOURCE_SHA256,
) -> dict[str, object]:
    """Build and independently validate the complete DISCOVERY endpoint shard."""

    authority, partition, donors, observed_sha256 = _load_sources(
        source_paths, expected_sha256
    )
    if authority.get("task_id") != TASK_ID or authority.get("split") != "FIT":
        raise DiscoveryShardError("authority identity changed")
    rows = authority.get("rows")
    partition_records = partition.get("records")
    donor_endpoints = donors.get("endpoints")
    if not isinstance(rows, list) or len(rows) != 128:
        raise DiscoveryShardError("authority must contain exactly 128 rows")
    if not isinstance(partition_records, list) or len(partition_records) != 32:
        raise DiscoveryShardError("partition must contain exactly 32 groups")
    if not isinstance(donor_endpoints, list) or len(donor_endpoints) != 256:
        raise DiscoveryShardError("donor manifest must contain exactly 256 endpoints")
    authority_by_row = {
        str(row["row_id"]): row for row in rows if isinstance(row, dict)
    }
    if len(authority_by_row) != 128:
        raise DiscoveryShardError("authority row IDs are malformed or duplicated")

    discovery_groups = {
        str(record["group_id"])
        for record in partition_records
        if isinstance(record, dict) and record.get("partition") == "DISCOVERY"
    }
    validation_groups = {
        str(record["group_id"])
        for record in partition_records
        if isinstance(record, dict) and record.get("partition") == "VALIDATION"
    }
    if (
        len(discovery_groups) != 16
        or len(validation_groups) != 16
        or discovery_groups & validation_groups
    ):
        raise DiscoveryShardError("DISCOVERY/VALIDATION group partition changed")

    selected = [
        _side_endpoint(endpoint, authority_by_row)
        for endpoint in donor_endpoints
        if isinstance(endpoint, dict) and endpoint.get("group_id") in discovery_groups
    ]
    selected.sort(key=lambda endpoint: str(endpoint["endpoint_id"]))
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "partition": "DISCOVERY",
        "source_sha256": observed_sha256,
        "endpoint_count": len(selected),
        "group_count": len(discovery_groups),
        "endpoints": selected,
    }
    endpoints_sha256 = validate_shard(
        payload,
        discovery_groups=discovery_groups,
        validation_groups=validation_groups,
    )
    payload["endpoints_sha256"] = endpoints_sha256
    return payload


def dry_run() -> dict[str, object]:
    payload = build_shard()
    return {
        "schema": f"{SCHEMA}_dryrun",
        "task_id": TASK_ID,
        "partition": "DISCOVERY",
        "source_sha256": payload["source_sha256"],
        "endpoint_count": payload["endpoint_count"],
        "group_count": payload["group_count"],
        "endpoints_sha256": payload["endpoints_sha256"],
        "prompt_text_emitted": False,
        "validation_endpoints_emitted": 0,
        "output_written": False,
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
    }


def write_create_only(output: Path, payload: Mapping[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dry_run(), sort_keys=True, indent=2, allow_nan=False))
        return 0
    payload = build_shard()
    write_create_only(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
