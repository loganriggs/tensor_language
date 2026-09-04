#!/usr/bin/env python3
# BQLANE: cpu
"""Compile the exact conditional physical plan for task14 FIT localization v2.

This module is deliberately model-free.  It reads only the frozen FIT authority,
the reviewed v2 partition/donor authorities, and frozen source/preregistration
bytes.  It enumerates every possible model call as replayable hash-chain chunks;
it neither imports torch nor opens a checkpoint, GPU, queue, outcome, or later
phase artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import queue
import stat
import threading
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA = "task14_fit_localization_v2_physical_compiler_v1"
CALL_SCHEMA = "task14_fit_localization_v2_call_v1"
CHUNK_SCHEMA = "task14_fit_localization_v2_call_chunk_v1"
EXPERIMENT_ID = "task14-subject-verb-agreement-fit-localization-v2"
TASK_ID = "subject_verb.number_agreement"
PHASE = "FIT"
WIDTH = 1152
BOUNDARIES = tuple(range(-1, 18))
POSITIONS = ("H", "Q")
SEEDS = (14001, 14002, 14003, 14004, 14005)
RANKS = (1, 2, 4)
FIT_STEPS = 400
LOGICAL_RELATIONS_PER_STEP = 32
INTERVENTION_BATCH_LIMIT = 192
EVALUATION_BATCH_LIMIT = 128
GPU_TIME_LIMIT_SECONDS = 8 * 60 * 60

V2_COMMIT = "8f41f51cdf7e073063201cc48760622607ce91b9"
V2_REVIEW_COMMIT = "2ffd6cf77998a6c7fb6af0c4e89c742bf1bbb923"
V2_REVIEW_SHA256 = "2905aeb040fad2d16062a22e3c4d32d9dd6953c468724ff51a80ab9fa849d384"

REPO_ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
MANIFEST_PATH = OPS / "circuit_battery_task14_fit_localization_v2_call_manifest.json"
CALL_INDEX_PATH = OPS / "circuit_battery_task14_fit_localization_v2_call_index.bin"
DRYRUN_PATH = OPS / "circuit_battery_task14_fit_localization_v2_compiler_dryrun.json"

FROZEN = {
    "fit_authority": (
        "basis_aligned/bilinear_quotient/ops/circuit_battery_task14_agreement_fit_authority.json",
        "e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f",
    ),
    "v2_builder": (
        "basis_aligned/bilinear_quotient/ops/build_task14_fit_localization_v2.py",
        "ac6cc964065204193a1c119c721b37dabd9f026ec56b4a4d3b0c0ce837f27d49",
    ),
    "v2_partition": (
        "basis_aligned/bilinear_quotient/ops/circuit_battery_task14_fit_localization_partition_v2.json",
        "1f43b767fb39082d7872629d1a8b700e90e055c9529d9d319fe483f77d91fad3",
    ),
    "v2_donors": (
        "basis_aligned/bilinear_quotient/ops/circuit_battery_task14_fit_localization_donors_v2.json",
        "ff702f2936e2445a247c6fca3a55d177e80974b2a5e14fb6de0a5fe2761db50a",
    ),
    "v2_tests": (
        "basis_aligned/bilinear_quotient/ops/test_build_task14_fit_localization_v2.py",
        "bd2623ebe8aafc28a59990c615abd2919591ac9b062cd57ce7ed49fc99374ccf",
    ),
    "v2_preregistration": (
        "basis_aligned/polynomial_causal/TASK14_SUBJECT_VERB_AGREEMENT_FIT_LOCALIZATION_V2_PREREGISTRATION_2026-09-04.md",
        "3ea31387f611d0d095895dec6ed0859e1d99b2ad91a5d5adfb7be178bf127f59",
    ),
    "v2_independent_review": (
        "basis_aligned/polynomial_causal/TASK14_SUBJECT_VERB_AGREEMENT_FIT_LOCALIZATION_V2_INDEPENDENT_REVIEW_2026-09-04.md",
        V2_REVIEW_SHA256,
    ),
    "spectral_derivation": (
        "basis_aligned/polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-04_0930.md",
        "d1c5eeb73b1e41a33ba5bc69ee26afa53e6518ad5de171dea3420d9af6091cfd",
    ),
    "experiment_spec": (
        "basis_aligned/bilinear_quotient/ops/circuit_experiment_spec.py",
        "64ba9b75d49dbc6129d592573fee454e27e2de661daef30ca35d457dbbbb093c",
    ),
    "artifact_package": (
        "basis_aligned/bilinear_quotient/ops/circuit_artifact_package.py",
        "6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc",
    ),
    "result_contract": (
        "basis_aligned/bilinear_quotient/ops/result_contract.py",
        "af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272",
    ),
    "model_source": (
        "jacclust/tt_model.py",
        "49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2",
    ),
    "model_facade": (
        "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    ),
    "fastload": (
        "basis_aligned/bilinear_quotient/ops/fastload.py",
        "5803de7f127d1f556470107b559c06daecf7fbc2bccf4574aeb1c347b6225d90",
    ),
    "fastload_dependency": (
        "basis_aligned/bilinear_quotient/ops/mlp_in_situ_usage_rank_map_probe.py",
        "c701af71491d29f33f5ad691f89380a9fa7c2d86514a61fd7423ad8a78fd4d16",
    ),
}

FIT_LOGICAL_SHA256 = "3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1"
PARTITION_RECORDS_SHA256 = "285092178ef25e5aee923a2b02ec791c6b2df83e7c47f185626cd5cfa507d08c"
DONOR_RECORDS_SHA256 = "6e1fc1fef2715e0c87f0e494646057957bad284f7b69b1e52dcc4ec0f3e6f905"
ENDPOINTS_SHA256 = "1b0deab978dbd3126ac09b22818609177b1b1da461eaa1812aa2d05bbb9d8438"

EXPECTED_RUNTIME = {
    "python_implementation": "CPython",
    "python_version": "3.12.14",
    "numpy_version": "2.5.2",
    "torch_version": "2.11.0+cu128",
    "torch_cuda_version": "12.8",
    "tiktoken_version": "0.14.0",
    "einops_version": "0.8.2",
}
CHECKPOINT = {
    "revision": "ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240",
    "config_sha256": "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c",
    "weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
    "weights_bytes": 2_067_738_635,
}


class CompileError(ValueError):
    """The requested plan differs from the exact v2 FIT compiler contract."""


_CALL_VISITOR: Any = None


def canonical_bytes(value: object, *, newline: bool = False) -> bytes:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return data + (b"\n" if newline else b"")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_read(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CompileError(f"not a regular file: {path}")
        blocks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            blocks.append(block)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise CompileError(f"file changed during read: {path}")
        return b"".join(blocks)
    finally:
        os.close(fd)


def strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise CompileError(f"{label} contains nonfinite JSON: {token}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise CompileError(f"{label} contains duplicate key: {key}")
            output[key] = value
        return output

    try:
        value = json.loads(
            raw, parse_constant=reject_constant, object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompileError(f"{label} is invalid JSON") from error
    if type(value) is not dict:
        raise CompileError(f"{label} is not an object")
    canonical_bytes(value)
    return value


def _load_frozen(role: str) -> bytes:
    relative, expected = FROZEN[role]
    raw = safe_read(REPO_ROOT / relative)
    if bytes_sha256(raw) != expected:
        raise CompileError(f"frozen role changed: {role}")
    return raw


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for role in FROZEN:
        _load_frozen(role)
    authority = strict_json(_load_frozen("fit_authority"), "FIT authority")
    partition = strict_json(_load_frozen("v2_partition"), "v2 partition")
    donors = strict_json(_load_frozen("v2_donors"), "v2 donors")
    if authority.get("split") != PHASE or authority.get("task_id") != TASK_ID:
        raise CompileError("FIT authority phase/task mismatch")
    if authority.get("split_records_sha256") != FIT_LOGICAL_SHA256:
        raise CompileError("FIT logical rows changed")
    if len(authority.get("rows", [])) != 128:
        raise CompileError("FIT row census changed")
    if partition.get("records_sha256") != PARTITION_RECORDS_SHA256 \
            or len(partition.get("records", [])) != 32:
        raise CompileError("partition logical authority changed")
    if donors.get("records_sha256") != DONOR_RECORDS_SHA256 \
            or donors.get("endpoints_sha256") != ENDPOINTS_SHA256 \
            or len(donors.get("records", [])) != 1088 \
            or len(donors.get("endpoints", [])) != 256:
        raise CompileError("donor logical authority changed")
    if set(r["partition"] for r in partition["records"]) != {"DISCOVERY", "VALIDATION"}:
        raise CompileError("partition labels changed")
    if any(r.get("partition") not in {"DISCOVERY", "VALIDATION"} for r in donors["records"]):
        raise CompileError("donor phase leaked")
    return authority, partition, donors


def _site(position: str, boundary: int) -> str:
    if position not in POSITIONS or boundary not in BOUNDARIES:
        raise CompileError("invalid residual site")
    return f"{position}:{boundary}"


def _row_maps(authority: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = {str(row["row_id"]): dict(row) for row in authority["rows"]}
    if len(rows) != 128:
        raise CompileError("duplicate FIT row identity")
    answer_ids: dict[str, int] = {}
    for row in rows.values():
        for side in ("base", "donor"):
            token = str(row[f"{side}_answer"])
            token_id = row[f"{side}_answer_id"]
            if type(token_id) is not int:
                raise CompileError("answer token ID is invalid")
            if token in answer_ids and answer_ids[token] != token_id:
                raise CompileError("answer token has inconsistent token IDs")
            answer_ids[token] = token_id
    if set(answer_ids) != {" is", " are"}:
        raise CompileError("answer vocabulary changed")
    endpoints: dict[str, dict[str, Any]] = {}
    for row in rows.values():
        for side in ("base", "donor"):
            endpoint_id = f"{row['row_id']}:{side}"
            ids = list(row[f"{side}_ids"])
            if not ids or any(type(token) is not int for token in ids):
                raise CompileError("endpoint token IDs are invalid")
            endpoints[endpoint_id] = {
                "endpoint_id": endpoint_id,
                "row_id": row["row_id"],
                "side": side,
                "family": row["transform_id"],
                "ids": ids,
                "sequence_length": len(ids),
                "answer_id": row[f"{side}_answer_id"],
                "foil_id": answer_ids[str(row[f"{side}_foil"])],
                "H_position": row[f"{side}_head_positions"][0],
                "H_positions": list(row[f"{side}_head_positions"]),
                "Q_position": row[f"{side}_prediction_position"],
                "subject_state": -1 if row[f"{side}_subject_number"] == "singular" else 1,
            }
    if len(endpoints) != 256:
        raise CompileError("endpoint identity collision")
    return rows, endpoints


def _validate_endpoint_table(
    donor_authority: Mapping[str, Any], endpoints: Mapping[str, Mapping[str, Any]],
) -> None:
    observed = {str(item["endpoint_id"]): item for item in donor_authority["endpoints"]}
    if set(observed) != set(endpoints):
        raise CompileError("endpoint table identity mismatch")
    for endpoint_id, item in observed.items():
        expected = endpoints[endpoint_id]
        if item["family"] != expected["family"] \
                or item["subject_state"] != expected["subject_state"]:
            raise CompileError("endpoint semantic binding mismatch")


def _record_maps(donors: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records = [dict(record) for record in donors["records"]]
    by_id = {str(record["record_id"]): record for record in records}
    if len(by_id) != 1088 or [record["ordinal"] for record in records] != list(range(1088)):
        raise CompileError("donor identity/order changed")
    return records, by_id


def _record_cell(record: Mapping[str, Any], endpoints: Mapping[str, Mapping[str, Any]]) -> str:
    state = endpoints[str(record["target_endpoint_id"])]["subject_state"]
    return "|".join((str(record["arm"]), str(record["family"]), str(record["matching"]), str(state)))


def _applicable(record: Mapping[str, Any], position: str) -> bool:
    return position == "Q" or not bool(record["q_only"])


def _is_opposite(record: Mapping[str, Any]) -> bool:
    return record["expected_relation"] == "opposite_subject_toward_donor"


def _required_answer_records(records: Sequence[Mapping[str, Any]], partition: str, position: str) -> list[dict[str, Any]]:
    return [
        dict(record) for record in records
        if record["partition"] == partition and _applicable(record, position) and _is_opposite(record)
    ]


def _all_applicable_records(records: Sequence[Mapping[str, Any]], partition: str, position: str) -> list[dict[str, Any]]:
    return [
        dict(record) for record in records
        if record["partition"] == partition and _applicable(record, position)
    ]


def _batch_binding(
    item_ids: Sequence[str], *, kind: str, position: str | None,
    endpoints: Mapping[str, Mapping[str, Any]], records_by_id: Mapping[str, Mapping[str, Any]],
    uses: Mapping[str, Sequence[str]] | None = None,
    extra_positions: Sequence[str] = (),
) -> tuple[int, str, str]:
    if any(extra not in POSITIONS or extra == position for extra in extra_positions):
        raise CompileError("invalid extra intervention position")
    items: list[dict[str, Any]] = []
    sequence_lengths: set[int] = set()
    for item_id in item_ids:
        if kind == "endpoint":
            endpoint = endpoints[item_id]
            sequence_lengths.add(int(endpoint["sequence_length"]))
            items.append({
                "endpoint_id": item_id,
                "ids": endpoint["ids"],
                "answer_id": endpoint["answer_id"],
                "foil_id": endpoint["foil_id"],
                "H_positions": endpoint["H_positions"],
                "Q_position": endpoint["Q_position"],
            })
        else:
            record = records_by_id[item_id]
            target = endpoints[str(record["target_endpoint_id"])]
            donor = endpoints[str(record["donor_endpoint_id"])]
            sequence_lengths.add(int(target["sequence_length"]))
            item = {
                "record_id": item_id,
                "target_endpoint_id": target["endpoint_id"],
                "donor_endpoint_id": donor["endpoint_id"],
                "target_ids": target["ids"],
                "donor_ids": donor["ids"],
                "donor_sequence_length": donor["sequence_length"],
                "target_answer_id": target["answer_id"],
                "target_foil_id": target["foil_id"],
                "target_position": target[f"{position}_position"],
                "donor_position": donor[f"{position}_position"],
            }
            if extra_positions:
                item["extra_positions"] = {
                    extra: {
                        "target_position": target[f"{extra}_position"],
                        "donor_position": donor[f"{extra}_position"],
                    }
                    for extra in extra_positions
                }
            if uses is not None:
                item["uses"] = list(uses.get(item_id, ()))
            items.append(item)
    if len(sequence_lengths) != 1:
        raise CompileError("physical batch mixes sequence lengths")
    length = next(iter(sequence_lengths))
    return length, canonical_sha256(items), canonical_sha256(list(item_ids))


def _call(
    *, stage: str, call_kind: str, branch: str, item_kind: str,
    item_ids: Sequence[str], position: str | None, boundary: int | None,
    endpoints: Mapping[str, Mapping[str, Any]], records_by_id: Mapping[str, Mapping[str, Any]],
    uses: Mapping[str, Sequence[str]] | None = None, retained: bool = False,
    fit: Mapping[str, Any] | None = None, variant: str | None = None,
    step_sha256: str | None = None, batch_ordinal: int = 0,
    participates_in_backward: bool = False,
    extra_positions: Sequence[str] = (),
    logical_step: Mapping[str, Any] | None = None,
    batch_count: int = 1,
) -> dict[str, Any]:
    length, binding, ids_digest = _batch_binding(
        item_ids, kind=item_kind, position=position, endpoints=endpoints,
        records_by_id=records_by_id, uses=uses, extra_positions=extra_positions,
    )
    core: dict[str, Any] = {
        "array_contracts": [
            {"contiguous": "C", "dtype": "float32", "name": "answer_logit", "retained": retained, "shape": [len(item_ids)]},
            {"contiguous": "C", "dtype": "float32", "name": "foil_logit", "retained": retained, "shape": [len(item_ids)]},
        ],
        "batch_binding_sha256": binding,
        "batch_ordinal": batch_ordinal,
        "batch_count": batch_count,
        "boundary": boundary,
        "branch": branch,
        "call_kind": call_kind,
        "cache_reads": _cache_reads(call_kind, position, boundary),
        "cache_writes": _cache_writes(call_kind),
        "forward_calls": 1,
        "item_count": len(item_ids),
        "item_ids": list(item_ids),
        "item_ids_sha256": ids_digest,
        "item_kind": item_kind,
        "item_uses": {
            item_id: list(uses.get(item_id, ())) for item_id in item_ids
        } if uses is not None else {},
        "logical_backward_after_this_call": participates_in_backward and batch_ordinal == batch_count - 1,
        "participates_in_backward": participates_in_backward,
        "phase": PHASE,
        "position": position,
        "extra_positions": list(extra_positions),
        "retained_output": retained,
        "schema": CALL_SCHEMA,
        "sequence_length": length,
        "stage": stage,
        "state_array_contracts": _state_array_contracts(call_kind, branch, len(item_ids)),
        "variant": variant,
    }
    if fit is not None:
        core["fit"] = dict(fit)
    if step_sha256 is not None:
        core["step_sha256"] = step_sha256
    if logical_step is not None:
        core["logical_step"] = dict(logical_step)
    return {**core, "call_id": canonical_sha256(core)}


def _state_array_contracts(call_kind: str, branch: str, batch_size: int) -> list[dict[str, Any]]:
    if call_kind == "native_cache_full_forward":
        output = [{
            "contiguous": "C", "dtype": "float32", "name": "fit_position_residuals",
            "retained": True, "shape": [batch_size, 38, WIDTH],
        }]
        if branch.endswith(":C"):
            output.append({
                "contiguous": "C", "dtype": "float32", "name": "c_second_head_residuals",
                "retained": True, "shape": [batch_size, 19, WIDTH],
            })
        return output
    if call_kind == "discovery_gradient_full_forward":
        return [{
            "contiguous": "C", "dtype": "float32", "name": "discovery_position_gradients",
            "retained": True, "shape": [batch_size, 38, WIDTH],
        }]
    return []


def _cache_reads(call_kind: str, position: str | None, boundary: int | None) -> list[str]:
    if call_kind.startswith("native") or call_kind == "discovery_gradient_full_forward":
        return []
    reads = ["fit_position_residuals"]
    if "projector" in call_kind or "necessity" in call_kind or "reader" in call_kind:
        reads.append("fitted_projector_registry")
    return reads


def _cache_writes(call_kind: str) -> list[str]:
    if call_kind == "native_cache_full_forward":
        return [
            "fit_position_residuals", "c_second_head_residuals", "native_answer_foil_logits",
        ]
    if call_kind == "discovery_gradient_full_forward":
        return ["discovery_H_Q_gradient_cache"]
    return []


def _batched_calls(
    *, items: Sequence[tuple[str, Sequence[str]]], batch_limit: int,
    stage: str, call_kind: str, branch: str, item_kind: str,
    position: str | None, boundary: int | None,
    endpoints: Mapping[str, Mapping[str, Any]], records_by_id: Mapping[str, Mapping[str, Any]],
    retained: bool = False, fit: Mapping[str, Any] | None = None,
    variant: str | None = None, step_sha256: str | None = None,
    participates_in_backward: bool = False,
    extra_positions: Sequence[str] = (),
    logical_step: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    by_length: dict[int, list[tuple[str, Sequence[str]]]] = defaultdict(list)
    for item_id, roles in items:
        if item_kind == "endpoint":
            length = int(endpoints[item_id]["sequence_length"])
        else:
            record = records_by_id[item_id]
            length = int(endpoints[str(record["target_endpoint_id"])]["sequence_length"])
        by_length[length].append((item_id, roles))
    batches: list[list[tuple[str, Sequence[str]]]] = []
    for length in sorted(by_length):
        group = by_length[length]
        for start in range(0, len(group), batch_limit):
            batches.append(group[start:start + batch_limit])
    for ordinal, batch in enumerate(batches):
        ids = [item[0] for item in batch]
        uses = {item_id: list(roles) for item_id, roles in batch}
        yield _call(
            stage=stage, call_kind=call_kind, branch=branch,
            item_kind=item_kind, item_ids=ids, position=position,
            boundary=boundary, endpoints=endpoints, records_by_id=records_by_id,
            uses=uses, retained=retained, fit=fit, variant=variant,
            step_sha256=step_sha256, batch_ordinal=ordinal,
            participates_in_backward=participates_in_backward,
            extra_positions=extra_positions,
            logical_step=logical_step, batch_count=len(batches),
        )


def _chain_seed(chunk_id: str) -> str:
    return canonical_sha256({"chunk_id": chunk_id, "schema": CHUNK_SCHEMA})


def _chain_step(current: str, call_id: str) -> str:
    return hashlib.sha256(bytes.fromhex(current) + bytes.fromhex(call_id)).hexdigest()


def _chunk(
    chunk_id: str, activation: str, calls: Iterable[Mapping[str, Any]], *,
    backward_calls: int = 0, optimizer_updates: int = 0,
    logical_update_root: str | None = None,
) -> dict[str, Any]:
    root = _chain_seed(chunk_id)
    count = examples = tokens = forwards = graph_batches = 0
    first = last = None
    kinds: dict[str, int] = defaultdict(int)
    for call in calls:
        call_id = str(call["call_id"])
        if first is None:
            first = call_id
        last = call_id
        if _CALL_VISITOR is not None:
            _CALL_VISITOR(chunk_id, dict(call))
        root = _chain_step(root, call_id)
        count += 1
        examples += int(call["item_count"])
        tokens += int(call["item_count"]) * int(call["sequence_length"])
        forwards += int(call["forward_calls"])
        graph_batches += int(bool(call["participates_in_backward"]))
        kinds[str(call["call_kind"])] += 1
    return {
        "activation": activation,
        "backward_calls": backward_calls,
        "backward_graph_batches": graph_batches,
        "call_count": count,
        "call_kind_counts": dict(sorted(kinds.items())),
        "call_root_sha256": root,
        "chunk_id": chunk_id,
        "example_evaluations": examples,
        "first_call_id": first,
        "forward_calls": forwards,
        "last_call_id": last,
        "logical_update_root_sha256": logical_update_root,
        "optimizer_updates": optimizer_updates,
        "schema": CHUNK_SCHEMA,
        "token_evaluations": tokens,
    }


def _endpoint_items(endpoint_ids: Sequence[str]) -> list[tuple[str, Sequence[str]]]:
    return [(endpoint_id, ()) for endpoint_id in endpoint_ids]


def _record_items(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, Sequence[str]]]:
    return [(str(record["record_id"]), ()) for record in records]


def _native_chunks(
    authority: Mapping[str, Any], partition: Mapping[str, Any],
    endpoints: Mapping[str, Mapping[str, Any]], records_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    discovery_groups = {
        int(record["group_number"]) for record in partition["records"]
        if record["partition"] == "DISCOVERY"
    }
    rows = [dict(row) for row in authority["rows"]]
    chunks: list[dict[str, Any]] = []
    native_calls: list[dict[str, Any]] = []
    gradient_calls: list[dict[str, Any]] = []
    for side in ("base", "donor"):
        for family in ("A1", "A2", "P", "C"):
            selected = sorted(
                (row for row in rows if row["transform_id"] == family),
                key=lambda row: int(row["group_number"]),
            )
            ids = [f"{row['row_id']}:{side}" for row in selected]
            native_calls.append(_call(
                stage="native_cache", call_kind="native_cache_full_forward",
                branch=f"native:{side}:{family}", item_kind="endpoint", item_ids=ids,
                position=None, boundary=None, endpoints=endpoints, records_by_id=records_by_id,
                retained=True,
            ))
            if family in {"A1", "A2"}:
                discovery_ids = [
                    f"{row['row_id']}:{side}" for row in selected
                    if int(row["group_number"]) in discovery_groups
                ]
                gradient_calls.append(_call(
                    stage="discovery_gradient", call_kind="discovery_gradient_full_forward",
                    branch=f"gradient:{side}:{family}", item_kind="endpoint", item_ids=discovery_ids,
                    position=None, boundary=None, endpoints=endpoints, records_by_id=records_by_id,
                    retained=False, participates_in_backward=True,
                ))
    chunks.append(_chunk("00_native_cache", "preflight_pass", native_calls))
    chunks.append(_chunk(
        "01_discovery_gradients", "native_cache_complete", gradient_calls,
        backward_calls=4,
    ))
    return chunks


def _ceiling_chunk(
    *, partition_name: str, position: str, boundary: int,
    records: Sequence[Mapping[str, Any]], endpoints: Mapping[str, Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]], activation: str,
) -> dict[str, Any]:
    selected = _required_answer_records(records, partition_name, position)
    calls = _batched_calls(
        items=_record_items(selected), batch_limit=EVALUATION_BATCH_LIMIT,
        stage=f"{partition_name.lower()}_full_ceiling",
        call_kind="full_state_intervention_forward",
        branch=f"full_ceiling:{partition_name}:{_site(position, boundary)}",
        item_kind="record", position=position, boundary=boundary,
        endpoints=endpoints, records_by_id=records_by_id, retained=True,
    )
    return _chunk(
        f"ceiling:{partition_name}:{_site(position, boundary)}", activation, calls,
    )


def _aggregate(record: Mapping[str, Any]) -> str | None:
    arm, family = str(record["arm"]), str(record["family"])
    if arm == "answer_change" and family in {"A1", "A2"}:
        return family
    if arm == "cross_syntax" and family in {"A1", "A2"}:
        return "X1" if family == "A1" else "X2"
    if arm == "P_positive_transfer":
        return "P"
    if arm in {"C_to_ordinary_singular", "ordinary_singular_to_C"}:
        return "CS"
    if arm == "P_zero_coordinate_control":
        return "L_P"
    if arm == "C_zero_coordinate_control":
        return "L_C"
    if arm in {"C_to_ordinary_plural_control", "ordinary_plural_to_C_control"}:
        return "L_CP"
    return None


def _fit_cycles(position: str, objective: str) -> tuple[str, ...]:
    if objective == "joint":
        return (
            ("A1", "A1", "A2", "A2", "X1", "X2", "P", "L_P")
            if position == "H" else
            ("A1", "A1", "A2", "A2", "X1", "X2", "P", "CS", "CS", "A_C", "A_C", "L_P", "L_C", "L_CP")
        )
    if objective not in {"A1_only", "A2_only"}:
        raise CompileError("invalid fit objective")
    family = objective[:2]
    return (
        (family, family, "P", "L_P")
        if position == "H" else
        (family, family, "P", f"CS_{family}", f"CS_{family}", "A_C", "A_C", "L_P", "L_C", "L_CP")
    )


def _fit_pools(
    *, position: str, objective: str, records: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    pools: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record["partition"] != "DISCOVERY" or not _applicable(record, position):
            continue
        aggregate = _aggregate(record)
        if aggregate is None:
            continue
        if aggregate == "CS" and objective in {"A1_only", "A2_only"}:
            family = objective[:2]
            target = endpoints[str(record["target_endpoint_id"])]
            donor = endpoints[str(record["donor_endpoint_id"])]
            ordinary_family = donor["family"] if target["family"] == "C" else target["family"]
            if ordinary_family != family:
                continue
            aggregate = f"CS_{family}"
        if objective in {"A1_only", "A2_only"} and aggregate in {"A1", "A2", "X1", "X2"}:
            family = objective[:2]
            if aggregate != family:
                continue
        pools[aggregate][_record_cell(record, endpoints)].append(str(record["record_id"]))
    if position == "Q":
        c_endpoints = sorted(
            endpoint_id for endpoint_id, endpoint in endpoints.items()
            if endpoint["family"] == "C" and any(
                record["partition"] == "DISCOVERY"
                and record["target_endpoint_id"] == endpoint_id
                for record in records
            )
        )
        for endpoint_id in c_endpoints:
            pools["A_C"][f"C_{endpoints[endpoint_id]['side']}"].append(endpoint_id)
    expected = set(_fit_cycles(position, objective))
    if set(pools) != expected:
        raise CompileError(
            f"fit aggregate coverage mismatch at {position}/{objective}: {set(pools)} != {expected}"
        )
    return {key: dict(value) for key, value in pools.items()}


def _ordered_pool(
    values: Sequence[str], *, seed: int, rank: int, site: str, objective: str,
) -> list[str]:
    prefix = f"task14-localization-v2-fit|{seed}|{rank}|{site}|{objective}|"
    return sorted(values, key=lambda value: hashlib.sha256((prefix + value).encode()).hexdigest())


def _normalizer_records(
    *, position: str, objective: str, records: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = defaultdict(list)
    family = objective[:2] if objective in {"A1_only", "A2_only"} else None
    for record in records:
        if record["partition"] != "DISCOVERY" or not _applicable(record, position):
            continue
        arm = str(record["arm"])
        if arm == "answer_change" and record["matching"] == "paired":
            if family is None or record["family"] == family:
                output[f"paired:{record['family']}"].append(str(record["record_id"]))
        if position == "Q" and arm in {"C_to_ordinary_singular", "ordinary_singular_to_C"}:
            target = endpoints[str(record["target_endpoint_id"])]
            donor = endpoints[str(record["donor_endpoint_id"])]
            ordinary = donor["family"] if target["family"] == "C" else target["family"]
            if family is None or ordinary == family:
                output[f"complete:{arm}:{ordinary}"].append(str(record["record_id"]))
    expected = 2 if position == "H" and family is None else 1 if position == "H" else 6 if family is None else 3
    if len(output) != expected:
        raise CompileError("normalizer reference-cell coverage mismatch")
    return dict(output)


def _fit_step_stream(
    *, position: str, boundary: int, objective: str, rank: int, seed: int,
    records: Sequence[Mapping[str, Any]], endpoints: Mapping[str, Mapping[str, Any]],
) -> Iterator[tuple[int, str, list[dict[str, str]], dict[str, list[str]]]]:
    site = _site(position, boundary)
    pools = _fit_pools(position=position, objective=objective, records=records, endpoints=endpoints)
    ordered: dict[tuple[str, str], list[str]] = {}
    cell_names: dict[str, list[str]] = {}
    for aggregate, cells in pools.items():
        cell_names[aggregate] = sorted(cells)
        for cell, values in cells.items():
            ordered[(aggregate, cell)] = _ordered_pool(
                values, seed=seed, rank=rank, site=site, objective=objective,
            )
    cell_cursor = defaultdict(int)
    value_cursor = defaultdict(int)
    cycle = _fit_cycles(position, objective)
    normalizers = _normalizer_records(
        position=position, objective=objective, records=records, endpoints=endpoints,
    )
    for step in range(FIT_STEPS):
        slots: list[dict[str, str]] = []
        for stream_index in range(step * LOGICAL_RELATIONS_PER_STEP, (step + 1) * LOGICAL_RELATIONS_PER_STEP):
            aggregate = cycle[stream_index % len(cycle)]
            cells = cell_names[aggregate]
            cell = cells[cell_cursor[aggregate] % len(cells)]
            cell_cursor[aggregate] += 1
            values = ordered[(aggregate, cell)]
            value = values[value_cursor[(aggregate, cell)] % len(values)]
            value_cursor[(aggregate, cell)] += 1
            slots.append({"aggregate": aggregate, "cell": cell, "item_id": value})
        step_core = {
            "normalizer_cells": normalizers,
            "objective": objective,
            "rank": rank,
            "seed": seed,
            "site": site,
            "slots": slots,
            "step": step,
        }
        yield step, canonical_sha256(step_core), slots, normalizers


def _fit_chunk(
    *, position: str, boundary: int, objective: str, rank: int, seed: int,
    records: Sequence[Mapping[str, Any]], endpoints: Mapping[str, Mapping[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]], activation: str,
) -> dict[str, Any]:
    site = _site(position, boundary)
    fit = {"objective": objective, "rank": rank, "seed": seed, "site": site}
    logical_root = canonical_sha256({"fit": fit, "schema": "task14_v2_update_stream_v1"})

    def calls() -> Iterator[dict[str, Any]]:
        nonlocal logical_root
        for step, step_hash, slots, normalizers in _fit_step_stream(
            position=position, boundary=boundary, objective=objective, rank=rank,
            seed=seed, records=records, endpoints=endpoints,
        ):
            logical_root = hashlib.sha256(
                bytes.fromhex(logical_root) + bytes.fromhex(step_hash)
            ).hexdigest()
            uses: dict[str, set[str]] = defaultdict(set)
            for cell, ids in normalizers.items():
                for record_id in ids:
                    uses[record_id].add(f"normalizer:{cell}")
            for slot in slots:
                if slot["aggregate"] != "A_C":
                    uses[slot["item_id"]].add(f"train:{slot['aggregate']}:{slot['cell']}")
            items = [(record_id, tuple(sorted(roles))) for record_id, roles in sorted(uses.items())]
            yield from _batched_calls(
                items=items, batch_limit=INTERVENTION_BATCH_LIMIT,
                stage="discovery_fit_update", call_kind="projector_intervention_train_forward",
                branch=f"fit:{site}:{objective}:rank{rank}:seed{seed}:step{step}",
                item_kind="record", position=position, boundary=boundary,
                endpoints=endpoints, records_by_id=records_by_id,
                retained=False, fit={**fit, "step": step}, step_sha256=step_hash,
                participates_in_backward=True,
                logical_step={"normalizer_cells": normalizers, "slots": slots},
            )
        final_records = _all_applicable_records(records, "DISCOVERY", position)
        yield from _batched_calls(
            items=_record_items(final_records), batch_limit=EVALUATION_BATCH_LIMIT,
            stage="discovery_fit_final", call_kind="final_discovery_projector_eval",
            branch=f"fit_final:{site}:{objective}:rank{rank}:seed{seed}",
            item_kind="record", position=position, boundary=boundary,
            endpoints=endpoints, records_by_id=records_by_id,
            retained=True, fit=fit,
        )

    chunk = _chunk(
        f"fit:{site}:{objective}:rank{rank}:seed{seed}", activation, calls(),
        backward_calls=FIT_STEPS, optimizer_updates=FIT_STEPS,
        logical_update_root="computed_after_stream",
    )
    # The generator has now consumed every step and finalized the nonlocal root.
    chunk["logical_update_root_sha256"] = logical_root
    return chunk


def _spectral_chunk(
    *, position: str, boundary: int, records: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Mapping[str, Any]], records_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    selected = _required_answer_records(records, "DISCOVERY", position)
    calls = _batched_calls(
        items=_record_items(selected), batch_limit=EVALUATION_BATCH_LIMIT,
        stage="spectral_diagnostic", call_kind="spectral_projector_intervention_forward",
        branch=f"spectral:{_site(position, boundary)}", item_kind="record",
        position=position, boundary=boundary, endpoints=endpoints,
        records_by_id=records_by_id, retained=True,
        fit={"objective": "spectral_diagnostic", "rank": 1, "seed": None, "site": _site(position, boundary)},
    )
    return _chunk(
        f"spectral:{_site(position, boundary)}",
        "site_is_retained_for_joint_rank1;diagnostic_only_never_selects", calls,
    )


def _projected_eval_chunk(
    *, partition_name: str, position: str, boundary: int, objective: str,
    rank: int, seed: int, records: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Mapping[str, Any]], records_by_id: Mapping[str, Mapping[str, Any]],
    activation: str,
) -> dict[str, Any]:
    selected = _all_applicable_records(records, partition_name, position)
    fit = {"objective": objective, "rank": rank, "seed": seed, "site": _site(position, boundary)}
    calls = _batched_calls(
        items=_record_items(selected), batch_limit=EVALUATION_BATCH_LIMIT,
        stage=f"{partition_name.lower()}_projected_evaluation",
        call_kind="locked_projector_intervention_forward",
        branch=f"eval:{partition_name}:{_site(position, boundary)}:{objective}:rank{rank}:seed{seed}",
        item_kind="record", position=position, boundary=boundary,
        endpoints=endpoints, records_by_id=records_by_id, retained=True, fit=fit,
    )
    return _chunk(
        f"eval:{partition_name}:{_site(position, boundary)}:{objective}:rank{rank}:seed{seed}",
        activation, calls,
    )


def _ordinary_validation_ids(
    authority: Mapping[str, Any], partition: Mapping[str, Any], side: str | None = None,
) -> list[str]:
    groups = {
        int(record["group_number"]) for record in partition["records"]
        if record["partition"] == "VALIDATION"
    }
    output: list[str] = []
    for row in sorted(authority["rows"], key=lambda value: (value["transform_id"], value["row_id"])):
        if row["transform_id"] not in {"A1", "A2"} or int(row["group_number"]) not in groups:
            continue
        for item_side in ("base", "donor"):
            if side is None or side == item_side:
                output.append(f"{row['row_id']}:{item_side}")
    return output


def _necessity_chunk_exact(
    *, authority: Mapping[str, Any], partition: Mapping[str, Any], boundary: int, seed: int,
    endpoints: Mapping[str, Mapping[str, Any]], records_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ids = _ordinary_validation_ids(authority, partition)
    calls = _batched_calls(
        items=_endpoint_items(ids), batch_limit=EVALUATION_BATCH_LIMIT,
        stage="validation_necessity", call_kind="necessity_neutralization_forward",
        branch=f"necessity:Q:{boundary}:seed{seed}", item_kind="endpoint",
        position="Q", boundary=boundary, endpoints=endpoints, records_by_id=records_by_id,
        retained=True, fit={"objective": "joint", "rank": 1, "seed": seed, "site": _site("Q", boundary)},
        variant="neutral_selected_Q",
    )
    return _chunk(
        f"necessity:Q:{boundary}:seed{seed}",
        "site_is_selected_Q_and_rank1_semantic_and_falsifier_gates_pass", calls,
    )


def _redundancy_chunk(
    *, authority: Mapping[str, Any], partition: Mapping[str, Any], first: int, second: int, seed: int,
    endpoints: Mapping[str, Mapping[str, Any]], records_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if first >= second:
        raise CompileError("top-two Q pair must be ordered")
    ids = _ordinary_validation_ids(authority, partition)

    def calls() -> Iterator[dict[str, Any]]:
        for variant, boundary in (("neutral_first_Q", first), ("neutral_second_Q", second), ("neutral_both_Q", first)):
            fit = {
                "first_site": _site("Q", first), "objective": "joint", "rank": 1,
                "second_site": _site("Q", second), "seed": seed,
            }
            yield from _batched_calls(
                items=_endpoint_items(ids), batch_limit=EVALUATION_BATCH_LIMIT,
                stage="validation_two_site_redundancy",
                call_kind="two_site_necessity_forward" if variant == "neutral_both_Q" else "necessity_neutralization_forward",
                branch=f"redundancy:Q:{first}:{second}:seed{seed}:{variant}",
                item_kind="endpoint", position="Q", boundary=boundary,
                endpoints=endpoints, records_by_id=records_by_id, retained=True,
                fit=fit, variant=variant,
            )

    return _chunk(
        f"redundancy:Q:{first}:{second}:seed{seed}",
        "pair_is_top_two_Q_and_selected_Q_single_necessity_fails", calls(),
    )


def _reader_chunk(
    *, h_boundary: int, q_boundary: int, seed: int, records: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Mapping[str, Any]], records_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if h_boundary >= q_boundary:
        raise CompileError("reader pair is not causally ordered")
    selected = [
        dict(record) for record in records
        if record["partition"] == "VALIDATION" and record["arm"] == "answer_change"
    ]
    variants = (
        "upstream_H_patch", "upstream_H_patch_then_native_Q_reset",
        "upstream_H_neutral", "upstream_H_neutral_then_donor_Q_insert",
        "downstream_Q_patch",
    )

    def calls() -> Iterator[dict[str, Any]]:
        for variant in variants:
            yield from _batched_calls(
                items=_record_items(selected), batch_limit=EVALUATION_BATCH_LIMIT,
                stage="validation_ordered_reader", call_kind="ordered_H_Q_reader_forward",
                branch=f"reader:H:{h_boundary}:Q:{q_boundary}:seed{seed}:{variant}",
                item_kind="record", position="H", boundary=h_boundary,
                endpoints=endpoints, records_by_id=records_by_id, retained=True,
                fit={
                    "H_site": _site("H", h_boundary), "Q_site": _site("Q", q_boundary),
                    "objective": "joint", "rank": 1, "seed": seed,
                }, variant=variant, extra_positions=("Q",),
            )

    return _chunk(
        f"reader:H:{h_boundary}:Q:{q_boundary}:seed{seed}",
        "sites_are_selected_and_ordered_and_semantics_pass_and_necessity_route_passes", calls(),
    )


def _fit_templates() -> Iterator[tuple[str, int, str, int, int, str]]:
    for position in POSITIONS:
        for boundary in BOUNDARIES:
            joint_activation = (
                "site_in_discovery_top3_H" if position == "H" else "site_discovery_eligible_Q"
            )
            for seed in SEEDS:
                yield position, boundary, "joint", 1, seed, joint_activation
            selected = "site_is_selected_H" if position == "H" else "site_is_selected_Q"
            for objective in ("A1_only", "A2_only"):
                for seed in SEEDS:
                    yield position, boundary, objective, 1, seed, selected
            for rank in (2, 4):
                for seed in SEEDS:
                    yield position, boundary, "joint", rank, seed, selected


def _retained_arrays() -> list[dict[str, Any]]:
    arrays = [
        {"name": "native_answer_foil_logits", "dtype": "float32", "shape": [256, 2], "when": "always", "raw_bytes": 2_048},
        {"name": "fit_position_residuals", "dtype": "float32", "shape": [256, 38, 1152], "when": "always", "raw_bytes": 44_826_624},
        {"name": "c_second_head_residuals", "dtype": "float32", "shape": [64, 19, 1152], "when": "always", "raw_bytes": 5_603_328},
        {"name": "discovery_position_gradients", "dtype": "float32", "shape": [64, 38, 1152], "when": "always", "raw_bytes": 11_206_656},
        {"name": "discovery_full_ceiling_margin", "dtype": "float32", "shape": [13_984], "when": "always", "raw_bytes": 55_936},
        {"name": "spectral_projector", "dtype": "float32", "shape": ["R_H+R_Q", 1152], "when": "retained_sites_nonempty", "raw_bytes": "4608*(R_H+R_Q)"},
        {"name": "spectral_finite_and_local_effect", "dtype": "float32", "shape": [2, "320*R_H+416*R_Q"], "when": "retained_sites_nonempty", "raw_bytes": "8*(320*R_H+416*R_Q)"},
        {"name": "joint_rank1_U", "dtype": "float32", "shape": ["R_H+R_Q", 5, 1152, 1], "when": "retained_sites_nonempty", "raw_bytes": "23040*(R_H+R_Q)"},
        {"name": "joint_rank1_trace", "dtype": "float32", "shape": ["R_H+R_Q", 5, 400], "when": "retained_sites_nonempty", "raw_bytes": "8000*(R_H+R_Q)"},
        {"name": "joint_rank1_discovery_margin", "dtype": "float32", "shape": [5, "336*R_H+544*R_Q"], "when": "retained_sites_nonempty", "raw_bytes": "20*(336*R_H+544*R_Q)"},
        {"name": "selected_family_rank1_U", "dtype": "float32", "shape": [2, 2, 5, 1152, 1], "when": "H_and_Q_selected", "raw_bytes": 92_160},
        {"name": "selected_family_rank1_trace", "dtype": "float32", "shape": [2, 2, 5, 400], "when": "H_and_Q_selected", "raw_bytes": 32_000},
        {"name": "selected_rank2_U", "dtype": "float32", "shape": [2, 5, 1152, 2], "when": "H_and_Q_selected", "raw_bytes": 92_160},
        {"name": "selected_rank4_U", "dtype": "float32", "shape": [2, 5, 1152, 4], "when": "H_and_Q_selected", "raw_bytes": 184_320},
        {"name": "selected_rank_falsifier_trace", "dtype": "float32", "shape": [2, 2, 5, 400], "when": "H_and_Q_selected", "raw_bytes": 32_000},
        {"name": "selected_nonjoint_discovery_margin", "dtype": "float32", "shape": [4, 5, 880], "when": "H_and_Q_selected", "raw_bytes": 70_400},
        {"name": "validation_full_ceiling_margin", "dtype": "float32", "shape": [736], "when": "H_and_Q_selected", "raw_bytes": 2_944},
        {"name": "validation_projected_margin", "dtype": "float32", "shape": [5, 5, 880], "when": "H_and_Q_selected", "raw_bytes": 88_000},
        {"name": "validation_single_necessity_margin", "dtype": "float32", "shape": [5, 64], "when": "semantic_and_rank_gates_pass", "raw_bytes": 1_280},
        {"name": "validation_two_site_necessity_margin", "dtype": "float32", "shape": [5, 3, 64], "when": "redundancy_branch", "raw_bytes": 3_840},
        {"name": "validation_reader_margin", "dtype": "float32", "shape": [5, 5, 192], "when": "ordered_reader_branch", "raw_bytes": 19_200},
    ]
    return [{**item, "contiguous": "C"} for item in arrays]


def _retained_byte_contract() -> dict[str, Any]:
    fixed = 2_048 + 44_826_624 + 5_603_328 + 11_206_656 + 55_936
    retained_site = lambda rh, rq: (
        4_608 * (rh + rq)
        + 8 * (320 * rh + 416 * rq)
        + 23_040 * (rh + rq)
        + 8_000 * (rh + rq)
        + 20 * (336 * rh + 544 * rq)
    )
    selected = 92_160 + 32_000 + 92_160 + 184_320 + 32_000 + 70_400 + 2_944 + 88_000
    necessity = 1_280
    redundancy = 3_840
    reader = 19_200
    maximum = fixed + retained_site(3, 19) + selected + necessity + redundancy + reader
    return {
        "fixed_raw_numeric_bytes": fixed,
        "formula": "fixed + retained_site(R_H,R_Q) + selected + necessity + redundancy + reader",
        "maximum_raw_numeric_bytes": maximum,
        "maximum_values": {
            "R_H": 3, "R_Q": 19, "necessity": True,
            "reader": True, "redundancy": True, "selected": True,
        },
        "minimum_valid_no_ceiling_raw_numeric_bytes": fixed,
        "non_array_metrics": "canonical_JSON_float64_scalars; not counted as raw numeric arrays",
        "reader_raw_numeric_bytes": reader,
        "redundancy_raw_numeric_bytes": redundancy,
        "retained_site_raw_numeric_bytes_formula": (
            "4608*(R_H+R_Q)+8*(320*R_H+416*R_Q)+23040*(R_H+R_Q)"
            "+8000*(R_H+R_Q)+20*(336*R_H+544*R_Q)"
        ),
        "selected_raw_numeric_bytes": selected,
        "single_necessity_raw_numeric_bytes": necessity,
        "array_order": {
            "endpoints": "FIT authority row order; base then donor",
            "sites": "H:-1..H:17 then Q:-1..Q:17 unless array name separates H/Q",
            "records": "v2 donor-manifest ordinal order after applicability filter",
            "seeds": "14001,14002,14003,14004,14005",
            "fit_configurations": "joint-rank1,A1_only-rank1,A2_only-rank1,joint-rank2,joint-rank4",
            "reader_variants": (
                "upstream_H_patch,upstream_H_patch_then_native_Q_reset,upstream_H_neutral,"
                "upstream_H_neutral_then_donor_Q_insert,downstream_Q_patch"
            ),
        },
    }


def _runtime_contract() -> dict[str, Any]:
    return {
        "arithmetic": {
            "model_weights": "float32",
            "model_residuals": "float32",
            "projector_parameters": "float32",
            "adam_state": "float32",
            "training_objective": "float32",
            "reported_metrics_and_selection": "float64_cpu",
            "training_median": "sorted_even_midpoint_with_autograd",
            "training_normalizer_gradient": "differentiate_through_current_projector_full_DISCOVERY_median_no_detach",
            "reported_quantile": "Hyndman_Fan_type7_float64",
            "tf32_allowed": False,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
        },
        "canary": {
            "pre_and_post_each_major_stage": True,
            "canary1_required_true": ["pa", "pb", "pc"],
            "canary1_required_finite": ["score_rank", "l1_cost", "ratio_5_6", "ratio_14_15"],
            "canary2_required_true": ["canary1", "atlases", "fingerprint_stable_vs_previous", "ALL"],
            "canary2_composition": "v2_layer17_mlp_plus_scalar",
            "canary2_fingerprint_sha256": "6b22b221a811382775e6a64b4198a61f2f9bcc55b826d0d12d0512d1a28be99c",
        },
        "checkpoint": {
            **CHECKPOINT,
            "verify_sha256_before_load": True,
            "verify_sha256_after_device_copy": True,
            "verify_sha256_after_each_major_stage": True,
            "verify_sha256_before_publication": True,
        },
        "deadline": {
            "hard_gpu_seconds": GPU_TIME_LIMIT_SECONDS,
            "external_watchdog_required": True,
            "monotonic_check_before_and_after_every_model_call": True,
            "no_call_may_start_without_remaining_reviewed_p99_call_budget": True,
            "no_automatic_retry": True,
            "preauthorization_throughput_receipt_required": True,
        },
        "dead_intervention_tripwires": {
            "synthetic_projector_formula_before_model_load": True,
            "live_exact_single_position_delta_check_every_call": True,
            "live_non_target_positions_bitwise_equal_every_call": True,
            "nonzero_expected_delta_must_be_live": True,
            "finite_collapsed_scientific_delta_is_gate_failure_not_instrument_error": True,
        },
        "interventions": {
            "sufficiency": "r_target + U@U.T@(r_donor-r_target) at exactly the registered token coordinate",
            "full_state_ceiling": "identity-projector replacement r_target <- r_donor at exactly one registered coordinate",
            "necessity": "r - u*(u.T@r-a0_DISCOVERY)",
            "two_site_order": "ascending boundary; second neutralization uses current activation after first",
            "reader_reset": "after H donor patch, replace selected-Q coefficient with native-target Q coefficient",
            "reader_rescue": "after H neutralization, insert natural-donor selected-Q coefficient",
        },
        "failure_semantics": {
            "nonfinite_or_hash_schema_runtime_optimizer_failure": "instrument_invalid",
            "finite_small_screen_or_recovery_ceiling": "scientific_gate_failure_not_instrument_invalid",
            "finite_collapsed_coordinate": "failed_state_hypothesis",
            "deadline_or_incomplete_call_index": "hard_abort_without_scientific_terminal",
            "partial_evidence_publication": False,
        },
        "expected_runtime": EXPECTED_RUNTIME,
        "model_forward": "wte->input_rmsnorm->18_native_blocks->final_rmsnorm->lm_head->30*tanh(logits/30)->float32",
        "intervention_call_semantics": (
            "every manifest forward reruns each target prompt from tokens through the full native prefix, applies "
            "the registered one- or two-site hook using only cached donor H/Q vectors, and continues the same "
            "forward; no full-sequence boundary or suffix cache exists"
        ),
        "model_execution_surface": (
            "future hash-bound full/suffix localization implementation with native-forward equivalence canary; "
            "facade validation only because its production dispatch shape is not this manifest's variable batch shape"
        ),
        "temporary_peak_storage": {
            "status": "implementation_dependent_not_falsely_claimed_exact_by_CPU_compiler",
            "largest_registered_batch": {"sequences": 192, "sequence_tokens": 8},
            "largest_projector_rank": 4,
            "simultaneous_training_graph_batches": 2,
            "retained_arrays_offloaded_to_CPU_after_each_call": True,
            "preauthorization_peak_receipt_required": True,
            "receipt_canary": (
                "hash-bound non-task canary at batch192_length8_rank4 with two forward graphs, one backward, "
                "one QR, one Adam update, float32, and the exact future producer/model execution surface"
            ),
            "receipt_required_fields": [
                "producer_sha256", "model_source_sha256", "checkpoint_weights_sha256",
                "runtime_versions", "cuda_device_identity", "max_memory_allocated_bytes",
                "max_memory_reserved_bytes", "call_shape", "receipt_sha256",
            ],
            "required_free_device_bytes_before_model_load": (
                "max(ceil(1.25*measured_max_memory_reserved_bytes),"
                "measured_max_memory_reserved_bytes+2147483648)"
            ),
            "runtime_guard": (
                "reset and read CUDA peak counters around every major stage; exceeding the separately approved "
                "receipt-derived cap or any OOM is hard_abort without scientific terminal or partial publication"
            ),
            "why_no_exact_allocator_number": (
                "PyTorch graph/allocator/workspace overhead depends on the future implementation and CUDA stack; "
                "inventing a byte total before that code exists would not be an auditable bound"
            ),
        },
        "namespace": {
            "name": "task14_fit_localization_v2_fit_v1",
            "create_only": True,
            "result": "basis_aligned/bilinear_quotient/circuit_battery_task14_fit_localization_v2_fit_v1_results.json",
            "evidence": "basis_aligned/bilinear_quotient/circuit_battery_task14_fit_localization_v2_fit_v1_evidence",
            "receipt": "basis_aligned/bilinear_quotient/circuit_battery_task14_fit_localization_v2_fit_v1_receipt.json",
            "publication": "linux_renameat2_noreplace_evidence_result_receipt_last_v1",
            "refuse_dangling_symlink_and_late_race": True,
        },
    }


def _dag() -> list[dict[str, Any]]:
    nodes = [
        {"node": "preflight", "after": [], "condition": "always", "failure": "instrument_invalid", "model_calls": False},
        {"node": "native_cache", "after": ["preflight"], "condition": "preflight_pass", "failure": "instrument_invalid", "model_calls": True},
        {"node": "discovery_gradients", "after": ["native_cache"], "condition": "native_cache_complete", "failure": "instrument_invalid", "model_calls": True},
        {"node": "discovery_full_ceilings", "after": ["discovery_gradients"], "condition": "gradient_cache_complete", "failure": "instrument_invalid_or_no_intervention_ceiling", "model_calls": True},
        {"node": "spectral_operator", "after": ["discovery_full_ceilings"], "condition": "retained_sites_known", "failure": "instrument_invalid", "model_calls": False, "selective": False},
        {"node": "joint_rank1_fits", "after": ["spectral_operator"], "condition": "H_and_Q_retained", "failure": "instrument_invalid", "model_calls": True},
        {"node": "spectral_finite_diagnostic", "after": ["joint_rank1_fits"], "condition": "joint_fits_complete", "failure": "instrument_invalid", "model_calls": True, "selective": False},
        {"node": "discovery_selection", "after": ["joint_rank1_fits"], "condition": "joint_fits_complete", "failure": "instrument_invalid", "model_calls": False},
        {"node": "selected_family_and_rank_fits", "after": ["discovery_selection"], "condition": "H_and_Q_selected", "failure": "instrument_invalid", "model_calls": True},
        {"node": "locked_validation", "after": ["selected_family_and_rank_fits"], "condition": "all_selected_fits_complete", "failure": "rank_or_semantic_terminal", "model_calls": True},
        {"node": "single_necessity", "after": ["locked_validation"], "condition": "rank1_semantic_and_falsifier_gates_pass", "failure": "sufficiency_only", "model_calls": True},
        {"node": "two_site_redundancy", "after": ["single_necessity"], "condition": "single_necessity_fails_and_top_two_exist", "failure": "sufficiency_only", "model_calls": True},
        {"node": "ordered_reader", "after": ["single_necessity", "two_site_redundancy"], "condition": "necessity_route_passes_and_H_precedes_Q", "failure": "reader_unresolved", "model_calls": True},
        {"node": "terminal_projection", "after": ["ordered_reader"], "condition": "reached_branch_complete", "failure": None, "model_calls": False},
    ]
    return [{**node, "guard_id": canonical_sha256(node)} for node in nodes]


def _compile_chunks() -> list[dict[str, Any]]:
    authority, partition, donors = load_inputs()
    rows, endpoints = _row_maps(authority)
    del rows
    _validate_endpoint_table(donors, endpoints)
    records, records_by_id = _record_maps(donors)
    chunks: list[dict[str, Any]] = []
    chunks.extend(_native_chunks(authority, partition, endpoints, records_by_id))
    for position in POSITIONS:
        for boundary in BOUNDARIES:
            chunks.append(_ceiling_chunk(
                partition_name="DISCOVERY", position=position, boundary=boundary,
                records=records, endpoints=endpoints, records_by_id=records_by_id,
                activation="gradient_cache_complete",
            ))
    for position, boundary, objective, rank, seed, activation in _fit_templates():
        chunks.append(_fit_chunk(
            position=position, boundary=boundary, objective=objective, rank=rank, seed=seed,
            records=records, endpoints=endpoints, records_by_id=records_by_id,
            activation=activation,
        ))
    for position in POSITIONS:
        for boundary in BOUNDARIES:
            chunks.append(_spectral_chunk(
                position=position, boundary=boundary, records=records,
                endpoints=endpoints, records_by_id=records_by_id,
            ))
            chunks.append(_ceiling_chunk(
                partition_name="VALIDATION", position=position, boundary=boundary,
                records=records, endpoints=endpoints, records_by_id=records_by_id,
                activation="site_is_selected_H_or_Q",
            ))
    for position, boundary, objective, rank, seed, _activation in _fit_templates():
        chunks.append(_projected_eval_chunk(
            partition_name="VALIDATION", position=position, boundary=boundary,
            objective=objective, rank=rank, seed=seed, records=records,
            endpoints=endpoints, records_by_id=records_by_id,
            activation="fit_is_selected_H_or_Q_configuration",
        ))
    for boundary in BOUNDARIES:
        for seed in SEEDS:
            chunks.append(_necessity_chunk_exact(
                authority=authority, partition=partition, boundary=boundary, seed=seed,
                endpoints=endpoints, records_by_id=records_by_id,
            ))
    for first_index, first in enumerate(BOUNDARIES):
        for second in BOUNDARIES[first_index + 1:]:
            for seed in SEEDS:
                chunks.append(_redundancy_chunk(
                    authority=authority, partition=partition, first=first, second=second,
                    seed=seed, endpoints=endpoints, records_by_id=records_by_id,
                ))
                chunks.append(_reader_chunk(
                    h_boundary=first, q_boundary=second, seed=seed, records=records,
                    endpoints=endpoints, records_by_id=records_by_id,
                ))
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise CompileError("chunk identity collision")
    return chunks


def visit_call_descriptors(visitor: Any) -> list[dict[str, Any]]:
    """Run the one canonical call traversal, visiting full descriptors in order."""
    global _CALL_VISITOR
    if _CALL_VISITOR is not None:
        raise CompileError("nested call-descriptor replay is forbidden")
    _CALL_VISITOR = visitor
    try:
        return _compile_chunks()
    finally:
        _CALL_VISITOR = None


def iter_call_descriptors() -> Iterator[tuple[str, dict[str, Any]]]:
    """Replay every canonical call descriptor in exact chunk/call order.

    A bounded producer queue keeps memory bounded without reopening mutable
    source paths after verification.  The compact checked-in index commits to
    every descriptor SHA-256; a future producer must use this iterator and
    compare each regenerated call ID before model access.
    """
    class ReplayStopped(Exception):
        pass

    sentinel = object()
    output: queue.Queue[Any] = queue.Queue(maxsize=8)
    stop = threading.Event()
    errors: list[BaseException] = []

    def put(value: Any) -> None:
        while not stop.is_set():
            try:
                output.put(value, timeout=0.1)
                return
            except queue.Full:
                continue
        raise ReplayStopped

    def visit(chunk_id: str, call: Mapping[str, Any]) -> None:
        put((chunk_id, dict(call)))

    def worker() -> None:
        try:
            visit_call_descriptors(visit)
        except ReplayStopped:
            pass
        except BaseException as error:  # propagated in the consumer thread
            errors.append(error)
        finally:
            try:
                put(sentinel)
            except ReplayStopped:
                pass

    thread = threading.Thread(target=worker, name="task14-v2-call-replay", daemon=True)
    thread.start()
    try:
        while True:
            value = output.get()
            if value is sentinel:
                if errors:
                    raise errors[0]
                break
            yield value
    finally:
        stop.set()
        thread.join(timeout=2.0)


def build_bundle() -> tuple[dict[str, Any], bytes]:
    call_index = bytearray()
    chunk_call_ids: dict[str, list[str]] = defaultdict(list)

    def visit(chunk_id: str, call: Mapping[str, Any]) -> None:
        call_id = str(call["call_id"])
        chunk_call_ids[chunk_id].append(call_id)
        call_index.extend(bytes.fromhex(call_id))

    chunks = visit_call_descriptors(visit)
    call_offset = 0
    for chunk in chunks:
        call_ids = chunk_call_ids[str(chunk["chunk_id"])]
        if len(call_ids) != int(chunk["call_count"]):
            raise CompileError("call index census differs from chunk census")
        encoded = b"".join(bytes.fromhex(call_id) for call_id in call_ids)
        chunk["call_index_count"] = len(call_ids)
        chunk["call_index_offset"] = call_offset
        chunk["call_index_slice_sha256"] = bytes_sha256(encoded)
        call_offset += len(call_ids)
    call_index_bytes = bytes(call_index)
    chunks_root = canonical_sha256(chunks)
    manifest: dict[str, Any] = {
        "artifact_closure": [
            {"path": path, "role": role, "sha256": digest}
            for role, (path, digest) in sorted(FROZEN.items())
        ],
        "call_chunk_count": len(chunks),
        "call_chunks": chunks,
        "call_chunks_root_sha256": chunks_root,
        "call_index": {
            "byte_count": len(call_index_bytes),
            "call_count": call_offset,
            "encoding": "ordered_raw_32_byte_SHA256_call_ids",
            "path": str(CALL_INDEX_PATH.relative_to(REPO_ROOT)),
            "sha256": bytes_sha256(call_index_bytes),
        },
        "conditional_price": _price_contract(chunks),
        "dag": _dag(),
        "experiment_id": EXPERIMENT_ID,
        "fit_only": {
            "allowed_authority_paths": [FROZEN["fit_authority"][0], FROZEN["v2_partition"][0], FROZEN["v2_donors"][0]],
            "forbidden_phases": ["SELECT", "TEST", "OOD"],
            "phase": PHASE,
        },
        "initialization": {
            "logical_rule": "SHA256 Rademacher matrix then reduced QR with positive R diagonal",
            "replay_required_before_fit": True,
            "seeds": list(SEEDS),
        },
        "model_contract": {
            "boundaries": list(BOUNDARIES), "positions": list(POSITIONS), "width": WIDTH,
            "checkpoint": CHECKPOINT, "runtime": EXPECTED_RUNTIME,
            "boundary_semantics": {
                "-1": "normalized embedding input before block 0",
                **{str(boundary): f"residual after complete block {boundary}; suffix resumes at block {boundary + 1}"
                   for boundary in range(18)},
            },
            "suffix_semantics_at_17": "final_rmsnorm_then_lm_head_then_softcap",
        },
        "physical_batching": {
            "equal_sequence_length_required": True,
            "evaluation_batch_limit": EVALUATION_BATCH_LIMIT,
            "fit_intervention_batch_limit": INTERVENTION_BATCH_LIMIT,
            "logical_relations_per_update": LOGICAL_RELATIONS_PER_STEP,
            "record_order": "frozen donor ordinal except exact preregistered SHA sampler",
            "runtime_replay": (
                "capture and hash-verify this compiler plus every artifact_closure role; regenerate each canonical "
                "call descriptor, including item_ids, token/position binding, optimizer uses, and A_C endpoint "
                "slots; require its call_id to equal the next 32-byte call-index entry before model access"
            ),
        },
        "retained_arrays": _retained_arrays(),
        "retained_byte_contract": _retained_byte_contract(),
        "runtime_and_publication": _runtime_contract(),
        "schema": SCHEMA,
        "science": {
            "decision_contract": {
                "authority": "v2_preregistration_sections_4_through_12_exactly",
                "available_ceiling": {"direction_at_least": 0.65, "mean_signed_effect_gt": 1e-6},
                "discovery": {
                    "eligible_H": "all mandatory H answer-changing cells available; paired A1/A2 direction >=0.80",
                    "eligible_Q": "eligible_H conditions plus all four C<->ordinary-singular cells available",
                    "retained_H": 3,
                    "retained_Q": "all eligible Q",
                    "selected_H": "largest float64 median-seed joint objective; tie earlier boundary",
                    "selected_Q": "v2 section 8 T_b onset rule; fallback finite argmax tie earlier and reader forced unresolved",
                    "top_two_Q": "descending raw float64 T_b; tie earlier boundary",
                },
                "fit": {
                    "steps": FIT_STEPS, "logical_relations_per_step": LOGICAL_RELATIONS_PER_STEP,
                    "learning_rate": "0.03*(1+cos(pi*t/399))/2",
                    "adam": {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "weight_decay": 0.0},
                    "all_five_seeds_must_be_healthy": True,
                },
                "validation_thresholds": {
                    "ordinary_paired_and_cross_noun": {"direction": 0.80, "recovery": 0.50},
                    "cross_syntax": {"direction": 0.75, "recovery": 0.40},
                    "P_positive": {"direction": 0.75, "recovery": 0.40},
                    "family_crossfit": {"direction": 0.70, "recovery": 0.35},
                    "C_singular_bidirectional": {"direction": 0.70, "recovery": 0.35},
                    "ordinary_number_attractor_cell_direction": 0.70,
                    "ordinary_P_and_paired_C_leakage": {"mean_output_max": 0.20, "median_coordinate_max": 0.20, "p90_coordinate_max": 0.50},
                    "C_ordinary_plural_leakage": {"mean_output_max": 0.25, "median_coordinate_max": 0.35, "p90_coordinate_max": 0.75},
                    "C_absolute_alignment": {"base_median_min": 0.50, "donor_median_min": 0.50, "each_side_positive_fraction_min": 0.80, "pooled_q1_strict_gt": 0.0},
                    "higher_rank_improvement_strict_gt": 0.10,
                    "single_necessity": {"ratio_min": 0.25, "positive_fraction_min": 0.65},
                    "redundancy": {"each_single_strict_max": 0.25, "joint_min": 0.50, "interaction_min": 0.20, "positive_fraction_min": 0.65},
                    "reader": {"reset_mediation_min": 0.70, "rescue_min": 0.70, "rescue_overshoot_max": 1.25},
                },
                "seed_gate": "medoid_pass AND median_of_five_pass AND at_least_four_of_five_pass",
                "terminal_precedence": [
                    "instrument_invalid", "no_intervention_ceiling",
                    "fit_binary_state_rejected_higher_rank_needed_or_better",
                    "fit_rank1_complete_subject_state_not_identified",
                    "fit_rank1_state_sufficiency_only",
                    "fit_rank1_state_and_ordered_reader_supported",
                    "fit_rank1_redundant_state_and_ordered_reader_supported",
                    "fit_rank1_state_supported_reader_unresolved",
                    "fit_rank1_two_site_redundant_state_reader_unresolved",
                ],
            },
            "spectral": {
                "operator": "A v = mean sigma/2 * [g*(delta^T v)+delta*(g^T v)]",
                "arithmetic": "float64_cpu_Lanczos_64_iterations_full_reorthogonalization",
                "lanczos_start": "normalized_SHA256_Rademacher(task14-v2-spectral-lanczos|site|dimension)",
                "reorthogonalization": "two_pass_modified_Gram_Schmidt_against_prior_vectors_in_iteration_order",
                "breakdown": "nonfinite_is_instrument_invalid;finite_beta_le_1e-12_is_valid_invariant_subspace_stop",
                "ritz_selection": "largest_algebraic_float64_eigenvalue;projector_sign_invariant;report_top_gap",
                "record_weighting": "same unweighted signed affirmative DISCOVERY cell means as the corresponding H/Q joint objective; omit controls and A_C",
                "cell_coefficients": {
                    "H": {"A1": 1.0, "A2": 1.0, "X1": 0.5, "X2": 0.5, "P": 0.5},
                    "Q": {"A1": 1.0, "A2": 1.0, "X1": 0.5, "X2": 0.5, "P": 0.5, "CS": 1.0},
                },
                "outputs": ["projector_distance", "finite_vs_local_Pearson", "finite_minus_local_RMSE"],
                "uses": "DISCOVERY_only_diagnostic_and_reported_initializer_candidate_not_used_by_registered_DAS",
                "success_predicate": False,
                "validation_selector": False,
                "registered_DAS_initialization_changed": False,
            },
            "terminal_precedence": [
                "instrument_invalid", "no_intervention_ceiling",
                "fit_binary_state_rejected_higher_rank_needed_or_better",
                "fit_rank1_complete_subject_state_not_identified",
                "fit_rank1_state_sufficiency_only",
                "fit_rank1_state_and_ordered_reader_supported",
                "fit_rank1_redundant_state_and_ordered_reader_supported",
                "fit_rank1_state_supported_reader_unresolved",
                "fit_rank1_two_site_redundant_state_reader_unresolved",
            ],
        },
        "source_commit": V2_COMMIT,
        "source_review_commit": V2_REVIEW_COMMIT,
        "source_review_sha256": V2_REVIEW_SHA256,
        "status": "prospective_compiler_review_only",
        "task_id": TASK_ID,
    }
    manifest["contract_sha256"] = canonical_sha256(manifest)
    validate_manifest(manifest)
    return manifest, call_index_bytes


def build_manifest() -> dict[str, Any]:
    return build_bundle()[0]


def _sum_price(chunks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        key: sum(int(chunk[key]) for chunk in chunks)
        for key in (
            "forward_calls", "backward_calls", "backward_graph_batches",
            "optimizer_updates", "example_evaluations", "token_evaluations",
        )
    }


def _price_contract(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    static = [
        chunk for chunk in chunks
        if chunk["chunk_id"] in {"00_native_cache", "01_discovery_gradients"}
        or str(chunk["chunk_id"]).startswith("ceiling:DISCOVERY:")
    ]
    all_templates = _sum_price(chunks)

    def take(chunk_id: str) -> Mapping[str, Any]:
        try:
            return by_id[chunk_id]
        except KeyError as error:
            raise CompileError(f"missing price chunk: {chunk_id}") from error

    # Cost is boundary-invariant within H and within Q because the authority rows
    # and equal-length batching are identical.  Representative boundaries make
    # this exact without choosing a scientific site.
    maximum_chunks: list[Mapping[str, Any]] = list(static)
    for boundary in BOUNDARIES:
        for seed in SEEDS:
            maximum_chunks.append(take(f"fit:Q:{boundary}:joint:rank1:seed{seed}"))
        maximum_chunks.append(take(f"spectral:Q:{boundary}"))
    for boundary in BOUNDARIES[:3]:
        for seed in SEEDS:
            maximum_chunks.append(take(f"fit:H:{boundary}:joint:rank1:seed{seed}"))
        maximum_chunks.append(take(f"spectral:H:{boundary}"))
    for position, boundary in (("H", BOUNDARIES[0]), ("Q", BOUNDARIES[1])):
        maximum_chunks.append(take(f"ceiling:VALIDATION:{position}:{boundary}"))
        for objective, rank in (("joint", 1), ("A1_only", 1), ("A2_only", 1), ("joint", 2), ("joint", 4)):
            for seed in SEEDS:
                if not (objective == "joint" and rank == 1):
                    maximum_chunks.append(take(
                        f"fit:{position}:{boundary}:{objective}:rank{rank}:seed{seed}"
                    ))
                maximum_chunks.append(take(
                    f"eval:VALIDATION:{position}:{boundary}:{objective}:rank{rank}:seed{seed}"
                ))
    for seed in SEEDS:
        maximum_chunks.append(take(f"necessity:Q:{BOUNDARIES[1]}:seed{seed}"))
        maximum_chunks.append(take(f"redundancy:Q:{BOUNDARIES[1]}:{BOUNDARIES[2]}:seed{seed}"))
        maximum_chunks.append(take(f"reader:H:{BOUNDARIES[0]}:Q:{BOUNDARIES[1]}:seed{seed}"))
    maximum_active = _sum_price(maximum_chunks)
    single_necessity_reader_path = _sum_price([
        chunk for chunk in maximum_chunks
        if not str(chunk["chunk_id"]).startswith("redundancy:")
    ])
    if maximum_active["optimizer_updates"] != 60_000:
        raise CompileError("maximum active logical update price changed")
    return {
        "all_conditional_templates_not_one_execution": all_templates,
        "hard_gpu_seconds": GPU_TIME_LIMIT_SECONDS,
        "logical_update_ceiling": 60_000,
        "maximum_active_upper_bound": maximum_active,
        "minimum_valid_prefix_before_no_ceiling_terminal": _sum_price(static),
        "reference_full_path_single_necessity_and_reader": single_necessity_reader_path,
        "price_terms_are_distinct": {
            "backward_calls": "one autograd backward per logical optimizer update",
            "backward_graph_batches": "forward batches whose graphs contribute to those backwards",
            "example_evaluations": "physical prompt-sequence evaluations across all forward batches",
            "forward_calls": "physical model/suffix forward batches",
            "optimizer_updates": "logical Adam updates; not model-forward calls",
            "token_evaluations": "sum(batch_size*equal_sequence_length) across physical forward batches",
        },
        "runtime_active_plan": "sum exact activated chunk receipts; root must be recomputed before each stage",
        "worst_case_active_formula": {
            "static": "native+gradients+all_38_discovery_ceiling_chunks",
            "joint_rank1": "all_19_Q plus exactly_3_H sites, five seeds, 400 updates",
            "selected_fits": "one_H+one_Q times A1_only+A2_only+rank2+rank4, five seeds, 400 updates",
            "diagnostic": "spectral finite calls on the same 22 retained sites",
            "validation": "one selected H and Q across full ceilings and five fit configurations x five seeds",
            "necessity": "selected_Q x five seeds",
            "redundancy": "one top_two_Q pair x five seeds only after singleton failure",
            "reader": "one ordered selected H,Q pair x five seeds only after a necessity route passes",
        },
        "maximum_is_conservative_and_branch_complete": (
            "Includes both two-site redundancy and its subsequent ordered-reader branch. "
            "Boundary representatives are exact because price is position-specific but boundary-invariant."
        ),
        "template_lookup_sha256": canonical_sha256(sorted(by_id)),
    }


def validate_manifest(value: Mapping[str, Any]) -> None:
    if value.get("schema") != SCHEMA or value.get("task_id") != TASK_ID:
        raise CompileError("manifest identity changed")
    if value.get("source_commit") != V2_COMMIT \
            or value.get("source_review_commit") != V2_REVIEW_COMMIT \
            or value.get("source_review_sha256") != V2_REVIEW_SHA256:
        raise CompileError("manifest source authority changed")
    expected_closure = [
        {"path": path, "role": role, "sha256": digest}
        for role, (path, digest) in sorted(FROZEN.items())
    ]
    if value.get("artifact_closure") != expected_closure:
        raise CompileError("artifact closure changed")
    observed = dict(value)
    contract = observed.pop("contract_sha256", None)
    if contract != canonical_sha256(observed):
        raise CompileError("manifest contract hash mismatch")
    chunks = value.get("call_chunks")
    if not isinstance(chunks, list) or value.get("call_chunk_count") != len(chunks):
        raise CompileError("chunk census mismatch")
    if value.get("call_chunks_root_sha256") != canonical_sha256(chunks):
        raise CompileError("chunk root mismatch")
    if len({chunk.get("chunk_id") for chunk in chunks}) != len(chunks):
        raise CompileError("duplicate chunk identity")
    for chunk in chunks:
        if chunk.get("schema") != CHUNK_SCHEMA or int(chunk.get("call_count", -1)) < 0:
            raise CompileError("malformed call chunk")
        if int(chunk["forward_calls"]) != int(chunk["call_count"]):
            raise CompileError("one-forward-per-call invariant changed")
        if int(chunk["optimizer_updates"]) not in {0, FIT_STEPS}:
            raise CompileError("optimizer update count changed")
        if int(chunk["backward_calls"]) not in {0, 4, FIT_STEPS}:
            raise CompileError("backward call count changed")
        if int(chunk.get("call_index_count", -1)) != int(chunk["call_count"]):
            raise CompileError("per-call index chunk census changed")
    index = value.get("call_index")
    if not isinstance(index, dict) \
            or index.get("encoding") != "ordered_raw_32_byte_SHA256_call_ids" \
            or int(index.get("byte_count", -1)) != 32 * int(index.get("call_count", -1)) \
            or int(index.get("call_count", -1)) != sum(int(chunk["call_count"]) for chunk in chunks):
        raise CompileError("per-call index contract changed")
    expected_offset = 0
    for chunk in chunks:
        if int(chunk.get("call_index_offset", -1)) != expected_offset:
            raise CompileError("per-call index offsets changed")
        expected_offset += int(chunk["call_index_count"])
    forbidden_paths = [
        item["path"] for item in value["artifact_closure"]
        if any(token in str(item["path"]).lower() for token in ("select_authority", "test_authority", "ood_authority", "results.json", "evidence"))
    ]
    if forbidden_paths:
        raise CompileError(f"future/outcome artifact leaked: {forbidden_paths}")
    if value["science"]["spectral"] != {
        "operator": "A v = mean sigma/2 * [g*(delta^T v)+delta*(g^T v)]",
        "arithmetic": "float64_cpu_Lanczos_64_iterations_full_reorthogonalization",
        "lanczos_start": "normalized_SHA256_Rademacher(task14-v2-spectral-lanczos|site|dimension)",
        "reorthogonalization": "two_pass_modified_Gram_Schmidt_against_prior_vectors_in_iteration_order",
        "breakdown": "nonfinite_is_instrument_invalid;finite_beta_le_1e-12_is_valid_invariant_subspace_stop",
        "ritz_selection": "largest_algebraic_float64_eigenvalue;projector_sign_invariant;report_top_gap",
        "record_weighting": "same unweighted signed affirmative DISCOVERY cell means as the corresponding H/Q joint objective; omit controls and A_C",
        "cell_coefficients": {
            "H": {"A1": 1.0, "A2": 1.0, "X1": 0.5, "X2": 0.5, "P": 0.5},
            "Q": {"A1": 1.0, "A2": 1.0, "X1": 0.5, "X2": 0.5, "P": 0.5, "CS": 1.0},
        },
        "outputs": ["projector_distance", "finite_vs_local_Pearson", "finite_minus_local_RMSE"],
        "uses": "DISCOVERY_only_diagnostic_and_reported_initializer_candidate_not_used_by_registered_DAS",
        "success_predicate": False,
        "validation_selector": False,
        "registered_DAS_initialization_changed": False,
    }:
        raise CompileError("spectral diagnostic became selective")
    if value["runtime_and_publication"]["deadline"]["hard_gpu_seconds"] != GPU_TIME_LIMIT_SECONDS:
        raise CompileError("GPU time ceiling changed")
    if value.get("runtime_and_publication") != _runtime_contract():
        raise CompileError("runtime/publication contract changed")
    if value.get("dag") != _dag():
        raise CompileError("conditional DAG changed")
    if value.get("fit_only") != {
        "allowed_authority_paths": [FROZEN["fit_authority"][0], FROZEN["v2_partition"][0], FROZEN["v2_donors"][0]],
        "forbidden_phases": ["SELECT", "TEST", "OOD"], "phase": PHASE,
    }:
        raise CompileError("FIT-only closure changed")
    arrays = value.get("retained_arrays")
    if not isinstance(arrays, list) or len({item["name"] for item in arrays}) != len(arrays):
        raise CompileError("retained array contract changed")
    if any(item.get("dtype") not in {"float32", "float64"} for item in arrays):
        raise CompileError("retained dtype changed")
    if arrays != _retained_arrays() or value.get("retained_byte_contract") != _retained_byte_contract():
        raise CompileError("retained array/byte contract changed")
    if value.get("conditional_price", {}).get("logical_update_ceiling") != 60_000:
        raise CompileError("logical update ceiling changed")


def validate_call_index(value: Mapping[str, Any], raw: bytes) -> None:
    index = value["call_index"]
    if len(raw) != int(index["byte_count"]) \
            or bytes_sha256(raw) != index["sha256"]:
        raise CompileError("per-call index bytes changed")
    for chunk in value["call_chunks"]:
        start = 32 * int(chunk["call_index_offset"])
        stop = start + 32 * int(chunk["call_index_count"])
        if bytes_sha256(raw[start:stop]) != chunk["call_index_slice_sha256"]:
            raise CompileError(f"per-call index slice changed: {chunk['chunk_id']}")
        ids = [raw[offset:offset + 32].hex() for offset in range(start, stop, 32)]
        root = _chain_seed(str(chunk["chunk_id"]))
        for call_id in ids:
            root = _chain_step(root, call_id)
        if root != chunk["call_root_sha256"]:
            raise CompileError(f"per-call chain replay failed: {chunk['chunk_id']}")


def check_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    expected, expected_index = build_bundle()
    raw = safe_read(path)
    observed = strict_json(raw, "materialized call manifest")
    if observed != expected or raw != canonical_bytes(expected, newline=True):
        raise CompileError("materialized call manifest differs from deterministic compiler")
    observed_index = safe_read(CALL_INDEX_PATH)
    validate_call_index(observed, observed_index)
    if observed_index != expected_index:
        raise CompileError("materialized per-call index differs from deterministic compiler")
    return expected


def dryrun_report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "call_chunk_count": manifest["call_chunk_count"],
        "call_chunks_root_sha256": manifest["call_chunks_root_sha256"],
        "call_index_call_count": manifest["call_index"]["call_count"],
        "call_index_sha256": manifest["call_index"]["sha256"],
        "compiler_model_calls": 0,
        "contract_sha256": manifest["contract_sha256"],
        "forbidden_phases_opened": [],
        "gpu_accesses": 0,
        "hard_gpu_seconds": GPU_TIME_LIMIT_SECONDS,
        "logical_update_ceiling": 60_000,
        "manifest_sha256": bytes_sha256(canonical_bytes(manifest, newline=True)),
        "outcome_artifacts_opened": [],
        "schema": "task14_fit_localization_v2_compiler_dryrun_v1",
        "status": "PASS",
    }


def write_artifacts(output_dir: Path) -> None:
    manifest, call_index = build_bundle()
    report = dryrun_report(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in ((MANIFEST_PATH.name, manifest), (DRYRUN_PATH.name, report)):
        with (output_dir / filename).open("xb") as handle:
            handle.write(canonical_bytes(value, newline=True))
    with (output_dir / CALL_INDEX_PATH.name).open("xb") as handle:
        handle.write(call_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--write-dir", type=Path)
    args = parser.parse_args()
    if args.check:
        manifest = check_manifest()
        report = dryrun_report(manifest)
        if safe_read(DRYRUN_PATH) != canonical_bytes(report, newline=True):
            raise CompileError("checked-in dry run differs from exact compiler")
        print(json.dumps(report, sort_keys=True))
    else:
        write_artifacts(args.write_dir)


if __name__ == "__main__":
    main()
