#!/usr/bin/env python3
# BQLANE: cpu
"""Compile the repaired exact conditional physical plan for task14 FIT localization v2.

This module is deliberately model-free.  It reads only the frozen FIT authority,
the reviewed v2 partition/donor authorities, and frozen source/preregistration
bytes.  It enumerates every possible model call as replayable hash-chain chunks;
it neither imports torch nor opens a checkpoint, GPU, queue, outcome, or later
phase artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import stat
import threading
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA = "task14_fit_localization_v2_physical_compiler_v2"
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
CANONICAL_CALL_CHUNK_COUNT = 3_821
CANONICAL_CALL_CHUNKS_ROOT_SHA256 = "073ed886dd051aae2610d1aa771bce6c3012e25dca007c7614283ea9cac732ef"
CANONICAL_CALL_COUNT = 743_881
CANONICAL_CALL_INDEX_SHA256 = "ae399e393d03af9b6232b7fc5339dd892b418ec7c88943735f8b72fc064c8ad9"

V2_COMMIT = "8f41f51cdf7e073063201cc48760622607ce91b9"
V2_REVIEW_COMMIT = "2ffd6cf77998a6c7fb6af0c4e89c742bf1bbb923"
V2_REVIEW_SHA256 = "2905aeb040fad2d16062a22e3c4d32d9dd6953c468724ff51a80ab9fa849d384"

REPO_ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
MANIFEST_PATH = OPS / "circuit_battery_task14_fit_localization_v2_call_manifest_v2.json"
CALL_INDEX_PATH = OPS / "circuit_battery_task14_fit_localization_v2_call_index_v2.bin"
DRYRUN_PATH = OPS / "circuit_battery_task14_fit_localization_v2_compiler_v2_dryrun.json"

BLOCKED_COMPILER_COMMIT = "ea16e22d28d125274ca4353f46e434c2826e0b02"
BLOCK_REVIEW_COMMIT = "45db7e2f2e2df3627c594b7df67dc0173aae318b"
BLOCK_REVIEW_SHA256 = "673389c02ec4d7e9122557fe4fb44ab9f90950ccf8e6efbbd310ac6d543548b1"

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
    "blocked_compiler_review": (
        "basis_aligned/polynomial_causal/TASK14_FIT_LOCALIZATION_V2_PHYSICAL_COMPILER_INDEPENDENT_REVIEW_2026-09-04.md",
        BLOCK_REVIEW_SHA256,
    ),
    "producer_acceptance": (
        "basis_aligned/polynomial_causal/TASK14_FIT_LOCALIZATION_V2_PRODUCER_ACCEPTANCE_2026-09-04.md",
        "1724fa6de7ece875cd633976841159302e04033ca008af6e6437ee159a935b46",
    ),
    "producer_acceptance_addendum": (
        "basis_aligned/polynomial_causal/TASK14_FIT_LOCALIZATION_V2_PRODUCER_ACCEPTANCE_ADDENDUM_2026-09-04.md",
        "c28e6dc2a453a08027673a2420bbf2053e94a0cb02b18a6f0579f747c81a4d96",
    ),
    "producer_acceptance_addendum2": (
        "basis_aligned/polynomial_causal/TASK14_FIT_LOCALIZATION_V2_PRODUCER_ACCEPTANCE_ADDENDUM2_2026-09-04.md",
        "30f11f8a6c4efd8e9dc6e3eb97cbb79bfbdbed21f0d27a622d268002824be18b",
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


class OperationalAbort(RuntimeError):
    """A preflight/runtime guard failed before a scientific terminal existed."""


TERMINALS = (
    "instrument_invalid",
    "no_intervention_ceiling",
    "fit_binary_state_rejected_higher_rank_needed_or_better",
    "fit_rank1_complete_subject_state_not_identified",
    "fit_rank1_state_sufficiency_only",
    "fit_rank1_state_and_ordered_reader_supported",
    "fit_rank1_redundant_state_and_ordered_reader_supported",
    "fit_rank1_state_supported_reader_unresolved",
    "fit_rank1_two_site_redundant_state_reader_unresolved",
)


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
            "synthetic_boundary_minus1_x0_and_live_v1_sensitive": True,
            "synthetic_boundary_nonnegative_preserves_target_x0_v1": True,
            "synthetic_composed_later_site_inherits_earlier_edited_trajectory": True,
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
            "preflight_hash_schema_runtime_nonfinite_incomplete_failure": "operational_abort_without_package",
            "completed_finite_optimizer_or_seed_health_failure": "instrument_invalid_publishable_scientific_terminal",
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
        "branch_projection": {
            "typed_state": "BranchState",
            "pure_projector": "project_terminal",
            "operational_fault_has_no_scientific_terminal": True,
            "all_inapplicable_nodes_explicitly_skipped": True,
            "single_and_redundancy_success_mutually_exclusive": True,
        },
        "call_index_replay": {
            "before_model": "preflight_global_call_index verifies all descriptors and all 32-byte entries",
            "active_chunk": "offset/count slice and root replay immediately before calls",
            "inactive_chunk": "canonical inactive_skip_zero_calls receipt",
            "active_path_root": True,
        },
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
            "preflight": "lstat_all_three_paths_then_future_renameat2_NOREPLACE_publication",
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


def _artifact_closure_contract() -> list[dict[str, str]]:
    return [
        {"path": path, "role": role, "sha256": digest}
        for role, (path, digest) in sorted(FROZEN.items())
    ]


def _fit_only_contract() -> dict[str, Any]:
    return {
        "allowed_authority_paths": [
            FROZEN["fit_authority"][0], FROZEN["v2_partition"][0],
            FROZEN["v2_donors"][0],
        ],
        "forbidden_phases": ["SELECT", "TEST", "OOD"],
        "phase": PHASE,
    }


def _initialization_contract() -> dict[str, Any]:
    return {
        "logical_rule": "SHA256 Rademacher matrix then reduced QR with positive R diagonal",
        "label": "task14-v2-das-init|position|boundary|objective|rank|seed|width",
        "rademacher_bit_order": "row_major_LSB_first_from_SHA256_counter_blocks",
        "counter_encoding": "unsigned_big_endian_8_bytes_starting_at_zero",
        "qr": "float32_reduced_QR_then_column_sign_from_R_diagonal_nonnegative_zero_maps_positive",
        "replay_required_before_fit": True,
        "ranks": list(RANKS),
        "seeds": list(SEEDS),
    }


def _model_contract() -> dict[str, Any]:
    return {
        "boundaries": list(BOUNDARIES),
        "positions": list(POSITIONS),
        "width": WIDTH,
        "checkpoint": CHECKPOINT,
        "runtime": EXPECTED_RUNTIME,
        "boundary_semantics": {
            "-1": "normalized embedding input after input RMSNorm and before block 0; suffix starts block 0",
            **{
                str(boundary): (
                    f"residual after complete block {boundary}; suffix resumes at block {boundary + 1}"
                )
                for boundary in range(18)
            },
        },
        "auxiliary_state_semantics": {
            "boundary_-1": (
                "edit normalized embedding before x0 exists; set x0 to the edited normalized embedding; "
                "derive v1 live from that edited x0 inside block0; never copy donor x0 or v1"
            ),
            "boundary_0_through_17": (
                "compute target native prefix x0 and v1; edit only the registered residual coordinate; "
                "preserve target x0 and v1; never copy donor x0 or v1"
            ),
            "composed_sites": (
                "apply ascending boundaries in one live forward; every later site sees the full trajectory "
                "caused by earlier edits, including its edited x0/live-derived v1 state; never restart from a native cache"
            ),
        },
        "suffix_semantics_at_17": "final_rmsnorm_then_lm_head_then_softcap",
    }


def _physical_batching_contract() -> dict[str, Any]:
    return {
        "equal_sequence_length_required": True,
        "evaluation_batch_limit": EVALUATION_BATCH_LIMIT,
        "fit_intervention_batch_limit": INTERVENTION_BATCH_LIMIT,
        "logical_relations_per_update": LOGICAL_RELATIONS_PER_STEP,
        "record_order": "frozen donor ordinal except exact preregistered SHA sampler",
        "runtime_replay": {
            "pre_model_global_preflight": (
                "safe-capture the entire call-index; verify byte count/hash; regenerate every canonical descriptor "
                "in global order and compare every 32-byte call id before model load"
            ),
            "active_chunk": (
                "seek the frozen offset/count slice; verify slice SHA and hash-chain root; regenerate descriptors "
                "from this compiler and compare each id before executing the chunk"
            ),
            "inactive_chunk": (
                "emit an explicit canonical skip receipt with chunk id, offset, count, slice SHA, activation guard, "
                "and evaluated false branch state; execute zero calls"
            ),
            "active_path_root": "SHA256 canonical ordered active-and-inactive chunk receipts",
        },
    }


def _decision_contract() -> dict[str, Any]:
    return {
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
            "steps": FIT_STEPS,
            "logical_relations_per_step": LOGICAL_RELATIONS_PER_STEP,
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
            "ordinary_P_and_paired_C_leakage": {
                "mean_output_max": 0.20, "median_coordinate_max": 0.20,
                "p90_coordinate_max": 0.50,
            },
            "C_ordinary_plural_leakage": {
                "mean_output_max": 0.25, "median_coordinate_max": 0.35,
                "p90_coordinate_max": 0.75,
            },
            "C_absolute_alignment": {
                "base_median_min": 0.50, "donor_median_min": 0.50,
                "each_side_positive_fraction_min": 0.80, "pooled_q1_strict_gt": 0.0,
            },
            "higher_rank_improvement_strict_gt": 0.10,
            "single_necessity": {"ratio_min": 0.25, "positive_fraction_min": 0.65},
            "redundancy": {
                "each_single_strict_max": 0.25, "joint_min": 0.50,
                "interaction_min": 0.20, "positive_fraction_min": 0.65,
            },
            "reader": {"reset_mediation_min": 0.70, "rescue_min": 0.70, "rescue_overshoot_max": 1.25},
        },
        "seed_gate": "medoid_pass AND median_of_five_pass AND at_least_four_of_five_pass",
        "terminal_precedence": list(TERMINALS),
    }


def _spectral_contract() -> dict[str, Any]:
    return {
        "operator": "A v = mean sigma/2 * [g*(delta^T v)+delta*(g^T v)]",
        "arithmetic": "float64_cpu_Lanczos_64_iterations_full_reorthogonalization",
        "lanczos_start": "normalized_SHA256_Rademacher(task14-v2-spectral-lanczos|site|dimension)",
        "reorthogonalization": "two_pass_modified_Gram_Schmidt_against_prior_vectors_in_iteration_order",
        "breakdown": "nonfinite_is_instrument_invalid;finite_beta_le_1e-12_is_valid_invariant_subspace_stop",
        "ritz_selection": "largest_algebraic_float64_eigenvalue;projector_sign_invariant;report_top_gap",
        "record_weighting": (
            "same unweighted signed affirmative DISCOVERY cell means as the corresponding H/Q joint objective; "
            "omit controls and A_C"
        ),
        "cell_coefficients": {
            "H": {"A1": 1.0, "A2": 1.0, "X1": 0.5, "X2": 0.5, "P": 0.5},
            "Q": {"A1": 1.0, "A2": 1.0, "X1": 0.5, "X2": 0.5, "P": 0.5, "CS": 1.0},
        },
        "outputs": ["projector_distance", "finite_vs_local_Pearson", "finite_minus_local_RMSE"],
        "uses": "DISCOVERY_only_diagnostic_and_reported_initializer_candidate_not_used_by_registered_DAS",
        "success_predicate": False,
        "validation_selector": False,
        "registered_DAS_initialization_changed": False,
    }


def _science_contract() -> dict[str, Any]:
    return {
        "decision_contract": _decision_contract(),
        "spectral": _spectral_contract(),
        "terminal_precedence": list(TERMINALS),
    }


def _identity_contract() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "source_commit": V2_COMMIT,
        "source_review_commit": V2_REVIEW_COMMIT,
        "source_review_sha256": V2_REVIEW_SHA256,
        "blocked_compiler_commit": BLOCKED_COMPILER_COMMIT,
        "block_review_commit": BLOCK_REVIEW_COMMIT,
        "block_review_sha256": BLOCK_REVIEW_SHA256,
        "acceptance_context": {
            "parent_commit": "ecb37c0ab",
            "parent_sha256": FROZEN["producer_acceptance"][1],
            "addendum_commit": "ea50dcfdf",
            "addendum_sha256": FROZEN["producer_acceptance_addendum"][1],
            "addendum2_commit": "3b3920ac8",
            "addendum2_sha256": FROZEN["producer_acceptance_addendum2"][1],
            "authority": "prospective_context_only_not_execution_authority",
        },
        "status": "prospective_repaired_compiler_review_only",
        "task_id": TASK_ID,
    }


def _branch_contract() -> dict[str, Any]:
    return {
        "schema": "task14_fit_localization_v2_branch_state_v1",
        "fields": {
            "operational_fault": "required_bool_default_false",
            "eligible_h_count": "nullable_nonnegative_int",
            "eligible_q_count": "nullable_nonnegative_int",
            "fit_health_stage": (
                "null_or_joint_rank1_failed_or_selected_family_rank_failed_or_selected_family_rank_passed; "
                "failure means every scheduled call in exactly that stage completed with finite evidence"
            ),
            "higher_rank_rescue": "nullable_bool",
            "semantic_gates_pass": "nullable_bool",
            "single_necessity_pass": "nullable_bool",
            "redundancy_available": "nullable_bool",
            "redundancy_pass": "nullable_bool",
            "h_before_q": "nullable_bool",
            "reader_pass": "nullable_bool",
        },
        "scientific_terminals": list(TERMINALS),
        "operational_disposition": "operational_abort_without_scientific_terminal_or_package",
        "skipped_nodes": "all inapplicable downstream fields must be null and every DAG node gets an explicit status",
        "exclusive_routes": "single-site necessity and two-site redundancy cannot both be successful",
        "projector": "project_terminal pure CPU function in this exact captured compiler",
    }


def _active_plan_contract() -> dict[str, Any]:
    return {
        "schema": "task14_fit_localization_v2_active_plan_state_v1",
        "source": "only frozen discovery/validation decisions projected by ActivePlanState",
        "selected_configuration_rule": "all and only registered fits/evaluations at exact selected H and Q",
        "redundancy_rule": "only the exact top-two-Q pair after selected-Q singleton necessity failure",
        "reader_rule": "only selected H<Q after either exclusive necessity route succeeds",
        "guard_evaluator": "evaluate_chunk_guard parses canonical chunk_id and verifies exact activation literal",
        "receipt_rule": "every chunk gets active_slice_verified or inactive_skip_zero_calls",
    }


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
        "artifact_closure": _artifact_closure_contract(),
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
        "active_plan_contract": _active_plan_contract(),
        "branch_state_contract": _branch_contract(),
        "conditional_price": _price_contract(chunks),
        "dag": _dag(),
        "fit_only": _fit_only_contract(),
        "initialization": _initialization_contract(),
        "model_contract": _model_contract(),
        "physical_batching": _physical_batching_contract(),
        "retained_arrays": _retained_arrays(),
        "retained_byte_contract": _retained_byte_contract(),
        "runtime_and_publication": _runtime_contract(),
        "science": _science_contract(),
        **_identity_contract(),
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
    observed = dict(value)
    contract = observed.pop("contract_sha256", None)
    if contract != canonical_sha256(observed):
        raise CompileError("manifest contract hash mismatch")
    expected_keys = {
        "artifact_closure", "active_plan_contract", "branch_state_contract",
        "call_chunk_count", "call_chunks", "call_chunks_root_sha256", "call_index",
        "conditional_price", "contract_sha256", "dag", "fit_only", "initialization",
        "model_contract", "physical_batching", "retained_arrays", "retained_byte_contract",
        "runtime_and_publication", "science", *_identity_contract().keys(),
    }
    if set(value) != expected_keys:
        raise CompileError("manifest top-level fields changed")
    for key, expected in _identity_contract().items():
        if value.get(key) != expected:
            raise CompileError(f"manifest identity/status changed: {key}")
    static_sections = {
        "artifact_closure": _artifact_closure_contract(),
        "active_plan_contract": _active_plan_contract(),
        "branch_state_contract": _branch_contract(),
        "dag": _dag(),
        "fit_only": _fit_only_contract(),
        "initialization": _initialization_contract(),
        "model_contract": _model_contract(),
        "physical_batching": _physical_batching_contract(),
        "retained_arrays": _retained_arrays(),
        "retained_byte_contract": _retained_byte_contract(),
        "runtime_and_publication": _runtime_contract(),
        "science": _science_contract(),
    }
    for key, expected in static_sections.items():
        if value.get(key) != expected:
            raise CompileError(f"canonical static contract changed: {key}")
    chunks = value.get("call_chunks")
    if not isinstance(chunks, list) or value.get("call_chunk_count") != len(chunks):
        raise CompileError("chunk census mismatch")
    if value.get("call_chunks_root_sha256") != canonical_sha256(chunks):
        raise CompileError("chunk root mismatch")
    if len(chunks) != CANONICAL_CALL_CHUNK_COUNT \
            or value.get("call_chunks_root_sha256") != CANONICAL_CALL_CHUNKS_ROOT_SHA256:
        raise CompileError("conditional chunk set differs from the frozen canonical census/root")
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
    if value.get("conditional_price") != _price_contract(chunks):
        raise CompileError("conditional price differs from supplied call chunks")
    index = value.get("call_index")
    if not isinstance(index, dict) \
            or index.get("encoding") != "ordered_raw_32_byte_SHA256_call_ids" \
            or int(index.get("byte_count", -1)) != 32 * int(index.get("call_count", -1)) \
            or int(index.get("call_count", -1)) != sum(int(chunk["call_count"]) for chunk in chunks):
        raise CompileError("per-call index contract changed")
    if int(index["call_count"]) != CANONICAL_CALL_COUNT \
            or index.get("sha256") != CANONICAL_CALL_INDEX_SHA256:
        raise CompileError("per-call index differs from the frozen canonical count/hash")
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


@dataclass(frozen=True)
class BranchState:
    """Typed, outcome-free inputs to the frozen terminal decision tree."""

    operational_fault: bool = False
    eligible_h_count: int | None = None
    eligible_q_count: int | None = None
    fit_health_stage: str | None = None
    higher_rank_rescue: bool | None = None
    semantic_gates_pass: bool | None = None
    single_necessity_pass: bool | None = None
    redundancy_available: bool | None = None
    redundancy_pass: bool | None = None
    h_before_q: bool | None = None
    reader_pass: bool | None = None


@dataclass(frozen=True)
class ActivePlanState:
    """Typed scientific selections from which every conditional chunk is derived."""

    preflight_pass: bool
    native_cache_complete: bool
    gradient_cache_complete: bool
    retained_h: tuple[int, ...]
    retained_q: tuple[int, ...]
    selected_h: int | None
    selected_q: int | None
    top_two_q: tuple[int, int] | None
    fit_health_stage: str | None
    semantic_and_falsifier_gates_pass: bool | None
    single_necessity_pass: bool | None
    redundancy_pass: bool | None


_BRANCH_FIELDS = tuple(BranchState.__dataclass_fields__)
_DAG_NODES = tuple(node["node"] for node in _dag())


def _terminal_id(disposition: str, terminal: str | None) -> str:
    return canonical_sha256({
        "branch_schema": "task14_fit_localization_v2_branch_state_v1",
        "disposition": disposition,
        "terminal": terminal,
    })


def _node_statuses(
    *, state: BranchState, disposition: str, terminal: str | None, route: str | None,
) -> dict[str, str]:
    statuses = {node: "skipped" for node in _DAG_NODES}
    if disposition == "operational_abort":
        statuses["preflight"] = "failed"
        statuses["terminal_projection"] = "completed_without_scientific_terminal"
        return statuses
    for node in ("preflight", "native_cache", "discovery_gradients", "discovery_full_ceilings"):
        statuses[node] = "completed"
    if terminal == "no_intervention_ceiling":
        statuses["terminal_projection"] = "completed"
        return statuses
    for node in (
        "spectral_operator", "joint_rank1_fits", "spectral_finite_diagnostic",
        "discovery_selection", "selected_family_and_rank_fits",
    ):
        statuses[node] = "completed"
    if terminal == "instrument_invalid":
        if state.fit_health_stage == "joint_rank1_failed":
            for node in (
                "spectral_finite_diagnostic", "discovery_selection",
                "selected_family_and_rank_fits",
            ):
                statuses[node] = "skipped"
            statuses["joint_rank1_fits"] = "failed_health"
        else:
            statuses["selected_family_and_rank_fits"] = "failed_health"
        statuses["terminal_projection"] = "completed"
        return statuses
    statuses["locked_validation"] = "completed"
    if terminal in {
        "fit_binary_state_rejected_higher_rank_needed_or_better",
        "fit_rank1_complete_subject_state_not_identified",
    }:
        statuses["terminal_projection"] = "completed"
        return statuses
    statuses["single_necessity"] = "completed"
    if route == "redundant":
        statuses["two_site_redundancy"] = "completed"
    elif state.single_necessity_pass is False and state.redundancy_available:
        statuses["two_site_redundancy"] = "failed"
    if route is not None and state.h_before_q:
        statuses["ordered_reader"] = "completed"
    statuses["terminal_projection"] = "completed"
    return statuses


def _require_none(state: BranchState, fields: Sequence[str], reason: str) -> None:
    changed = [field for field in fields if getattr(state, field) is not None]
    if changed:
        raise CompileError(f"{reason}; fields must be explicit skipped/null: {changed}")


def project_terminal(state: BranchState | Mapping[str, Any]) -> dict[str, Any]:
    """Pure executable projection from a complete branch state to one disposition."""
    if isinstance(state, Mapping):
        unknown = set(state) - set(_BRANCH_FIELDS)
        if unknown:
            raise CompileError(f"unknown branch-state fields: {sorted(unknown)}")
        state = BranchState(**dict(state))
    if not isinstance(state, BranchState):
        raise CompileError("branch state must be BranchState or exact mapping")

    terminal: str | None
    route: str | None = None
    if state.operational_fault:
        _require_none(state, _BRANCH_FIELDS[1:], "operational abort has no scientific state")
        disposition, terminal = "operational_abort", None
    else:
        if not isinstance(state.eligible_h_count, int) or not isinstance(state.eligible_q_count, int):
            raise CompileError("eligible H/Q counts are required integers")
        if state.eligible_h_count < 0 or state.eligible_q_count < 0:
            raise CompileError("eligible H/Q counts cannot be negative")
        if state.eligible_h_count == 0 or state.eligible_q_count == 0:
            _require_none(
                state,
                ("fit_health_stage", "higher_rank_rescue",
                 "semantic_gates_pass", "single_necessity_pass", "redundancy_available",
                 "redundancy_pass", "h_before_q", "reader_pass"),
                "empty H/Q branch ends before fitting",
            )
            disposition, terminal = "scientific_terminal", "no_intervention_ceiling"
        else:
            if state.fit_health_stage in {"joint_rank1_failed", "selected_family_rank_failed"}:
                _require_none(
                    state,
                    ("higher_rank_rescue", "semantic_gates_pass", "single_necessity_pass",
                     "redundancy_available", "redundancy_pass", "h_before_q", "reader_pass"),
                    "completed finite optimizer-health failure precedes scientific branches",
                )
                disposition, terminal = "scientific_terminal", "instrument_invalid"
            else:
                if state.fit_health_stage != "selected_family_rank_passed":
                    raise CompileError(
                        "branch projection requires an exact completed fit-health stage; "
                        "incomplete schedules are operational faults"
                    )
                if state.higher_rank_rescue is None:
                    raise CompileError("higher-rank falsifier decision is required")
                if state.higher_rank_rescue:
                    _require_none(
                        state,
                        ("semantic_gates_pass", "single_necessity_pass", "redundancy_available",
                         "redundancy_pass", "h_before_q", "reader_pass"),
                        "higher-rank rescue ends rank1 claim",
                    )
                    disposition, terminal = (
                        "scientific_terminal",
                        "fit_binary_state_rejected_higher_rank_needed_or_better",
                    )
                else:
                    if state.semantic_gates_pass is None:
                        raise CompileError("semantic gate decision is required")
                    if not state.semantic_gates_pass:
                        _require_none(
                            state,
                            ("single_necessity_pass", "redundancy_available", "redundancy_pass",
                             "h_before_q", "reader_pass"),
                            "semantic failure precedes necessity",
                        )
                        disposition, terminal = (
                            "scientific_terminal", "fit_rank1_complete_subject_state_not_identified",
                        )
                    else:
                        if state.single_necessity_pass is None:
                            raise CompileError("single-site necessity decision is required")
                        if state.single_necessity_pass:
                            _require_none(
                                state, ("redundancy_available", "redundancy_pass"),
                                "single-site success skips the redundancy branch",
                            )
                            route = "single"
                        else:
                            if state.redundancy_available is None:
                                raise CompileError("redundancy availability is required after singleton failure")
                            if not state.redundancy_available:
                                _require_none(
                                    state, ("redundancy_pass", "h_before_q", "reader_pass"),
                                    "unavailable redundancy ends necessity path",
                                )
                                disposition, terminal = (
                                    "scientific_terminal", "fit_rank1_state_sufficiency_only",
                                )
                                route = None
                            else:
                                if state.redundancy_pass is None:
                                    raise CompileError("redundancy decision is required when available")
                                if not state.redundancy_pass:
                                    _require_none(
                                        state, ("h_before_q", "reader_pass"),
                                        "failed redundancy ends necessity path",
                                    )
                                    disposition, terminal = (
                                        "scientific_terminal", "fit_rank1_state_sufficiency_only",
                                    )
                                    route = None
                                else:
                                    route = "redundant"
                        if route is not None:
                            if state.h_before_q is None:
                                raise CompileError("H<Q ordering decision is required after necessity succeeds")
                            if not state.h_before_q:
                                _require_none(state, ("reader_pass",), "unordered H/Q skips reader test")
                                terminal = (
                                    "fit_rank1_state_supported_reader_unresolved"
                                    if route == "single" else
                                    "fit_rank1_two_site_redundant_state_reader_unresolved"
                                )
                            else:
                                if state.reader_pass is None:
                                    raise CompileError("reader decision is required for ordered H<Q sites")
                                if state.reader_pass:
                                    terminal = (
                                        "fit_rank1_state_and_ordered_reader_supported"
                                        if route == "single" else
                                        "fit_rank1_redundant_state_and_ordered_reader_supported"
                                    )
                                else:
                                    terminal = (
                                        "fit_rank1_state_supported_reader_unresolved"
                                        if route == "single" else
                                        "fit_rank1_two_site_redundant_state_reader_unresolved"
                                    )
                            disposition = "scientific_terminal"

    if disposition == "scientific_terminal" and terminal not in TERMINALS:
        raise CompileError("terminal projector did not produce exactly one registered terminal")
    projection = {
        "branch_state": asdict(state),
        "disposition": disposition,
        "package_allowed": disposition == "scientific_terminal",
        "scientific_terminal": terminal,
        "terminal_id": _terminal_id(disposition, terminal),
        "node_statuses": _node_statuses(
            state=state, disposition=disposition, terminal=terminal, route=route,
        ),
    }
    projection["projection_sha256"] = canonical_sha256(projection)
    return projection


def deadline_check(
    clock: Any, *, start: float, limit_seconds: float, reviewed_p99_seconds: float,
    phase: str,
) -> float:
    """Fail before a call whose reviewed p99 cannot fit in the remaining budget."""
    now = float(clock())
    fields = (now, float(start), float(limit_seconds), float(reviewed_p99_seconds))
    if not all(math.isfinite(item) for item in fields) or now < start \
            or limit_seconds <= 0 or reviewed_p99_seconds < 0:
        raise OperationalAbort(f"invalid monotonic deadline state before {phase}")
    if now - start + reviewed_p99_seconds > limit_seconds:
        raise OperationalAbort(f"deadline would be exceeded by {phase}")
    return now


def deadline_check_after(clock: Any, *, start: float, limit_seconds: float, phase: str) -> float:
    now = float(clock())
    if not math.isfinite(now) or now < start or now - start > limit_seconds:
        raise OperationalAbort(f"deadline exceeded after {phase}")
    return now


def guarded_terminal_projection(
    state: BranchState, *, action: Any, clock: Any, start: float,
    limit_seconds: float, reviewed_p99_seconds: float,
) -> dict[str, Any]:
    """Testable guard wrapper: deadline faults project only to operational abort."""
    try:
        deadline_check(
            clock, start=start, limit_seconds=limit_seconds,
            reviewed_p99_seconds=reviewed_p99_seconds, phase="guarded_action",
        )
        action()
        deadline_check_after(
            clock, start=start, limit_seconds=limit_seconds, phase="guarded_action",
        )
    except OperationalAbort:
        return project_terminal(BranchState(operational_fault=True))
    return project_terminal(state)


def preflight_namespace_absent(paths: Sequence[Path]) -> dict[str, Any]:
    """Use lstat so files, directories, and dangling symlinks all block publication."""
    checked: list[str] = []
    for path in paths:
        absolute = path.resolve(strict=False)
        try:
            os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            raise OperationalAbort(f"reserved namespace already exists: {path}")
        checked.append(str(absolute))
    receipt = {"create_exclusive": True, "paths": checked, "status": "all_absent"}
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def apply_boundary_edit(
    *, boundary: int, target_residual: Sequence[float], donor_residual: Sequence[float],
    coordinate: int, target_x0: Sequence[float] | None = None,
    target_v1: Sequence[float] | None = None, derive_v1: Any = None,
) -> dict[str, tuple[float, ...]]:
    """Pure scalar-coordinate tripwire for the frozen x0/v1 intervention rules."""
    if boundary not in BOUNDARIES or not 0 <= coordinate < len(target_residual) \
            or len(target_residual) != len(donor_residual):
        raise CompileError("invalid synthetic boundary edit")
    edited = list(float(item) for item in target_residual)
    edited[coordinate] = float(donor_residual[coordinate])
    if boundary == -1:
        if target_x0 is not None or target_v1 is not None or derive_v1 is None:
            raise CompileError("boundary -1 must establish edited x0 and live-derived v1")
        x0 = tuple(edited)
        v1 = tuple(float(item) for item in derive_v1(x0))
    else:
        if target_x0 is None or target_v1 is None or derive_v1 is not None:
            raise CompileError("boundary >=0 must preserve target-prefix x0/v1")
        x0 = tuple(float(item) for item in target_x0)
        v1 = tuple(float(item) for item in target_v1)
    return {"residual": tuple(edited), "x0": x0, "v1": v1}


def apply_composed_boundary_edits(
    *, first_boundary: int, second_boundary: int,
    initial_state: Mapping[str, Sequence[float] | None],
    first_donor: Sequence[float], second_donor: Sequence[float],
    coordinate: int, derive_v1: Any, propagate_to_second: Any,
) -> dict[str, Any]:
    """Tripwire ensuring a later intervention consumes the earlier edited trajectory."""
    if not first_boundary < second_boundary:
        raise CompileError("composed sites must be strictly ascending")
    first = apply_boundary_edit(
        boundary=first_boundary,
        target_residual=initial_state["residual"],  # type: ignore[arg-type]
        donor_residual=first_donor,
        coordinate=coordinate,
        target_x0=initial_state.get("x0"),  # type: ignore[arg-type]
        target_v1=initial_state.get("v1"),  # type: ignore[arg-type]
        derive_v1=derive_v1 if first_boundary == -1 else None,
    )
    live = propagate_to_second(first)
    if not isinstance(live, Mapping):
        raise CompileError("composed-site propagation must return a state mapping")
    second = apply_boundary_edit(
        boundary=second_boundary,
        target_residual=live["residual"],
        donor_residual=second_donor,
        coordinate=coordinate,
        target_x0=live["x0"],
        target_v1=live["v1"],
    )
    return {"first": first, "live_at_second": dict(live), "second": second}


def _raw_call_ids(raw: bytes, offset: int, count: int) -> list[str]:
    start, stop = 32 * offset, 32 * (offset + count)
    return [raw[index:index + 32].hex() for index in range(start, stop, 32)]


def _validate_active_plan_state(state: ActivePlanState) -> None:
    if state.native_cache_complete and not state.preflight_pass:
        raise CompileError("native cache cannot complete before preflight")
    if state.gradient_cache_complete and not state.native_cache_complete:
        raise CompileError("gradient cache cannot complete before native cache")
    if not state.gradient_cache_complete and (
        state.retained_h or state.retained_q or state.selected_h is not None
        or state.fit_health_stage is not None
    ):
        raise CompileError("scientific selections require a complete gradient cache")
    for name, sites, maximum in (
        ("retained_h", state.retained_h, 3), ("retained_q", state.retained_q, 19),
    ):
        if len(sites) != len(set(sites)) or len(sites) > maximum \
                or any(site not in BOUNDARIES for site in sites):
            raise CompileError(f"invalid {name} sites")
    if tuple(sorted(state.retained_h)) != state.retained_h \
            or tuple(sorted(state.retained_q)) != state.retained_q:
        raise CompileError("retained sites must be sorted in boundary order")
    if state.selected_h is not None and state.selected_h not in state.retained_h:
        raise CompileError("selected H is not retained")
    if state.selected_q is not None and state.selected_q not in state.retained_q:
        raise CompileError("selected Q is not retained")
    if (state.selected_h is None) != (state.selected_q is None):
        raise CompileError("H and Q selection must be jointly present or absent")
    allowed_health = {
        None, "joint_rank1_failed", "selected_family_rank_failed", "selected_family_rank_passed",
    }
    if state.fit_health_stage not in allowed_health:
        raise CompileError("unknown fit-health stage")
    if (state.retained_h and state.retained_q) and state.fit_health_stage is None:
        raise CompileError("retained H/Q sites require a completed fit-health stage")
    if state.fit_health_stage == "joint_rank1_failed" and state.selected_h is not None:
        raise CompileError("joint-rank1 health failure precedes site selection")
    if state.fit_health_stage in {"selected_family_rank_failed", "selected_family_rank_passed"} \
            and state.selected_h is None:
        raise CompileError("selected-family/rank health stage requires selected H/Q")
    if state.fit_health_stage != "selected_family_rank_passed" \
            and state.semantic_and_falsifier_gates_pass is not None:
        raise CompileError("semantic gates are unavailable before healthy selected-fit completion")
    if state.top_two_q is not None:
        if len(set(state.top_two_q)) != 2 \
                or any(site not in state.retained_q for site in state.top_two_q):
            raise CompileError("top-two Q pair is invalid")
    if state.semantic_and_falsifier_gates_pass and state.selected_q is None:
        raise CompileError("semantic pass requires selected H/Q")
    if state.semantic_and_falsifier_gates_pass \
            and state.fit_health_stage != "selected_family_rank_passed":
        raise CompileError("semantic gates require healthy completion of all selected fits")
    if state.semantic_and_falsifier_gates_pass is not True:
        if state.single_necessity_pass is not None or state.redundancy_pass is not None:
            raise CompileError("absent/failed semantic gate skips necessity branches")
    elif state.single_necessity_pass is None:
        raise CompileError("semantic pass requires singleton necessity decision")
    if state.single_necessity_pass:
        if state.redundancy_pass is not None:
            raise CompileError("single necessity success skips redundancy")
    elif state.single_necessity_pass is False:
        if state.top_two_q is None and state.redundancy_pass is not None:
            raise CompileError("redundancy cannot execute without a top-two Q pair")
        if state.top_two_q is not None and state.redundancy_pass is None:
            raise CompileError("available redundancy requires an evaluated result")


def evaluate_chunk_guard(chunk: Mapping[str, Any], state: ActivePlanState) -> bool:
    """Evaluate a chunk from its canonical ID; never trust a free-text guard."""
    _validate_active_plan_state(state)
    has_both_positions = bool(state.retained_h and state.retained_q)
    chunk_id = str(chunk["chunk_id"])
    parts = chunk_id.split(":")
    if chunk_id == "00_native_cache":
        expected_guard, active = "preflight_pass", state.preflight_pass
    elif chunk_id == "01_discovery_gradients":
        expected_guard, active = "native_cache_complete", state.native_cache_complete
    elif parts[0] == "ceiling" and parts[1] == "DISCOVERY":
        expected_guard, active = "gradient_cache_complete", state.gradient_cache_complete
    elif parts[0] == "fit":
        position, boundary, objective, rank = parts[1], int(parts[2]), parts[3], int(parts[4][4:])
        if objective == "joint" and rank == 1:
            if position == "H":
                expected_guard = "site_in_discovery_top3_H"
                active = has_both_positions and boundary in state.retained_h
            else:
                expected_guard = "site_discovery_eligible_Q"
                active = has_both_positions and boundary in state.retained_q
        else:
            expected_guard = f"site_is_selected_{position}"
            active = boundary == (state.selected_h if position == "H" else state.selected_q)
    elif parts[0] == "spectral":
        position, boundary = parts[1], int(parts[2])
        expected_guard = "site_is_retained_for_joint_rank1;diagnostic_only_never_selects"
        active = (
            has_both_positions and state.fit_health_stage != "joint_rank1_failed"
            and boundary in (state.retained_h if position == "H" else state.retained_q)
        )
    elif parts[0] == "ceiling" and parts[1] == "VALIDATION":
        position, boundary = parts[2], int(parts[3])
        expected_guard = "site_is_selected_H_or_Q"
        active = (
            state.fit_health_stage == "selected_family_rank_passed"
            and boundary == (state.selected_h if position == "H" else state.selected_q)
        )
    elif parts[0] == "eval":
        position, boundary = parts[2], int(parts[3])
        expected_guard = "fit_is_selected_H_or_Q_configuration"
        active = (
            state.fit_health_stage == "selected_family_rank_passed"
            and boundary == (state.selected_h if position == "H" else state.selected_q)
        )
    elif parts[0] == "necessity":
        boundary = int(parts[2])
        expected_guard = "site_is_selected_Q_and_rank1_semantic_and_falsifier_gates_pass"
        active = boundary == state.selected_q and state.semantic_and_falsifier_gates_pass
    elif parts[0] == "redundancy":
        pair = (int(parts[2]), int(parts[3]))
        expected_guard = "pair_is_top_two_Q_and_selected_Q_single_necessity_fails"
        active = (
            state.semantic_and_falsifier_gates_pass
            and state.single_necessity_pass is False
            and state.top_two_q is not None
            and set(pair) == set(state.top_two_q)
        )
    elif parts[0] == "reader":
        h_boundary, q_boundary = int(parts[2]), int(parts[4])
        expected_guard = "sites_are_selected_and_ordered_and_semantics_pass_and_necessity_route_passes"
        route_pass = state.single_necessity_pass is True or state.redundancy_pass is True
        active = (
            state.semantic_and_falsifier_gates_pass and route_pass
            and h_boundary == state.selected_h and q_boundary == state.selected_q
            and h_boundary < q_boundary
        )
    else:
        raise CompileError(f"unrecognized canonical chunk id: {chunk_id}")
    if chunk.get("activation") != expected_guard:
        raise CompileError(f"chunk activation guard text changed: {chunk_id}")
    return bool(active)


def derive_active_chunk_ids(value: Mapping[str, Any], state: ActivePlanState) -> list[str]:
    return [
        str(chunk["chunk_id"])
        for chunk in value["call_chunks"]
        if evaluate_chunk_guard(chunk, state)
    ]


def validate_execution_compatibility(
    active_state: ActivePlanState, branch_state: BranchState,
) -> dict[str, Any]:
    """Bind the activated physical path to the separately projected terminal path."""
    _validate_active_plan_state(active_state)
    projection = project_terminal(branch_state)
    if branch_state.operational_fault:
        if active_state.preflight_pass or active_state.native_cache_complete \
                or active_state.gradient_cache_complete or active_state.retained_h \
                or active_state.retained_q or active_state.selected_h is not None:
            raise CompileError("operational abort cannot claim an activated scientific path")
        return projection
    if not (
        active_state.preflight_pass and active_state.native_cache_complete
        and active_state.gradient_cache_complete
    ):
        raise CompileError("a scientific terminal requires complete preflight/native/gradient prefix")
    if branch_state.eligible_h_count != len(active_state.retained_h) \
            or branch_state.eligible_q_count != len(active_state.retained_q):
        raise CompileError("terminal H/Q census differs from active physical path")
    if branch_state.fit_health_stage != active_state.fit_health_stage:
        raise CompileError("terminal fit-health stage differs from active physical path")
    if branch_state.semantic_gates_pass is not active_state.semantic_and_falsifier_gates_pass:
        raise CompileError("terminal semantic decision differs from active physical path")
    if branch_state.single_necessity_pass is not active_state.single_necessity_pass:
        raise CompileError("terminal singleton decision differs from active physical path")
    if branch_state.redundancy_pass is not active_state.redundancy_pass:
        raise CompileError("terminal redundancy decision differs from active physical path")
    if branch_state.redundancy_available is not None:
        if branch_state.redundancy_available != (active_state.top_two_q is not None):
            raise CompileError("terminal redundancy availability differs from active physical path")
    if branch_state.h_before_q is not None:
        ordered = (
            active_state.selected_h is not None and active_state.selected_q is not None
            and active_state.selected_h < active_state.selected_q
        )
        if branch_state.h_before_q != ordered:
            raise CompileError("terminal H<Q decision differs from selected physical sites")
    active_ids_for_contract = {
        "terminal": projection["scientific_terminal"],
        "active_state_sha256": canonical_sha256(asdict(active_state)),
        "branch_state_sha256": canonical_sha256(asdict(branch_state)),
    }
    active_ids_for_contract["compatibility_sha256"] = canonical_sha256(active_ids_for_contract)
    return {**projection, **active_ids_for_contract}


def preflight_global_call_index(value: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    """Regenerate and compare every descriptor before any future model access."""
    validate_call_index(value, raw)
    expected_chunks = iter(value["call_chunks"])
    current = next(expected_chunks, None)
    position = 0
    observed_count = 0
    for chunk_id, call in iter_call_descriptors():
        if current is None or chunk_id != current["chunk_id"]:
            raise CompileError("global descriptor traversal differs from materialized chunk order")
        expected = raw[32 * (position + observed_count):32 * (position + observed_count + 1)].hex()
        if call.get("call_id") != expected:
            raise CompileError(f"global descriptor mismatch: {chunk_id}:{observed_count}")
        observed_count += 1
        if observed_count == int(current["call_index_count"]):
            position += observed_count
            observed_count = 0
            current = next(expected_chunks, None)
    if current is not None or observed_count != 0 or position != int(value["call_index"]["call_count"]):
        raise CompileError("global descriptor replay ended at the wrong index position")
    receipt = {
        "call_count": position,
        "call_index_sha256": bytes_sha256(raw),
        "status": "whole_global_index_verified_before_model",
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def active_path_receipts(
    value: Mapping[str, Any], raw: bytes, state: ActivePlanState,
    branch_state: BranchState,
) -> dict[str, Any]:
    """Verify each active slice and emit an explicit receipt for every inactive chunk."""
    validate_call_index(value, raw)
    compatibility = validate_execution_compatibility(state, branch_state)
    active = derive_active_chunk_ids(value, state)
    canonical_order = [str(chunk["chunk_id"]) for chunk in value["call_chunks"]]
    receipts: list[dict[str, Any]] = []
    active_set = set(active)
    for chunk in value["call_chunks"]:
        chunk_id = str(chunk["chunk_id"])
        is_active = chunk_id in active_set
        offset, count = int(chunk["call_index_offset"]), int(chunk["call_index_count"])
        ids = _raw_call_ids(raw, offset, count)
        root = _chain_seed(chunk_id)
        for call_id in ids:
            root = _chain_step(root, call_id)
        if root != chunk["call_root_sha256"]:
            raise CompileError(f"active-path slice replay failed: {chunk_id}")
        receipt = {
            "activation_guard": chunk["activation"],
            "call_count": count,
            "call_index_offset": offset,
            "call_index_slice_sha256": chunk["call_index_slice_sha256"],
            "chunk_id": chunk_id,
            "guard_value": is_active,
            "status": "active_slice_verified" if is_active else "inactive_skip_zero_calls",
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        receipts.append(receipt)
    result = {
        "active_chunk_ids": active,
        "active_plan_state": asdict(state),
        "branch_projection": compatibility,
        "active_count": len(active),
        "inactive_count": len(receipts) - len(active),
        "receipts": receipts,
    }
    result["active_path_root_sha256"] = canonical_sha256(receipts)
    result["execution_path_root_sha256"] = canonical_sha256({
        "active_path_root_sha256": result["active_path_root_sha256"],
        "compatibility_sha256": compatibility["compatibility_sha256"],
    })
    return result


def replay_active_path(
    value: Mapping[str, Any], raw: bytes, state: ActivePlanState,
    branch_state: BranchState, visitor: Any,
) -> dict[str, Any]:
    """Second canonical traversal: verify active slices and immediately expose exact calls.

    A future producer must call ``preflight_global_call_index`` first, before model
    load. It then uses this traversal—not a caller-authored call list—to execute
    each active descriptor. Inactive descriptors are regenerated but never passed
    to ``visitor`` and receive explicit skip receipts.
    """
    plan = active_path_receipts(value, raw, state, branch_state)
    active = set(plan["active_chunk_ids"])
    chunk_iterator = iter(value["call_chunks"])
    current = next(chunk_iterator, None)
    local_index = 0
    executed = 0
    for chunk_id, call in iter_call_descriptors():
        if current is None or chunk_id != current["chunk_id"]:
            raise CompileError("active replay chunk order differs from canonical manifest")
        if chunk_id in active:
            absolute = int(current["call_index_offset"]) + local_index
            expected = raw[32 * absolute:32 * (absolute + 1)].hex()
            if call.get("call_id") != expected:
                raise CompileError(f"active descriptor differs at {chunk_id}:{local_index}")
            visitor(chunk_id, dict(call))
            executed += 1
        local_index += 1
        if local_index == int(current["call_index_count"]):
            current = next(chunk_iterator, None)
            local_index = 0
    if current is not None or local_index:
        raise CompileError("active descriptor traversal ended before canonical completion")
    expected_executed = sum(
        int(chunk["call_count"]) for chunk in value["call_chunks"]
        if str(chunk["chunk_id"]) in active
    )
    if executed != expected_executed:
        raise CompileError("active descriptor execution census changed")
    result = {**plan, "executed_descriptor_count": executed}
    result["active_replay_receipt_sha256"] = canonical_sha256({
        "execution_path_root_sha256": plan["execution_path_root_sha256"],
        "executed_descriptor_count": executed,
    })
    return result


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
        "schema": "task14_fit_localization_v2_compiler_v2_dryrun_v1",
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
