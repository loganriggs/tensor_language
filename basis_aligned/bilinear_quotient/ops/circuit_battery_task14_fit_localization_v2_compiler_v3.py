#!/usr/bin/env python3
# BQLANE: cpu
"""Compile the stagewise v3 exact physical plan for task14 FIT localization v2.

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


SCHEMA = "task14_fit_localization_v2_physical_compiler_v3"
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
CANONICAL_CALL_CHUNKS_ROOT_SHA256 = "136adcb09391a50a686645adb47671c18033042ab49a3747f7fc9ef0004cd54b"
CANONICAL_CALL_COUNT = 743_881
CANONICAL_CALL_INDEX_SHA256 = "f51dce007b549f5f51000c831cea67eca301cd9bbe10b1fb4799196a738dd9a8"
CANONICAL_CALL_SHAPE_COUNT = 5_271
CANONICAL_CALL_SHAPES_ROOT_SHA256 = "0f3f373ed6c9c992a6637effe946974fd6c368d3aeb5814c38367c3c202bffb6"
CANONICAL_SHAPE_MULTIPLICITY_ROOT_SHA256 = "UNFROZEN"
CANONICAL_STAGE_CALL_CONTRACT_ROOT_SHA256 = "UNFROZEN"

V2_COMMIT = "8f41f51cdf7e073063201cc48760622607ce91b9"
V2_REVIEW_COMMIT = "2ffd6cf77998a6c7fb6af0c4e89c742bf1bbb923"
V2_REVIEW_SHA256 = "2905aeb040fad2d16062a22e3c4d32d9dd6953c468724ff51a80ab9fa849d384"

REPO_ROOT = Path(__file__).resolve().parents[3]
OPS = Path(__file__).resolve().parent
MANIFEST_PATH = OPS / "circuit_battery_task14_fit_localization_v2_call_manifest_v3.json"
CALL_INDEX_PATH = OPS / "circuit_battery_task14_fit_localization_v2_call_index_v3.bin"
DRYRUN_PATH = OPS / "circuit_battery_task14_fit_localization_v2_compiler_v3_dryrun.json"

BLOCKED_COMPILER_COMMIT = "6b7fb09ff30080e73cad0414d8315db660e04ca0"
BLOCK_REVIEW_COMMIT = "60892e3994250b7f58330f4b2a84f8ed4126c928"
BLOCK_REVIEW_SHA256 = "3131fffd0b6c8cd18789b69e4909b0002ca3e90f2c965391c07444f56b63756a"

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
        "basis_aligned/polynomial_causal/TASK14_FIT_LOCALIZATION_V2_PHYSICAL_COMPILER_V2_INDEPENDENT_REVIEW_2026-09-04.md",
        BLOCK_REVIEW_SHA256,
    ),
    "initial_compiler_block_review": (
        "basis_aligned/polynomial_causal/TASK14_FIT_LOCALIZATION_V2_PHYSICAL_COMPILER_INDEPENDENT_REVIEW_2026-09-04.md",
        "673389c02ec4d7e9122557fe4fb44ab9f90950ccf8e6efbbd310ac6d543548b1",
    ),
    "v3_preregistration": (
        "basis_aligned/polynomial_causal/TASK14_SUBJECT_VERB_AGREEMENT_FIT_LOCALIZATION_V2_PHYSICAL_COMPILER_PREREGISTRATION_V3_2026-09-04.md",
        "280b0a60272c5395285b47d4531e09c4eae6ab4b1eca673d05bcfe22bcec2209",
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
    "mathematical_review_1230": (
        "basis_aligned/polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-04_1230.md",
        "92afa14d63285be9c36d0773cc09f65bf67de8503389fea135c241dd236d742b",
    ),
    "v3_prefreeze_checklist": (
        "basis_aligned/polynomial_causal/TASK14_FIT_LOCALIZATION_V2_COMPILER_V3_PREFREEZE_CHECKLIST_2026-09-04.md",
        "b5e0c92e650caac2c36ba85a227a6fe14eaa7611f411ebca378e74945c316930",
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
    """The requested plan differs from the exact v3 FIT compiler contract."""


class OperationalAbort(RuntimeError):
    """A preflight/runtime guard failed before a scientific terminal existed."""


@dataclass(frozen=True)
class TimingCapability:
    stage: str
    call_shape_sha256: str
    p99_seconds: float
    reviewed_receipt_sha256: str
    token_id: str
    _seal: object


@dataclass(frozen=True)
class DeadlineCapability:
    timing_token_ids: tuple[str, ...]
    deadline_id: str
    _seal: object


@dataclass(frozen=True)
class TimingAuthorization:
    timing_token_ids: tuple[str, ...]
    authorization_id: str
    _seal: object


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


def capture_input_bytes() -> tuple[bytes, bytes, bytes]:
    captured = {role: _load_frozen(role) for role in FROZEN}
    return captured["fit_authority"], captured["v2_partition"], captured["v2_donors"]


def parse_captured_inputs(
    captured: tuple[bytes, bytes, bytes],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if type(captured) is not tuple or len(captured) != 3 \
            or any(type(item) is not bytes for item in captured):
        raise CompileError("captured compiler inputs must be exact immutable bytes tuple")
    authority = strict_json(captured[0], "captured FIT authority")
    partition = strict_json(captured[1], "captured v2 partition")
    donors = strict_json(captured[2], "captured v2 donors")
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


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return parse_captured_inputs(capture_input_bytes())


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
    stages: set[str] = set()
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
        stages.add(str(call["stage"]))
    if len(stages) != 1:
        raise CompileError("chunk must contain exactly one physical call stage")
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
        "stage": next(iter(stages)),
        "token_evaluations": tokens,
    }


def _endpoint_items(endpoint_ids: Sequence[str]) -> list[tuple[str, Sequence[str]]]:
    return [(endpoint_id, ()) for endpoint_id in endpoint_ids]


def _record_items(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, Sequence[str]]]:
    return [(str(record["record_id"]), ()) for record in records]


def _native_chunks(
    authority: Mapping[str, Any], partition: Mapping[str, Any],
    endpoints: Mapping[str, Mapping[str, Any]], records_by_id: Mapping[str, Mapping[str, Any]],
    only: str | None = None,
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
                    stage="discovery_gradients", call_kind="discovery_gradient_full_forward",
                    branch=f"gradient:{side}:{family}", item_kind="endpoint", item_ids=discovery_ids,
                    position=None, boundary=None, endpoints=endpoints, records_by_id=records_by_id,
                    retained=False, participates_in_backward=True,
                ))
    if only in {None, "native_cache"}:
        chunks.append(_chunk("00_native_cache", "preflight_pass", native_calls))
    if only in {None, "discovery_gradients"}:
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
        stage=("discovery_full_ceilings" if partition_name == "DISCOVERY" else "validation_full_ceilings"),
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
    causal_stage = (
        "joint_rank1_fits"
        if objective == "joint" and rank == 1 else "selected_family_and_rank_fits"
    )
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
                stage=causal_stage, call_kind="projector_intervention_train_forward",
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
            stage=causal_stage, call_kind="final_discovery_projector_eval",
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
        stage="spectral_finite_diagnostic", call_kind="spectral_projector_intervention_forward",
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
        stage="locked_validation",
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
        stage="single_necessity", call_kind="necessity_neutralization_forward",
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
                stage="two_site_redundancy",
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
                stage="ordered_reader", call_kind="ordered_H_Q_reader_forward",
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
            "clock_start": "before_any_model_checkpoint_or_CUDA_access",
            "monotonic_check_before_and_after_every_model_call": True,
            "no_call_may_start_without_remaining_reviewed_p99_call_budget": True,
            "no_automatic_retry": True,
            "preauthorization_throughput_receipt_required_per_distinct_physical_call_shape": True,
            "receipt_digest_status": "not_yet_known_compiler_does_not_invent_it",
            "binding_authority": "future_producer_and_authorization_must_bind_exact_independently_reviewed_receipts",
            "compiler_start_deadline_status": "blocked_without_process_local_TimingAuthorization_no_issuer_in_v3",
            "future_timing_authorization_requirements": (
                "exact allowlisted receipt and independent-review digests, complete frozen physical-call-shape "
                "coverage, reviewed fixed bootstrap/publication bounds, and longest compatible path <=28800"
            ),
            "compatible_path_algorithm": "longest_compatible_stage_path reverse-topological recurrence with witness",
            "authorization_inequality": "bootstrap_seconds + longest_path_seconds + publication_seconds <= 28800",
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
            "typed_state": "one_distinct_frozen_dataclass_per_completed_stage",
            "pure_projector": "project_stagewise_terminal",
            "operational_fault_has_no_scientific_terminal": True,
            "all_inapplicable_nodes_explicitly_skipped": True,
            "single_and_redundancy_success_mutually_exclusive": True,
            "execution_blocker": (
                "this compiler validates typed/rule-consistent stage decisions but does not reconstruct them "
                "from primitive model evidence; a future producer/result validator must do so exactly"
            ),
        },
        "call_index_replay": {
            "before_model": (
                "preflight_global_call_index verifies all descriptors and all 32-byte entries, captures all frozen "
                "input bytes/parsed objects once, and issues an unforgeable process-local GlobalPreflightToken"
            ),
            "source_reopen_after_preflight": False,
            "causal_stage": "only the next stage named by a predecessor-bound process-local StageCapability",
            "active_chunk": "offset/count slice and root replay immediately before calls from captured inputs",
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


def _dag(
    stage_call_contract: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    calls_by_stage = {
        str(item["stage"]): dict(item) for item in (stage_call_contract or ())
    }
    specs = (
        ("preflight", (), ("preflight.token",), (), "always", ("native_cache",)),
        ("native_cache", ("preflight.token",), (
            "native_cache.complete", "native_cache.evidence_sha256",
        ), (), "always", ("discovery_gradients",)),
        ("discovery_gradients", ("native_cache.complete", "native_cache.evidence_sha256"), (
            "discovery_gradients.complete", "discovery_gradients.all_values_finite",
            "discovery_gradients.gradient_denominators_above_threshold",
            "discovery_gradients.invalid_gradient_denominator_count",
            "discovery_gradients.evidence_sha256",
        ), (), "always", ("discovery_full_ceilings", "terminal_projection")),
        ("discovery_full_ceilings", (
            "native_cache.evidence_sha256", "discovery_gradients.complete",
            "discovery_gradients.gradient_denominators_above_threshold",
            "discovery_gradients.evidence_sha256",
        ), (
            "discovery_full_ceilings.complete", "discovery_full_ceilings.all_values_finite",
            "discovery_full_ceilings.natural_margin_denominators_above_threshold",
            "discovery_full_ceilings.invalid_natural_margin_denominator_count",
            "discovery_full_ceilings.eligible_h_count", "discovery_full_ceilings.eligible_h_ranked_scores",
            "discovery_full_ceilings.retained_h", "discovery_full_ceilings.eligible_q_count",
            "discovery_full_ceilings.retained_q", "discovery_full_ceilings.top_three_h_evidence_sha256",
            "discovery_full_ceilings.evidence_sha256",
        ), (), "always", ("joint_rank1_fits", "terminal_projection")),
        ("joint_rank1_fits", (
            "native_cache.evidence_sha256", "discovery_full_ceilings.evidence_sha256",
            "discovery_full_ceilings.natural_margin_denominators_above_threshold",
            "discovery_full_ceilings.retained_h", "discovery_full_ceilings.retained_q",
        ), (
            "joint_rank1_fits.all_scheduled_calls_complete",
            "joint_rank1_fits.finite_optimizer_seed_health_ok", "joint_rank1_fits.evidence_sha256",
        ), ("discovery_full_ceilings.retained_h", "discovery_full_ceilings.retained_q"),
         "both_retained_nonempty", ("spectral_finite_diagnostic", "terminal_projection")),
        ("spectral_finite_diagnostic", (
            "discovery_full_ceilings.retained_h", "discovery_full_ceilings.retained_q",
            "joint_rank1_fits.finite_optimizer_seed_health_ok", "joint_rank1_fits.evidence_sha256",
        ), ("spectral_finite_diagnostic.diagnostic_complete", "spectral_finite_diagnostic.evidence_sha256"),
         ("joint_rank1_fits.finite_optimizer_seed_health_ok",), "joint_fit_health_pass", ("discovery_selection",)),
        ("discovery_selection", (
            "discovery_full_ceilings.retained_h", "discovery_full_ceilings.retained_q",
            "discovery_full_ceilings.evidence_sha256",
            "joint_rank1_fits.evidence_sha256", "joint_rank1_fits.finite_optimizer_seed_health_ok",
            "spectral_finite_diagnostic.diagnostic_complete",
            "spectral_finite_diagnostic.evidence_sha256",
        ), (
            "discovery_selection.selected_h", "discovery_selection.selected_q", "discovery_selection.top_two_q",
            "discovery_selection.h_objective_scores", "discovery_selection.q_t_scores",
            "discovery_selection.reader_selection_eligible", "discovery_selection.selection_evidence_sha256",
        ), ("joint_rank1_fits.finite_optimizer_seed_health_ok",), "joint_fit_health_pass",
         ("selected_family_and_rank_fits",)),
        ("selected_family_and_rank_fits", (
            "native_cache.evidence_sha256", "joint_rank1_fits.evidence_sha256",
            "discovery_selection.selected_h", "discovery_selection.selected_q",
            "discovery_selection.selection_evidence_sha256",
        ), (
            "selected_family_and_rank_fits.all_scheduled_calls_complete",
            "selected_family_and_rank_fits.finite_optimizer_seed_health_ok",
            "selected_family_and_rank_fits.evidence_sha256",
        ), ("discovery_selection.selected_h", "discovery_selection.selected_q"), "sites_selected",
         ("validation_full_ceilings", "terminal_projection")),
        ("validation_full_ceilings", (
            "native_cache.evidence_sha256", "discovery_selection.selected_h",
            "discovery_selection.selected_q", "discovery_selection.selection_evidence_sha256",
            "selected_family_and_rank_fits.finite_optimizer_seed_health_ok",
        ), ("validation_full_ceilings.complete", "validation_full_ceilings.evidence_sha256"),
         ("selected_family_and_rank_fits.finite_optimizer_seed_health_ok",), "selected_fit_health_pass",
         ("locked_validation",)),
        ("locked_validation", (
            "discovery_selection.selected_h", "discovery_selection.selected_q",
            "selected_family_and_rank_fits.evidence_sha256", "validation_full_ceilings.evidence_sha256",
            "selected_family_and_rank_fits.finite_optimizer_seed_health_ok",
        ), ("locked_validation.higher_rank_rescue", "locked_validation.semantic_gates_pass", "locked_validation.evidence_sha256"),
         ("selected_family_and_rank_fits.finite_optimizer_seed_health_ok",), "selected_fit_health_pass",
         ("single_necessity", "terminal_projection")),
        ("single_necessity", (
            "discovery_selection.selected_h", "discovery_selection.selected_q",
            "discovery_selection.top_two_q", "discovery_selection.reader_selection_eligible",
            "selected_family_and_rank_fits.evidence_sha256",
            "locked_validation.semantic_gates_pass", "locked_validation.evidence_sha256",
        ), ("single_necessity.single_necessity_pass", "single_necessity.evidence_sha256"),
         ("locked_validation.semantic_gates_pass",), "rank1_semantic_gates_pass",
         ("two_site_redundancy", "ordered_reader", "terminal_projection")),
        ("two_site_redundancy", (
            "discovery_selection.selected_h", "discovery_selection.selected_q",
            "discovery_selection.top_two_q", "discovery_selection.reader_selection_eligible",
            "single_necessity.single_necessity_pass", "single_necessity.evidence_sha256",
            "selected_family_and_rank_fits.evidence_sha256", "locked_validation.evidence_sha256",
        ), ("two_site_redundancy.redundancy_pass", "two_site_redundancy.evidence_sha256"),
         ("discovery_selection.top_two_q", "single_necessity.single_necessity_pass"),
         "single_failed_and_top_two_available", ("ordered_reader", "terminal_projection")),
        ("ordered_reader", (
            "discovery_selection.selected_h", "discovery_selection.selected_q",
            "discovery_selection.reader_selection_eligible", "single_necessity.single_necessity_pass",
            "two_site_redundancy.redundancy_pass", "single_necessity.evidence_sha256",
            "selected_family_and_rank_fits.evidence_sha256", "locked_validation.evidence_sha256",
        ), ("ordered_reader.reader_pass", "ordered_reader.evidence_sha256"),
         ("discovery_selection.reader_selection_eligible", "single_necessity.single_necessity_pass",
          "two_site_redundancy.redundancy_pass"), "necessity_route_pass_and_H_before_Q", ("terminal_projection",)),
        ("terminal_projection", (
            "discovery_gradients.gradient_denominators_above_threshold",
            "discovery_full_ceilings.eligible_h_count", "discovery_full_ceilings.eligible_q_count",
            "discovery_full_ceilings.natural_margin_denominators_above_threshold",
            "discovery_full_ceilings.retained_h", "discovery_full_ceilings.retained_q",
            "joint_rank1_fits.finite_optimizer_seed_health_ok",
            "selected_family_and_rank_fits.finite_optimizer_seed_health_ok",
            "locked_validation.higher_rank_rescue", "locked_validation.semantic_gates_pass",
            "single_necessity.single_necessity_pass", "two_site_redundancy.redundancy_pass",
            "ordered_reader.reader_pass",
        ), ("terminal_projection.scientific_terminal",), (), "first_registered_terminal_by_precedence", ()),
    )
    nodes: list[dict[str, Any]] = []
    for node, reads, writes, guard_reads, guard, successors in specs:
        optional_by_node = {
            "ordered_reader": {"two_site_redundancy.redundancy_pass"},
            "terminal_projection": {
                "discovery_full_ceilings.eligible_h_count", "discovery_full_ceilings.eligible_q_count",
                "discovery_full_ceilings.natural_margin_denominators_above_threshold",
                "discovery_full_ceilings.retained_h", "discovery_full_ceilings.retained_q",
                "joint_rank1_fits.finite_optimizer_seed_health_ok",
                "selected_family_and_rank_fits.finite_optimizer_seed_health_ok",
                "locked_validation.higher_rank_rescue", "locked_validation.semantic_gates_pass",
                "single_necessity.single_necessity_pass", "two_site_redundancy.redundancy_pass",
                "ordered_reader.reader_pass",
            },
        }
        optional_reads = optional_by_node.get(node, set())
        transition_reads_by_node = {
            "discovery_gradients": (
                "discovery_gradients.gradient_denominators_above_threshold",
            ),
            "discovery_full_ceilings": (
                "discovery_full_ceilings.natural_margin_denominators_above_threshold",
                "discovery_full_ceilings.retained_h", "discovery_full_ceilings.retained_q",
            ),
            "joint_rank1_fits": ("joint_rank1_fits.finite_optimizer_seed_health_ok",),
            "selected_family_and_rank_fits": (
                "selected_family_and_rank_fits.finite_optimizer_seed_health_ok",
            ),
            "locked_validation": (
                "locked_validation.higher_rank_rescue", "locked_validation.semantic_gates_pass",
            ),
            "single_necessity": (
                "single_necessity.single_necessity_pass", "discovery_selection.selected_h",
                "discovery_selection.selected_q", "discovery_selection.top_two_q",
                "discovery_selection.reader_selection_eligible",
            ),
            "two_site_redundancy": (
                "two_site_redundancy.redundancy_pass", "discovery_selection.selected_h",
                "discovery_selection.selected_q", "discovery_selection.reader_selection_eligible",
            ),
        }
        item = {
            "node": node,
            "required_reads": [field for field in reads if field not in optional_reads],
            "optional_reads": [field for field in reads if field in optional_reads],
            "writes": list(writes),
            "transition_reads": list(transition_reads_by_node.get(node, ())),
            "guard": {
                "name": guard,
                "required_reads": [field for field in guard_reads if field not in optional_reads],
                "optional_reads": [field for field in guard_reads if field in optional_reads],
            },
            "physical_call_stage": node if node in CALL_STAGES else None,
            "calls": calls_by_stage.get(node) if node in CALL_STAGES else None,
            "successors": list(successors),
            "capability_input": "GlobalPreflightToken" if node == "preflight" else "process_local_predecessor_StageCapability",
            "operational_failure": "operational_abort_no_scientific_terminal_no_package",
        }
        if node in {
            "discovery_gradients", "discovery_full_ceilings",
            "joint_rank1_fits", "selected_family_and_rank_fits",
        }:
            item["finite_completed_health_failure"] = "instrument_invalid"
        if node == "discovery_full_ceilings":
            item["finite_completed_empty_H_or_Q"] = "no_intervention_ceiling"
        nodes.append({**item, "node_contract_sha256": canonical_sha256(item)})
    _validate_dag(nodes)
    return nodes


def _validate_dag(nodes: Sequence[Mapping[str, Any]]) -> None:
    if type(nodes) not in {list, tuple} or not nodes:
        raise CompileError("stage DAG must be a nonempty exact sequence")
    writers: dict[str, str] = {}
    seen: set[str] = set()
    for item in nodes:
        node = item.get("node")
        if type(node) is not str or node in seen:
            raise CompileError("stage DAG has invalid/duplicate node")
        required = item.get("required_reads")
        optional = item.get("optional_reads")
        writes = item.get("writes")
        guard = item.get("guard")
        if type(required) is not list or type(optional) is not list \
                or type(writes) is not list or type(guard) is not dict \
                or type(item.get("transition_reads")) is not list \
                or type(guard.get("required_reads")) is not list \
                or type(guard.get("optional_reads")) is not list:
            raise CompileError("stage DAG dataflow fields malformed")
        all_reads = set(required) | set(optional)
        if set(required) & set(optional) \
                or not (set(guard["required_reads"]) | set(guard["optional_reads"])).issubset(all_reads):
            raise CompileError(f"stage DAG read/guard classification malformed at {node}")
        for field in writes:
            if type(field) is not str or field in writers:
                raise CompileError(f"stage DAG derived field has multiple writers: {field}")
            writers[field] = node
        call_stage = item.get("physical_call_stage")
        if call_stage is not None and (call_stage != node or call_stage not in CALL_STAGES):
            raise CompileError("stage DAG physical-call binding changed")
        calls = item.get("calls")
        if call_stage is not None:
            if calls is not None and (
                type(calls) is not dict or calls.get("stage") != node
                or calls.get("stage_call_contract_sha256") != canonical_sha256({
                    key: calls[key] for key in calls if key != "stage_call_contract_sha256"
                })
            ):
                raise CompileError("stage DAG exact calls binding changed")
        elif calls is not None:
            raise CompileError("zero-call DAG node declares physical calls")
        for successor in item.get("successors", []):
            if type(successor) is not str:
                raise CompileError("stage DAG successor malformed")
        seen.add(node)
    payload_types = globals().get("_PAYLOAD_TYPES", {})
    if payload_types:
        for item in nodes:
            node = item["node"]
            if node in payload_types:
                expected_writes = {
                    f"{node}.{field}" for field in payload_types[node].__dataclass_fields__
                }
                if set(item["writes"]) != expected_writes:
                    raise CompileError(f"DAG writes differ from exact {node} payload fields")
    names = [item["node"] for item in nodes]
    positions = {node: index for index, node in enumerate(names)}
    for item in nodes:
        for successor in item["successors"]:
            if successor not in positions or positions[successor] <= positions[item["node"]]:
                raise CompileError("stage DAG is cyclic or not topologically ordered")
    predecessors = {name: set() for name in names}
    for item in nodes:
        for successor in item["successors"]:
            predecessors[successor].add(item["node"])
    dominators = {names[0]: {names[0]}}
    for name in names[1:]:
        dominators[name] = set(names)
    changed = True
    while changed:
        changed = False
        for name in names[1:]:
            preds = predecessors[name]
            if not preds:
                raise CompileError("non-root stage is unreachable")
            common = set.intersection(*(dominators[pred] for pred in preds))
            update = {name} | common
            if update != dominators[name]:
                dominators[name] = update
                changed = True
    for item in nodes:
        node = item["node"]
        for field in item["required_reads"] + item["guard"]["required_reads"]:
            writer = writers.get(field)
            if writer is None or writer not in dominators[node]:
                raise CompileError(f"required field writer does not dominate {node}: {field}")
        for field in item["optional_reads"] + item["guard"]["optional_reads"]:
            writer = writers.get(field)
            if writer is None or positions[writer] >= positions[node]:
                raise CompileError(f"optional field is not from a prior branch at {node}: {field}")
        for field in item["transition_reads"]:
            writer = writers.get(field)
            if writer is None or (writer != node and writer not in dominators[node]):
                raise CompileError(f"transition reads unavailable field at {node}: {field}")
    for item in nodes:
        core = {key: item[key] for key in item if key != "node_contract_sha256"}
        if item.get("node_contract_sha256") != canonical_sha256(core):
            raise CompileError("stage DAG node contract digest changed")


def longest_compatible_stage_path(stage_weights: Mapping[str, float]) -> dict[str, Any]:
    """Exact DAG longest path; weights remain unavailable until reviewed timing exists."""
    nodes = _dag()
    names = [item["node"] for item in nodes]
    if type(stage_weights) is not dict or set(stage_weights) != set(names):
        raise CompileError("longest-path weights must cover exactly every frozen DAG node")
    cost: dict[str, float] = {}
    witness: dict[str, tuple[str, ...]] = {}
    for item in reversed(nodes):
        node = item["node"]
        weight = stage_weights[node]
        if type(weight) not in {int, float} or not math.isfinite(float(weight)) or weight < 0:
            raise CompileError("stage weights must be finite nonnegative numbers excluding bool")
        successors = item["successors"]
        if successors:
            chosen = min(successors, key=lambda successor: (-cost[successor], names.index(successor)))
            cost[node] = float(weight) + cost[chosen]
            witness[node] = (node,) + witness[chosen]
        else:
            cost[node] = float(weight)
            witness[node] = (node,)
    return {
        "seconds": cost["preflight"],
        "witness": list(witness["preflight"]),
        "algorithm": "reverse_topological_max_successor_ties_earlier_frozen_order",
    }


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
        "logical_rule": "exact frozen-v2 per-entry SHA256 Rademacher matrix then reduced QR with positive R diagonal",
        "entry_text": (
            "task14-localization-v2-init|seed|rank|site|objective_name|d|j; "
            "all values are base-10 text with no padding"
        ),
        "rademacher_rule": (
            "+1 when the low bit of the first SHA256 digest byte is zero; -1 otherwise"
        ),
        "entry_order": "d_major_then_j_increasing",
        "counter_encoding": None,
        "qr": "float32_reduced_QR_then_column_sign_from_R_diagonal_nonnegative_zero_maps_positive",
        "replay_required_before_fit": True,
        "ranks": list(RANKS),
        "seeds": list(SEEDS),
    }


def initialization_entry_sign(
    *, seed: int, rank: int, site: str, objective_name: str, d: int, j: int,
) -> int:
    """Replay the frozen v2 initializer literally, one matrix entry at a time."""
    _exact_int(seed, "initialization seed")
    _exact_int(rank, "initialization rank", minimum=1)
    _exact_int(d, "initialization row", minimum=0)
    _exact_int(j, "initialization column", minimum=0)
    if seed not in SEEDS or rank not in RANKS or type(site) is not str \
            or site not in {_site(position, boundary) for position in POSITIONS for boundary in BOUNDARIES} \
            or type(objective_name) is not str or objective_name not in {"joint", "A1_only", "A2_only"}:
        raise CompileError("initialization coordinate differs from frozen v2 domain")
    text = "|".join((
        "task14-localization-v2-init", str(seed), str(rank), site,
        objective_name, str(d), str(j),
    ))
    return 1 if hashlib.sha256(text.encode("ascii")).digest()[0] & 1 == 0 else -1


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
            "causal_stage_order": list(CALL_STAGES),
            "stage_capability": (
                "process-local identity-sealed capability binds canonical manifest/index preflight token, exact "
                "predecessor receipt, and only outcomes available after the preceding completed stage"
            ),
            "active_chunk": (
                "seek the frozen offset/count slice; verify slice SHA and hash-chain root; regenerate descriptors "
                "from preflight-captured inputs and compare each id immediately before executing the chunk"
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
        "breakdown": "nonfinite_is_operational_abort_no_package;finite_beta_le_1e-12_is_valid_invariant_subspace_stop",
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


def _stage_state_contract() -> dict[str, Any]:
    return {
        "schema": "task14_fit_localization_v2_stage_state_v3",
        "exact_types": {
            "bool": "type(value) is bool",
            "count_and_site": "type(value) is int; bool and float forbidden",
            "site_collection": "type(value) is tuple; sorted unique boundaries",
            "sha256": "exactly 64 lowercase hexadecimal characters",
        },
        "payload_types": {
            stage: _PAYLOAD_TYPE_NAMES[stage] for stage in _PAYLOAD_TYPE_NAMES
        },
        "future_fields": "structurally impossible: each stage has a distinct dataclass",
        "eligible_H_vs_retained_H": "len(retained_h)==min(3,eligible_h_count)",
        "eligible_Q_vs_retained_Q": "len(retained_q)==eligible_q_count",
        "scientific_terminals": list(TERMINALS),
        "operational_disposition": "operational_abort_without_scientific_terminal_or_package",
        "health_invalidity": (
            "instrument_invalid after completed finite gradient or natural-margin denominator evidence below its "
            "frozen threshold, or after every scheduled physical call in joint_rank1_fits or "
            "selected_family_and_rank_fits completed with finite optimizer/seed-health evidence"
        ),
        "projector": "project_stagewise_terminal pure CPU function in this exact captured compiler",
        "evidence_trust_boundary": (
            "payload evidence digests are format bindings, not proof of model-derived values; execution stays "
            "blocked until a separately reviewed producer recomputes every payload from retained primitive evidence"
        ),
    }


def _stage_replay_contract() -> dict[str, Any]:
    return {
        "schema": "task14_fit_localization_v2_stage_replay_v3",
        "global_preflight": "full canonical manifest/index replay before model access",
        "captured_inputs": "owned by the process-local global token; no filesystem reopen",
        "stage_order": list(CALL_STAGES),
        "future_selection_forbidden": True,
        "selected_configuration_rule": "all and only registered fits/evaluations at exact selected H and Q after selection",
        "redundancy_rule": "only the exact top-two-Q pair after selected-Q singleton necessity failure",
        "reader_rule": "only selected H<Q after either exclusive necessity route succeeds",
        "guard_evaluator": "_stage_chunk_active from typed predecessor history",
        "receipt_rule": "every chunk gets active_slice_verified or inactive_skip_zero_calls",
        "capability_rule": "every physical stage receipt binds the exact live predecessor capability identity",
    }


CALL_STAGES = (
    "native_cache", "discovery_gradients", "discovery_full_ceilings",
    "joint_rank1_fits", "spectral_finite_diagnostic", "selected_family_and_rank_fits",
    "validation_full_ceilings", "locked_validation", "single_necessity",
    "two_site_redundancy", "ordered_reader",
)

_PAYLOAD_TYPE_NAMES = {
    "native_cache": "NativeState",
    "discovery_gradients": "GradientState",
    "discovery_full_ceilings": "CeilingState",
    "joint_rank1_fits": "JointFitState",
    "spectral_finite_diagnostic": "SpectralState",
    "discovery_selection": "SelectionState",
    "selected_family_and_rank_fits": "SelectedFitState",
    "validation_full_ceilings": "ValidationCeilingState",
    "locked_validation": "ValidationState",
    "single_necessity": "NecessityState",
    "two_site_redundancy": "RedundancyState",
    "ordered_reader": "ReaderState",
}


def _compile_stage_chunks(
    stage: str,
    captured_inputs: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if stage not in CALL_STAGES:
        raise CompileError(f"unknown physical call stage: {stage}")
    authority, partition, donors = captured_inputs if captured_inputs is not None else load_inputs()
    rows, endpoints = _row_maps(authority)
    del rows
    _validate_endpoint_table(donors, endpoints)
    records, records_by_id = _record_maps(donors)
    chunks: list[dict[str, Any]] = []
    if stage in {"native_cache", "discovery_gradients"}:
        chunks.extend(_native_chunks(
            authority, partition, endpoints, records_by_id, only=stage,
        ))
    elif stage == "discovery_full_ceilings":
        for position in POSITIONS:
            for boundary in BOUNDARIES:
                chunks.append(_ceiling_chunk(
                    partition_name="DISCOVERY", position=position, boundary=boundary,
                    records=records, endpoints=endpoints, records_by_id=records_by_id,
                    activation="gradient_cache_complete",
                ))
    elif stage == "joint_rank1_fits":
        for position, boundary, objective, rank, seed, activation in _fit_templates():
            if objective == "joint" and rank == 1:
                chunks.append(_fit_chunk(
                    position=position, boundary=boundary, objective=objective, rank=rank, seed=seed,
                    records=records, endpoints=endpoints, records_by_id=records_by_id,
                    activation=activation,
                ))
    elif stage == "spectral_finite_diagnostic":
        for position in POSITIONS:
            for boundary in BOUNDARIES:
                chunks.append(_spectral_chunk(
                    position=position, boundary=boundary, records=records,
                    endpoints=endpoints, records_by_id=records_by_id,
                ))
    elif stage == "selected_family_and_rank_fits":
        for position, boundary, objective, rank, seed, activation in _fit_templates():
            if not (objective == "joint" and rank == 1):
                chunks.append(_fit_chunk(
                    position=position, boundary=boundary, objective=objective, rank=rank, seed=seed,
                    records=records, endpoints=endpoints, records_by_id=records_by_id,
                    activation=activation,
                ))
    elif stage == "validation_full_ceilings":
        for position in POSITIONS:
            for boundary in BOUNDARIES:
                chunks.append(_ceiling_chunk(
                    partition_name="VALIDATION", position=position, boundary=boundary,
                    records=records, endpoints=endpoints, records_by_id=records_by_id,
                    activation="site_is_selected_H_or_Q",
                ))
    elif stage == "locked_validation":
        for position, boundary, objective, rank, seed, _activation in _fit_templates():
            chunks.append(_projected_eval_chunk(
                partition_name="VALIDATION", position=position, boundary=boundary,
                objective=objective, rank=rank, seed=seed, records=records,
                endpoints=endpoints, records_by_id=records_by_id,
                activation="fit_is_selected_H_or_Q_configuration",
            ))
    elif stage == "single_necessity":
        for boundary in BOUNDARIES:
            for seed in SEEDS:
                chunks.append(_necessity_chunk_exact(
                    authority=authority, partition=partition, boundary=boundary, seed=seed,
                    endpoints=endpoints, records_by_id=records_by_id,
                ))
    elif stage == "two_site_redundancy":
        for first_index, first in enumerate(BOUNDARIES):
            for second in BOUNDARIES[first_index + 1:]:
                for seed in SEEDS:
                    chunks.append(_redundancy_chunk(
                        authority=authority, partition=partition, first=first, second=second,
                        seed=seed, endpoints=endpoints, records_by_id=records_by_id,
                    ))
    elif stage == "ordered_reader":
        for first_index, first in enumerate(BOUNDARIES):
            for second in BOUNDARIES[first_index + 1:]:
                for seed in SEEDS:
                    chunks.append(_reader_chunk(
                        h_boundary=first, q_boundary=second, seed=seed, records=records,
                        endpoints=endpoints, records_by_id=records_by_id,
                    ))
    return chunks


def _compile_chunks(
    captured_inputs: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    inputs = captured_inputs if captured_inputs is not None else load_inputs()
    chunks = [
        chunk for stage in CALL_STAGES
        for chunk in _compile_stage_chunks(stage, inputs)
    ]
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise CompileError("chunk identity collision")
    return chunks


def _compiler_visit_call_descriptors(
    visitor: Any,
    captured_inputs: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Compiler-build-only traversal; runtime code must use sealed ``replay_stage``."""
    global _CALL_VISITOR
    if _CALL_VISITOR is not None:
        raise CompileError("nested call-descriptor replay is forbidden")
    _CALL_VISITOR = visitor
    try:
        return _compile_chunks(captured_inputs)
    finally:
        _CALL_VISITOR = None


def _compiler_visit_stage_call_descriptors(
    stage: str, visitor: Any,
    captured_inputs: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Compiler-build-only stage traversal, never a runtime execution API."""
    global _CALL_VISITOR
    if _CALL_VISITOR is not None:
        raise CompileError("nested call-descriptor replay is forbidden")
    _CALL_VISITOR = visitor
    try:
        return _compile_stage_chunks(stage, captured_inputs)
    finally:
        _CALL_VISITOR = None


def _compiler_iter_call_descriptors(
    captured_inputs: tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Stream canonical descriptors while building or globally preflighting.

    A bounded producer queue keeps memory bounded without reopening mutable
    source paths after verification.  The compact checked-in index commits to
    every descriptor SHA-256; a future producer must use this iterator and
    compare each regenerated call ID before model access.  This private helper
    never authorizes execution; only ``replay_stage`` can expose live calls.
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
            _compiler_visit_call_descriptors(visit, captured_inputs)
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


def _stage_ranges(chunks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Commit the contiguous causal stage layout of chunks and call-index slices."""
    result: list[dict[str, Any]] = []
    cursor = 0
    call_cursor = 0
    for stage in CALL_STAGES:
        start = cursor
        while cursor < len(chunks) and _chunk_stage(str(chunks[cursor]["chunk_id"])) == stage:
            chunk = chunks[cursor]
            if int(chunk["call_index_offset"]) != call_cursor:
                raise CompileError("stage range encountered a noncontiguous call-index offset")
            call_cursor += int(chunk["call_index_count"])
            cursor += 1
        selected = chunks[start:cursor]
        if not selected:
            raise CompileError(f"physical stage has no canonical chunks: {stage}")
        core = {
            "stage": stage,
            "chunk_offset": start,
            "chunk_count": len(selected),
            "call_index_offset": int(selected[0]["call_index_offset"]),
            "call_count": sum(int(chunk["call_index_count"]) for chunk in selected),
            "chunk_ids_sha256": canonical_sha256([chunk["chunk_id"] for chunk in selected]),
        }
        result.append({**core, "stage_range_sha256": canonical_sha256(core)})
    if cursor != len(chunks):
        raise CompileError("chunks are not in exact causal stage order")
    return result


def physical_call_shape(call: Mapping[str, Any]) -> dict[str, Any]:
    """Return a tractable equivalence class for identical physical operations.

    Item identities, lexical cell names, and array/cache labels are deliberately
    excluded.  Tensor shapes, suffix boundary, intervention operations, loss
    operations, graph retention, and every workload-changing count remain.
    """
    if type(call) is not dict or call.get("schema") != CALL_SCHEMA:
        raise CompileError("physical call shape requires an exact canonical call mapping")
    fit = call.get("fit")
    rank = None if fit is None else fit.get("rank")
    if rank is not None:
        _exact_int(rank, "physical call rank", minimum=1)
    uses_histogram: dict[str, int] = defaultdict(int)
    for roles in call["item_uses"].values():
        generalized = []
        for role in roles:
            parts = str(role).split(":")
            generalized.append(
                "normalizer" if parts[0] == "normalizer"
                else f"train:{parts[1]}" if len(parts) >= 2 else parts[0]
            )
        uses_histogram[canonical_sha256(sorted(generalized))] += 1
    logical_step = call.get("logical_step") or {}
    normalizers = logical_step.get("normalizer_cells", {})
    slots = logical_step.get("slots", [])
    slot_histogram: dict[str, int] = defaultdict(int)
    for slot in slots:
        slot_histogram[str(slot["aggregate"])] += 1

    def array_signature(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return sorted(({
            "contiguous": item["contiguous"], "dtype": item["dtype"],
            "retained": item["retained"], "shape": item["shape"],
        } for item in items), key=lambda item: canonical_bytes(item))

    cache_classes = {
        "fit_position_residuals": "residual_position_cache",
        "c_second_head_residuals": "second_head_residual_cache",
        "native_answer_foil_logits": "two_logit_cache",
        "fitted_projector_registry": "projector_parameter_cache",
        "discovery_H_Q_gradient_cache": "gradient_position_cache",
    }

    def cache_signature(items: Sequence[str]) -> dict[str, int]:
        output: dict[str, int] = defaultdict(int)
        for item in items:
            output[cache_classes.get(item, "unknown_cache_class")] += 1
        return dict(sorted(output.items()))

    variant_operations = {
        None: "registered_single_site_operation",
        "neutral_selected_Q": "one_site_projector_neutralization",
        "neutral_first_Q": "one_of_two_site_projector_neutralization",
        "neutral_second_Q": "one_of_two_site_projector_neutralization",
        "neutral_both_Q": "two_site_projector_neutralization",
        "upstream_H_patch": "H_sufficiency_patch",
        "upstream_H_patch_then_native_Q_reset": "H_patch_then_Q_reset",
        "upstream_H_neutral": "H_neutralization",
        "upstream_H_neutral_then_donor_Q_insert": "H_neutral_then_Q_rescue",
        "downstream_Q_patch": "Q_sufficiency_patch",
    }
    if call["variant"] not in variant_operations:
        raise CompileError("physical call has unknown intervention operation")
    core = {
        "stage": call["stage"],
        "call_kind": call["call_kind"],
        "item_count": _exact_int(call["item_count"], "physical item count", minimum=1),
        "sequence_length": _exact_int(call["sequence_length"], "physical sequence length", minimum=1),
        "item_kind": call["item_kind"],
        "position": call["position"],
        "boundary": call["boundary"],
        "extra_positions": list(call["extra_positions"]),
        "projector_rank": rank,
        "intervention_operation": variant_operations[call["variant"]],
        "cache_read_classes": cache_signature(call["cache_reads"]),
        "cache_write_classes": cache_signature(call["cache_writes"]),
        "retained_output": _exact_bool(call["retained_output"], "physical retained output"),
        "array_contracts": array_signature(call["array_contracts"]),
        "state_array_contracts": array_signature(call["state_array_contracts"]),
        "item_use_role_histogram": dict(sorted(uses_histogram.items())),
        "normalizer_cell_sizes": sorted(len(value) for value in normalizers.values()),
        "logical_slot_histogram": dict(sorted(slot_histogram.items())),
        "participates_in_backward": _exact_bool(
            call["participates_in_backward"], "physical participates_in_backward",
        ),
        "logical_backward_after_this_call": _exact_bool(
            call["logical_backward_after_this_call"], "physical logical backward marker",
        ),
        "backward_graph_batch_count": (
            _exact_int(call["batch_count"], "physical graph batch count", minimum=1)
            if call["logical_backward_after_this_call"] else 0
        ),
    }
    return {**core, "call_shape_sha256": canonical_sha256(core)}


def _shape_multiplicity_contract(
    counts: Mapping[str, int], shapes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for shape_id in sorted(shapes):
        count = counts.get(shape_id)
        if type(count) is not int or count <= 0:
            raise CompileError("physical call shape lacks positive template multiplicity")
        rows.append({
            "call_shape_sha256": shape_id,
            "stage": shapes[shape_id]["stage"],
            "template_call_count": count,
        })
    if set(counts) != set(shapes):
        raise CompileError("shape multiplicity keys differ from shape registry")
    return rows


def _stage_call_contract(
    chunks: Sequence[Mapping[str, Any]],
    shape_multiplicities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for stage in CALL_STAGES:
        stage_chunks = [chunk for chunk in chunks if chunk["stage"] == stage]
        stage_shapes = [row for row in shape_multiplicities if row["stage"] == stage]
        core = {
            "stage": stage,
            "chunk_count": len(stage_chunks),
            "chunk_ids_sha256": canonical_sha256([chunk["chunk_id"] for chunk in stage_chunks]),
            "chunks_root_sha256": canonical_sha256(stage_chunks),
            "template_call_count": sum(int(chunk["call_count"]) for chunk in stage_chunks),
            "shape_class_count": len(stage_shapes),
            "shape_multiplicities_root_sha256": canonical_sha256(stage_shapes),
            "physical_price": _sum_price(stage_chunks),
        }
        result.append({**core, "stage_call_contract_sha256": canonical_sha256(core)})
    return result


def build_bundle() -> tuple[dict[str, Any], bytes]:
    call_index = bytearray()
    chunk_call_ids: dict[str, list[str]] = defaultdict(list)
    call_shapes: dict[str, dict[str, Any]] = {}
    call_shape_counts: dict[str, int] = defaultdict(int)

    def visit(chunk_id: str, call: Mapping[str, Any]) -> None:
        call_id = str(call["call_id"])
        chunk_call_ids[chunk_id].append(call_id)
        call_index.extend(bytes.fromhex(call_id))
        shape = physical_call_shape(dict(call))
        if shape["call_shape_sha256"] in call_shapes \
                and call_shapes[shape["call_shape_sha256"]] != shape:
            raise CompileError("physical call-shape hash collision across execution signatures")
        call_shapes[shape["call_shape_sha256"]] = shape
        call_shape_counts[shape["call_shape_sha256"]] += 1

    chunks = _compiler_visit_call_descriptors(visit)
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
    shape_multiplicities = _shape_multiplicity_contract(call_shape_counts, call_shapes)
    stage_call_contract = _stage_call_contract(chunks, shape_multiplicities)
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
        "compiler_source_sha256": bytes_sha256(safe_read(Path(__file__))),
        "stage_replay_contract": _stage_replay_contract(),
        "stage_state_contract": _stage_state_contract(),
        "stage_call_contract": stage_call_contract,
        "stage_ranges": _stage_ranges(chunks),
        "conditional_price": _price_contract(chunks),
        "dag": _dag(stage_call_contract),
        "fit_only": _fit_only_contract(),
        "initialization": _initialization_contract(),
        "model_contract": _model_contract(),
        "physical_batching": _physical_batching_contract(),
        "physical_call_shape_count": len(call_shapes),
        "physical_call_shapes": [call_shapes[key] for key in sorted(call_shapes)],
        "physical_call_shapes_root_sha256": canonical_sha256(
            [call_shapes[key] for key in sorted(call_shapes)]
        ),
        "physical_call_shape_multiplicities": shape_multiplicities,
        "physical_call_shape_multiplicities_root_sha256": canonical_sha256(shape_multiplicities),
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
        "artifact_closure", "stage_replay_contract", "stage_state_contract", "stage_ranges",
        "call_chunk_count", "call_chunks", "call_chunks_root_sha256", "call_index",
        "compiler_source_sha256",
        "conditional_price", "contract_sha256", "dag", "fit_only", "initialization",
        "model_contract", "physical_batching", "physical_call_shape_count", "physical_call_shapes",
        "physical_call_shapes_root_sha256", "retained_arrays", "retained_byte_contract",
        "runtime_and_publication", "science", *_identity_contract().keys(),
    }
    if set(value) != expected_keys:
        raise CompileError("manifest top-level fields changed")
    for key, expected in _identity_contract().items():
        if value.get(key) != expected:
            raise CompileError(f"manifest identity/status changed: {key}")
    _exact_sha(value.get("compiler_source_sha256"), "compiler source SHA-256")
    if value["compiler_source_sha256"] != bytes_sha256(safe_read(Path(__file__))):
        raise CompileError("manifest compiler source digest differs from executing source")
    static_sections = {
        "artifact_closure": _artifact_closure_contract(),
        "stage_replay_contract": _stage_replay_contract(),
        "stage_state_contract": _stage_state_contract(),
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
    shapes = value.get("physical_call_shapes")
    if type(shapes) is not list or not shapes:
        raise CompileError("physical call-shape registry missing")
    if shapes != sorted(shapes, key=lambda item: item.get("call_shape_sha256", "")) \
            or len({item.get("call_shape_sha256") for item in shapes}) != len(shapes):
        raise CompileError("physical call-shape registry order/census changed")
    for shape in shapes:
        observed_shape = dict(shape)
        digest = observed_shape.pop("call_shape_sha256", None)
        if digest != canonical_sha256(observed_shape):
            raise CompileError("physical call-shape digest changed")
    shapes_root = canonical_sha256(shapes)
    if value.get("physical_call_shape_count") != len(shapes) \
            or value.get("physical_call_shapes_root_sha256") != shapes_root:
        raise CompileError("physical call-shape census/root changed")
    if CANONICAL_CALL_SHAPE_COUNT >= 0 and (
        len(shapes) != CANONICAL_CALL_SHAPE_COUNT
        or shapes_root != CANONICAL_CALL_SHAPES_ROOT_SHA256
    ):
        raise CompileError("physical call shapes differ from frozen canonical registry")
    chunks = value.get("call_chunks")
    if type(chunks) is not list or type(value.get("call_chunk_count")) is not int \
            or value.get("call_chunk_count") != len(chunks):
        raise CompileError("chunk census mismatch")
    if value.get("call_chunks_root_sha256") != canonical_sha256(chunks):
        raise CompileError("chunk root mismatch")
    if len(chunks) != CANONICAL_CALL_CHUNK_COUNT \
            or (CANONICAL_CALL_CHUNKS_ROOT_SHA256 != "UNFROZEN" and
                value.get("call_chunks_root_sha256") != CANONICAL_CALL_CHUNKS_ROOT_SHA256):
        raise CompileError("conditional chunk set differs from the frozen canonical census/root")
    if len({chunk.get("chunk_id") for chunk in chunks}) != len(chunks):
        raise CompileError("duplicate chunk identity")
    for chunk in chunks:
        if chunk.get("schema") != CHUNK_SCHEMA or int(chunk.get("call_count", -1)) < 0:
            raise CompileError("malformed call chunk")
        if chunk.get("stage") != _chunk_stage(str(chunk.get("chunk_id"))):
            raise CompileError("chunk physical stage differs from canonical chunk identity")
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
    if value.get("stage_ranges") != _stage_ranges(chunks):
        raise CompileError("causal stage ranges/order changed")
    index = value.get("call_index")
    if type(index) is not dict or set(index) != {"byte_count", "call_count", "encoding", "path", "sha256"} \
            or index.get("encoding") != "ordered_raw_32_byte_SHA256_call_ids" \
            or index.get("path") != str(CALL_INDEX_PATH.relative_to(REPO_ROOT)) \
            or type(index.get("byte_count")) is not int or type(index.get("call_count")) is not int \
            or index.get("byte_count") != 32 * index.get("call_count") \
            or index.get("call_count") != sum(int(chunk["call_count"]) for chunk in chunks):
        raise CompileError("per-call index contract changed")
    _exact_sha(index.get("sha256"), "call index SHA-256")
    if index["call_count"] != CANONICAL_CALL_COUNT \
            or (CANONICAL_CALL_INDEX_SHA256 != "UNFROZEN" and
                index.get("sha256") != CANONICAL_CALL_INDEX_SHA256):
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
class GlobalPreflightToken:
    manifest_contract_sha256: str
    call_chunks_root_sha256: str
    call_index_sha256: str
    call_count: int
    compiler_sha256: str
    token_id: str
    _seal: object


@dataclass(frozen=True)
class NativeState:
    complete: bool
    evidence_sha256: str


@dataclass(frozen=True)
class GradientState:
    complete: bool
    all_values_finite: bool
    gradient_denominators_above_threshold: bool
    invalid_gradient_denominator_count: int
    evidence_sha256: str


@dataclass(frozen=True)
class CeilingState:
    complete: bool
    all_values_finite: bool
    natural_margin_denominators_above_threshold: bool
    invalid_natural_margin_denominator_count: int
    eligible_h_count: int
    eligible_h_ranked_scores: tuple[tuple[int, float], ...]
    retained_h: tuple[int, ...]
    eligible_q_count: int
    retained_q: tuple[int, ...]
    top_three_h_evidence_sha256: str
    evidence_sha256: str


@dataclass(frozen=True)
class JointFitState:
    all_scheduled_calls_complete: bool
    finite_optimizer_seed_health_ok: bool
    evidence_sha256: str


@dataclass(frozen=True)
class SpectralState:
    diagnostic_complete: bool
    evidence_sha256: str


@dataclass(frozen=True)
class SelectionState:
    selected_h: int
    selected_q: int
    top_two_q: tuple[int, int] | None
    h_objective_scores: tuple[tuple[int, float], ...]
    q_t_scores: tuple[tuple[int, float], ...]
    reader_selection_eligible: bool
    selection_evidence_sha256: str


@dataclass(frozen=True)
class SelectedFitState:
    all_scheduled_calls_complete: bool
    finite_optimizer_seed_health_ok: bool
    evidence_sha256: str


@dataclass(frozen=True)
class ValidationCeilingState:
    complete: bool
    evidence_sha256: str


@dataclass(frozen=True)
class ValidationState:
    higher_rank_rescue: bool
    semantic_gates_pass: bool | None
    evidence_sha256: str


@dataclass(frozen=True)
class NecessityState:
    single_necessity_pass: bool
    evidence_sha256: str


@dataclass(frozen=True)
class RedundancyState:
    redundancy_pass: bool
    evidence_sha256: str


@dataclass(frozen=True)
class ReaderState:
    reader_pass: bool
    evidence_sha256: str


@dataclass(frozen=True)
class StageCapability:
    next_stage: str
    predecessor_id: str
    root_token_id: str
    predecessor_completion_id: str | None
    capability_chain_root_sha256: str
    capability_id: str
    _seal: object


@dataclass(frozen=True)
class ChunkReplayReceipt:
    chunk_id: str
    stage: str
    activation_guard: str
    guard_evaluated: bool
    guard_state_sha256: str
    status: str
    call_index_offset: int
    template_call_count: int
    executed_call_count: int
    call_index_slice_sha256: str
    call_root_sha256: str
    forward_calls: int
    backward_calls: int
    backward_graph_batches: int
    optimizer_updates: int
    example_evaluations: int
    token_evaluations: int
    receipt_id: str


@dataclass(frozen=True)
class StageReplayReceipt:
    stage: str
    capability_id: str
    active_path_root_sha256: str
    chunk_receipts_root_sha256: str
    chunk_receipts: tuple[ChunkReplayReceipt, ...]
    template_call_count: int
    executed_call_count: int
    forward_calls: int
    backward_calls: int
    backward_graph_batches: int
    optimizer_updates: int
    example_evaluations: int
    token_evaluations: int
    receipt_id: str
    _seal: object


@dataclass(frozen=True)
class StageCompletionReceipt:
    stage: str
    capability_id: str
    payload_type: str
    evidence_sha256: str
    replay_receipt_id: str | None
    replay_active_path_root_sha256: str | None
    replay_executed_call_count: int
    replay_work_ledger_sha256: str
    completion_id: str
    _seal: object


@dataclass(frozen=True)
class OperationalAbortState:
    completed_stages: tuple[str, ...]
    failed_stage: str
    reason: str
    node_statuses: tuple[tuple[str, str], ...]
    active_chunk_id: str | None
    active_chunk_call_offset: int | None
    attempted_call_count: int
    completed_call_count: int
    completed_call_root_sha256: str
    completed_slice_count: int
    completed_slice_root_sha256: str
    forward_calls: int
    backward_calls: int
    backward_graph_batches: int
    optimizer_updates: int
    example_evaluations: int
    token_evaluations: int
    scientific_terminal: None = None
    package_allowed: bool = False


@dataclass(frozen=True)
class ScientificTerminalState:
    completed_stages: tuple[str, ...]
    terminal: str
    terminal_id: str
    node_statuses: tuple[tuple[str, str], ...]
    root_token_id: str
    capability_chain_root_sha256: str
    predecessor_completion_id: str
    package_allowed: bool = True
    _seal: object | None = None


@dataclass(frozen=True)
class _PreflightContext:
    manifest_bytes: bytes
    call_index_bytes: bytes
    input_bytes: tuple[bytes, bytes, bytes]
    compiler_source_bytes: bytes


_GLOBAL_TOKENS: dict[str, dict[str, Any]] = {}
_GLOBAL_CONTEXTS: dict[str, _PreflightContext] = {}
_STAGE_CAPABILITIES: dict[str, dict[str, Any]] = {}
_STAGE_REPLAYS: dict[str, dict[str, Any]] = {}
_STAGE_COMPLETIONS: dict[str, dict[str, Any]] = {}
_SCIENTIFIC_TERMINALS: dict[str, dict[str, Any]] = {}


def _exact_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise CompileError(f"{name} must be exact bool")
    return value


def _exact_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise CompileError(f"{name} must be exact int excluding bool")
    if minimum is not None and value < minimum:
        raise CompileError(f"{name} is below minimum")
    return value


def _exact_site_tuple(value: Any, name: str, *, maximum: int) -> tuple[int, ...]:
    if type(value) is not tuple:
        raise CompileError(f"{name} must be exact tuple")
    if len(value) > maximum or len(value) != len(set(value)):
        raise CompileError(f"{name} has invalid census/duplicates")
    for item in value:
        _exact_int(item, f"{name} site")
        if item not in BOUNDARIES:
            raise CompileError(f"{name} site outside boundaries")
    if value != tuple(sorted(value)):
        raise CompileError(f"{name} must be sorted")
    return value


def _exact_ranked_site_scores(
    value: Any, name: str,
) -> tuple[tuple[int, float], ...]:
    if type(value) is not tuple:
        raise CompileError(f"{name} must be exact tuple")
    observed: list[tuple[int, float]] = []
    for pair in value:
        if type(pair) is not tuple or len(pair) != 2:
            raise CompileError(f"{name} entries must be exact (int,float) tuples")
        site, score = pair
        _exact_int(site, f"{name} site")
        if site not in BOUNDARIES or type(score) is not float or not math.isfinite(score):
            raise CompileError(f"{name} has invalid site/score")
        observed.append((site, score))
    if len({site for site, _score in observed}) != len(observed):
        raise CompileError(f"{name} contains duplicate sites")
    if observed != sorted(observed, key=lambda item: (-item[1], item[0])):
        raise CompileError(f"{name} must use score-descending, earlier-boundary tie order")
    return value


def _exact_site_score_table(value: Any, name: str) -> tuple[tuple[int, float], ...]:
    if type(value) is not tuple:
        raise CompileError(f"{name} must be exact tuple")
    observed: list[tuple[int, float]] = []
    for pair in value:
        if type(pair) is not tuple or len(pair) != 2:
            raise CompileError(f"{name} entries must be exact (int,float) tuples")
        site, score = pair
        _exact_int(site, f"{name} site")
        if site not in BOUNDARIES or type(score) is not float or not math.isfinite(score):
            raise CompileError(f"{name} has invalid site/score")
        observed.append((site, score))
    if tuple(observed) != tuple(sorted(observed)) or len({site for site, _ in observed}) != len(observed):
        raise CompileError(f"{name} must be boundary-sorted and unique")
    return value


def _exact_sha(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or value != value.lower():
        raise CompileError(f"{name} must be exact SHA-256 text")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise CompileError(f"{name} is not hexadecimal") from error
    return value


def _validate_global_token(
    token: GlobalPreflightToken, manifest: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    record = _GLOBAL_TOKENS.get(getattr(token, "token_id", ""))
    if type(token) is not GlobalPreflightToken or record is None \
            or record["seal"] is not token._seal:
        raise CompileError("missing or forged process-local global preflight token")
    context = _GLOBAL_CONTEXTS.get(token.token_id)
    if type(context) is not _PreflightContext:
        raise CompileError("global preflight token has no immutable captured context")
    captured_manifest = strict_json(context.manifest_bytes, "captured canonical manifest")
    if manifest is not None and canonical_bytes(manifest, newline=True) != context.manifest_bytes:
        raise CompileError("post-preflight manifest substitution is forbidden")
    exact = {
        "manifest_contract_sha256": captured_manifest["contract_sha256"],
        "call_chunks_root_sha256": CANONICAL_CALL_CHUNKS_ROOT_SHA256,
        "call_index_sha256": CANONICAL_CALL_INDEX_SHA256,
        "call_count": CANONICAL_CALL_COUNT,
        "compiler_sha256": token.compiler_sha256,
    }
    if token.token_id != canonical_sha256(exact) or any(
        getattr(token, key) != value for key, value in exact.items()
    ):
        raise CompileError("global preflight token binding changed")
    if token.compiler_sha256 != captured_manifest["compiler_source_sha256"]:
        raise CompileError("global token compiler source binding changed")
    if bytes_sha256(context.compiler_source_bytes) != token.compiler_sha256:
        raise CompileError("captured compiler source bytes changed")
    return captured_manifest


def _public_dataclass(value: object) -> dict[str, Any]:
    if not hasattr(value, "__dataclass_fields__"):
        return {}
    return {
        field: getattr(value, field)
        for field in value.__dataclass_fields__
        if not field.startswith("_")
    }


def _zero_progress(capability_id: str) -> dict[str, Any]:
    return {
        "active_chunk_id": None,
        "active_chunk_call_offset": None,
        "attempted_call_count": 0,
        "completed_call_count": 0,
        "completed_call_root_sha256": canonical_sha256({
            "schema": "task14_v3_partial_completed_call_root_v1",
            "capability_id": capability_id,
        }),
        "completed_slice_count": 0,
        "completed_slice_root_sha256": canonical_sha256({
            "schema": "task14_v3_partial_completed_slice_root_v1",
            "capability_id": capability_id,
        }),
        "forward_calls": 0,
        "backward_calls": 0,
        "backward_graph_batches": 0,
        "optimizer_updates": 0,
        "example_evaluations": 0,
        "token_evaluations": 0,
    }


def _issue_capability(
    next_stage: str, predecessor_id: str, payload: object, completed: tuple[str, ...],
    *, root_token_id: str, predecessor_completion: StageCompletionReceipt | None = None,
) -> StageCapability:
    if root_token_id not in _GLOBAL_TOKENS:
        raise CompileError("successor capability lacks a live global root")
    predecessor_completion_id = None
    prior_chain = root_token_id
    if predecessor_completion is not None:
        completion_record = _STAGE_COMPLETIONS.get(predecessor_completion.completion_id)
        if completion_record is None or completion_record["seal"] is not predecessor_completion._seal:
            raise CompileError("successor capability lacks its process-local completion receipt")
        completion_core = completion_record["core"]
        public_completion = {
            "stage": predecessor_completion.stage,
            "capability_id": predecessor_completion.capability_id,
            "payload_type": predecessor_completion.payload_type,
            "evidence_sha256": predecessor_completion.evidence_sha256,
            "replay_receipt_id": predecessor_completion.replay_receipt_id,
            "replay_active_path_root_sha256": predecessor_completion.replay_active_path_root_sha256,
            "replay_executed_call_count": predecessor_completion.replay_executed_call_count,
            "replay_work_ledger_sha256": predecessor_completion.replay_work_ledger_sha256,
        }
        if predecessor_completion.completion_id != canonical_sha256(completion_core) \
                or any(public_completion[key] != completion_core[key] for key in public_completion):
            raise CompileError("public completion receipt fields differ from sealed core")
        predecessor_completion_id = predecessor_completion.completion_id
        prior_capability = _STAGE_CAPABILITIES.get(predecessor_id)
        if prior_capability is None:
            raise CompileError("successor capability lacks prior capability state")
        prior_chain = prior_capability["core"]["capability_chain_root_sha256"]
    chain_root = canonical_sha256({
        "prior_chain_root_sha256": prior_chain,
        "predecessor_completion_id": predecessor_completion_id,
    })
    public_payload = (
        _public_dataclass(payload)
    )
    core = {
        "next_stage": next_stage,
        "predecessor_id": predecessor_id,
        "root_token_id": root_token_id,
        "predecessor_completion_id": predecessor_completion_id,
        "capability_chain_root_sha256": chain_root,
        "payload_type": type(payload).__name__,
        "payload": public_payload,
        "completed_stages": list(completed),
    }
    capability_id = canonical_sha256(core)
    seal = object()
    capability = StageCapability(
        next_stage, predecessor_id, root_token_id, predecessor_completion_id,
        chain_root, capability_id, seal,
    )
    _STAGE_CAPABILITIES[capability_id] = {
        "seal": seal, "predecessor_id": predecessor_id, "payload": payload,
        "completed": completed, "attempted": False, "completion_attempted": False, "consumed": False,
        "replay_id": None, "completion_id": None, "core": core,
        "progress": _zero_progress(capability_id),
    }
    return capability


def _capability_record(capability: StageCapability) -> tuple[str, object, tuple[str, ...]]:
    if type(capability) is not StageCapability:
        raise CompileError("stage replay requires exact StageCapability")
    record = _STAGE_CAPABILITIES.get(capability.capability_id)
    if record is None or record["seal"] is not capability._seal \
            or record["predecessor_id"] != capability.predecessor_id:
        raise CompileError("missing or forged process-local stage capability")
    core = record["core"]
    if capability.capability_id != canonical_sha256(core) or any(
        getattr(capability, key) != core[key]
        for key in (
            "next_stage", "predecessor_id", "root_token_id",
            "predecessor_completion_id", "capability_chain_root_sha256",
        )
    ):
        raise CompileError("public stage-capability fields differ from sealed core")
    if core["payload"] != _public_dataclass(record["payload"]) \
            or core["completed_stages"] != list(record["completed"]):
        raise CompileError("sealed capability payload/history changed")
    return record["predecessor_id"], record["payload"], record["completed"]


def start_stagewise_execution(token: GlobalPreflightToken) -> StageCapability:
    _validate_global_token(token)
    record = _GLOBAL_TOKENS[token.token_id]
    if record["started"]:
        raise CompileError("global preflight token is one-shot and already started")
    record["started"] = True
    return _issue_capability(
        "native_cache", token.token_id, token, ("preflight",), root_token_id=token.token_id,
    )


def abort_stage(capability: StageCapability, reason: Any) -> OperationalAbortState:
    _predecessor, _payload, completed = _capability_record(capability)
    record = _STAGE_CAPABILITIES[capability.capability_id]
    if record["consumed"]:
        raise CompileError("stage capability is already consumed")
    if type(reason) is not str or not reason:
        raise CompileError("operational abort reason must be nonempty exact str")
    nodes = tuple(node["node"] for node in _dag())
    statuses = tuple(
        (node, "completed" if node in completed else "failed" if node == capability.next_stage else "skipped")
        for node in nodes
    )
    record["consumed"] = True
    progress = record["progress"]
    return OperationalAbortState(
        completed, capability.next_stage, reason, statuses,
        **{key: progress[key] for key in (
            "active_chunk_id", "active_chunk_call_offset", "attempted_call_count",
            "completed_call_count", "completed_call_root_sha256",
            "completed_slice_count", "completed_slice_root_sha256", "forward_calls",
            "backward_calls", "backward_graph_batches", "optimizer_updates",
            "example_evaluations", "token_evaluations",
        )},
    )


def abort_preflight(reason: Any) -> OperationalAbortState:
    """Represent a fault before a global token exists; never a scientific package."""
    if type(reason) is not str or not reason:
        raise CompileError("preflight abort reason must be nonempty exact str")
    nodes = tuple(node["node"] for node in _dag())
    statuses = tuple(
        (node, "failed" if node == "preflight" else "skipped") for node in nodes
    )
    progress = _zero_progress("preflight")
    return OperationalAbortState(
        (), "preflight", reason, statuses,
        **{key: progress[key] for key in (
            "active_chunk_id", "active_chunk_call_offset", "attempted_call_count",
            "completed_call_count", "completed_call_root_sha256",
            "completed_slice_count", "completed_slice_root_sha256", "forward_calls",
            "backward_calls", "backward_graph_batches", "optimizer_updates",
            "example_evaluations", "token_evaluations",
        )},
    )


_PAYLOAD_TYPES = {
    "native_cache": NativeState,
    "discovery_gradients": GradientState,
    "discovery_full_ceilings": CeilingState,
    "joint_rank1_fits": JointFitState,
    "spectral_finite_diagnostic": SpectralState,
    "discovery_selection": SelectionState,
    "selected_family_and_rank_fits": SelectedFitState,
    "validation_full_ceilings": ValidationCeilingState,
    "locked_validation": ValidationState,
    "single_necessity": NecessityState,
    "two_site_redundancy": RedundancyState,
    "ordered_reader": ReaderState,
}


def _history(capability: StageCapability) -> dict[str, object]:
    history: dict[str, object] = {}
    capability_id = capability.capability_id
    while capability_id in _STAGE_CAPABILITIES:
        prior = _STAGE_CAPABILITIES[capability_id]
        payload = prior["payload"]
        stage = prior["completed"][-1]
        history[stage] = payload
        capability_id = prior["predecessor_id"]
    return history


def _completed_stage_capability_ids(capability: StageCapability) -> tuple[tuple[str, str], ...]:
    """Return completed stage/capability pairs in causal order."""
    pairs: list[tuple[str, str]] = []
    capability_id = capability.predecessor_id
    while capability_id in _STAGE_CAPABILITIES:
        record = _STAGE_CAPABILITIES[capability_id]
        if record.get("completion_id") is not None:
            pairs.append((record["core"]["next_stage"], capability_id))
        capability_id = record["predecessor_id"]
    return tuple(reversed(pairs))


def _root_global_token_id(capability: StageCapability) -> str:
    predecessor = capability.capability_id
    while predecessor in _STAGE_CAPABILITIES:
        predecessor = _STAGE_CAPABILITIES[predecessor]["predecessor_id"]
    if predecessor not in _GLOBAL_TOKENS:
        raise CompileError("stage capability has no live global-preflight root")
    return predecessor


def _validate_stage_payload(stage: str, payload: object, history: Mapping[str, object]) -> None:
    expected = _PAYLOAD_TYPES.get(stage)
    if expected is None or type(payload) is not expected:
        raise CompileError(f"{stage} requires exact {getattr(expected, '__name__', 'payload')} type")
    if hasattr(payload, "evidence_sha256"):
        _exact_sha(payload.evidence_sha256, f"{stage} evidence SHA-256")
    if isinstance(payload, NativeState):
        if not _exact_bool(payload.complete, f"{stage}.complete"):
            raise CompileError(f"{stage} incomplete is an operational abort, not a receipt")
    elif isinstance(payload, GradientState):
        if type(history.get("native_cache")) is not NativeState:
            raise CompileError("gradient state lacks native cache evidence")
        if not _exact_bool(payload.complete, "discovery_gradients.complete"):
            raise CompileError("incomplete gradient schedule is an operational abort")
        if not _exact_bool(payload.all_values_finite, "gradient all_values_finite"):
            raise CompileError("nonfinite gradient evidence is an operational abort")
        valid = _exact_bool(
            payload.gradient_denominators_above_threshold,
            "gradient_denominators_above_threshold",
        )
        invalid_count = _exact_int(
            payload.invalid_gradient_denominator_count,
            "invalid_gradient_denominator_count", minimum=0,
        )
        if valid != (invalid_count == 0):
            raise CompileError("gradient denominator validity/count disagree")
    elif isinstance(payload, CeilingState):
        if type(history.get("native_cache")) is not NativeState \
                or type(history.get("discovery_gradients")) is not GradientState:
            raise CompileError("ceiling state lacks native/gradient evidence")
        if not _exact_bool(payload.complete, "discovery ceilings complete"):
            raise CompileError("incomplete discovery ceiling schedule is operational abort")
        if not _exact_bool(payload.all_values_finite, "ceiling all_values_finite"):
            raise CompileError("nonfinite discovery ceiling evidence is operational abort")
        valid_margin = _exact_bool(
            payload.natural_margin_denominators_above_threshold,
            "natural_margin_denominators_above_threshold",
        )
        invalid_margin_count = _exact_int(
            payload.invalid_natural_margin_denominator_count,
            "invalid_natural_margin_denominator_count", minimum=0,
        )
        if valid_margin != (invalid_margin_count == 0):
            raise CompileError("natural-margin denominator validity/count disagree")
        eh = _exact_int(payload.eligible_h_count, "eligible_h_count", minimum=0)
        eq = _exact_int(payload.eligible_q_count, "eligible_q_count", minimum=0)
        ranked = _exact_ranked_site_scores(
            payload.eligible_h_ranked_scores, "eligible_h_ranked_scores",
        )
        rh = _exact_site_tuple(payload.retained_h, "retained_h", maximum=3)
        rq = _exact_site_tuple(payload.retained_q, "retained_q", maximum=19)
        if len(ranked) != eh or len(rh) != min(3, eh):
            raise CompileError("retained H census must equal min(3, eligible H count)")
        if rh != tuple(sorted(site for site, _score in ranked[:3])):
            raise CompileError("retained H sites differ from exact top-three score/tie ranking")
        if len(rq) != eq:
            raise CompileError("Q retains exactly the full eligible set")
        if not set(rq).issubset({site for site, _score in ranked}):
            raise CompileError("eligible Q must be a subset of eligible H")
        expected_top3_sha = canonical_sha256({
            "schema": "task14_v3_top_three_h_evidence_v1",
            "eligible_h_count": eh,
            "eligible_h_ranked_scores": [list(item) for item in ranked],
            "retained_h": list(rh),
        })
        if payload.top_three_h_evidence_sha256 != expected_top3_sha:
            raise CompileError("top-three H evidence digest changed")
    elif isinstance(payload, (JointFitState, SelectedFitState)):
        required = (
            ("native_cache", NativeState),
            ("discovery_full_ceilings", CeilingState),
        ) if isinstance(payload, JointFitState) else (
            ("native_cache", NativeState),
            ("discovery_selection", SelectionState),
            ("joint_rank1_fits", JointFitState),
        )
        if any(type(history.get(name)) is not kind for name, kind in required):
            raise CompileError(f"{stage} lacks required native/projector-selection evidence")
        if not _exact_bool(payload.all_scheduled_calls_complete, "all_scheduled_calls_complete"):
            raise CompileError("incomplete fit schedule is operational abort, not a health receipt")
        _exact_bool(payload.finite_optimizer_seed_health_ok, "finite_optimizer_seed_health_ok")
    elif isinstance(payload, SpectralState):
        if type(history.get("joint_rank1_fits")) is not JointFitState:
            raise CompileError("spectral diagnostic lacks joint-fit projector evidence")
        if not _exact_bool(payload.diagnostic_complete, "diagnostic_complete"):
            raise CompileError("incomplete spectral diagnostic is operational abort")
    elif isinstance(payload, SelectionState):
        ceiling = history.get("discovery_full_ceilings")
        joint = history.get("joint_rank1_fits")
        spectral = history.get("spectral_finite_diagnostic")
        if type(ceiling) is not CeilingState or type(joint) is not JointFitState \
                or type(spectral) is not SpectralState:
            raise CompileError("selection lacks ceiling/joint-fit/spectral completion evidence")
        h = _exact_int(payload.selected_h, "selected_h")
        q = _exact_int(payload.selected_q, "selected_q")
        h_scores = _exact_site_score_table(payload.h_objective_scores, "h_objective_scores")
        q_scores = _exact_site_score_table(payload.q_t_scores, "q_t_scores")
        reader_eligible = _exact_bool(payload.reader_selection_eligible, "reader_selection_eligible")
        if h not in ceiling.retained_h or q not in ceiling.retained_q:
            raise CompileError("selected site is not retained")
        if tuple(site for site, _ in h_scores) != ceiling.retained_h \
                or tuple(site for site, _ in q_scores) != ceiling.retained_q:
            raise CompileError("selection score tables differ from retained sites")
        expected_h = min(h_scores, key=lambda item: (-item[1], item[0]))[0]
        if h != expected_h:
            raise CompileError("selected H differs from exact objective argmax/tie rule")
        q_by_site = dict(q_scores)
        q_max = max(q_by_site.values())
        onset = next((
            site for site in BOUNDARIES
            if site in q_by_site and site - 1 in q_by_site
            and q_by_site[site] >= 0.90 * q_max and q_by_site[site - 1] < 0.50 * q_max
        ), None) if q_max > 1e-6 else None
        expected_q = onset if onset is not None else min(q_scores, key=lambda item: (-item[1], item[0]))[0]
        if q != expected_q or reader_eligible != (onset is not None):
            raise CompileError("selected Q differs from exact onset/fallback rule")
        expected_top_two = (
            tuple(sorted(site for site, _score in sorted(q_scores, key=lambda item: (-item[1], item[0]))[:2]))
            if len(q_scores) >= 2 else None
        )
        if payload.top_two_q is not None:
            pair = _exact_site_tuple(payload.top_two_q, "top_two_q", maximum=2)
            if len(pair) != 2 or any(site not in ceiling.retained_q for site in pair):
                raise CompileError("top-two Q must contain two retained sites")
        if payload.top_two_q != expected_top_two:
            raise CompileError("top-two Q differs from independent raw-T ranking")
        expected_selection_sha = canonical_sha256({
            "schema": "task14_v3_discovery_selection_evidence_v1",
            "selected_h": h, "selected_q": q,
            "top_two_q": None if expected_top_two is None else list(expected_top_two),
            "h_objective_scores": [list(item) for item in h_scores],
            "q_t_scores": [list(item) for item in q_scores],
            "reader_selection_eligible": reader_eligible,
        })
        if payload.selection_evidence_sha256 != expected_selection_sha:
            raise CompileError("selection evidence digest changed")
    elif isinstance(payload, ValidationCeilingState):
        if type(history.get("native_cache")) is not NativeState \
                or type(history.get("discovery_selection")) is not SelectionState:
            raise CompileError("validation ceilings lack native/selected-site evidence")
        if not _exact_bool(payload.complete, "validation ceilings complete"):
            raise CompileError("incomplete validation ceilings are operational abort")
    elif isinstance(payload, ValidationState):
        if type(history.get("selected_family_and_rank_fits")) is not SelectedFitState \
                or type(history.get("validation_full_ceilings")) is not ValidationCeilingState:
            raise CompileError("locked validation lacks selected-fit/full-ceiling evidence")
        higher = _exact_bool(payload.higher_rank_rescue, "higher_rank_rescue")
        if higher:
            if payload.semantic_gates_pass is not None:
                raise CompileError("semantic gates are skipped/null after higher-rank rescue")
        else:
            _exact_bool(payload.semantic_gates_pass, "semantic_gates_pass")
    elif isinstance(payload, NecessityState):
        if type(history.get("selected_family_and_rank_fits")) is not SelectedFitState \
                or type(history.get("locked_validation")) is not ValidationState:
            raise CompileError("necessity lacks fitted-projector/locked-validation evidence")
        _exact_bool(payload.single_necessity_pass, "single_necessity_pass")
    elif isinstance(payload, RedundancyState):
        if type(history.get("selected_family_and_rank_fits")) is not SelectedFitState \
                or type(history.get("locked_validation")) is not ValidationState \
                or type(history.get("single_necessity")) is not NecessityState:
            raise CompileError("redundancy lacks projector/validation/single-site evidence")
        _exact_bool(payload.redundancy_pass, "redundancy_pass")
    elif isinstance(payload, ReaderState):
        if type(history.get("selected_family_and_rank_fits")) is not SelectedFitState \
                or type(history.get("locked_validation")) is not ValidationState \
                or type(history.get("single_necessity")) is not NecessityState:
            raise CompileError("reader lacks projector/validation/necessity evidence")
        _exact_bool(payload.reader_pass, "reader_pass")


def _payload_evidence_sha256(payload: object) -> str:
    value = getattr(payload, "evidence_sha256", None)
    if value is None:
        value = getattr(payload, "selection_evidence_sha256", None)
    return _exact_sha(value, f"{type(payload).__name__} evidence SHA-256")


def _chunk_receipt_core(receipt: ChunkReplayReceipt) -> dict[str, Any]:
    return {
        field: getattr(receipt, field)
        for field in receipt.__dataclass_fields__
        if field != "receipt_id"
    }


def _replay_receipt_core(receipt: StageReplayReceipt) -> dict[str, Any]:
    return {
        "stage": receipt.stage,
        "capability_id": receipt.capability_id,
        "active_path_root_sha256": receipt.active_path_root_sha256,
        "chunk_receipts_root_sha256": receipt.chunk_receipts_root_sha256,
        "chunk_receipts": [
            {**_chunk_receipt_core(item), "receipt_id": item.receipt_id}
            for item in receipt.chunk_receipts
        ],
        "template_call_count": receipt.template_call_count,
        "executed_call_count": receipt.executed_call_count,
        "forward_calls": receipt.forward_calls,
        "backward_calls": receipt.backward_calls,
        "backward_graph_batches": receipt.backward_graph_batches,
        "optimizer_updates": receipt.optimizer_updates,
        "example_evaluations": receipt.example_evaluations,
        "token_evaluations": receipt.token_evaluations,
    }


def _validate_replay_receipt(
    receipt: StageReplayReceipt, capability: StageCapability, stage: str,
) -> dict[str, Any]:
    if type(receipt) is not StageReplayReceipt:
        raise CompileError("physical stage requires exact StageReplayReceipt")
    record = _STAGE_REPLAYS.get(receipt.receipt_id)
    if record is None or record["seal"] is not receipt._seal:
        raise CompileError("missing or forged process-local replay receipt")
    core = _replay_receipt_core(receipt)
    if core != record["core"] or receipt.receipt_id != canonical_sha256(core) \
            or receipt.stage != stage or receipt.capability_id != capability.capability_id:
        raise CompileError("public replay receipt fields differ from sealed core")
    for child in receipt.chunk_receipts:
        child_core = _chunk_receipt_core(child)
        if child.receipt_id != canonical_sha256(child_core):
            raise CompileError("chunk replay receipt ID changed")
    if receipt.chunk_receipts_root_sha256 != canonical_sha256([
        {**_chunk_receipt_core(item), "receipt_id": item.receipt_id}
        for item in receipt.chunk_receipts
    ]):
        raise CompileError("chunk replay receipt root changed")
    return core


def complete_stage(
    capability: StageCapability, payload: object,
    replay_receipt: StageReplayReceipt | None = None,
) -> StageCapability:
    _predecessor, _prior_payload, completed = _capability_record(capability)
    capability_record = _STAGE_CAPABILITIES[capability.capability_id]
    if capability_record["consumed"] or capability_record["completion_attempted"]:
        raise CompileError("stage capability is one-shot and already consumed")
    # Consume the completion attempt before inspecting any caller-supplied receipt
    # or payload.  A malformed receipt cannot be followed by a valid retry.
    capability_record["completion_attempted"] = True
    stage = capability.next_stage
    if stage in CALL_STAGES:
        replay_core = _validate_replay_receipt(replay_receipt, capability, stage)  # type: ignore[arg-type]
        if capability_record["replay_id"] != replay_receipt.receipt_id:
            raise CompileError("replay receipt is not the unique attempt for this stage")
    elif replay_receipt is not None:
        raise CompileError("zero-call stage cannot consume a physical replay receipt")
    else:
        replay_core = None
    _validate_stage_payload(stage, payload, _history(capability))
    history = {**_history(capability), stage: payload}
    next_stage = {
        "native_cache": "discovery_gradients",
        "spectral_finite_diagnostic": "discovery_selection",
        "discovery_selection": "selected_family_and_rank_fits",
        "validation_full_ceilings": "locked_validation",
        "ordered_reader": "terminal_projection",
    }.get(stage)
    if stage == "discovery_gradients":
        next_stage = (
            "discovery_full_ceilings"
            if payload.gradient_denominators_above_threshold else "terminal_projection"
        )
    elif stage == "discovery_full_ceilings":
        next_stage = (
            "terminal_projection" if (
                not payload.natural_margin_denominators_above_threshold
                or not payload.retained_h or not payload.retained_q
            ) else "joint_rank1_fits"
        )
    elif stage == "joint_rank1_fits":
        next_stage = (
            "spectral_finite_diagnostic"
            if payload.finite_optimizer_seed_health_ok else "terminal_projection"
        )
    elif stage == "selected_family_and_rank_fits":
        next_stage = (
            "validation_full_ceilings"
            if payload.finite_optimizer_seed_health_ok else "terminal_projection"
        )
    elif stage == "locked_validation":
        next_stage = (
            "single_necessity"
            if not payload.higher_rank_rescue and payload.semantic_gates_pass else "terminal_projection"
        )
    elif stage == "single_necessity":
        selection = history["discovery_selection"]
        if payload.single_necessity_pass:
            next_stage = (
                "ordered_reader"
                if selection.reader_selection_eligible and selection.selected_h < selection.selected_q
                else "terminal_projection"
            )
        else:
            next_stage = "two_site_redundancy" if selection.top_two_q is not None else "terminal_projection"
    elif stage == "two_site_redundancy":
        selection = history["discovery_selection"]
        next_stage = (
            "ordered_reader"
            if payload.redundancy_pass and selection.reader_selection_eligible
            and selection.selected_h < selection.selected_q
            else "terminal_projection"
        )
    if next_stage is None:
        raise CompileError(f"no frozen transition from {stage}")
    evidence_sha256 = _payload_evidence_sha256(payload)
    replay_work = {
        key: 0 if replay_core is None else replay_core[key]
        for key in (
            "template_call_count", "executed_call_count", "forward_calls",
            "backward_calls", "backward_graph_batches", "optimizer_updates",
            "example_evaluations", "token_evaluations",
        )
    }
    completion_core = {
        "schema": "task14_fit_localization_v2_stage_completion_v3",
        "stage": stage,
        "capability_id": capability.capability_id,
        "payload_type": type(payload).__name__,
        "payload": _public_dataclass(payload),
        "evidence_sha256": evidence_sha256,
        "replay_receipt_id": None if replay_receipt is None else replay_receipt.receipt_id,
        "replay_active_path_root_sha256": (
            None if replay_receipt is None else replay_receipt.active_path_root_sha256
        ),
        "replay_executed_call_count": replay_work["executed_call_count"],
        "replay_work_ledger_sha256": canonical_sha256(replay_work),
    }
    completion_id, completion_seal = canonical_sha256(completion_core), object()
    completion = StageCompletionReceipt(
        stage, capability.capability_id, type(payload).__name__, evidence_sha256,
        completion_core["replay_receipt_id"], completion_core["replay_active_path_root_sha256"],
        replay_work["executed_call_count"], completion_core["replay_work_ledger_sha256"],
        completion_id, completion_seal,
    )
    _STAGE_COMPLETIONS[completion_id] = {"seal": completion_seal, "core": completion_core}
    capability_record["completion_id"] = completion_id
    capability_record["consumed"] = True
    return _issue_capability(
        next_stage, capability.capability_id, payload, completed + (stage,),
        root_token_id=capability.root_token_id, predecessor_completion=completion,
    )


def project_stagewise_terminal(capability: StageCapability) -> ScientificTerminalState:
    _predecessor, _payload, completed = _capability_record(capability)
    if capability.next_stage != "terminal_projection":
        raise CompileError("scientific terminal requested before branch completion")
    record = _STAGE_CAPABILITIES[capability.capability_id]
    if record["consumed"]:
        raise CompileError("terminal capability is one-shot and already consumed")
    history = _history(capability)
    gradient = history.get("discovery_gradients")
    ceiling = history.get("discovery_full_ceilings")
    if type(gradient) is not GradientState:
        raise CompileError("scientific terminal lacks completed finite gradient state")
    if not gradient.gradient_denominators_above_threshold:
        terminal = "instrument_invalid"
    else:
        if type(ceiling) is not CeilingState:
            raise CompileError("terminal lacks discovery ceiling state")
        if not ceiling.natural_margin_denominators_above_threshold:
            terminal = "instrument_invalid"
        elif not ceiling.retained_h or not ceiling.retained_q:
            terminal = "no_intervention_ceiling"
        else:
            joint = history.get("joint_rank1_fits")
            if type(joint) is not JointFitState:
                raise CompileError("terminal lacks joint-fit state")
            if not joint.finite_optimizer_seed_health_ok:
                terminal = "instrument_invalid"
            else:
                selected_fit = history.get("selected_family_and_rank_fits")
                if type(selected_fit) is not SelectedFitState:
                    raise CompileError("terminal lacks selected-fit state")
                if not selected_fit.finite_optimizer_seed_health_ok:
                    terminal = "instrument_invalid"
                else:
                    validation = history.get("locked_validation")
                    if type(validation) is not ValidationState:
                        raise CompileError("terminal lacks locked-validation state")
                    if validation.higher_rank_rescue:
                        terminal = "fit_binary_state_rejected_higher_rank_needed_or_better"
                    elif not validation.semantic_gates_pass:
                        terminal = "fit_rank1_complete_subject_state_not_identified"
                    else:
                        necessity = history.get("single_necessity")
                        selection = history.get("discovery_selection")
                        if type(necessity) is not NecessityState or type(selection) is not SelectionState:
                            raise CompileError("terminal lacks necessity/selection state")
                        redundancy = history.get("two_site_redundancy")
                        if not necessity.single_necessity_pass \
                                and (type(redundancy) is not RedundancyState or not redundancy.redundancy_pass):
                            terminal = "fit_rank1_state_sufficiency_only"
                        else:
                            route = "single" if necessity.single_necessity_pass else "redundant"
                            reader = history.get("ordered_reader")
                            if type(reader) is ReaderState and reader.reader_pass:
                                terminal = (
                                    "fit_rank1_state_and_ordered_reader_supported" if route == "single" else
                                    "fit_rank1_redundant_state_and_ordered_reader_supported"
                                )
                            else:
                                terminal = (
                                    "fit_rank1_state_supported_reader_unresolved" if route == "single" else
                                    "fit_rank1_two_site_redundant_state_reader_unresolved"
                                )
    if terminal not in TERMINALS:
        raise CompileError("stagewise projector produced unknown terminal")
    all_nodes = tuple(node["node"] for node in _dag())
    statuses = tuple(
        (node, "completed" if node in completed or node == "terminal_projection" else "skipped")
        for node in all_nodes
    )
    statuses = tuple(
        (node, "completed_health_invalid")
        if terminal == "instrument_invalid" and (
            (node == "discovery_gradients" and not gradient.gradient_denominators_above_threshold)
            or (
                node == "discovery_full_ceilings" and isinstance(ceiling, CeilingState)
                and not ceiling.natural_margin_denominators_above_threshold
            )
            or (
                node == "joint_rank1_fits" and isinstance(history.get("joint_rank1_fits"), JointFitState)
                and not history["joint_rank1_fits"].finite_optimizer_seed_health_ok
            )
            or (
                node == "selected_family_and_rank_fits"
                and isinstance(history.get("selected_family_and_rank_fits"), SelectedFitState)
                and not history["selected_family_and_rank_fits"].finite_optimizer_seed_health_ok
            )
        ) else (node, status)
        for node, status in statuses
    )
    terminal_core = {
        "schema": "task14_fit_localization_v2_stagewise_terminal_v3",
        "terminal": terminal,
        "completed_stages": list(completed),
        "node_statuses": [list(item) for item in statuses],
        "root_token_id": capability.root_token_id,
        "capability_id": capability.capability_id,
        "capability_chain_root_sha256": capability.capability_chain_root_sha256,
        "predecessor_completion_id": capability.predecessor_completion_id,
        "evidence_chain": [
            {
                "stage": stage,
                "evidence_sha256": _payload_evidence_sha256(history[stage]),
                "completion_id": _STAGE_CAPABILITIES[next_id]["completion_id"],
            }
            for stage, next_id in _completed_stage_capability_ids(capability)
        ],
    }
    terminal_id, terminal_seal = canonical_sha256(terminal_core), object()
    record["consumed"] = True
    result = ScientificTerminalState(
        completed_stages=completed, terminal=terminal, terminal_id=terminal_id,
        node_statuses=statuses, root_token_id=capability.root_token_id,
        capability_chain_root_sha256=capability.capability_chain_root_sha256,
        predecessor_completion_id=capability.predecessor_completion_id,  # type: ignore[arg-type]
        package_allowed=True, _seal=terminal_seal,
    )
    _SCIENTIFIC_TERMINALS[terminal_id] = {"seal": terminal_seal, "core": terminal_core}
    return result


def validate_scientific_terminal(value: ScientificTerminalState) -> None:
    if type(value) is not ScientificTerminalState:
        raise CompileError("scientific terminal must have exact type")
    record = _SCIENTIFIC_TERMINALS.get(value.terminal_id)
    if record is None or record["seal"] is not value._seal:
        raise CompileError("missing or forged process-local scientific terminal")
    public = {
        "schema": "task14_fit_localization_v2_stagewise_terminal_v3",
        "terminal": value.terminal,
        "completed_stages": list(value.completed_stages),
        "node_statuses": [list(item) for item in value.node_statuses],
        "root_token_id": value.root_token_id,
        "capability_id": record["core"]["capability_id"],
        "capability_chain_root_sha256": value.capability_chain_root_sha256,
        "predecessor_completion_id": value.predecessor_completion_id,
        "evidence_chain": record["core"]["evidence_chain"],
    }
    if value.package_allowed is not True or public != record["core"] \
            or value.terminal_id != canonical_sha256(public):
        raise CompileError("public scientific terminal fields differ from sealed core")


_TIMING_CAPABILITIES: dict[str, dict[str, Any]] = {}
_TIMING_AUTHORIZATIONS: dict[str, dict[str, Any]] = {}
_DEADLINE_CAPABILITIES: dict[str, dict[str, Any]] = {}
RESERVED_NAMESPACE_PATHS = (
    "basis_aligned/bilinear_quotient/circuit_battery_task14_fit_localization_v2_fit_v1_results.json",
    "basis_aligned/bilinear_quotient/circuit_battery_task14_fit_localization_v2_fit_v1_evidence",
    "basis_aligned/bilinear_quotient/circuit_battery_task14_fit_localization_v2_fit_v1_receipt.json",
)
_NAMESPACE_ROOT = REPO_ROOT


def validate_timing_receipt_schema(receipt: Mapping[str, Any]) -> TimingCapability:
    """Schema-check a candidate receipt; this CPU compiler does not authorize it."""
    expected_keys = {
        "schema", "stage", "call_shape_sha256", "p99_seconds",
        "independent_review_sha256", "receipt_sha256",
    }
    if type(receipt) is not dict or set(receipt) != expected_keys \
            or receipt["schema"] != "task14_v3_physical_call_shape_timing_v1":
        raise CompileError("timing receipt schema/fields changed")
    stage = receipt["stage"]
    if type(stage) is not str or stage not in CALL_STAGES:
        raise CompileError("timing receipt has unknown stage")
    call_shape = _exact_sha(receipt["call_shape_sha256"], "call-shape SHA")
    review = _exact_sha(receipt["independent_review_sha256"], "timing review SHA")
    p99 = receipt["p99_seconds"]
    if type(p99) is not float or not math.isfinite(p99) or p99 <= 0:
        raise CompileError("timing p99 must be positive finite exact float")
    core = {key: receipt[key] for key in sorted(expected_keys - {"receipt_sha256"})}
    if receipt["receipt_sha256"] != canonical_sha256(core):
        raise CompileError("timing receipt digest changed")
    token_core = {"stage": stage, "call_shape_sha256": call_shape, "p99_seconds": p99,
                  "reviewed_receipt_sha256": receipt["receipt_sha256"], "review_sha256": review}
    token_id, seal = canonical_sha256(token_core), object()
    if token_id in _TIMING_CAPABILITIES:
        raise CompileError("timing receipt schema token was already registered")
    token = TimingCapability(stage, call_shape, p99, receipt["receipt_sha256"], token_id, seal)
    _TIMING_CAPABILITIES[token_id] = {
        "seal": seal, "core": token_core,
        "status": "schema_only_unapproved_until_exact_external_allowlist",
        "independent_review_sha256": review,
    }
    return token


def _validate_timing_capability(timing: TimingCapability) -> dict[str, Any]:
    if type(timing) is not TimingCapability:
        raise OperationalAbort("timing capability has wrong type")
    record = _TIMING_CAPABILITIES.get(timing.token_id)
    if record is None or record["seal"] is not timing._seal:
        raise OperationalAbort("missing or forged timing capability")
    core = record["core"]
    public = {
        "stage": timing.stage,
        "call_shape_sha256": timing.call_shape_sha256,
        "p99_seconds": timing.p99_seconds,
        "reviewed_receipt_sha256": timing.reviewed_receipt_sha256,
        "review_sha256": record["independent_review_sha256"],
    }
    if public != core or timing.token_id != canonical_sha256(core):
        raise OperationalAbort("public timing capability fields differ from sealed core")
    return record


def start_deadline(
    clock: Any, timings: tuple[TimingCapability, ...], *, authorization: TimingAuthorization,
    hard_limit_seconds: int = GPU_TIME_LIMIT_SECONDS,
) -> DeadlineCapability:
    if type(hard_limit_seconds) is not int or hard_limit_seconds != GPU_TIME_LIMIT_SECONDS:
        raise OperationalAbort("hard deadline must be exact frozen 28800 seconds")
    if type(timings) is not tuple or not timings:
        raise OperationalAbort("deadline requires a nonempty exact timing-token tuple")
    authorization_record = _TIMING_AUTHORIZATIONS.get(
        getattr(authorization, "authorization_id", ""),
    )
    if type(authorization) is not TimingAuthorization or authorization_record is None \
            or authorization_record["seal"] is not authorization._seal:
        raise OperationalAbort(
            "this compiler has no timing authorization; a future reviewed successor must allowlist exact receipts"
        )
    authorization_core = authorization_record.get("core")
    if type(authorization_core) is not dict \
            or authorization.authorization_id != canonical_sha256(authorization_core) \
            or authorization.timing_token_ids != tuple(authorization_core.get("timing_token_ids", ())):
        raise OperationalAbort("public timing authorization differs from sealed future authority")
    token_ids: list[str] = []
    for timing in timings:
        _validate_timing_capability(timing)
        token_ids.append(timing.token_id)
    if len(token_ids) != len(set(token_ids)):
        raise OperationalAbort("deadline timing tokens contain duplicates")
    if tuple(token_ids) != authorization.timing_token_ids \
            or tuple(token_ids) != authorization_record["timing_token_ids"]:
        raise OperationalAbort("timing set differs from exact external authorization")
    now = clock()
    if type(now) not in {int, float} or not math.isfinite(float(now)):
        raise OperationalAbort("initial monotonic clock value is invalid")
    core = {"timing_token_ids": token_ids, "start": float(now), "hard_limit_seconds": GPU_TIME_LIMIT_SECONDS}
    deadline_id, seal = canonical_sha256(core), object()
    if deadline_id in _DEADLINE_CAPABILITIES:
        raise OperationalAbort("deadline capability for this exact start is one-shot")
    deadline = DeadlineCapability(tuple(token_ids), deadline_id, seal)
    _DEADLINE_CAPABILITIES[deadline_id] = {
        "seal": seal, "start": float(now), "last_now": float(now),
        "timing_token_ids": tuple(token_ids), "pending_stage": None, "core": core,
    }
    return deadline


def deadline_check(
    clock: Any, *, deadline: DeadlineCapability, timing: TimingCapability,
    stage: str, call_shape_sha256: str,
    hard_limit_seconds: int = GPU_TIME_LIMIT_SECONDS,
) -> float:
    """Use exact global limit and process-local reviewed per-call-shape p99 token."""
    if type(hard_limit_seconds) is not int or hard_limit_seconds != GPU_TIME_LIMIT_SECONDS:
        raise OperationalAbort("hard deadline must be exact frozen 28800 seconds")
    _exact_sha(call_shape_sha256, "deadline call-shape SHA")
    state = _DEADLINE_CAPABILITIES.get(getattr(deadline, "deadline_id", ""))
    timing_state = _TIMING_CAPABILITIES.get(getattr(timing, "token_id", ""))
    if type(deadline) is not DeadlineCapability or state is None \
            or state["seal"] is not deadline._seal:
        raise OperationalAbort("missing or forged process-local deadline capability")
    if deadline.deadline_id != canonical_sha256(state["core"]) \
            or deadline.timing_token_ids != tuple(state["core"]["timing_token_ids"]):
        raise OperationalAbort("public deadline capability fields differ from sealed core")
    if state["pending_stage"] is not None:
        raise OperationalAbort("previous physical call lacks its after-call deadline check")
    _validate_timing_capability(timing)
    if timing_state is None or timing.token_id not in state["timing_token_ids"] \
            or stage != timing.stage or call_shape_sha256 != timing.call_shape_sha256:
        raise OperationalAbort("missing/wrong reviewed physical-call-shape timing capability")
    now = clock()
    if type(now) not in {int, float}:
        raise OperationalAbort("monotonic clock returned wrong type")
    now = float(now)
    if not math.isfinite(now) or now < state["last_now"]:
        raise OperationalAbort("monotonic clock rollback/nonfinite timestamp")
    if now - state["start"] + timing.p99_seconds > GPU_TIME_LIMIT_SECONDS:
        raise OperationalAbort(f"deadline would be exceeded by {stage}")
    state["last_now"] = now
    state["pending_stage"] = stage
    return now


def deadline_check_after(
    clock: Any, *, deadline: DeadlineCapability, stage: str,
    hard_limit_seconds: int = GPU_TIME_LIMIT_SECONDS,
) -> float:
    if type(hard_limit_seconds) is not int or hard_limit_seconds != GPU_TIME_LIMIT_SECONDS:
        raise OperationalAbort("hard deadline must be exact frozen 28800 seconds")
    state = _DEADLINE_CAPABILITIES.get(getattr(deadline, "deadline_id", ""))
    if type(deadline) is not DeadlineCapability or state is None \
            or state["seal"] is not deadline._seal:
        raise OperationalAbort("missing or forged process-local deadline capability")
    if deadline.deadline_id != canonical_sha256(state["core"]) \
            or deadline.timing_token_ids != tuple(state["core"]["timing_token_ids"]):
        raise OperationalAbort("public deadline capability fields differ from sealed core")
    if type(stage) is not str or state["pending_stage"] != stage:
        raise OperationalAbort("after-call deadline stage differs from pending physical call")
    now = clock()
    if type(now) not in {int, float}:
        raise OperationalAbort("invalid monotonic timestamp")
    now = float(now)
    if not math.isfinite(now) or now < state["last_now"] \
            or now - state["start"] > GPU_TIME_LIMIT_SECONDS:
        raise OperationalAbort(f"deadline/clock rollback after {stage}")
    state["last_now"] = now
    state["pending_stage"] = None
    return now


def preflight_namespace_absent() -> dict[str, Any]:
    """Check only the exact frozen nonempty namespace; tests patch the private root."""
    if not isinstance(_NAMESPACE_ROOT, Path) or not _NAMESPACE_ROOT.is_dir():
        raise CompileError("private namespace root must be an existing exact Path directory")
    paths = [_NAMESPACE_ROOT / relative for relative in RESERVED_NAMESPACE_PATHS]
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
    _exact_int(boundary, "boundary edit boundary")
    _exact_int(coordinate, "boundary edit coordinate")
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
    _exact_int(first_boundary, "first composed boundary")
    _exact_int(second_boundary, "second composed boundary")
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


def preflight_global_call_index(
    value: Mapping[str, Any], raw: bytes,
) -> GlobalPreflightToken:
    """Regenerate and compare every descriptor before any future model access."""
    if type(value) is not dict or type(raw) is not bytes:
        raise CompileError("preflight requires exact manifest dict and immutable index bytes")
    validate_manifest(value)
    validate_call_index(value, raw)
    captured_compiler_bytes = safe_read(Path(__file__))
    compiler_sha256 = bytes_sha256(captured_compiler_bytes)
    if compiler_sha256 != value["compiler_source_sha256"]:
        raise CompileError("captured compiler bytes differ from frozen manifest source digest")
    input_bytes = capture_input_bytes()
    captured_inputs = parse_captured_inputs(input_bytes)
    expected_chunks = iter(value["call_chunks"])
    current = next(expected_chunks, None)
    position = 0
    observed_count = 0
    for chunk_id, call in _compiler_iter_call_descriptors(captured_inputs):
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
    core = {
        "manifest_contract_sha256": value["contract_sha256"],
        "call_chunks_root_sha256": CANONICAL_CALL_CHUNKS_ROOT_SHA256,
        "call_index_sha256": CANONICAL_CALL_INDEX_SHA256,
        "call_count": CANONICAL_CALL_COUNT,
        "compiler_sha256": compiler_sha256,
    }
    token_id = canonical_sha256(core)
    seal = object()
    token = GlobalPreflightToken(**core, token_id=token_id, _seal=seal)
    if token_id in _GLOBAL_TOKENS:
        raise CompileError("global preflight for this exact run is one-shot")
    _GLOBAL_TOKENS[token_id] = {"seal": seal, "started": False}
    _GLOBAL_CONTEXTS[token_id] = _PreflightContext(
        canonical_bytes(value, newline=True), bytes(raw), input_bytes, captured_compiler_bytes,
    )
    return token


def _chunk_stage(chunk_id: str) -> str:
    parts = chunk_id.split(":")
    if chunk_id == "00_native_cache":
        return "native_cache"
    if chunk_id == "01_discovery_gradients":
        return "discovery_gradients"
    if parts[:2] == ["ceiling", "DISCOVERY"]:
        return "discovery_full_ceilings"
    if parts[0] == "fit":
        return "joint_rank1_fits" if parts[3:5] == ["joint", "rank1"] else "selected_family_and_rank_fits"
    if parts[0] == "spectral":
        return "spectral_finite_diagnostic"
    if parts[:2] == ["ceiling", "VALIDATION"]:
        return "validation_full_ceilings"
    if parts[0] == "eval":
        return "locked_validation"
    return {
        "necessity": "single_necessity",
        "redundancy": "two_site_redundancy",
        "reader": "ordered_reader",
    }.get(parts[0]) or (_ for _ in ()).throw(CompileError(f"unknown chunk id: {chunk_id}"))


def _stage_chunk_active(stage: str, chunk: Mapping[str, Any], history: Mapping[str, object]) -> bool:
    if stage in {"native_cache", "discovery_gradients", "discovery_full_ceilings"}:
        return True
    ceiling = history.get("discovery_full_ceilings")
    if type(ceiling) is not CeilingState:
        raise CompileError(f"{stage} lacks prior ceiling state")
    parts = str(chunk["chunk_id"]).split(":")
    if stage in {"joint_rank1_fits", "spectral_finite_diagnostic"}:
        position, boundary = parts[1], int(parts[2])
        return boundary in (ceiling.retained_h if position == "H" else ceiling.retained_q)
    selection = history.get("discovery_selection")
    if type(selection) is not SelectionState:
        raise CompileError(f"{stage} lacks prior selection state")
    if stage == "selected_family_and_rank_fits":
        position, boundary = parts[1], int(parts[2])
        return boundary == (selection.selected_h if position == "H" else selection.selected_q)
    if stage == "validation_full_ceilings":
        position, boundary = parts[2], int(parts[3])
        return boundary == (selection.selected_h if position == "H" else selection.selected_q)
    if stage == "locked_validation":
        position, boundary = parts[2], int(parts[3])
        return boundary == (selection.selected_h if position == "H" else selection.selected_q)
    if stage == "single_necessity":
        return int(parts[2]) == selection.selected_q
    if stage == "two_site_redundancy":
        return selection.top_two_q is not None and {
            int(parts[2]), int(parts[3]),
        } == set(selection.top_two_q)
    if stage == "ordered_reader":
        return int(parts[2]) == selection.selected_h and int(parts[4]) == selection.selected_q
    raise CompileError(f"no active-chunk rule for {stage}")


def replay_stage(
    token: GlobalPreflightToken, capability: StageCapability, visitor: Any,
) -> StageReplayReceipt:
    """Replay exactly the current physical stage using only predecessor outcomes."""
    value = _validate_global_token(token)
    context = _GLOBAL_CONTEXTS[token.token_id]
    raw = context.call_index_bytes
    _capability_record(capability)
    if _root_global_token_id(capability) != token.token_id:
        raise CompileError("stage capability and global preflight token have different roots")
    stage = capability.next_stage
    if stage not in CALL_STAGES:
        raise CompileError("current stage has no physical calls")
    capability_record = _STAGE_CAPABILITIES[capability.capability_id]
    if capability_record["attempted"] or capability_record["consumed"]:
        raise CompileError("physical stage is one-shot; replay/retry is forbidden")
    capability_record["attempted"] = True
    history = _history(capability)
    guard_state_sha256 = canonical_sha256({
        stage_name: _public_dataclass(history[stage_name]) for stage_name in sorted(history)
    })
    chunks = [chunk for chunk in value["call_chunks"] if _chunk_stage(str(chunk["chunk_id"])) == stage]
    if not chunks:
        raise CompileError("canonical physical stage has no chunks")
    for chunk in chunks:
        start = 32 * int(chunk["call_index_offset"])
        stop = start + 32 * int(chunk["call_index_count"])
        encoded = raw[start:stop]
        if bytes_sha256(encoded) != chunk["call_index_slice_sha256"]:
            raise CompileError("stage-local call-index slice digest changed")
        chain = _chain_seed(str(chunk["chunk_id"]))
        for offset in range(0, len(encoded), 32):
            chain = _chain_step(chain, encoded[offset:offset + 32].hex())
        if chain != chunk["call_root_sha256"]:
            raise CompileError("stage-local call-index root replay changed")
    by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    active_ids = {
        chunk_id for chunk_id, chunk in by_id.items()
        if _stage_chunk_active(stage, chunk, history)
    }
    local_counts = defaultdict(int)
    progress = capability_record["progress"]

    def checked_visit(chunk_id: str, call: Mapping[str, Any]) -> None:
        if chunk_id not in by_id:
            raise CompileError("stage compiler emitted a future or prior chunk")
        chunk = by_id[chunk_id]
        local = local_counts[chunk_id]
        if local >= int(chunk["call_index_count"]):
            raise CompileError("stage emitted too many descriptors")
        absolute = int(chunk["call_index_offset"]) + local
        expected = raw[32 * absolute:32 * (absolute + 1)].hex()
        if call.get("call_id") != expected:
            raise CompileError(f"stage descriptor mismatch: {chunk_id}:{local}")
        local_counts[chunk_id] += 1
        if chunk_id in active_ids:
            progress["active_chunk_id"] = chunk_id
            progress["active_chunk_call_offset"] = absolute
            progress["attempted_call_count"] += 1
            # Conservatively charge the whole attempted call before entering an
            # opaque future producer callback.  A callback failure may happen
            # after the model/backward work, so incurred work must not vanish.
            progress["forward_calls"] += int(call["forward_calls"])
            progress["backward_calls"] += int(bool(call["logical_backward_after_this_call"]))
            progress["backward_graph_batches"] += int(bool(call["participates_in_backward"]))
            progress["optimizer_updates"] += int(bool(call["logical_backward_after_this_call"]))
            progress["example_evaluations"] += int(call["item_count"])
            progress["token_evaluations"] += int(call["item_count"]) * int(call["sequence_length"])
            try:
                visitor(chunk_id, dict(call))
            except Exception as error:
                raise OperationalAbort(
                    f"physical visitor failed at {chunk_id} global call {absolute}: {type(error).__name__}"
                ) from error
            progress["completed_call_count"] += 1
            progress["completed_call_root_sha256"] = hashlib.sha256(
                bytes.fromhex(progress["completed_call_root_sha256"])
                + bytes.fromhex(str(call["call_id"]))
            ).hexdigest()
            if local_counts[chunk_id] == int(chunk["call_index_count"]):
                slice_id = canonical_sha256({
                    "chunk_id": chunk_id,
                    "call_index_offset": int(chunk["call_index_offset"]),
                    "call_index_count": int(chunk["call_index_count"]),
                    "call_index_slice_sha256": str(chunk["call_index_slice_sha256"]),
                    "call_root_sha256": str(chunk["call_root_sha256"]),
                })
                progress["completed_slice_count"] += 1
                progress["completed_slice_root_sha256"] = hashlib.sha256(
                    bytes.fromhex(progress["completed_slice_root_sha256"])
                    + bytes.fromhex(slice_id)
                ).hexdigest()

    root_token_id = _root_global_token_id(capability)
    captured_context = _GLOBAL_CONTEXTS[root_token_id]
    captured_inputs = parse_captured_inputs(captured_context.input_bytes)
    observed_chunks = _compiler_visit_stage_call_descriptors(stage, checked_visit, captured_inputs)
    index_fields = {"call_index_count", "call_index_offset", "call_index_slice_sha256"}
    frozen_core = [{key: item for key, item in chunk.items() if key not in index_fields} for chunk in chunks]
    if observed_chunks != frozen_core:
        raise CompileError("stage chunk reconstruction differs from frozen manifest")
    if any(local_counts[str(chunk["chunk_id"])] != int(chunk["call_count"]) for chunk in chunks):
        raise CompileError("stage descriptor census incomplete")
    receipts: list[ChunkReplayReceipt] = []
    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        active = chunk_id in active_ids
        child_core = {
            "chunk_id": chunk_id,
            "stage": stage,
            "activation_guard": str(chunk["activation"]),
            "guard_evaluated": active,
            "guard_state_sha256": guard_state_sha256,
            "status": "active_completed" if active else "inactive_skip_zero_calls",
            "call_index_offset": int(chunk["call_index_offset"]),
            "template_call_count": int(chunk["call_index_count"]),
            "executed_call_count": int(chunk["call_index_count"]) if active else 0,
            "call_index_slice_sha256": str(chunk["call_index_slice_sha256"]),
            "call_root_sha256": str(chunk["call_root_sha256"]),
            "forward_calls": int(chunk["forward_calls"]) if active else 0,
            "backward_calls": int(chunk["backward_calls"]) if active else 0,
            "backward_graph_batches": int(chunk["backward_graph_batches"]) if active else 0,
            "optimizer_updates": int(chunk["optimizer_updates"]) if active else 0,
            "example_evaluations": int(chunk["example_evaluations"]) if active else 0,
            "token_evaluations": int(chunk["token_evaluations"]) if active else 0,
        }
        receipts.append(ChunkReplayReceipt(
            **child_core, receipt_id=canonical_sha256(child_core),
        ))
    child_rows = [
        {**_chunk_receipt_core(item), "receipt_id": item.receipt_id} for item in receipts
    ]
    chunk_receipts_root = canonical_sha256(child_rows)
    active_root = canonical_sha256([
        {"chunk_receipt_id": item.receipt_id, "status": item.status} for item in receipts
    ])
    sums = {
        key: sum(getattr(item, key) for item in receipts)
        for key in (
            "template_call_count", "executed_call_count", "forward_calls", "backward_calls",
            "backward_graph_batches", "optimizer_updates", "example_evaluations", "token_evaluations",
        )
    }
    for key in (
        "executed_call_count", "forward_calls", "backward_calls", "backward_graph_batches",
        "optimizer_updates", "example_evaluations", "token_evaluations",
    ):
        if progress[key if key != "executed_call_count" else "completed_call_count"] != sums[key]:
            raise CompileError(f"completed physical work ledger differs from active chunks: {key}")
    core = {
        "stage": stage,
        "capability_id": capability.capability_id,
        "active_path_root_sha256": active_root,
        "chunk_receipts_root_sha256": chunk_receipts_root,
        "chunk_receipts": child_rows,
        **sums,
    }
    receipt_id, seal = canonical_sha256(core), object()
    receipt = StageReplayReceipt(
        stage, capability.capability_id, active_root, chunk_receipts_root, tuple(receipts),
        sums["template_call_count"], sums["executed_call_count"], sums["forward_calls"],
        sums["backward_calls"], sums["backward_graph_batches"], sums["optimizer_updates"],
        sums["example_evaluations"], sums["token_evaluations"], receipt_id, seal,
    )
    _STAGE_REPLAYS[receipt_id] = {"seal": seal, "core": core}
    capability_record["replay_id"] = receipt_id
    return receipt


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
        "physical_call_shape_count": manifest["physical_call_shape_count"],
        "stage_order": list(CALL_STAGES),
        "schema": "task14_fit_localization_v2_compiler_v3_dryrun_v1",
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
