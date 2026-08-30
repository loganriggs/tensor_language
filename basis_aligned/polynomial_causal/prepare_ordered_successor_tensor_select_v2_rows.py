#!/usr/bin/env python3
"""Source-closed, outcome-blind row freezer for ordered-successor SELECT v2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
from typing import Any, Callable, Mapping, NamedTuple

import torch

import ordered_successor_digit_lexicon_v2 as registry
import ordered_successor_tensor_select_registry_v2 as protocol
from ordered_successor_masks_v1 import OrderedLexicon, SuccessorMasks, build_ordered_successor_masks
import ordered_successor_tensor_discovery_v1 as v1
import ordered_successor_tensor_select_statistics_v1 as statistics
import prepare_block3_native_down_behavioral_port_v1_rows as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = ROOT / "basis_aligned/bilinear_quotient"
AMENDMENT = HERE / "ORDERED_SUCCESSOR_TENSOR_SELECT_V2_AMENDMENT.md"
AUDIT = HERE / "ordered_successor_tensor_select_v2_rows_independent_audit.json"
CACHE = BQ / ".rowcache_ordered_successor_tensor_select_v2"
RECEIPT = BQ / "ordered_successor_tensor_select_v2_rows_receipt.json"
FAILURE = BQ / "ordered_successor_tensor_select_v2_rows_failure.json"
LOCK = Path("/workspace/runs/.ordered_successor_tensor_select_v2_rows.lock")
N_SELECT = 192
START_DOCUMENT_INDEX = 200_000
CANDIDATE_DOCUMENTS = 4_096
ROW_LENGTH = 257
PREFIX_LENGTH = 32
MIN_POSITIONS = 200
MIN_DOCUMENTS = 30
V2_ARM_NAMES = protocol.ARM_NAMES
V2_SCHEMA = "ordered_successor_tensor_select_v2"
OWN_SOURCES = (
    AMENDMENT,
    HERE / "ordered_successor_digit_lexicon_v2.py",
    HERE / "test_ordered_successor_digit_lexicon_v2.py",
    HERE / "ordered_successor_tensor_select_registry_v2.py",
    Path(__file__).resolve(),
    HERE / "test_prepare_ordered_successor_tensor_select_v2_rows.py",
    HERE / "ordered_successor_tensor_select_statistics_v1.py",
    HERE / "test_ordered_successor_tensor_select_statistics_v1.py",
    ROOT / "jacclust/__init__.py",
    ROOT / "jacclust/tt_model.py",
)
STATISTICS_SOURCES = tuple(ROOT / relative for relative in statistics.SOURCE_PATHS)
SOURCE_PATHS = tuple(dict.fromkeys(
    (*OWN_SOURCES, *STATISTICS_SOURCES, *base.SOURCE_PATHS)
))


def file_sha256(path: Path) -> str:
    return base.file_sha256(path)


def tensor_sha256(value: torch.Tensor) -> str:
    return base.tensor_sha256(value)


def _stable_json(path: Path) -> tuple[dict[str, Any], str]:
    before = file_sha256(path)
    raw = path.read_bytes()
    after = file_sha256(path)
    if before != after or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"JSON changed during stable read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value, before


def source_closure(commit: str) -> dict[str, str]:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    answer = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"successor v2 row source drift: {relative}")
        answer[relative] = digest
    return answer


def validate_independent_audit(path: Path = AUDIT) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise RuntimeError("successor v2 independent row audit is absent")
    audit, digest = _stable_json(path)
    required = {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    }
    if (
        set(audit) != required
        or audit.get("schema") != "ordered_successor_tensor_select_v2_rows_independent_audit"
        or audit.get("status") != "GO" or audit.get("outcome_access") is not False
        or type(audit.get("tests_passed")) is not int or audit["tests_passed"] < 1
        or not isinstance(audit.get("reviewer"), str) or not audit["reviewer"]
    ):
        raise RuntimeError("successor v2 independent row audit is not an exact GO")
    commit, hashes = audit.get("audited_source_commit"), audit.get("audited_source_hashes")
    if not isinstance(commit, str) or len(commit) != 40 or not isinstance(hashes, dict):
        raise RuntimeError("successor v2 independent audit binding is malformed")
    if source_closure(commit) != hashes:
        raise RuntimeError("successor v2 independent audit source closure changed")
    return audit, digest


class RunClaim(NamedTuple):
    descriptor: int
    inode: int
    nonce: str


def acquire_claim(path: Path = LOCK) -> RunClaim:
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (nonce + "\n").encode("ascii"))
        os.fsync(descriptor)
        return RunClaim(descriptor, os.fstat(descriptor).st_ino, nonce)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def require_claim(claim: RunClaim, path: Path = LOCK) -> None:
    if not path.is_file() or path.stat().st_ino != claim.inode or path.read_text() != (
        claim.nonce + "\n"
    ):
        raise RuntimeError("successor v2 row claim changed")


def release_claim(claim: RunClaim, path: Path = LOCK) -> None:
    try:
        if path.exists() and path.stat().st_ino == claim.inode:
            path.unlink()
    finally:
        os.close(claim.descriptor)


def build_masks(rows: torch.Tensor, lexicon: OrderedLexicon) -> SuccessorMasks:
    return build_ordered_successor_masks(rows, lexicon, window=128, first_prediction=64)


def powered_census(masks: SuccessorMasks) -> dict[str, dict[str, int]]:
    masks.validate_partition()
    answer = {
        name: {"positions": int(mask.sum()), "documents": int(mask.any(1).sum())}
        for name, mask in masks.named_cells().items()
    }
    for name in statistics.POWERED_CELLS:
        value = answer[name]
        value["passed"] = (
            value["positions"] >= MIN_POSITIONS and value["documents"] >= MIN_DOCUMENTS
        )
    return answer


def allocate_powered_select(
    candidate_rows: torch.Tensor,
    candidate_records: list[dict[str, Any]],
    lexicon: OrderedLexicon,
    *,
    mask_builder: Callable[[torch.Tensor, OrderedLexicon], SuccessorMasks] = build_masks,
) -> tuple[torch.Tensor, list[dict[str, Any]], SuccessorMasks, dict[str, dict[str, int]]]:
    """Deterministic support-first allocation followed by earliest unused fill."""

    if (
        not torch.is_tensor(candidate_rows) or candidate_rows.device.type != "cpu"
        or candidate_rows.dtype != torch.long or candidate_rows.ndim != 2
        or candidate_rows.shape[1] != ROW_LENGTH
        or len(candidate_rows) != len(candidate_records)
        or len(candidate_rows) < N_SELECT
    ):
        raise ValueError("successor candidate rows are malformed")
    masks = mask_builder(candidate_rows, lexicon)
    masks.validate_partition()
    chosen: list[int] = []
    positions = {name: 0 for name in statistics.POWERED_CELLS}
    documents = {name: 0 for name in statistics.POWERED_CELLS}
    named = masks.named_cells()
    for index in range(len(candidate_rows)):
        contributes = False
        for name in statistics.POWERED_CELLS:
            amount = int(named[name][index].sum())
            if amount and (positions[name] < MIN_POSITIONS or documents[name] < MIN_DOCUMENTS):
                contributes = True
        if not contributes:
            continue
        chosen.append(index)
        for name in statistics.POWERED_CELLS:
            amount = int(named[name][index].sum())
            positions[name] += amount
            documents[name] += int(amount > 0)
        if len(chosen) > N_SELECT:
            raise RuntimeError("powered successor support requires more than 192 documents")
        if all(
            positions[name] >= MIN_POSITIONS and documents[name] >= MIN_DOCUMENTS
            for name in statistics.POWERED_CELLS
        ):
            break
    if not all(
        positions[name] >= MIN_POSITIONS and documents[name] >= MIN_DOCUMENTS
        for name in statistics.POWERED_CELLS
    ):
        raise RuntimeError("candidate scan cannot power every successor cell")
    selected = set(chosen)
    for index in range(len(candidate_rows)):
        if len(selected) == N_SELECT:
            break
        selected.add(index)
    if len(selected) != N_SELECT:
        raise RuntimeError("candidate scan cannot fill 192 unique successor documents")
    ordered = sorted(selected)
    index_tensor = torch.tensor(ordered, dtype=torch.long)
    rows = candidate_rows.index_select(0, index_tensor).contiguous()
    records = []
    for ordinal, index in enumerate(ordered):
        record = dict(candidate_records[index])
        record["candidate_scan_ordinal"] = index
        record["source_document_ordinal"] = ordinal
        record["row_index"] = ordinal
        records.append(record)
    selected_masks = mask_builder(rows, lexicon)
    census = powered_census(selected_masks)
    if any(census[name]["passed"] is not True for name in statistics.POWERED_CELLS):
        raise RuntimeError("selected successor role failed its powered census")
    return rows, records, selected_masks, census


def _mask_hashes(masks: SuccessorMasks) -> dict[str, str]:
    return {
        **{name: tensor_sha256(mask.contiguous()) for name, mask in masks.named_cells().items()},
        "eligible_target": tensor_sha256(masks.eligible_target.contiguous()),
        "pair_index": tensor_sha256(masks.pair_index.contiguous()),
    }


def _validate_payload(path: Path, entry: Mapping[str, Any]) -> torch.Tensor:
    before = file_sha256(path)
    if before != entry.get("file_sha256"):
        raise RuntimeError("successor v2 row payload hash changed")
    rows = torch.load(path, map_location="cpu", weights_only=True)
    if (
        file_sha256(path) != before or not torch.is_tensor(rows)
        or rows.device.type != "cpu" or rows.dtype != torch.long
        or tuple(rows.shape) != (N_SELECT, ROW_LENGTH) or not rows.is_contiguous()
        or tensor_sha256(rows) != entry.get("tensor_sha256")
    ):
        raise RuntimeError("successor v2 row payload semantic replay failed")
    return rows


def _validate_manifest(path: Path, expected: Mapping[str, Any], digest: str) -> None:
    observed, actual = _stable_json(path)
    if actual != digest or observed != expected:
        raise RuntimeError("successor v2 row manifest semantic replay failed")


def _artifact_snapshot(entry: Mapping[str, Any], manifest_sha256: str) -> dict[str, str]:
    row_path = Path(str(entry["path"]))
    manifest_path = CACHE / "select_manifest.json"
    if not row_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("successor v2 installed artifact is absent")
    observed = {
        "rows": file_sha256(row_path),
        "manifest": file_sha256(manifest_path),
    }
    expected = {
        "rows": str(entry["file_sha256"]),
        "manifest": manifest_sha256,
    }
    if observed != expected:
        raise RuntimeError("successor v2 installed artifact hash changed")
    return observed


def _terminal_absent() -> None:
    if RECEIPT.exists() or FAILURE.exists():
        raise RuntimeError("successor v2 row terminal already exists")


def _source_identity(
    canonical: Mapping[str, Any], parquet: Path, encoding: Any,
) -> dict[str, Any]:
    gate = canonical["ordered_manifest_local_parquet_identity_gate"]
    return {
        "fineweb_revision": base.BASE.local.PINNED_REVISION,
        "parquet_path": str(parquet.resolve()),
        "parquet_size": parquet.stat().st_size,
        "parquet_sha256": file_sha256(parquet),
        "ordered_manifest_gate": gate,
        "tokenizer_name": registry.ENCODING_NAME,
        "tokenizer_sha256": registry.ENCODING_SHA256,
        "lexicon_registry_sha256": registry.REGISTRY_SHA256,
    }


def discover_prior_registry_files() -> tuple[Path, ...]:
    """Return the inherited recursive census minus this transaction's own manifest."""

    own_manifest = (CACHE / "select_manifest.json").resolve()
    return tuple(
        path for path in base.discover_registry_files()
        if path.resolve() != own_manifest
    )


def _verify_prior_registry_snapshot(
    *, registry_files: tuple[Path, ...], prior: Any,
    registry_hashes: Mapping[str, str], tensor_hashes: Mapping[str, str],
    waiver_proofs: list[dict[str, Any]], nonrow_proofs: list[dict[str, Any]],
    parquet: Path,
) -> None:
    current_registry = discover_prior_registry_files()
    if current_registry != registry_files:
        raise RuntimeError("successor v2 prior registry membership changed")
    (
        current_prior, current_registry_hashes, current_tensor_hashes,
        current_waiver_proofs, current_nonrow_proofs,
    ) = base.load_registry_exclusions(current_registry)
    if discover_prior_registry_files() != current_registry:
        raise RuntimeError("successor v2 prior registry changed during replay")
    if current_registry_hashes != dict(registry_hashes):
        raise RuntimeError("successor v2 prior registry bytes changed")
    if current_tensor_hashes != dict(tensor_hashes) or current_prior != prior:
        raise RuntimeError("successor v2 prior row exclusions changed")
    if current_waiver_proofs != waiver_proofs or current_nonrow_proofs != nonrow_proofs:
        raise RuntimeError("successor v2 prior registry classifications changed")
    if parquet.stat().st_size != base.BASE.local.PINNED_SIZE or (
        file_sha256(parquet) != base.BASE.local.PINNED_SHA256
    ):
        raise RuntimeError("successor v2 pinned FineWeb parquet changed")


def _protected_replay(
    *, commit: str, sources: Mapping[str, str], audit_sha256: str,
    registry_files: tuple[Path, ...], prior: Any, registry_hashes: Mapping[str, str],
    tensor_hashes: Mapping[str, str], waiver_proofs: list[dict[str, Any]],
    nonrow_proofs: list[dict[str, Any]], canonical: Mapping[str, Any], parquet: Path,
    encoding: Any, source_identity: Mapping[str, Any],
) -> None:
    protocol.validate_registry()
    if source_closure(commit) != dict(sources):
        raise RuntimeError("successor v2 sources changed")
    if validate_independent_audit()[1] != audit_sha256:
        raise RuntimeError("successor v2 audit changed")
    _verify_prior_registry_snapshot(
        registry_files=registry_files, registry_hashes=registry_hashes,
        tensor_hashes=tensor_hashes, prior=prior, waiver_proofs=waiver_proofs,
        nonrow_proofs=nonrow_proofs, parquet=parquet,
    )
    registry.validate_encoding(encoding)
    if _source_identity(canonical, parquet, encoding) != dict(source_identity):
        raise RuntimeError("successor v2 source/tokenizer identity changed")


def _write_json_create_only(
    value: Mapping[str, Any], path: Path, *, before_link: Callable[[], None],
) -> None:
    normalized = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as sink:
            descriptor = -1
            sink.write(json.dumps(normalized, indent=2, allow_nan=False) + "\n")
            sink.flush(); os.fsync(sink.fileno())
        replay, _ = _stable_json(temporary)
        if replay != normalized:
            raise RuntimeError("successor v2 terminal JSON replay failed")
        before_link()
        os.link(temporary, path)
        base._fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def freeze_locked(claim: RunClaim) -> dict[str, Any]:
    require_claim(claim); _terminal_absent()
    if CACHE.exists():
        raise RuntimeError("successor v2 row cache already exists")
    audit, audit_sha256 = validate_independent_audit()
    commit = audit["audited_source_commit"]
    sources = source_closure(commit)
    protocol.validate_registry()
    lexicon, encoding = registry.load_pinned_lexicon()
    canonical, parquet = base.BASE.validate_ordered_source()
    registry_files = discover_prior_registry_files()
    prior, registry_hashes, tensor_hashes, waiver_proofs, nonrow_proofs = (
        base.load_registry_exclusions(registry_files)
    )
    if prior[1] and max(prior[1]) >= START_DOCUMENT_INDEX:
        raise RuntimeError("successor v2 start is not beyond every historical dataset index")
    identity = _source_identity(canonical, parquet, encoding)
    candidates, candidate_records = base.harvest_fresh_documents(
        base.BASE.local.parquet_texts([parquet]), encoding.encode_ordinary, prior,
        start_document_index=START_DOCUMENT_INDEX,
        n_source_documents=CANDIDATE_DOCUMENTS,
        token_length=ROW_LENGTH,
    )
    rows, records, masks, census = allocate_powered_select(
        candidates, candidate_records, lexicon,
    )
    disjointness = base.validate_disjointness(rows, records, prior)
    support_sha256 = v1.support_sha256(rows, (lexicon,), {lexicon.name: masks})
    manifest = {
        "schema": "ordered_successor_tensor_select_v2_rows_manifest",
        "role": "SELECT", "document_records": records,
        "powered_census": census, "mask_hashes": _mask_hashes(masks),
        "support_sha256": support_sha256,
        "lexicon_registry_sha256": registry.REGISTRY_SHA256,
        "protocol_registry_sha256": protocol.REGISTRY_SHA256,
        "v2_arm_names": list(V2_ARM_NAMES),
    }
    staging = CACHE.with_name(f".{CACHE.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        row_path = staging / "select_rows.pt"
        torch.save(rows, row_path)
        with row_path.open("rb") as handle: os.fsync(handle.fileno())
        entry = {
            "path": str((CACHE / row_path.name).resolve()),
            "shape": [N_SELECT, ROW_LENGTH], "dtype": "torch.int64",
            "file_sha256": file_sha256(row_path), "tensor_sha256": tensor_sha256(rows),
        }
        manifest["row_entry"] = entry
        manifest_path = staging / "select_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        with manifest_path.open("rb") as handle: os.fsync(handle.fileno())
        manifest_sha256 = file_sha256(manifest_path)
        _validate_payload(row_path, entry)
        _validate_manifest(manifest_path, manifest, manifest_sha256)
        _protected_replay(
            commit=commit, sources=sources, audit_sha256=audit_sha256,
            registry_files=registry_files, prior=prior, registry_hashes=registry_hashes,
            tensor_hashes=tensor_hashes, waiver_proofs=waiver_proofs,
            nonrow_proofs=nonrow_proofs, canonical=canonical, parquet=parquet,
            encoding=encoding, source_identity=identity,
        )
        _terminal_absent()
        if CACHE.exists():
            raise RuntimeError("successor v2 row cache appeared before install")
        require_claim(claim)
        base._fsync_directory(staging)
        os.replace(staging, CACHE); base._fsync_directory(CACHE.parent)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    installed_rows = _validate_payload(Path(entry["path"]), entry)
    installed_manifest = CACHE / "select_manifest.json"
    _validate_manifest(installed_manifest, manifest, manifest_sha256)
    replay_candidates, replay_records = base.harvest_fresh_documents(
        base.BASE.local.parquet_texts([parquet]), encoding.encode_ordinary, prior,
        start_document_index=START_DOCUMENT_INDEX,
        n_source_documents=CANDIDATE_DOCUMENTS, token_length=ROW_LENGTH,
    )
    replay_rows, replay_selected_records, replay_masks, replay_census = allocate_powered_select(
        replay_candidates, replay_records, lexicon,
    )
    if (
        not torch.equal(replay_rows, installed_rows) or replay_selected_records != records
        or replay_census != census or _mask_hashes(replay_masks) != manifest["mask_hashes"]
    ):
        raise RuntimeError("successor v2 deterministic allocation replay changed")
    receipt = {
        "schema": "ordered_successor_tensor_select_v2_rows_receipt",
        "status": "frozen_before_any_successor_select_model_forward",
        "authorized_role": "SELECT", "authorized_for_training": False,
        "source_commit": commit, "source_hashes": sources,
        "independent_audit": {"path": str(AUDIT), "file_sha256": audit_sha256},
        "v1_status_preserved": "PROSPECTIVE_NO_GO",
        "v2_arm_names": list(V2_ARM_NAMES), "omitted_v1_diagnostics": [v1.CURRENT_ONLY, v1.V1_ONLY],
        "protocol_registry_sha256": protocol.REGISTRY_SHA256,
        "selection": {
            "start_dataset_document_index": START_DOCUMENT_INDEX,
            "candidate_documents": CANDIDATE_DOCUMENTS, "select_documents": N_SELECT,
            "rows_per_document": 1, "row_length": ROW_LENGTH,
            "scored_positions": [64, 256], "algorithm": "support_first_then_earliest_unused",
        },
        "entry": entry,
        "manifest": {"path": str(installed_manifest), "file_sha256": manifest_sha256},
        "source_identity": identity, "registry_files": registry_hashes,
        "prior_row_tensors": tensor_hashes, "disjointness": disjointness,
        "historical_max_dataset_document_index": max(prior[1]) if prior[1] else None,
        "powered_census": census, "support_sha256": support_sha256,
        "failed_unmaterialized_registry_waivers": waiver_proofs,
        "exact_nonrow_registry_artifacts": nonrow_proofs,
        "outcome_access": {"model_imported": False, "checkpoint_loaded": False,
                           "model_forward_calls": 0, "scientific_outcomes_read": False},
        "artifact_order": ["rows", "manifest", "receipt_last"],
    }

    def final_guard() -> None:
        before = _artifact_snapshot(entry, manifest_sha256)
        _validate_payload(Path(entry["path"]), entry)
        _validate_manifest(installed_manifest, manifest, manifest_sha256)
        _protected_replay(
            commit=commit, sources=sources, audit_sha256=audit_sha256,
            registry_files=registry_files, prior=prior, registry_hashes=registry_hashes,
            tensor_hashes=tensor_hashes, waiver_proofs=waiver_proofs,
            nonrow_proofs=nonrow_proofs, canonical=canonical, parquet=parquet,
            encoding=encoding, source_identity=identity,
        )
        if _artifact_snapshot(entry, manifest_sha256) != before:
            raise RuntimeError("successor v2 installed artifacts changed during final replay")
        _terminal_absent()
        require_claim(claim)

    _write_json_create_only(receipt, RECEIPT, before_link=final_guard)
    return receipt


def freeze() -> dict[str, Any]:
    claim = acquire_claim()
    try:
        return freeze_locked(claim)
    except BaseException as error:
        if not RECEIPT.exists() and not FAILURE.exists():
            failure = {
                "schema": "ordered_successor_tensor_select_v2_rows_failure",
                "status": "terminal_failure_no_receipt", "error_type": type(error).__name__,
                "error": str(error), "cache_exists": CACHE.exists(), "outcome_access": False,
            }
            try:
                def failure_guard() -> None:
                    _terminal_absent()
                    if CACHE.exists() is not failure["cache_exists"]:
                        raise RuntimeError("successor v2 failure cache state changed")
                    require_claim(claim)
                _write_json_create_only(failure, FAILURE, before_link=failure_guard)
            except BaseException:
                pass
        raise
    finally:
        release_claim(claim)


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
