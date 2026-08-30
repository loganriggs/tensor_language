#!/usr/bin/env python3
"""Source-closed, outcome-blind row freezer for ordered-successor SELECT v3.

The owner is fail-closed behind a separately published exact-source independent audit.
Importing it performs no row, corpus, checkpoint, model, GPU, or outcome access.
"""

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

import ordered_successor_digit_lexicon_v2 as digit_registry
import ordered_successor_tensor_select_registry_v2 as protocol
import ordered_successor_tensor_select_v3_budget as budget
import prepare_ordered_successor_tensor_select_v2_rows as v2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = ROOT / "basis_aligned/bilinear_quotient"
AMENDMENT = HERE / "ORDERED_SUCCESSOR_TENSOR_SELECT_V3_ROWS_AMENDMENT.md"
AUDIT = HERE / "ordered_successor_tensor_select_v3_rows_independent_audit.json"
CACHE = BQ / ".rowcache_ordered_successor_tensor_select_v3"
RECEIPT = BQ / "ordered_successor_tensor_select_v3_rows_receipt.json"
FAILURE = BQ / "ordered_successor_tensor_select_v3_rows_failure.json"
TERMINAL = BQ / "ordered_successor_tensor_select_v3_rows_terminal_claim.json"
LOCK = Path("/workspace/runs/.ordered_successor_tensor_select_v3_rows.lock")
MANIFEST_NAME = "select_manifest.json"
ROWS_NAME = "select_rows.pt"

N_SELECT = budget.V3_SELECT_DOCUMENTS
START_DOCUMENT_INDEX = v2.START_DOCUMENT_INDEX
CANDIDATE_DOCUMENTS = v2.CANDIDATE_DOCUMENTS
ROW_LENGTH = v2.ROW_LENGTH
PREFIX_LENGTH = v2.PREFIX_LENGTH
MIN_POSITIONS = v2.MIN_POSITIONS
MIN_DOCUMENTS = v2.MIN_DOCUMENTS
ARM_NAMES = protocol.ARM_NAMES
PAIR_NAMES = v2.PAIR_NAMES

FREEZER = Path(__file__).resolve()
TEST = HERE / "test_prepare_ordered_successor_tensor_select_v3_rows.py"
OWN_SOURCES = (
    AMENDMENT,
    HERE / "ordered_successor_tensor_select_v3_budget.py",
    HERE / "test_ordered_successor_tensor_select_v3_budget.py",
    FREEZER,
    TEST,
    budget.V2_AUDIT,
    budget.V2_FAILURE,
)
SOURCE_PATHS = tuple(dict.fromkeys((*OWN_SOURCES, *v2.SOURCE_PATHS)))


def file_sha256(path: Path) -> str:
    return v2.file_sha256(path)


def tensor_sha256(value: torch.Tensor) -> str:
    return v2.tensor_sha256(value)


def _stable_json(path: Path) -> tuple[dict[str, Any], str]:
    before = file_sha256(path)
    raw = path.read_bytes()
    after = file_sha256(path)
    if before != after or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"successor v3 JSON changed during stable read: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"successor v3 JSON root is not an object: {path}")
    return value, before


def source_closure(commit: str) -> dict[str, str]:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    answer: dict[str, str] = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"successor v3 row source drift: {relative}")
        answer[relative] = digest
    return answer


def validate_independent_audit(path: Path = AUDIT) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise RuntimeError("successor v3 independent row audit is absent")
    audit, digest = _stable_json(path)
    required = {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    }
    if (
        set(audit) != required
        or audit.get("schema") != "ordered_successor_tensor_select_v3_rows_independent_audit"
        or audit.get("status") != "GO"
        or audit.get("outcome_access") is not False
        or type(audit.get("tests_passed")) is not int
        or audit["tests_passed"] < 1
        or not isinstance(audit.get("reviewer"), str)
        or not audit["reviewer"]
    ):
        raise RuntimeError("successor v3 independent row audit is not an exact GO")
    commit = audit.get("audited_source_commit")
    hashes = audit.get("audited_source_hashes")
    if not isinstance(commit, str) or len(commit) != 40 or not isinstance(hashes, dict):
        raise RuntimeError("successor v3 independent audit binding is malformed")
    if source_closure(commit) != hashes:
        raise RuntimeError("successor v3 independent audit source closure changed")
    return audit, digest


class RunClaim(NamedTuple):
    descriptor: int
    device: int
    inode: int
    nonce: str
    path: Path


def acquire_claim(path: Path = LOCK) -> RunClaim:
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32)
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"successor v3 row namespace is locked: {path}") from error
    try:
        os.write(descriptor, (nonce + "\n").encode("ascii"))
        os.fsync(descriptor)
        stat = os.fstat(descriptor)
        return RunClaim(descriptor, stat.st_dev, stat.st_ino, nonce, path)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def require_claim(claim: RunClaim, path: Path | None = None) -> None:
    path = claim.path if path is None else path
    original = os.fstat(claim.descriptor)
    if (original.st_dev, original.st_ino) != (claim.device, claim.inode):
        raise RuntimeError("successor v3 row claim changed")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, OSError) as error:
        raise RuntimeError("successor v3 row claim changed") from error
    try:
        before = os.fstat(descriptor)
        raw = os.read(descriptor, 4096)
        overflow = os.read(descriptor, 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_stat = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise RuntimeError("successor v3 row claim changed") from error
    identity = (claim.device, claim.inode)
    if (
        (before.st_dev, before.st_ino) != identity
        or (after.st_dev, after.st_ino) != identity
        or (path_stat.st_dev, path_stat.st_ino) != identity
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or overflow
        or raw != (claim.nonce + "\n").encode("ascii")
    ):
        raise RuntimeError("successor v3 row claim changed")


def release_claim(claim: RunClaim, path: Path | None = None) -> None:
    path = claim.path if path is None else path
    try:
        try:
            require_claim(claim, path)
        except RuntimeError:
            pass
        else:
            path.unlink()
    finally:
        os.close(claim.descriptor)


def _terminal_absent() -> None:
    if RECEIPT.exists() or FAILURE.exists() or TERMINAL.exists():
        raise RuntimeError("successor v3 row terminal already exists")


def _v2_lineage_snapshot() -> dict[str, Any]:
    lineage = budget.validate_v2_lineage()
    state = {
        "lineage": lineage,
        "v2_cache_exists": v2.CACHE.exists(),
        "v2_receipt_exists": v2.RECEIPT.exists(),
        "v2_failure_exists": v2.FAILURE.exists(),
    }
    if state["v2_cache_exists"] or state["v2_receipt_exists"] or not state["v2_failure_exists"]:
        raise RuntimeError("spent successor v2 terminal namespace changed")
    return state


def allocate_powered_select(
    candidate_rows: torch.Tensor,
    candidate_records: list[dict[str, Any]],
    lexicon: Any,
    *,
    mask_builder: Callable[[torch.Tensor, Any], Any] = v2.build_masks,
) -> budget.BudgetAllocation:
    return budget.allocate_v3_budget(
        candidate_rows,
        candidate_records,
        lexicon,
        budget=N_SELECT,
        mask_builder=mask_builder,
        require_registered_stopping_count=True,
    )


def support_sha256(rows: torch.Tensor, lexicon: Any, masks: Any) -> str:
    if (
        not torch.is_tensor(rows)
        or rows.device.type != "cpu"
        or rows.dtype != torch.long
        or tuple(rows.shape) != (N_SELECT, ROW_LENGTH)
        or not rows.is_contiguous()
        or bool((rows < 0).any())
        or bool((rows >= 50_257).any())
    ):
        raise ValueError("successor v3 support rows are malformed")
    if lexicon.name != "decimal_digits_v2" or lexicon.items != digit_registry.DIGIT_TOKEN_IDS:
        raise ValueError("successor v3 support lexicon changed")
    masks.validate_partition()
    digest = hashlib.sha256()
    digest.update(tensor_sha256(rows).encode("ascii"))
    digest.update(digit_registry.REGISTRY_SHA256.encode("ascii"))
    digest.update(lexicon.name.encode("utf-8"))
    digest.update(tensor_sha256(masks.eligible_target.contiguous()).encode("ascii"))
    for name, mask in masks.named_cells().items():
        digest.update(name.encode("ascii"))
        digest.update(tensor_sha256(mask.contiguous()).encode("ascii"))
    digest.update(tensor_sha256(masks.pair_index.contiguous()).encode("ascii"))
    return digest.hexdigest()


def _mask_hashes(masks: Any) -> dict[str, str]:
    return {
        **{
            name: tensor_sha256(mask.contiguous())
            for name, mask in masks.named_cells().items()
        },
        "eligible_target": tensor_sha256(masks.eligible_target.contiguous()),
        "pair_index": tensor_sha256(masks.pair_index.contiguous()),
    }


def _validate_payload(path: Path, entry: Mapping[str, Any]) -> torch.Tensor:
    before = file_sha256(path)
    if before != entry.get("file_sha256"):
        raise RuntimeError("successor v3 row payload hash changed")
    rows = torch.load(path, map_location="cpu", weights_only=True)
    if (
        file_sha256(path) != before
        or not torch.is_tensor(rows)
        or rows.device.type != "cpu"
        or rows.dtype != torch.long
        or tuple(rows.shape) != (N_SELECT, ROW_LENGTH)
        or not rows.is_contiguous()
        or tensor_sha256(rows) != entry.get("tensor_sha256")
    ):
        raise RuntimeError("successor v3 row payload semantic replay failed")
    return rows


def _validate_manifest(path: Path, expected: Mapping[str, Any], digest: str) -> None:
    observed, actual = _stable_json(path)
    if actual != digest or observed != expected:
        raise RuntimeError("successor v3 row manifest semantic replay failed")


def _artifact_snapshot(entry: Mapping[str, Any], manifest_sha256: str) -> dict[str, str]:
    rows = Path(str(entry["path"]))
    manifest = CACHE / MANIFEST_NAME
    if not rows.is_file() or not manifest.is_file():
        raise RuntimeError("successor v3 installed artifact is absent")
    observed = {"rows": file_sha256(rows), "manifest": file_sha256(manifest)}
    expected = {"rows": str(entry["file_sha256"]), "manifest": manifest_sha256}
    if observed != expected:
        raise RuntimeError("successor v3 installed artifact hash changed")
    return observed


def discover_prior_registry_files() -> tuple[Path, ...]:
    """Exclude exactly this transaction's separately validated in-progress manifest."""

    own_manifest = (CACHE / MANIFEST_NAME).resolve()
    return tuple(
        path for path in v2.base.discover_registry_files()
        if path.resolve() != own_manifest
    )


def _verify_prior_registry_snapshot(
    *,
    registry_files: tuple[Path, ...],
    prior: Any,
    registry_hashes: Mapping[str, str],
    tensor_hashes: Mapping[str, str],
    waiver_proofs: list[dict[str, Any]],
    nonrow_proofs: list[dict[str, Any]],
    parquet: Path,
) -> None:
    current = discover_prior_registry_files()
    if current != registry_files:
        raise RuntimeError("successor v3 prior registry membership changed")
    replay = v2.base.load_registry_exclusions(current)
    if discover_prior_registry_files() != current:
        raise RuntimeError("successor v3 prior registry changed during replay")
    current_prior, current_registry, current_tensors, current_waivers, current_nonrows = replay
    if current_registry != dict(registry_hashes):
        raise RuntimeError("successor v3 prior registry bytes changed")
    if current_tensors != dict(tensor_hashes) or current_prior != prior:
        raise RuntimeError("successor v3 prior row exclusions changed")
    if current_waivers != waiver_proofs or current_nonrows != nonrow_proofs:
        raise RuntimeError("successor v3 prior registry classifications changed")
    if (
        parquet.stat().st_size != v2.base.BASE.local.PINNED_SIZE
        or file_sha256(parquet) != v2.base.BASE.local.PINNED_SHA256
    ):
        raise RuntimeError("successor v3 pinned FineWeb parquet changed")


def _source_identity(canonical: Mapping[str, Any], parquet: Path, encoding: Any) -> dict[str, Any]:
    return v2._source_identity(canonical, parquet, encoding)


def _protected_replay(
    *,
    commit: str,
    sources: Mapping[str, str],
    audit_sha256: str,
    v2_lineage: Mapping[str, Any],
    registry_files: tuple[Path, ...],
    prior: Any,
    registry_hashes: Mapping[str, str],
    tensor_hashes: Mapping[str, str],
    waiver_proofs: list[dict[str, Any]],
    nonrow_proofs: list[dict[str, Any]],
    canonical: Mapping[str, Any],
    parquet: Path,
    encoding: Any,
    source_identity: Mapping[str, Any],
) -> None:
    protocol.validate_registry()
    budget.prospective_status()
    if source_closure(commit) != dict(sources):
        raise RuntimeError("successor v3 sources changed")
    if validate_independent_audit()[1] != audit_sha256:
        raise RuntimeError("successor v3 audit changed")
    if _v2_lineage_snapshot() != dict(v2_lineage):
        raise RuntimeError("successor v2 parent lineage changed")
    _verify_prior_registry_snapshot(
        registry_files=registry_files,
        prior=prior,
        registry_hashes=registry_hashes,
        tensor_hashes=tensor_hashes,
        waiver_proofs=waiver_proofs,
        nonrow_proofs=nonrow_proofs,
        parquet=parquet,
    )
    digit_registry.validate_encoding(encoding)
    if _source_identity(canonical, parquet, encoding) != dict(source_identity):
        raise RuntimeError("successor v3 source/tokenizer identity changed")


def _stage_json(value: Mapping[str, Any], path: Path) -> tuple[dict[str, Any], Path, str]:
    normalized = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as sink:
            descriptor = -1
            sink.write(json.dumps(normalized, indent=2, allow_nan=False) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        replay, _ = _stable_json(temporary)
        if replay != normalized:
            raise RuntimeError("successor v3 terminal JSON replay failed")
        return normalized, temporary, file_sha256(temporary)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _late_fsync(path: Path) -> None:
    try:
        v2.base._fsync_directory(path.parent)
    except OSError:
        # A successful hard link is already the authoritative state transition.
        pass


def _publish_terminal_json(
    value: Mapping[str, Any], *, kind: str, claim: RunClaim,
    before_claim: Callable[[], None],
) -> None:
    """Win one shared claim, then link exactly one mutually exclusive terminal."""

    if kind not in ("receipt", "failure"):
        raise ValueError("successor v3 terminal kind is malformed")
    path = RECEIPT if kind == "receipt" else FAILURE
    opposite = FAILURE if kind == "receipt" else RECEIPT
    normalized, payload_tmp, payload_sha256 = _stage_json(value, path)
    claim_value = {
        "schema": "ordered_successor_tensor_select_v3_rows_terminal_claim",
        "status": f"{kind}_claimed_before_terminal_link",
        "kind": kind,
        "target_path": str(path.resolve()),
        "payload_sha256": payload_sha256,
    }
    _normalized_claim, claim_tmp, claim_sha256 = _stage_json(claim_value, TERMINAL)
    try:
        before_claim()
        require_claim(claim)
        _terminal_absent()
        # This single O_EXCL hard-link is the serialization point. A competing
        # receipt/failure publisher must lose here before it can link its payload.
        os.link(claim_tmp, TERMINAL)
        _late_fsync(TERMINAL)
        # The staged payload was semantically replayed before the shared claim.
        # There are no fallible callbacks after the serialization point.
        if file_sha256(TERMINAL) != claim_sha256 or opposite.exists() or path.exists():
            raise RuntimeError("successor v3 terminal aggregate changed after shared claim")
        os.link(payload_tmp, path)
        _late_fsync(path)
    finally:
        try:
            payload_tmp.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            claim_tmp.unlink(missing_ok=True)
        except OSError:
            pass


def freeze_locked(claim: RunClaim) -> dict[str, Any]:
    require_claim(claim)
    _terminal_absent()
    if CACHE.exists():
        raise RuntimeError("successor v3 row cache already exists")
    audit, audit_sha256 = validate_independent_audit()
    commit = audit["audited_source_commit"]
    sources = source_closure(commit)
    v2_lineage = _v2_lineage_snapshot()
    protocol.validate_registry()
    lexicon, encoding = digit_registry.load_pinned_lexicon()
    canonical, parquet = v2.base.BASE.validate_ordered_source()
    registry_files = discover_prior_registry_files()
    prior, registry_hashes, tensor_hashes, waiver_proofs, nonrow_proofs = (
        v2.base.load_registry_exclusions(registry_files)
    )
    if prior[1] and max(prior[1]) >= START_DOCUMENT_INDEX:
        raise RuntimeError("successor v3 start is not beyond every historical dataset index")
    identity = _source_identity(canonical, parquet, encoding)
    candidates, candidate_records = v2.base.harvest_fresh_documents(
        v2.base.BASE.local.parquet_texts([parquet]),
        encoding.encode_ordinary,
        prior,
        start_document_index=START_DOCUMENT_INDEX,
        n_source_documents=CANDIDATE_DOCUMENTS,
        token_length=ROW_LENGTH,
    )
    allocation = allocate_powered_select(candidates, candidate_records, lexicon)
    rows = allocation.selected_rows
    records = list(allocation.selected_records)
    masks = v2.build_masks(rows, lexicon)
    census = dict(allocation.census)
    pair_census = dict(allocation.pair_occupancy)
    disjointness = v2.base.validate_disjointness(rows, records, prior)
    support_digest = support_sha256(rows, lexicon, masks)
    manifest: dict[str, Any] = {
        "schema": "ordered_successor_tensor_select_v3_rows_manifest",
        "role": "SELECT",
        "document_records": records,
        "support_first_count": allocation.support_first_count,
        "support_first_last_candidate": allocation.support_first_last_candidate,
        "powered_census": census,
        "mask_hashes": _mask_hashes(masks),
        "support_sha256": support_digest,
        "pair_names": list(PAIR_NAMES),
        "pair_occupancy": pair_census,
        "lexicon_registry_sha256": digit_registry.REGISTRY_SHA256,
        "protocol_registry_sha256": protocol.REGISTRY_SHA256,
        "arm_names": list(ARM_NAMES),
        "v2_lineage": v2_lineage,
    }
    staging = CACHE.with_name(f".{CACHE.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    staging.mkdir(parents=False, exist_ok=False)
    try:
        row_path = staging / ROWS_NAME
        torch.save(rows, row_path)
        with row_path.open("rb") as handle:
            os.fsync(handle.fileno())
        entry = {
            "path": str((CACHE / ROWS_NAME).resolve()),
            "shape": [N_SELECT, ROW_LENGTH],
            "dtype": "torch.int64",
            "file_sha256": file_sha256(row_path),
            "tensor_sha256": tensor_sha256(rows),
        }
        manifest["row_entry"] = entry
        manifest_path = staging / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        with manifest_path.open("rb") as handle:
            os.fsync(handle.fileno())
        manifest_sha256 = file_sha256(manifest_path)
        _validate_payload(row_path, entry)
        _validate_manifest(manifest_path, manifest, manifest_sha256)
        _protected_replay(
            commit=commit,
            sources=sources,
            audit_sha256=audit_sha256,
            v2_lineage=v2_lineage,
            registry_files=registry_files,
            prior=prior,
            registry_hashes=registry_hashes,
            tensor_hashes=tensor_hashes,
            waiver_proofs=waiver_proofs,
            nonrow_proofs=nonrow_proofs,
            canonical=canonical,
            parquet=parquet,
            encoding=encoding,
            source_identity=identity,
        )
        _terminal_absent()
        if CACHE.exists():
            raise RuntimeError("successor v3 row cache appeared before install")
        require_claim(claim)
        v2.base._fsync_directory(staging)
        os.replace(staging, CACHE)
        v2.base._fsync_directory(CACHE.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    installed_rows = _validate_payload(Path(entry["path"]), entry)
    installed_manifest = CACHE / MANIFEST_NAME
    _validate_manifest(installed_manifest, manifest, manifest_sha256)
    replay_candidates, replay_records = v2.base.harvest_fresh_documents(
        v2.base.BASE.local.parquet_texts([parquet]),
        encoding.encode_ordinary,
        prior,
        start_document_index=START_DOCUMENT_INDEX,
        n_source_documents=CANDIDATE_DOCUMENTS,
        token_length=ROW_LENGTH,
    )
    replay = allocate_powered_select(replay_candidates, replay_records, lexicon)
    replay_masks = v2.build_masks(replay.selected_rows, lexicon)
    if (
        not torch.equal(replay.selected_rows, installed_rows)
        or list(replay.selected_records) != records
        or dict(replay.census) != census
        or dict(replay.pair_occupancy) != pair_census
        or _mask_hashes(replay_masks) != manifest["mask_hashes"]
        or replay.support_first_count != allocation.support_first_count
        or replay.support_first_last_candidate != allocation.support_first_last_candidate
    ):
        raise RuntimeError("successor v3 deterministic allocation replay changed")

    receipt = {
        "schema": "ordered_successor_tensor_select_v3_rows_receipt",
        "status": "frozen_before_any_successor_select_model_forward",
        "authorized_role": "SELECT",
        "authorized_for_training": False,
        "source_commit": commit,
        "source_hashes": sources,
        "independent_audit": {"path": str(AUDIT), "file_sha256": audit_sha256},
        "v2_lineage": v2_lineage,
        "arm_names": list(ARM_NAMES),
        "protocol_registry_sha256": protocol.REGISTRY_SHA256,
        "selection": {
            "start_dataset_document_index": START_DOCUMENT_INDEX,
            "candidate_documents": CANDIDATE_DOCUMENTS,
            "select_documents": N_SELECT,
            "rows_per_document": 1,
            "row_length": ROW_LENGTH,
            "scored_positions": [64, 256],
            "algorithm": "support_first_then_earliest_unused",
            "support_first_count": allocation.support_first_count,
            "support_first_last_candidate": allocation.support_first_last_candidate,
            "budget_margin_documents": budget.V3_MARGIN_DOCUMENTS,
        },
        "entry": entry,
        "manifest": {"path": str(installed_manifest), "file_sha256": manifest_sha256},
        "source_identity": identity,
        "registry_files": registry_hashes,
        "prior_row_tensors": tensor_hashes,
        "disjointness": disjointness,
        "historical_max_dataset_document_index": max(prior[1]) if prior[1] else None,
        "powered_census": census,
        "pair_names": list(PAIR_NAMES),
        "pair_occupancy": pair_census,
        "support_sha256": support_digest,
        "failed_unmaterialized_registry_waivers": waiver_proofs,
        "exact_nonrow_registry_artifacts": nonrow_proofs,
        "outcome_access": {
            "model_imported": False,
            "checkpoint_loaded": False,
            "model_forward_calls": 0,
            "scientific_outcomes_read": False,
        },
        "terminal_claim": {"path": str(TERMINAL), "kind": "receipt"},
        "artifact_order": ["rows", "manifest", "shared_terminal_claim", "receipt_last"],
    }

    def final_guard() -> None:
        before = _artifact_snapshot(entry, manifest_sha256)
        _validate_payload(Path(entry["path"]), entry)
        _validate_manifest(installed_manifest, manifest, manifest_sha256)
        _protected_replay(
            commit=commit,
            sources=sources,
            audit_sha256=audit_sha256,
            v2_lineage=v2_lineage,
            registry_files=registry_files,
            prior=prior,
            registry_hashes=registry_hashes,
            tensor_hashes=tensor_hashes,
            waiver_proofs=waiver_proofs,
            nonrow_proofs=nonrow_proofs,
            canonical=canonical,
            parquet=parquet,
            encoding=encoding,
            source_identity=identity,
        )
        if _artifact_snapshot(entry, manifest_sha256) != before:
            raise RuntimeError("successor v3 installed artifacts changed during final replay")
        _terminal_absent()
        require_claim(claim)

    _publish_terminal_json(
        receipt, kind="receipt", claim=claim, before_claim=final_guard,
    )
    return receipt


def freeze() -> dict[str, Any]:
    claim = acquire_claim()
    try:
        return freeze_locked(claim)
    except BaseException as error:
        if not RECEIPT.exists() and not FAILURE.exists() and not TERMINAL.exists():
            failure = {
                "schema": "ordered_successor_tensor_select_v3_rows_failure",
                "status": "terminal_failure_no_receipt",
                "error_type": type(error).__name__,
                "error": str(error),
                "cache_exists": CACHE.exists(),
                "outcome_access": False,
            }
            try:
                def failure_guard() -> None:
                    _terminal_absent()
                    if CACHE.exists() is not failure["cache_exists"]:
                        raise RuntimeError("successor v3 failure cache state changed")
                    require_claim(claim)

                _publish_terminal_json(
                    failure, kind="failure", claim=claim, before_claim=failure_guard,
                )
            except BaseException:
                pass
        raise
    finally:
        release_claim(claim)


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
