"""CPU-only lifecycle contract for the frozen early-MLP suffix-transport experiment.

This module is intentionally unable to import the model or run a forward pass.  It
owns fresh-role licenses, source-content closure, create-only artifacts, protected
snapshots, and the final-attempt/terminal-authority ordering.  Numerical modules may
depend on it; it must never depend on them.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
RUN_LOCK = Path("/workspace/runs/.early_mlp_suffix_transport_v1.lock")

PREREGISTRATION = HERE / "EARLY_MLP_SUFFIX_TRANSPORT_V1_PREREGISTRATION.md"
IMPLEMENTATION_AMENDMENT = (
    HERE / "EARLY_MLP_SUFFIX_TRANSPORT_V1_IMPLEMENTATION_AMENDMENT.md"
)
PURE_CONTRACT = HERE / "early_mlp_suffix_transport_v1.py"
PURE_TEST = HERE / "test_early_mlp_suffix_transport_v1.py"
OBSERVED_EXECUTION_CLOSURE = (
    HERE / "bilin18_observed_model_facade.py",
    HERE / "test_bilin18_observed_model_facade.py",
    HERE / "bilin18_frozen_ship_program.py",
    HERE / "test_bilin18_frozen_ship_program.py",
    HERE / "bilin18_observed_adapter.py",
    HERE / "test_bilin18_observed_adapter.py",
)
# These stage owners must exist, be tested, be tracked, and be part of the frozen
# source closure before a fresh row role may be materialized.  Naming them here is
# intentionally a fail-closed implementation gate: fit data must not be exposed and
# then used to finish the programs or final evaluator outcome-adaptively.
NUMERICAL_STAGE_CLOSURE = (
    HERE / "early_mlp_suffix_transport_v1_fit.py",
    HERE / "test_early_mlp_suffix_transport_v1_fit.py",
    HERE / "early_mlp_suffix_transport_v1_programs.py",
    HERE / "test_early_mlp_suffix_transport_v1_programs.py",
    HERE / "early_mlp_suffix_transport_v1_final.py",
    HERE / "test_early_mlp_suffix_transport_v1_final.py",
)
FROZEN_SHA256 = {
    PREREGISTRATION: "11577380d65c813cf9e80e92002de9569928d293747c278c065939b3f3b24193",
    IMPLEMENTATION_AMENDMENT:
        "f4d019352c9443cbbea3f1f78a025fa94e0ba51c5c3a91e33d81a141b0c6e4a7",
    PURE_CONTRACT: "11a2e05057ae8c3b4e8fd397635cbc1c7be8327e53b1cdcaa825a37cc70d2339",
    PURE_TEST: "53a876dc4595893929b3a415a602f674d83d4edac1cfafab1c882fcf2c6732da",
}
SOURCE_CLOSURE = (
    PREREGISTRATION,
    IMPLEMENTATION_AMENDMENT,
    HERE / "EARLY_MLP_SUFFIX_TRANSPORT_V1_N_WRITE_SEMANTICS.md",
    HERE / "EARLY_MLP_SUFFIX_TRANSPORT_V1_GRAPH_IDENTITY.md",
    PURE_CONTRACT,
    PURE_TEST,
    Path(__file__),
    HERE / "test_early_mlp_suffix_transport_v1_lifecycle.py",
    HERE / "early_mlp_suffix_transport_v1_statistics.py",
    HERE / "test_early_mlp_suffix_transport_v1_statistics.py",
    HERE / "early_mlp_suffix_transport_v1_rows.py",
    HERE / "test_early_mlp_suffix_transport_v1_rows.py",
    HERE / "early_mlp_suffix_transport_v1_row_freezer.py",
    HERE / "test_early_mlp_suffix_transport_v1_row_freezer.py",
    HERE / "early_mlp_suffix_transport_v1_runtime.py",
    HERE / "test_early_mlp_suffix_transport_v1_runtime.py",
    HERE / "early_mlp_suffix_transport_v1_inherited.py",
    HERE / "test_early_mlp_suffix_transport_v1_inherited.py",
    HERE / "early_mlp_suffix_transport_v1_capabilities.py",
    HERE / "test_early_mlp_suffix_transport_v1_capabilities.py",
    *OBSERVED_EXECUTION_CLOSURE,
    *NUMERICAL_STAGE_CLOSURE,
)

ROLE_NAMES = (
    "early_mlp_suffix_transport_v1_fit",
    "early_mlp_suffix_transport_v1_validation",
    "early_mlp_suffix_transport_v1_final",
)
ROLE_LICENSES = {
    ROLE_NAMES[0]: {"training": True, "selection": False, "final_scoring": False},
    ROLE_NAMES[1]: {"training": False, "selection": True, "final_scoring": False},
    ROLE_NAMES[2]: {
        "training": False,
        "selection": False,
        "final_scoring": True,
        "requires_programs_unlock": True,
    },
}


@dataclass(frozen=True)
class CandidateTriple:
    index: int
    fit_n: int
    fit_skip: int
    validation_n: int
    validation_skip: int
    final_n: int
    final_skip: int


def candidate_triple(index: int) -> CandidateTriple:
    if not isinstance(index, int) or index < 0:
        raise ValueError("candidate index must be a nonnegative integer")
    return CandidateTriple(
        index=index,
        fit_n=384,
        fit_skip=43000 + 12000 * index,
        validation_n=192,
        validation_skip=47000 + 12000 * index,
        final_n=192,
        final_skip=51000 + 12000 * index,
    )


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path = BQ

    @property
    def cache(self) -> Path:
        return self.root / ".rowcache_early_mlp_suffix_transport_v1"

    @property
    def rows_receipt(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_rows_receipt.json"

    @property
    def rows_manifest(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_rows_manifest.json"

    @property
    def collision_manifest(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_rows_collision_manifest.json"

    @property
    def fit_ledger(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_fit_ledger.pt"

    @property
    def fit_manifest(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_fit_manifest.json"

    @property
    def fit_receipt(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_fit_receipt.json"

    @property
    def programs(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_programs.pt"

    @property
    def programs_receipt(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_programs_receipt.json"

    @property
    def final_attempt(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_final_attempt.json"

    @property
    def final_result(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_final_result.pt"

    @property
    def final_manifest(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_final_manifest.json"

    @property
    def final_authority(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_final_authority.json"

    @property
    def integrity_failure(self) -> Path:
        return self.root / "early_mlp_suffix_transport_v1_integrity_failure.json"

    def output_files(self) -> tuple[Path, ...]:
        return (
            self.rows_receipt, self.rows_manifest, self.collision_manifest,
            self.fit_ledger, self.fit_manifest, self.fit_receipt,
            self.programs, self.programs_receipt, self.final_attempt,
            self.final_result, self.final_manifest, self.final_authority,
            self.integrity_failure,
        )

    def assert_stage_preconditions(self, stage: str) -> None:
        required: tuple[Path, ...]
        forbidden: tuple[Path, ...]
        if stage == "rows":
            required = ()
            forbidden = self.output_files()
            if self.cache.exists():
                raise RuntimeError("row cache namespace already exists")
        elif stage == "fit":
            required = (self.rows_receipt, self.rows_manifest)
            forbidden = self.output_files()[3:]
        elif stage == "programs":
            required = (self.fit_ledger, self.fit_manifest, self.fit_receipt)
            forbidden = self.output_files()[6:]
        elif stage == "final_attempt":
            required = (self.programs, self.programs_receipt)
            forbidden = self.output_files()[8:]
        elif stage == "final_authority":
            required = (self.final_attempt, self.final_result, self.final_manifest)
            forbidden = (self.final_authority, self.integrity_failure)
        else:
            raise ValueError(f"unknown lifecycle stage: {stage}")
        missing = [str(path) for path in required if not path.is_file()]
        existing = [str(path) for path in forbidden if path.exists()]
        if missing or existing:
            raise RuntimeError(
                f"stage {stage} ordering failed; missing={missing}; existing={existing}"
            )


PATHS = ArtifactPaths()
_FINAL_ROLE_LOADS = 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def logical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("tensor_sha256 requires a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def artifact_binding(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"binding target is absent: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")


def atomic_create_bytes(data: bytes, path: Path) -> None:
    """Atomically publish bytes without any overwrite window."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite create-only artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def atomic_create_json(value: Mapping[str, Any], path: Path) -> None:
    atomic_create_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"), path)


def atomic_create_torch(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite create-only artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_run_claim(lock_path: Path = RUN_LOCK) -> Iterator[str]:
    nonce = secrets.token_hex(32)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("suffix-transport namespace is already claimed") from error
    with os.fdopen(descriptor, "w") as handle:
        handle.write(nonce)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        require_run_claim(nonce, lock_path)
        yield nonce
        require_run_claim(nonce, lock_path)
    finally:
        if lock_path.is_file() and lock_path.read_text() == nonce:
            lock_path.unlink()


def require_run_claim(nonce: str, lock_path: Path = RUN_LOCK) -> None:
    if not lock_path.is_file() or lock_path.read_text() != nonce:
        raise RuntimeError("suffix-transport run claim is absent or not owned")


def verify_frozen_inputs() -> None:
    for path, expected in FROZEN_SHA256.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"frozen suffix-transport input changed: {path}")


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=check, capture_output=True, text=True,
    )


def source_closure_paths() -> tuple[Path, ...]:
    return tuple(dict.fromkeys(path.resolve() for path in SOURCE_CLOSURE))


def require_numerical_source_closure() -> tuple[Path, ...]:
    """Refuse row exposure until the complete real numerical pipeline is frozen.

    Checking both filesystem and git membership catches two distinct mistakes:
    implementing a stage without binding it into the experiment, and declaring a
    future stage name without actually implementing and committing it.
    """

    required = tuple(
        dict.fromkeys(path.resolve() for path in (
            *OBSERVED_EXECUTION_CLOSURE, *NUMERICAL_STAGE_CLOSURE,
        ))
    )
    closure = set(source_closure_paths())
    omitted = [str(path.relative_to(ROOT.resolve())) for path in required if path not in closure]
    absent = [str(path.relative_to(ROOT.resolve())) for path in required if not path.is_file()]
    untracked = [
        str(path.relative_to(ROOT.resolve())) for path in required
        if path.is_file() and _git(
            "ls-files", "--error-unmatch", "--",
            str(path.relative_to(ROOT.resolve())), check=False,
        ).returncode
    ]
    if omitted or absent or untracked:
        raise RuntimeError(
            "suffix-transport numerical source closure is incomplete; "
            f"omitted={omitted}; absent={absent}; untracked={untracked}"
        )
    return required


def _source_relative_paths() -> tuple[str, ...]:
    return tuple(
        str(path.relative_to(ROOT.resolve())) for path in source_closure_paths()
    )


def freeze_source_closure(*, require_origin: bool = True) -> dict[str, Any]:
    """Bind exact committed source bytes, optionally requiring pushed HEAD."""

    verify_frozen_inputs()
    resolved = source_closure_paths()
    if not resolved:
        raise RuntimeError("source closure is empty")
    source_hashes: dict[str, str] = {}
    for path in resolved:
        relative = str(path.relative_to(ROOT.resolve()))
        if _git("ls-files", "--error-unmatch", "--", relative, check=False).returncode:
            raise RuntimeError(f"source closure path is not tracked: {relative}")
        if _git("diff", "--quiet", "HEAD", "--", relative, check=False).returncode or (
            _git("diff", "--cached", "--quiet", "HEAD", "--", relative, check=False).returncode
        ):
            raise RuntimeError(f"source closure path is dirty: {relative}")
        source_hashes[relative] = file_sha256(path)
    commit = _git("rev-parse", "HEAD").stdout.strip()
    if require_origin and commit != _git("rev-parse", "origin/main").stdout.strip():
        raise RuntimeError("row authority requires the source commit pushed to origin/main")
    verify_source_closure(commit, source_hashes, require_current=True)
    return {"source_commit": commit, "source_hashes": source_hashes}


def verify_source_closure(
    source_commit: str, source_hashes: Mapping[str, str], *, require_current: bool = True,
) -> None:
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise RuntimeError("source commit is malformed")
    if _git("cat-file", "-e", f"{source_commit}^{{commit}}", check=False).returncode:
        raise RuntimeError("source commit is unavailable")
    if set(source_hashes) != set(_source_relative_paths()):
        raise RuntimeError("source closure path set is incomplete or changed")
    for relative, expected in sorted(source_hashes.items()):
        blob = _git("show", f"{source_commit}:{relative}", check=False)
        if blob.returncode or hashlib.sha256(blob.stdout.encode()).hexdigest() != expected:
            # Text-mode subprocess can transform bytes. Fall back to the literal blob.
            raw = subprocess.run(
                ["git", "show", f"{source_commit}:{relative}"], cwd=ROOT,
                check=False, capture_output=True,
            )
            if raw.returncode or hashlib.sha256(raw.stdout).hexdigest() != expected:
                raise RuntimeError(f"source content does not match commit: {relative}")
        if require_current:
            current = ROOT / relative
            if not current.is_file() or file_sha256(current) != expected:
                raise RuntimeError(f"current source content drifted: {relative}")


def protected_snapshot(paths: Iterable[Path]) -> dict[str, Any]:
    return {
        str(path.resolve()): (
            artifact_binding(path) if path.is_file() else {"path": str(path.resolve()), "absent": True}
        )
        for path in paths
    }


def require_protected_snapshot(paths: Iterable[Path], expected: Mapping[str, Any]) -> None:
    if protected_snapshot(paths) != dict(expected):
        raise RuntimeError("protected suffix-transport inputs drifted")


@dataclass(frozen=True)
class RoleIdentity:
    documents: frozenset[str]
    full_rows: frozenset[str]
    prefixes32: frozenset[str]


def role_identity(rows: torch.Tensor, records: Sequence[Mapping[str, Any]]) -> RoleIdentity:
    if not torch.is_tensor(rows) or rows.ndim != 2 or rows.shape[1] < 32 or (
        rows.dtype != torch.long
    ) or len(records) != len(rows):
        raise ValueError("role rows/provenance are malformed")
    documents = []
    for record in records:
        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("role provenance lacks document_id")
        documents.append(document_id)
    return RoleIdentity(
        documents=frozenset(documents),
        full_rows=frozenset(tensor_sha256(row) for row in rows),
        prefixes32=frozenset(tensor_sha256(row[:32]) for row in rows),
    )


def collision_report(
    candidates: Mapping[str, RoleIdentity], prior: Mapping[str, RoleIdentity],
) -> dict[str, Any]:
    combined = {**prior, **candidates}
    collisions = []
    names = sorted(combined)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            if left in prior and right in prior:
                continue
            a, b = combined[left], combined[right]
            counts = {
                "documents": len(a.documents & b.documents),
                "full_rows": len(a.full_rows & b.full_rows),
                "prefixes32": len(a.prefixes32 & b.prefixes32),
            }
            if any(counts.values()):
                collisions.append({"left": left, "right": right, **counts})
    return {"collision_free": not collisions, "collisions": collisions}


def _validate_rows_receipt(
    receipt: Mapping[str, Any], paths: ArtifactPaths = PATHS,
) -> None:
    """Reconstruct and validate the complete CPU row transaction before a role load."""
    import early_mlp_suffix_transport_v1_row_freezer as freezer
    import early_mlp_suffix_transport_v1_rows as row_contract

    exact_keys = {
        "schema_version", "status", "authority", "authorized_for_scored_experiments",
        "authorized_for_training", "chosen_candidate_index", "chosen_decision",
        "collision_history_sha256", "collision_manifest", "entries",
        "role_record_counts", "role_record_hashes", "role_licenses",
        "registry_census", "source_closure", "source_identity", "receipt_kind",
        "rows_manifest", "document_provenance",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != exact_keys:
        raise RuntimeError("row receipt schema changed")
    fixed = {
        "schema_version": 1,
        "status": "row_roles_frozen_before_any_model_forward",
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "receipt_kind": "early_mlp_suffix_transport_v1_rows",
        "role_licenses": ROLE_LICENSES,
    }
    if any(receipt.get(key) != value for key, value in fixed.items()):
        raise RuntimeError("row receipt fixed fields changed")
    source_closure = receipt["source_closure"]
    if not isinstance(source_closure, Mapping) or set(source_closure) != {
        "source_commit", "source_hashes",
    }:
        raise RuntimeError("row receipt source closure schema changed")
    verify_source_closure(
        source_closure["source_commit"], source_closure["source_hashes"],
    )

    # Revalidate current canonical data/protected identities without requiring that
    # unrelated later commits leave HEAD equal to the row-publication commit.
    gate, source = freezer.validate_ordered_source()
    _, census = row_contract.load_canonical_prior()
    import tiktoken
    source_identity = freezer._source_identity(
        gate, source, tiktoken.get_encoding("gpt2"),
    )
    if receipt["registry_census"] != census or receipt["source_identity"] != source_identity:
        raise RuntimeError("row receipt canonical registry/source identity changed")

    candidate_index = receipt["chosen_candidate_index"]
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) \
            or candidate_index < 0:
        raise RuntimeError("row receipt candidate index is malformed")
    chosen = receipt["chosen_decision"]
    row_contract.validate_collision_report(chosen, candidate_index)
    if chosen["accepted"] is not True:
        raise RuntimeError("row receipt chosen decision is not accepted")

    collision_binding = receipt["collision_manifest"]
    if candidate_index == 0:
        expected_absent = {
            "path": str(paths.collision_manifest.resolve()), "absent": True,
        }
        if collision_binding != expected_absent or paths.collision_manifest.exists():
            raise RuntimeError("unexpected collision manifest for candidate zero")
        history = [chosen]
    else:
        if collision_binding != artifact_binding(paths.collision_manifest):
            raise RuntimeError("collision manifest binding changed")
        collision = json.loads(paths.collision_manifest.read_text())
        if not isinstance(collision, Mapping) or set(collision) != {
            "schema_version", "status", "authority", "reports", "reports_sha256",
        } or collision.get("schema_version") != 1 or collision.get(
            "status"
        ) != "rejected_candidates_hash_only" or collision.get("authority") != "none":
            raise RuntimeError("collision manifest schema changed")
        reports = collision["reports"]
        if not isinstance(reports, list) or len(reports) != candidate_index \
                or collision["reports_sha256"] != logical_json_sha256(reports):
            raise RuntimeError("collision manifest reports changed")
        history = [*reports, chosen]
    row_contract.validate_collision_history(history, candidate_index)
    if receipt["collision_history_sha256"] != row_contract.collision_history_hash(history):
        raise RuntimeError("collision history binding changed")

    entries = receipt["entries"]
    if not isinstance(entries, Mapping) or set(entries) != set(ROLE_NAMES):
        raise RuntimeError("row receipt entries changed")
    provenance = receipt["document_provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "schema_version", "sets",
    } or provenance["schema_version"] != 1 or not isinstance(
        provenance["sets"], Mapping
    ) or set(provenance["sets"]) != set(ROLE_NAMES):
        raise RuntimeError("row receipt provenance schema changed")
    if not isinstance(receipt["role_record_counts"], Mapping) or set(
        receipt["role_record_counts"]
    ) != set(ROLE_NAMES) or not isinstance(receipt["role_record_hashes"], Mapping) \
            or set(receipt["role_record_hashes"]) != set(ROLE_NAMES):
        raise RuntimeError("row receipt provenance census schema changed")
    short_roles = dict(zip(ROLE_NAMES, row_contract.ROLES, strict=True))
    for role in ROLE_NAMES:
        entry = entries[role]
        if not isinstance(entry, Mapping) or set(entry) != {
            "cache_path", "cache_file_sha256", "shape_full",
            "tensor_full_raw_sha256", "tensor_bytes_raw_sha256",
        }:
            raise RuntimeError(f"row entry schema changed: {role}")
        short = short_roles[role]
        expected_name = freezer.expected_filename(short, candidate_index)
        cache_path = Path(entry["cache_path"])
        expected_shape = chosen["candidate"][short]["n"]
        records = provenance["sets"][role]
        if cache_path != paths.cache / expected_name or not isinstance(records, list) \
                or any(not isinstance(record, Mapping) or set(record) != {
                    "document_id", "dataset_document_index", "chunk_id", "token_start",
                } for record in records) or entry["shape_full"] != [
                    expected_shape, row_contract.TOKEN_LENGTH,
                ] or receipt["role_record_counts"][role] != len(records) \
                or len(records) != expected_shape or receipt["role_record_hashes"][
                    role
                ] != logical_json_sha256(records):
            raise RuntimeError(f"row entry/provenance binding changed: {role}")
        hashes = chosen["role_identity_hashes"][short]
        ordered_provenance = [
            [
                record["document_id"], record["dataset_document_index"],
                record["chunk_id"], record["token_start"],
            ]
            for record in records
        ]
        if hashes["ordered_tensor_raw"] != entry["tensor_bytes_raw_sha256"] \
                or hashes["ordered_provenance"] != logical_json_sha256(
                    ordered_provenance
                ):
            raise RuntimeError(f"row decision binding changed: {role}")

    if receipt["rows_manifest"] != artifact_binding(paths.rows_manifest):
        raise RuntimeError("rows manifest binding changed")
    manifest = json.loads(paths.rows_manifest.read_text())
    expected_manifest = dict(receipt)
    for key in ("receipt_kind", "rows_manifest", "document_provenance"):
        expected_manifest.pop(key)
    expected_manifest["status"] = "rows_frozen_before_any_model_forward"
    if manifest != expected_manifest:
        raise RuntimeError("rows manifest content changed")


def _expected_final_cache(receipt: Mapping[str, Any]) -> dict[str, Any]:
    entry = receipt["entries"][ROLE_NAMES[2]]
    path = Path(entry["cache_path"])
    return {
        "path": str(path.resolve()),
        "sha256": entry["cache_file_sha256"],
        "bytes": path.stat().st_size,
        "shape_full": list(entry["shape_full"]),
        "tensor_full_raw_sha256": entry["tensor_full_raw_sha256"],
    }


def load_programs_unlock(paths: ArtifactPaths = PATHS) -> dict[str, Any]:
    if not paths.programs_receipt.is_file() or not paths.programs.is_file():
        raise RuntimeError("canonical programs unlock is absent")
    receipt = json.loads(paths.programs_receipt.read_text())
    required = {
        "schema_version": 1,
        "status": "frozen_programs_before_final",
        "authority": "early_mlp_suffix_transport_v1_programs_unlock",
        "authorized_for_final_scoring": True,
        "rows_receipt": artifact_binding(paths.rows_receipt),
        "programs": artifact_binding(paths.programs),
    }
    if set(receipt) != set(required) | {
        "source_commit", "source_hashes", "protected_before",
    } or any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeError("canonical programs unlock binding changed")
    verify_source_closure(receipt["source_commit"], receipt["source_hashes"])
    return dict(receipt)


def _validate_final_attempt(
    paths: ArtifactPaths, rows_receipt: Mapping[str, Any], unlock: Mapping[str, Any],
    lock_nonce: str,
) -> dict[str, Any]:
    if not paths.final_attempt.is_file():
        raise RuntimeError("final attempt must exist before final deserialization")
    attempt = json.loads(paths.final_attempt.read_text())
    required = {
        "schema_version": 1,
        "status": "attempt_frozen_before_final_deserialization",
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "authorized_for_final_scoring": False,
        "requested_role": ROLE_NAMES[2],
        "rows_receipt": artifact_binding(paths.rows_receipt),
        "programs": artifact_binding(paths.programs),
        "programs_receipt": artifact_binding(paths.programs_receipt),
        "final_cache": _expected_final_cache(rows_receipt),
        "source_commit": unlock["source_commit"],
        "source_hashes": unlock["source_hashes"],
        "protected_before": unlock["protected_before"],
        "lock_nonce": lock_nonce,
        "final_role_loads_before_attempt": 0,
        "final_evaluations_before_attempt": 0,
    }
    if attempt != required:
        raise RuntimeError("final attempt binding changed")
    return dict(attempt)


def load_roles(
    requested: Sequence[str], *, operation: str, lock_nonce: str,
    paths: ArtifactPaths = PATHS, lock_path: Path = RUN_LOCK,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Validate every entry but deserialize only explicitly requested roles."""

    global _FINAL_ROLE_LOADS
    require_run_claim(lock_nonce, lock_path)
    receipt = json.loads(paths.rows_receipt.read_text())
    _validate_rows_receipt(receipt, paths)
    if not requested or len(set(requested)) != len(requested) or any(
        role not in ROLE_NAMES for role in requested
    ):
        raise ValueError("invalid or duplicate requested role")
    allowed = {
        "training": (ROLE_NAMES[0],),
        "selection": (ROLE_NAMES[1],),
        "final": (ROLE_NAMES[2],),
    }
    if operation not in allowed or tuple(requested) != allowed[operation]:
        raise RuntimeError("requested roles violate the operation license")
    if operation == "training":
        paths.assert_stage_preconditions("fit")
    elif operation == "selection":
        paths.assert_stage_preconditions("programs")
    else:
        if any(path.exists() for path in (
            paths.final_result, paths.final_manifest, paths.final_authority,
            paths.integrity_failure,
        )):
            raise RuntimeError("final namespace is already spent")
        unlock = load_programs_unlock(paths)
        _validate_final_attempt(paths, receipt, unlock, lock_nonce)
        if _FINAL_ROLE_LOADS != 0:
            raise RuntimeError("final role may be deserialized exactly once")
        _FINAL_ROLE_LOADS += 1
    loaded: dict[str, torch.Tensor] = {}
    for role, entry in receipt["entries"].items():
        path = Path(entry["cache_path"])
        if not path.is_file() or file_sha256(path) != entry["cache_file_sha256"]:
            raise RuntimeError(f"row cache binding changed: {role}")
        if role not in requested:
            continue
        value = torch.load(path, map_location="cpu", weights_only=True)
        value = value["rows"] if isinstance(value, Mapping) and "rows" in value else value
        if not torch.is_tensor(value) or value.dtype != torch.long or tuple(
            value.shape
        ) != tuple(entry["shape_full"]) or tensor_sha256(value) != entry[
            "tensor_full_raw_sha256"
        ]:
            raise RuntimeError(f"row cache tensor changed: {role}")
        loaded[role] = value.contiguous()
    return dict(receipt), loaded


def write_final_attempt(
    *, paths: ArtifactPaths, source_closure: Mapping[str, Any],
    protected_before: Mapping[str, Any], lock_nonce: str,
    lock_path: Path = RUN_LOCK,
) -> dict[str, Any]:
    require_run_claim(lock_nonce, lock_path)
    paths.assert_stage_preconditions("final_attempt")
    verify_source_closure(
        source_closure["source_commit"], source_closure["source_hashes"],
    )
    rows_receipt = json.loads(paths.rows_receipt.read_text())
    _validate_rows_receipt(rows_receipt, paths)
    unlock = load_programs_unlock(paths)
    if source_closure != {
        "source_commit": unlock["source_commit"],
        "source_hashes": unlock["source_hashes"],
    } or dict(protected_before) != unlock["protected_before"]:
        raise RuntimeError("final attempt differs from canonical programs unlock")
    attempt = {
        "schema_version": 1,
        "status": "attempt_frozen_before_final_deserialization",
        "authority": "none",
        "authorized_for_scored_experiments": False,
        "authorized_for_final_scoring": False,
        "requested_role": ROLE_NAMES[2],
        "rows_receipt": artifact_binding(paths.rows_receipt),
        "programs": artifact_binding(paths.programs),
        "programs_receipt": artifact_binding(paths.programs_receipt),
        "final_cache": _expected_final_cache(rows_receipt),
        "source_commit": source_closure["source_commit"],
        "source_hashes": dict(source_closure["source_hashes"]),
        "protected_before": dict(protected_before),
        "lock_nonce": lock_nonce,
        "final_role_loads_before_attempt": 0,
        "final_evaluations_before_attempt": 0,
    }
    atomic_create_json(attempt, paths.final_attempt)
    if json.loads(paths.final_attempt.read_text()) != attempt:
        raise RuntimeError("final attempt did not reload exactly")
    return attempt
