"""CPU-only source/input/artifact lifecycle for early-MLP/context cross v1.

This module deliberately has no model loader or forward capability.  It verifies
inherited row/source pins, document disjointness, committed source closure, a fresh
namespace, create-only publication, and receipt-last ordering primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

import compilation_mask_cut_rank_v1_gpu_adapter as inherited
import early_mlp_context_cross_v1_measurements as measurement


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
ROW_RECEIPT = BQ / ".rowcache/fineweb_oracle_v2_receipt.json"
ROLE_ENTRY_NAMES = {
    "skip7000": "n192_skip7000",
    "skip11000": "n192_skip11000",
}
ROW_FILES = {
    "skip7000": BQ / ".rowcache/fineweb_n192_skip7000.pt",
    "skip11000": BQ / ".rowcache/fineweb_n192_skip11000.pt",
}
FROZEN_FILE_SHA256 = {
    ROW_RECEIPT: "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16",
    ROW_FILES["skip7000"]:
        "d66c1ee7807bc6b9bd7d0ddba5cdd7e3bc64926b00320a10675a2f817d67128c",
    ROW_FILES["skip11000"]:
        "b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868",
    HERE / "compilation_mask_cut_rank_v1_bilin18_backend.py":
        "738f4988fe5b87a7329f833bc7117cc417adcb9834da06533b00bc8b320c18e0",
}
FROZEN_RAW_ROW_SHA256 = {
    "skip7000": "10d66676c804569eaa501d0c3c425f357d1d4305eb2581f1e9a5403504f054c0",
    "skip11000": "5d6c1697f6d05860e4235c21e6324e3451d47924565d8edb62e06fbe37b3a1fa",
}
SOURCE_RELATIVE_PATHS = (
    "basis_aligned/polynomial_causal/EARLY_MLP_CONTEXT_CROSS_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/EARLY_MLP_CONTEXT_CROSS_V1_IMPLEMENTATION_AMENDMENT.md",
    "basis_aligned/polynomial_causal/early_mlp_context_cross_v1.py",
    "basis_aligned/polynomial_causal/test_early_mlp_context_cross_v1.py",
    "basis_aligned/polynomial_causal/early_mlp_context_cross_v1_statistics.py",
    "basis_aligned/polynomial_causal/test_early_mlp_context_cross_v1_statistics.py",
    "basis_aligned/polynomial_causal/early_mlp_context_cross_v1_measurements.py",
    "basis_aligned/polynomial_causal/test_early_mlp_context_cross_v1_measurements.py",
    "basis_aligned/polynomial_causal/early_mlp_context_cross_v1_lifecycle.py",
    "basis_aligned/polynomial_causal/test_early_mlp_context_cross_v1_lifecycle.py",
    "basis_aligned/polynomial_causal/early_mlp_context_cross_v1_bilin18_backend.py",
    "basis_aligned/polynomial_causal/test_early_mlp_context_cross_v1_bilin18_backend.py",
    "basis_aligned/polynomial_causal/run_early_mlp_context_cross_v1.py",
    "basis_aligned/polynomial_causal/test_run_early_mlp_context_cross_v1.py",
    "basis_aligned/polynomial_causal/compilation_mask_cut_rank_v1_bilin18_backend.py",
    "basis_aligned/polynomial_causal/compilation_mask_cut_rank_v1_gpu_adapter.py",
    "basis_aligned/polynomial_causal/compilation_mask_cut_rank_v1_measurements.py",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "jacclust/tt_model.py",
)
DEFAULT_NAMESPACE = "early_mlp_context_cross_v1_measurement_wave_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise RuntimeError(f"frozen input hash differs: {path}")


def verify_inherited_files() -> None:
    for path, expected in FROZEN_FILE_SHA256.items():
        verify_file(path, expected)


@dataclass(frozen=True, slots=True)
class RoleRows:
    role: str
    wave: inherited.RowWave
    row_file_sha256: str
    document_identity_set_sha256: str

    def __post_init__(self) -> None:
        if self.role not in ROLE_ENTRY_NAMES or not isinstance(
            self.wave, inherited.RowWave
        ) or self.row_file_sha256 != FROZEN_FILE_SHA256[ROW_FILES[self.role]] or (
            len(self.document_identity_set_sha256) != 64
        ) or self.wave.row_count != measurement.ROW_COUNT or self.wave.document_count != (
            measurement.ROLE_DOCUMENT_COUNTS[self.role]
        ):
            raise ValueError("role row binding changed")


@dataclass(frozen=True, slots=True)
class TwoRoleRows:
    skip7000: RoleRows
    skip11000: RoleRows
    disjointness_sha256: str

    def __post_init__(self) -> None:
        if self.skip7000.role != "skip7000" or self.skip11000.role != (
            "skip11000"
        ) or len(self.disjointness_sha256) != 64:
            raise ValueError("two-role binding changed")


def _document_sets(receipt: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    sets = receipt.get("document_provenance", {}).get("sets", {})
    output: dict[str, frozenset[str]] = {}
    for role, entry_name in ROLE_ENTRY_NAMES.items():
        records = sets.get(entry_name)
        if not isinstance(records, list) or len(records) != measurement.ROW_COUNT:
            raise RuntimeError("row provenance role is absent or changed")
        identifiers = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(
                record.get("document_id"), str
            ) or not record["document_id"]:
                raise RuntimeError("row provenance document identity changed")
            identifiers.append(record["document_id"])
        output[role] = frozenset(identifiers)
        if len(output[role]) != measurement.ROLE_DOCUMENT_COUNTS[role]:
            raise RuntimeError("source-document count changed")
    return output


def load_two_roles() -> TwoRoleRows:
    """Verify exact files before/after load and prove cross-role disjointness."""

    verify_inherited_files()
    receipt_before = file_sha256(ROW_RECEIPT)
    raw_receipt = json.loads(ROW_RECEIPT.read_text(encoding="utf-8"))
    documents = _document_sets(raw_receipt)
    if documents["skip7000"] & documents["skip11000"]:
        raise RuntimeError("evaluation roles share a source document")
    role_rows: dict[str, RoleRows] = {}
    for role, entry_name in ROLE_ENTRY_NAMES.items():
        row_path = ROW_FILES[role]
        before = file_sha256(row_path)
        wave = inherited.load_row_wave(ROW_RECEIPT, entry_name)
        after = file_sha256(row_path)
        if before != FROZEN_FILE_SHA256[row_path] or after != before:
            raise RuntimeError("row cache changed while loading")
        raw = torch.load(row_path, map_location="cpu", weights_only=True)
        tensor = raw["rows"] if isinstance(raw, dict) and set(raw) == {"rows"} else raw
        if inherited.raw_tensor_sha256(tensor) != FROZEN_RAW_ROW_SHA256[role]:
            raise RuntimeError("raw row tensor differs from amendment")
        role_rows[role] = RoleRows(
            role=role, wave=wave, row_file_sha256=before,
            document_identity_set_sha256=logical_sha256(sorted(documents[role])),
        )
    if file_sha256(ROW_RECEIPT) != receipt_before:
        raise RuntimeError("row receipt changed while loading roles")
    disjointness = logical_sha256({
        "skip7000": sorted(documents["skip7000"]),
        "skip11000": sorted(documents["skip11000"]),
        "intersection": [],
    })
    return TwoRoleRows(
        skip7000=role_rows["skip7000"], skip11000=role_rows["skip11000"],
        disjointness_sha256=disjointness,
    )


def committed_source_closure() -> inherited.SourceClosure:
    """Require exact working bytes at a pushed HEAD for every execution source."""

    return inherited.committed_source_closure(REPO, SOURCE_RELATIVE_PATHS)


@dataclass(frozen=True, slots=True)
class OutputPaths:
    authority: Path
    payload: Path
    manifest: Path
    receipt: Path
    failure: Path
    lock: Path

    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.authority, self.payload, self.manifest, self.receipt,
            self.failure, self.lock,
        )

    def require_pristine(self) -> None:
        existing = [str(path) for path in self.all_paths() if path.exists()]
        if existing:
            raise RuntimeError(f"cross namespace is already spent: {existing}")


def output_paths(
    directory: Path = HERE, namespace: str = DEFAULT_NAMESPACE,
) -> OutputPaths:
    if not namespace or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in namespace
    ):
        raise ValueError("output namespace is not a safe lowercase identifier")
    root = directory.resolve()
    return OutputPaths(
        authority=root / f"{namespace}_authority.json",
        payload=root / f"{namespace}_payload.pt",
        manifest=root / f"{namespace}_manifest.json",
        receipt=root / f"{namespace}_receipt.json",
        failure=root / f"{namespace}_failure.json",
        lock=root / f".{namespace}.lock",
    )


RunLock = inherited.RunLock


def publish_json_create_only(
    path: Path, value: Mapping[str, Any], lock: RunLock,
) -> None:
    inherited._publish_bytes_create_only(path, inherited._json_bytes(value), lock)


def publish_torch_create_only(path: Path, value: Any, lock: RunLock) -> None:
    inherited._publish_bytes_create_only(path, inherited._torch_bytes(value), lock)


def validate_contract() -> None:
    if set(ROLE_ENTRY_NAMES) != set(measurement.ROLE_DOCUMENT_COUNTS) or set(
        ROW_FILES
    ) != set(ROLE_ENTRY_NAMES) or len(SOURCE_RELATIVE_PATHS) != len(
        set(SOURCE_RELATIVE_PATHS)
    ):
        raise RuntimeError("lifecycle role/source registry changed")


validate_contract()
