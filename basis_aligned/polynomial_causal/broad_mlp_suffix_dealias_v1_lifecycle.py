"""Source/input/artifact lifecycle for broad-MLP suffix de-alias v1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import broad_mlp_suffix_dealias_v1_measurements as measurement
import early_mlp_context_cross_v1_lifecycle as parent


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_NAMESPACE = "broad_mlp_suffix_dealias_v1_measurement_wave_v1"
PARENT_NAMESPACE = "early_mlp_context_cross_v1_measurement_wave_v1"
PARENT_PATHS = parent.output_paths(namespace=PARENT_NAMESPACE)
PROTECTED_SHA256 = {
    HERE / "BROAD_MLP_SUFFIX_DEALIAS_V1_PREREGISTRATION.md":
        "065f1458a93685432596abf43c6628935e93c4e84d8210cb19d7f346d1e9b24f",
    PARENT_PATHS.authority:
        "2bca3a523c3fbcb95b67766033c54d908d00784378dab0216572147780834463",
    PARENT_PATHS.payload:
        "2bf8c5c61038a6b9cc5437357e4ba45373a18f3312e6a80f3c86d1207251d279",
    PARENT_PATHS.manifest:
        "4ca06eb4b2d854469317e487d3dac7a06ef86a1c441e3fe54db03b2505744e9f",
    PARENT_PATHS.receipt:
        "82af48ef6a553038316004dfcf1e82eb10f9d717fde3f8021c805c0afd79da43",
}
EXPECTED_PARENT = {
    "source_closure_sha256": "207f9e91ae4d16af293563f65337bb7dfe8666542cd37186cb9ec14b7cd9e437",
    "program_bank_sha256": "10f253d1f89109b864fda7dff6d16b40212326600146d0b429006c017af6e443",
    "shared_program_sha256": measurement.SHARED_PROGRAM_SHA256,
    "model_realization_sha256": measurement.MODEL_REALIZATION_SHA256,
    "component_tree_sha256": measurement.COMPONENT_TREE_SHA256,
    "two_role_authority_sha256": "b64e087d2bd9c3ae85a85816a1d15fc6d043aaf0be27553b0accc6e6a4c9dd6d",
}

NEW_SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/BROAD_MLP_SUFFIX_DEALIAS_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/BROAD_MLP_SUFFIX_DEALIAS_V1_IMPLEMENTATION_AMENDMENT.md",
    "basis_aligned/polynomial_causal/broad_mlp_suffix_dealias_v1.py",
    "basis_aligned/polynomial_causal/test_broad_mlp_suffix_dealias_v1.py",
    "basis_aligned/polynomial_causal/broad_mlp_suffix_dealias_v1_measurements.py",
    "basis_aligned/polynomial_causal/test_broad_mlp_suffix_dealias_v1_measurements.py",
    "basis_aligned/polynomial_causal/broad_mlp_suffix_dealias_v1_lifecycle.py",
    "basis_aligned/polynomial_causal/test_broad_mlp_suffix_dealias_v1_lifecycle.py",
    "basis_aligned/polynomial_causal/broad_mlp_suffix_dealias_v1_bilin18_backend.py",
    "basis_aligned/polynomial_causal/test_broad_mlp_suffix_dealias_v1_bilin18_backend.py",
    "basis_aligned/polynomial_causal/run_broad_mlp_suffix_dealias_v1.py",
    "basis_aligned/polynomial_causal/test_run_broad_mlp_suffix_dealias_v1.py",
    "basis_aligned/polynomial_causal/score_broad_mlp_suffix_dealias_v1.py",
    "basis_aligned/polynomial_causal/test_score_broad_mlp_suffix_dealias_v1.py",
)
SOURCE_RELATIVE_PATHS = tuple(sorted(set((*parent.SOURCE_RELATIVE_PATHS, *NEW_SOURCE_PATHS))))


def file_sha256(path: Path) -> str:
    return parent.file_sha256(path)


def verify_protected_files() -> None:
    for path, expected in PROTECTED_SHA256.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"protected input differs: {path}")


def parent_authority() -> dict[str, Any]:
    verify_protected_files()
    value = json.loads(PARENT_PATHS.authority.read_text(encoding="utf-8"))
    receipt = json.loads(PARENT_PATHS.receipt.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(receipt, dict) or receipt.get(
        "status"
    ) != "complete_two_role_measurement_receipt_last" or receipt.get(
        "authoritative_measurement"
    ) is not True or value.get("status") != "frozen_before_any_measurement_outcome" or (
        value.get("source_closure_sha256") != EXPECTED_PARENT["source_closure_sha256"]
    ) or value.get("program_bank_sha256") != EXPECTED_PARENT["program_bank_sha256"] or (
        value.get("shared_program_sha256") != EXPECTED_PARENT["shared_program_sha256"]
    ) or value.get("two_role_authority_sha256") != EXPECTED_PARENT[
        "two_role_authority_sha256"
    ] or value.get("model_binding", {}).get("model_realization_sha256") != (
        EXPECTED_PARENT["model_realization_sha256"]
    ) or value.get("model_binding", {}).get("component_tree_sha256") != (
        EXPECTED_PARENT["component_tree_sha256"]
    ):
        raise RuntimeError("parent authority semantics changed")
    return value


def load_two_roles():
    verify_protected_files()
    return parent.load_two_roles()


def committed_source_closure():
    source = parent.inherited.committed_source_closure(REPO, SOURCE_RELATIVE_PATHS)
    completed = subprocess.run(
        (
            "git", "-C", str(REPO), "merge-base", "--is-ancestor",
            source.source_commit, "origin/main",
        ),
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError("launch commit is not an ancestor of origin/main")
    return source


def verify_source_closure(source) -> None:
    for relative, expected in source.path_sha256s:
        path = REPO / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError("source closure changed during transaction")


@dataclass(frozen=True, slots=True)
class OutputPaths:
    authority: Path
    payload: Path
    manifest: Path
    receipt: Path
    failure: Path
    lock: Path

    def all_paths(self) -> tuple[Path, ...]:
        return self.authority, self.payload, self.manifest, self.receipt, self.failure, self.lock

    def require_pristine(self) -> None:
        existing = [str(path) for path in self.all_paths() if path.exists()]
        if existing:
            raise RuntimeError(f"broad-MLP namespace is already spent: {existing}")


def output_paths(directory: Path = HERE, namespace: str = DEFAULT_NAMESPACE) -> OutputPaths:
    if not namespace or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in namespace):
        raise ValueError("unsafe output namespace")
    root = directory.resolve()
    return OutputPaths(
        authority=root / f"{namespace}_authority.json",
        payload=root / f"{namespace}_payload.pt",
        manifest=root / f"{namespace}_manifest.json",
        receipt=root / f"{namespace}_receipt.json",
        failure=root / f"{namespace}_failure.json",
        lock=root / f".{namespace}.lock",
    )


RunLock = parent.RunLock


def publish_json_create_only(path: Path, value: Mapping[str, Any], lock: RunLock) -> None:
    parent.publish_json_create_only(path, value, lock)


def publish_torch_create_only(path: Path, value: Any, lock: RunLock) -> None:
    parent.publish_torch_create_only(path, value, lock)
