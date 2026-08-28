"""CPU-only inherited-evidence capability for suffix-transport v1.

This module validates every inherited numerical object before deserializing it and
returns only private CPU masters for the two rank-64 bases and exact affine
initialization.  It deliberately does not import the model, load fresh or historical
rows, run a forward pass, or expose an original-MLP capability.  Public construction
is pinned to the canonical inherited paths; caller-selected roots exist only in
private test helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import MappingProxyType
from typing import Any, Mapping

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"

SOURCE_COMMIT = "bd9a58207b41b05c1b76f2be730771e89cab54ff"
SHIP_REALIZATION_SHA256 = (
    "21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e"
)
D_MODEL = 1152
CODE_DIM = 64


@dataclass(frozen=True)
class ArtifactPin:
    filename: str
    sha256: str
    bytes: int


PINS = (
    ArtifactPin(
        "early_mlp_state_complete_compiler_v21_final_authority.json",
        "659051ed8e2d34a2d755d1942f4112161294831e724d6697f4c3e2ef466f6987",
        11_767,
    ),
    ArtifactPin(
        "early_mlp_state_complete_compiler_v21_final_result.pt",
        "c73f2a7f6099de9e28550b02d7d02904fe37477c65cb8c5c9c6f4beed9bfb5cd",
        1_594_891,
    ),
    ArtifactPin(
        "early_mlp_state_complete_compiler_v21_programs_receipt.json",
        "c9c67bdd14a34dd83192a02d49705d0ed7043e2f9751d042250f44395f88ec2c",
        11_269,
    ),
    ArtifactPin(
        "early_mlp_state_complete_compiler_v21_programs.pt",
        "36a8e5203ec72d8c8f30909dba9241d1bf2a4a2d3fd980d8c558e28c3c0b614e",
        186_250_188,
    ),
    ArtifactPin(
        "joint_early_mlp_pca_composition_authoritative_v3_bases.pt",
        "0eee01f39087548a479486d068404f78c4bdc2fd930932add162212da31fe4d9",
        601_081,
    ),
    ArtifactPin(
        "joint_early_mlp_pca_composition_authoritative_v3_basis_receipt.json",
        "b81adb4c78255613997de4cbfc8ffd9e8eec233b40950915a14005ba3efcba0f",
        3_947,
    ),
)

TERMINAL_CHAIN_PINS = (
    ArtifactPin(
        "early_mlp_state_complete_compiler_v21_final_manifest.json",
        "21c1a3718fd040f702b31db0a574b886a979f2f18d4faac6941f704d696305dc",
        18_654,
    ),
    ArtifactPin(
        "early_mlp_state_complete_compiler_v21_final_attempt.json",
        "e20c2b94b5647bd77333c220488b49d7eec12633ca656040c4384f59aa7c2838",
        14_906,
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def raw_tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("raw tensor hash requires a tensor")
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"inherited artifact is absent: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _verify_pinned_files(
    root: Path = BQ, pins: tuple[ArtifactPin, ...] = PINS,
) -> dict[str, dict[str, Any]]:
    """Hash all inherited files before any torch deserialization."""

    if not pins or len({pin.filename for pin in pins}) != len(pins):
        raise ValueError("inherited artifact pins are empty or duplicated")
    bindings: dict[str, dict[str, Any]] = {}
    for pin in pins:
        path = root / pin.filename
        observed = _binding(path)
        if observed["sha256"] != pin.sha256 or observed["bytes"] != pin.bytes:
            raise RuntimeError(f"inherited artifact binding changed: {pin.filename}")
        bindings[pin.filename] = observed
    return bindings


def _strict_json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise RuntimeError(f"duplicate JSON key in inherited receipt: {key}")
            output[key] = value
        return output

    before = file_sha256(path)
    value = json.loads(path.read_text(), object_pairs_hook=reject_duplicates)
    after = file_sha256(path)
    if before != after or (expected_sha256 is not None and after != expected_sha256):
        raise RuntimeError(f"inherited JSON mutated while loading: {path.name}")
    if not isinstance(value, dict):
        raise RuntimeError(f"inherited JSON root is not an object: {path.name}")
    return value


def verify_historical_source_closure(
    source_commit: str,
    source_hashes: Mapping[str, str],
    *,
    root: Path = ROOT,
    expected_count: int | None = None, require_current: bool = True,
) -> None:
    """Verify receipt-bound blobs at their historical commit, not current HEAD."""

    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise RuntimeError("historical source commit is malformed")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise RuntimeError("historical source closure is empty")
    if expected_count is not None and len(source_hashes) != expected_count:
        raise RuntimeError("historical source closure path count changed")
    available = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=root,
        capture_output=True,
    )
    if available.returncode:
        raise RuntimeError("historical source commit is unavailable")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"], cwd=root,
        capture_output=True,
    )
    if ancestor.returncode:
        raise RuntimeError("historical source commit is not an ancestor of HEAD")
    for relative, expected in sorted(source_hashes.items()):
        if not isinstance(relative, str) or not relative or not isinstance(
            expected, str
        ) or len(expected) != 64 or Path(relative).is_absolute() or ".." in Path(
            relative
        ).parts:
            raise RuntimeError("historical source closure entry is malformed")
        blob = subprocess.run(
            ["git", "show", f"{source_commit}:{relative}"], cwd=root,
            capture_output=True,
        )
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != expected:
            raise RuntimeError(f"historical source content changed: {relative}")
        if require_current:
            current = root / relative
            if current.is_symlink() or not current.is_file() or file_sha256(current) != expected:
                raise RuntimeError(f"current inherited source content changed: {relative}")


def _require_binding(
    name: str, observed: Any, expected: Mapping[str, Any],
) -> None:
    if observed != dict(expected):
        raise RuntimeError(f"inherited {name} binding changed")


def _validate_execution_closure(value: Any, *, name: str) -> None:
    expected = {
        "outer_model_returned": True,
        "hook_restored_and_inert": True,
        "component_tree_before": SHIP_REALIZATION_SHA256,
        "component_tree_after": SHIP_REALIZATION_SHA256,
    }
    if value != expected:
        raise RuntimeError(f"inherited {name} execution closure changed")


def validate_v21_metadata(
    authority: Mapping[str, Any],
    receipt: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, str]:
    """Cross-check the negative terminal authority, result, and program unlock."""

    exact_authority_keys = {
        "attempt", "authority", "authorized_for_scored_experiments", "claim_scope",
        "execution_closure", "final_cache", "final_evaluations", "final_role_loads",
        "integrity", "last_write_rule", "manifest", "package_admitted",
        "program_bundle", "program_unlock", "result", "schema_version",
        "source_commit", "source_hashes", "status",
    }
    exact_receipt_keys = {
        "authority", "authorized_for_final_scoring", "authorized_for_training",
        "execution_closure", "final_rulings_sha256", "frozen_contents",
        "implementation_amendment_sha256", "programs_artifact_bytes",
        "programs_artifact_path", "programs_artifact_sha256", "protocol_sha256",
        "rows_receipt_path", "rows_receipt_sha256", "site1_manifest", "source_commit",
        "source_hashes", "status",
    }
    fixed_authority = {
        "schema_version": 1,
        "status": "authoritative_negative_v21_final",
        "authority": "compiler_v21_scientific_outcome",
        "authorized_for_scored_experiments": True,
        "source_commit": SOURCE_COMMIT,
        "integrity": True,
        "package_admitted": False,
        "claim_scope": "negative",
        "final_role_loads": 1,
        "final_evaluations": 1,
    }
    if not isinstance(authority, Mapping) or set(authority) != exact_authority_keys or any(
        authority.get(key) != expected for key, expected in fixed_authority.items()
    ):
        raise RuntimeError("v2.1 terminal authority metadata changed")
    fixed_receipt = {
        "status": "frozen_v21_programs_controls_strata_prices_before_final",
        "authority": "compiler_v21_final_unlock",
        "authorized_for_training": False,
        "authorized_for_final_scoring": True,
        "source_commit": SOURCE_COMMIT,
    }
    if not isinstance(receipt, Mapping) or set(receipt) != exact_receipt_keys or any(
        receipt.get(key) != expected for key, expected in fixed_receipt.items()
    ):
        raise RuntimeError("v2.1 program unlock metadata changed")

    _require_binding(
        "final result", authority.get("result"),
        bindings["early_mlp_state_complete_compiler_v21_final_result.pt"],
    )
    _require_binding(
        "program unlock", authority.get("program_unlock"),
        bindings["early_mlp_state_complete_compiler_v21_programs_receipt.json"],
    )
    _require_binding(
        "program bundle", authority.get("program_bundle"),
        bindings["early_mlp_state_complete_compiler_v21_programs.pt"],
    )
    _require_binding(
        "final manifest", authority.get("manifest"),
        bindings["early_mlp_state_complete_compiler_v21_final_manifest.json"],
    )
    _require_binding(
        "final attempt", authority.get("attempt"),
        bindings["early_mlp_state_complete_compiler_v21_final_attempt.json"],
    )
    program_binding = bindings[
        "early_mlp_state_complete_compiler_v21_programs.pt"
    ]
    if receipt.get("programs_artifact_path") != program_binding["path"] or receipt.get(
        "programs_artifact_sha256"
    ) != program_binding["sha256"] or receipt.get(
        "programs_artifact_bytes"
    ) != program_binding["bytes"]:
        raise RuntimeError("v2.1 unlock no longer binds the program bundle")
    for container_name, container in (("authority", authority), ("receipt", receipt)):
        _validate_execution_closure(container.get("execution_closure"), name=container_name)
    source_hashes = authority.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or len(source_hashes) != 60 or (
        receipt.get("source_hashes") != source_hashes
    ):
        raise RuntimeError("v2.1 inherited source closures disagree")
    return source_hashes


def _finite_tensor(name: str, value: Any, shape: tuple[int, ...]) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != torch.float32:
        raise RuntimeError(f"{name} tensor schema changed")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} contains nonfinite values")
    return value.detach().cpu().contiguous().clone()


def validate_affine_initializations(bundle: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    exact_bundle_keys = {
        "authority", "authorized_for_final_scoring", "authorized_for_training",
        "candidate_ledgers", "controls", "family_representatives",
        "final_rulings_sha256", "implementation_amendment_sha256",
        "pipeline_contexts", "prices", "programs", "protocol_sha256",
        "rows_receipt_sha256", "schema_version", "selection_receipts",
        "site0_training_authorization", "stage_bindings", "status", "strata",
    }
    if not isinstance(bundle, Mapping) or set(bundle) != exact_bundle_keys or bundle.get("schema_version") != 1 or bundle.get(
        "status"
    ) != "frozen_v21_program_bundle_pending_final_unlock" or bundle.get(
        "authority"
    ) != "compiler_v21_program_bundle" or bundle.get(
        "authorized_for_training"
    ) is not False or bundle.get("authorized_for_final_scoring") is not False:
        raise RuntimeError("v2.1 program bundle metadata changed")
    programs = bundle.get("programs")
    if not isinstance(programs, Mapping) or set(programs) != {"true", "shuffle", "mean"}:
        raise RuntimeError("v2.1 program families changed")
    true = programs["true"]
    if not isinstance(true, Mapping) or set(true) != {0, 1}:
        raise RuntimeError("v2.1 true affine sites changed")
    output: dict[int, dict[str, Any]] = {}
    selections = bundle.get("selection_receipts")
    expected_selected = {0: "B_l5_r64", 1: "B_l6_r64"}
    expected_lambdas = {0: 0.01, 1: 0.1}
    if not isinstance(selections, Mapping):
        raise RuntimeError("v2.1 selection receipts changed")
    exact_keys = {
        "grammar", "interface", "family", "mean", "scale", "bias", "left",
        "right", "lambda", "rank",
    }
    for site in (0, 1):
        state = true[site]
        selection = selections.get(f"true_site{site}")
        if not isinstance(state, Mapping) or set(state) != exact_keys or state.get(
            "grammar"
        ) != "affine" or state.get("interface") != "state_complete_p" or state.get(
            "family"
        ) != "B_state_complete_affine_euclidean" or state.get("rank") != CODE_DIM or not isinstance(
            selection, Mapping
        ) or selection.get("selected") != expected_selected[site] or selection.get(
            "selected_family"
        ) != "B_state_complete_affine_euclidean" or state.get("lambda") != expected_lambdas[site]:
            raise RuntimeError(f"v2.1 affine state schema changed at site {site}")
        copied = {
            "grammar": state["grammar"],
            "interface": state["interface"],
            "mean": _finite_tensor(f"site{site}.mean", state["mean"], (D_MODEL,)),
            "scale": _finite_tensor(f"site{site}.scale", state["scale"], (D_MODEL,)),
            "bias": _finite_tensor(f"site{site}.bias", state["bias"], (CODE_DIM,)),
            "left": _finite_tensor(
                f"site{site}.left", state["left"], (D_MODEL, CODE_DIM)
            ),
            "right": _finite_tensor(
                f"site{site}.right", state["right"], (CODE_DIM, CODE_DIM)
            ),
        }
        if bool(torch.any(copied["scale"] <= 0)) or not math.isfinite(float(state["lambda"])):
            raise RuntimeError(f"v2.1 affine normalization changed at site {site}")
        ledger = bundle["candidate_ledgers"].get(f"true_site{site}")
        selected = ledger.get(expected_selected[site]) if isinstance(ledger, Mapping) else None
        selected_state = selected.get("state") if isinstance(selected, Mapping) else None
        if not isinstance(selected_state, Mapping) or set(selected_state) != exact_keys:
            raise RuntimeError(f"v2.1 selected candidate ledger changed at site {site}")
        for key in exact_keys:
            left, right = state[key], selected_state[key]
            equal = torch.equal(left, right) if torch.is_tensor(left) else left == right
            if not equal:
                raise RuntimeError(
                    f"v2.1 true program differs from selected ledger at site {site}.{key}"
                )
        output[site] = copied
    return output


def validate_bases(
    payload: Mapping[str, Any], receipt: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[int, torch.Tensor], Mapping[str, str], str]:
    artifact = bindings[
        "joint_early_mlp_pca_composition_authoritative_v3_bases.pt"
    ]
    exact_receipt_keys = {
        "artifact_bytes", "artifact_path", "artifact_sha256", "authority",
        "authorized_for_scored_experiments", "authorized_for_training",
        "basis_row_receipt", "freeze_rule", "preregistration_sha256",
        "schema_version", "ship_realization_sha256", "site_basis_sha256",
        "source_commit", "source_hashes", "status",
    }
    exact_payload_keys = {
        "authority", "authorized_for_training", "basis_row_receipt",
        "capture_positions", "pca_seed", "projection_rank", "schema_version",
        "ship_realization_sha256", "sites", "source_commit", "source_hashes",
        "status", "support_rank",
    }
    fixed_receipt = {
        "schema_version": 2,
        "status": "frozen_before_evaluation",
        "authority": "canonical_fineweb_basis_split",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "ship_realization_sha256": SHIP_REALIZATION_SHA256,
        "artifact_path": artifact["path"],
        "artifact_sha256": artifact["sha256"],
        "artifact_bytes": artifact["bytes"],
    }
    if not isinstance(receipt, Mapping) or set(receipt) != exact_receipt_keys or any(
        receipt.get(key) != expected for key, expected in fixed_receipt.items()
    ):
        raise RuntimeError("v3 basis receipt metadata changed")
    fixed_payload = {
        "schema_version": 2,
        "status": "frozen_before_evaluation",
        "authority": "canonical_fineweb_basis_split",
        "authorized_for_training": False,
        "projection_rank": CODE_DIM,
        "support_rank": 256,
        "pca_seed": 161803,
        "ship_realization_sha256": SHIP_REALIZATION_SHA256,
        "capture_positions": "64::3 over 256 model-input positions",
    }
    if not isinstance(payload, Mapping) or set(payload) != exact_payload_keys or any(
        payload.get(key) != expected for key, expected in fixed_payload.items()
    ):
        raise RuntimeError("v3 basis payload metadata changed")
    if payload.get("basis_row_receipt") != receipt.get("basis_row_receipt"):
        raise RuntimeError("v3 basis row receipt binding changed")
    sites = payload.get("sites")
    if not isinstance(sites, Mapping) or set(sites) != {0, 1}:
        raise RuntimeError("v3 basis sites changed")
    receipt_hashes = receipt.get("site_basis_sha256")
    if not isinstance(receipt_hashes, Mapping) or set(receipt_hashes) != {"0", "1"}:
        raise RuntimeError("v3 basis receipt tensor hashes changed")
    output: dict[int, torch.Tensor] = {}
    exact_site_keys = {
        "basis", "basis_sha256", "captured_energy_fraction",
        "captured_residual_rms", "captured_residual_sha256",
        "captured_residual_shape", "gram_max_abs_error", "projected_correction_rms",
        "support_singular_values",
    }
    for site in (0, 1):
        record = sites[site]
        if not isinstance(record, Mapping) or set(record) != exact_site_keys or (
            record.get("basis_sha256") != receipt_hashes[str(site)]
        ):
            raise RuntimeError(f"v3 basis record changed at site {site}")
        basis = _finite_tensor(f"basis{site}", record["basis"], (D_MODEL, CODE_DIM))
        if raw_tensor_sha256(basis) != record["basis_sha256"]:
            raise RuntimeError(f"v3 basis tensor hash changed at site {site}")
        gram_error = float(
            (basis.double().T @ basis.double() - torch.eye(CODE_DIM, dtype=torch.float64))
            .abs().max()
        )
        if not math.isfinite(gram_error) or gram_error > 2e-4:
            raise RuntimeError(f"v3 basis lost orthonormality at site {site}")
        output[site] = basis
    source_hashes = receipt.get("source_hashes")
    source_commit = receipt.get("source_commit")
    if not isinstance(source_hashes, Mapping) or len(source_hashes) != 17 or not isinstance(
        source_commit, str
    ) or payload.get("source_hashes") != source_hashes or payload.get(
        "source_commit"
    ) != source_commit:
        raise RuntimeError("v3 basis inherited source closures disagree")
    return output, source_hashes, source_commit


def _tensor_tree_hash(bases: Mapping[int, torch.Tensor], states: Mapping[int, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for site in (0, 1):
        digest.update(f"basis{site}".encode())
        digest.update(raw_tensor_sha256(bases[site]).encode())
        for key in ("grammar", "interface"):
            digest.update(f"site{site}.{key}".encode())
            digest.update(str(states[site][key]).encode())
        for key in ("mean", "scale", "bias", "left", "right"):
            digest.update(f"site{site}.{key}".encode())
            digest.update(raw_tensor_sha256(states[site][key]).encode())
        digest.update(f"site{site}.full_product".encode())
        digest.update(raw_tensor_sha256(
            states[site]["left"] @ states[site]["right"]
        ).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class ValidatedInherited:
    bindings: Mapping[str, Mapping[str, Any]]
    ship_realization_sha256: str
    compiler_source_commit: str
    basis_source_commit: str
    snapshot_sha256: str
    full_product_sha256: Mapping[int, str]


class LoadedInitialization:
    """Private immutable masters that issue verified storage-disjoint clones."""

    __slots__ = (
        "__authority", "__bases", "__expected_snapshot", "__sealed", "__states",
    )

    def __init__(
        self, bases: Mapping[int, torch.Tensor], states: Mapping[int, Mapping[str, Any]],
        authority: ValidatedInherited,
    ) -> None:
        object.__setattr__(self, "_LoadedInitialization__sealed", False)
        object.__setattr__(self, "_LoadedInitialization__bases", {
            site: value.clone() for site, value in bases.items()
        })
        object.__setattr__(self, "_LoadedInitialization__states", {
            site: {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in state.items()
            }
            for site, state in states.items()
        })
        object.__setattr__(
            self, "_LoadedInitialization__expected_snapshot", authority.snapshot_sha256,
        )
        object.__setattr__(self, "_LoadedInitialization__authority", authority)
        self._require_pristine()
        object.__setattr__(self, "_LoadedInitialization__sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_LoadedInitialization__sealed", False):
            raise AttributeError("inherited initialization capability is sealed")
        object.__setattr__(self, name, value)

    @property
    def authority(self) -> ValidatedInherited:
        return self.__authority

    def _require_pristine(self) -> None:
        observed = _tensor_tree_hash(self.__bases, self.__states)
        if observed != self.__expected_snapshot:
            raise RuntimeError("inherited initialization master mutated")
        for site in (0, 1):
            product = self.__states[site]["left"] @ self.__states[site]["right"]
            if raw_tensor_sha256(product) != self.__authority.full_product_sha256[site]:
                raise RuntimeError("inherited full-product initialization mutated")

    def clone_bases(self) -> dict[int, torch.Tensor]:
        self._require_pristine()
        return {site: value.clone() for site, value in self.__bases.items()}

    def clone_affine_states(self) -> dict[int, dict[str, Any]]:
        self._require_pristine()
        return {
            site: {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in state.items()
            }
            for site, state in self.__states.items()
        }

    def make_program(self, route: str):
        import early_mlp_suffix_transport_v1_runtime as runtime

        return runtime.JointAffineProgram.from_v21_states(
            self.clone_affine_states(), route=route,
        )


def _load_inherited_initialization(root: Path, *, verify_git: bool) -> LoadedInitialization:
    """Private parameterized helper; production callers use the canonical wrapper."""

    bindings = _verify_pinned_files(root, PINS + TERMINAL_CHAIN_PINS)
    authority_path = root / "early_mlp_state_complete_compiler_v21_final_authority.json"
    receipt_path = root / "early_mlp_state_complete_compiler_v21_programs_receipt.json"
    basis_receipt_path = root / "joint_early_mlp_pca_composition_authoritative_v3_basis_receipt.json"
    authority = _strict_json(
        authority_path,
        bindings["early_mlp_state_complete_compiler_v21_final_authority.json"]["sha256"],
    )
    receipt = _strict_json(
        receipt_path,
        bindings["early_mlp_state_complete_compiler_v21_programs_receipt.json"]["sha256"],
    )
    basis_receipt = _strict_json(
        basis_receipt_path,
        bindings[
            "joint_early_mlp_pca_composition_authoritative_v3_basis_receipt.json"
        ]["sha256"],
    )
    v21_sources = validate_v21_metadata(authority, receipt, bindings)
    if verify_git:
        verify_historical_source_closure(
            SOURCE_COMMIT, v21_sources, expected_count=60,
        )

    bundle = torch.load(
        root / "early_mlp_state_complete_compiler_v21_programs.pt",
        map_location="cpu", weights_only=True,
    )
    affine_states = validate_affine_initializations(bundle)
    if file_sha256(root / "early_mlp_state_complete_compiler_v21_programs.pt") != bindings[
        "early_mlp_state_complete_compiler_v21_programs.pt"
    ]["sha256"]:
        raise RuntimeError("v2.1 program bundle mutated while loading")
    del bundle
    basis_payload = torch.load(
        root / "joint_early_mlp_pca_composition_authoritative_v3_bases.pt",
        map_location="cpu", weights_only=True,
    )
    bases, basis_sources, basis_commit = validate_bases(
        basis_payload, basis_receipt, bindings,
    )
    if file_sha256(root / "joint_early_mlp_pca_composition_authoritative_v3_bases.pt") != bindings[
        "joint_early_mlp_pca_composition_authoritative_v3_bases.pt"
    ]["sha256"]:
        raise RuntimeError("v3 basis artifact mutated while loading")
    del basis_payload
    if verify_git:
        verify_historical_source_closure(
            basis_commit, basis_sources, expected_count=17,
        )
        # Recheck both historical/current closures after all tensor deserialization;
        # otherwise a source could drift after its first closure replay.
        verify_historical_source_closure(
            SOURCE_COMMIT, v21_sources, expected_count=60,
        )
        verify_historical_source_closure(
            basis_commit, basis_sources, expected_count=17,
        )
    final_bindings = _verify_pinned_files(root, PINS + TERMINAL_CHAIN_PINS)
    if final_bindings != bindings:
        raise RuntimeError("inherited artifact snapshot changed while loading")
    snapshot = _tensor_tree_hash(bases, affine_states)
    full_product_hashes = MappingProxyType({
        site: raw_tensor_sha256(state["left"] @ state["right"])
        for site, state in affine_states.items()
    })
    validated = ValidatedInherited(
        bindings=MappingProxyType({
            key: MappingProxyType(dict(value)) for key, value in bindings.items()
        }),
        ship_realization_sha256=SHIP_REALIZATION_SHA256,
        compiler_source_commit=SOURCE_COMMIT,
        basis_source_commit=basis_commit,
        snapshot_sha256=snapshot,
        full_product_sha256=full_product_hashes,
    )
    return LoadedInitialization(bases, affine_states, validated)


def load_canonical_initialization() -> LoadedInitialization:
    """Issue the sole canonical, nonauthorizing inherited initialization capability."""

    return _load_inherited_initialization(BQ, verify_git=True)
