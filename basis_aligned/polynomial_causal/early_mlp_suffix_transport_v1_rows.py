"""Pure row-identity boundary for early-MLP suffix transport v1.

This module performs no network access and imports no model runtime.  It validates
candidate fit/validation/final tensors, constructs their identity sets, and decides
whether a candidate triple collides internally or with a frozen prior-registry
snapshot.  Rejected-candidate reports contain counts and hashes only: they never
serialize tokens, document identifiers, or dataset indices.

The eventual CPU harvester owns source traversal and artifact publication.  Keeping
the collision decision pure makes its fail-closed behavior independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from early_mlp_suffix_transport_v1_lifecycle import CandidateTriple, candidate_triple


TOKEN_LENGTH = 513
ROLES = ("fit", "validation", "final")
IDENTITY_KEYS = ("documents", "dataset_indices", "full_rows", "prefix32")
ORDERED_HASH_KEYS = (
    "ordered_tensor_raw", "ordered_provenance", "ordered_row_provenance_binding",
)
PAIR_KEYS = ("fit__validation", "fit__final", "validation__final")
PRIVACY_RULE = "counts_and_sha256_only_no_raw_candidate_identities"
COLLISION_EVIDENCE_SEMANTICS = (
    "sum_of_categorywise_unique_intersections_and_internal_duplicate_excesses;"
    "zero_is_collision_free;not_unique_collided_rows"
)
HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"

# This allowlist is prospective.  Recursive artifact discovery is forbidden: it is
# sensitive to unrelated later experiments and misses code-OOD and frozen-ship
# objects outside BQ.  File hashes bind the exact registry census before harvesting.
CANONICAL_REGISTRY_FILES = {
    BQ / ".rowcache/fineweb_oracle_v2_receipt.json":
        "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16",
    BQ / "joint_early_mlp_pca_composition_authoritative_v3_basis_receipt.json":
        "b81adb4c78255613997de4cbfc8ffd9e8eec233b40950915a14005ba3efcba0f",
    BQ / "early_mlp_state_complete_compiler_v2_rows_receipt.json":
        "23319ece1d8542d51e024bde0e2253d740b08ad18ad4f2d8565ba5120473fd82",
    BQ / "early_mlp_state_complete_compiler_v21_rows_receipt.json":
        "7fa45e9a9e77e6622167fbf024400177f581cdd40958c1fb722ca13d8fcc018b",
    HERE / "code_oracle_corpus_v2_manifest.json":
        "a19def47d44a581a72a6cb8f0d91ac7ae0bb2121007f43964f4f1dcb526cb9ec",
    BQ / "whole_model_heldout_results.json":
        "0f7e86df9004372ef282908eb86aa806eac463cae83db567ba892c9590129c4d",
}

# path -> (serialized-file SHA256, raw long-tensor SHA256, optional payload key)
CANONICAL_ROW_TENSORS = {
    BQ / ".rowcache/fineweb_n96_skip80.pt": (
        "94bc1fb3e3a6a061541e555295e0af8c50ae6068fdff84e95a69c25844091eda",
        "a703cadb1a5e27497cba43d21bca889a1d765b861c3da311a1dc4dfeb28b21cc", None),
    BQ / ".rowcache/fineweb_n480_skip80.pt": (
        "2acf75382486988a1e124a1a575ef3230af43aa1b1507d80dee02eefc7bba496",
        "343d92ce07f78572e3233120d3361814c63f69fa76e97e58b62d1d6c8f24497f", None),
    BQ / ".rowcache/fineweb_n96_skip1200.pt": (
        "21707551f35d13818c10ac59e12e9445ef076d0522371fe779691bfab719d34f",
        "d6302f327983e8233509e0ad8a05aa84fad88784861a9f8d10575b325be83dda", None),
    BQ / ".rowcache/fineweb_n192_skip7000.pt": (
        "d66c1ee7807bc6b9bd7d0ddba5cdd7e3bc64926b00320a10675a2f817d67128c",
        "10d66676c804569eaa501d0c3c425f357d1d4305eb2581f1e9a5403504f054c0", None),
    BQ / ".rowcache/fineweb_n192_skip11000.pt": (
        "b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868",
        "5d6c1697f6d05860e4235c21e6324e3451d47924565d8edb62e06fbe37b3a1fa", None),
    BQ / ".rowcache_compiler_v2/fineweb_n480_skip27000.pt": (
        "cf3abe833dec8ccfc09afef0ff1bdcc74b2ba8a37dc86e9c954c238bb6b7c276",
        "fc2b3b8ced2f5449e6494dc9b5127c95717b938b37db74fbcc4d9458e8d39442", None),
    BQ / ".rowcache_compiler_v2/fineweb_n192_skip31000.pt": (
        "0d66fc0958da4fa8c0aedbe5b4203d474382acf6b5c0ebe77b53a54505a91ac9",
        "f415e3b7a148104435592b8e482d875de24ae832603799d42315b415309e6ca2", None),
    BQ / ".rowcache_compiler_v2/fineweb_n192_skip35000.pt": (
        "c9de7d6386668f24414018b330f4182a17c4c73fb2babb395c0654a52f9a3acd",
        "a58c0a8cfc1ecd27417384470c71f0f8054793976b61833f8af30268a47cd398", None),
    BQ / ".rowcache_compiler_v21/fineweb_n192_skip39000.pt": (
        "6982853519be38f627fec91532f2e622573f57304bd8b04fe08220491122c8da",
        "f6fbbb1b84a276d23a5fc248b5b103a1a427bae717e7fc8fd2e73bc67800124e", None),
    BQ / "bilin18_eval_tokens_large.pt": (
        "bb2b00699e511245bb68069be1fe5559777170fb78a6dc9218830454f38e3cd7",
        "57bdaae071d4b61081bb58f1a48fe352c3f139ab31472aac36105e4729a05d2d", None),
    HERE / "code_oracle_corpus_v2.pt": (
        "6750a72b4232d4d4687946bb379457210555a660c9ef2e0d4967a63ddfaf2d9d",
        "62adc15414802bc6e181f1fb7be380cfb39052675d39a53ef48ef73b39f03e79", "rows"),
}

PROTECTED_NONROW_FILES = {
    BQ / "early_mlp_state_complete_compiler_v21_final_authority.json":
        "659051ed8e2d34a2d755d1942f4112161294831e724d6697f4c3e2ef466f6987",
    BQ / "early_mlp_state_complete_compiler_v21_final_result.pt":
        "c73f2a7f6099de9e28550b02d7d02904fe37477c65cb8c5c9c6f4beed9bfb5cd",
    BQ / "early_mlp_state_complete_compiler_v21_programs_receipt.json":
        "c9c67bdd14a34dd83192a02d49705d0ed7043e2f9751d042250f44395f88ec2c",
    BQ / "early_mlp_state_complete_compiler_v21_programs.pt":
        "36a8e5203ec72d8c8f30909dba9241d1bf2a4a2d3fd980d8c558e28c3c0b614e",
    BQ / "joint_early_mlp_pca_composition_authoritative_v3_bases.pt":
        "0eee01f39087548a479486d068404f78c4bdc2fd930932add162212da31fe4d9",
    Path("/workspace/runs/bilin18_frozen_ship_v2_manifest.json"):
        "21c89c4d1bd03e1c4be34023781c027b13d2c98202b855938488e33c99e9ba04",
    Path("/workspace/runs/bilin18_frozen_ship_v2.pt"):
        "fe21ead35b1dcb3c0914a36b04d7be36e9c3f179c57bc63eee62bd78d34fe9df",
}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_raw_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor_raw_sha256 requires a tensor")
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def _hash_strings(values: set[str]) -> str:
    return _sha256_json(sorted(values))


def _hash_integers(values: set[int]) -> str:
    return _sha256_json(sorted(values))


def _hash_rows(values: set[tuple[int, ...]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(values):
        digest.update(len(row).to_bytes(4, "little", signed=False))
        for token in row:
            digest.update(int(token).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _row_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


@dataclass(frozen=True)
class IdentitySets:
    documents: frozenset[str]
    dataset_indices: frozenset[int]
    full_rows: frozenset[tuple[int, ...]]
    prefix32: frozenset[tuple[int, ...]]

    @classmethod
    def empty(cls) -> "IdentitySets":
        return cls(frozenset(), frozenset(), frozenset(), frozenset())

    def counts(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "dataset_indices": len(self.dataset_indices),
            "full_rows": len(self.full_rows),
            "prefix32": len(self.prefix32),
        }

    def hashes(self) -> dict[str, str]:
        return {
            "documents": _hash_strings(set(self.documents)),
            "dataset_indices": _hash_integers(set(self.dataset_indices)),
            "full_rows": _hash_rows(set(self.full_rows)),
            "prefix32": _hash_rows(set(self.prefix32)),
        }


def _records_from_registry(payload: Any) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if not isinstance(payload, dict):
        return records
    sets = payload.get("document_provenance", {}).get("sets", {})
    if isinstance(sets, dict):
        for values in sets.values():
            if not isinstance(values, list):
                raise RuntimeError("document provenance set is not a list")
            records.extend(value for value in values if isinstance(value, dict))
    # Code OOD uses row_provenance and is excluded primarily by token identity.
    code_sets = payload.get("row_provenance", {})
    if isinstance(code_sets, dict):
        for values in code_sets.values():
            if not isinstance(values, list):
                raise RuntimeError("code row provenance set is not a list")
            records.extend(value for value in values if isinstance(value, dict))
    return records


def load_canonical_prior(
    *,
    registry_files: Mapping[Path, str] | None = None,
    row_tensors: Mapping[Path, tuple[str, str, str | None]] | None = None,
    protected_files: Mapping[Path, str] | None = None,
) -> tuple[IdentitySets, dict[str, Any]]:
    """Hash and census the exact prospective registry allowlist.

    Prior-vs-prior overlap is intentionally unioned, not rejected: compiler-v2.1
    prospectively remaps two compiler-v2 row tensors.  Only candidate-vs-union and
    candidate-role-vs-role overlap is scientifically relevant.
    """
    registries = dict(CANONICAL_REGISTRY_FILES if registry_files is None else registry_files)
    tensors = dict(CANONICAL_ROW_TENSORS if row_tensors is None else row_tensors)
    protected = dict(PROTECTED_NONROW_FILES if protected_files is None else protected_files)
    documents: set[str] = set()
    indices: set[int] = set()
    full_rows: set[tuple[int, ...]] = set()
    prefixes: set[tuple[int, ...]] = set()
    observed_registry_hashes: dict[str, str] = {}
    observed_tensor_hashes: dict[str, dict[str, str]] = {}
    observed_protected_hashes: dict[str, str] = {}

    for path, expected in sorted(registries.items(), key=lambda item: str(item[0])):
        if not path.is_file():
            raise RuntimeError(f"canonical registry is missing: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(f"canonical registry hash changed: {path}")
        observed_registry_hashes[str(path.resolve())] = observed
        payload = json.loads(path.read_text())
        for record in _records_from_registry(payload):
            document = record.get("document_id")
            index = record.get("dataset_document_index")
            if isinstance(document, str) and document:
                documents.add(document)
            if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
                indices.add(index)

    for path, (file_expected, tensor_expected, payload_key) in sorted(
        tensors.items(), key=lambda item: str(item[0])
    ):
        if not path.is_file():
            raise RuntimeError(f"canonical row tensor is missing: {path}")
        before = file_sha256(path)
        if before != file_expected:
            raise RuntimeError(f"canonical row file hash changed: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        after = file_sha256(path)
        if after != before:
            raise RuntimeError(f"canonical row file changed while loading: {path}")
        value = payload[payload_key] if payload_key is not None \
            and isinstance(payload, dict) and payload_key in payload else payload
        if not isinstance(value, torch.Tensor) or value.dtype != torch.long \
                or value.ndim != 2 or value.shape[1] < 32:
            raise RuntimeError(f"canonical row payload has invalid shape or dtype: {path}")
        raw_hash = tensor_raw_sha256(value)
        if raw_hash != tensor_expected:
            raise RuntimeError(f"canonical row tensor hash changed: {path}")
        observed_tensor_hashes[str(path.resolve())] = {
            "file_sha256": before, "tensor_raw_sha256": raw_hash,
        }
        for row in value:
            values = tuple(int(token) for token in row.tolist())
            full_rows.add(values)
            prefixes.add(values[:32])

    for path, expected in sorted(protected.items(), key=lambda item: str(item[0])):
        if not path.is_file():
            raise RuntimeError(f"protected inherited object is missing: {path}")
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(f"protected inherited object hash changed: {path}")
        observed_protected_hashes[str(path.resolve())] = observed

    identities = IdentitySets(
        documents=frozenset(documents),
        dataset_indices=frozenset(indices),
        full_rows=frozenset(full_rows),
        prefix32=frozenset(prefixes),
    )
    census = {
        "registry_files": observed_registry_hashes,
        "row_tensors": observed_tensor_hashes,
        "protected_nonrow_files": observed_protected_hashes,
        "identity_counts": identities.counts(),
        "identity_hashes": identities.hashes(),
        "discovery_rule": "exact_prospective_allowlist_no_recursive_discovery",
    }
    return identities, census


def _required_rows(triple: CandidateTriple) -> dict[str, int]:
    return {
        "fit": triple.fit_n,
        "validation": triple.validation_n,
        "final": triple.final_n,
    }


def _validate_record(record: Mapping[str, Any], row_index: int) -> tuple[str, int]:
    required = {
        "document_id", "dataset_document_index", "chunk_id", "token_start",
    }
    if set(record) != required:
        raise RuntimeError(
            f"row {row_index} provenance schema changed: {sorted(record)}"
        )
    document = record["document_id"]
    dataset_index = record["dataset_document_index"]
    chunk_id = record["chunk_id"]
    token_start = record["token_start"]
    if not isinstance(document, str) or not document:
        raise RuntimeError(f"row {row_index} has invalid document_id")
    if isinstance(dataset_index, bool) or not isinstance(dataset_index, int) \
            or dataset_index < 0:
        raise RuntimeError(f"row {row_index} has invalid dataset_document_index")
    if isinstance(chunk_id, bool) or not isinstance(chunk_id, int) or chunk_id < 0:
        raise RuntimeError(f"row {row_index} has invalid chunk_id")
    if isinstance(token_start, bool) or not isinstance(token_start, int) \
            or token_start != chunk_id * TOKEN_LENGTH:
        raise RuntimeError(f"row {row_index} has inconsistent token_start")
    return document, dataset_index


def role_identities(
    *,
    role: str,
    rows: torch.Tensor,
    records: Sequence[Mapping[str, Any]],
    triple: CandidateTriple,
) -> tuple[IdentitySets, dict[str, int], dict[str, str]]:
    """Validate one role and return identity sets plus duplicate diagnostics."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    expected = _required_rows(triple)[role]
    minimum_index = {
        "fit": triple.fit_skip,
        "validation": triple.validation_skip,
        "final": triple.final_skip,
    }[role]
    if not isinstance(rows, torch.Tensor) or rows.dtype != torch.long \
            or rows.ndim != 2 or tuple(rows.shape) != (expected, TOKEN_LENGTH):
        raise RuntimeError(
            f"{role} rows must be long[{expected},{TOKEN_LENGTH}], got "
            f"{getattr(rows, 'dtype', None)} {getattr(rows, 'shape', None)}"
        )
    if len(records) != expected:
        raise RuntimeError(f"{role} provenance count {len(records)} != {expected}")

    documents: set[str] = set()
    indices: set[int] = set()
    document_to_index: dict[str, int] = {}
    index_to_document: dict[int, str] = {}
    row_values: list[tuple[int, ...]] = []
    prefixes: list[tuple[int, ...]] = []
    provenance_units: set[tuple[str, int, int]] = set()
    ordered_provenance: list[list[Any]] = []
    ordered_bindings: list[list[Any]] = []
    previous_position: tuple[int, int] | None = None
    for offset, (row, record) in enumerate(zip(rows, records, strict=True)):
        document, dataset_index = _validate_record(record, offset)
        chunk_id = int(record["chunk_id"])
        if dataset_index < minimum_index:
            raise RuntimeError(f"{role} provenance precedes its registered skip")
        position = (dataset_index, chunk_id)
        if previous_position is not None and position <= previous_position:
            raise RuntimeError(f"{role} provenance is not in canonical source order")
        previous_position = position
        if document in document_to_index and document_to_index[document] != dataset_index:
            raise RuntimeError(f"{role} maps one document_id to multiple dataset indices")
        if dataset_index in index_to_document and index_to_document[dataset_index] != document:
            raise RuntimeError(f"{role} maps one dataset index to multiple document_ids")
        document_to_index[document] = dataset_index
        index_to_document[dataset_index] = document
        unit = (document, dataset_index, int(record["chunk_id"]))
        if unit in provenance_units:
            raise RuntimeError(f"{role} repeats a document/chunk provenance unit")
        provenance_units.add(unit)
        documents.add(document)
        indices.add(dataset_index)
        values = tuple(int(token) for token in row.tolist())
        if min(values) < 0 or max(values) >= 50_257:
            raise RuntimeError(f"{role} contains an out-of-vocabulary token")
        row_values.append(values)
        prefixes.append(values[:32])
        provenance = [
            document, dataset_index, chunk_id, int(record["token_start"]),
        ]
        ordered_provenance.append(provenance)
        ordered_bindings.append([*provenance, _row_sha256(row)])

    identities = IdentitySets(
        documents=frozenset(documents),
        dataset_indices=frozenset(indices),
        full_rows=frozenset(row_values),
        prefix32=frozenset(prefixes),
    )
    duplicates = {
        "full_rows": len(row_values) - len(identities.full_rows),
        "prefix32": len(prefixes) - len(identities.prefix32),
    }
    ordered_hashes = {
        "ordered_tensor_raw": tensor_raw_sha256(rows),
        "ordered_provenance": _sha256_json(ordered_provenance),
        "ordered_row_provenance_binding": _sha256_json(ordered_bindings),
    }
    return identities, duplicates, ordered_hashes


def _intersection_counts(left: IdentitySets, right: IdentitySets) -> dict[str, int]:
    return {
        "documents": len(left.documents & right.documents),
        "dataset_indices": len(left.dataset_indices & right.dataset_indices),
        "full_rows": len(left.full_rows & right.full_rows),
        "prefix32": len(left.prefix32 & right.prefix32),
    }


def adjudicate_candidate(
    *,
    candidate_index: int,
    rows_by_role: Mapping[str, torch.Tensor],
    records_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    prior: IdentitySets,
) -> dict[str, Any]:
    """Return a hash-only collision decision for one frozen candidate triple."""
    if set(rows_by_role) != set(ROLES) or set(records_by_role) != set(ROLES):
        raise RuntimeError("candidate must contain exactly fit, validation, and final roles")
    triple = candidate_triple(candidate_index)
    identities: dict[str, IdentitySets] = {}
    duplicate_counts: dict[str, dict[str, int]] = {}
    ordered_hashes: dict[str, dict[str, str]] = {}
    for role in ROLES:
        identities[role], duplicate_counts[role], ordered_hashes[role] = role_identities(
            role=role,
            rows=rows_by_role[role],
            records=records_by_role[role],
            triple=triple,
        )

    prior_collisions = {
        role: _intersection_counts(identities[role], prior) for role in ROLES
    }
    pair_collisions: dict[str, dict[str, int]] = {}
    for left, right in (("fit", "validation"), ("fit", "final"),
                        ("validation", "final")):
        pair_collisions[f"{left}__{right}"] = _intersection_counts(
            identities[left], identities[right]
        )
    collision_evidence_count = sum(
        sum(values.values()) for values in prior_collisions.values()
    ) + sum(sum(values.values()) for values in pair_collisions.values()) \
        + sum(sum(values.values()) for values in duplicate_counts.values())
    accepted = collision_evidence_count == 0
    return {
        "schema_version": 1,
        "candidate_index": candidate_index,
        "candidate": {
            "fit": {"n": triple.fit_n, "skip": triple.fit_skip},
            "validation": {"n": triple.validation_n, "skip": triple.validation_skip},
            "final": {"n": triple.final_n, "skip": triple.final_skip},
        },
        "accepted": accepted,
        "collision_evidence_count": collision_evidence_count,
        "collision_evidence_semantics": COLLISION_EVIDENCE_SEMANTICS,
        "role_identity_counts": {
            role: identities[role].counts() for role in ROLES
        },
        "role_identity_hashes": {
            role: {**identities[role].hashes(), **ordered_hashes[role]} for role in ROLES
        },
        "internal_duplicate_counts": duplicate_counts,
        "prior_collision_counts": prior_collisions,
        "cross_role_collision_counts": pair_collisions,
        "privacy_rule": PRIVACY_RULE,
    }


def _exact_keys(value: Any, expected: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        observed = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise RuntimeError(f"{context} schema changed: {observed}")
    return value


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{context} must be a literal nonnegative integer")
    return value


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"{context} must be a lowercase SHA256")
    return value


def validate_collision_report(report: Mapping[str, Any], expected_index: int) -> None:
    top = _exact_keys(report, {
        "schema_version", "candidate_index", "candidate", "accepted",
        "collision_evidence_count", "collision_evidence_semantics",
        "role_identity_counts", "role_identity_hashes", "internal_duplicate_counts",
        "prior_collision_counts", "cross_role_collision_counts", "privacy_rule",
    }, "collision report")
    if top["schema_version"] != 1:
        raise RuntimeError("collision report schema version changed")
    if _nonnegative_int(top["candidate_index"], "candidate_index") != expected_index:
        raise RuntimeError("collision history is not contiguous and ordered")
    triple = candidate_triple(expected_index)
    expected_candidate = {
        "fit": {"n": triple.fit_n, "skip": triple.fit_skip},
        "validation": {"n": triple.validation_n, "skip": triple.validation_skip},
        "final": {"n": triple.final_n, "skip": triple.final_skip},
    }
    if top["candidate"] != expected_candidate:
        raise RuntimeError("collision report candidate schedule changed")
    if top["privacy_rule"] != PRIVACY_RULE:
        raise RuntimeError("collision report privacy rule changed")
    if top["collision_evidence_semantics"] != COLLISION_EVIDENCE_SEMANTICS:
        raise RuntimeError("collision evidence semantics changed")

    counts = _exact_keys(top["role_identity_counts"], set(ROLES), "role counts")
    hashes = _exact_keys(top["role_identity_hashes"], set(ROLES), "role hashes")
    duplicates = _exact_keys(
        top["internal_duplicate_counts"], set(ROLES), "duplicate counts",
    )
    prior = _exact_keys(top["prior_collision_counts"], set(ROLES), "prior counts")
    pairs = _exact_keys(
        top["cross_role_collision_counts"], set(PAIR_KEYS), "cross-role counts",
    )
    required_rows = _required_rows(triple)
    evidence = 0
    for role in ROLES:
        role_counts = _exact_keys(counts[role], set(IDENTITY_KEYS), f"{role} counts")
        for key in IDENTITY_KEYS:
            _nonnegative_int(role_counts[key], f"{role}.{key}")
        if role_counts["documents"] != role_counts["dataset_indices"]:
            raise RuntimeError(f"{role} document/index counts disagree")
        role_duplicates = _exact_keys(
            duplicates[role], {"full_rows", "prefix32"}, f"{role} duplicates",
        )
        for key in ("full_rows", "prefix32"):
            duplicate = _nonnegative_int(role_duplicates[key], f"{role}.{key} duplicates")
            if role_counts[key] + duplicate != required_rows[role]:
                raise RuntimeError(f"{role} unique-plus-duplicate count changed")
            evidence += duplicate
        role_hashes = _exact_keys(
            hashes[role], set(IDENTITY_KEYS + ORDERED_HASH_KEYS), f"{role} hashes",
        )
        for key, value in role_hashes.items():
            _sha256(value, f"{role}.{key}")
        role_prior = _exact_keys(prior[role], set(IDENTITY_KEYS), f"{role} prior")
        for key, value in role_prior.items():
            evidence += _nonnegative_int(value, f"{role}.prior.{key}")
    for pair in PAIR_KEYS:
        pair_counts = _exact_keys(pairs[pair], set(IDENTITY_KEYS), f"{pair} counts")
        for key, value in pair_counts.items():
            evidence += _nonnegative_int(value, f"{pair}.{key}")
    declared = _nonnegative_int(
        top["collision_evidence_count"], "collision_evidence_count",
    )
    if declared != evidence:
        raise RuntimeError("collision evidence count does not recompute")
    accepted = top["accepted"]
    if not isinstance(accepted, bool):
        raise RuntimeError("collision decision must be a literal boolean")
    if accepted != (declared == 0):
        raise RuntimeError("collision decision disagrees with collision evidence")


def validate_collision_history(
    reports: Sequence[Mapping[str, Any]], chosen_index: int,
) -> None:
    """Require every earlier candidate to be rejected and the chosen one accepted."""
    if isinstance(chosen_index, bool) or not isinstance(chosen_index, int) \
            or chosen_index < 0:
        raise ValueError("chosen_index must be a nonnegative integer")
    if len(reports) != chosen_index + 1:
        raise RuntimeError("collision history must contain candidates 0..chosen_index")
    for expected, report in enumerate(reports):
        validate_collision_report(report, expected)
        accepted = report["accepted"]
        if expected < chosen_index and accepted:
            raise RuntimeError("selection skipped an earlier collision-free candidate")
        if expected == chosen_index and not accepted:
            raise RuntimeError("chosen candidate is not collision-free")


def collision_history_hash(reports: Sequence[Mapping[str, Any]]) -> str:
    if not reports:
        raise RuntimeError("collision history is empty")
    validate_collision_history(reports, len(reports) - 1)
    return _sha256_json(list(reports))
