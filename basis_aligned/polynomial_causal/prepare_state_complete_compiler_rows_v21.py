#!/usr/bin/env python3
"""Prospectively remap v2 rows and freeze a new final split for compiler v2.1."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
PROTOCOL = HERE / "early_mlp_state_complete_compiler_v21_preregistration.json"
IMPLEMENTATION_AMENDMENT = (
    HERE / "early_mlp_state_complete_compiler_v21_implementation_amendment.json"
)
IMPLEMENTATION_AMENDMENT_SHA256 = (
    "0ac129411a89b94867635253493e53f04aef77e0361bea2fe6af589b7ae5d988"
)
OLD_RECEIPT = BQ / "early_mlp_state_complete_compiler_v2_rows_receipt.json"
RETRY1_FAILURE = BQ / "early_mlp_state_complete_compiler_v2_site0_retry1_manifest.json"
PARENT_FAILURE = BQ / "early_mlp_state_complete_compiler_v2_site0_manifest.json"
DIAGNOSTIC_RECEIPT = (
    BQ / "early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1_receipt.json"
)
HARVESTER = HERE / "local_fineweb_harvest.py"
DEDUP = BQ / "bilin18_eval_tokens_large.pt"
CENSUS = BQ / "census_lib.py"
RECEIPT = BQ / "early_mlp_state_complete_compiler_v21_rows_receipt.json"
MANIFEST = BQ / "early_mlp_state_complete_compiler_v21_rows_manifest.json"
PROGRAMS_ARTIFACT = BQ / "early_mlp_state_complete_compiler_v21_programs.pt"
PROGRAMS_RECEIPT = BQ / "early_mlp_state_complete_compiler_v21_programs_receipt.json"
SITE0_TRAINING_RECEIPT = BQ / (
    "early_mlp_state_complete_compiler_v21_site0_training_receipt.json"
)
SITE0_LEDGER_ARTIFACT = BQ / "early_mlp_state_complete_compiler_v21_site0_ledger.pt"
SITE0_LEDGER_RECEIPT = BQ / "early_mlp_state_complete_compiler_v21_site0_ledger_receipt.json"
SITE1_LEDGER_ARTIFACT = BQ / "early_mlp_state_complete_compiler_v21_site1_ledger.pt"
SITE1_LEDGER_RECEIPT = BQ / "early_mlp_state_complete_compiler_v21_site1_ledger_receipt.json"
CACHE = BQ / ".rowcache_compiler_v21"
LOCK = Path("/workspace/runs/.early_mlp_state_complete_compiler_v21_rows.lock")
OUTPUTS = (RECEIPT, MANIFEST)
FINAL_SPEC = (192, 39000)
T_LEN = 513
MODEL_LEN = 257
TOKEN_VOCAB = 50257
TOKEN_FREQUENCY_BOUNDARIES = [1, 2, 4, 8, 16, 32, 64, 128]
CAUSAL_CAPTURE_COUNT = 192 * 64
VALIDATION_TOKEN_COUNT = 192 * 192
FIT_CAPTURE_COUNT = 480 * 64
ROWS_RECEIPT_SHA256 = "7fa45e9a9e77e6622167fbf024400177f581cdd40958c1fb722ca13d8fcc018b"
ROWS_MANIFEST_SHA256 = "1cfe4b383c8298ca79164dc42098a9c9e5bce8c47b4411605e1aa1fb2dc44975"

PINS = {
    PROTOCOL: "69c1bb3bdb0cbf576c41775d4a0881c20a1154642a5ac8deacef780fe9b08ee3",
    OLD_RECEIPT: "23319ece1d8542d51e024bde0e2253d740b08ad18ad4f2d8565ba5120473fd82",
    RETRY1_FAILURE: "2eb0ef098a93d5562bb1abd0b3e94187a461e86cc1c3aec055a1bb719632829a",
    PARENT_FAILURE: "0903b0822b935e7dd6225da46dd1e58064ec275b80fd9c599685cea8b8b05f36",
    DIAGNOSTIC_RECEIPT: "fffecb9a3d99a4f6b7f615c96caf9bd7e2ac9c4d4788d610df7edf18d6a1d9fd",
    HARVESTER: "87d9abeaf1182811650c35bcae25b0373687d2e87aede895bc9f2bc440b90b04",
    DEDUP: "bb2b00699e511245bb68069be1fe5559777170fb78a6dc9218830454f38e3cd7",
    CENSUS: "f51c19e83f46dc363a2c5dad1887b55ab5dd9b3684294e940583a6814881cf1f",
}
OLD_IDENTITIES = {
    "compiler_fit": {
        "cache_file_sha256": "cf3abe833dec8ccfc09afef0ff1bdcc74b2ba8a37dc86e9c954c238bb6b7c276",
        "tensor_full_raw_sha256": "fc2b3b8ced2f5449e6494dc9b5127c95717b938b37db74fbcc4d9458e8d39442",
        "tensor_prefix257_raw_sha256": "3d78e2c43406e83af5f724cbb907d4aade430fef5211ee08103471e7f0eb56ad",
        "provenance_records_sha256": "ed56c053521598956ac56bd221303c54a8fba5897da31188f1a7f9e7384015c1",
    },
    "compiler_validation": {
        "cache_file_sha256": "0d66fc0958da4fa8c0aedbe5b4203d474382acf6b5c0ebe77b53a54505a91ac9",
        "tensor_full_raw_sha256": "f415e3b7a148104435592b8e482d875de24ae832603799d42315b415309e6ca2",
        "tensor_prefix257_raw_sha256": "19cb5da7c458cde4565a9b354e183c62deed22706020dfbdafc3f7d003b52c62",
        "provenance_records_sha256": "c6c424e13a8a8181d37d3394bd7a77d7e6f3f1abc4be395453fb9a6414f13799",
    },
    "compiler_final": {
        "cache_file_sha256": "c9de7d6386668f24414018b330f4182a17c4c73fb2babb395c0654a52f9a3acd",
        "tensor_full_raw_sha256": "a58c0a8cfc1ecd27417384470c71f0f8054793976b61833f8af30268a47cd398",
        "tensor_prefix257_raw_sha256": "36c8c38b9e0fd909569aaf582ae42941929d9bba63453c286d78306b12e1f576",
        "provenance_records_sha256": "321378a751d3d51cad61718a1551cc3a755733708313a0c79d31de4fc4bd88a8",
    },
}
ORIGINAL_ABSENT = (
    BQ / "early_mlp_state_complete_compiler_v2_site0_programs.pt",
    BQ / "early_mlp_state_complete_compiler_v2_site0_receipt.json",
    BQ / "early_mlp_state_complete_compiler_v2_site0_results.json",
    BQ / "early_mlp_state_complete_compiler_v2_site0_retry1_programs.pt",
    BQ / "early_mlp_state_complete_compiler_v2_site0_retry1_receipt.json",
    BQ / "early_mlp_state_complete_compiler_v2_site0_retry1_results.json",
)

sys.path.insert(0, str(HERE))
import prepare_state_complete_compiler_rows_v2 as old  # noqa: E402
import early_mlp_affine_compiler_v1 as affine_v1  # noqa: E402
import early_mlp_state_complete_compiler_v2 as compiler  # noqa: E402
import state_complete_compiler_fit_v2 as compiler_fit  # noqa: E402
import state_complete_compiler_selection_v2 as selection  # noqa: E402

PINS.update(old.PRIOR_RECEIPTS)
PINS.update({
    Path(old.__file__): "e24267acc5e0e3d908a33f6d402cf4a693eeeb68e3f143f2a3fe022db7bacb1e",
    HERE / "test_prepare_state_complete_compiler_rows_v2.py": (
        "872dda9bd24d50e72665a64aebd9e47cb87e4faf0b4d65acfc97d7514a86382d"
    ),
})
SOURCE_CLOSURE = tuple(dict.fromkeys((
    Path(__file__),
    HERE / "test_prepare_state_complete_compiler_rows_v21.py",
    PROTOCOL,
    Path(old.__file__),
    HERE / "test_prepare_state_complete_compiler_rows_v2.py",
    HARVESTER,
    CENSUS,
)))
_RETRY1_TRANSITIVE_SOURCES = tuple(
    ROOT / relative
    for relative in json.loads(RETRY1_FAILURE.read_text()).get("source_hashes", {})
)
PROGRAM_SOURCE_CLOSURE = tuple(dict.fromkeys((
    HERE / "early_mlp_state_complete_compiler_v21.py",
    HERE / "test_early_mlp_state_complete_compiler_v21.py",
    HERE / "early_mlp_state_complete_compiler_v21_site0.py",
    HERE / "test_early_mlp_state_complete_compiler_v21_site0.py",
    PROTOCOL,
    IMPLEMENTATION_AMENDMENT,
    Path(__file__),
    HERE / "test_prepare_state_complete_compiler_rows_v21.py",
    *_RETRY1_TRANSITIVE_SOURCES,
)))
PROTECTED = tuple(dict.fromkeys((*PINS, *ORIGINAL_ABSENT)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return old.tensor_sha256(value)


def logical_json_sha256(value: Any) -> str:
    return old.logical_json_sha256(value)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def expected_fit_permutation_sha256(receipt: Mapping[str, Any]) -> str:
    records = receipt.get("document_provenance", {}).get("sets", {}).get(
        "compiler_fit_v21"
    )
    if not isinstance(records, list) or len(records) != 480 or any(
        not isinstance(record, Mapping) or not isinstance(record.get("document_id"), str)
        or not record["document_id"] for record in records
    ):
        raise RuntimeError("v2.1 fit permutation provenance changed")
    row_permutation = compiler_fit.document_block_permutation(
        [record["document_id"] for record in records], compiler_fit.FIT_SEED,
    )
    expanded = compiler_fit.expand_capture_permutation(row_permutation)
    if expanded.dtype != torch.long or tuple(expanded.shape) != (480 * 64,):
        raise RuntimeError("v2.1 expanded fit permutation changed")
    return tensor_sha256(expanded)


def write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_torch_atomic(value: torch.Tensor, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_committed_clean(path: Path) -> None:
    relative = path.resolve().relative_to(ROOT.resolve())
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(relative)], cwd=ROOT,
        capture_output=True, text=True,
    )
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=ROOT,
    )
    if tracked.returncode or dirty.returncode:
        raise RuntimeError(f"v2.1 row source is not committed and clean: {relative}")


def protected_snapshot() -> dict[str, str | None]:
    return {str(path): file_sha256(path) if path.is_file() else None for path in PROTECTED}


def verify_inputs() -> tuple[dict[str, Any], str, dict[str, str]]:
    for path, expected in PINS.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"pinned v2.1 row input changed: {path}")
    if any(path.exists() for path in ORIGINAL_ABSENT):
        raise RuntimeError("failed compiler namespace gained an output")
    source_hashes = {}
    for path in SOURCE_CLOSURE:
        require_committed_clean(path)
        relative = path.resolve().relative_to(ROOT.resolve())
        source_hashes[str(relative)] = file_sha256(path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    origin = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != origin:
        raise RuntimeError("v2.1 row designation requires HEAD==origin/main")
    receipt = json.loads(OLD_RECEIPT.read_text())
    if receipt.get("status") != "frozen_before_any_label_or_gradient_capture":
        raise RuntimeError("old v2 row authority changed")
    for role, identities in OLD_IDENTITIES.items():
        entry = receipt.get("entries", {}).get(role, {})
        records = receipt.get("document_provenance", {}).get("sets", {}).get(role)
        path = Path(entry.get("cache_path", ""))
        if not path.is_file() or file_sha256(path) != identities["cache_file_sha256"]:
            raise RuntimeError(f"old v2 cache identity changed for {role}")
        for key in ("tensor_full_raw_sha256", "tensor_prefix257_raw_sha256"):
            if entry.get(key) != identities[key]:
                raise RuntimeError(f"old v2 tensor receipt changed for {role}:{key}")
        if not isinstance(records, list) or logical_json_sha256(records) != identities[
            "provenance_records_sha256"
        ]:
            raise RuntimeError(f"old v2 provenance changed for {role}")
    return receipt, head, source_hashes


def remapped_entry(
    old_receipt: Mapping[str, Any], source_role: str, designation: str,
) -> dict[str, Any]:
    entry = copy.deepcopy(old_receipt["entries"][source_role])
    entry.update({
        "source_role": source_role,
        "v21_designation": designation,
        "cache_file_sha256": OLD_IDENTITIES[source_role]["cache_file_sha256"],
    })
    return entry


def build(before: Mapping[str, str | None] | None = None) -> dict[str, Any]:
    if any(path.exists() for path in OUTPUTS) or CACHE.exists():
        raise RuntimeError("refusing to overwrite v2.1 rows or receipt")
    before = dict(protected_snapshot() if before is None else before)
    old_receipt, source_commit, source_hashes = verify_inputs()
    gate = old_receipt["ordered_manifest_gate"]
    source = Path(gate["source_local_path"])
    if not source.is_file() or file_sha256(source) != gate["source_sha256"]:
        raise RuntimeError("pinned local FineWeb source changed")

    # This intentionally deserializes old caches only to construct exact exclusion
    # sets.  The receipt records this phase; no model forward or outcome is computed.
    earlier_receipts = [json.loads(path.read_text()) for path in old.PRIOR_RECEIPTS]
    prior = old.prior_exclusions([*earlier_receipts, old_receipt])
    prior_documents, prior_rows, prior_prefixes = prior

    import tiktoken
    import local_fineweb_harvest as harvest

    reference = torch.load(DEDUP, map_location="cpu", weights_only=True)
    seen = {tuple(row[:32].tolist()) for row in reference}
    encoding = tiktoken.get_encoding("gpt2")
    tensors, provenance = harvest.harvest_texts(
        harvest.parquet_texts([source]), (FINAL_SPEC,), encoding.encode_ordinary, seen
    )
    final = tensors[FINAL_SPEC]
    records = provenance[FINAL_SPEC]
    if tuple(final.shape) != (FINAL_SPEC[0], T_LEN) or final.dtype != torch.long:
        raise RuntimeError("invalid v2.1 final tensor")
    if len(records) != FINAL_SPEC[0]:
        raise RuntimeError("invalid v2.1 final provenance count")
    documents = {record["document_id"] for record in records}
    full_rows = {tensor_sha256(row) for row in final}
    prefixes = {tuple(row[:32].tolist()) for row in final}
    if documents & prior_documents or full_rows & prior_rows or prefixes & prior_prefixes:
        raise RuntimeError("v2.1 final overlaps a prior role")

    staging = CACHE.with_name(f"{CACHE.name}.tmp.{os.getpid()}")
    staging.mkdir(parents=True)
    try:
        filename = f"fineweb_n{FINAL_SPEC[0]}_skip{FINAL_SPEC[1]}.pt"
        staged = staging / filename
        write_torch_atomic(final, staged)
        os.replace(staging, CACHE)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    final_path = CACHE / filename
    entries = {
        "compiler_fit_v21": remapped_entry(old_receipt, "compiler_fit", "fit_reuse"),
        "compiler_validation_v21": remapped_entry(
            old_receipt, "compiler_final", "prospective_validation_remap"
        ),
        "compiler_final_v21": {
            "request": {"n": FINAL_SPEC[0], "skip": FINAL_SPEC[1]},
            "shape_full": list(final.shape),
            "shape_model_prefix": [FINAL_SPEC[0], MODEL_LEN],
            "dtype": str(final.dtype),
            "tensor_full_raw_sha256": tensor_sha256(final),
            "tensor_prefix257_raw_sha256": tensor_sha256(final[:, :MODEL_LEN]),
            "unique_document_count": len(documents),
            "document_ids_sha256": logical_json_sha256(
                [record["document_id"] for record in records]
            ),
            "provenance_records_sha256": logical_json_sha256(records),
            "cache_path": str(final_path.resolve()),
            "cache_file_sha256": file_sha256(final_path),
            "prior_document_overlap": 0,
            "prior_full_row_overlap": 0,
            "prior_prefix32_overlap": 0,
        },
    }
    receipt = {
        "schema_version": 1,
        "receipt_kind": "early_mlp_state_complete_compiler_v21_rows",
        "status": "frozen_before_any_v21_validation_model_forward",
        "authority": "compiler_v21_prospective_role_designation",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "training_license_sites": [],
        "role_licenses": {
            "compiler_fit_v21": {
                "training": True, "selection": False, "final_scoring": False
            },
            "compiler_validation_v21": {
                "training": False, "selection": True, "final_scoring": False
            },
            "compiler_final_v21": {
                "training": False, "selection": False, "final_scoring": True,
                "requires_final_unlock_authority": True
            },
            "old_compiler_validation": "forbidden"
        },
        "protocol_sha256": PINS[PROTOCOL],
        "old_rows_receipt_sha256": PINS[OLD_RECEIPT],
        "retry1_failure_manifest_sha256": PINS[RETRY1_FAILURE],
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "role_designation": {
            "compiler_fit_v21": "old compiler_fit; training reuse only",
            "compiler_validation_v21": "old compiler_final; never again final",
            "compiler_final_v21": "new n192 skip39000; final only",
            "forbidden_old_compiler_validation": "spent; identity pinned; never load in v2.1 runner",
        },
        "provenance_correction": (
            "Old compiler_final was deserialized in preflights for integrity/disjointness "
            "only; it was never model-forwarded, scored, fit, selected on, or outcome-summarized."
        ),
        "exclusion_only_deserialization": {
            "performed": True,
            "roles": "every role in both earlier row authorities and all three old v2 roles",
            "purpose": "construct pinned document/full-row/prefix32 exclusion sets only",
            "model_forward": False,
        },
        "entries": entries,
        "spent_validation_identity": {
            **copy.deepcopy(old_receipt["entries"]["compiler_validation"]),
            "cache_file_sha256": OLD_IDENTITIES["compiler_validation"]["cache_file_sha256"],
            "v21_license": "forbidden",
        },
        "document_provenance": {
            "schema_version": 1,
            "sets": {
                "compiler_fit_v21": copy.deepcopy(
                    old_receipt["document_provenance"]["sets"]["compiler_fit"]
                ),
                "compiler_validation_v21": copy.deepcopy(
                    old_receipt["document_provenance"]["sets"]["compiler_final"]
                ),
                "compiler_final_v21": records,
            },
        },
        "ordered_manifest_gate": gate,
        "source_identity_rechecked": {
            "path": str(source.resolve()),
            "size": source.stat().st_size,
            "sha256": file_sha256(source),
        },
        "implementation_hashes": {
            "row_preparer": file_sha256(Path(__file__)),
            "old_row_preparer": file_sha256(Path(old.__file__)),
            "local_harvester": PINS[HARVESTER],
            "census_lib": PINS[CENSUS],
            "dedup_reference_file": PINS[DEDUP],
            "gpt2_encoding": harvest.encoding_fingerprint(encoding),
        },
        "disjointness_gates": {
            "old_fit_mapped_validation_pairwise_from_pinned_v2_receipt": True,
            "new_final_document_disjoint_from_every_prior_role": True,
            "new_final_full_row_disjoint_from_every_prior_role": True,
            "new_final_prefix32_disjoint_from_every_prior_role": True,
            "old_v2_roles_disjoint_from_every_earlier_authority": True,
        },
        "loader_semantics": (
            "Byte-validate every cache and provenance set; deserialize only requested roles. "
            "Fit/selection may request fit+validation only; final is forbidden until program freeze."
        ),
    }
    _, final_commit, final_source_hashes = verify_inputs()
    if final_commit != source_commit or final_source_hashes != source_hashes:
        raise RuntimeError("v2.1 row source closure changed during harvest")
    final_entry = receipt["entries"]["compiler_final_v21"]
    if file_sha256(Path(final_entry["cache_path"])) != final_entry["cache_file_sha256"]:
        raise RuntimeError("v2.1 final cache changed before row authority")
    after = protected_snapshot()
    if after != before:
        raise RuntimeError("protected compiler lineage changed during v2.1 row harvest")
    manifest = {
        "schema_version": 1,
        "status": "completed_v21_rows_pending_last_written_receipt",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "protocol_sha256": PINS[PROTOCOL],
        "retry1_failure_manifest_sha256": PINS[RETRY1_FAILURE],
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "final_cache_path": final_entry["cache_path"],
        "final_cache_sha256": final_entry["cache_file_sha256"],
        "protected_before": before,
        "protected_after": after,
    }
    write_json_atomic(manifest, MANIFEST)
    receipt["manifest_sha256"] = file_sha256(MANIFEST)
    receipt["last_written_authority_rule"] = (
        "Receipt is written only after repeated pin/source/absence/cache/protected checks."
    )
    write_json_atomic(receipt, RECEIPT)
    return receipt


def _candidate_specs() -> dict[str, tuple[str, str, float | None, int]]:
    specs: dict[str, tuple[str, str, float | None, int]] = {}
    affine_families = {
        "A": ("A_v1_like_z_only_affine_euclidean", "z_only_c"),
        "B": ("B_state_complete_affine_euclidean", "state_complete_p"),
        "C": ("C_state_complete_affine_causal", "state_complete_p"),
    }
    for prefix, (family, interface) in affine_families.items():
        for lambda_index, ridge in enumerate(affine_v1.LAMBDA_GRID):
            for rank in affine_v1.RANK_GRID:
                specs[f"{prefix}_l{lambda_index}_r{rank}"] = (
                    family, interface, float(ridge), int(rank),
                )
    for prefix, family in (
        ("D", "D_state_complete_native_euclidean"),
        ("E", "E_state_complete_native_causal"),
    ):
        for k in compiler.NATIVE_K_GRID:
            specs[f"{prefix}_k{k}"] = (
                family, "state_complete_p", None, int(k),
            )
    if len(specs) != 108:
        raise RuntimeError("registered compiler candidate grid is not 108 cells")
    return specs


def _finite_tensor(
    value: Any, shape: tuple[int, ...], *, floating: bool = True,
) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise RuntimeError(f"program tensor shape changed: expected {shape}")
    if floating and value.dtype != torch.float32:
        raise RuntimeError("serialized program tensor is not float32")
    if not floating and value.dtype != torch.long:
        raise RuntimeError("program index tensor is not int64")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("program tensor is nonfinite")
    return value


def _validate_candidate_state(
    name: str, state: Any, spec: tuple[str, str, float | None, int],
) -> None:
    if not isinstance(state, Mapping):
        raise RuntimeError(f"candidate state is not a mapping: {name}")
    family, interface, ridge, size = spec
    if state.get("family") != family or state.get("interface") != interface:
        raise RuntimeError(f"candidate family/interface changed: {name}")
    if ridge is not None:
        if state.get("grammar") != "affine" or state.get("rank") != size or (
            state.get("lambda") != ridge
        ):
            raise RuntimeError(f"affine candidate metadata changed: {name}")
        _finite_tensor(state.get("mean"), (compiler.D_MODEL,))
        scale = _finite_tensor(state.get("scale"), (compiler.D_MODEL,))
        if bool((scale <= 0).any()):
            raise RuntimeError(f"affine candidate has nonpositive scale: {name}")
        _finite_tensor(state.get("bias"), (compiler.COEFFICIENT_DIM,))
        _finite_tensor(state.get("left"), (compiler.D_MODEL, size))
        _finite_tensor(state.get("right"), (size, compiler.COEFFICIENT_DIM))
    else:
        if state.get("grammar") != "native" or state.get("k") != size:
            raise RuntimeError(f"native candidate metadata changed: {name}")
        _finite_tensor(state.get("left"), (size, compiler.D_MODEL))
        _finite_tensor(state.get("right"), (size, compiler.D_MODEL))
        _finite_tensor(
            state.get("projected_decoder"), (size, compiler.COEFFICIENT_DIM),
        )
        _finite_tensor(state.get("beta"), (compiler.COEFFICIENT_DIM,))
        indices = _finite_tensor(state.get("indices"), (size,), floating=False)
        if bool((indices < 0).any()) or bool((indices >= compiler.NATIVE_PRODUCTS).any()) or (
            indices.unique().numel() != size
        ):
            raise RuntimeError(f"native candidate indices are invalid: {name}")


def state_logical_sha256(state: Mapping[str, Any]) -> str:
    logical = {}
    for key in sorted(state):
        value = state[key]
        logical[key] = ({
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": tensor_sha256(value),
        } if torch.is_tensor(value) else value)
    return logical_json_sha256(logical)


def _validate_full_native_control(
    value: Any, site: int, *, context: str,
    upstream_state_sha256: str, validation_document_ids_sha256: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "state", "context", "upstream_state_sha256",
        "validation_document_ids_sha256", "scorer", "integrity_gates",
        "observed", "measurement_sha256",
    }:
        raise RuntimeError(f"full-native site{site} control schema changed")
    if value.get("context") != context or value.get(
        "upstream_state_sha256"
    ) != upstream_state_sha256 or value.get(
        "validation_document_ids_sha256"
    ) != validation_document_ids_sha256 or value.get("scorer") != (
        "CUDA float32 per-token; float64 row/aggregate"
    ):
        raise RuntimeError(f"full-native site{site} context binding changed")
    state = value["state"]
    spec = (
        "full_native_ceiling_control", "state_complete_p", None,
        compiler.NATIVE_PRODUCTS,
    )
    _validate_candidate_state(f"full_native_site{site}", state, spec)
    gates = value["integrity_gates"]
    if gates != {
        "algebra_identity": True,
        "physical_identity": True,
        "poison_zero_original_calls": True,
        "row_ce_identity": True,
    }:
        raise RuntimeError(f"full-native site{site} integrity gates changed")
    observed = value["observed"]
    if not isinstance(observed, Mapping) or set(observed) != {
        "physical_max_abs_error", "physical_reference_scale", "physical_tolerance",
        "target_original_mlp_calls", "capture_call_counters",
        "scored_arm_call_counters", "max_row_ce_abs_error",
    }:
        raise RuntimeError(f"full-native site{site} observations changed")
    physical = float(observed["physical_max_abs_error"])
    physical_scale = float(observed["physical_reference_scale"])
    physical_tolerance = float(observed["physical_tolerance"])
    row_ce = float(observed["max_row_ce_abs_error"])
    expected_tolerance = 4e-6 * max(1.0, physical_scale)
    if not torch.isfinite(torch.tensor([
        physical, physical_scale, physical_tolerance, row_ce,
    ])).all() or physical_scale < 0 or not abs(
        physical_tolerance - expected_tolerance
    ) <= 1e-12 * max(1.0, expected_tolerance) or (
        physical > physical_tolerance or row_ce > 2e-6
        or observed["target_original_mlp_calls"] != 0
    ):
        raise RuntimeError(f"full-native site{site} observations fail the frozen gate")
    capture_calls = observed["capture_call_counters"]
    scored_calls = observed["scored_arm_call_counters"]
    expected_capture = ({0: 24, 1: 0, 2: 0} if site == 0 else {
        0: 0, 1: 24, 2: 0,
    })
    expected_scored = ({0: 0, 1: 24, 2: 0} if site == 0 else {
        0: 0, 1: 0, 2: 0,
    })
    if capture_calls != expected_capture or scored_calls != expected_scored:
        raise RuntimeError(f"full-native site{site} call counters changed")
    measurement = {
        "context": context,
        "upstream_state_sha256": upstream_state_sha256,
        "validation_document_ids_sha256": validation_document_ids_sha256,
        "scorer": value["scorer"],
        "state_sha256": state_logical_sha256(state),
        "integrity_gates": gates,
        "observed": observed,
    }
    if value.get("measurement_sha256") != logical_json_sha256(measurement):
        raise RuntimeError(f"full-native site{site} measurement binding changed")


def _same_value(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return bool(
            torch.is_tensor(left) and torch.is_tensor(right)
            and left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return bool(
            isinstance(left, Mapping) and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_same_value(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return bool(
            isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_same_value(a, b) for a, b in zip(left, right))
        )
    return left == right


def _total_shuffle_selection(candidates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for name, candidate in candidates.items():
        state, metrics = candidate["state"], candidate["metrics"]
        recovery = float(metrics["recovery"])
        if state["family"] in selection.FAMILY_ORDER and torch.isfinite(
            torch.tensor(recovery)
        ):
            rows.append({"name": name, "state": state, "metrics": metrics})
    if not rows:
        raise RuntimeError("v2.1 shuffle bank has no finite B-E candidate")
    best = max(float(row["metrics"]["recovery"]) for row in rows)
    threshold = 0.99 * best if best > 0 else best
    near = [row for row in rows if float(row["metrics"]["recovery"]) >= threshold]
    selected = min(near, key=selection.candidate_key)
    return {
        "selector": "selection_permissive_nondeployable_shuffle_null",
        "selected": selected["name"],
        "selected_family": selected["state"]["family"],
        "best_signed_recovery": best,
        "recovery_slack": 0.99,
        "copy_is_eligibility": False,
        "eligible": sorted(row["name"] for row in near),
    }


def _constant_price() -> dict[str, int]:
    basis_reals = compiler.D_MODEL * compiler.COEFFICIENT_DIM
    return {
        "basis_reals": basis_reals,
        "program_reals": compiler.COEFFICIENT_DIM,
        "total_reals": basis_reals + compiler.COEFFICIENT_DIM,
    }


def _pipeline_price(site0: Mapping[str, Any], site1: Mapping[str, Any]) -> dict[str, Any]:
    prices = [selection.state_price(site0), selection.state_price(site1)]
    return {
        "site0": prices[0], "site1": prices[1],
        "total_reals": int(sum(price["total_reals"] for price in prices)),
    }


def derive_token_frequency_strata(
    fit_rows: torch.Tensor, validation_rows: torch.Tensor, boundaries: list[int],
) -> dict[str, Any]:
    if tuple(fit_rows.shape[1:]) != (T_LEN,) or tuple(
        validation_rows.shape[1:]
    ) != (T_LEN,) or fit_rows.dtype != torch.long or validation_rows.dtype != torch.long:
        raise RuntimeError("v2.1 token-frequency rows have wrong shape or dtype")
    if boundaries != TOKEN_FREQUENCY_BOUNDARIES:
        raise RuntimeError("v2.1 token-frequency boundaries changed")
    fit_targets = fit_rows[:, 65:MODEL_LEN].contiguous()
    validation_targets = validation_rows[:, 65:MODEL_LEN].contiguous()
    if bool((fit_targets < 0).any()) or bool((fit_targets >= TOKEN_VOCAB).any()) or bool(
        (validation_targets < 0).any()
    ) or bool((validation_targets >= TOKEN_VOCAB).any()):
        raise RuntimeError("v2.1 token-frequency target is outside GPT-2 vocabulary")
    fit_counts = torch.bincount(fit_targets.flatten(), minlength=TOKEN_VOCAB).long()
    boundary_tensor = torch.tensor(boundaries, dtype=torch.long)
    assignments = torch.bucketize(
        fit_counts.index_select(0, validation_targets.flatten()),
        boundary_tensor, right=True,
    ).view_as(validation_targets)
    counts = torch.bincount(assignments.flatten(), minlength=len(boundaries) + 1).tolist()
    return {
        "rule": "fit target counts at positions 64..255; torch.bucketize right=True",
        "boundaries": boundaries,
        "counts": counts,
        "fit_token_counts_sha256": tensor_sha256(fit_counts),
        "validation_assignment_sha256": tensor_sha256(assignments),
    }


def derive_causal_weights(
    omission_losses: torch.Tensor, target_second_moments: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    for name, value in (
        ("omission losses", omission_losses),
        ("target second moments", target_second_moments),
    ):
        if not torch.is_tensor(value) or value.dtype != torch.float64 or tuple(
            value.shape
        ) != (compiler.COEFFICIENT_DIM,) or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"v2.1 causal {name} changed")
    losses = omission_losses
    moments = target_second_moments
    if bool((moments < 0).any()):
        raise RuntimeError("v2.1 target coefficient second moments are negative")
    raw = losses.abs() / moments.clamp_min(1e-12)
    positive = raw[raw > 0]
    if positive.numel() == 0:
        raise RuntimeError("v2.1 causal omission audit has no positive direction")
    floor = float(torch.quantile(positive, 0.05))
    weights = raw.clamp_min(floor)
    weights = (weights / weights.mean()).float().contiguous()
    return floor, weights


def derive_causal_audit(
    full_oracle_row_ce: torch.Tensor, omit_row_ce: torch.Tensor,
    target_p_square_sums: torch.Tensor, target_p_count: int,
) -> dict[str, Any]:
    for name, value, shape in (
        ("full oracle row CE", full_oracle_row_ce, (192,)),
        ("omit row CE", omit_row_ce, (compiler.COEFFICIENT_DIM, 192)),
        ("target p square sums", target_p_square_sums, (compiler.COEFFICIENT_DIM,)),
    ):
        if not torch.is_tensor(value) or value.dtype != torch.float64 or tuple(
            value.shape
        ) != shape or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"v2.1 causal audit {name} changed")
    if target_p_count != CAUSAL_CAPTURE_COUNT or bool(
        (target_p_square_sums < 0).any()
    ):
        raise RuntimeError("v2.1 causal audit target sufficient statistics changed")
    omission_losses = (
        omit_row_ce.mean(dim=1) - full_oracle_row_ce.mean()
    ).double().contiguous()
    moments = (target_p_square_sums / target_p_count).double().contiguous()
    floor, weights = derive_causal_weights(omission_losses, moments)
    return {
        "omission_losses": omission_losses,
        "target_second_moments": moments,
        "positive_floor": floor,
        "weights": weights,
    }


def _load_role_cache_for_strata(receipt: Mapping[str, Any], role: str) -> torch.Tensor:
    entry = receipt.get("entries", {}).get(role, {})
    path = Path(entry.get("cache_path", ""))
    if not path.is_file() or file_sha256(path) != entry.get("cache_file_sha256"):
        raise RuntimeError(f"v2.1 strata role cache changed: {role}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    expected_rows = 480 if role == "compiler_fit_v21" else 192
    if tuple(value.shape) != (expected_rows, T_LEN) or value.dtype != torch.long:
        raise RuntimeError(f"v2.1 strata role cache shape changed: {role}")
    return value


def _validate_exact_call_counter(value: Any, expected: Mapping[int, int], label: str) -> None:
    if not isinstance(value, Mapping) or value != expected:
        raise RuntimeError(f"v2.1 {label} original-MLP call counter changed")


def _validate_context_diagnostics(
    context: Any, *, stage: str, name: str, expected_upstream: str,
    candidates: Mapping[str, Any],
) -> None:
    required = {
        "upstream_state_sha256", "scorer", "teacher_denominator",
        "teacher_kl_sum", "teacher_token_count", "copy_baseline",
        "copy_ce_sum", "copy_token_count", "call_counters",
    }
    if not isinstance(context, Mapping) or set(context) != required or context.get(
        "upstream_state_sha256"
    ) != expected_upstream or context.get(
        "scorer"
    ) != "CUDA float32 per-token; float64 row/aggregate":
        raise RuntimeError(f"v2.1 {name} context binding changed")
    denominator = float(context["teacher_denominator"])
    teacher_sum = float(context["teacher_kl_sum"])
    teacher_count = context["teacher_token_count"]
    copy_baseline = float(context["copy_baseline"])
    copy_sum = float(context["copy_ce_sum"])
    copy_count = context["copy_token_count"]
    numeric = torch.tensor(
        [denominator, teacher_sum, copy_baseline, copy_sum], dtype=torch.float64,
    )
    if not bool(torch.isfinite(numeric).all()) or denominator <= 0 or teacher_sum <= 0 or (
        teacher_count != VALIDATION_TOKEN_COUNT
    ) or denominator != teacher_sum / teacher_count or not isinstance(
        copy_count, int
    ) or not 0 < copy_count <= VALIDATION_TOKEN_COUNT or copy_baseline != (
        copy_sum / copy_count
    ):
        raise RuntimeError(f"v2.1 {name} scorer sufficient statistics changed")

    calls = context["call_counters"]
    if not isinstance(calls, Mapping) or set(calls) != {
        "fit_capture", "validation_capture", "teacher", "copy_baseline",
        "candidates",
    }:
        raise RuntimeError(f"v2.1 {name} call-counter schema changed")
    if stage == "site0":
        expected = {
            "fit_capture": {0: 60, 1: 60, 2: 0},
            "validation_capture": {0: 24, 1: 0, 2: 0},
            "teacher": {0: 24, 1: 24, 2: 0},
            "copy_baseline": {0: 0, 1: 24, 2: 0},
            "candidate": {0: 0, 1: 24, 2: 0},
        }
    else:
        expected = {
            "fit_capture": {0: 0, 1: 60, 2: 0},
            "validation_capture": {0: 0, 1: 24, 2: 0},
            "teacher": {0: 0, 1: 24, 2: 0},
            "copy_baseline": {0: 0, 1: 0, 2: 0},
            "candidate": {0: 0, 1: 0, 2: 0},
        }
    for arm in ("fit_capture", "validation_capture", "teacher", "copy_baseline"):
        _validate_exact_call_counter(calls[arm], expected[arm], f"{name} {arm}")
    candidate_calls = calls["candidates"]
    if not isinstance(candidate_calls, Mapping) or set(candidate_calls) != set(candidates):
        raise RuntimeError(f"v2.1 {name} candidate call-counter coverage changed")
    for candidate_name in candidates:
        _validate_exact_call_counter(
            candidate_calls[candidate_name], expected["candidate"],
            f"{name} {candidate_name}",
        )


def _validate_scorer_metrics(
    metrics: Any, context: Mapping[str, Any], expected_price: Mapping[str, Any],
    label: str,
) -> None:
    required_metrics = {
        "candidate_teacher_kl", "oracle_denominator_kl", "remaining_kl_ratio",
        "recovery", "global_ce", "copy_ce", "copy_count", "copy_worsening",
        "price", "raw_sufficient_statistics",
    }
    required_raw = {
        "candidate_teacher_kl_sum", "candidate_teacher_kl_count",
        "global_ce_sum", "global_ce_count", "copy_ce_sum", "copy_ce_count",
    }
    denominator = float(context["teacher_denominator"])
    baseline_copy = float(context["copy_baseline"])
    expected_copy_count = context["copy_token_count"]
    if not isinstance(metrics, Mapping) or set(metrics) != required_metrics:
        raise RuntimeError(f"v2.1 {label} metric schema changed")
    raw = metrics["raw_sufficient_statistics"]
    if not isinstance(raw, Mapping) or set(raw) != required_raw:
        raise RuntimeError(f"v2.1 {label} raw scorer schema changed")
    kl_sum = float(raw["candidate_teacher_kl_sum"])
    global_sum = float(raw["global_ce_sum"])
    copy_sum = float(raw["copy_ce_sum"])
    kl_count = raw["candidate_teacher_kl_count"]
    global_count = raw["global_ce_count"]
    copy_count = raw["copy_ce_count"]
    numeric = torch.tensor([kl_sum, global_sum, copy_sum], dtype=torch.float64)
    if not bool(torch.isfinite(numeric).all()) or (
        kl_count != VALIDATION_TOKEN_COUNT
        or global_count != VALIDATION_TOKEN_COUNT
        or copy_count != expected_copy_count
    ):
        raise RuntimeError(f"v2.1 {label} scorer counts changed")
    candidate_kl = kl_sum / kl_count
    global_ce = global_sum / global_count
    copy_ce = copy_sum / copy_count
    expected = {
        "candidate_teacher_kl": candidate_kl,
        "oracle_denominator_kl": denominator,
        "remaining_kl_ratio": candidate_kl / denominator,
        "recovery": 1.0 - candidate_kl / denominator,
        "global_ce": global_ce,
        "copy_ce": copy_ce,
        "copy_count": copy_count,
        "copy_worsening": copy_ce - baseline_copy,
        "price": dict(expected_price),
        "raw_sufficient_statistics": dict(raw),
    }
    if any(not _same_value(metrics.get(key), value) for key, value in expected.items()):
        raise RuntimeError(f"v2.1 {label} selector metric does not recompute")


def _validate_candidate_sufficient_statistics(
    candidates: Mapping[str, Any], context: Mapping[str, Any], label: str,
) -> None:
    """Recompute every selector-facing scalar from serialized raw sums/counts."""

    for candidate_name, candidate in candidates.items():
        _validate_scorer_metrics(
            candidate.get("metrics") if isinstance(candidate, Mapping) else None,
            context, selection.state_price(candidate["state"]),
            f"{label}:{candidate_name}",
        )


def _validate_mean_score(
    value: Any, context: Mapping[str, Any], *, stage: str,
    expected_upstream: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "context", "upstream_state_sha256", "metrics", "call_counter",
    } or value.get("context") != (
        "baseline" if stage == "site0" else "mean_site0"
    ) or value.get("upstream_state_sha256") != expected_upstream:
        raise RuntimeError(f"v2.1 {stage} mean-score context changed")
    _validate_scorer_metrics(
        value["metrics"], context, _constant_price(), f"{stage}:mean_score",
    )
    expected_calls = ({0: 0, 1: 24, 2: 0} if stage == "site0" else {
        0: 0, 1: 0, 2: 0,
    })
    _validate_exact_call_counter(
        value["call_counter"], expected_calls, f"{stage} mean-score",
    )


def _validate_mean_site1_diagnostics(value: Any, programs: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "context", "upstream_state_sha256", "scorer", "p_sum", "p_sum_sha256",
        "p_count", "capture_call_counter",
    } or value.get("context") != "mean_site0" or value.get(
        "upstream_state_sha256"
    ) != state_logical_sha256(programs["mean"][0]) or value.get(
        "scorer"
    ) != "CUDA float32 capture; float64 coefficient sums":
        raise RuntimeError("v2.1 mean-site1 provenance changed")
    p_sum = value["p_sum"]
    if not torch.is_tensor(p_sum) or p_sum.dtype != torch.float64 or tuple(
        p_sum.shape
    ) != (compiler.COEFFICIENT_DIM,) or not bool(torch.isfinite(p_sum).all()) or (
        value.get("p_sum_sha256") != tensor_sha256(p_sum)
    ) or value.get("p_count") != FIT_CAPTURE_COUNT:
        raise RuntimeError("v2.1 mean-site1 sufficient statistics changed")
    expected_bias = (p_sum / FIT_CAPTURE_COUNT).float().contiguous()
    mean_site1 = programs["mean"][1]
    mean_bias = mean_site1.get("bias") if isinstance(mean_site1, Mapping) else None
    if not torch.is_tensor(mean_bias) or not torch.equal(mean_bias, expected_bias):
        raise RuntimeError("v2.1 mean-site1 bias does not recompute")
    _validate_exact_call_counter(
        value["capture_call_counter"], {0: 0, 1: 60, 2: 0},
        "mean-site1 fit capture",
    )


def _validate_stage_binding(
    bundle: Mapping[str, Any], stage: str, artifact_path: Path, receipt_path: Path,
    ledger_names: set[str],
) -> None:
    bindings = bundle.get("stage_bindings")
    binding = bindings.get(stage) if isinstance(bindings, Mapping) else None
    if not isinstance(binding, Mapping) or set(binding) != {
        "artifact_path", "artifact_sha256", "artifact_bytes",
        "receipt_path", "receipt_sha256", "receipt_bytes",
    }:
        raise RuntimeError(f"v2.1 {stage} stage binding is incomplete")
    expected_binding = {
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path) if artifact_path.is_file() else None,
        "artifact_bytes": artifact_path.stat().st_size if artifact_path.is_file() else None,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_sha256": file_sha256(receipt_path) if receipt_path.is_file() else None,
        "receipt_bytes": receipt_path.stat().st_size if receipt_path.is_file() else None,
    }
    if binding != expected_binding:
        raise RuntimeError(f"v2.1 {stage} stage files are not exactly bound")
    receipt = json.loads(receipt_path.read_text())
    required_receipt = {
        "status": f"frozen_v21_{stage}_preselector_ledger",
        "authority": f"compiler_v21_{stage}_preselector_ledger",
        "authorized_for_training": False,
        "authorized_for_final_scoring": False,
        "protocol_sha256": PINS[PROTOCOL],
        "implementation_amendment_sha256": IMPLEMENTATION_AMENDMENT_SHA256,
        "rows_receipt_sha256": file_sha256(RECEIPT),
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
    }
    if any(receipt.get(key) != value for key, value in required_receipt.items()):
        raise RuntimeError(f"v2.1 {stage} preselector receipt changed")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get(
        "status"
    ) != f"pending_v21_{stage}_preselector_ledger" or payload.get(
        "authorized_for_training"
    ) is not False or payload.get("authorized_for_final_scoring") is not False:
        raise RuntimeError(f"v2.1 {stage} preselector artifact changed")
    external = payload.get("candidate_ledgers")
    embedded = bundle.get("candidate_ledgers")
    if not isinstance(external, Mapping) or set(external) != ledger_names or any(
        not _same_value(external[name], embedded[name]) for name in ledger_names
    ):
        raise RuntimeError(f"v2.1 {stage} external ledgers differ from program bundle")
    controls = payload.get("controls")
    bundle_controls = bundle.get("controls")
    programs = bundle.get("programs")
    if not isinstance(bundle_controls, Mapping) or not isinstance(programs, Mapping) or any(
        not isinstance(programs.get(arm), Mapping) for arm in ("true", "shuffle", "mean")
    ):
        raise RuntimeError(f"v2.1 {stage} bundle controls/programs are incomplete")
    needed_controls = ({"full_native_site0"} if stage == "site0" else {
        "full_native_site1_true_context", "full_native_site1_shuffle_context",
    })
    if not needed_controls.issubset(bundle_controls):
        raise RuntimeError(f"v2.1 {stage} bundle controls are incomplete")
    if stage == "site0":
        expected_controls = {
            "mean_site0": programs["mean"][0],
            "full_native_site0": bundle_controls["full_native_site0"],
        }
        upstream = {"true_site0": "baseline", "shuffle_site0": "baseline"}
    else:
        expected_controls = {
            "mean_site1": programs["mean"][1],
            "full_native_site1_true_context": bundle_controls[
                "full_native_site1_true_context"
            ],
            "full_native_site1_shuffle_context": bundle_controls[
                "full_native_site1_shuffle_context"
            ],
        }
        upstream = {
            "true_site1": state_logical_sha256(programs["true"][0]),
            "shuffle_site1": state_logical_sha256(programs["shuffle"][0]),
        }
    if not _same_value(controls, expected_controls):
        raise RuntimeError(f"v2.1 {stage} external controls differ from program bundle")
    diagnostics = payload.get("diagnostics")
    diagnostic_keys = {
        "fit_permutation_sha256", "capture_hashes", "contexts",
    } | ({"mean_control"} if stage == "site1" else {"mean_score"})
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != diagnostic_keys:
        raise RuntimeError(f"v2.1 {stage} external diagnostics are incomplete")
    permutation_hash = diagnostics["fit_permutation_sha256"]
    captures = diagnostics["capture_hashes"]
    contexts = diagnostics["contexts"]
    row_receipt = json.loads(RECEIPT.read_text())
    expected_permutation_hash = expected_fit_permutation_sha256(row_receipt)
    capture_keys = ({
        "fit_original", "fit_shuffled", "validation_site0",
    } if stage == "site0" else {
        "true_fit_site1", "shuffle_fit_site1",
        "true_validation_site1", "shuffle_validation_site1", "mean_fit_site1",
    })
    if permutation_hash != expected_permutation_hash or not isinstance(
        captures, Mapping
    ) or set(captures) != capture_keys or any(
        not _is_sha256(value) for value in captures.values()
    ):
        raise RuntimeError(f"v2.1 {stage} capture/permutation hashes changed")
    if not isinstance(contexts, Mapping) or set(contexts) != ledger_names:
        raise RuntimeError(f"v2.1 {stage} context diagnostics changed")
    for name, expected_upstream in upstream.items():
        _validate_context_diagnostics(
            contexts[name], stage=stage, name=name,
            expected_upstream=expected_upstream, candidates=external[name],
        )
        _validate_candidate_sufficient_statistics(
            external[name], contexts[name], name,
        )
    if stage == "site1":
        _validate_mean_site1_diagnostics(diagnostics["mean_control"], programs)
    else:
        _validate_mean_score(
            diagnostics["mean_score"], contexts["true_site0"],
            stage="site0", expected_upstream="baseline",
        )


def _validate_site0_training_authorization(
    bundle: Mapping[str, Any], programs: Mapping[str, Any],
) -> None:
    binding = bundle.get("site0_training_authorization")
    if not isinstance(binding, Mapping) or set(binding) != {
        "path", "sha256", "bytes", "receipt",
    } or binding.get("path") != str(SITE0_TRAINING_RECEIPT.resolve()) or not (
        SITE0_TRAINING_RECEIPT.is_file()
    ) or binding.get("sha256") != file_sha256(SITE0_TRAINING_RECEIPT) or binding.get(
        "bytes"
    ) != SITE0_TRAINING_RECEIPT.stat().st_size:
        raise RuntimeError("v2.1 site0 training authorization binding changed")
    receipt = json.loads(SITE0_TRAINING_RECEIPT.read_text())
    if not _same_value(binding.get("receipt"), receipt) or receipt.get(
        "status"
    ) != "frozen_v21_site0_programs_after_outer_return" or receipt.get(
        "authority"
    ) != "compiler_v21_site0_to_site1_training_unlock" or receipt.get(
        "authorized_for_training"
    ) is not True or receipt.get("training_license_sites") != [1] or receipt.get(
        "authorized_for_final_scoring"
    ) is not False or receipt.get("outer_model_returned") is not True or receipt.get(
        "hook_restored_and_inert"
    ) is not True:
        raise RuntimeError("v2.1 site0 training authorization changed")
    expected_hashes = {
        arm: state_logical_sha256(programs[arm][0]) for arm in ("true", "shuffle")
    }
    if receipt.get("selected_state_sha256") != expected_hashes or receipt.get(
        "mean_state_sha256"
    ) != state_logical_sha256(programs["mean"][0]) or receipt.get(
        "stage_binding"
    ) != bundle.get("stage_bindings", {}).get("site0"):
        raise RuntimeError("v2.1 site0 training states differ from deployed bundle")


def _validate_historical_row_authority(receipt: Mapping[str, Any]) -> None:
    if file_sha256(RECEIPT) != ROWS_RECEIPT_SHA256 or not MANIFEST.is_file() or (
        file_sha256(MANIFEST) != ROWS_MANIFEST_SHA256
    ) or receipt.get("manifest_sha256") != ROWS_MANIFEST_SHA256:
        raise RuntimeError("v2.1 realized row authority identity changed")
    manifest = json.loads(MANIFEST.read_text())
    source_commit = receipt.get("source_commit")
    source_hashes = receipt.get("source_hashes")
    if manifest.get("source_commit") != source_commit or manifest.get(
        "source_hashes"
    ) != source_hashes or not isinstance(source_commit, str) or not isinstance(
        source_hashes, Mapping
    ) or not source_hashes:
        raise RuntimeError("v2.1 historical row source closure changed")
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=ROOT,
        capture_output=True,
    )
    if commit_check.returncode:
        raise RuntimeError("v2.1 historical row source commit is absent")
    for relative, expected in source_hashes.items():
        committed = subprocess.run(
            ["git", "show", f"{source_commit}:{relative}"], cwd=ROOT,
            capture_output=True,
        )
        if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != expected:
            raise RuntimeError(f"v2.1 historical row source blob changed: {relative}")
    if manifest.get("protected_before") != manifest.get("protected_after"):
        raise RuntimeError("v2.1 historical row protected snapshot changed")


def _validate_program_bundle(bundle: Any) -> None:
    if not isinstance(bundle, Mapping):
        raise RuntimeError("v2.1 program artifact is not a mapping")
    programs = bundle.get("programs")
    if not isinstance(programs, Mapping) or set(programs) != {"true", "shuffle", "mean"}:
        raise RuntimeError("v2.1 program artifact lacks the three registered arms")
    contexts = bundle.get("pipeline_contexts")
    if contexts != {
        "true": {0: "baseline", 1: "true_site0"},
        "shuffle": {0: "baseline", 1: "shuffle_site0"},
        "mean": {0: "baseline", 1: "mean_site0"},
    }:
        raise RuntimeError("v2.1 program artifact is not autoregressive by arm")

    ledger_names = {
        "true_site0", "true_site1", "shuffle_site0", "shuffle_site1",
    }
    ledgers, receipts = bundle.get("candidate_ledgers"), bundle.get("selection_receipts")
    specs = _candidate_specs()
    if not isinstance(ledgers, Mapping) or set(ledgers) != ledger_names:
        raise RuntimeError("v2.1 program artifact candidate ledgers are incomplete")
    if not isinstance(receipts, Mapping) or set(receipts) != ledger_names:
        raise RuntimeError("v2.1 program artifact selection receipts are incomplete")
    for name in sorted(ledger_names):
        ledger = ledgers[name]
        if not isinstance(ledger, Mapping) or set(ledger) != set(specs):
            raise RuntimeError(f"v2.1 candidate ledger is not the exact A-E grid: {name}")
        for candidate_name, candidate in ledger.items():
            if not isinstance(candidate, Mapping) or not isinstance(
                candidate.get("metrics"), Mapping
            ):
                raise RuntimeError(f"v2.1 candidate record is malformed: {name}")
            _validate_candidate_state(candidate_name, candidate.get("state"), specs[candidate_name])
            metrics = candidate["metrics"]
            for metric in ("recovery", "copy_worsening"):
                value = float(metrics.get(metric, float("nan")))
                if not bool(torch.isfinite(torch.tensor(value))):
                    raise RuntimeError(f"v2.1 candidate metric is nonfinite: {name}:{metric}")
            if metrics.get("price") != selection.state_price(candidate["state"]):
                raise RuntimeError(f"v2.1 candidate price changed: {name}:{candidate_name}")
        expected = (
            selection.freeze_validation_selection(ledger)
            if name.startswith("true_") else _total_shuffle_selection(ledger)
        )
        if not _same_value(receipts[name], expected):
            raise RuntimeError(f"v2.1 selection receipt does not recompute: {name}")
        arm, site_text = name.split("_site")
        site = int(site_text)
        if not _same_value(programs[arm][site], ledger[expected["selected"]]["state"]):
            raise RuntimeError(f"v2.1 deployed state differs from selected ledger: {name}")

    _validate_stage_binding(
        bundle, "site0", SITE0_LEDGER_ARTIFACT, SITE0_LEDGER_RECEIPT,
        {"true_site0", "shuffle_site0"},
    )
    _validate_stage_binding(
        bundle, "site1", SITE1_LEDGER_ARTIFACT, SITE1_LEDGER_RECEIPT,
        {"true_site1", "shuffle_site1"},
    )
    _validate_site0_training_authorization(bundle, programs)

    for arm in ("true", "shuffle", "mean"):
        if not isinstance(programs[arm], Mapping) or set(programs[arm]) != {0, 1}:
            raise RuntimeError(f"v2.1 {arm} program lacks exact sites 0 and 1")
    for site, state in programs["mean"].items():
        if not isinstance(state, Mapping) or set(state) != {
            "grammar", "interface", "family", "bias",
        } or state.get("grammar") != "constant" or state.get(
            "interface"
        ) != "state_complete_p" or state.get("family") != "fit_mean_control":
            raise RuntimeError(f"v2.1 mean site{site} grammar changed")
        _finite_tensor(state.get("bias"), (compiler.COEFFICIENT_DIM,))

    controls = bundle.get("controls")
    control_names = {
        "full_native_site0", "full_native_site1_true_context",
        "full_native_site1_shuffle_context",
        "copy_constrained_shuffle_sensitivity_site0",
        "copy_constrained_shuffle_sensitivity_site1",
    }
    if not isinstance(controls, Mapping) or set(controls) != control_names:
        raise RuntimeError("v2.1 program artifact controls are incomplete")
    validation_identity = bundle.get("strata", {}).get(
        "validation_document_ids_sha256"
    ) if isinstance(bundle.get("strata"), Mapping) else None
    _validate_full_native_control(
        controls["full_native_site0"], 0, context="baseline",
        upstream_state_sha256="baseline",
        validation_document_ids_sha256=validation_identity,
    )
    _validate_full_native_control(
        controls["full_native_site1_true_context"], 1, context="true_site0",
        upstream_state_sha256=state_logical_sha256(programs["true"][0]),
        validation_document_ids_sha256=validation_identity,
    )
    _validate_full_native_control(
        controls["full_native_site1_shuffle_context"], 1, context="shuffle_site0",
        upstream_state_sha256=state_logical_sha256(programs["shuffle"][0]),
        validation_document_ids_sha256=validation_identity,
    )
    for site in (0, 1):
        ledger = ledgers[f"shuffle_site{site}"]
        try:
            sensitivity = {
                "status": "selected",
                "selection": selection.freeze_control_selection(ledger),
            }
        except RuntimeError as exc:
            sensitivity = {"status": "empty", "failure_message": str(exc)}
        if controls[f"copy_constrained_shuffle_sensitivity_site{site}"] != sensitivity:
            raise RuntimeError(f"v2.1 shuffle sensitivity does not recompute: site{site}")

    strata = bundle.get("strata")
    if not isinstance(strata, Mapping) or set(strata) != {
        "source", "validation_document_ids_sha256", "token_frequency",
        "causal_omission_audit",
    } or strata.get("source") != "compiler_validation_v21":
        raise RuntimeError("v2.1 program artifact strata schema changed")
    identity = strata.get("validation_document_ids_sha256")
    row_receipt = json.loads(RECEIPT.read_text())
    validation_records = row_receipt.get("document_provenance", {}).get(
        "sets", {}
    ).get("compiler_validation_v21")
    validation_entry = row_receipt.get("entries", {}).get("compiler_validation_v21", {})
    if not isinstance(validation_records, list) or not validation_records or any(
        not isinstance(record, Mapping) or "document_id" not in record
        for record in validation_records
    ):
        raise RuntimeError("v2.1 row authority lacks mapped-validation provenance")
    expected_identity = logical_json_sha256([
        record["document_id"] for record in validation_records
    ])
    if validation_entry.get("document_ids_sha256") != expected_identity or (
        identity != expected_identity
    ):
        raise RuntimeError("v2.1 strata validation document identity changed")
    frequency = strata.get("token_frequency")
    if not isinstance(frequency, Mapping) or set(frequency) != {
        "rule", "boundaries", "counts", "fit_token_counts_sha256",
        "validation_assignment_sha256",
    }:
        raise RuntimeError("v2.1 token-frequency strata changed")
    boundaries, counts = frequency["boundaries"], frequency["counts"]
    fit_rows = _load_role_cache_for_strata(row_receipt, "compiler_fit_v21")
    validation_rows = _load_role_cache_for_strata(
        row_receipt, "compiler_validation_v21"
    )
    expected_frequency = derive_token_frequency_strata(
        fit_rows, validation_rows, boundaries,
    )
    if frequency != expected_frequency:
        raise RuntimeError("v2.1 token-frequency strata do not recompute")

    audit = strata.get("causal_omission_audit")
    if not isinstance(audit, Mapping) or set(audit) != {
        "context", "upstream_state_sha256", "validation_document_ids_sha256",
        "scorer", "quantile_currency", "rule", "full_oracle_row_ce",
        "omit_row_ce", "full_oracle_row_ce_sha256", "omit_row_ce_sha256",
        "target_p_square_sums", "target_p_square_sums_sha256", "target_p_count",
        "omission_losses", "target_second_moments", "positive_floor", "weights",
        "call_counters",
    } or audit.get("context") != "true_site0" or audit.get("rule") != (
        "abs(loss)/max(second_moment,1e-12); positive 5pct floor; mean-one"
    ) or audit.get("upstream_state_sha256") != state_logical_sha256(
        programs["true"][0]
    ) or audit.get("validation_document_ids_sha256") != expected_identity or audit.get(
        "scorer"
    ) != "CUDA float32 per-token; float64 row/aggregate" or audit.get(
        "quantile_currency"
    ) != "torch.float64 q=0.05 interpolation=linear":
        raise RuntimeError("v2.1 causal omission audit schema changed")
    full_row_ce = audit["full_oracle_row_ce"]
    omit_row_ce = audit["omit_row_ce"]
    square_sums = audit["target_p_square_sums"]
    if audit.get("full_oracle_row_ce_sha256") != tensor_sha256(full_row_ce) or audit.get(
        "omit_row_ce_sha256"
    ) != tensor_sha256(omit_row_ce) or audit.get(
        "target_p_square_sums_sha256"
    ) != tensor_sha256(square_sums):
        raise RuntimeError("v2.1 causal omission sufficient-stat hashes changed")
    derived = derive_causal_audit(
        full_row_ce, omit_row_ce, square_sums, audit.get("target_p_count"),
    )
    if any(not _same_value(audit.get(key), value) for key, value in derived.items()):
        raise RuntimeError("v2.1 causal omission weights do not recompute")
    counters = audit.get("call_counters")
    if not isinstance(counters, Mapping) or set(counters) != {
        "full_oracle", "omissions"
    } or counters["full_oracle"] != {0: 0, 1: 0, 2: 0} or counters[
        "omissions"
    ] != {0: 0, 1: 0, 2: 0}:
        raise RuntimeError("v2.1 causal omission call counters changed")

    prices = bundle.get("prices")
    expected_prices = {
        "true": _pipeline_price(programs["true"][0], programs["true"][1]),
        "shuffle": _pipeline_price(programs["shuffle"][0], programs["shuffle"][1]),
        "mean": {
            "site0": _constant_price(), "site1": _constant_price(),
            "total_reals": 2 * _constant_price()["total_reals"],
        },
    }
    if prices != expected_prices:
        raise RuntimeError("v2.1 pipeline prices do not recompute from deployed states")


def validate_final_unlock(path: Path) -> dict[str, Any]:
    if path.resolve() != PROGRAMS_RECEIPT.resolve() or not path.is_file():
        raise RuntimeError("v2.1 final unlock authority is absent")
    if not IMPLEMENTATION_AMENDMENT.is_file() or file_sha256(
        IMPLEMENTATION_AMENDMENT
    ) != IMPLEMENTATION_AMENDMENT_SHA256:
        raise RuntimeError("v2.1 implementation amendment changed")
    if not RECEIPT.is_file() or file_sha256(RECEIPT) != ROWS_RECEIPT_SHA256:
        raise RuntimeError("v2.1 realized row receipt changed")
    _validate_historical_row_authority(json.loads(RECEIPT.read_text()))
    unlock = json.loads(path.read_text())
    required = {
        "status": "frozen_v21_programs_controls_strata_prices_before_final",
        "authority": "compiler_v21_final_unlock",
        "authorized_for_training": False,
        "authorized_for_final_scoring": True,
        "protocol_sha256": PINS[PROTOCOL],
        "implementation_amendment_sha256": IMPLEMENTATION_AMENDMENT_SHA256,
        "rows_receipt_path": str(RECEIPT.resolve()),
        "rows_receipt_sha256": file_sha256(RECEIPT),
        "programs_artifact_path": str(PROGRAMS_ARTIFACT.resolve()),
    }
    for key, expected in required.items():
        if unlock.get(key) != expected:
            raise RuntimeError(f"v2.1 final unlock changed at {key}")
    if not PROGRAMS_ARTIFACT.is_file() or unlock.get(
        "programs_artifact_sha256"
    ) != file_sha256(PROGRAMS_ARTIFACT) or unlock.get(
        "programs_artifact_bytes"
    ) != PROGRAMS_ARTIFACT.stat().st_size:
        raise RuntimeError("v2.1 final unlock program artifact binding changed")
    expected_contents = {
        "true_program_sites": [0, 1],
        "shuffle_program_sites": [0, 1],
        "mean_program_sites": [0, 1],
        "candidate_ledgers_frozen": True,
        "controls_frozen": True,
        "strata_frozen": True,
        "standalone_prices_frozen": True,
        "preselector_stage_receipts_bound": True,
        "strata_derivations_recomputed": True,
        "site1_full_native_contexts": ["true", "shuffle"],
    }
    if unlock.get("frozen_contents") != expected_contents:
        raise RuntimeError("v2.1 final unlock contents are incomplete")

    try:
        bundle = torch.load(PROGRAMS_ARTIFACT, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError("v2.1 final unlock program artifact is unreadable") from exc
    if not isinstance(bundle, Mapping):
        raise RuntimeError("v2.1 program artifact is not a mapping")
    bundle_required = {
        "schema_version": 1,
        "status": "frozen_v21_program_bundle_pending_final_unlock",
        "authority": "compiler_v21_program_bundle",
        "authorized_for_training": False,
        "authorized_for_final_scoring": False,
        "protocol_sha256": PINS[PROTOCOL],
        "implementation_amendment_sha256": IMPLEMENTATION_AMENDMENT_SHA256,
        "rows_receipt_sha256": file_sha256(RECEIPT),
    }
    for key, expected in bundle_required.items():
        if bundle.get(key) != expected:
            raise RuntimeError(f"v2.1 program artifact changed at {key}")
    _validate_program_bundle(bundle)

    source_hashes = unlock.get("source_hashes")
    expected_sources = {
        str(source.resolve().relative_to(ROOT.resolve()))
        for source in PROGRAM_SOURCE_CLOSURE
    }
    if not isinstance(source_hashes, dict) or set(source_hashes) != expected_sources:
        raise RuntimeError("v2.1 final unlock lacks the exact program source closure")
    source_commit = unlock.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise RuntimeError("v2.1 final unlock lacks source commit")
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], cwd=ROOT,
        capture_output=True,
    )
    if commit_check.returncode:
        raise RuntimeError("v2.1 final unlock source commit does not exist")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    origin = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, head], cwd=ROOT,
        capture_output=True,
    )
    if head != origin or ancestor.returncode:
        raise RuntimeError("v2.1 final unlock source commit is not synchronized")
    for relative, expected in source_hashes.items():
        source = (ROOT / relative).resolve()
        try:
            source.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"v2.1 source escapes repository: {relative}") from exc
        if not source.is_file() or file_sha256(source) != expected:
            raise RuntimeError(f"v2.1 final unlock source changed: {relative}")
        committed = subprocess.run(
            ["git", "show", f"{source_commit}:{relative}"], cwd=ROOT,
            capture_output=True,
        )
        if committed.returncode or hashlib.sha256(committed.stdout).hexdigest() != expected:
            raise RuntimeError(f"v2.1 final unlock source is not commit-bound: {relative}")
    return unlock


def load_roles_and_validate(
    roles: Iterable[str], *, final_unlock_receipt: Path | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    requested = tuple(dict.fromkeys(roles))
    valid = ("compiler_fit_v21", "compiler_validation_v21", "compiler_final_v21")
    if not requested or any(role not in valid for role in requested):
        raise ValueError(f"invalid v2.1 row roles: {requested}")
    if "compiler_final_v21" in requested:
        if final_unlock_receipt is None:
            raise RuntimeError("v2.1 final is locked until programs/controls/strata/prices freeze")
        validate_final_unlock(final_unlock_receipt)
    old_receipt, _, _ = verify_inputs()
    if not RECEIPT.is_file():
        raise RuntimeError("v2.1 row receipt is absent")
    receipt = json.loads(RECEIPT.read_text())
    _validate_historical_row_authority(receipt)
    required = {
        "status": "frozen_before_any_v21_validation_model_forward",
        "authority": "compiler_v21_prospective_role_designation",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "training_license_sites": [],
        "protocol_sha256": PINS[PROTOCOL],
        "old_rows_receipt_sha256": PINS[OLD_RECEIPT],
        "retry1_failure_manifest_sha256": PINS[RETRY1_FAILURE],
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"v2.1 row receipt changed at {key}")
    expected_licenses = {
        "compiler_fit_v21": {
            "training": True, "selection": False, "final_scoring": False
        },
        "compiler_validation_v21": {
            "training": False, "selection": True, "final_scoring": False
        },
        "compiler_final_v21": {
            "training": False, "selection": False, "final_scoring": True,
            "requires_final_unlock_authority": True
        },
        "old_compiler_validation": "forbidden"
    }
    if receipt.get("role_licenses") != expected_licenses:
        raise RuntimeError("v2.1 role-scoped row licenses changed")
    if not all(receipt.get("disjointness_gates", {}).values()):
        raise RuntimeError("v2.1 row disjointness authority is incomplete")
    mapping = {
        "compiler_fit_v21": ("compiler_fit", 480),
        "compiler_validation_v21": ("compiler_final", 192),
        "compiler_final_v21": (None, 192),
    }
    rows: dict[str, torch.Tensor] = {}
    for role, (source_role, count) in mapping.items():
        entry = receipt.get("entries", {}).get(role, {})
        path = Path(entry.get("cache_path", ""))
        if not path.is_file() or file_sha256(path) != entry.get("cache_file_sha256"):
            raise RuntimeError(f"v2.1 serialized cache changed for {role}")
        records = receipt.get("document_provenance", {}).get("sets", {}).get(role)
        if not isinstance(records, list) or len(records) != count or logical_json_sha256(
            records
        ) != entry.get("provenance_records_sha256"):
            raise RuntimeError(f"v2.1 provenance changed for {role}")
        if source_role is not None and entry.get("cache_path") != old_receipt[
            "entries"
        ][source_role]["cache_path"]:
            raise RuntimeError(f"v2.1 remap path changed for {role}")
        if role not in requested:
            continue
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(tensor.shape) != (count, T_LEN) or tensor.dtype != torch.long:
            raise RuntimeError(f"v2.1 tensor shape changed for {role}")
        if tensor_sha256(tensor) != entry.get("tensor_full_raw_sha256") or tensor_sha256(
            tensor[:, :MODEL_LEN]
        ) != entry.get("tensor_prefix257_raw_sha256"):
            raise RuntimeError(f"v2.1 tensor content changed for {role}")
        rows[role] = tensor
    return receipt, rows


def load_final_for_scoring(
    final_unlock_receipt: Path,
) -> tuple[dict[str, Any], torch.Tensor]:
    receipt, rows = load_roles_and_validate(
        ("compiler_final_v21",), final_unlock_receipt=final_unlock_receipt
    )
    return receipt, rows["compiler_final_v21"]


def main() -> None:
    if any(path.exists() for path in OUTPUTS) or CACHE.exists():
        raise RuntimeError("refusing to overwrite v2.1 row outputs")
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"v2.1 row build already claimed: {LOCK}") from error
    before = protected_snapshot()
    try:
        receipt = build(before)
        print(json.dumps({
            "status": receipt["status"],
            "protocol_sha256": receipt["protocol_sha256"],
            "roles": {
                role: {
                    "request": entry.get("request"),
                    "source_role": entry.get("source_role"),
                    "tensor_prefix257_raw_sha256": entry["tensor_prefix257_raw_sha256"],
                    "unique_document_count": entry["unique_document_count"],
                }
                for role, entry in receipt["entries"].items()
            },
            "disjointness_gates": receipt["disjointness_gates"],
        }, indent=2))
        print(f"wrote {RECEIPT}")
    except BaseException as error:
        if not RECEIPT.exists():
            manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
            manifest.update({
                "schema_version": 1,
                "status": "failed_compiler_v21_row_build",
                "authorized_for_scored_experiments": False,
                "authorized_for_training": False,
                "protocol_sha256": PINS[PROTOCOL],
                "retry1_failure_manifest_sha256": PINS[RETRY1_FAILURE],
                "failure_type": type(error).__name__,
                "failure_message": str(error),
                "protected_after": protected_snapshot(),
                "recovery": "Preserve cache/manifest; use a new namespace for any retry."
            })
            write_json_atomic(manifest, MANIFEST)
        raise
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    main()
