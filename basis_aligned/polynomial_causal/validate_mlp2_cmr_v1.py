#!/usr/bin/env python3
"""Source-closed finite VALIDATION run for the frozen MLP2 CMR v1 assay."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Callable

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for source_root in (ROOT, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
import mlp2_cmr_v1_validation_runtime as runtime
import mlp2_cmr_v1_validation_statistics as statistics
import project_mlp2_cmr_v1_validation_rows as projection


PREREG = HERE / "MLP2_CMR_V1_PREREGISTRATION.md"
ADDENDUM = HERE / "MLP2_CMR_V1_VALIDATION_ADDENDUM.md"
ROLE_ROWS = HERE / "mlp2_cmr_v1_validation_rows.pt"
ROLE_MANIFEST = HERE / "mlp2_cmr_v1_validation_rows_manifest.json"
ROLE_RECEIPT = HERE / "mlp2_cmr_v1_validation_rows_receipt.json"
FIT_BUNDLE = HERE / "mlp2_cmr_v1_fit_mean_bundle.pt"
FIT_RESULT = HERE / "mlp2_cmr_v1_fit_mean_result.json"
FIT_RECEIPT = HERE / "mlp2_cmr_v1_fit_mean_receipt.json"
SUFFIX_BUNDLE = HERE / "mlp2_cmr_v1_suffix_v2_bundle.pt"
SUFFIX_RESULT = HERE / "mlp2_cmr_v1_suffix_v2_result.json"
SUFFIX_RECEIPT = HERE / "mlp2_cmr_v1_suffix_v2_receipt.json"
CORRECTION = HERE / "mlp2_cmr_v1_suffix_v2_overlap_correction.json"
CORRECTION_RECEIPT = HERE / "mlp2_cmr_v1_suffix_v2_overlap_correction_receipt.json"
CALIBRATION_BUNDLE = HERE / "mlp2_cmr_v1_fit_selector_calibration_bundle.pt"
CALIBRATION_RESULT = HERE / "mlp2_cmr_v1_fit_selector_calibration_result.json"
CALIBRATION_RECEIPT = HERE / "mlp2_cmr_v1_fit_selector_calibration_receipt.json"
AUTHORITY = HERE / "mlp2_cmr_v1_validation_authority.json"
LEDGER = HERE / "mlp2_cmr_v1_validation_ledger.pt"
RESULT = HERE / "mlp2_cmr_v1_validation_result.json"
RECEIPT = HERE / "mlp2_cmr_v1_validation_receipt.json"
FAILURE = HERE / "mlp2_cmr_v1_validation_failure.json"
LOCK = HERE / ".mlp2_cmr_v1_validation.lock"

PARENT_PATHS = {
    "role_rows": ROLE_ROWS,
    "role_manifest": ROLE_MANIFEST,
    "role_receipt": ROLE_RECEIPT,
    "fit_bundle": FIT_BUNDLE,
    "fit_result": FIT_RESULT,
    "fit_receipt": FIT_RECEIPT,
    "suffix_bundle": SUFFIX_BUNDLE,
    "suffix_result": SUFFIX_RESULT,
    "suffix_receipt": SUFFIX_RECEIPT,
    "correction": CORRECTION,
    "correction_receipt": CORRECTION_RECEIPT,
    "calibration_bundle": CALIBRATION_BUNDLE,
    "calibration_result": CALIBRATION_RESULT,
    "calibration_receipt": CALIBRATION_RECEIPT,
}
EXPECTED_PARENTS = {
    "role_rows": "f0436268b6a17f1c4c47621ff16d542fe7c20a6579a3fee6e10bce241cee90db",
    "role_manifest": "4ff1155386cec27daf7702675797ec8ed0a1a5534f07c4f24a31fea9ef384a40",
    "role_receipt": "98e077918ab132a6d3e1cc6ffc5d03f3295f3e30adfe7670cf088843b43acebe",
    "fit_bundle": "043bb52b9580d9c9c342460e5bb80ff579db01486b3b6c6672bf5fba77e46f8e",
    "fit_result": "65c1ee33f0399d6489cae0227442d479a9d59b9be98f619d92423cfd39fc7833",
    "fit_receipt": "9dc14d909a1b4aafd33c67dc7a3d066db4ccc9cb83c7059fe7aaf499ca9e5efa",
    "suffix_bundle": "cb3f8d3caecab86881eba825785cabd58c1b7ac8e2aa1eb93b459168cff17ce1",
    "suffix_result": "ab08dc0f0a71b5daf21228991b9e78a272aa74d226d97189ac414a546dc16f62",
    "suffix_receipt": "b61c7308409ec64dc05601206bda21e1f4e24097871ba8dff0c92bc84e761e1f",
    "correction": "ffd5a826962f09ffec0af6c842eaf0bf64530423b827f6239556bc43db9d7ff4",
    "correction_receipt": "dd557dc6366503bea2f3f7649d6312abc8a89857bb277ea4e844d8822e4e968a",
    "calibration_bundle": "3f9aa5ff69530a099c7859b454298eca48cd0789413b412f599746793fd6c1fa",
    "calibration_result": "e30ae749d59dedad4c17159d5b29af1c4c0c79e3f620e794e3a590f3b049c08c",
    "calibration_receipt": "08267122572157203ccf87f9d901d9c4efdfb41c9bb3b4f0d34f1f1f4e669b52",
}

SOURCE_CLOSURE = tuple(dict.fromkeys((
    PREREG, ADDENDUM, Path(__file__).resolve(),
    HERE / "test_validate_mlp2_cmr_v1.py",
    Path(runtime.__file__).resolve(), HERE / "test_mlp2_cmr_v1_validation_runtime.py",
    Path(statistics.__file__).resolve(), HERE / "test_mlp2_cmr_v1_validation_statistics.py",
    HERE / "mlp2_cmr_v1_physical_program.py",
    HERE / "test_mlp2_cmr_v1_physical_program.py",
    *tuple(Path(path).resolve() for path in projection.SOURCE_CLOSURE),
    Path(facade.__file__).resolve(), ROOT / "jacclust/tt_model.py",
)))


def file_sha256(path: Path) -> str:
    return projection.base.file_sha256(path)


def canonical_json_bytes(value: Any) -> bytes:
    return projection.base.canonical_json_bytes(value)


def committed_source() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    hashes = {}
    if len(SOURCE_CLOSURE) != len(set(SOURCE_CLOSURE)):
        raise RuntimeError("validation source closure contains duplicates")
    for path in SOURCE_CLOSURE:
        relative = path.relative_to(ROOT)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"validation source differs from commit: {relative}")
        hashes[str(relative)] = digest
    return commit, hashes


def protected_inputs() -> tuple[dict[str, str], dict[str, bytes]]:
    captured = {name: path.read_bytes() for name, path in PARENT_PATHS.items()}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in captured.items()}
    if hashes != EXPECTED_PARENTS:
        raise RuntimeError("MLP2 validation protected parent changed")
    role_manifest = json.loads(captured["role_manifest"])
    role_receipt = json.loads(captured["role_receipt"])
    fit_receipt = json.loads(captured["fit_receipt"])
    suffix_receipt = json.loads(captured["suffix_receipt"])
    correction_receipt = json.loads(captured["correction_receipt"])
    calibration_receipt = json.loads(captured["calibration_receipt"])
    if (
        role_manifest.get("contains_roles") != ["VALIDATION"]
        or role_manifest.get("output_sha256") != EXPECTED_PARENTS["role_rows"]
        or role_receipt.get("output_sha256") != EXPECTED_PARENTS["role_rows"]
        or role_receipt.get("manifest_sha256") != EXPECTED_PARENTS["role_manifest"]
        or role_receipt.get("authorized_for_validation_model_forward_input") is not True
        or role_receipt.get("authorized_for_replication") is not False
        or fit_receipt.get("bundle_sha256") != EXPECTED_PARENTS["fit_bundle"]
        or fit_receipt.get("result_sha256") != EXPECTED_PARENTS["fit_result"]
        or suffix_receipt.get("bundle_sha256") != EXPECTED_PARENTS["suffix_bundle"]
        or suffix_receipt.get("result_sha256") != EXPECTED_PARENTS["suffix_result"]
        or suffix_receipt.get("authorized_for_validation") is not True
        or suffix_receipt.get("authorized_for_replication") is not False
        or correction_receipt.get(
            "authorized_for_validation_with_original_selector_receipt"
        ) is not True
        or correction_receipt.get("result_sha256") != EXPECTED_PARENTS["correction"]
        or calibration_receipt.get("bundle_sha256") != EXPECTED_PARENTS[
            "calibration_bundle"
        ]
        or calibration_receipt.get("result_sha256") != EXPECTED_PARENTS[
            "calibration_result"
        ]
        or calibration_receipt.get("authorized_for_validation_implementation") is not True
        or calibration_receipt.get("authorized_for_replication") is not False
    ):
        raise RuntimeError("MLP2 validation parent receipt joins changed")
    return hashes, captured


def validate_claim(nonce: str, inode: tuple[int, int]) -> None:
    descriptor = os.open(LOCK, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        stat = os.fstat(descriptor)
        payload = os.read(descriptor, 4096)
    finally:
        os.close(descriptor)
    if (stat.st_dev, stat.st_ino) != inode or json.loads(payload).get("nonce") != nonce:
        raise RuntimeError("MLP2 validation claim changed")


def _capability_system():
    seal = object()
    minted = False

    class Capability:
        __slots__ = ("nonce", "inode", "authority_sha256", "consumed")

        def __init__(self, provided_seal, nonce, inode, authority_sha256):
            if provided_seal is not seal:
                raise TypeError("validation capability is not directly constructible")
            self.nonce = nonce
            self.inode = inode
            self.authority_sha256 = authority_sha256
            self.consumed = False

        def __copy__(self):
            raise TypeError("validation capability cannot be copied")

        def __deepcopy__(self, memo):
            raise TypeError("validation capability cannot be copied")

    def mint(nonce, inode, authority_sha256):
        nonlocal minted
        if minted:
            raise RuntimeError("validation capability was already minted")
        validate_claim(nonce, inode)
        authority = json.loads(AUTHORITY.read_bytes())
        if file_sha256(AUTHORITY) != authority_sha256 or authority.get(
            "status"
        ) != "authority_frozen_before_validation_model_access" or authority.get(
            "authorized_role"
        ) != "VALIDATION" or authority.get("authorized_forward_calls") != {
            arm: runtime.CALLS for arm in runtime.ALL_ARMS
        }:
            raise RuntimeError("validation authority semantics changed")
        minted = True
        return Capability(seal, nonce, inode, authority_sha256)

    def consume(capability):
        if type(capability) is not Capability or capability.consumed:
            raise RuntimeError("fresh validation capability required")
        validate_claim(capability.nonce, capability.inode)
        if file_sha256(AUTHORITY) != capability.authority_sha256:
            raise RuntimeError("validation capability authority changed")
        capability.consumed = True

    return Capability, mint, consume


_ValidationCapability, _mint_capability, _consume_capability = _capability_system()


def guard_inputs(
    source_hashes: dict[str, str], parents: dict[str, str], nonce: str,
    inode: tuple[int, int], authority_hash: str,
) -> None:
    validate_claim(nonce, inode)
    for relative, expected in source_hashes.items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError("MLP2 validation source changed during execution")
    current, _ = protected_inputs()
    if current != parents or file_sha256(AUTHORITY) != authority_hash or (
        RECEIPT.exists() or FAILURE.exists()
    ):
        raise RuntimeError("MLP2 validation protected snapshot changed")
    validate_claim(nonce, inode)


def final_guard(
    source_hashes: dict[str, str], parents: dict[str, str], nonce: str,
    inode: tuple[int, int], authority_hash: str, ledger_hash: str,
    result_hash: str,
) -> None:
    guard_inputs(source_hashes, parents, nonce, inode, authority_hash)
    if file_sha256(LEDGER) != ledger_hash or file_sha256(RESULT) != result_hash:
        raise RuntimeError("MLP2 validation terminal outputs changed")
    validate_claim(nonce, inode)


def _load_torch(value: bytes) -> Any:
    return torch.load(io.BytesIO(value), map_location="cpu", weights_only=True)


def _selector_gauge_passes(result: dict[str, Any]) -> bool:
    audit = result.get("gauge_and_permutation_audit", {})
    channel = audit.get("channel_permutation", {})
    dyadic = audit.get("dyadic_reciprocal", {})
    general = audit.get("general_reciprocal_functional", {})
    return (
        channel == {
            "derangement_equivariant": True,
            "hash_random_equivariant": True,
            "suffix_support_equivariant": True,
        }
        and dyadic == {
            "canonical_down_max_abs_error": 0.0,
            "derangement_exact": True,
            "hash_random_exact": True,
        }
        and general == {
            "canonical_down_max_relative_error":
                general.get("canonical_down_max_relative_error"),
            "hash_byte_replay_required": False,
        }
        and isinstance(general["canonical_down_max_relative_error"], float)
        and 0.0 <= general["canonical_down_max_relative_error"] <= 5e-15
    )


def collect(
    parent_bytes: dict[str, bytes], capability: _ValidationCapability,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    _consume_capability(capability)
    role = _load_torch(parent_bytes["role_rows"])
    role_summary = projection.validate_role(role)
    rows = role["rows"].contiguous()
    eligible = role["eligible_mask"].contiguous()
    fit = _load_torch(parent_bytes["fit_bundle"])
    suffix = _load_torch(parent_bytes["suffix_bundle"])
    calibration = _load_torch(parent_bytes["calibration_bundle"])
    fit_result = json.loads(parent_bytes["fit_result"])
    suffix_result = json.loads(parent_bytes["suffix_result"])
    correction = json.loads(parent_bytes["correction"])
    calibration_result = json.loads(parent_bytes["calibration_result"])
    mean = fit["mean"]
    supports = {arm: suffix["supports"][arm].clone() for arm in runtime.PHYSICAL_ARMS}
    support_hashes = runtime.validate_supports(supports)
    if support_hashes != suffix_result.get("tensor_hashes", {}).get("supports") or any(
        not torch.equal(supports[arm], fit["supports"][arm])
        for arm in ("LOCAL", "RMS", "MASS")
    ) or correction.get("support_hashes") != support_hashes:
        raise RuntimeError("frozen selector support identity changed")
    cells = statistics.validation_cells(
        rows, eligible, calibration["fit_token_counts"],
        calibration["frequency_boundaries"],
    )
    epsilon = calibration["epsilon_grid"].clone().contiguous()
    if epsilon.shape != (28,) or calibration_result.get("epsilon_grid") != epsilon.tolist():
        raise RuntimeError("frozen validation epsilon grid changed")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != (
        "NVIDIA GeForce RTX 5090"
    ):
        raise RuntimeError("MLP2 validation requires the registered RTX 5090")
    device = torch.device("cuda:0")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    checkpoint_after_load = facade.validate_snapshot()
    if checkpoint != checkpoint_after_load:
        raise RuntimeError("checkpoint changed across validation model load")
    programs, program_receipts = runtime.build_physical_programs(
        model.transformer.h[runtime.SITE].mlp, mean, supports,
    )
    materialization = runtime.physical_materialization_replay(programs, device=device)
    if materialization["bit_exact"] is not True:
        raise RuntimeError("physical MLP2 materialization replay failed")

    ledgers: dict[str, dict[int, dict[str, statistics.CellSums]]] = {
        arm: {} for arm in runtime.ALL_ARMS
    }
    call_ledger = runtime.new_call_ledger()
    margin_counts = torch.zeros(runtime.DOCUMENTS, epsilon.numel(), dtype=torch.long)
    margin_support = torch.zeros(runtime.DOCUMENTS, dtype=torch.long)
    geometry_batches = []
    additivity = torch.zeros(runtime.DOCUMENTS, 3, dtype=torch.float64)
    precision_audit = None

    with torch.inference_mode():
        for start in range(0, runtime.DOCUMENTS, runtime.BATCH):
            stop = start + runtime.BATCH
            tokens = rows[start:stop, :-1].to(device).contiguous()
            batch_rows = rows[start:stop]
            batch_eligible = eligible[start:stop]
            batch_cells = {name: value[start:stop] for name, value in cells.items()}
            observed_additivity = []

            def observe(product: torch.Tensor, mlp: torch.nn.Module) -> None:
                if observed_additivity:
                    raise RuntimeError("native MLP2 additivity observer repeated")
                observed_additivity.append(runtime.additivity_batch(
                    product, mlp.Down.weight, mean, supports["SUFFIX"], batch_eligible,
                ))

            native = runtime.forward_arm(
                model, tokens, "NATIVE", programs, call_ledger,
                native_mlp2_observer=observe,
            )
            if len(observed_additivity) != 1:
                raise RuntimeError("native MLP2 additivity observer missing")
            additivity[start:stop] = observed_additivity[0]
            counts, support = statistics.native_margin_counts(
                native, batch_eligible, epsilon,
            )
            margin_counts[start:stop] = counts
            margin_support[start:stop] = support
            ledgers["NATIVE"].update(statistics.reduce_arm_batch(
                native, native, batch_rows, batch_cells, tuple(range(start, stop)),
            ))
            suffix_logits = None
            for arm in ("ZERO", *runtime.PHYSICAL_ARMS):
                candidate = runtime.forward_arm(
                    model, tokens, arm, programs, call_ledger,
                )
                ledgers[arm].update(statistics.reduce_arm_batch(
                    native, candidate, batch_rows, batch_cells, tuple(range(start, stop)),
                ))
                if arm == "SUFFIX":
                    suffix_logits = candidate
                    if start == 0:
                        precision_audit = statistics.enforce_float32_precision_audit(
                            native, candidate, batch_rows, batch_eligible,
                        )
                else:
                    del candidate
            if suffix_logits is None:
                raise RuntimeError("SUFFIX logits were not retained for signed geometry")
            signed_logits = {}
            for arm in runtime.SIGNED_T:
                candidate = runtime.forward_arm(
                    model, tokens, arm, programs, call_ledger,
                )
                ledgers[arm].update(statistics.reduce_arm_batch(
                    native, candidate, batch_rows, batch_cells, tuple(range(start, stop)),
                ))
                signed_logits[arm] = candidate
            geometry_batches.append(statistics.reduce_signed_geometry_batch(
                native, suffix_logits, signed_logits, batch_cells,
            ))
            del native, suffix_logits, signed_logits, tokens
            print(f"MLP2 validation batch {stop // runtime.BATCH}/{runtime.CALLS}", flush=True)
    torch.cuda.synchronize(device)
    if precision_audit is None or precision_audit.get("passed") is not True or not (
        runtime.call_ledger_passes(call_ledger)
    ) or not torch.equal(additivity[:, 0].long(), eligible.sum(1).long()):
        raise RuntimeError("MLP2 validation precision/call/additivity ledger failed")
    gauge_pass = _selector_gauge_passes(suffix_result)
    protocol_audits = {
        "exact_price_and_support_replay": (
            all(receipt["stored_scalar_values"] == 1_770_624
                for receipt in program_receipts.values())
            and support_hashes == suffix_result["tensor_hashes"]["supports"]
        ),
        "gauge_and_permutation_replay": gauge_pass,
        "physical_materialization_replay": materialization["bit_exact"],
        "physical_call_ledger_replay": runtime.call_ledger_passes(call_ledger),
        "float32_cpu_float64_precision_audit": precision_audit["passed"],
    }
    if not all(protocol_audits.values()):
        raise RuntimeError("MLP2 validation protocol audit failed")
    ledger_bundle = {
        "schema": "mlp2_cmr_v1_validation_ledger",
        "ledgers": runtime.pack_ledgers(ledgers),
        "margin_counts": margin_counts,
        "margin_support_counts": margin_support,
        "epsilon_grid": epsilon,
        "geometry": runtime.pack_geometry(geometry_batches),
        "additivity": additivity,
    }
    score = runtime.score_validation_bundle(
        ledger_bundle, protocol_audits=protocol_audits,
    )
    result = {
        "schema": "mlp2_cmr_v1_validation_result",
        "status": (
            "validation_passed_replication_implementation_authorized"
            if score["validation_passed"]
            else "validation_failed_replication_remains_sealed"
        ),
        "checkpoint": checkpoint.__dict__,
        "checkpoint_after_load": checkpoint_after_load.__dict__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "model_dtype": str(torch.bfloat16),
        "role_summary": role_summary,
        "support_hashes": support_hashes,
        "program_receipts": program_receipts,
        "physical_materialization": materialization,
        "selector_gauge_and_permutation_audit": suffix_result[
            "gauge_and_permutation_audit"
        ],
        "call_ledger": call_ledger,
        "precision_audit": precision_audit,
        "protocol_audits": protocol_audits,
        "score": score,
        "fit_reference": {
            "fit_observations": fit_result["fit_observations"],
            "selector_documents": suffix_result["documents"],
            "calibration_documents": calibration_result["documents"],
        },
        "runtime_seconds": time.time() - started,
        "validation_opened": True,
        "replication_opened": False,
        "raw_logits_published": False,
        "per_token_losses_published": False,
        "validation_targets_published": False,
    }
    del model, role, ledgers, programs
    torch.cuda.empty_cache()
    return ledger_bundle, result


def validate_output_semantics(
    bundle: Any, result: Any, parent_bytes: dict[str, bytes],
) -> None:
    if not isinstance(result, dict) or result.get("schema") != (
        "mlp2_cmr_v1_validation_result"
    ) or result.get("replication_opened") is not False or result.get(
        "raw_logits_published"
    ) is not False or result.get("per_token_losses_published") is not False or (
        result.get("validation_targets_published") is not False
    ):
        raise RuntimeError("MLP2 validation result schema/forbidden output changed")
    replay_score = runtime.score_validation_bundle(
        bundle, protocol_audits=result["protocol_audits"],
    )
    if replay_score != result.get("score") or result.get("status") != (
        "validation_passed_replication_implementation_authorized"
        if replay_score["validation_passed"]
        else "validation_failed_replication_remains_sealed"
    ) or replay_score["replication_authorized"] != replay_score["validation_passed"]:
        raise RuntimeError("MLP2 validation score replay failed")
    role = _load_torch(parent_bytes["role_rows"])
    if projection.validate_role(role) != result.get("role_summary"):
        raise RuntimeError("MLP2 validation role replay failed")
    if result.get("checkpoint") != result.get("checkpoint_after_load") or result.get(
        "checkpoint", {}
    ).get("weights_sha256") != facade.WEIGHTS_SHA256 or result.get("device_name") != (
        "NVIDIA GeForce RTX 5090"
    ) or result.get("model_dtype") != str(torch.bfloat16) or not runtime.call_ledger_passes(
        result.get("call_ledger", {})
    ) or result.get("precision_audit", {}).get("passed") is not True:
        raise RuntimeError("MLP2 validation runtime replay failed")


def main() -> None:
    namespace = (AUTHORITY, LEDGER, RESULT, RECEIPT, FAILURE, LOCK)
    if any(path.exists() for path in namespace):
        raise RuntimeError("MLP2 validation namespace already exists")
    source_commit, source_hashes = committed_source()
    parents, parent_bytes = protected_inputs()
    experiment_id = "bilin18_mlp2_cmr_v1_validation"
    nonce = secrets.token_hex(32)
    inode = None
    authority = None
    try:
        projection.base.write_create_only(LOCK, canonical_json_bytes({
            "experiment_id": experiment_id, "nonce": nonce,
        }))
        stat = LOCK.stat(follow_symlinks=False)
        inode = (stat.st_dev, stat.st_ino)
        validate_claim(nonce, inode)
        authority = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "status": "authority_frozen_before_validation_model_access",
            "source_commit": source_commit,
            "source_hashes": source_hashes,
            "parents": parents,
            "authorized_role": "VALIDATION",
            "authorized_forward_calls": {
                arm: runtime.CALLS for arm in runtime.ALL_ARMS
            },
            "authorized_outputs": [LEDGER.name, RESULT.name, RECEIPT.name],
            "replication_authorized": False,
            "forbidden_outputs": [
                "raw logits", "per-token losses", "raw validation targets",
                "REPLICATION data or outcome",
            ],
        }
        projection.base.write_create_only(AUTHORITY, canonical_json_bytes(authority))
        authority_hash = file_sha256(AUTHORITY)
        capability = _mint_capability(nonce, inode, authority_hash)
        ledger_bundle, result = collect(parent_bytes, capability)
        if not capability.consumed:
            raise RuntimeError("validation capability was not consumed")
        guard_inputs(source_hashes, parents, nonce, inode, authority_hash)
        projection.base.publish_torch_create_only(LEDGER, ledger_bundle)
        ledger_hash_before = file_sha256(LEDGER)
        replay_bundle = torch.load(LEDGER, map_location="cpu", weights_only=True)
        ledger_hash = file_sha256(LEDGER)
        if ledger_hash != ledger_hash_before:
            raise RuntimeError("validation ledger changed across semantic load")
        result.update({
            "authority_sha256": authority_hash,
            "ledger_sha256": ledger_hash,
            "source_commit": source_commit,
            "source_hashes": source_hashes,
            "parents": parents,
        })
        projection.base.write_create_only(RESULT, canonical_json_bytes(result))
        result_hash = file_sha256(RESULT)
        replay_result = json.loads(RESULT.read_bytes())
        if replay_result != result:
            raise RuntimeError("validation JSON semantic replay failed")
        validate_output_semantics(replay_bundle, replay_result, parent_bytes)
        final_guard(
            source_hashes, parents, nonce, inode, authority_hash, ledger_hash,
            result_hash,
        )
        receipt = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "status": "mlp2_cmr_v1_validation_complete_receipt_last",
            "authority_sha256": authority_hash,
            "ledger_sha256": ledger_hash,
            "result_sha256": result_hash,
            "source_commit": source_commit,
            "source_hashes": source_hashes,
            "parents": parents,
            "validation_passed": result["score"]["validation_passed"],
            "authorized_for_replication_implementation": result["score"][
                "replication_authorized"
            ],
            "authorized_for_replication_execution": False,
        }

        def receipt_guard() -> None:
            final_guard(
                source_hashes, parents, nonce, inode, authority_hash, ledger_hash,
                result_hash,
            )
            if RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("validation terminal namespace changed")
            validate_claim(nonce, inode)

        projection.write_create_only_guarded(
            RECEIPT, canonical_json_bytes(receipt), before_link=receipt_guard,
        )
    except BaseException as error:
        if inode is not None and not RECEIPT.exists() and not FAILURE.exists():
            partial = {
                path.name: file_sha256(path) if path.exists() else None
                for path in (AUTHORITY, LEDGER, RESULT)
            }
            failure = {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "status": "mlp2_cmr_v1_validation_failed_invalid_no_scientific_decision",
                "error_type": type(error).__name__,
                "error": str(error),
                "partial_outputs": partial,
                "replication_opened": False,
            }

            def failure_guard() -> None:
                validate_claim(nonce, inode)
                for relative, expected in source_hashes.items():
                    if file_sha256(ROOT / relative) != expected:
                        raise RuntimeError("validation source changed before failure")
                current, _ = protected_inputs()
                current_partial = {
                    path.name: file_sha256(path) if path.exists() else None
                    for path in (AUTHORITY, LEDGER, RESULT)
                }
                if current != parents or current_partial != partial or (
                    RECEIPT.exists() or FAILURE.exists()
                ):
                    raise RuntimeError("validation failure snapshot changed")
                validate_claim(nonce, inode)

            try:
                projection.write_create_only_guarded(
                    FAILURE, canonical_json_bytes(failure), before_link=failure_guard,
                )
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    main()
