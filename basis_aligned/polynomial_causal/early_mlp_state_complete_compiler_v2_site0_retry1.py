#!/usr/bin/env python3
"""Versioned compiler-v2 site-0 retry with one unified CUDA scorer currency."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
PROTOCOL = HERE / "early_mlp_state_complete_compiler_v2_site0_retry1_protocol.json"
PARENT_FAILURE = BQ / "early_mlp_state_complete_compiler_v2_site0_manifest.json"
DIAGNOSTIC_PROTOCOL = HERE / "early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1_protocol.json"
DIAGNOSTIC_RESULT = BQ / "early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1.json"
DIAGNOSTIC_MANIFEST = BQ / "early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1_manifest.json"
DIAGNOSTIC_RECEIPT = BQ / "early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1_receipt.json"
V1_AUTHORITY = BQ / "early_mlp_affine_compiler_v1_authority.json"
V1_RESULT = BQ / "early_mlp_affine_compiler_v1_results.json"
V1_ERRATUM = BQ / "early_mlp_affine_compiler_v1_erratum.json"
PREFLIGHT_V1_FAILURE = BQ / "early_mlp_state_complete_compiler_v2_preflight_manifest.json"
ARTIFACT = BQ / "early_mlp_state_complete_compiler_v2_site0_retry1_programs.pt"
RECEIPT = BQ / "early_mlp_state_complete_compiler_v2_site0_retry1_receipt.json"
RESULT = BQ / "early_mlp_state_complete_compiler_v2_site0_retry1_results.json"
MANIFEST = BQ / "early_mlp_state_complete_compiler_v2_site0_retry1_manifest.json"
LOCK = Path("/workspace/runs/.early_mlp_state_complete_compiler_v2_site0_retry1.lock")
OUTPUTS = (ARTIFACT, RECEIPT, RESULT, MANIFEST)
ORIGINAL_ABSENT = (
    BQ / "early_mlp_state_complete_compiler_v2_site0_programs.pt",
    BQ / "early_mlp_state_complete_compiler_v2_site0_receipt.json",
    BQ / "early_mlp_state_complete_compiler_v2_site0_results.json",
)

sys.path.insert(0, str(HERE))
import code_ood_oracle as code_oracle  # noqa: E402
import early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1 as diagnostic  # noqa: E402
import early_mlp_state_complete_compiler_v2_site0 as failed  # noqa: E402
import frozen_ship_oracle_v2 as frozen  # noqa: E402
import joint_early_mlp_oracle_factorial_authoritative as exact_runner  # noqa: E402
import joint_early_mlp_pca_composition_authoritative_v3 as v3  # noqa: E402
import prepare_fineweb_oracle_rows as old_rows  # noqa: E402
import prepare_state_complete_compiler_rows_v2 as fresh_rows  # noqa: E402
import state_complete_compiler_fit_v2 as fit  # noqa: E402
import state_complete_compiler_runtime_v2 as runtime  # noqa: E402
import state_complete_compiler_selection_v2 as selection  # noqa: E402


PINS = {
    **failed.PINS,
    PROTOCOL: "81379b84a65452d90e15f6260008d155ba8bbf1fc21702b7427adc0b663148dd",
    PARENT_FAILURE: "0903b0822b935e7dd6225da46dd1e58064ec275b80fd9c599685cea8b8b05f36",
    DIAGNOSTIC_PROTOCOL: "faf3de94d6e2bf7f33db52f5bdf074c10926535d10c18ced22d303f3d8ea5418",
    DIAGNOSTIC_RESULT: "307b74ab95856d943ceaad06f246cfa9cf70466dda84cb80bd559dabf64b634e",
    DIAGNOSTIC_MANIFEST: "8ce7b8cb38399eb5ac502ef9d0b78825cf97977cfa86384c355bde2d9f041c00",
    DIAGNOSTIC_RECEIPT: "fffecb9a3d99a4f6b7f615c96caf9bd7e2ac9c4d4788d610df7edf18d6a1d9fd",
    V1_AUTHORITY: "2c1ad6ca099d4910f83531fbc958f070da600f554799947c5118284e7b939e28",
    V1_RESULT: "f189cd4f98641ef54ba15687f7368d853233c5adb12cd98c8b6a2be798ec2051",
    V1_ERRATUM: "0196b5c0402b3a0be03b613453e7976ed5f9f0cfa75565e4c1dfa8d8871251b4",
    PREFLIGHT_V1_FAILURE: (
        "219f5fb7a02d782f4d04b54fe8f1d6a961d601323971513ddc3945d649a31526"
    ),
}
SOURCE_CLOSURE = tuple(dict.fromkeys((
    Path(__file__),
    HERE / "test_early_mlp_state_complete_compiler_v2_site0_retry1.py",
    PROTOCOL,
    HERE / "early_mlp_state_complete_compiler_v2_preflight_r2.py",
    HERE / "test_early_mlp_state_complete_compiler_v2_preflight_r2.py",
    *diagnostic.SOURCE_CLOSURE,
)))
PROTECTED = tuple(dict.fromkeys((*PINS, *ORIGINAL_ABSENT, *failed.PROTECTED)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return code_oracle.tensor_sha256(value.detach().cpu().contiguous())


def write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_torch_atomic(value: Any, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def protected_snapshot() -> dict[str, str | None]:
    return {str(path): file_sha256(path) if path.is_file() else None for path in PROTECTED}


def verify_pins_and_sources() -> tuple[str, dict[str, str]]:
    for path, expected in PINS.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"pinned compiler-v2 retry1 input changed: {path}")
    if any(path.exists() for path in ORIGINAL_ABSENT):
        raise RuntimeError("parent failed namespace gained an output after failure")
    parent = json.loads(PARENT_FAILURE.read_text())
    if parent.get("status") != "failed_compiler_v2_site0" or parent.get(
        "source_commit"
    ) != "33d0629af2fe2161eee3d62d58bd2b31b1025cb0":
        raise RuntimeError("parent site0 failure status changed")
    receipt = json.loads(DIAGNOSTIC_RECEIPT.read_text())
    if receipt.get("scorer_only_retry_licensed") is not True or receipt.get(
        "representation_retry_licensed"
    ) is not False:
        raise RuntimeError("numeric diagnostic did not license scorer-only retry")
    for authority, label in ((parent, "parent failure"), (receipt, "numeric diagnostic")):
        for relative, expected in authority.get("source_hashes", {}).items():
            path = ROOT / relative
            if not path.is_file() or file_sha256(path) != expected:
                raise RuntimeError(f"{label} source lineage changed: {relative}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    origin = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != origin:
        raise RuntimeError("compiler-v2 retry1 requires HEAD==origin/main")
    hashes = {}
    for path in SOURCE_CLOSURE:
        relative = path.resolve().relative_to(ROOT.resolve())
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)], cwd=ROOT,
            capture_output=True, text=True,
        )
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=ROOT,
        )
        if tracked.returncode or dirty.returncode:
            raise RuntimeError(f"retry1 source is not committed and clean: {relative}")
        hashes[str(relative)] = file_sha256(path)
    return head, hashes


@torch.no_grad()
def cached_teacher_bank(
    sa: Any, hook: runtime.StateCompleteCorrectionHook, rows: torch.Tensor,
    twall: Mapping[int, Any], all_attention: frozenset[int],
) -> tuple[torch.Tensor, float, float, torch.Tensor, dict[str, Any]]:
    """Cache logits on CPU but compute every metric in the frozen CUDA currency."""

    parts: dict[str, list[torch.Tensor]] = {"OON": [], "NON": []}
    targets_parts, idx_parts = [], []
    counters = {}
    for name, states, allowed in (
        ("OON", {0: "O", 1: "O"}, {0, 1}),
        ("NON", {1: "O"}, {1}),
    ):
        hook.configure(states)
        with runtime.OriginalMLPCallGuard(sa.H, allowed) as guard:
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8].to(sa.DEV)
                idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
                logits = sa.fwd_arm(
                    idx, all_attention, twall, frozenset(range(18))
                ).float()[:, 64:]
                parts[name].append(logits.cpu().contiguous())
                if name == "OON":
                    targets_parts.append(targets.cpu())
                    idx_parts.append(idx.cpu())
        guard.assert_contract(require_allowed_calls=True)
        counters[name] = dict(guard.counts)
    teacher, non = torch.cat(parts["OON"]), torch.cat(parts["NON"])
    targets_all, idx_all = torch.cat(targets_parts), torch.cat(idx_parts)
    copy_all = failed._copy_mask(idx_all, targets_all)[:, 64:]
    kl_sum = 0.0
    ce_sum = 0.0
    copy_sum = 0.0
    copy_count = 0
    teacher_rows = []
    token_count = 0
    for start in range(0, len(rows), 8):
        stop = min(start + 8, len(rows))
        t = teacher[start:stop].to(sa.DEV)
        n = non[start:stop].to(sa.DEV)
        target = targets_all[start:stop, 64:].to(sa.DEV)
        t_logp = F.log_softmax(t, dim=-1)
        n_logp = F.log_softmax(n, dim=-1)
        kl = (t_logp.exp() * (t_logp - n_logp)).sum(dim=-1)
        kl_sum += float(kl.double().sum())
        token_count += kl.numel()
        non_ce = F.cross_entropy(
            n.reshape(-1, n.shape[-1]), target.reshape(-1), reduction="none"
        ).view(stop - start, -1)
        teacher_ce = F.cross_entropy(
            t.reshape(-1, t.shape[-1]), target.reshape(-1), reduction="none"
        ).view(stop - start, -1)
        teacher_rows.append(teacher_ce.double().mean(dim=1).cpu())
        ce_sum += float(non_ce.double().sum())
        copy = copy_all[start:stop].to(sa.DEV)
        if bool(copy.any()):
            copy_sum += float(non_ce[copy].double().sum())
            copy_count += int(copy.sum())
    denominator = kl_sum / token_count
    if not denominator > 0.0:
        raise RuntimeError("retry1 OON/NON teacher-KL denominator is not positive")
    return teacher, denominator, copy_sum / max(copy_count, 1), torch.cat(
        teacher_rows
    ), {
        "NON_global_ce": ce_sum / token_count,
        "calls": counters,
        "validation_rows": int(len(rows)),
        "validation_tokens": int(token_count),
        "scorer": "CUDA float32 per-token; float64 accumulation",
    }


@torch.no_grad()
def score_live_arm(
    sa: Any, hook: runtime.StateCompleteCorrectionHook, rows: torch.Tensor,
    twall: Mapping[int, Any], all_attention: frozenset[int],
    teacher: torch.Tensor, baseline_copy_ce: float,
    *, states: Mapping[int, str], allowed: set[int], program_name: str | None = None,
) -> tuple[dict[str, float], dict[int, int]]:
    hook.configure(states, **({} if program_name is None else {"program_name": program_name}))
    kl_sum = 0.0
    token_count = 0
    ce_sum = 0.0
    copy_sum = 0.0
    copy_count = 0
    row_drift = []
    logit_max = 0.0
    with runtime.OriginalMLPCallGuard(sa.H, allowed) as guard:
        row_offset = 0
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8].to(sa.DEV)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            candidate = sa.fwd_arm(
                idx, all_attention, twall, frozenset(range(18))
            ).float()[:, 64:]
            t = teacher[row_offset:row_offset + len(batch)].to(sa.DEV)
            target = targets[:, 64:]
            t_logp = F.log_softmax(t, dim=-1)
            c_logp = F.log_softmax(candidate, dim=-1)
            kl = (t_logp.exp() * (t_logp - c_logp)).sum(dim=-1)
            kl_sum += float(kl.double().sum())
            token_count += kl.numel()
            c_ce = F.cross_entropy(
                candidate.reshape(-1, candidate.shape[-1]),
                target.reshape(-1), reduction="none",
            ).view(len(batch), -1)
            t_ce = F.cross_entropy(
                t.reshape(-1, t.shape[-1]), target.reshape(-1), reduction="none"
            ).view(len(batch), -1)
            row_drift.append(
                (c_ce.double().mean(dim=1) - t_ce.double().mean(dim=1)).abs().cpu()
            )
            ce_sum += float(c_ce.double().sum())
            copy = failed._copy_mask(idx, targets)[:, 64:]
            if bool(copy.any()):
                copy_sum += float(c_ce[copy].double().sum())
                copy_count += int(copy.sum())
            logit_max = max(logit_max, float((candidate - t).abs().max()))
            row_offset += len(batch)
    guard.assert_contract(require_allowed_calls=True)
    copy_ce = copy_sum / max(copy_count, 1)
    drift = torch.cat(row_drift)
    return {
        "candidate_teacher_kl": kl_sum / token_count,
        "global_ce": ce_sum / token_count,
        "copy_ce": copy_ce,
        "copy_count": copy_count,
        "copy_worsening": copy_ce - baseline_copy_ce,
        "row_ce_max_abs_drift": float(drift.max()),
        "row_ce_mean_abs_drift": float(drift.mean()),
        "logit_max_abs_drift": logit_max,
    }, dict(guard.counts)


@torch.no_grad()
def full_native_gate(
    sa: Any, hook: runtime.StateCompleteCorrectionHook, rows: torch.Tensor,
    twall: Mapping[int, Any], all_attention: frozenset[int], state: Mapping[str, Any],
    capture: Mapping[str, torch.Tensor], teacher: torch.Tensor,
    baseline_copy_ce: float, basis: torch.Tensor,
) -> dict[str, Any]:
    device_state = failed._device_state(state, sa.DEV)
    predicted_parts = []
    for start in range(0, capture["z"].shape[0], 1024):
        predicted_parts.append(runtime.runtime_projected_output(
            capture["z"][start:start + 1024].to(sa.DEV), device_state
        ).cpu())
    coefficient_error = torch.cat(predicted_parts) - capture["p"]
    physical_parts = []
    basis_device = basis.to(sa.DEV).float()
    for start in range(0, coefficient_error.shape[0], 1024):
        physical_parts.append((
            coefficient_error[start:start + 1024].to(sa.DEV) @ basis_device.T
        ).cpu())
    physical_max = float(torch.cat(physical_parts).abs().max())
    physical_scale = max(1.0, float(
        (capture["c"] @ basis.cpu().float().T).abs().max()
    ))
    physical_tolerance = 4e-6 * physical_scale
    if physical_max > physical_tolerance:
        raise RuntimeError(
            f"retry1 full-native physical gate failed: {physical_max}>{physical_tolerance}"
        )
    hook.programs = {"full_native": {0: device_state}}
    metrics, calls = score_live_arm(
        sa, hook, rows, twall, all_attention, teacher, baseline_copy_ce,
        states={0: "Q", 1: "O"}, allowed={1}, program_name="full_native",
    )
    if metrics["row_ce_max_abs_drift"] > 2e-6:
        raise RuntimeError(
            "retry1 full-native unified row-CE gate failed: "
            f"{metrics['row_ce_max_abs_drift']}>2e-6"
        )
    return {
        "passed": True,
        "physical_max_abs_error": physical_max,
        "physical_tolerance": physical_tolerance,
        **metrics,
        "poison_calls": calls,
    }


@torch.no_grad()
def score_candidate(
    sa: Any, hook: runtime.StateCompleteCorrectionHook, rows: torch.Tensor,
    twall: Mapping[int, Any], all_attention: frozenset[int], name: str,
    state: Mapping[str, Any], teacher: torch.Tensor, denominator: float,
    baseline_copy_ce: float,
) -> tuple[dict[str, Any], dict[int, int]]:
    hook.programs = {name: {0: failed._device_state(state, sa.DEV)}}
    raw, calls = score_live_arm(
        sa, hook, rows, twall, all_attention, teacher, baseline_copy_ce,
        states={0: "Q", 1: "O"}, allowed={1}, program_name=name,
    )
    metrics = selection.direct_recovery(raw.pop("candidate_teacher_kl"), denominator)
    metrics.update(raw)
    metrics["price"] = selection.state_price(state)
    return metrics, calls


def artifact_payload(
    candidates: Mapping[str, Mapping[str, Any]], frozen_selection: Mapping[str, Any],
    controls: Mapping[str, Any], diagnostics_payload: Mapping[str, Any],
    source_commit: str, source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = failed.artifact_payload(
        candidates, frozen_selection, controls, diagnostics_payload,
        source_commit, source_hashes,
    )
    payload.update({
        "status": "pending_site0_retry1_last_written_authority_receipt",
        "authority": "compiler_v2_site0_retry1_pending",
        "authorized_for_training": False,
        "training_license_sites": [],
        "retry_namespace": 1,
        "parent_failure_manifest_sha256": PINS[PARENT_FAILURE],
        "numeric_diagnostic_receipt_sha256": PINS[DIAGNOSTIC_RECEIPT],
        "retry1_protocol_sha256": PINS[PROTOCOL],
        "scorer_currency": "CUDA float32 per-token; float64 accumulation",
    })
    return payload


def validate_pending_outputs(
    *, require_receipt_absent: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if require_receipt_absent and RECEIPT.exists():
        raise RuntimeError("retry1 authority receipt exists before finalization")
    payload = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
    result = json.loads(RESULT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if payload.get("status") != "pending_site0_retry1_last_written_authority_receipt" or (
        payload.get("authority") != "compiler_v2_site0_retry1_pending"
    ):
        raise RuntimeError("retry1 pending artifact identity changed")
    if payload.get("authorized_for_training") is not False or payload.get(
        "training_license_sites"
    ) != []:
        raise RuntimeError("retry1 pending artifact self-authorized training")
    if payload.get("parent_failure_manifest_sha256") != PINS[PARENT_FAILURE]:
        raise RuntimeError("retry1 artifact parent failure changed")
    selected = payload.get("selection", {}).get("selected")
    if selected not in payload.get("candidates", {}):
        raise RuntimeError("retry1 selected program is absent")
    if set(payload.get("controls", {})) != {"mean", "shuffle", "full_native"}:
        raise RuntimeError("retry1 controls are incomplete")
    artifact_sha256 = file_sha256(ARTIFACT)
    if result.get("status") != "completed_site0_validation_pending_authority_retry1" or (
        result.get("authorized_for_training") is not False
    ) or result.get("artifact_sha256") != artifact_sha256:
        raise RuntimeError("retry1 pending result binding changed")
    if manifest.get(
        "status"
    ) != "completed_integrity_pending_last_written_authority_receipt_retry1" or (
        manifest.get("authorized_for_training") is not False
    ):
        raise RuntimeError("retry1 finalized pending manifest identity changed")
    if manifest.get("artifact_sha256") != artifact_sha256 or manifest.get(
        "result_sha256"
    ) != file_sha256(RESULT):
        raise RuntimeError("retry1 pending manifest output binding changed")
    if manifest.get("component_tree_unchanged") is not True or manifest.get(
        "hook_restored_and_inert"
    ) is not True or manifest.get("outer_sa_main_returned") is not True:
        raise RuntimeError("retry1 outer integrity closure is incomplete")
    if manifest.get("protected_after_outer") != manifest.get("protected_before"):
        raise RuntimeError("retry1 protected state changed before authority")
    for row in (payload, result, manifest):
        if row.get("parent_failure_manifest_sha256") != PINS[PARENT_FAILURE] or row.get(
            "numeric_diagnostic_receipt_sha256"
        ) != PINS[DIAGNOSTIC_RECEIPT] or row.get("retry1_protocol_sha256") != PINS[PROTOCOL]:
            raise RuntimeError("retry1 pending lineage identity changed")
        if row.get("source_commit") != payload.get("source_commit") or row.get(
            "source_hashes"
        ) != payload.get("source_hashes"):
            raise RuntimeError("retry1 pending source closure changed")
    return payload, result, manifest


def validate_artifact() -> tuple[dict[str, Any], dict[str, Any]]:
    payload, result, manifest = validate_pending_outputs(require_receipt_absent=False)
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("status") != "frozen_site0_retry1_last_written_authority" or receipt.get(
        "authority"
    ) != "compiler_v2_sequential_site0_validation_freeze_retry1":
        raise RuntimeError("retry1 authority receipt identity changed")
    if receipt.get("authorized_for_training") is not True or receipt.get(
        "training_license_sites"
    ) != [1] or receipt.get("authorized_for_scored_experiments") is not False:
        raise RuntimeError("retry1 authority training scope changed")
    bindings = {
        "artifact_path": str(ARTIFACT.resolve()),
        "artifact_sha256": file_sha256(ARTIFACT),
        "artifact_bytes": ARTIFACT.stat().st_size,
        "result_path": str(RESULT.resolve()),
        "result_sha256": file_sha256(RESULT),
        "manifest_path": str(MANIFEST.resolve()),
        "manifest_sha256": file_sha256(MANIFEST),
        "parent_failure_manifest_sha256": PINS[PARENT_FAILURE],
        "numeric_diagnostic_receipt_sha256": PINS[DIAGNOSTIC_RECEIPT],
        "retry1_protocol_sha256": PINS[PROTOCOL],
        "source_commit": payload["source_commit"],
        "source_hashes": payload["source_hashes"],
    }
    if any(receipt.get(key) != value for key, value in bindings.items()):
        raise RuntimeError("retry1 authority receipt binding changed")
    if receipt.get("selected") != payload["selection"]["selected"] or receipt.get(
        "selected_family"
    ) != payload["selection"]["selected_family"]:
        raise RuntimeError("retry1 authority selection binding changed")
    if result.get("artifact_sha256") != receipt["artifact_sha256"] or manifest.get(
        "result_sha256"
    ) != receipt["result_sha256"]:
        raise RuntimeError("retry1 transitive authority binding changed")
    return payload, receipt


def run_claimed(before: Mapping[str, str | None]) -> None:
    source_commit, source_hashes = verify_pins_and_sources()
    old_receipt, _ = old_rows.validate_receipt()
    row_receipt, rows_full = fresh_rows.load_roles_and_validate(
        ("compiler_fit", "compiler_validation")
    )
    rows = {role: tensor[:, :257].contiguous() for role, tensor in rows_full.items()}
    document_ids = [record["document_id"] for record in
                    row_receipt["document_provenance"]["sets"]["compiler_fit"]]
    code_rows, _ = code_oracle.load_frozen_corpus()
    frozen.validate_frozen_ship_pair(old_receipt)
    bases_payload, _ = v3.validate_basis_pair()
    manifest = {
        "schema_version": 1,
        "status": "running_compiler_v2_site0_retry1",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "parent_failure_manifest_sha256": PINS[PARENT_FAILURE],
        "numeric_diagnostic_receipt_sha256": PINS[DIAGNOSTIC_RECEIPT],
        "retry1_protocol_sha256": PINS[PROTOCOL],
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "protected_before": dict(before),
    }
    write_json_atomic(manifest, MANIFEST)
    torch.manual_seed(exact_runner.SHIP_SEED)
    torch.cuda.manual_seed_all(exact_runner.SHIP_SEED)
    sys.path.insert(0, str(BQ))
    import ship_error_attrib as sa  # noqa: PLC0415
    started = time.time()
    callback_state: dict[str, bool] = {}

    def callback(twall: dict, all_attention: frozenset[int], _: float) -> None:
        realization, _ = frozen.restore_ship_realization(
            sa, twall, all_attention, old_receipt, code_rows
        )
        if realization != failed.SHIP_HASH:
            raise RuntimeError("frozen ship realization changed")
        component_before = exact_runner.component_tree_sha256(sa, twall, all_attention)
        basis = bases_payload["sites"][0]["basis"].to(sa.DEV).float()
        hook = runtime.StateCompleteCorrectionHook({
            0: basis,
            1: bases_payload["sites"][1]["basis"].to(sa.DEV).float(),
        }, {})
        prior_hook = sa.add_oracle_correction
        sa.add_oracle_correction = hook
        try:
            captured, fit_calls = failed.capture_site0_fit(
                sa, hook, rows["compiler_fit"], twall, all_attention
            )
            capture_hashes = {key: tensor_sha256(value) for key, value in captured.items()}
            fit_permutation = fit.expand_capture_permutation(
                fit.document_block_permutation(document_ids, failed.FIT_SEED)
            )
            validation_capture, validation_calls = failed.capture_site0_validation(
                sa, hook, rows["compiler_validation"], twall, all_attention
            )
            (teacher, denominator, baseline_copy, _,
             teacher_diagnostics) = cached_teacher_bank(
                sa, hook, rows["compiler_validation"], twall, all_attention
            )
            replay, replay_calls = score_live_arm(
                sa, hook, rows["compiler_validation"], twall, all_attention,
                teacher, baseline_copy, states={0: "O", 1: "O"}, allowed={0, 1},
            )
            if replay["logit_max_abs_drift"] != 0.0 or replay[
                "row_ce_max_abs_drift"
            ] != 0.0:
                raise RuntimeError(f"retry1 OON replay is not exact: {replay}")
            full_native_control = failed.full_native_state(sa.H[0], basis)
            full_native = full_native_gate(
                sa, hook, rows["compiler_validation"], twall, all_attention,
                full_native_control, validation_capture, teacher, baseline_copy, basis,
            )

            candidates, fit_diagnostics = failed.build_site0_candidates(
                captured, sa.H[0], basis, sa.DEV
            )
            shuffled_capture = failed.shuffled_fit_capture(captured, fit_permutation)
            shuffled_candidates, shuffle_fit_diagnostics = failed.build_site0_candidates(
                shuffled_capture, sa.H[0], basis, sa.DEV
            )
            mean_control = failed._cpu_state(fit.constant_state(captured["p"]))
            bank = {}
            calls = {
                "fit": fit_calls,
                "validation_capture": validation_calls,
                "OON_replay": replay_calls,
                "full_native": full_native["poison_calls"],
                **teacher_diagnostics.pop("calls"),
            }
            for index, (name, state) in enumerate(sorted(candidates.items())):
                metrics, candidate_calls = score_candidate(
                    sa, hook, rows["compiler_validation"], twall, all_attention,
                    name, state, teacher, denominator, baseline_copy,
                )
                bank[name] = {"state": state, "metrics": metrics}
                calls[name] = candidate_calls
                print(f"compiler-v2 retry1 validation {index + 1}/{len(candidates)} {name}",
                      flush=True)
            frozen_selection = selection.freeze_validation_selection(bank)
            shuffle_bank = {}
            for index, (name, state) in enumerate(sorted(shuffled_candidates.items())):
                metrics, candidate_calls = score_candidate(
                    sa, hook, rows["compiler_validation"], twall, all_attention,
                    name, state, teacher, denominator, baseline_copy,
                )
                shuffle_bank[name] = {"state": state, "metrics": metrics}
                calls[f"shuffle_{name}"] = candidate_calls
                print(
                    f"compiler-v2 retry1 shuffle validation "
                    f"{index + 1}/{len(shuffled_candidates)} {name}", flush=True,
                )
            frozen_shuffle = selection.freeze_control_selection(shuffle_bank)
            selected_shuffle = frozen_shuffle["selected"]
            controls = {
                "mean": mean_control,
                "shuffle": {
                    "state": failed._cpu_state(shuffled_candidates[selected_shuffle]),
                    "selection": frozen_shuffle,
                },
                "full_native": full_native_control,
            }
            diagnostics_payload = {
                "fit_capture_tensor_sha256": capture_hashes,
                "fit_capture_shapes": {key: list(value.shape) for key, value in captured.items()},
                "fit_adjoint_clip_threshold": fit_diagnostics["adjoint_clip_threshold"],
                "fit_document_permutation_sha256": tensor_sha256(fit_permutation),
                "fit_document_permutation_moved": int((fit_permutation != torch.arange(
                    len(fit_permutation))).sum()),
                "fit": fit_diagnostics,
                "shuffle_fit": shuffle_fit_diagnostics,
                "validation_capture_tensor_sha256": {
                    key: tensor_sha256(value) for key, value in validation_capture.items()
                },
                "teacher_kl_denominator_OON_vs_NON": denominator,
                "teacher": teacher_diagnostics,
                "OON_replay": replay,
                "full_native_validation_gate": full_native,
                "call_counters": calls,
                "scorer_currency": "CUDA float32 per-token; float64 accumulation",
            }
            payload = artifact_payload(
                candidates, frozen_selection, controls, diagnostics_payload,
                source_commit, source_hashes,
            )
            write_torch_atomic(payload, ARTIFACT)
            artifact_sha256 = file_sha256(ARTIFACT)
            result = {
                "schema_version": 1,
                "status": "completed_site0_validation_pending_authority_retry1",
                "authorized_for_scored_experiments": False,
                "authorized_for_training": False,
                "scope": "fit and validation only; no final rows, site1 claim, executable recovery, or whole-model credit",
                "parent_failure_manifest_sha256": PINS[PARENT_FAILURE],
                "numeric_diagnostic_receipt_sha256": PINS[DIAGNOSTIC_RECEIPT],
                "retry1_protocol_sha256": PINS[PROTOCOL],
                "selection": frozen_selection,
                "validation": {name: dict(row["metrics"]) for name, row in bank.items()},
                "shuffle_selection": frozen_shuffle,
                "shuffle_validation": {
                    name: dict(row["metrics"]) for name, row in shuffle_bank.items()
                },
                "OON_replay": replay,
                "full_native_validation_gate": full_native,
                "teacher_kl_denominator_OON_vs_NON": denominator,
                "candidate_count": len(candidates),
                "shuffle_candidate_count": len(shuffled_candidates),
                "artifact_sha256": artifact_sha256,
                "source_commit": source_commit,
                "source_hashes": source_hashes,
                "runtime_s": round(time.time() - started, 1),
            }
            write_json_atomic(result, RESULT)
            component_after = exact_runner.component_tree_sha256(sa, twall, all_attention)
            if component_after != component_before:
                raise RuntimeError("component tree changed during compiler-v2 retry1")
            after = protected_snapshot()
            if after != dict(before):
                raise RuntimeError("compiler-v2 retry1 changed protected artifacts")
            manifest.update({
                "status": "passed_callback_integrity_pending_outer_return_retry1",
                "result_sha256": file_sha256(RESULT),
                "artifact_sha256": artifact_sha256,
                "component_tree_unchanged": True,
                "protected_after_callback": after,
            })
            write_json_atomic(manifest, MANIFEST)
        finally:
            hook.clear()
            sa.add_oracle_correction = prior_hook
            exact_runner.require_inert_correction_state(sa)
            callback_state["hook_restored_and_inert"] = True

    sa.run_oracle_content_screen = callback
    sa.main(oracle_content_screen=True)
    if callback_state.get("hook_restored_and_inert") is not True:
        raise RuntimeError("retry1 callback did not certify restored inert hook state")
    after_outer = protected_snapshot()
    if after_outer != dict(before):
        raise RuntimeError("compiler-v2 retry1 changed protected artifacts after outer return")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("status") != "passed_callback_integrity_pending_outer_return_retry1":
        raise RuntimeError("retry1 callback did not leave a pending integrity manifest")
    manifest.update({
        "status": "completed_integrity_pending_last_written_authority_receipt_retry1",
        "authorized_for_training": False,
        "hook_restored_and_inert": True,
        "outer_sa_main_returned": True,
        "protected_after_outer": after_outer,
    })
    write_json_atomic(manifest, MANIFEST)
    payload, _, _ = validate_pending_outputs()
    receipt = {
        "schema_version": 1,
        "status": "frozen_site0_retry1_last_written_authority",
        "authority": "compiler_v2_sequential_site0_validation_freeze_retry1",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": True,
        "training_license_sites": [1],
        "parent_failure_manifest_sha256": PINS[PARENT_FAILURE],
        "numeric_diagnostic_receipt_sha256": PINS[DIAGNOSTIC_RECEIPT],
        "retry1_protocol_sha256": PINS[PROTOCOL],
        "artifact_path": str(ARTIFACT.resolve()),
        "artifact_sha256": file_sha256(ARTIFACT),
        "artifact_bytes": ARTIFACT.stat().st_size,
        "result_path": str(RESULT.resolve()),
        "result_sha256": file_sha256(RESULT),
        "manifest_path": str(MANIFEST.resolve()),
        "manifest_sha256": file_sha256(MANIFEST),
        "selected": payload["selection"]["selected"],
        "selected_family": payload["selection"]["selected_family"],
        "freeze_rule": (
            "Atomic receipt written only after callback integrity, hook restoration, "
            "outer sa.main return, and exact pending-output validation."
        ),
        "source_commit": source_commit,
        "source_hashes": source_hashes,
    }
    write_json_atomic(receipt, RECEIPT)


def main() -> None:
    existing = [str(path) for path in OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite compiler-v2 retry1 outputs: {existing}")
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"compiler-v2 retry1 already claimed: {LOCK}") from error
    before = protected_snapshot()
    try:
        run_claimed(before)
    except BaseException as error:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        manifest.update({
            "schema_version": 1,
            "status": "failed_compiler_v2_site0_retry1",
            "authorized_for_scored_experiments": False,
            "authorized_for_training": False,
            "parent_failure_manifest_sha256": PINS[PARENT_FAILURE],
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "protected_after": protected_snapshot(),
            "recovery": "Preserve outputs; never overwrite or relabel parent failure.",
        })
        write_json_atomic(manifest, MANIFEST)
        raise
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    main()
