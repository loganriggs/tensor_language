#!/usr/bin/env python3
"""Validation-only numeric diagnosis of the preserved compiler-v2 site-0 failure.

This is not a retry and cannot select or authorize a compiler.  It compares
predeclared algebraically equivalent full-native evaluation orders solely to
locate the frozen row-CE integrity failure preserved in the pinned manifest.
"""

from __future__ import annotations

import hashlib
import json
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
FAILURE = BQ / "early_mlp_state_complete_compiler_v2_site0_manifest.json"
FAILURE_SHA256 = "0903b0822b935e7dd6225da46dd1e58064ec275b80fd9c599685cea8b8b05f36"
PROTOCOL = HERE / "early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1_protocol.json"
PROTOCOL_SHA256 = "faf3de94d6e2bf7f33db52f5bdf074c10926535d10c18ced22d303f3d8ea5418"
RESULT = BQ / "early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1.json"
MANIFEST = BQ / "early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1_manifest.json"
RECEIPT = BQ / "early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1_receipt.json"
LOCK = Path("/workspace/runs/.early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1.lock")
OUTPUTS = (RESULT, MANIFEST, RECEIPT)

sys.path.insert(0, str(HERE))
import code_ood_oracle as code_oracle  # noqa: E402
import early_mlp_state_complete_compiler_v2 as compiler  # noqa: E402
import early_mlp_state_complete_compiler_v2_preflight as preflight  # noqa: E402
import early_mlp_state_complete_compiler_v2_site0 as site0  # noqa: E402
import frozen_ship_oracle_v2 as frozen  # noqa: E402
import joint_early_mlp_oracle_factorial_authoritative as exact_runner  # noqa: E402
import joint_early_mlp_pca_composition_authoritative_v3 as v3  # noqa: E402
import prepare_fineweb_oracle_rows as old_rows  # noqa: E402
import prepare_state_complete_compiler_rows_v2 as fresh_rows  # noqa: E402
import state_complete_compiler_runtime_v2 as runtime  # noqa: E402


SOURCE_CLOSURE = tuple(dict.fromkeys((
    Path(__file__),
    HERE / "test_early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1.py",
    PROTOCOL,
    *site0.SOURCE_CLOSURE,
)))
ORIGINAL_ABSENT = (site0.ARTIFACT, site0.RECEIPT, site0.RESULT)
PROTECTED = tuple(dict.fromkeys((FAILURE, *ORIGINAL_ABSENT, *site0.PROTECTED)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def protected_snapshot() -> dict[str, str | None]:
    return {str(path): file_sha256(path) if path.is_file() else None for path in PROTECTED}


def verify_sources() -> tuple[str, dict[str, str]]:
    if file_sha256(FAILURE) != FAILURE_SHA256:
        raise RuntimeError("pinned compiler-v2 site0 failure changed")
    if file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("numeric diagnostic protocol changed")
    failure = json.loads(FAILURE.read_text())
    if failure.get("status") != "failed_compiler_v2_site0" or not str(
        failure.get("failure_message", "")
    ).startswith("full-native live row-CE gate failed"):
        raise RuntimeError("pinned failure is not the registered integrity failure")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    origin = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != origin:
        raise RuntimeError("numeric diagnostic requires HEAD==origin/main")
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
            raise RuntimeError(f"diagnostic source is not committed and clean: {relative}")
        hashes[str(relative)] = file_sha256(path)
    return head, hashes


def variant_states(block: Any, basis: torch.Tensor) -> dict[str, dict[str, Any]]:
    """Predeclared equivalent full-native serialization/evaluation orders."""

    left, right, down, bias = preflight._native_tensors(block)
    left32 = left.detach().cpu().float().contiguous()
    right32 = right.detach().cpu().float().contiguous()
    down32 = down.detach().cpu().float().contiguous()
    bias32 = bias.detach().cpu().float().contiguous()
    basis32 = basis.detach().cpu().float().contiguous()
    canonical64 = compiler.project_native_weights(
        left32, right32, down32, bias32, basis32
    )
    common32 = {"left": left32, "right": right32}
    common64 = {"left": left32.double(), "right": right32.double()}
    return {
        "canonical_q64cast_eval32": {
            **{key: value.float() for key, value in canonical64.items()
               if key != "indices"}, "mode": "native32",
        },
        "uncanonical_q64cast_eval32": {
            **common32,
            "projected_decoder": (down32.double().T @ basis32.double()).float(),
            "beta": (bias32.double() @ basis32.double()).float(),
            "mode": "native32",
        },
        "uncanonical_q32_eval32": {
            **common32,
            "projected_decoder": down32.T @ basis32,
            "beta": bias32 @ basis32,
            "mode": "native32",
        },
        "canonical_eval64": {
            **{key: value.double() for key, value in canonical64.items()
               if key != "indices"}, "mode": "native64",
        },
        "uncanonical_eval64": {
            **common64,
            "projected_decoder": down32.double().T @ basis32.double(),
            "beta": bias32.double() @ basis32.double(),
            "mode": "native64",
        },
        "staged_original_association_eval32": {
            **common32, "down": down32, "bias": bias32, "basis": basis32,
            "mode": "staged32",
        },
        "staged_original_association_eval64": {
            **common64, "down": down32.double(), "bias": bias32.double(),
            "basis": basis32.double(), "mode": "staged64",
        },
    }


def projected_output(z: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor:
    flat = z.reshape(-1, compiler.D_MODEL)
    mode = state["mode"]
    if mode in ("native32", "native64"):
        dtype = torch.float32 if mode == "native32" else torch.float64
        x = flat.to(dtype)
        products = (x @ state["left"].to(x.device).T) * (
            x @ state["right"].to(x.device).T
        )
        return products @ state["projected_decoder"].to(x.device) + state[
            "beta"
        ].to(x.device)
    if mode in ("staged32", "staged64"):
        dtype = torch.float32 if mode == "staged32" else torch.float64
        x = flat.to(dtype)
        products = (x @ state["left"].to(x.device).T) * (
            x @ state["right"].to(x.device).T
        )
        full = products @ state["down"].to(x.device).T + state["bias"].to(x.device)
        return full @ state["basis"].to(x.device)
    raise ValueError(f"unknown numeric diagnostic mode: {mode}")


class VariantHook(runtime.StateCompleteCorrectionHook):
    variant: Mapping[str, Any] | None = None

    def __call__(self, site: int, block: Any, z: torch.Tensor, mo: torch.Tensor) -> torch.Tensor:
        if self.site_states.get(site, "N") == "Q":
            if site != 0 or self.variant is None:
                raise RuntimeError("numeric diagnostic Q is restricted to site 0")
            self.calls[site] = self.calls.get(site, 0) + 1
            basis = self.bases[site].to(z.device)
            predicted = projected_output(z, self.variant)
            live_mo = mo.float().reshape(-1, compiler.D_MODEL) @ basis
            coefficients = predicted - live_mo.to(predicted.dtype)
            delta = (coefficients @ basis.to(predicted.dtype).T).view_as(mo)
            return mo + delta.to(mo.dtype)
        return super().__call__(site, block, z, mo)


@torch.no_grad()
def physical_metrics(
    state: Mapping[str, Any], capture: Mapping[str, torch.Tensor],
    basis: torch.Tensor, device: Any,
) -> dict[str, float]:
    parts = []
    for start in range(0, capture["z"].shape[0], 1024):
        parts.append(projected_output(
            capture["z"][start:start + 1024].to(device), state
        ).float().cpu())
    coefficient_error = torch.cat(parts) - capture["p"].float()
    physical = coefficient_error.to(device) @ basis.to(device).float().T
    return {
        "coefficient_max_abs_error": float(coefficient_error.abs().max()),
        "physical_max_abs_error": float(physical.abs().max()),
    }


@torch.no_grad()
def row_ce_metrics(
    sa: Any, hook: VariantHook, state: Mapping[str, Any] | None, rows: torch.Tensor,
    teacher: torch.Tensor, twall: Mapping[int, Any],
    all_attention: frozenset[int],
) -> dict[str, Any]:
    if state is None:
        hook.variant = None
        hook.configure({0: "O", 1: "O"})
        allowed = {0, 1}
    else:
        hook.variant = state
        hook.programs = {"numeric": {0: {"placeholder": True}}}
        hook.configure({0: "Q", 1: "O"}, program_name="numeric")
        allowed = {1}
    drift32, drift64 = [], []
    kl_sum = 0.0
    token_count = 0
    logit_max = 0.0
    logit_mean_numerator = 0.0
    logit_count = 0
    with runtime.OriginalMLPCallGuard(sa.H, allowed) as guard:
        row_offset = 0
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8].to(sa.DEV)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            candidate = sa.fwd_arm(
                idx, all_attention, twall, frozenset(range(18))
            ).float()[:, 64:]
            teacher_batch = teacher[row_offset:row_offset + len(batch)].to(sa.DEV)
            logit_error = (candidate - teacher_batch).abs()
            logit_max = max(logit_max, float(logit_error.max()))
            logit_mean_numerator += float(logit_error.double().sum())
            logit_count += logit_error.numel()
            target = targets[:, 64:]
            for dtype, sink in ((torch.float32, drift32), (torch.float64, drift64)):
                t_ce = F.cross_entropy(
                    teacher_batch.to(dtype).reshape(-1, teacher_batch.shape[-1]),
                    target.reshape(-1), reduction="none",
                ).view(len(batch), -1).double().mean(dim=1)
                c_ce = F.cross_entropy(
                    candidate.to(dtype).reshape(-1, candidate.shape[-1]),
                    target.reshape(-1), reduction="none",
                ).view(len(batch), -1).double().mean(dim=1)
                sink.append((c_ce - t_ce).abs().cpu())
            teacher_logp = F.log_softmax(teacher_batch, dim=-1)
            candidate_logp = F.log_softmax(candidate, dim=-1)
            kl = (teacher_logp.exp() * (teacher_logp - candidate_logp)).sum(dim=-1)
            kl_sum += float(kl.double().sum())
            token_count += kl.numel()
            row_offset += len(batch)
    guard.assert_contract(require_allowed_calls=True)
    d32, d64 = torch.cat(drift32), torch.cat(drift64)
    return {
        "row_ce_float32_max_abs_drift": float(d32.max()),
        "row_ce_float32_mean_abs_drift": float(d32.mean()),
        "row_ce_float64_max_abs_drift": float(d64.max()),
        "row_ce_float64_mean_abs_drift": float(d64.mean()),
        "logit_max_abs_drift": logit_max,
        "logit_mean_abs_drift": logit_mean_numerator / logit_count,
        "teacher_kl": kl_sum / token_count,
        "poison_calls": dict(guard.counts),
    }


def run(before: Mapping[str, str | None]) -> None:
    source_commit, source_hashes = verify_sources()
    old_receipt, _ = old_rows.validate_receipt()
    _, loaded = fresh_rows.load_roles_and_validate(("compiler_validation",))
    rows = loaded["compiler_validation"][:, :257].contiguous()
    code_rows, _ = code_oracle.load_frozen_corpus()
    frozen.validate_frozen_ship_pair(old_receipt)
    bases_payload, _ = v3.validate_basis_pair()
    manifest = {
        "schema_version": 1,
        "status": "running_full_native_numeric_diagnostic_v1",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "scope": "validation-only numeric integrity diagnosis; no fit, selection, retry, support, threshold, or recovery credit",
        "pinned_failure_sha256": FAILURE_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
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

    def callback(twall: dict, all_attention: frozenset[int], _: float) -> None:
        realization, _ = frozen.restore_ship_realization(
            sa, twall, all_attention, old_receipt, code_rows
        )
        if realization != site0.SHIP_HASH:
            raise RuntimeError("frozen ship realization changed")
        basis = bases_payload["sites"][0]["basis"].to(sa.DEV).float()
        hook = VariantHook({
            0: basis,
            1: bases_payload["sites"][1]["basis"].to(sa.DEV).float(),
        }, {})
        prior_hook = sa.add_oracle_correction
        sa.add_oracle_correction = hook
        try:
            capture, capture_calls = site0.capture_site0_validation(
                sa, hook, rows, twall, all_attention
            )
            teacher, denominator, _, _, teacher_diagnostics = site0.teacher_bank(
                sa, hook, rows, twall, all_attention
            )
            metrics = {}
            variants = {
                name: {
                    key: value.to(sa.DEV) if torch.is_tensor(value) else value
                    for key, value in state.items()
                }
                for name, state in variant_states(sa.H[0], basis).items()
            }
            metrics["OON_replay_same_device_scorer"] = row_ce_metrics(
                sa, hook, None, rows, teacher, twall, all_attention
            )
            print("full-native numeric diagnostic OON replay", flush=True)
            for name, state in variants.items():
                metrics[name] = {
                    **physical_metrics(state, capture, basis, sa.DEV),
                    **row_ce_metrics(
                        sa, hook, state, rows, teacher, twall, all_attention
                    ),
                }
                print(f"full-native numeric diagnostic {name}", flush=True)
            result = {
                "schema_version": 1,
                "status": "completed_full_native_numeric_diagnostic_v1",
                "authorized_for_scored_experiments": False,
                "authorized_for_training": False,
                "scope": manifest["scope"],
                "pinned_failure_sha256": FAILURE_SHA256,
                "protocol_sha256": PROTOCOL_SHA256,
                "frozen_row_ce_tolerance": 2e-6,
                "teacher_kl_denominator_OON_vs_NON": denominator,
                "capture_calls": capture_calls,
                "teacher_diagnostics": teacher_diagnostics,
                "variants": metrics,
                "source_commit": source_commit,
                "source_hashes": source_hashes,
                "runtime_s": round(time.time() - started, 1),
            }
            write_json_atomic(result, RESULT)
            canonical = metrics["canonical_q64cast_eval32"]
            replay = metrics["OON_replay_same_device_scorer"]
            scorer_retry_licensed = (
                replay["row_ce_float32_max_abs_drift"] <= 2e-6
                and canonical["row_ce_float32_max_abs_drift"] <= 2e-6
            )
            receipt = {
                "schema_version": 1,
                "status": "frozen_full_native_numeric_diagnostic_v1",
                "authority": "compiler_v2_integrity_diagnostic_only",
                "authorized_for_scored_experiments": False,
                "authorized_for_training": False,
                "parent_failure_manifest_sha256": FAILURE_SHA256,
                "protocol_sha256": PROTOCOL_SHA256,
                "result_path": str(RESULT.resolve()),
                "result_sha256": file_sha256(RESULT),
                "scorer_only_retry_licensed": scorer_retry_licensed,
                "representation_retry_licensed": False,
                "frozen_row_ce_tolerance": 2e-6,
                "source_commit": source_commit,
                "source_hashes": source_hashes,
            }
            write_json_atomic(receipt, RECEIPT)
            after = protected_snapshot()
            if after != dict(before):
                raise RuntimeError("numeric diagnostic changed protected artifacts")
            manifest.update({
                "status": "completed_full_native_numeric_diagnostic_v1",
                "result_sha256": file_sha256(RESULT),
                "receipt_sha256": file_sha256(RECEIPT),
                "protected_after": after,
            })
            write_json_atomic(manifest, MANIFEST)
        finally:
            hook.clear()
            sa.add_oracle_correction = prior_hook
            exact_runner.require_inert_correction_state(sa)

    sa.run_oracle_content_screen = callback
    sa.main(oracle_content_screen=True)


def main() -> None:
    existing = [str(path) for path in OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite numeric diagnostic outputs: {existing}")
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"numeric diagnostic already claimed: {LOCK}") from error
    before = protected_snapshot()
    try:
        run(before)
    except BaseException as error:
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        manifest.update({
            "status": "failed_full_native_numeric_diagnostic_v1",
            "authorized_for_scored_experiments": False,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "protected_after": protected_snapshot(),
        })
        write_json_atomic(manifest, MANIFEST)
        raise
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    main()
