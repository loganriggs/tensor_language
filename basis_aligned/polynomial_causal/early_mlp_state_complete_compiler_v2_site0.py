#!/usr/bin/env python3
"""Capture, fit, validate, and freeze the compiler-v2 site-0 program bank."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
PREREG = HERE / "early_mlp_state_complete_compiler_v2_preregistration.json"
SOLVER_PROTOCOL = HERE / "early_mlp_state_complete_compiler_v2_solver_protocol.json"
INTERPRETATION_RECEIPT = (
    HERE / "early_mlp_state_complete_compiler_v2_interpretation_receipt.json"
)
SOLVER_CORRECTION_RECEIPT = (
    HERE / "early_mlp_state_complete_compiler_v2_solver_correction_receipt.json"
)
ROWS_RECEIPT = BQ / "early_mlp_state_complete_compiler_v2_rows_receipt.json"
PREFLIGHT = BQ / "early_mlp_state_complete_compiler_v2_preflight_r2.json"
PREFLIGHT_MANIFEST = BQ / "early_mlp_state_complete_compiler_v2_preflight_r2_manifest.json"
V3_AUTHORITY = BQ / "joint_early_mlp_pca_composition_authoritative_v3_authority.json"
V3_RESULT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_results.json"
V3_MANIFEST = BQ / "joint_early_mlp_pca_composition_authoritative_v3_manifest.json"
V3_BASIS = BQ / "joint_early_mlp_pca_composition_authoritative_v3_bases.pt"
V3_BASIS_RECEIPT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_basis_receipt.json"
OLD_ROWS_RECEIPT = BQ / ".rowcache/fineweb_oracle_v2_receipt.json"
FROZEN_STATE = Path("/workspace/runs/bilin18_frozen_ship_v2.pt")
FROZEN_MANIFEST = Path("/workspace/runs/bilin18_frozen_ship_v2_manifest.json")
PINS = {
    PREREG: "45b0a6c055779449bf5fee815a0ecc7471336e95963db67e74166a2270978d54",
    SOLVER_PROTOCOL: "9b01d94f44a55ee5306ea48d912ab8c8815cb2522fdee082eeca56c1c29a3103",
    INTERPRETATION_RECEIPT: (
        "7c3d98acfa5f127fd20926acb4c511451183f9fd360e67d17d7675a2ebbb7e25"
    ),
    SOLVER_CORRECTION_RECEIPT: (
        "546ac9ca72eba86747a1ca43fdcab5ceb9834d7e0916e4e13fcbccca824cfee1"
    ),
    ROWS_RECEIPT: "23319ece1d8542d51e024bde0e2253d740b08ad18ad4f2d8565ba5120473fd82",
    PREFLIGHT: "f73cf247fa91d37d48a20a880bb46e16afe8149b9179971b06c8b5354e8eefc9",
    PREFLIGHT_MANIFEST: "8f78472c0bbcaf4c8b4d89e22a900264073ee0c11eca81887fe777473169c102",
    V3_AUTHORITY: "38cf5a349e4426b4ed3227ad11f37499ffa9c3959de5d85d606580aa32c39f1e",
    V3_RESULT: "c3408feb031165b747346107841e2e82066aa80ea223ecde845f085d30006587",
    V3_MANIFEST: "cae4a3092213d3e1d0983448576f9a4c3966a092d731d56ab16169a8c82b7588",
    V3_BASIS: "0eee01f39087548a479486d068404f78c4bdc2fd930932add162212da31fe4d9",
    V3_BASIS_RECEIPT: "b81adb4c78255613997de4cbfc8ffd9e8eec233b40950915a14005ba3efcba0f",
    OLD_ROWS_RECEIPT: "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16",
    FROZEN_STATE: "fe21ead35b1dcb3c0914a36b04d7be36e9c3f179c57bc63eee62bd78d34fe9df",
    FROZEN_MANIFEST: "21c89c4d1bd03e1c4be34023781c027b13d2c98202b855938488e33c99e9ba04",
}
ARTIFACT = BQ / "early_mlp_state_complete_compiler_v2_site0_programs.pt"
RECEIPT = BQ / "early_mlp_state_complete_compiler_v2_site0_receipt.json"
RESULT = BQ / "early_mlp_state_complete_compiler_v2_site0_results.json"
MANIFEST = BQ / "early_mlp_state_complete_compiler_v2_site0_manifest.json"
LOCK = Path("/workspace/runs/.early_mlp_state_complete_compiler_v2_site0.lock")
OUTPUTS = (ARTIFACT, RECEIPT, RESULT, MANIFEST)
SHIP_HASH = "21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e"
FIT_SEED = 271828

sys.path.insert(0, str(HERE))
import code_ood_oracle as code_oracle  # noqa: E402
import early_mlp_affine_compiler_v1 as affine_v1  # noqa: E402
import early_mlp_state_complete_compiler_v2 as compiler  # noqa: E402
import early_mlp_state_complete_compiler_v2_preflight as preflight  # noqa: E402
import frozen_ship_oracle_v2 as frozen  # noqa: E402
import joint_early_mlp_oracle_factorial_authoritative as exact_runner  # noqa: E402
import joint_early_mlp_pca_composition_authoritative_v3 as v3  # noqa: E402
import prepare_fineweb_oracle_rows as old_rows  # noqa: E402
import prepare_state_complete_compiler_rows_v2 as fresh_rows  # noqa: E402
import state_complete_compiler_fit_v2 as fit  # noqa: E402
import state_complete_compiler_runtime_v2 as runtime  # noqa: E402
import state_complete_compiler_selection_v2 as selection  # noqa: E402
import state_complete_compiler_solver_v2 as native_solver  # noqa: E402


SOURCE_CLOSURE = (
    Path(__file__),
    HERE / "test_early_mlp_state_complete_compiler_v2_site0.py",
    INTERPRETATION_RECEIPT,
    SOLVER_CORRECTION_RECEIPT,
    HERE / "early_mlp_state_complete_compiler_v2.py",
    HERE / "test_early_mlp_state_complete_compiler_v2.py",
    HERE / "state_complete_compiler_runtime_v2.py",
    HERE / "test_state_complete_compiler_runtime_v2.py",
    HERE / "state_complete_compiler_solver_v2.py",
    HERE / "test_state_complete_compiler_solver_v2.py",
    HERE / "state_complete_compiler_fit_v2.py",
    HERE / "test_state_complete_compiler_fit_v2.py",
    HERE / "state_complete_compiler_selection_v2.py",
    HERE / "test_state_complete_compiler_selection_v2.py",
    HERE / "prepare_state_complete_compiler_rows_v2.py",
    HERE / "test_prepare_state_complete_compiler_rows_v2.py",
    HERE / "early_mlp_affine_compiler_v1.py",
    HERE / "test_early_mlp_affine_compiler_v1.py",
    *preflight.SOURCE_CLOSURE,
    *v3.SOURCE_CLOSURE,
    *exact_runner.SOURCE_CLOSURE,
)
PROTECTED = tuple(dict.fromkeys((*PINS, *exact_runner.PROTECTED_EXISTING)))


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


def verify_pins_and_sources() -> dict[str, str]:
    for path, expected in PINS.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"pinned compiler-v2 site0 input changed: {path}")
    preflight_result = json.loads(PREFLIGHT.read_text())
    if preflight_result.get("status") != "passed_prelabel_preflight":
        raise RuntimeError("compiler-v2 native preflight is not a pass")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    origin = subprocess.run(["git", "rev-parse", "origin/main"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if head != origin:
        raise RuntimeError("compiler-v2 site0 requires HEAD==origin/main")
    hashes = {}
    for path in dict.fromkeys(SOURCE_CLOSURE):
        relative = path.resolve().relative_to(ROOT.resolve())
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)], cwd=ROOT,
            capture_output=True, text=True,
        )
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(relative)],
                               cwd=ROOT)
        if tracked.returncode != 0 or dirty.returncode != 0:
            raise RuntimeError(f"behavior source is not committed and clean: {relative}")
        hashes[str(relative)] = file_sha256(path)
    return hashes


class FrozenParameters:
    def __init__(self, module: Any) -> None:
        self.module = module
        self.flags: list[bool] = []

    def __enter__(self) -> "FrozenParameters":
        self.flags = [parameter.requires_grad for parameter in self.module.parameters()]
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for parameter, flag in zip(self.module.parameters(), self.flags, strict=True):
            parameter.requires_grad_(flag)


def _valid_mask(targets: torch.Tensor) -> torch.Tensor:
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[:, :64] = False
    return valid


def _copy_mask(idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    copy = torch.zeros_like(targets, dtype=torch.bool)
    for lag in range(64):
        past = torch.roll(idx, lag, dims=1)
        if lag:
            past[:, :lag] = -1
        copy |= past == targets
    return copy & _valid_mask(targets)


def capture_site0_fit(
    sa: Any, hook: runtime.StateCompleteCorrectionHook, rows: torch.Tensor,
    twall: Mapping[int, Any], all_attention: frozenset[int],
) -> tuple[dict[str, torch.Tensor], dict[int, int]]:
    """OON teacher capture with a detached site-0 coefficient leaf."""

    hook.configure({1: "O"}, capture_site=0, capture_adjoint=True)
    with FrozenParameters(sa.m), runtime.OriginalMLPCallGuard(sa.H, {0, 1}) as guard:
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8].to(sa.DEV)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            logits = sa.fwd_arm(idx, all_attention, twall, frozenset(range(18))).float()
            ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
            ).view_as(targets)
            loss = ce[_valid_mask(targets)].mean()
            loss.backward()
            hook.collect_pending_adjoint()
    guard.assert_contract(require_allowed_calls=True)
    captured = hook.captured()
    expected = len(rows) * 64
    expected_shapes = {"z": (expected, compiler.D_MODEL),
                       "p": (expected, compiler.COEFFICIENT_DIM),
                       "mo": (expected, compiler.COEFFICIENT_DIM),
                       "c": (expected, compiler.COEFFICIENT_DIM),
                       "adjoint": (expected, compiler.COEFFICIENT_DIM)}
    if {key: tuple(value.shape) for key, value in captured.items()} != expected_shapes:
        raise RuntimeError("site0 fit capture shape changed")
    return captured, dict(guard.counts)


@torch.no_grad()
def capture_site0_validation(
    sa: Any, hook: runtime.StateCompleteCorrectionHook, rows: torch.Tensor,
    twall: Mapping[int, Any], all_attention: frozenset[int],
) -> tuple[dict[str, torch.Tensor], dict[int, int]]:
    hook.configure({}, capture_site=0)
    with runtime.OriginalMLPCallGuard(sa.H, {0}) as guard:
        for start in range(0, len(rows), 8):
            idx = rows[start:start + 8, :-1].to(sa.DEV).contiguous()
            sa.fwd_arm(idx, all_attention, twall, frozenset(range(18)))
    guard.assert_contract(require_allowed_calls=True)
    return hook.captured(), dict(guard.counts)


def _cpu_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: (value.detach().cpu().float().contiguous() if torch.is_tensor(value)
                  and key != "indices" else value.detach().cpu().long().contiguous()
                  if torch.is_tensor(value) else value)
            for key, value in state.items()}


def _device_state(state: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: (value.to(device).float() if torch.is_tensor(value) and key != "indices"
                  else value.to(device).long() if torch.is_tensor(value) else value)
            for key, value in state.items()}


def build_site0_candidates(
    captured: Mapping[str, torch.Tensor], block: Any, basis: torch.Tensor, device: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    clipped, threshold = fit.clip_fit_adjoints(captured["adjoint"])
    candidates: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {"adjoint_clip_threshold": threshold}
    a_states = fit.euclidean_affine_states(
        captured["z"], captured["c"], interface="z_only_c",
        family="A_v1_like_z_only_affine_euclidean",
    )
    b_states = fit.euclidean_affine_states(
        captured["z"], captured["p"], interface="state_complete_p",
        family="B_state_complete_affine_euclidean",
    )
    c_states, diagnostics["causal_affine"] = fit.causal_affine_states(
        captured["z"], captured["p"], clipped, device=device,
    )
    for prefix, states in (("A", a_states), ("B", b_states), ("C", c_states)):
        for (ridge, rank), state in states.items():
            name = f"{prefix}_l{affine_v1.LAMBDA_GRID.index(ridge)}_r{rank}"
            candidates[name] = _cpu_state(state)

    left, right, down, bias = preflight._native_tensors(block)
    q = down.detach().cpu().float().T @ basis.detach().cpu().float()
    phi = fit.native_features(
        captured["z"], left.detach().cpu(), right.detach().cpu(), device=device
    )
    d_states, diagnostics["native_euclidean"] = fit.native_states(
        phi, captured["p"], left.detach().cpu(), right.detach().cpu(), q,
        adjoint=None, family="D_state_complete_native_euclidean", device=device,
    )
    e_states, diagnostics["native_causal"] = fit.native_states(
        phi, captured["p"], left.detach().cpu(), right.detach().cpu(), q,
        adjoint=clipped, family="E_state_complete_native_causal", device=device,
    )
    for prefix, states in (("D", d_states), ("E", e_states)):
        for k, state in states.items():
            candidates[f"{prefix}_k{k}"] = _cpu_state(state)
    diagnostics["candidate_count"] = len(candidates)
    diagnostics["family_counts"] = {
        family: sum(state["family"] == family for state in candidates.values())
        for family in selection.ALL_FAMILIES
    }
    return candidates, diagnostics


def shuffled_fit_capture(
    captured: Mapping[str, torch.Tensor], permutation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Permute only fit p-labels; keep z, live mo, and adjoints in place."""

    permutation = permutation.long()
    if tuple(permutation.shape) != (captured["p"].shape[0],):
        raise ValueError("shuffle permutation does not align with fit captures")
    shuffled_p = captured["p"].index_select(0, permutation)
    return {
        "z": captured["z"],
        "p": shuffled_p,
        "mo": captured["mo"],
        "c": shuffled_p - captured["mo"],
        "adjoint": captured["adjoint"],
    }


def full_native_state(block: Any, basis: torch.Tensor) -> dict[str, Any]:
    left, right, down, bias = preflight._native_tensors(block)
    state = compiler.project_native_weights(
        left.detach().cpu(), right.detach().cpu(), down.detach().cpu(),
        bias.detach().cpu(), basis.detach().cpu(),
    )
    state.update({
        "grammar": "native", "interface": "state_complete_p",
        "family": "full_native_ceiling_control", "k": compiler.NATIVE_PRODUCTS,
    })
    return _cpu_state(state)


@torch.no_grad()
def teacher_bank(
    sa: Any, hook: runtime.StateCompleteCorrectionHook, rows: torch.Tensor,
    twall: Mapping[int, Any], all_attention: frozenset[int],
) -> tuple[torch.Tensor, float, float, torch.Tensor, dict[str, Any]]:
    """Cache OON valid-position logits and score the NON teacher-KL denominator."""

    teacher_parts = []
    non_parts = []
    targets_parts = []
    idx_parts = []
    counters = {}
    for name, states, allowed, sink in (
        ("OON", {0: "O", 1: "O"}, {0, 1}, teacher_parts),
        ("NON", {1: "O"}, {1}, non_parts),
    ):
        hook.configure(states)
        with runtime.OriginalMLPCallGuard(sa.H, allowed) as guard:
            for start in range(0, len(rows), 8):
                batch = rows[start:start + 8].to(sa.DEV)
                idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
                logits = sa.fwd_arm(idx, all_attention, twall, frozenset(range(18))).float()
                sink.append(logits[:, 64:].detach().cpu().contiguous())
                if name == "OON":
                    targets_parts.append(targets[:, 64:].cpu())
                    idx_parts.append(idx.cpu())
        guard.assert_contract(require_allowed_calls=True)
        counters[name] = dict(guard.counts)
    teacher = torch.cat(teacher_parts)
    non = torch.cat(non_parts)
    targets = torch.cat(targets_parts)
    idx_all = torch.cat(idx_parts)
    valid = torch.ones(teacher.shape[:2], dtype=torch.bool)
    denominator = selection.token_weighted_teacher_kl(teacher, non, valid)
    non_ce = F.cross_entropy(non.reshape(-1, non.shape[-1]), targets.reshape(-1),
                             reduction="none").view_as(targets)
    copy = _copy_mask(idx_all, torch.cat([torch.zeros(
        len(idx_all), 64, dtype=targets.dtype), targets
    ], dim=1))[:, 64:]
    teacher_ce = F.cross_entropy(
        teacher.reshape(-1, teacher.shape[-1]), targets.reshape(-1), reduction="none"
    ).view_as(targets)
    teacher_row_ce = teacher_ce.mean(dim=1)
    copy_ce = float(non_ce[copy].mean()) if bool(copy.any()) else math.nan
    return teacher, denominator, copy_ce, teacher_row_ce, {
        "NON_global_ce": float(non_ce.mean()), "calls": counters,
        "validation_rows": int(len(rows)),
        "validation_tokens": int(targets.numel()),
    }


@torch.no_grad()
def full_native_validation_gate(
    sa: Any, hook: runtime.StateCompleteCorrectionHook, rows: torch.Tensor,
    twall: Mapping[int, Any], all_attention: frozenset[int],
    state: Mapping[str, Any], validation_capture: Mapping[str, torch.Tensor],
    teacher: torch.Tensor, teacher_row_ce: torch.Tensor, basis: torch.Tensor,
) -> dict[str, Any]:
    """Fail closed on live physical and poison-gated validation-CE identity."""

    device_state = _device_state(state, sa.DEV)
    predicted_parts = []
    for start in range(0, validation_capture["z"].shape[0], 1024):
        z = validation_capture["z"][start:start + 1024].to(sa.DEV)
        predicted_parts.append(
            runtime.runtime_projected_output(z, device_state).detach().cpu()
        )
    predicted = torch.cat(predicted_parts)
    coefficient_error = predicted - validation_capture["p"]
    physical_parts = []
    basis_device = basis.to(sa.DEV).float()
    for start in range(0, coefficient_error.shape[0], 1024):
        physical_parts.append(
            (coefficient_error[start:start + 1024].to(sa.DEV) @ basis_device.T)
            .detach().cpu()
        )
    physical_error = torch.cat(physical_parts)
    physical_max = float(physical_error.abs().max())
    physical_scale = max(1.0, float((validation_capture["c"] @ basis.cpu().float().T)
                                    .abs().max()))
    physical_tolerance = 4e-6 * physical_scale
    if physical_max > physical_tolerance:
        raise RuntimeError(
            f"full-native live physical gate failed: {physical_max}>{physical_tolerance}"
        )

    hook.programs = {"full_native": {0: device_state}}
    hook.configure({0: "Q", 1: "O"}, program_name="full_native")
    row_ce_parts = []
    kl_sum = 0.0
    token_count = 0
    with runtime.OriginalMLPCallGuard(sa.H, {1}) as guard:
        row_offset = 0
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8].to(sa.DEV)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            logits = sa.fwd_arm(idx, all_attention, twall, frozenset(range(18))).float()
            candidate = logits[:, 64:]
            teacher_batch = teacher[row_offset:row_offset + len(batch)].to(sa.DEV)
            teacher_logp = F.log_softmax(teacher_batch, dim=-1)
            candidate_logp = F.log_softmax(candidate, dim=-1)
            kl = (teacher_logp.exp() * (teacher_logp - candidate_logp)).sum(dim=-1)
            kl_sum += float(kl.double().sum())
            token_count += kl.numel()
            ce = F.cross_entropy(
                candidate.reshape(-1, candidate.shape[-1]),
                targets[:, 64:].reshape(-1), reduction="none",
            ).view(len(batch), -1)
            row_ce_parts.append(ce.double().mean(dim=1).cpu())
            row_offset += len(batch)
    guard.assert_contract(require_allowed_calls=True)
    row_ce = torch.cat(row_ce_parts)
    row_ce_drift = (row_ce - teacher_row_ce.double()).abs()
    max_row_ce_drift = float(row_ce_drift.max())
    if max_row_ce_drift > 2e-6:
        raise RuntimeError(
            f"full-native live row-CE gate failed: {max_row_ce_drift}>2e-6"
        )
    return {
        "passed": True,
        "physical_max_abs_error": physical_max,
        "physical_tolerance": physical_tolerance,
        "validation_row_ce_max_abs_drift": max_row_ce_drift,
        "validation_row_ce_mean_abs_drift": float(row_ce_drift.mean()),
        "validation_teacher_kl": kl_sum / token_count,
        "poison_calls": dict(guard.counts),
    }


@torch.no_grad()
def score_candidate(
    sa: Any, hook: runtime.StateCompleteCorrectionHook, rows: torch.Tensor,
    twall: Mapping[int, Any], all_attention: frozenset[int],
    name: str, state: Mapping[str, Any], teacher: torch.Tensor,
    denominator: float, baseline_copy_ce: float,
) -> tuple[dict[str, Any], dict[int, int]]:
    hook.programs = {name: {0: _device_state(state, sa.DEV)}}
    hook.configure({0: "Q", 1: "O"}, program_name=name)
    kl_sum = 0.0
    token_count = 0
    ce_sum = 0.0
    copy_sum = 0.0
    copy_count = 0
    with runtime.OriginalMLPCallGuard(sa.H, {1}) as guard:
        row_offset = 0
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8].to(sa.DEV)
            idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            logits = sa.fwd_arm(idx, all_attention, twall, frozenset(range(18))).float()
            candidate = logits[:, 64:]
            teacher_batch = teacher[row_offset:row_offset + len(batch)].to(sa.DEV)
            teacher_logp = F.log_softmax(teacher_batch, dim=-1)
            candidate_logp = F.log_softmax(candidate, dim=-1)
            kl = (teacher_logp.exp() * (teacher_logp - candidate_logp)).sum(dim=-1)
            kl_sum += float(kl.double().sum())
            token_count += kl.numel()
            target = targets[:, 64:]
            ce = F.cross_entropy(candidate.reshape(-1, candidate.shape[-1]),
                                 target.reshape(-1), reduction="none").view_as(target)
            ce_sum += float(ce.double().sum())
            copy = _copy_mask(idx, targets)[:, 64:]
            if bool(copy.any()):
                copy_sum += float(ce[copy].double().sum())
                copy_count += int(copy.sum())
            row_offset += len(batch)
    guard.assert_contract(require_allowed_calls=True)
    candidate_kl = kl_sum / token_count
    metrics = selection.direct_recovery(candidate_kl, denominator)
    metrics.update({
        "global_ce": ce_sum / token_count,
        "copy_ce": copy_sum / max(copy_count, 1),
        "copy_count": copy_count,
        "copy_worsening": copy_sum / max(copy_count, 1) - baseline_copy_ce,
        "price": selection.state_price(state),
    })
    return metrics, dict(guard.counts)


def artifact_payload(
    candidates: Mapping[str, Mapping[str, Any]], selection_receipt: Mapping[str, Any],
    controls: Mapping[str, Any], diagnostics: Mapping[str, Any],
    source_commit: str, source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    retained = {name: _cpu_state(state) for name, state in candidates.items()}
    return {
        "schema_version": 1,
        "status": "frozen_before_any_site1_capture",
        "authority": "compiler_v2_sequential_site0_validation_freeze",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": True,
        "training_license_sites": [1],
        "preregistration_sha256": PINS[PREREG],
        "solver_protocol_sha256": PINS[SOLVER_PROTOCOL],
        "interpretation_receipt_sha256": PINS[INTERPRETATION_RECEIPT],
        "solver_correction_receipt_sha256": PINS[SOLVER_CORRECTION_RECEIPT],
        "rows_receipt_sha256": PINS[ROWS_RECEIPT],
        "preflight_sha256": PINS[PREFLIGHT],
        "candidates": retained,
        "selection": dict(selection_receipt),
        "controls": dict(controls),
        "diagnostics": dict(diagnostics),
        "source_commit": source_commit,
        "source_hashes": dict(source_hashes),
        "forbidden_artifact_contents": [
            "tokens", "row indices", "labels", "cached z/p/c/mo/adjoints",
            "native phi features", "original checkpoint pointers", "validation logits",
        ],
    }


def validate_artifact() -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
    receipt = json.loads(RECEIPT.read_text())
    if payload.get("status") != "frozen_before_any_site1_capture":
        raise RuntimeError("site0 artifact is not frozen")
    if receipt.get("artifact_sha256") != file_sha256(ARTIFACT):
        raise RuntimeError("site0 artifact receipt hash changed")
    if receipt.get("authorized_for_training") is not True or receipt.get(
        "training_license_sites"
    ) != [1]:
        raise RuntimeError("site0 artifact training scope changed")
    selected = payload.get("selection", {}).get("selected")
    if selected not in payload.get("candidates", {}):
        raise RuntimeError("site0 selected program is absent")
    controls = payload.get("controls", {})
    if set(controls) != {"mean", "shuffle", "full_native"}:
        raise RuntimeError("site0 registered controls are incomplete")
    if controls["shuffle"].get("state", {}).get("family") not in selection.ALL_FAMILIES:
        raise RuntimeError("site0 shuffle control lacks a selected A-E program")
    if controls["mean"].get("grammar") != "constant":
        raise RuntimeError("site0 mean control is invalid")
    if controls["full_native"].get("k") != compiler.NATIVE_PRODUCTS:
        raise RuntimeError("site0 full-native control is invalid")
    return payload, receipt


def run_claimed(before: Mapping[str, str | None]) -> None:
    source_hashes = verify_pins_and_sources()
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
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                   capture_output=True, text=True).stdout.strip()
    manifest = {
        "schema_version": 1, "status": "running_compiler_v2_site0",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "preregistration_sha256": PINS[PREREG],
        "solver_protocol_sha256": PINS[SOLVER_PROTOCOL],
        "interpretation_receipt_sha256": PINS[INTERPRETATION_RECEIPT],
        "solver_correction_receipt_sha256": PINS[SOLVER_CORRECTION_RECEIPT],
        "rows_receipt_sha256": PINS[ROWS_RECEIPT],
        "source_commit": source_commit, "source_hashes": source_hashes,
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
        if realization != SHIP_HASH:
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
            captured, fit_calls = capture_site0_fit(
                sa, hook, rows["compiler_fit"], twall, all_attention
            )
            capture_hashes = {key: tensor_sha256(value) for key, value in captured.items()}
            candidates, fit_diagnostics = build_site0_candidates(
                captured, sa.H[0], basis, sa.DEV
            )
            fit_permutation = fit.expand_capture_permutation(
                fit.document_block_permutation(document_ids, FIT_SEED)
            )
            shuffled_capture = shuffled_fit_capture(captured, fit_permutation)
            shuffled_candidates, shuffled_fit_diagnostics = build_site0_candidates(
                shuffled_capture, sa.H[0], basis, sa.DEV
            )
            mean_control = _cpu_state(fit.constant_state(captured["p"]))
            full_native_control = full_native_state(sa.H[0], basis)
            validation_capture, validation_calls = capture_site0_validation(
                sa, hook, rows["compiler_validation"], twall, all_attention
            )
            (teacher, denominator, baseline_copy, teacher_row_ce,
             teacher_diagnostics) = teacher_bank(
                sa, hook, rows["compiler_validation"], twall, all_attention
            )
            full_native_gate = full_native_validation_gate(
                sa, hook, rows["compiler_validation"], twall, all_attention,
                full_native_control, validation_capture, teacher, teacher_row_ce, basis,
            )
            bank = {}
            call_counters = {"fit": fit_calls, "validation_capture": validation_calls,
                             **teacher_diagnostics.pop("calls")}
            for index, (name, state) in enumerate(sorted(candidates.items())):
                metrics, calls = score_candidate(
                    sa, hook, rows["compiler_validation"], twall, all_attention,
                    name, state, teacher, denominator, baseline_copy,
                )
                bank[name] = {"state": state, "metrics": metrics}
                call_counters[name] = calls
                print(f"compiler-v2 site0 validation {index + 1}/{len(candidates)} {name}",
                      flush=True)
            frozen_selection = selection.freeze_validation_selection(bank)
            shuffle_bank = {}
            for index, (name, state) in enumerate(sorted(shuffled_candidates.items())):
                metrics, calls = score_candidate(
                    sa, hook, rows["compiler_validation"], twall, all_attention,
                    name, state, teacher, denominator, baseline_copy,
                )
                shuffle_bank[name] = {"state": state, "metrics": metrics}
                call_counters[f"shuffle_{name}"] = calls
                print(
                    f"compiler-v2 site0 shuffle validation "
                    f"{index + 1}/{len(shuffled_candidates)} {name}", flush=True,
                )
            frozen_shuffle_selection = selection.freeze_control_selection(shuffle_bank)
            selected_shuffle = frozen_shuffle_selection["selected"]
            controls = {
                "mean": mean_control,
                "shuffle": {
                    "state": _cpu_state(shuffled_candidates[selected_shuffle]),
                    "selection": frozen_shuffle_selection,
                },
                "full_native": full_native_control,
            }
            diagnostics = {
                "fit_capture_tensor_sha256": capture_hashes,
                "fit_capture_shapes": {key: list(value.shape) for key, value in captured.items()},
                "fit_adjoint_clip_threshold": fit_diagnostics["adjoint_clip_threshold"],
                "fit_document_permutation_sha256": tensor_sha256(fit_permutation),
                "fit_document_permutation_moved": int((fit_permutation != torch.arange(
                    len(fit_permutation))).sum()),
                "fit": fit_diagnostics,
                "shuffle_fit": shuffled_fit_diagnostics,
                "validation_capture_tensor_sha256": {
                    key: tensor_sha256(value) for key, value in validation_capture.items()
                },
                "teacher_kl_denominator_OON_vs_NON": denominator,
                "teacher": teacher_diagnostics,
                "full_native_validation_gate": full_native_gate,
                "call_counters": call_counters,
            }
            payload = artifact_payload(
                candidates, frozen_selection, controls, diagnostics,
                source_commit, source_hashes
            )
            write_torch_atomic(payload, ARTIFACT)
            receipt = {
                "schema_version": 1,
                "status": "frozen_before_any_site1_capture",
                "authority": "compiler_v2_sequential_site0_validation_freeze",
                "authorized_for_scored_experiments": False,
                "authorized_for_training": True,
                "training_license_sites": [1],
                "preregistration_sha256": PINS[PREREG],
                "solver_protocol_sha256": PINS[SOLVER_PROTOCOL],
                "interpretation_receipt_sha256": PINS[INTERPRETATION_RECEIPT],
                "solver_correction_receipt_sha256": PINS[SOLVER_CORRECTION_RECEIPT],
                "rows_receipt_sha256": PINS[ROWS_RECEIPT],
                "artifact_path": str(ARTIFACT.resolve()),
                "artifact_sha256": file_sha256(ARTIFACT),
                "artifact_bytes": ARTIFACT.stat().st_size,
                "selected": frozen_selection["selected"],
                "selected_family": frozen_selection["selected_family"],
                "freeze_rule": "Written and validated before any site1 fit-state forward.",
                "source_commit": source_commit, "source_hashes": source_hashes,
            }
            write_json_atomic(receipt, RECEIPT)
            validate_artifact()
            result = {
                "schema_version": 1,
                "status": "completed_site0_validation_freeze",
                "authorized_for_scored_experiments": False,
                "authorized_for_training": False,
                "scope": "fit and validation only; no compiler_final rows, site1 claim, executable recovery, or whole-model credit",
                "preregistration_sha256": PINS[PREREG],
                "solver_protocol_sha256": PINS[SOLVER_PROTOCOL],
                "interpretation_receipt_sha256": PINS[INTERPRETATION_RECEIPT],
                "solver_correction_receipt_sha256": PINS[SOLVER_CORRECTION_RECEIPT],
                "selection": frozen_selection,
                "validation": {name: dict(row["metrics"]) for name, row in bank.items()},
                "shuffle_selection": frozen_shuffle_selection,
                "shuffle_validation": {
                    name: dict(row["metrics"]) for name, row in shuffle_bank.items()
                },
                "full_native_validation_gate": full_native_gate,
                "teacher_kl_denominator_OON_vs_NON": denominator,
                "candidate_count": len(candidates),
                "shuffle_candidate_count": len(shuffled_candidates),
                "artifact_sha256": receipt["artifact_sha256"],
                "source_commit": source_commit, "source_hashes": source_hashes,
                "runtime_s": round(time.time() - started, 1),
            }
            write_json_atomic(result, RESULT)
            component_after = exact_runner.component_tree_sha256(sa, twall, all_attention)
            if component_after != component_before:
                raise RuntimeError("component tree changed during compiler-v2 site0")
            manifest.update({
                "status": "completed_site0_validation_freeze",
                "result_sha256": file_sha256(RESULT),
                "artifact_sha256": receipt["artifact_sha256"],
                "receipt_sha256": file_sha256(RECEIPT),
                "component_tree_unchanged": True,
                "protected_after": protected_snapshot(),
            })
            if manifest["protected_after"] != dict(before):
                raise RuntimeError("compiler-v2 site0 changed protected artifacts")
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
        raise RuntimeError(f"refusing to overwrite compiler-v2 site0 outputs: {existing}")
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"compiler-v2 site0 already claimed: {LOCK}") from error
    before = protected_snapshot()
    try:
        run_claimed(before)
    except BaseException as error:
        failure = {
            "schema_version": 1, "status": "failed_compiler_v2_site0",
            "authorized_for_scored_experiments": False,
            "authorized_for_training": False,
            "failure_type": type(error).__name__, "failure_message": str(error),
            "protected_after": protected_snapshot(),
            "recovery": "Preserve outputs and use a versioned retry namespace.",
        }
        manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        manifest.update(failure)
        write_json_atomic(manifest, MANIFEST)
        raise
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    main()
