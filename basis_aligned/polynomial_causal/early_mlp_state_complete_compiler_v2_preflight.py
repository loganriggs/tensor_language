#!/usr/bin/env python3
"""Fail-closed model/type/full-native preflight for compiler v2.

This phase reads no compiler-fit labels, gradients, validation outcomes, or final
outcomes.  It verifies that the frozen model really has the registered ungated
bilinear algebra and that a standalone float32 K=4608 program reproduces its
projected output within a scale-aware roundoff bound.
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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
PREREG = HERE / "early_mlp_state_complete_compiler_v2_preregistration.json"
ROWS_RECEIPT = BQ / "early_mlp_state_complete_compiler_v2_rows_receipt.json"
V3_AUTHORITY = BQ / "joint_early_mlp_pca_composition_authoritative_v3_authority.json"
V3_RESULT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_results.json"
V3_MANIFEST = BQ / "joint_early_mlp_pca_composition_authoritative_v3_manifest.json"
V3_BASIS = BQ / "joint_early_mlp_pca_composition_authoritative_v3_bases.pt"
V3_BASIS_RECEIPT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_basis_receipt.json"
OLD_ROWS_RECEIPT = BQ / ".rowcache/fineweb_oracle_v2_receipt.json"
FROZEN_STATE = Path("/workspace/runs/bilin18_frozen_ship_v2.pt")
FROZEN_MANIFEST = Path("/workspace/runs/bilin18_frozen_ship_v2_manifest.json")
MODEL_SNAPSHOT = Path(
    "/workspace/.hf_home/hub/"
    "models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/"
    "snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240"
)
PINS = {
    PREREG: "45b0a6c055779449bf5fee815a0ecc7471336e95963db67e74166a2270978d54",
    ROWS_RECEIPT: "23319ece1d8542d51e024bde0e2253d740b08ad18ad4f2d8565ba5120473fd82",
    V3_AUTHORITY: "38cf5a349e4426b4ed3227ad11f37499ffa9c3959de5d85d606580aa32c39f1e",
    V3_RESULT: "c3408feb031165b747346107841e2e82066aa80ea223ecde845f085d30006587",
    V3_MANIFEST: "cae4a3092213d3e1d0983448576f9a4c3966a092d731d56ab16169a8c82b7588",
    V3_BASIS: "0eee01f39087548a479486d068404f78c4bdc2fd930932add162212da31fe4d9",
    V3_BASIS_RECEIPT: "b81adb4c78255613997de4cbfc8ffd9e8eec233b40950915a14005ba3efcba0f",
    OLD_ROWS_RECEIPT: "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16",
    FROZEN_STATE: "fe21ead35b1dcb3c0914a36b04d7be36e9c3f179c57bc63eee62bd78d34fe9df",
    FROZEN_MANIFEST: "21c89c4d1bd03e1c4be34023781c027b13d2c98202b855938488e33c99e9ba04",
    MODEL_SNAPSHOT / "config.json": "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c",
    MODEL_SNAPSHOT / "pytorch_model.bin": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
}
RESULT = BQ / "early_mlp_state_complete_compiler_v2_preflight.json"
MANIFEST = BQ / "early_mlp_state_complete_compiler_v2_preflight_manifest.json"
LOCK = Path("/workspace/runs/.early_mlp_state_complete_compiler_v2_preflight.lock")
OUTPUTS = (RESULT, MANIFEST)
SHIP_HASH = "21ddc9ffdb7703aa570f88c5c7f4fa9fe007a988a1a7a3fd91058ee76a25ab8e"
SERIALIZED_RELATIVE_TOLERANCE = 2e-6
PHYSICAL_RELATIVE_TOLERANCE = 4e-6
DOUBLE_RELATIVE_TOLERANCE = 2e-9

sys.path.insert(0, str(HERE))
import code_ood_oracle as code_oracle  # noqa: E402
import early_mlp_state_complete_compiler_v2 as compiler  # noqa: E402
import frozen_ship_oracle_v2 as frozen  # noqa: E402
import joint_early_mlp_oracle_factorial_authoritative as exact_runner  # noqa: E402
import joint_early_mlp_pca_composition_authoritative_v3 as v3  # noqa: E402
import prepare_fineweb_oracle_rows as old_rows  # noqa: E402
import prepare_state_complete_compiler_rows_v2 as fresh_rows  # noqa: E402
import state_complete_compiler_runtime_v2 as runtime  # noqa: E402


SOURCE_CLOSURE = (
    Path(__file__),
    HERE / "test_early_mlp_state_complete_compiler_v2_preflight.py",
    HERE / "early_mlp_state_complete_compiler_v2.py",
    HERE / "test_early_mlp_state_complete_compiler_v2.py",
    HERE / "state_complete_compiler_runtime_v2.py",
    HERE / "test_state_complete_compiler_runtime_v2.py",
    HERE / "prepare_state_complete_compiler_rows_v2.py",
    HERE / "test_prepare_state_complete_compiler_rows_v2.py",
    *exact_runner.SOURCE_CLOSURE,
)
PROTECTED = tuple(dict.fromkeys((
    *PINS,
    *exact_runner.PROTECTED_EXISTING,
)))


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


def verify_pins_and_sources() -> dict[str, str]:
    for path, expected in PINS.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"pinned compiler-v2 preflight input changed: {path}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    origin = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if head != origin:
        raise RuntimeError(f"compiler-v2 preflight requires HEAD==origin/main: {head}!={origin}")
    hashes = {}
    for path in dict.fromkeys(SOURCE_CLOSURE):
        relative = path.resolve().relative_to(ROOT.resolve())
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)], cwd=ROOT,
            capture_output=True, text=True,
        )
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=ROOT,
        )
        if tracked.returncode != 0 or dirty.returncode != 0:
            raise RuntimeError(f"behavior source is not committed and clean: {relative}")
        hashes[str(relative)] = file_sha256(path)
    return hashes


def scaled_tolerance(value: torch.Tensor, relative: float) -> dict[str, float]:
    if relative <= 0.0:
        raise ValueError("relative tolerance must be positive")
    scale = max(1.0, float(value.detach().float().abs().max()))
    return {"relative_multiplier": float(relative), "scale_max_1": scale,
            "tolerance": relative * scale}


def _native_tensors(block: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mlp = block.mlp
    for name in ("Left", "Right", "Down", "Down_bias"):
        if not hasattr(mlp, name):
            raise RuntimeError(f"MLP lacks registered ungated bilinear field {name}")
    left, right, down, bias = (
        mlp.Left.weight, mlp.Right.weight, mlp.Down.weight, mlp.Down_bias
    )
    if mlp.Left.bias is not None or mlp.Right.bias is not None or mlp.Down.bias is not None:
        raise RuntimeError("registered native algebra requires bias-free L/R/Down maps")
    expected = {
        "left": (compiler.NATIVE_PRODUCTS, compiler.D_MODEL),
        "right": (compiler.NATIVE_PRODUCTS, compiler.D_MODEL),
        "down": (compiler.D_MODEL, compiler.NATIVE_PRODUCTS),
        "bias": (compiler.D_MODEL,),
    }
    observed = {"left": tuple(left.shape), "right": tuple(right.shape),
                "down": tuple(down.shape), "bias": tuple(bias.shape)}
    if observed != expected:
        raise RuntimeError(f"native tensor shapes changed: {observed}")
    return left, right, down, bias


@torch.no_grad()
def preflight_site(block: Any, basis: torch.Tensor, site: int, device: Any) -> dict[str, Any]:
    left, right, down, bias = _native_tensors(block)
    basis_cpu = basis.detach().cpu().float().contiguous()
    state64 = compiler.project_native_weights(
        left.detach().cpu(), right.detach().cpu(), down.detach().cpu(),
        bias.detach().cpu().float(), basis_cpu,
    )
    state32 = {
        key: (value.to(device).float().contiguous() if torch.is_tensor(value) else value)
        for key, value in state64.items()
    }
    state32.update({"grammar": "native", "interface": "state_complete_p"})
    values = torch.linspace(-1.0, 1.0, 6 * compiler.D_MODEL, device=device)
    dtype = left.dtype
    z = values.view(2, 3, compiler.D_MODEL).to(dtype)
    original = block.mlp(z).float()
    projected = original.reshape(-1, compiler.D_MODEL) @ basis.to(device).float()
    predicted = runtime.runtime_projected_output(z, state32)
    serialized_error = float((predicted - projected).abs().max())
    serialized_bound = scaled_tolerance(projected, SERIALIZED_RELATIVE_TOLERANCE)
    if serialized_error > serialized_bound["tolerance"]:
        raise RuntimeError(
            f"MLP{site} serialized full-native projection failed: "
            f"error={serialized_error} bound={serialized_bound}"
        )

    z64 = z.detach().cpu().double().reshape(-1, compiler.D_MODEL)
    direct64 = ((z64 @ left.detach().cpu().double().T)
                * (z64 @ right.detach().cpu().double().T)) @ down.detach().cpu().double().T
    direct64 += bias.detach().cpu().double()
    projected64 = direct64 @ basis_cpu.double()
    predicted64 = compiler.native_projected_output(z64, state64)
    double_error = float((predicted64 - projected64).abs().max())
    double_bound = scaled_tolerance(projected64, DOUBLE_RELATIVE_TOLERANCE)
    if double_error > double_bound["tolerance"]:
        raise RuntimeError(
            f"MLP{site} double full-native algebra failed: "
            f"error={double_error} bound={double_bound}"
        )

    deployed = torch.cos(values).view_as(original).float()
    coefficients = predicted - deployed.reshape(-1, compiler.D_MODEL) @ basis.to(device).float()
    corrected = deployed + (coefficients @ basis.to(device).float().T).view_as(deployed)
    expected = (
        deployed
        - ((deployed.reshape(-1, compiler.D_MODEL) @ basis.to(device).float())
           @ basis.to(device).float().T).view_as(deployed)
        + ((projected @ basis.to(device).float().T).view_as(deployed))
    )
    physical_error = float((corrected - expected).abs().max())
    physical_bound = scaled_tolerance(expected, PHYSICAL_RELATIVE_TOLERANCE)
    if physical_error > physical_bound["tolerance"]:
        raise RuntimeError(
            f"MLP{site} state-complete physical identity failed: "
            f"error={physical_error} bound={physical_bound}"
        )
    return {
        "site": site,
        "type_gate": "ungated D[(Lz)odot(Rz)]+Down_bias",
        "shapes": {"left": list(left.shape), "right": list(right.shape),
                   "down": list(down.shape), "bias": list(bias.shape)},
        "dtypes": {"left": str(left.dtype), "right": str(right.dtype),
                   "down": str(down.dtype), "bias": str(bias.dtype)},
        "serialized_full_native": {"max_abs_error": serialized_error, **serialized_bound},
        "double_algebra": {"max_abs_error": double_error, **double_bound},
        "state_complete_physical": {"max_abs_error": physical_error, **physical_bound},
        "standalone_price": compiler.native_program_price(
            compiler.NATIVE_PRODUCTS, include_basis=True
        ),
        "program_tensor_sha256": {
            key: code_oracle.tensor_sha256(value.detach().cpu().contiguous())
            for key, value in state32.items() if torch.is_tensor(value)
        },
    }


def run_claimed(before: Mapping[str, str | None]) -> None:
    source_hashes = verify_pins_and_sources()
    old_receipt, _ = old_rows.validate_receipt()
    fresh_rows.load_and_validate()
    code_rows, _ = code_oracle.load_frozen_corpus()
    frozen.validate_frozen_ship_pair(old_receipt)
    bases_payload, _ = v3.validate_basis_pair()
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "status": "running_prelabel_preflight",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": False,
        "preregistration_sha256": PINS[PREREG],
        "rows_receipt_sha256": PINS[ROWS_RECEIPT],
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
        if realization != SHIP_HASH:
            raise RuntimeError("frozen ship realization changed")
        exact_runner.require_inert_correction_state(sa)
        component_before = exact_runner.component_tree_sha256(sa, twall, all_attention)
        config = sa.m.config.to_dict()
        type_config = {"gated": config.get("gated"),
                       "squared_mlp": config.get("squared_mlp")}
        if type_config != {"gated": False, "squared_mlp": False}:
            raise RuntimeError(f"model is not the registered ungated bilinear form: {type_config}")
        sites = {
            str(site): preflight_site(
                sa.H[site], bases_payload["sites"][site]["basis"], site, sa.DEV
            )
            for site in (0, 1)
        }
        exact_runner.require_inert_correction_state(sa)
        component_after = exact_runner.component_tree_sha256(sa, twall, all_attention)
        if component_after != component_before:
            raise RuntimeError("component tree changed during compiler-v2 preflight")
        result = {
            "schema_version": 1,
            "status": "passed_prelabel_preflight",
            "authorized_for_scored_experiments": False,
            "authorized_for_training": False,
            "scope": "type/algebra/runtime adequacy only; no compiler labels, gradients, validation, final scoring, or recovery credit",
            "preregistration_sha256": PINS[PREREG],
            "rows_receipt_sha256": PINS[ROWS_RECEIPT],
            "ship_realization_sha256": realization,
            "model_type_config": type_config,
            "sites": sites,
            "component_tree_unchanged": True,
            "source_commit": source_commit,
            "source_hashes": source_hashes,
            "runtime_s": round(time.time() - started, 1),
        }
        write_json_atomic(result, RESULT)
        manifest.update({
            "status": "passed_prelabel_preflight",
            "result_sha256": file_sha256(RESULT),
            "ship_realization_sha256": realization,
            "protected_after": protected_snapshot(),
        })
        if manifest["protected_after"] != dict(before):
            raise RuntimeError("preflight changed protected artifacts")
        write_json_atomic(manifest, MANIFEST)

    sa.run_oracle_content_screen = callback
    sa.main(oracle_content_screen=True)


def main() -> None:
    existing = [str(path) for path in OUTPUTS if path.exists()]
    if existing:
        raise RuntimeError(f"refusing to overwrite compiler-v2 preflight outputs: {existing}")
    try:
        LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"compiler-v2 preflight already claimed: {LOCK}") from error
    before = protected_snapshot()
    try:
        run_claimed(before)
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "failed_prelabel_preflight",
            "authorized_for_scored_experiments": False,
            "authorized_for_training": False,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "protected_after": protected_snapshot(),
            "recovery": "Preserve this namespace and use a versioned retry.",
        }
        if not MANIFEST.exists():
            write_json_atomic(failure, MANIFEST)
        else:
            manifest = json.loads(MANIFEST.read_text())
            manifest.update(failure)
            write_json_atomic(manifest, MANIFEST)
        raise
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    main()
