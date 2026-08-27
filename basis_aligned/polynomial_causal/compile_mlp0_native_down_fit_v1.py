#!/usr/bin/env python3
"""Fit and serialize MLP0 native-Down programs without evaluation-row access.

This process consumes only the historical Stage-0 fit role.  Its output bundles and
all fit-derived choices are frozen before the separate evaluation authority exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch


ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
PC = ROOT / "basis_aligned" / "polynomial_causal"
sys.path.insert(0, str(BQ))
sys.path.insert(0, str(PC))

import mlp0_quotient_worst_cell as stage  # noqa: E402
from mlp0_native_down_program import (  # noqa: E402
    canonical_balanced_factors,
    common_exact_product_price,
    compact_centered_codebook,
    deterministic_centroid_derangement,
    fit_reduced_rank_from_statistics,
    load_program,
    matched_hierarchy_rank,
    program_price_bytes,
    serialize_program,
)


AUTHORITY = BQ / "mlp0_native_down_hierarchy_v1_fit_authority.json"
FIT_RECEIPT = BQ / "mlp0_native_down_hierarchy_v1_fit_receipt.json"
PROGRAMS = BQ / "mlp0_native_down_hierarchy_v1_programs"
FAILURE = BQ / "mlp0_native_down_hierarchy_v1_fit_failure.json"
LOCK = Path("/workspace/runs/.bilin18_mlp0_native_down_fit_v1.lock")
RIDGE_FRACTION = 1e-4
RUNG_RANKS = (256, 512)
BATCH = 8
CAPTURE = {"h": None}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_authority() -> dict:
    if not AUTHORITY.is_file():
        raise RuntimeError("fit authority absent")
    authority = json.loads(AUTHORITY.read_text())
    if (authority.get("status") != "frozen_before_any_native_down_fit_model_forward"
            or authority.get("fit_receipt_path") != str(FIT_RECEIPT)
            or authority.get("program_directory") != str(PROGRAMS)):
        raise RuntimeError("fit authority identity/status mismatch")
    for raw, expected in authority.get("source_hashes", {}).items():
        if file_sha256(Path(raw)) != expected:
            raise RuntimeError(f"authority-bound source changed: {raw}")
    for raw, expected in authority.get("model_files", {}).items():
        if file_sha256(Path(raw)) != expected:
            raise RuntimeError(f"authority-bound model file changed: {raw}")
    if FIT_RECEIPT.exists() or PROGRAMS.exists() or FAILURE.exists():
        raise RuntimeError("fit namespace is already spent")
    return authority


def down_input_hook(module, args) -> None:
    if len(args) != 1:
        raise RuntimeError("unexpected Down input signature")
    CAPTURE["h"] = args[0].detach().float()


@torch.no_grad()
def collect_product_statistics(
    fit_rows: torch.Tensor,
    assignments: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | int]:
    """Two deterministic passes: mean h, then covariance and per-state h sums."""
    if stage.H is None:
        raise RuntimeError("model not loaded")
    n_positions = int(fit_rows.shape[0] * (fit_rows.shape[1] - 1))
    sum_h = torch.zeros(stage.HID if hasattr(stage, "HID") else 4608, dtype=torch.float64)
    for start in range(0, len(fit_rows), BATCH):
        idx = fit_rows[start:start + BATCH, :-1].to(stage.DEV).contiguous()
        CAPTURE["h"] = None
        stage.fwd(idx, "O")
        if CAPTURE["h"] is None:
            raise RuntimeError("Down product-state capture missing")
        sum_h += CAPTURE["h"].reshape(-1, 4608).double().sum(0).cpu()
    mean_h = (sum_h / n_positions).float()

    covariance = torch.zeros(4608, 4608, dtype=torch.float32, device=stage.DEV)
    state_sums = {
        name: torch.zeros(int(values.max()) + 1, 4608, dtype=torch.float64)
        for name, values in assignments.items()
    }
    for start in range(0, len(fit_rows), BATCH):
        idx = fit_rows[start:start + BATCH, :-1].to(stage.DEV).contiguous()
        CAPTURE["h"] = None
        stage.fwd(idx, "O")
        h = CAPTURE["h"].reshape(-1, 4608).float()
        centered = h - mean_h.to(stage.DEV)
        covariance.addmm_(centered.T, centered)
        centered_cpu = centered.double().cpu()
        token_cpu = idx.reshape(-1).cpu()
        for name, values in assignments.items():
            codes = values[token_cpu]
            state_sums[name].index_add_(0, codes, centered_cpu)
        print(f"fit covariance {min(start + BATCH, len(fit_rows))}/{len(fit_rows)}", flush=True)
    covariance /= n_positions
    return {
        "n_positions": n_positions,
        "mean_h": mean_h,
        "covariance": covariance,
        "state_sums": state_sums,
    }


def centered_null(codebook: dict) -> tuple[torch.Tensor, dict]:
    permuted, report = deterministic_centroid_derangement(
        codebook["centroids"], codebook["masses"]
    )
    masses = codebook["masses"].float()
    mean = (permuted * masses.unsqueeze(1)).sum(0) / masses.sum()
    return permuted - mean, report


def residual_cross_covariance(
    covariance: torch.Tensor,
    down_weight: torch.Tensor,
    state_sums: torch.Tensor | None,
    centroids: torch.Tensor | None,
    n_positions: int,
) -> torch.Tensor:
    cross = covariance @ down_weight.T
    if state_sums is not None:
        if centroids is None or state_sums.shape[0] != centroids.shape[0]:
            raise ValueError("state sums and codebook differ")
        cross = cross - state_sums.to(cross.device).float().T @ centroids.to(cross.device).float() / n_positions
    return cross


def physical_program(
    coefficient: torch.Tensor,
    basis: torch.Tensor,
    rank: int,
    mean_h: torch.Tensor,
    mean_down: torch.Tensor,
    centroids: torch.Tensor,
    assignments: torch.Tensor,
    masses: torch.Tensor | None,
) -> dict:
    left, right = canonical_balanced_factors(basis[:, :rank], coefficient)
    left_q = left.to(torch.bfloat16).float().cpu()
    right_q = right.to(torch.bfloat16).float().cpu()
    centroids_q = centroids.to(torch.bfloat16).float().cpu()
    if masses is None:
        mean_b = torch.zeros_like(mean_down)
    else:
        mean_b = (centroids_q * masses.float().unsqueeze(1)).sum(0) / masses.sum()
    intercept = mean_down.cpu() - mean_b - left_q @ (right_q @ mean_h.cpu())
    return {
        "rank": rank,
        "intercept": intercept,
        "left": left,
        "right": right,
        "centroids": centroids,
        "assignments": assignments,
    }


@torch.no_grad()
def compile_fit() -> dict:
    started = time.time()
    authority = validate_authority()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    staging = PROGRAMS.with_name(f".{PROGRAMS.name}.tmp.{os.getpid()}")
    try:
        _, fit_full = stage.load_frozen_role("fit")
        fit_rows = fit_full[:, :stage.T + 1].contiguous()
        stage.load_model()
        hooks = stage.register_hooks()
        assert stage.H is not None
        capture_hook = stage.H[0].mlp.Down.register_forward_pre_hook(down_input_hook)
        try:
            state = stage.fit_state(fit_rows)
            expected_stage = json.loads(stage.FIT_RECEIPT.read_text())
            if stage.fit_receipt_payload(state, fit_rows) != expected_stage:
                raise RuntimeError("Stage-0 fit constants no longer replay")

            q = compact_centered_codebook(
                state["token_table"], state["token_count"], state["q64_labels"]
            )
            a = compact_centered_codebook(
                state["token_table"], state["token_count"], state["a64_labels"]
            )
            if (len(q["centroids"]), len(a["centroids"])) != (25, 31):
                raise RuntimeError("fit occupancy changed from frozen Stage 0")
            q_null, q_null_report = centered_null(q)
            a_null, a_null_report = centered_null(a)
            statistics = collect_product_statistics(
                fit_rows, {"Q": q["assignments"], "A": a["assignments"]}
            )
        finally:
            capture_hook.remove()
            stage.STATE["arm"] = "O"
            for hook in hooks:
                hook.remove()

        covariance = statistics["covariance"]
        mean_h = statistics["mean_h"]
        n_positions = statistics["n_positions"]
        down_weight = stage.H[0].mlp.Down.weight.detach().float().to(stage.DEV)
        mean_down = down_weight.cpu() @ mean_h
        codebooks = {
            "C": (torch.empty(0, stage.D), torch.empty(0, dtype=torch.long), None, None),
            "Q": (q["centroids"], q["assignments"], q["masses"], statistics["state_sums"]["Q"][:-1]),
            "Qnull": (q_null, q["assignments"], q["masses"], statistics["state_sums"]["Q"][:-1]),
            "A": (a["centroids"], a["assignments"], a["masses"], statistics["state_sums"]["A"][:-1]),
            "Anull": (a_null, a["assignments"], a["masses"], statistics["state_sums"]["A"][:-1]),
        }
        ranks = {"C": {256: 256, 512: 512}}
        for name, codebook in (("Q", q), ("A", a)):
            ranks[name] = {
                rung: matched_hierarchy_rank(rung, len(codebook["centroids"]))
                for rung in RUNG_RANKS
            }
            ranks[name + "null"] = dict(ranks[name])

        fits = {}
        for name, (centroids, _, _, sums) in codebooks.items():
            cross = residual_cross_covariance(
                covariance, down_weight, sums, centroids, n_positions
            )
            maximum = max(ranks[name].values()) + (0 if name.endswith("null") else 1)
            fits[name] = fit_reduced_rank_from_statistics(
                covariance, cross, maximum, ridge_fraction=RIDGE_FRACTION
            )
            print(f"fit {name} rank {maximum}", flush=True)

        staging.mkdir(parents=True)
        receipts = {}
        for name in ("C", "Q", "Qnull", "A", "Anull"):
            centroids, assignments, masses, _ = codebooks[name]
            for rung, rank in ranks[name].items():
                label = f"{name}{rank}_at_C{rung}"
                program = physical_program(
                    fits[name]["coefficient"], fits[name]["output_basis"], rank,
                    mean_h, mean_down, centroids, assignments, masses
                )
                receipt = serialize_program(staging / f"{label}.bin", program)
                loaded = load_program(Path(receipt["path"]))
                if loaded["rank"] != rank:
                    raise RuntimeError("serialized rank did not replay")
                receipts[label] = receipt

        price_gates = {}
        for hierarchy in ("Q", "A"):
            for rung in RUNG_RANKS:
                rank = ranks[hierarchy][rung]
                centroids, assignments, masses, _ = codebooks[hierarchy]
                probe_label = f"{hierarchy}{rank + 1}_price_probe_at_C{rung}"
                probe = physical_program(
                    fits[hierarchy]["coefficient"], fits[hierarchy]["output_basis"], rank + 1,
                    mean_h, mean_down, centroids, assignments, masses
                )
                probe_receipt = serialize_program(staging / f"{probe_label}.bin", probe)
                receipts[probe_label] = probe_receipt
                ceiling = receipts[f"C{rung}_at_C{rung}"]["bytes"]
                admitted = receipts[f"{hierarchy}{rank}_at_C{rung}"]["bytes"]
                price_gates[f"{hierarchy}_at_C{rung}"] = {
                    "rank": rank, "admitted_bytes": admitted, "ceiling_bytes": ceiling,
                    "next_rank": rank + 1, "next_rank_bytes": probe_receipt["bytes"],
                    "admitted_le_ceiling": admitted <= ceiling,
                    "next_rank_gt_ceiling": probe_receipt["bytes"] > ceiling,
                }
        if not all(gate["admitted_le_ceiling"] and gate["next_rank_gt_ceiling"]
                   for gate in price_gates.values()):
            raise RuntimeError("physical matched-price maximality failed")
        for rung in RUNG_RANKS:
            for hierarchy in ("Q", "A"):
                parent = receipts[f"{hierarchy}{ranks[hierarchy][rung]}_at_C{rung}"]["bytes"]
                null = receipts[f"{hierarchy}null{ranks[hierarchy][rung]}_at_C{rung}"]["bytes"]
                if parent != null:
                    raise RuntimeError("structured null price differs from parent")

        os.replace(staging, PROGRAMS)
        for receipt in receipts.values():
            receipt["path"] = str((PROGRAMS / Path(receipt["path"]).name).resolve())
        result = {
            "schema_version": 1,
            "receipt_kind": "mlp0_native_down_hierarchy_v1_fit",
            "status": "frozen_before_evaluation_authority",
            "authority": authority,
            "fit_rows": {"shape": list(fit_rows.shape), "sha256": tensor_sha256(fit_rows)},
            "fit_statistics": {
                "n_positions": n_positions,
                "mean_h_sha256": tensor_sha256(mean_h),
                "covariance_sha256": tensor_sha256(covariance),
                "ridge_fraction": RIDGE_FRACTION,
                "operation_order": "two passes; float64 CPU mean/state sums; float32 CUDA covariance",
            },
            "construction": {
                "occupancy": {"Q": len(q["centroids"]), "A": len(a["centroids"])},
                "sentinels": {"Q": q["sentinel"], "A": a["sentinel"]},
                "ranks": ranks,
                "q_null": q_null_report,
                "a_null": a_null_report,
                "common_exact_product_price": common_exact_product_price(),
            },
            "price_gates": price_gates,
            "programs": receipts,
            "runtime_s": time.time() - started,
        }
        write_json_atomic(result, FIT_RECEIPT)
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        os.close(lock_fd)
        LOCK.unlink(missing_ok=True)


def authoritative_entry() -> None:
    try:
        result = compile_fit()
        print(json.dumps({
            "status": result["status"], "ranks": result["construction"]["ranks"],
            "price_gates": result["price_gates"], "runtime_s": result["runtime_s"],
        }, indent=2), flush=True)
        print(f"wrote {FIT_RECEIPT} and {PROGRAMS}", flush=True)
    except BaseException as error:
        if not FIT_RECEIPT.exists() and not FAILURE.exists():
            write_json_atomic({
                "schema_version": 1,
                "status": "failed_closed_without_evaluation_authority",
                "error_type": type(error).__name__, "error": str(error),
                "authority_sha256": file_sha256(AUTHORITY) if AUTHORITY.exists() else None,
            }, FAILURE)
        LOCK.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    authoritative_entry()
