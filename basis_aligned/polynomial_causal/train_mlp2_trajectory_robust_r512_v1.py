#!/usr/bin/env python3
"""Fit the preregistered paired-trajectory MLP2 student without opening evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for source_root in (ROOT, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
import mlp2_trajectory_robust_objective as objective
import run_mlp0_c512_mlp2_full512_composition_v1 as composition
import run_mlp2_rank512_refit_v1 as refit
from mlp0_native_down_program import load_program

PREREG = HERE / "MLP2_TRAJECTORY_ROBUST_R512_V1_PREREGISTRATION.md"
ROWS_RECEIPT = composition.ROWS_RECEIPT
BUNDLE = HERE / "mlp2_trajectory_robust_r512_v1_fit_bundle.pt"
RESULT = HERE / "mlp2_trajectory_robust_r512_v1_fit_result.json"
RECEIPT = HERE / "mlp2_trajectory_robust_r512_v1_fit_receipt.json"
FAILURE = HERE / "mlp2_trajectory_robust_r512_v1_fit_failure.json"
LOCK = Path("/workspace/runs/.mlp2_trajectory_robust_r512_v1_fit.lock")

SITE = 2
SCORING = slice(64, 256)
FIT_DOCUMENTS = 160
DEV_DOCUMENTS = 32
TOKENS_PER_DOCUMENT = 192
STEPS = 1200
BATCH_PER_BACKGROUND = 512
LEARNING_RATE = 3e-4
SEED = 2026082941


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_inputs() -> tuple[dict, Path]:
    receipt = json.loads(ROWS_RECEIPT.read_text())
    composition.validate_row_receipt(receipt, receipt["source_hashes"])
    if receipt["roles"]["TRAIN"] != {
        "authorized_for_training": True, "authorized_for_evaluation": False,
    }:
        raise RuntimeError("composition TRAIN role authority changed")
    train = receipt["entries"]["TRAIN"]
    train_path = Path(train["path"])
    if file_sha256(train_path) != train["file_sha256"]:
        raise RuntimeError("paired-trajectory training rows changed")
    composition.validate_parents()
    if any(path.exists() for path in (BUNDLE, RESULT, RECEIPT, FAILURE, LOCK)):
        raise RuntimeError("paired-trajectory fit namespace already exists")
    return receipt, train_path


def relative_shift(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape or first.numel() == 0:
        raise ValueError("paired tensors must have one equal nonempty shape")
    denominator = (first - first.mean(0, keepdim=True)).square().sum()
    return float(torch.sqrt((second - first).square().sum() / denominator.clamp_min(1e-30)))


@torch.no_grad()
def capture_background(
    model, rows: torch.Tensor, device: torch.device, *, c512: bool,
    c512_tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    states: list[torch.Tensor] = []
    writes: list[torch.Tensor] = []
    counts = {"outer": 0, "attention": 0, "native_mlp": 0,
              "c512": 0, "site2_capture": 0}
    with torch.inference_mode():
        for start in range(0, rows.shape[0], 4):
            tokens = rows[start:start + 4, :-1].to(device).contiguous()
            captured: list[tuple[torch.Tensor, torch.Tensor]] = []

            def attention(event: facade.AttentionEvent):
                counts["attention"] += 1
                return event.block.attn(event.state, event.first_value)

            def mlp(event: facade.EarlyMLPEvent):
                if c512 and event.site == 0:
                    counts["c512"] += 1
                    return composition.c512_write(event, c512_tensors)
                write = event.block.mlp(event.state)
                counts["native_mlp"] += 1
                if event.site == SITE:
                    captured.append((
                        event.state[:, SCORING].detach().float().cpu(),
                        write[:, SCORING].detach().float().cpu(),
                    ))
                    counts["site2_capture"] += 1
                return write

            facade.forward_with_dispatch(model, tokens, attention, mlp)
            counts["outer"] += 1
            if len(captured) != 1:
                raise RuntimeError("paired-trajectory site-2 capture count changed")
            states.append(captured[0][0]); writes.append(captured[0][1])
    expected = {
        "outer": 48, "attention": 48 * 18,
        "native_mlp": 48 * (17 if c512 else 18),
        "c512": 48 if c512 else 0, "site2_capture": 48,
    }
    if counts != expected:
        raise RuntimeError(f"paired-trajectory call census changed: {counts}")
    return (torch.cat(states).reshape(-1, refit.WIDTH),
            torch.cat(writes).reshape(-1, refit.WIDTH), counts)


@torch.no_grad()
def background_dev_loss(candidate, x, y, energy: float) -> float:
    squared_error = 0.0
    count = 0
    for start in range(0, x.shape[0], 1024):
        xb = x[start:start + 1024].to(candidate.left.device).float()
        yb = y[start:start + 1024].to(candidate.left.device).float()
        squared_error += float((candidate(xb) - yb).square().sum())
        count += yb.numel()
    return squared_error / count / energy


def train(candidate, native_x, native_y, second_x, second_y,
          c512_dev_x, c512_dev_y, device):
    split = FIT_DOCUMENTS * TOKENS_PER_DOCUMENT
    nx, ny, sx, sy = (value[:split] for value in
                       (native_x, native_y, second_x, second_y))
    ndx, ndy = native_x[split:], native_y[split:]
    cdx, cdy = c512_dev_x[split:], c512_dev_y[split:]
    native_energy = float(objective.centered_target_energy(ny))
    c512_energy = float(objective.centered_target_energy(c512_dev_y[:split]))
    second_energy = native_energy if second_x is native_x else c512_energy
    optimizer = torch.optim.Adam(candidate.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(SEED)

    def observe() -> tuple[float, float]:
        return (background_dev_loss(candidate, ndx, ndy, native_energy),
                background_dev_loss(candidate, cdx, cdy, c512_energy))

    initial = observe()
    baseline = initial
    best_losses = initial
    best = refit.program_state(candidate)
    curve = [{"step": 0, "native_normalized_mse": initial[0],
              "c512_normalized_mse": initial[1],
              "worst_normalized_mse": max(initial)}]
    candidate.train()
    for step in range(1, STEPS + 1):
        ni = torch.randint(0, nx.shape[0], (BATCH_PER_BACKGROUND,), generator=generator)
        si = torch.randint(0, sx.shape[0], (BATCH_PER_BACKGROUND,), generator=generator)
        nxb, nyb = nx[ni].to(device).float(), ny[ni].to(device).float()
        sxb, syb = sx[si].to(device).float(), sy[si].to(device).float()
        optimizer.zero_grad(set_to_none=True)
        loss, report = objective.balanced_background_loss(
            candidate(nxb), nyb, candidate(sxb), syb, native_energy, second_energy,
        )
        loss.backward(); optimizer.step()
        if step % 25 == 0:
            observed = observe()
            curve.append({"step": step,
                          "native_normalized_mse": observed[0],
                          "c512_normalized_mse": observed[1],
                          "worst_normalized_mse": max(observed),
                          "train_balanced_loss": float(loss.detach()),
                          "train_native_normalized_mse": float(
                              report["native_normalized_mse"].detach()),
                          "train_second_normalized_mse": float(
                              report["c512_normalized_mse"].detach())})
            if objective.retain_checkpoint(*observed, *best_losses, *baseline):
                best_losses = observed
                best = refit.program_state(candidate)
    candidate.load_state_dict(best); candidate.eval()
    return candidate, curve, {
        "initial_dev_normalized_mse": {"native": initial[0], "c512": initial[1]},
        "best_dev_normalized_mse": {"native": best_losses[0], "c512": best_losses[1]},
        "best_dev_nrmse": {"native": best_losses[0] ** 0.5,
                           "c512": best_losses[1] ** 0.5},
        "target_energy": {"native": native_energy, "c512": c512_energy},
        "steps": STEPS,
    }


def main() -> None:
    receipt, train_path = validate_inputs()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(descriptor, f"{os.getpid()}\n".encode()); os.fsync(descriptor); os.close(descriptor)
    started = time.time()
    try:
        rows = torch.load(train_path, map_location="cpu", weights_only=True)
        if rows.dtype != torch.long or tuple(rows.shape) != (192, 257):
            raise RuntimeError("paired-trajectory TRAIN tensor changed")
        device = torch.device("cuda")
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.set_float32_matmul_precision("high")
        model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
        c512 = load_program(composition.C512_PATH)
        c512_tensors = {key: c512[key].to(device)
                        for key in ("intercept", "left", "right")}
        native_x, native_y, native_calls = capture_background(
            model, rows, device, c512=False, c512_tensors=c512_tensors)
        c512_x, c512_y, c512_calls = capture_background(
            model, rows, device, c512=True, c512_tensors=c512_tensors)
        parent, _ = composition.stable_torch(
            composition.FULL_BUNDLE, composition.FULL_BUNDLE_SHA)
        initial_state = parent["programs"]["FULL512"]
        continued = refit.build_from_state(initial_state, device)
        robust = refit.build_from_state(initial_state, device)
        continued, continued_curve, continued_fit = train(
            continued, native_x, native_y, native_x, native_y,
            c512_x, c512_y, device)
        robust, robust_curve, robust_fit = train(
            robust, native_x, native_y, c512_x, c512_y,
            c512_x, c512_y, device)
        parents = {
            "rows_receipt_sha256": file_sha256(ROWS_RECEIPT),
            "train_rows_sha256": receipt["entries"]["TRAIN"]["file_sha256"],
            "c512_sha256": composition.C512_SHA,
            "full512_bundle_sha256": composition.FULL_BUNDLE_SHA,
        }
        bundle = {"schema": "mlp2_trajectory_robust_r512_v1_fit_bundle",
                  "programs": {"CONTINUE512": refit.program_state(continued),
                               "ROBUST512": refit.program_state(robust)},
                  "parents": parents,
                  "fit": {"CONTINUE512": continued_fit,
                          "ROBUST512": robust_fit},
                  "curves": {"CONTINUE512": continued_curve,
                             "ROBUST512": robust_curve},
                  "evaluation_opened": False}
        refit.atomic_torch(BUNDLE, bundle)
        result = {
            "schema": "mlp2_trajectory_robust_r512_v1_fit_result",
            "status": "training_complete_evaluation_unopened",
            "runtime_seconds": time.time() - started,
            "fit": {"CONTINUE512": continued_fit, "ROBUST512": robust_fit},
            "trajectory_shift_nrmse": {
                "pre_mlp2_state": relative_shift(native_x, c512_x),
                "native_mlp2_write": relative_shift(native_y, c512_y),
            },
            "calls": {"native": native_calls, "c512": c512_calls},
            "parents": parents, "checkpoint": checkpoint.__dict__,
            "bundle_sha256": file_sha256(BUNDLE),
            "documents": {"fit": FIT_DOCUMENTS, "dev": DEV_DOCUMENTS},
            "evaluation_opened": False,
        }
        refit.atomic_json(RESULT, result)
        receipt_out = {
            "schema": "mlp2_trajectory_robust_r512_v1_fit_receipt",
            "status": "fit_complete_receipt_last_evaluation_unopened",
            "result_sha256": file_sha256(RESULT),
            "bundle_sha256": file_sha256(BUNDLE),
            "parents": parents, "evaluation_opened": False,
        }
        refit.atomic_json(RECEIPT, receipt_out)
        print(json.dumps(result, sort_keys=True, indent=2))
    except BaseException as exc:
        if not RECEIPT.exists() and not FAILURE.exists():
            refit.atomic_json(FAILURE, {
                "schema": "mlp2_trajectory_robust_r512_v1_fit_failure",
                "status": "terminal_failure_no_receipt", "error": repr(exc),
                "bundle_exists": BUNDLE.exists(), "result_exists": RESULT.exists(),
                "evaluation_opened": False,
            })
        raise
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
