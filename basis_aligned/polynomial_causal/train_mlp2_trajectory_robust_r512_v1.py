#!/usr/bin/env python3
"""Fit the preregistered paired-trajectory MLP2 student without opening evaluation."""

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
RUNNER = Path(__file__).resolve()
OBJECTIVE = Path(objective.__file__).resolve()
TEST = HERE / "test_train_mlp2_trajectory_robust_r512_v1.py"
OBJECTIVE_TEST = HERE / "test_mlp2_trajectory_robust_objective.py"
AUDIT = HERE / "mlp2_trajectory_robust_r512_v1_fit_independent_audit.json"
SOURCE_PATHS = tuple(dict.fromkeys((
    PREREG, RUNNER, OBJECTIVE, TEST, OBJECTIVE_TEST,
    *composition.SOURCE_PATHS,
    Path(facade.__file__).resolve(), composition.RUNNER, refit.RUNNER,
    HERE / "mlp0_native_down_program.py",
)))
ROWS_RECEIPT = composition.ROWS_RECEIPT
AUTHORITY = HERE / "mlp2_trajectory_robust_r512_v1_fit_authority.json"
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
PROGRAM_FLOATS = 1_770_624


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def committed_sources() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                     text=True).strip()
    if commit != subprocess.check_output(["git", "rev-parse", "origin/main"],
                                         cwd=ROOT, text=True).strip():
        raise RuntimeError("fit source commit is not pushed")
    hashes = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"uncommitted fit source: {relative}")
        hashes[relative] = digest
    return commit, hashes


def validate_independent_audit(sources: Mapping[str, str]) -> tuple[dict, str]:
    raw = AUDIT.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    if set(value) != {"schema", "status", "outcome_access", "audited_source_commit",
                      "audited_source_hashes", "tests_passed", "reviewer"} or (
        value.get("schema") != "mlp2_trajectory_robust_r512_v1_fit_independent_audit"
    ) or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or value.get("audited_source_hashes") != dict(sources) \
            or not isinstance(value.get("tests_passed"), int) \
            or value["tests_passed"] < 1:
        raise RuntimeError("fit independent audit is not an exact source-bound GO")
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str):
        raise RuntimeError("fit audit commit changed")
    for relative, expected in sources.items():
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError("fit audit historical source binding changed")
    return value, digest


def validate_row_metadata(receipt: dict) -> Path:
    composition.validate_row_receipt(receipt, receipt["source_hashes"])
    if receipt["roles"]["TRAIN"] != {
        "authorized_for_training": True, "authorized_for_evaluation": False,
    }:
        raise RuntimeError("composition TRAIN role authority changed")
    train = receipt["entries"]["TRAIN"]
    if train["file_sha256"] != (
        "efb7dad2ba187e1d265773bc6d4b5ea133366a646cbc2bc07b4929e6815bd599"
    ):
        raise RuntimeError("exact preregistered TRAIN file hash changed")
    train_path = Path(train["path"])
    if file_sha256(train_path) != train["file_sha256"]:
        raise RuntimeError("paired-trajectory training rows changed")
    return train_path


def expected_capture_census(c512: bool) -> dict[str, Any]:
    native = {str(site): 48 for site in range(18)}
    if c512:
        native["0"] = 0
    return {"outer_calls": 48, "outer_returns": 48,
            "attention_sites": {str(site): 48 for site in range(18)},
            "native_mlp_sites": native, "c512": 48 if c512 else 0,
            "site2_capture": 48}


def validate_program_state(state: Any) -> None:
    shapes = {"left": (512, 1152), "right": (512, 1152),
              "down": (1152, 512), "bias": (1152,)}
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise RuntimeError("rank-512 program field family changed")
    if sum(value.numel() for value in state.values()) != PROGRAM_FLOATS:
        raise RuntimeError("rank-512 program scalar price changed")
    for key, shape in shapes.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 \
                or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise RuntimeError(f"rank-512 program tensor changed: {key}")


def protected_snapshot(authority: dict, sources: dict[str, str]) -> dict[str, Any]:
    if committed_sources()[1] != sources:
        raise RuntimeError("fit source closure changed")
    _, audit_sha = validate_independent_audit(sources)
    receipt, receipt_sha = composition.stable_json(
        ROWS_RECEIPT, authority["parents"]["rows_receipt_sha256"])
    train_path = validate_row_metadata(receipt)
    checkpoint = Path(facade.DEFAULT_SNAPSHOT)
    snapshot = {"sources": sources, "audit_sha256": audit_sha,
                "rows_receipt_sha256": receipt_sha,
                "train_rows_sha256": file_sha256(train_path),
                "c512_sha256": file_sha256(composition.C512_PATH),
                "full512_bundle_sha256": file_sha256(composition.FULL_BUNDLE),
                "checkpoint_config_sha256": file_sha256(checkpoint / "config.json"),
                "checkpoint_weights_sha256": file_sha256(checkpoint / "pytorch_model.bin")}
    if snapshot != authority["protected"]:
        raise RuntimeError("fit protected snapshot changed")
    return snapshot


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
    counts = {"outer_calls": 0, "outer_returns": 0,
              "attention_sites": {str(site): 0 for site in range(18)},
              "native_mlp_sites": {str(site): 0 for site in range(18)},
              "c512": 0, "site2_capture": 0}
    with torch.inference_mode():
        for start in range(0, rows.shape[0], 4):
            tokens = rows[start:start + 4, :-1].to(device).contiguous()
            captured: list[tuple[torch.Tensor, torch.Tensor]] = []

            def attention(event: facade.AttentionEvent):
                counts["attention_sites"][str(event.site)] += 1
                return event.block.attn(event.state, event.first_value)

            def mlp(event: facade.EarlyMLPEvent):
                if c512 and event.site == 0:
                    counts["c512"] += 1
                    return composition.c512_write(event, c512_tensors)
                write = event.block.mlp(event.state)
                counts["native_mlp_sites"][str(event.site)] += 1
                if event.site == SITE:
                    captured.append((
                        event.state[:, SCORING].detach().float().cpu(),
                        write[:, SCORING].detach().float().cpu(),
                    ))
                    counts["site2_capture"] += 1
                return write

            counts["outer_calls"] += 1
            facade.forward_with_dispatch(model, tokens, attention, mlp)
            counts["outer_returns"] += 1
            if len(captured) != 1:
                raise RuntimeError("paired-trajectory site-2 capture count changed")
            states.append(captured[0][0]); writes.append(captured[0][1])
    expected = expected_capture_census(c512)
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


def optimization_status(fit: dict, curve: list[dict]) -> str:
    worst_nrmse = max(fit["best_dev_nrmse"].values())
    tail = [point["worst_normalized_mse"] for point in curve[-5:]]
    still_improving = (len(tail) == 5
                       and all(tail[index + 1] < tail[index]
                               for index in range(1, len(tail) - 1))
                       and tail[-1] <= 0.99 * tail[0])
    return ("optimization_inconclusive" if worst_nrmse > 0.25 and still_improving
            else "fit_complete")


def validate_bundle(value: Any, parents: dict[str, str]) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema", "programs", "parents", "fit", "curves", "statuses",
        "evaluation_opened",
    } or value.get("schema") != "mlp2_trajectory_robust_r512_v1_fit_bundle" \
            or value.get("parents") != parents \
            or value.get("evaluation_opened") is not False \
            or set(value.get("programs", {})) != {"CONTINUE512", "ROBUST512"} \
            or set(value.get("fit", {})) != {"CONTINUE512", "ROBUST512"} \
            or set(value.get("curves", {})) != {"CONTINUE512", "ROBUST512"} \
            or set(value.get("statuses", {})) != {"CONTINUE512", "ROBUST512"}:
        raise RuntimeError("paired-trajectory fit bundle schema changed")
    for state in value["programs"].values():
        validate_program_state(state)
    for name in ("CONTINUE512", "ROBUST512"):
        if [point.get("step") for point in value["curves"][name]] != list(
            range(0, STEPS + 1, 25)
        ):
            raise RuntimeError("paired-trajectory development schedule changed")
        numeric = []
        for mapping in (value["fit"][name]["initial_dev_normalized_mse"],
                        value["fit"][name]["best_dev_normalized_mse"],
                        value["fit"][name]["best_dev_nrmse"],
                        value["fit"][name]["target_energy"]):
            numeric.extend(mapping.values())
        for point in value["curves"][name]:
            numeric.extend(item for item in point.values()
                           if isinstance(item, (int, float)))
        if not numeric or not all(math.isfinite(float(item)) for item in numeric):
            raise RuntimeError("paired-trajectory fit contains nonfinite metrics")
        if value["statuses"][name] != optimization_status(
            value["fit"][name], value["curves"][name]
        ):
            raise RuntimeError("paired-trajectory optimization status changed")


def terminal_guard(claim, authority: dict, sources: dict[str, str],
                   *, publishing: str, artifacts: Mapping[Path, str]) -> None:
    composition.row_life.base.require_claim(claim, LOCK)
    if publishing not in {"bundle", "result", "receipt", "failure"}:
        raise ValueError("unknown fit publication boundary")
    if RECEIPT.exists() or FAILURE.exists():
        raise RuntimeError("fit terminal artifact raced publication")
    on_disk_authority, _ = composition.stable_json(AUTHORITY)
    if on_disk_authority != authority:
        raise RuntimeError("fit authority bytes or semantics changed")
    protected_snapshot(authority, sources)
    for path, expected in artifacts.items():
        if file_sha256(path) != expected:
            raise RuntimeError("fit artifact changed before publication")
    if RECEIPT.exists() or FAILURE.exists():
        raise RuntimeError("fit terminal artifact raced late publication")
    on_disk_authority, _ = composition.stable_json(AUTHORITY)
    if on_disk_authority != authority:
        raise RuntimeError("fit authority changed late in publication")
    composition.row_life.base.require_claim(claim, LOCK)


def nested_equal(first: Any, second: Any) -> bool:
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return first.dtype == second.dtype and first.shape == second.shape \
            and torch.equal(first, second)
    if isinstance(first, dict) and isinstance(second, dict):
        return set(first) == set(second) and all(
            nested_equal(first[key], second[key]) for key in first)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            nested_equal(a, b) for a, b in zip(first, second))
    return first == second


def main() -> None:
    if any(path.exists() for path in (
        AUTHORITY, BUNDLE, RESULT, RECEIPT, FAILURE, LOCK,
    )):
        raise RuntimeError("paired-trajectory fit namespace already exists")
    claim = composition.row_life.base.acquire_claim(LOCK)
    authority: dict[str, Any] | None = None
    sources: dict[str, str] | None = None
    published: dict[Path, str] = {}
    try:
        commit, sources = committed_sources()
        audit, audit_sha = validate_independent_audit(sources)
        row_receipt, row_receipt_sha = composition.stable_json(ROWS_RECEIPT)
        train_path = validate_row_metadata(row_receipt)
        checkpoint_path = Path(facade.DEFAULT_SNAPSHOT)
        protected = {
            "sources": sources, "audit_sha256": audit_sha,
            "rows_receipt_sha256": row_receipt_sha,
            "train_rows_sha256": file_sha256(train_path),
            "c512_sha256": file_sha256(composition.C512_PATH),
            "full512_bundle_sha256": file_sha256(composition.FULL_BUNDLE),
            "checkpoint_config_sha256": file_sha256(checkpoint_path / "config.json"),
            "checkpoint_weights_sha256": file_sha256(
                checkpoint_path / "pytorch_model.bin"),
        }
        parents = {"rows_receipt_sha256": row_receipt_sha,
                   "train_rows_sha256": protected["train_rows_sha256"],
                   "c512_sha256": composition.C512_SHA,
                   "full512_bundle_sha256": composition.FULL_BUNDLE_SHA}
        if parents["c512_sha256"] != protected["c512_sha256"] or (
            parents["full512_bundle_sha256"] != protected["full512_bundle_sha256"]
        ):
            raise RuntimeError("fit parent bytes changed before authority")
        authority = {
            "schema": "mlp2_trajectory_robust_r512_v1_fit_authority",
            "status": "spent_before_train_row_or_model_access",
            "source_commit": commit, "source_hashes": sources,
            "audit_sha256": audit_sha, "audit_reviewer": audit["reviewer"],
            "parents": parents, "protected": protected,
            "authorized_role": "TRAIN", "fit_documents": 160,
            "dev_documents": 32, "evaluation_opened": False,
        }

        def authority_guard() -> None:
            composition.row_life.base.require_claim(claim, LOCK)
            if any(path.exists() for path in (
                AUTHORITY, BUNDLE, RESULT, RECEIPT, FAILURE,
            )) or committed_sources() != (commit, sources):
                raise RuntimeError("fit authority inputs or namespace changed")
            composition.row_life.base.require_claim(claim, LOCK)

        refit.atomic_json(AUTHORITY, authority, pre_link_check=authority_guard)
        authority_sha = file_sha256(AUTHORITY)
        protected_snapshot(authority, sources)
        started = time.time()
        train_entry = row_receipt["entries"]["TRAIN"]
        rows, rows_sha = composition.stable_torch(
            train_path, train_entry["file_sha256"])
        if rows_sha != parents["train_rows_sha256"] or rows.dtype != torch.long \
                or tuple(rows.shape) != (192, 257) \
                or refit.row_life.tensor_sha256(rows) != train_entry["tensor_sha256"]:
            raise RuntimeError("paired-trajectory TRAIN tensor changed")
        composition.validate_parents()
        device = torch.device("cuda")
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        torch.set_float32_matmul_precision("high")
        model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
        protected_snapshot(authority, sources)
        c512 = load_program(composition.C512_PATH)
        if file_sha256(composition.C512_PATH) != composition.C512_SHA:
            raise RuntimeError("C512 bytes changed across program load")
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
        statuses = {"CONTINUE512": optimization_status(continued_fit, continued_curve),
                    "ROBUST512": optimization_status(robust_fit, robust_curve)}
        bundle = {"schema": "mlp2_trajectory_robust_r512_v1_fit_bundle",
                  "programs": {"CONTINUE512": refit.program_state(continued),
                               "ROBUST512": refit.program_state(robust)},
                  "parents": parents,
                  "fit": {"CONTINUE512": continued_fit,
                          "ROBUST512": robust_fit},
                  "curves": {"CONTINUE512": continued_curve,
                             "ROBUST512": robust_curve},
                  "statuses": statuses,
                  "evaluation_opened": False}
        validate_bundle(bundle, parents)
        refit.atomic_torch(BUNDLE, bundle, pre_link_check=lambda: terminal_guard(
            claim, authority, sources, publishing="bundle", artifacts={}))
        published[BUNDLE] = file_sha256(BUNDLE)
        reloaded_bundle, bundle_sha = composition.stable_torch(BUNDLE, published[BUNDLE])
        validate_bundle(reloaded_bundle, parents)
        if not nested_equal(reloaded_bundle, bundle):
            raise RuntimeError("fit bundle semantic roundtrip changed")
        continued_fit = reloaded_bundle["fit"]["CONTINUE512"]
        robust_fit = reloaded_bundle["fit"]["ROBUST512"]
        statuses = reloaded_bundle["statuses"]
        result = {
            "schema": "mlp2_trajectory_robust_r512_v1_fit_result",
            "status": ("optimization_inconclusive" if
                       "optimization_inconclusive" in statuses.values()
                       else "training_complete_evaluation_unopened"),
            "runtime_seconds": time.time() - started,
            "fit": {"CONTINUE512": continued_fit, "ROBUST512": robust_fit},
            "trajectory_shift_nrmse": {
                "pre_mlp2_state": relative_shift(native_x, c512_x),
                "native_mlp2_write": relative_shift(native_y, c512_y),
            },
            "calls": {"native": native_calls, "c512": c512_calls},
            "parents": parents, "checkpoint": checkpoint.__dict__,
            "authority_sha256": authority_sha, "bundle_sha256": bundle_sha,
            "program_price": {"rank": 512, "products_per_token": 512,
                              "float32_coefficients_each": PROGRAM_FLOATS,
                              "native_mlp2_calls_at_deployment": 0},
            "documents": {"fit": FIT_DOCUMENTS, "dev": DEV_DOCUMENTS},
            "evaluation_opened": False,
        }
        refit.atomic_json(RESULT, result, pre_link_check=lambda: terminal_guard(
            claim, authority, sources, publishing="result", artifacts=published))
        published[RESULT] = file_sha256(RESULT)
        if composition.stable_json(RESULT, published[RESULT])[0] != result:
            raise RuntimeError("fit result semantic replay changed")
        receipt_out = {
            "schema": "mlp2_trajectory_robust_r512_v1_fit_receipt",
            "status": "fit_complete_receipt_last_evaluation_unopened",
            "result_sha256": published[RESULT], "bundle_sha256": bundle_sha,
            "authority_sha256": authority_sha, "parents": parents,
            "evaluation_opened": False,
        }
        refit.atomic_json(RECEIPT, receipt_out, pre_link_check=lambda: terminal_guard(
            claim, authority, sources, publishing="receipt", artifacts=published))
        print(json.dumps(result, sort_keys=True, indent=2))
    except BaseException as exc:
        if not RECEIPT.exists() and not FAILURE.exists():
            failure = {
                "schema": "mlp2_trajectory_robust_r512_v1_fit_failure",
                "status": "terminal_failure_no_receipt", "error": repr(exc),
                "bundle_exists": BUNDLE.exists(), "result_exists": RESULT.exists(),
                "artifact_hashes": {str(path): digest for path, digest in published.items()},
                "authority_sha256": (file_sha256(AUTHORITY)
                                      if AUTHORITY.exists() else None),
                "evaluation_opened": False}
            if authority is not None and sources is not None:
                refit.atomic_json(FAILURE, failure, pre_link_check=lambda: terminal_guard(
                    claim, authority, sources, publishing="failure", artifacts=published))
            else:
                refit.atomic_json(FAILURE, failure)
        raise
    finally:
        composition.row_life.base.release_claim(claim, LOCK)


if __name__ == "__main__":
    main()
