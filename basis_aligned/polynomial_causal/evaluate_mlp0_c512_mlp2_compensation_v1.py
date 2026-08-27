#!/usr/bin/env python3
"""Authority-bound evaluator for the physical C512 -> MLP2 factorial."""

from __future__ import annotations

from contextlib import contextmanager
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


ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
PC = ROOT / "basis_aligned" / "polynomial_causal"
sys.path[:0] = [str(ROOT), str(BQ), str(PC)]

import evaluate_mlp0_c512_mlp1_interchange_v1 as base  # noqa: E402
from mlp0_c512_mlp1_interchange import (  # noqa: E402
    capture_through_mlp1, document_derangement,
)
from mlp0_c512_mlp2_compensation import (  # noqa: E402
    capture_mlp2_paths, norm_matched_native_write, physical_mlp2_matrix,
    post_mlp1_paths, suffix_from_mlp2,
)
from mlp0_c512_mlp2_evaluator_contract import (  # noqa: E402
    BATCH_WINDOWS, ARM_CARRIED_PATH, build_unit_identity, carried_inputs_for_arm,
    contrast_logits, control_contract_sha256, control_realization_sha256,
    coverage_by_wave, expected_call_contract, unit_identity_hashes,
    verify_derangement,
)
from mlp0_native_down_program import load_program  # noqa: E402
from prepare_mlp0_c512_mlp2_compensation_v1_rows import (  # noqa: E402
    RECEIPT as ROW_RECEIPT, load_frozen_rows,
)
from prepare_mlp0_quotient_stage0_v1_rows import load_frozen_role  # noqa: E402
from score_mlp0_c512_mlp2_compensation_v1 import (  # noqa: E402
    CONTRASTS, MARGINS, score_result, validate_integrity,
)


D = 1152
V = 50257
T = 256
PROGRAM_KEY = "C512_at_C512"
AUTHORITY = BQ / "mlp0_c512_mlp2_compensation_v1_eval_authority.json"
OUT = BQ / "mlp0_c512_mlp2_compensation_v1_results.json"
FAILURE = BQ / "mlp0_c512_mlp2_compensation_v1_failure.json"
LOCK = Path("/workspace/runs/.bilin18_mlp0_c512_mlp2_compensation_v1.lock")
FIT_RECEIPT = BQ / "mlp0_native_down_hierarchy_v1_fit_receipt.json"
STAGE0_FIT_RECEIPT = BQ / "mlp0_quotient_stage0_v2_fit_receipt.json"
STAGE0_ROW_RECEIPT = BQ / "mlp0_quotient_stage0_v1_rows_receipt.json"
INHERITED_RESULT = BQ / "mlp0_c512_mlp1_interchange_v3_results.json"
INHERITED_AUTHORITY = BQ / "mlp0_c512_mlp1_interchange_v3_eval_authority.json"
CELL_NAMES = [
    f"pos{pos}_freq{freq}_prev{prev}_dev{dev}"
    for pos in range(2) for freq in range(2)
    for prev in range(2) for dev in range(2)
]


file_sha256 = base.file_sha256
tensor_sha256 = base.tensor_sha256
closure_sha256 = base.closure_sha256
write_json_atomic = base.write_json_atomic
C512Down = base.C512Down
install_native = base.install_native
install_candidate = base.install_candidate
pair_effects = base.pair_effects
add_unit_cells = base.add_unit_cells


def write_json_create_only(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            descriptor = None
            handle.write(json.dumps(payload, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def acquire_lock(path: Path | None = None) -> int:
    path = LOCK if path is None else path
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"MLP2 evaluation lock is already owned: {path}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
    except BaseException:
        release_owned_lock(descriptor, path)
        raise
    return descriptor


def release_owned_lock(descriptor: int, path: Path | None = None) -> None:
    path = LOCK if path is None else path
    try:
        if path.exists() and os.stat(path).st_ino == os.fstat(descriptor).st_ino:
            path.unlink()
    finally:
        os.close(descriptor)


def load_domain() -> tuple[dict[str, Any], dict[str, object], torch.Tensor]:
    receipt, rows = load_frozen_rows()
    windows = torch.cat([rows[:, :257], rows[:, 256:513]], dim=0).contiguous()
    identity = build_unit_identity(receipt)
    if tuple(windows.shape) != (1256, 257):
        raise RuntimeError("frozen MLP2 evaluation-window shape changed")
    if len(identity["row_to_unit"]) != len(windows):
        raise RuntimeError("row-to-source-document mapping changed")
    return {
        "rows": windows,
        "unit_ids": torch.tensor(identity["row_to_unit"], dtype=torch.long),
        "n_units": 384,
    }, identity, rows


def json_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def inherited_currency_contract() -> tuple[dict[str, object], str]:
    """Bind every input to the inherited scoring currency without a model forward."""
    payload = json.loads(INHERITED_RESULT.read_text())
    if (payload.get("authority", {}).get("status")
            != "frozen_before_any_c512_mlp1_evaluation_forward"
            or payload.get("inference", {}).get("integrity_passes") is not True):
        raise RuntimeError("inherited C512/MLP1 currency result is not authoritative")
    value = payload.get("fit_frozen_centered_logit_rms")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise RuntimeError("inherited centered capped-logit RMS is invalid")
    authority_sha = file_sha256(INHERITED_AUTHORITY)
    if payload.get("authority_file_sha256") != authority_sha:
        raise RuntimeError("inherited result does not bind its authority file")
    if payload.get("authority") != json.loads(INHERITED_AUTHORITY.read_text()):
        raise RuntimeError("inherited result's embedded authority differs from its file")
    _, fit_full = load_frozen_role("fit")
    fit_rows = fit_full[:, :257].contiguous()
    constants = json.loads(STAGE0_FIT_RECEIPT.read_text())["constants"]
    token_count = torch.bincount(
        fit_rows[:, :-1].reshape(-1), minlength=V
    ).to(torch.int64)
    punctuation = base.punctuation_table().to(torch.uint8)
    contract = {
        "prior_result_path": str(INHERITED_RESULT.resolve()),
        "prior_result_sha256": file_sha256(INHERITED_RESULT),
        "prior_authority_path": str(INHERITED_AUTHORITY.resolve()),
        "prior_authority_sha256": authority_sha,
        "centered_capped_logit_rms": float(value),
        "stage0_row_receipt_sha256": file_sha256(STAGE0_ROW_RECEIPT),
        "stage0_fit_receipt_sha256": file_sha256(STAGE0_FIT_RECEIPT),
        "fit_rows_tensor_sha256": tensor_sha256(fit_rows),
        "fit_rows_shape": list(fit_rows.shape),
        "token_count_tensor_sha256": tensor_sha256(token_count),
        "frequency_median": float(constants["frequency_median"]),
        "pre_mlp0_raw_residual_norm_median": float(
            constants["pre_mlp0_raw_residual_norm_median"]
        ),
        "punctuation_table_sha256": tensor_sha256(punctuation),
        "valid_mask_definition": "fit_token_count[current_token] > 0",
        "logit_cap_definition": "30*tanh(raw_logits/30)",
        "nrmse_definition": (
            "RMS(center(candidate_capped)-center(reference_capped)) / "
            "inherited_centered_capped_logit_rms"
        ),
    }
    return contract, json_sha256(contract)


def inherited_currency() -> tuple[float, str]:
    contract, digest = inherited_currency_contract()
    return float(contract["centered_capped_logit_rms"]), digest


def empty_ledgers(n_units: int = 384) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    return {
        contrast: {
            metric: {
                "sums": torch.zeros(n_units, 16, dtype=torch.float64),
                "counts": torch.zeros(n_units, 16, dtype=torch.float64),
            }
            for metric in MARGINS
        }
        for contrast in CONTRASTS
    }


class PhaseSiteCounter:
    """Count completed native MLP calls in the exact registered execution phase."""

    def __init__(self, blocks: Any, expected: Mapping[str, Mapping[str, int]]):
        self.expected = {phase: dict(values) for phase, values in expected.items()}
        self.counts = {
            phase: {site: 0 for site in values} for phase, values in self.expected.items()
        }
        self.current: str | None = None
        self.hooks = [
            block.mlp.register_forward_hook(self._hook(site))
            for site, block in enumerate(blocks)
        ]

    def _hook(self, site: int):
        def record(module: Any, args: Any, output: Any) -> None:
            if self.current is None:
                raise RuntimeError(f"unphased MLP call at site {site}")
            key = str(site)
            if self.current == "crossed_suffix_replay" and site in (1, 2):
                self.counts["crossed_forbidden_teacher"][key] += 1
                return
            if key in self.counts[self.current]:
                self.counts[self.current][key] += 1
                return
            # MLP0 runs while capturing MLP1, and MLP1 is already captured when
            # the physical MLP2 interface is evaluated. They are intentional.
            allowed_ignored = {
                "mlp1_teacher_capture": {0},
                "mlp2_teacher_capture": set(),
            }
            if site not in allowed_ignored.get(self.current, set()):
                raise RuntimeError(f"unexpected site {site} in phase {self.current}")
        return record

    @contextmanager
    def phase(self, name: str):
        if name not in self.counts or name == "crossed_forbidden_teacher":
            raise ValueError(f"unregistered execution phase: {name}")
        if self.current is not None:
            raise RuntimeError("nested execution phases are forbidden")
        self.current = name
        try:
            yield
        finally:
            self.current = None

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()


def derangement_groups(cells: torch.Tensor, unit_ids: torch.Tensor) -> torch.Tensor:
    if cells.ndim != 2 or unit_ids.shape != (cells.shape[0],):
        raise ValueError("derangement cell/unit shapes differ")
    wave = (unit_ids >= 192).long()[:, None]
    return cells.long() + 16 * wave


@torch.no_grad()
def capture_interfaces(
    model: Any, blocks: Any, idx: torch.Tensor, mlp0: Any,
    original: Any, original_forward: Any, proxy: C512Down, poisoned: Any,
    counter: PhaseSiteCounter,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    with counter.phase("mlp1_teacher_capture"):
        install_native(mlp0, original, original_forward)
        exact = capture_through_mlp1(model, blocks, idx)
        install_candidate(mlp0, proxy, original, poisoned)
        candidate = capture_through_mlp1(model, blocks, idx)
    paths = post_mlp1_paths(exact, candidate)
    with counter.phase("mlp2_teacher_capture"):
        interfaces = capture_mlp2_paths(blocks, paths)
    return exact, candidate, interfaces


@torch.no_grad()
def prepare_domain(
    domain: Mapping[str, Any], model: Any, blocks: Any, mlp0: Any,
    original: Any, original_forward: Any, proxy: C512Down, poisoned: Any,
    counter: PhaseSiteCounter, token_count: torch.Tensor, punctuation: torch.Tensor,
    frequency_median: float, norm_median: float,
) -> dict[str, Any]:
    rows = domain["rows"]
    unit_ids = domain["unit_ids"]
    delta = torch.empty(len(rows), T, D, dtype=torch.float32)
    cells_all = torch.empty(len(rows), T, dtype=torch.uint8)
    valid_all = torch.empty(len(rows), T, dtype=torch.bool)
    carried_max = {"x0": 0.0, "v1": 0.0}
    for start in range(0, len(rows), BATCH_WINDOWS):
        stop = min(start + BATCH_WINDOWS, len(rows))
        idx = rows[start:stop, :-1].to("cuda").contiguous()
        exact, candidate, interfaces = capture_interfaces(
            model, blocks, idx, mlp0, original, original_forward, proxy,
            poisoned, counter,
        )
        cells, valid = base.cell_map(
            idx, exact["pre_mlp0"], token_count, punctuation,
            frequency_median, norm_median,
        )
        delta[start:stop] = (interfaces["C"]["m"] - interfaces["O"]["m"]).cpu()
        cells_all[start:stop] = cells.byte().cpu()
        valid_all[start:stop] = valid.cpu()
        for key in carried_max:
            carried_max[key] = max(
                carried_max[key], float((exact[key] - candidate[key]).abs().max())
            )
    recipient_units = unit_ids[:, None].expand(-1, T).reshape(-1)
    groups = derangement_groups(cells_all.long(), unit_ids).reshape(-1)
    permutation = document_derangement(recipient_units, groups)
    donor_units = recipient_units[permutation]
    donor_groups = groups[permutation]
    checks = verify_derangement(
        permutation, recipient_units, donor_units, groups, donor_groups
    )
    if not all(checks.values()):
        raise RuntimeError(f"registered MLP2 derangement failed: {checks}")
    realization = control_realization_sha256(
        permutation, recipient_units, donor_units, groups
    )
    return {
        "delta": delta,
        "cells": cells_all.long(),
        "valid": valid_all,
        "permutation": permutation,
        "control_checks": checks,
        "control_realization_sha256": realization,
        "control_realization_receipt": {
            "permutation_sha256": tensor_sha256(permutation.long()),
            "recipient_units_sha256": tensor_sha256(recipient_units.long()),
            "donor_units_sha256": tensor_sha256(donor_units.long()),
            "recipient_groups_sha256": tensor_sha256(groups.long()),
            "donor_groups_sha256": tensor_sha256(donor_groups.long()),
            "positions": int(permutation.numel()),
        },
        "carried_state_identity_max": carried_max,
    }


def replay_error(
    replay: dict[str, float], arm_raw: torch.Tensor, arm_capped: torch.Tensor,
    parent_raw: torch.Tensor, parent_capped: torch.Tensor, target: torch.Tensor,
) -> None:
    replay["raw_logits_max_abs"] = max(
        replay["raw_logits_max_abs"], float((arm_raw - parent_raw).abs().max())
    )
    replay["capped_logits_max_abs"] = max(
        replay["capped_logits_max_abs"],
        float((arm_capped - parent_capped).abs().max()),
    )
    arm_ce = F.cross_entropy(arm_capped.flatten(0, 1), target.flatten())
    parent_ce = F.cross_entropy(parent_capped.flatten(0, 1), target.flatten())
    replay["ce_abs"] = max(replay["ce_abs"], float((arm_ce - parent_ce).abs()))


@torch.no_grad()
def evaluate_domain(
    domain: Mapping[str, Any], prepared: Mapping[str, Any], model: Any,
    blocks: Any, mlp0: Any, original: Any, original_forward: Any,
    proxy: C512Down, poisoned: Any, counter: PhaseSiteCounter,
    logit_scale: float, replay: dict[str, dict[str, float]],
    token_count: torch.Tensor, punctuation: torch.Tensor,
    frequency_median: float, norm_median: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    rows, unit_ids = domain["rows"], domain["unit_ids"]
    ledgers = empty_ledgers()
    flat_delta = prepared["delta"].reshape(-1, D)
    permutation = prepared["permutation"]
    same_delta_max = 0.0
    native_norm_max = 0.0
    for start in range(0, len(rows), BATCH_WINDOWS):
        stop = min(start + BATCH_WINDOWS, len(rows))
        idx = rows[start:stop, :-1].to("cuda").contiguous()
        target = rows[start:stop, 1:].to("cuda").contiguous()
        exact, candidate, interfaces = capture_interfaces(
            model, blocks, idx, mlp0, original, original_forward, proxy,
            poisoned, counter,
        )
        cells, valid = base.cell_map(
            idx, exact["pre_mlp0"], token_count, punctuation,
            frequency_median, norm_median,
        )
        if (not torch.equal(cells.cpu(), prepared["cells"][start:stop])
                or not torch.equal(valid.cpu(), prepared["valid"][start:stop])):
            raise RuntimeError("scoring cell/valid realization differs from preparation")
        realized_delta = interfaces["C"]["m"] - interfaces["O"]["m"]
        expected_delta = prepared["delta"][start:stop].to("cuda")
        same_delta_max = max(
            same_delta_max, float((realized_delta - expected_delta).abs().max())
        )
        matrix = physical_mlp2_matrix(interfaces)
        indices = torch.arange(start * T, stop * T)
        shuffled = flat_delta[permutation[indices]].reshape(stop - start, T, D).to("cuda")
        native = norm_matched_native_write(realized_delta, interfaces["O"]["m"])
        native_norm_max = max(
            native_norm_max,
            float((native.float().norm(dim=-1)
                   - realized_delta.float().norm(dim=-1)).abs().max()),
        )
        posts = {
            **matrix,
            "O0": interfaces["O"]["s"],
            "C0": interfaces["C"]["s"],
            "CS": matrix["CO"] + shuffled,
            "ON": matrix["OO"] + native,
        }
        outputs: dict[str, torch.Tensor] = {}
        raws: dict[str, torch.Tensor] = {}
        with counter.phase("crossed_suffix_replay"):
            for arm, post in posts.items():
                carried, x0 = carried_inputs_for_arm(arm, interfaces)
                if arm in {"OO", "CC", "O0", "C0"}:
                    raw, capped = suffix_from_mlp2(
                        model, blocks, post, carried, x0, return_raw=True
                    )
                    raws[arm], outputs[arm] = raw, capped
                else:
                    outputs[arm] = suffix_from_mlp2(
                        model, blocks, post, carried, x0
                    )

        with counter.phase("parent_replay_mlp_sites"):
            install_native(mlp0, original, original_forward)
            exact_raw, exact_live = base.full_forward(model, blocks, idx, "live")
            exact_omit_raw, exact_omit = base.full_forward(
                model, blocks, idx, "mlp2_omit"
            )
            install_candidate(mlp0, proxy, original, poisoned)
            candidate_raw, candidate_live = base.full_forward(model, blocks, idx, "live")
            candidate_omit_raw, candidate_omit = base.full_forward(
                model, blocks, idx, "mlp2_omit"
            )
        for name, arm, parent_raw, parent_capped in (
            ("exact_live", "OO", exact_raw, exact_live),
            ("candidate_live", "CC", candidate_raw, candidate_live),
            ("exact_mlp2_omit", "O0", exact_omit_raw, exact_omit),
            ("candidate_mlp2_omit", "C0", candidate_omit_raw, candidate_omit),
        ):
            replay_error(
                replay[name], raws[arm], outputs[arm], parent_raw, parent_capped, target
            )
        for contrast, (reference, candidate_logits) in contrast_logits(outputs).items():
            effects = pair_effects(reference, candidate_logits, target, logit_scale)
            for metric, values in effects.items():
                add_unit_cells(
                    ledgers[contrast][metric], unit_ids[start:stop],
                    prepared["cells"][start:stop], prepared["valid"][start:stop],
                    values.cpu(),
                )
        print(f"evaluated FineWeb windows {stop}/{len(rows)}", flush=True)
    return ledgers, {
        "same_realization_delta_max_abs": same_delta_max,
        "native_control_norm_max_abs": native_norm_max,
    }


def verify_authority_file() -> dict[str, object]:
    if not AUTHORITY.is_file():
        raise RuntimeError("frozen MLP2 evaluation authority is absent")
    relative = str(AUTHORITY.relative_to(ROOT))
    blob = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    if hashlib.sha256(blob).hexdigest() != file_sha256(AUTHORITY):
        raise RuntimeError("MLP2 authority is not byte-identical to committed HEAD")
    authority = json.loads(AUTHORITY.read_text())
    if (authority.get("status")
            != "frozen_before_any_c512_mlp2_compensation_evaluation_forward"
            or authority.get("output_path") != str(OUT)
            or authority.get("failure_path") != str(FAILURE)):
        raise RuntimeError("MLP2 authority status or namespace changed")
    for raw, expected in authority.get("source_hashes", {}).items():
        if file_sha256(Path(raw)) != expected:
            raise RuntimeError(f"authority-bound source changed: {raw}")
    for raw, expected in authority.get("model_files", {}).items():
        if file_sha256(Path(raw)) != expected:
            raise RuntimeError(f"authority-bound model file changed: {raw}")
    if OUT.exists() or FAILURE.exists():
        raise RuntimeError("MLP2 evaluation namespace is already spent")
    return authority


def verify_preflight_artifacts(
    authority: Mapping[str, object], frozen_rows: torch.Tensor, program_path: Path,
) -> dict[str, str]:
    source_hashes = {raw: file_sha256(Path(raw)) for raw in authority["source_hashes"]}
    roles = authority.get("model_file_roles", {})
    if set(roles) != {"config", "checkpoint"} or len(set(roles.values())) != 2:
        raise RuntimeError("authority model roles are incomplete or aliased")
    if any(raw not in authority.get("model_files", {}) for raw in roles.values()):
        raise RuntimeError("authority model role is outside the bound model files")
    observed = {
        "source_closure_sha256": closure_sha256(source_hashes),
        "row_receipt_sha256": file_sha256(ROW_RECEIPT),
        "row_tensor_sha256": tensor_sha256(frozen_rows),
        "c512_program_sha256": file_sha256(program_path),
        "model_checkpoint_sha256": file_sha256(Path(roles["checkpoint"])),
        "model_config_sha256": file_sha256(Path(roles["config"])),
        "inherited_currency_sha256": inherited_currency()[1],
        "control_contract_sha256": control_contract_sha256(),
    }
    if observed != authority["integrity_contract"]["bound_hashes"]:
        raise RuntimeError("MLP2 preflight artifacts differ from frozen authority")
    value, digest = inherited_currency()
    if digest != observed["inherited_currency_sha256"]:
        raise RuntimeError("inherited scoring-currency contract changed")
    if value != authority["integrity_contract"]["inherited_centered_capped_logit_rms"]:
        raise RuntimeError("inherited scoring currency changed")
    return observed


def poison_canary(mlp0: Any, original: Any, original_forward: Any) -> int:
    return base.poison_canary(mlp0, original, original_forward)


@torch.no_grad()
def main() -> None:
    started = time.time()
    authority = verify_authority_file()
    original = original_forward = mlp0 = counter = None
    try:
        domain, identity, frozen_rows = load_domain()
        contract = authority["integrity_contract"]
        if (unit_identity_hashes(identity) != contract["unit_identity_hashes"]
                or len(domain["rows"]) != contract["n_eval_windows"]):
            raise RuntimeError("runtime source-document identity differs from authority")
        _, fit_full = load_frozen_role("fit")
        fit_rows = fit_full[:, :257].contiguous()
        fit_constants = json.loads(STAGE0_FIT_RECEIPT.read_text())["constants"]
        frequency_median = float(fit_constants["frequency_median"])
        norm_median = float(fit_constants["pre_mlp0_raw_residual_norm_median"])
        token_count = torch.bincount(
            fit_rows[:, :-1].reshape(-1), minlength=V
        ).float().to("cuda")
        punctuation = base.punctuation_table().to("cuda")
        fit_receipt = json.loads(FIT_RECEIPT.read_text())
        program_receipt = fit_receipt["programs"][PROGRAM_KEY]
        program_path = Path(program_receipt["path"])
        if (file_sha256(program_path) != program_receipt["sha256"]
                or program_path.stat().st_size != program_receipt["bytes"]):
            raise RuntimeError("C512 program changed")
        expected_program = {
            "key": PROGRAM_KEY,
            "path": str(program_path.resolve()),
            "bytes": program_receipt["bytes"],
            "sha256": program_receipt["sha256"],
            "rank": 512,
            "n_centroids": 0,
            "fit_receipt_sha256": file_sha256(FIT_RECEIPT),
        }
        if authority.get("program_contract") != expected_program:
            raise RuntimeError("authority-bound C512 program/fit chain changed")
        observed_hashes = verify_preflight_artifacts(authority, frozen_rows, program_path)
        logit_scale, _ = inherited_currency()
        model = base.load_authority_model(authority)
        blocks = model.transformer.h
        mlp0 = blocks[0].mlp
        original = mlp0.Down
        original_forward = original.forward
        proxy = C512Down(load_program(program_path)).to("cuda")
        original_poison_calls = 0

        def poisoned_original(*args: Any, **kwargs: Any) -> Any:
            nonlocal original_poison_calls
            original_poison_calls += 1
            raise RuntimeError("candidate reached poisoned original MLP0 Down")

        counter = PhaseSiteCounter(blocks, contract["exact_phase_site_call_counts"])
        canary_calls = poison_canary(mlp0, original, original_forward)
        prepared = prepare_domain(
            domain, model, blocks, mlp0, original, original_forward, proxy,
            poisoned_original, counter, token_count, punctuation,
            frequency_median, norm_median,
        )
        replay = {
            parent: {
                "raw_logits_max_abs": 0.0,
                "capped_logits_max_abs": 0.0,
                "ce_abs": 0.0,
            }
            for parent in (
                "exact_live", "candidate_live", "exact_mlp2_omit",
                "candidate_mlp2_omit",
            )
        }
        ledgers, diagnostics = evaluate_domain(
            domain, prepared, model, blocks, mlp0, original, original_forward,
            proxy, poisoned_original, counter, logit_scale, replay,
            token_count, punctuation, frequency_median, norm_median,
        )
        install_native(mlp0, original, original_forward)
        tolerances = contract["parent_replay_tolerances"]
        for values in replay.values():
            values["passes"] = all(
                values[key] <= tolerances[key]
                for key in ("raw_logits_max_abs", "capped_logits_max_abs", "ce_abs")
            )
        carried_max = prepared["carried_state_identity_max"]
        carried_tolerance = contract["carried_state_identity_tolerance"]
        control_checks = {
            **prepared["control_checks"],
            "control_realization_sha256": prepared["control_realization_sha256"],
            "native_control_norm_max_abs": diagnostics["native_control_norm_max_abs"],
        }
        control_checks["passes"] = bool(
            all(prepared["control_checks"].values())
            and diagnostics["native_control_norm_max_abs"]
            <= contract["native_control_norm_tolerance"]
        )
        integrity = {
            "call_counts": {
                "candidate_original_down_calls": original_poison_calls,
                "poison_canary_calls": canary_calls,
                "c512_proxy_calls": proxy.calls,
            },
            "phase_site_call_counts": counter.counts,
            "observed_hashes": observed_hashes,
            "parent_replay": replay,
            "same_realization_delta": {
                "max_abs": diagnostics["same_realization_delta_max_abs"],
                "passes": diagnostics["same_realization_delta_max_abs"]
                <= contract["same_realization_delta_tolerance"],
            },
            "carried_state_identity": {
                "x0_max_abs": carried_max["x0"],
                "v1_max_abs": carried_max["v1"],
                "passes": max(carried_max.values()) <= carried_tolerance,
            },
            "control_checks": control_checks,
            "scoring_currency": {
                "centered_capped_logit_rms": logit_scale,
                "matches_authority": (
                    logit_scale == contract["inherited_centered_capped_logit_rms"]
                ),
            },
        }
        if not validate_integrity(authority, integrity):
            raise RuntimeError(f"runtime integrity differs from authority: {integrity}")
        coverage = coverage_by_wave(
            prepared["valid"], domain["unit_ids"], identity["wave_labels"]
        )
        raw = {
            "schema_version": 1,
            "experiment": "mlp0_c512_mlp2_compensation_v1",
            "authority": authority,
            "authority_file_sha256": file_sha256(AUTHORITY),
            "unit_identity": identity,
            "rows": {
                "receipt_sha256": file_sha256(ROW_RECEIPT),
                "tensor_sha256": tensor_sha256(frozen_rows),
            },
            "program": program_receipt,
            "fit_frozen_centered_capped_logit_rms": logit_scale,
            "inherited_currency_contract": inherited_currency_contract()[0],
            "cell_names": CELL_NAMES,
            "coverage": coverage,
            "integrity": integrity,
            "control_realization_receipt": prepared[
                "control_realization_receipt"
            ],
            "sufficient_statistics": {
                contrast: {
                    metric: {
                        "sums": values["sums"].tolist(),
                        "counts": values["counts"].tolist(),
                    }
                    for metric, values in metrics.items()
                }
                for contrast, metrics in ledgers.items()
            },
        }
        raw["inference"] = score_result(raw)
        raw["runtime_s"] = time.time() - started
        write_json_atomic(raw, OUT)
        print(json.dumps({
            "coverage": coverage,
            "integrity": integrity,
            "decisions": raw["inference"]["decisions"],
            "runtime_s": raw["runtime_s"],
        }, indent=2), flush=True)
        print(f"wrote {OUT}", flush=True)
    finally:
        if counter is not None:
            counter.close()
        if original is not None and original_forward is not None and mlp0 is not None:
            try:
                install_native(mlp0, original, original_forward)
            except Exception:
                pass


def authoritative_entry() -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = acquire_lock()
    try:
        if OUT.exists() or FAILURE.exists():
            raise RuntimeError("MLP2 result namespace was spent before owned execution")
        try:
            main()
        except BaseException as error:
            if not OUT.exists() and not FAILURE.exists():
                write_json_create_only({
                    "schema_version": 1,
                    "status": "failed_closed_without_scientific_result",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "authority_sha256": (
                        file_sha256(AUTHORITY) if AUTHORITY.exists() else None
                    ),
                }, FAILURE)
            raise
    finally:
        release_owned_lock(lock_fd)


if __name__ == "__main__":
    authoritative_entry()
