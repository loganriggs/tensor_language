#!/usr/bin/env python3
"""Source-closed fit-only numerical transaction for Block-3 family F.

The authority is published before the n480 tensor, parent tensor payloads, checkpoint
weights, or any teacher/student outcome is deserialized.  Validation and final loaders
are neither imported nor referenced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
sys.path.insert(0, str(ROOT))

import bilin18_observed_model_facade as facade
import block3_consequence_family_f_call_ledger as call_contract
import block3_consequence_family_f_lifecycle as life
import block3_consequence_fit as core
import collect_block3_native_gate_fit_v1 as collector
import native_gate_subset as subset


HERE = life.HERE
AUTHORITY = life.AUTHORITY
PROGRAMS = life.PROGRAMS
RESULTS = life.RESULTS
RECEIPT = life.RECEIPT
FAILURE = life.FAILURE
LOCK = life.LOCK
RUNNER = HERE / "fit_block3_consequence_family_f_v1.py"
DEVICE = "cuda"
WIDTH = 1152
LAYER = 3
MODEL_TOKENS = 256
POSITION_START = 64
POSITION_STOP = 256
LOGICAL_BATCH = 8
MICROBATCH = 2
PREFILTER = 1024
BUDGETS = (256, 512)
RANDOM_SEED = 2026082907
RIDGE = 1e-6
MAX_WALL_SECONDS = 45 * 60
MAX_ALLOCATED_CUDA_BYTES = 30 * 1024 ** 3

SOURCE_PATHS = tuple(dict.fromkeys((*life.SOURCE_PATHS,
    "basis_aligned/polynomial_causal/BLOCK3_NATIVE_GATE_SUBSET_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/fit_block3_consequence_family_f_v1.py",
    "basis_aligned/polynomial_causal/test_fit_block3_consequence_family_f_v1.py",
    "basis_aligned/polynomial_causal/block3_consequence_family_f_call_ledger.py",
    "basis_aligned/polynomial_causal/test_block3_consequence_family_f_call_ledger.py",
)))


def logical_sha256(value: Any) -> str:
    return life.logical_sha256(value)


def file_sha256(path: Path) -> str:
    return collector.file_sha256(path)


def require_resource_ceiling(started: float) -> tuple[float, int]:
    """Enforce the preregistered hard ceiling throughout, not only at return."""

    elapsed = time.time() - started
    maximum_allocated = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    if elapsed > MAX_WALL_SECONDS or maximum_allocated > MAX_ALLOCATED_CUDA_BYTES:
        raise RuntimeError(
            "family-F resource ceiling exceeded: "
            f"seconds={elapsed:.3f}, bytes={maximum_allocated}"
        )
    return elapsed, maximum_allocated


def source_closure() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"family-F execution source is not committed: {relative}")
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"live family-F execution source differs from commit: {relative}")
        hashes[relative] = digest
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": logical_sha256(body)}


def verify_source_closure(binding: Mapping[str, Any]) -> None:
    if set(binding) != {"commit", "paths", "sha256"} or set(
        binding["paths"]
    ) != set(SOURCE_PATHS) or logical_sha256({
        "commit": binding["commit"], "paths": binding["paths"],
    }) != binding["sha256"]:
        raise RuntimeError("family-F execution source closure is malformed")
    for relative, digest in binding["paths"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"family-F execution source drift: {relative}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", binding["commit"], "origin/main"],
        cwd=ROOT, check=True,
    )


def authority(
    source: Mapping[str, Any], prior: Mapping[str, Any], rows: Mapping[str, Any],
    checkpoint: facade.CheckpointReceipt,
) -> dict[str, Any]:
    protocol = {**life.protocol(), "authorized_for_fit_execution": True}
    body = {
        "schema": "block3_consequence_family_f_v1_authority",
        "status": "frozen_before_any_family_f_row_tensor_parent_tensor_checkpoint_weight_or_outcome_load",
        "source_closure": dict(source),
        "prior_artifact_binding": dict(prior),
        "row_binding": dict(rows),
        "checkpoint": asdict(checkpoint),
        "protocol": protocol,
        "output_paths": {
            "programs": str(PROGRAMS), "results": str(RESULTS),
            "receipt": str(RECEIPT), "failure": str(FAILURE),
        },
        "authorized_for_fit_execution": True,
        "authorized_for_validation": False,
        "authorized_for_final": False,
        "authorized_for_global_ledger_credit": False,
    }
    return {**body, "authority_sha256": logical_sha256(body)}


def verify_frozen_inputs(
    source: Mapping[str, Any], prior: Mapping[str, Any], rows: Mapping[str, Any],
    checkpoint: facade.CheckpointReceipt,
) -> None:
    verify_source_closure(source)
    life.verify_prior_artifact_binding(prior)
    life.verify_row_binding(rows)
    if facade.validate_snapshot(verify_weights_sha256=True) != checkpoint:
        raise RuntimeError("family-F execution checkpoint drift")


def load_rows_after_authority(binding: Mapping[str, Any]) -> torch.Tensor:
    if not AUTHORITY.is_file():
        raise RuntimeError("family-F rows cannot load before authority publication")
    before = file_sha256(life.ROWS)
    raw = torch.load(life.ROWS, map_location="cpu", weights_only=True)
    rows = raw["rows"] if isinstance(raw, dict) and set(raw) == {"rows"} else raw
    if before != life.ROWS_FILE_SHA256 or file_sha256(life.ROWS) != before or not (
        torch.is_tensor(rows)
    ) or tuple(rows.shape) != (life.ROW_COUNT, life.ROW_WIDTH) or rows.dtype != torch.long or (
        collector.tensor_sha256(rows) != life.ROWS_RAW_SHA256
    ) or binding.get("row_file_sha256") != before:
        raise RuntimeError("family-F rows failed post-authority replay")
    return rows.contiguous()


def load_parent_tensors_after_authority(
    frozen_authority: Mapping[str, Any], *, started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the two tensor parents under hash-before/load/hash-after guards."""

    if not AUTHORITY.is_file():
        raise RuntimeError("family-F parent tensors cannot load before authority publication")
    published = json.loads(AUTHORITY.read_text())
    if published != dict(frozen_authority):
        raise RuntimeError("family-F authority changed before parent tensor load")
    prior = frozen_authority.get("prior_artifact_binding", {})
    expected_hashes = prior.get("file_sha256s", {})
    paths = (collector.PAYLOAD, life.PRIOR_PATHS[4])
    before: dict[Path, str] = {}
    for path in paths:
        relative = str(path.relative_to(ROOT))
        before[path] = file_sha256(path)
        if expected_hashes.get(relative) != before[path]:
            raise RuntimeError(f"family-F parent tensor binding changed: {relative}")
    require_resource_ceiling(started)
    parent_payload = torch.load(paths[0], map_location="cpu", weights_only=True)
    raw_a_programs = torch.load(paths[1], map_location="cpu", weights_only=True)
    require_resource_ceiling(started)
    for path in paths:
        if file_sha256(path) != before[path]:
            raise RuntimeError(f"family-F parent tensor changed during load: {path.name}")
    if parent_payload.get("authority_sha256") != prior.get(
        "collector_authority_sha256"
    ) or parent_payload.get("prefilter_indices", torch.empty(0)).shape != (
        PREFILTER,
    ) or raw_a_programs.get("fit_authority_sha256") != prior.get(
        "fit_authority_sha256"
    ):
        raise RuntimeError("family-F post-authority parent tensor joins failed")
    return parent_payload, raw_a_programs


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    tensors = sorted(
        list(model.named_parameters()) + list(model.named_buffers()), key=lambda item: item[0],
    )
    for name, value in tensors:
        tensor = value.detach().cpu().contiguous()
        header = json.dumps({
            "name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        }, sort_keys=True, separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(slots=True)
class Prefix:
    post: torch.Tensor
    x0: torch.Tensor
    first_value: torch.Tensor
    z: torch.Tensor
    native_write: torch.Tensor

    def take(self, start: int, stop: int) -> "Prefix":
        return Prefix(*(
            value[start:stop] for value in (
                self.post, self.x0, self.first_value, self.z, self.native_write,
            )
        ))


def prefix_to_mlp3(model: torch.nn.Module, tokens: torch.Tensor, calls: Any,
                   *, phase: str, arm: str, donor: bool = False) -> Prefix:
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (WIDTH,))
        x0, first_value = x, None
        for site in range(LAYER + 1):
            block = model.transformer.h[site]
            h = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, first_value = block.attn(F.rms_norm(h, (WIDTH,)), first_value)
            post = h + attention
            z = F.rms_norm(post, (WIDTH,))
            if site == LAYER:
                native_write = block.mlp(z)
                calls.record_prefix(phase, arm, donor=donor)
                return Prefix(post, x0, first_value, z, native_write)
            x = post + block.mlp(z)
    raise AssertionError("family-F prefix did not reach MLP3")


def suffix_raw_logits(
    model: torch.nn.Module, prefix: Prefix, write: torch.Tensor, calls: Any,
    *, phase: str, arm: str, teacher: bool,
) -> torch.Tensor:
    x = prefix.post + write
    x0 = prefix.x0
    first_value = prefix.first_value
    for site in range(4, 18):
        block = model.transformer.h[site]
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, first_value = block.attn(F.rms_norm(x, (WIDTH,)), first_value)
        x = x + attention
        x = x + block.mlp(F.rms_norm(x, (WIDTH,)))
    raw = model.lm_head(F.rms_norm(x, (WIDTH,)))
    if teacher:
        calls.record_teacher_suffix(phase, arm)
    else:
        calls.record_student_suffix(phase, arm)
    return raw


def full_raw_logits(model: torch.nn.Module, tokens: torch.Tensor, calls: Any) -> torch.Tensor:
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(tokens), (WIDTH,))
        x0, first_value = x, None
        for block in model.transformer.h:
            x, first_value = block(x, first_value, x0)
        raw = model.lm_head(F.rms_norm(x, (WIDTH,)))
        calls.record_outer_replay()
        return raw


def scored_logits(raw: torch.Tensor) -> torch.Tensor:
    return core.softcap_scored_raw_logits(
        raw, start=POSITION_START, stop=POSITION_STOP,
    )


def softcap_logits(raw: torch.Tensor) -> torch.Tensor:
    """Apply the model's output softcap to an already selected raw-logit slice."""

    return (30.0 * torch.tanh(raw / 30.0)).float()


def _program_payload(program: subset.NativeGateSubsetProgram) -> dict[str, torch.Tensor]:
    return {
        "indices": program.indices.detach().cpu().contiguous(),
        "left": program.left.detach().cpu().contiguous(),
        "right": program.right.detach().cpu().contiguous(),
        "decoder": program.decoder.detach().cpu().contiguous(),
        "bias": program.bias.detach().cpu().contiguous(),
    }


def _materialize_program(
    payload: Mapping[str, torch.Tensor], device: str = DEVICE,
) -> subset.NativeGateSubsetProgram:
    required = {"indices", "left", "right", "decoder", "bias"}
    if set(payload) != required:
        raise RuntimeError("family-F stored program schema changed")
    return subset.NativeGateSubsetProgram(**{
        name: value.detach().to(device).contiguous() for name, value in payload.items()
    })


def _teacher_for_logical_batch(
    *, model: torch.nn.Module, rows: torch.Tensor, start: int, arm: str,
    donor_rows: torch.Tensor, calls: Any, phase: str,
) -> tuple[Prefix, torch.Tensor, torch.Tensor]:
    indices = torch.arange(start, start + LOGICAL_BATCH, dtype=torch.long)
    tokens = rows[indices, :MODEL_TOKENS].to(DEVICE)
    target_prefix = prefix_to_mlp3(model, tokens, calls, phase=phase, arm=arm)
    if arm == "teacher_document_derangement":
        donor = donor_rows[indices]
        donor_tokens = rows[donor, :MODEL_TOKENS].to(DEVICE)
        teacher_prefix = prefix_to_mlp3(
            model, donor_tokens, calls, phase=phase, arm=arm, donor=True,
        )
    else:
        teacher_prefix = target_prefix
    with torch.no_grad():
        teacher_raw = suffix_raw_logits(
            model, teacher_prefix, teacher_prefix.native_write, calls,
            phase=phase, arm=arm, teacher=True,
        )
        teacher_raw = teacher_raw[:, POSITION_START:POSITION_STOP].detach()
        teacher = softcap_logits(teacher_raw).detach()
        if arm == "teacher_row_reversal":
            teacher_raw = teacher_raw.flip(0).contiguous()
            teacher = teacher.flip(0).contiguous()
    return target_prefix, teacher_raw, teacher


def _score_epoch_trace(
    *, epoch: int, document_kl: float, row_kl: float, scores: torch.Tensor,
    gradient_norm_max: float,
) -> dict[str, Any]:
    frozen = scores.detach().double()
    trace = core.ScoreTrace(
        epoch=epoch, document_kl=document_kl, row_kl=row_kl,
        score_min=float(frozen.min()), score_max=float(frozen.max()),
        score_sum=float(frozen.sum()),
        saturated_zero=float((frozen <= 1e-6).double().mean()),
        saturated_one=float((frozen >= 1 - 1e-6).double().mean()),
        gradient_norm_max=gradient_norm_max,
    )
    return asdict(trace)


def validate_native_replay(
    ordinary_full_raw: torch.Tensor, teacher_raw: torch.Tensor,
    teacher: torch.Tensor,
) -> dict[str, float]:
    """Compare autonomous and ordinary execution before and after the softcap."""

    ordinary_raw = ordinary_full_raw[:, POSITION_START:POSITION_STOP]
    if ordinary_raw.shape != teacher_raw.shape or teacher.shape != teacher_raw.shape:
        raise RuntimeError("family-F native replay tensor shapes changed")
    raw_difference = ordinary_raw - teacher_raw
    raw_absolute = float(raw_difference.abs().max())
    raw_relative = raw_absolute / max(
        float(teacher_raw.abs().max()), torch.finfo(torch.float32).tiny,
    )
    ordinary = softcap_logits(ordinary_raw)
    difference = ordinary - teacher
    absolute = float(difference.abs().max())
    relative = absolute / max(
        float(teacher.abs().max()), torch.finfo(torch.float32).tiny,
    )
    self_kl = float(core.teacher_kl_by_row(teacher, teacher).abs().max())
    replay = {
        "raw_max_absolute": raw_absolute,
        "raw_max_relative": raw_relative,
        "max_absolute": absolute,
        "max_relative": relative,
        "teacher_self_kl_max": self_kl,
    }
    if raw_relative > core.REPLAY_RELATIVE_LIMIT or (
        relative > core.REPLAY_RELATIVE_LIMIT
    ) or self_kl > 2e-7:
        raise RuntimeError("family-F native autonomous suffix replay failed")
    return replay


def fit_score_arm(
    *, model: torch.nn.Module, rows: torch.Tensor, row_to_document: torch.Tensor,
    row_weights: torch.Tensor, donor_rows: torch.Tensor,
    balanced_left: torch.Tensor, balanced_right: torch.Tensor,
    native_down: torch.Tensor, native_bias: torch.Tensor,
    prefilter_indices: torch.Tensor, arm: str, calls: Any,
    ordinary_replay: bool, started: float,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, float] | None]:
    if arm not in {
        "teacher", "teacher_row_reversal", "teacher_document_derangement",
    }:
        raise ValueError("unregistered family-F score arm")
    scores = torch.nn.Parameter(torch.full(
        (PREFILTER,), 0.5, dtype=torch.float64, device=DEVICE,
    ))
    optimizer = torch.optim.Adam(
        [scores], lr=0.02, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )
    traces: list[dict[str, Any]] = []
    replay: dict[str, float] | None = None
    phase = "score_fit"
    for epoch in range(8):
        epoch_document_kl = 0.0
        epoch_row_kl = 0.0
        epoch_rows = 0
        epoch_grad_max = 0.0
        for start in range(0, life.ROW_COUNT, LOGICAL_BATCH):
            require_resource_ceiling(started)
            prefix, teacher_raw, teacher = _teacher_for_logical_batch(
                model=model, rows=rows, start=start, arm=arm, donor_rows=donor_rows,
                calls=calls, phase=phase,
            )
            if ordinary_replay and epoch == 0 and start == 0:
                ordinary_raw = full_raw_logits(
                    model, rows[start:start + LOGICAL_BATCH, :MODEL_TOKENS].to(DEVICE), calls,
                )
                replay = validate_native_replay(ordinary_raw, teacher_raw, teacher)
                del ordinary_raw
            logical_row_losses: list[torch.Tensor] = []
            closures = []
            for micro_start in range(0, LOGICAL_BATCH, MICROBATCH):
                micro_stop = micro_start + MICROBATCH
                def loss_closure(micro_start=micro_start, micro_stop=micro_stop):
                    micro_prefix = prefix.take(micro_start, micro_stop)
                    write = core.consequence_score_write(
                        micro_prefix.z, balanced_left, balanced_right, native_down,
                        native_bias, prefilter_indices, scores,
                    )
                    student_raw = suffix_raw_logits(
                        model, micro_prefix, write, calls, phase=phase, arm=arm,
                        teacher=False,
                    )
                    student = scored_logits(student_raw)
                    row_loss = core.teacher_kl_by_row(
                        teacher[micro_start:micro_stop], student,
                    )
                    logical_row_losses.append(row_loss.detach())
                    global_rows = slice(start + micro_start, start + micro_stop)
                    return core.document_balanced_batch_loss(
                        row_loss, row_weights[global_rows].to(DEVICE),
                        document_count=life.DOCUMENT_COUNT,
                    )
                closures.append(loss_closure)
            gradient_norm = core.logical_batch_adam_closure_step(
                optimizer, [scores], closures, max_grad_norm=1.0,
            )
            with torch.no_grad():
                scores.copy_(core.project_capped_simplex(scores, 512))
            calls.record_optimizer_step(phase, arm, backwards=4, projection=True)
            all_row_losses = torch.cat(logical_row_losses).double().cpu()
            weights = row_weights[start:start + LOGICAL_BATCH]
            epoch_document_kl += float((all_row_losses * weights).sum()) / life.DOCUMENT_COUNT
            epoch_row_kl += float(all_row_losses.sum())
            epoch_rows += len(all_row_losses)
            epoch_grad_max = max(epoch_grad_max, gradient_norm)
            del prefix, teacher, closures, logical_row_losses, all_row_losses
            del teacher_raw
            require_resource_ceiling(started)
        traces.append(_score_epoch_trace(
            epoch=epoch + 1, document_kl=epoch_document_kl,
            row_kl=epoch_row_kl / epoch_rows, scores=scores,
            gradient_norm_max=epoch_grad_max,
        ))
    return scores.detach().cpu().double().contiguous(), traces, replay


def _affine_write(
    program: subset.NativeGateSubsetProgram, z: torch.Tensor,
    scale: torch.Tensor, correction: torch.Tensor,
) -> torch.Tensor:
    """Execute exactly the float32 program that publication would fold."""

    left = F.linear(z, program.left)
    right = F.linear(z, program.right)
    features = left * right
    decoder = program.decoder * scale.to(program.decoder)
    bias = program.bias + correction.to(program.bias)
    output = F.linear(features, decoder)
    shape = (1,) * (output.ndim - 1) + (program.width,)
    return output + bias.reshape(shape)


def fit_affine_arm(
    *, model: torch.nn.Module, rows: torch.Tensor, row_weights: torch.Tensor,
    donor_rows: torch.Tensor, program: subset.NativeGateSubsetProgram,
    arm: str, calls: Any, started: float,
) -> tuple[subset.NativeGateSubsetProgram, dict[str, Any], list[dict[str, Any]]]:
    scale = torch.nn.Parameter(torch.ones((), dtype=torch.float64, device=DEVICE))
    correction = torch.nn.Parameter(torch.zeros(WIDTH, dtype=torch.float64, device=DEVICE))
    optimizer = torch.optim.Adam(
        [scale, correction], lr=0.005, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0,
    )
    traces: list[dict[str, Any]] = []
    phase = "affine_fit"
    for epoch in range(4):
        epoch_document_kl = 0.0
        epoch_row_kl = 0.0
        epoch_rows = 0
        epoch_grad_max = 0.0
        for start in range(0, life.ROW_COUNT, LOGICAL_BATCH):
            require_resource_ceiling(started)
            prefix, _teacher_raw, teacher = _teacher_for_logical_batch(
                model=model, rows=rows, start=start, arm=arm, donor_rows=donor_rows,
                calls=calls, phase=phase,
            )
            logical_row_losses: list[torch.Tensor] = []
            closures = []
            for micro_start in range(0, LOGICAL_BATCH, MICROBATCH):
                micro_stop = micro_start + MICROBATCH
                def loss_closure(micro_start=micro_start, micro_stop=micro_stop):
                    micro_prefix = prefix.take(micro_start, micro_stop)
                    write = _affine_write(program, micro_prefix.z, scale, correction)
                    student_raw = suffix_raw_logits(
                        model, micro_prefix, write, calls, phase=phase, arm=arm,
                        teacher=False,
                    )
                    row_loss = core.teacher_kl_by_row(
                        teacher[micro_start:micro_stop], scored_logits(student_raw),
                    )
                    logical_row_losses.append(row_loss.detach())
                    global_rows = slice(start + micro_start, start + micro_stop)
                    return core.document_balanced_batch_loss(
                        row_loss, row_weights[global_rows].to(DEVICE),
                        document_count=life.DOCUMENT_COUNT,
                    )
                closures.append(loss_closure)
            gradient_norm = core.logical_batch_adam_closure_step(
                optimizer, [scale, correction], closures, max_grad_norm=1.0,
            )
            calls.record_optimizer_step(phase, arm, backwards=4, projection=False)
            all_row_losses = torch.cat(logical_row_losses).double().cpu()
            weights = row_weights[start:start + LOGICAL_BATCH]
            epoch_document_kl += float((all_row_losses * weights).sum()) / life.DOCUMENT_COUNT
            epoch_row_kl += float(all_row_losses.sum())
            epoch_rows += len(all_row_losses)
            epoch_grad_max = max(epoch_grad_max, gradient_norm)
            del prefix, teacher, closures, logical_row_losses, all_row_losses
            del _teacher_raw
            require_resource_ceiling(started)
        traces.append({
            "epoch": epoch + 1, "document_kl": epoch_document_kl,
            "row_kl": epoch_row_kl / epoch_rows,
            "gradient_norm_max": epoch_grad_max,
            "scale": float(scale.detach()),
            "correction_rms": float(correction.detach().square().mean().sqrt()),
            "correction_norm": float(correction.detach().norm()),
        })
    folded = core.fold_affine_calibration(
        program, scale.detach().float(), correction.detach().float(),
    )
    parameters = {
        "scale_float64": float(scale.detach()),
        "correction_float64": correction.detach().cpu().double().contiguous(),
    }
    return folded, parameters, traces


def build_programs(
    *, left: torch.Tensor, right: torch.Tensor, native_down: torch.Tensor,
    native_bias: torch.Tensor, prefilter_indices: torch.Tensor,
    gram: torch.Tensor, cross: torch.Tensor, permuted_cross: torch.Tensor,
    scores: Mapping[str, torch.Tensor],
) -> tuple[
    dict[str, subset.NativeGateSubsetProgram], dict[str, torch.Tensor],
]:
    supports_by_arm = {
        arm: core.stable_nested_supports(score, prefilter_indices.cpu(), BUDGETS)
        for arm, score in scores.items()
    }
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    random_local = torch.randperm(PREFILTER, generator=generator)
    random_supports = {
        budget: prefilter_indices.cpu()[random_local[:budget]].clone()
        for budget in BUDGETS
    }
    programs: dict[str, subset.NativeGateSubsetProgram] = {}
    for budget in BUDGETS:
        real_support = supports_by_arm["teacher"][budget]
        definitions = {
            f"real_F_post_refit_k{budget}": (real_support, cross),
            f"random_post_refit_k{budget}": (random_supports[budget], cross),
            f"same_support_permuted_cross_post_refit_k{budget}": (
                real_support, permuted_cross,
            ),
            f"row_reversal_selector_post_refit_k{budget}": (
                supports_by_arm["teacher_row_reversal"][budget], cross,
            ),
            f"document_derangement_selector_post_refit_k{budget}": (
                supports_by_arm["teacher_document_derangement"][budget], cross,
            ),
        }
        for name, (support, fit_cross) in definitions.items():
            programs[name] = core.refit_joint_program(
                left=left, right=right, bias=native_bias,
                prefilter_indices=prefilter_indices.cpu(), gram=gram, cross=fit_cross,
                global_support=support, relative_ridge=RIDGE,
            )
        decoder = native_down[:, real_support.to(native_down.device)]
        programs[f"real_F_binary_native_down_k{budget}"] = subset.build_program(
            left, right, native_bias, real_support.to(left.device), decoder,
        )
    supports = {
        f"{arm}_k{budget}": support.clone()
        for arm, arm_supports in supports_by_arm.items()
        for budget, support in arm_supports.items()
    }
    supports.update({
        f"random_k{budget}": support for budget, support in random_supports.items()
    })
    return programs, supports


def _stacked_fit_nrmse(
    program: subset.NativeGateSubsetProgram, prefilter_indices: torch.Tensor,
    gram: torch.Tensor, cross: torch.Tensor, write_energy: torch.Tensor,
) -> float:
    lookup = {int(gate): local for local, gate in enumerate(prefilter_indices.tolist())}
    local = torch.tensor([lookup[int(gate)] for gate in program.indices.detach().cpu().tolist()])
    selected_gram = gram[local][:, local]
    selected_cross = cross[local]
    decoder_t = program.decoder.detach().cpu().double().T
    energy = float(write_energy)
    sse = energy - 2 * float((decoder_t * selected_cross).sum()) + float(
        (decoder_t * (selected_gram @ decoder_t)).sum()
    )
    if sse < -1e-9 * max(energy, 1.0):
        raise RuntimeError("family-F stacked fit SSE is materially negative")
    return math.sqrt(max(sse, 0.0) / energy)


def deployed_polarization_replay(
    program: subset.NativeGateSubsetProgram,
) -> dict[str, float]:
    coordinates = torch.linspace(
        -1.0, 1.0, 3 * program.width, dtype=program.left.dtype,
        device=program.left.device,
    ).reshape(3, program.width)
    u = coordinates
    v = 0.375 * coordinates.flip(-1)
    with torch.no_grad():
        direct = program.write(u + v)
        typed = sum(program.terms(u, v).values()) + program.bias
    absolute = float((direct - typed).abs().max())
    relative = absolute / max(float(typed.abs().max()), torch.finfo(torch.float32).tiny)
    if not math.isfinite(relative) or relative > core.REPLAY_RELATIVE_LIMIT:
        raise RuntimeError("family-F deployed direct/polarized replay failed")
    return {"max_absolute": absolute, "max_relative": relative}


def report_fit_arms(
    *, model: torch.nn.Module, rows: torch.Tensor, row_weights: torch.Tensor,
    calls: call_contract.FamilyFCallLedger,
    balanced_left: torch.Tensor, balanced_right: torch.Tensor,
    native_down: torch.Tensor, native_bias: torch.Tensor,
    prefilter_indices: torch.Tensor, teacher_scores: torch.Tensor,
    programs: Mapping[str, subset.NativeGateSubsetProgram], started: float,
) -> dict[str, dict[str, float]]:
    report_names = call_contract.REPORT_STUDENT_ARMS
    if set(report_names) != set(programs) | {"continuous_teacher_F1"}:
        raise RuntimeError("family-F reporting arm registry differs from programs")
    accumulators = {
        arm: {"weighted_kl": 0.0, "row_kl": 0.0, "sse": 0.0, "energy": 0.0}
        for arm in report_names
    }
    for start in range(0, life.ROW_COUNT, LOGICAL_BATCH):
        require_resource_ceiling(started)
        tokens = rows[start:start + LOGICAL_BATCH, :MODEL_TOKENS].to(DEVICE)
        prefix = prefix_to_mlp3(
            model, tokens, calls, phase="postfit_report",
            arm=call_contract.REPORT_SHARED_ARM,
        )
        with torch.no_grad():
            teacher = scored_logits(suffix_raw_logits(
                model, prefix, prefix.native_write, calls, phase="postfit_report",
                arm=call_contract.REPORT_SHARED_ARM, teacher=True,
            )).detach()
            for arm in report_names:
                if arm == "continuous_teacher_F1":
                    write = core.consequence_score_write(
                        prefix.z, balanced_left, balanced_right, native_down,
                        native_bias, prefilter_indices, teacher_scores.to(DEVICE),
                    )
                else:
                    write = programs[arm].write(prefix.z)
                student = scored_logits(suffix_raw_logits(
                    model, prefix, write, calls, phase="postfit_report",
                    arm=arm, teacher=False,
                ))
                row_kl = core.teacher_kl_by_row(teacher, student).double().cpu()
                weights = row_weights[start:start + LOGICAL_BATCH]
                accumulators[arm]["weighted_kl"] += float((row_kl * weights).sum())
                accumulators[arm]["row_kl"] += float(row_kl.sum())
                local_error = (
                    write[:, POSITION_START:POSITION_STOP]
                    - prefix.native_write[:, POSITION_START:POSITION_STOP]
                ).double()
                local_energy = (
                    prefix.native_write[:, POSITION_START:POSITION_STOP]
                    - native_bias.reshape(1, 1, -1)
                ).double()
                accumulators[arm]["sse"] += float(local_error.square().sum())
                accumulators[arm]["energy"] += float(local_energy.square().sum())
        del prefix, teacher
        require_resource_ceiling(started)
    return {
        arm: {
            "document_balanced_teacher_kl": values["weighted_kl"] / life.DOCUMENT_COUNT,
            "row_mean_teacher_kl": values["row_kl"] / life.ROW_COUNT,
            "summed_write_nrmse": math.sqrt(values["sse"] / values["energy"]),
        }
        for arm, values in accumulators.items()
    }


def reconstruct_programs_from_sealed_parents(
    *, value: Mapping[str, Any], left: torch.Tensor, right: torch.Tensor,
    native_down: torch.Tensor, native_bias: torch.Tensor,
    parent_payload: Mapping[str, Any], raw_a_programs: Mapping[str, Any],
) -> tuple[dict[str, subset.NativeGateSubsetProgram], dict[str, torch.Tensor]]:
    """Independently rebuild every published array from frozen scores and parents."""

    programs, supports = build_programs(
        left=left, right=right, native_down=native_down, native_bias=native_bias,
        prefilter_indices=parent_payload["prefilter_indices"].long().contiguous(),
        gram=parent_payload["prefilter_gram"].double(),
        cross=parent_payload["prefilter_cross"].double(),
        permuted_cross=parent_payload["prefilter_permuted_cross"].double(),
        scores=value["scores"],
    )
    a_payload = raw_a_programs.get("programs", {}).get("activation_selected_k512")
    if not isinstance(a_payload, Mapping):
        raise RuntimeError("family-F sealed family-A K512 program is absent")
    programs["family_A_uncalibrated_k512"] = _materialize_program(
        a_payload, device=str(left.device),
    )
    affine_bases = {
        "teacher_F_k512": "real_F_post_refit_k512",
        "family_A_k512": "family_A_uncalibrated_k512",
        "random_k512": "random_post_refit_k512",
        "same_support_permuted_cross_k512": (
            "same_support_permuted_cross_post_refit_k512"
        ),
    }
    for arm, base in affine_bases.items():
        parameters = value["affine_parameters"][arm]
        programs[f"affine_{arm}"] = core.fold_affine_calibration(
            programs[base], float(parameters["scale_float64"]),
            parameters["correction_float64"].to(left.device).float(),
        )
    return programs, supports


def _require_exact_program_payload(
    observed: Mapping[str, torch.Tensor], expected: subset.NativeGateSubsetProgram,
    *, arm: str,
) -> None:
    expected_payload = _program_payload(expected)
    if set(observed) != set(expected_payload):
        raise RuntimeError(f"family-F reconstructed program schema changed: {arm}")
    for name, tensor in expected_payload.items():
        if not torch.equal(observed[name], tensor):
            raise RuntimeError(
                f"family-F reconstructed program tensor changed: {arm}/{name}"
            )


def semantic_validate_program_artifact(
    value: Mapping[str, Any], *, expected_authority_sha256: str,
    prefilter: torch.Tensor,
    reconstructed_programs: Mapping[str, subset.NativeGateSubsetProgram] | None = None,
    family_a_supports: Mapping[int, torch.Tensor] | None = None,
) -> None:
    required = {
        "schema", "authority_sha256", "scores", "supports", "programs",
        "affine_parameters", "promotive_programs", "nonpromotive_programs",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get(
        "schema"
    ) != "block3_consequence_family_f_v1_programs" or value.get(
        "authority_sha256"
    ) != expected_authority_sha256 or set(value["scores"]) != set(
        call_contract.SCORE_ARMS
    ):
        raise RuntimeError("family-F program artifact schema changed")
    if not torch.is_tensor(prefilter) or prefilter.shape != (PREFILTER,) or (
        prefilter.dtype != torch.long
    ):
        raise RuntimeError("family-F semantic prefilter is malformed")
    prefilter = prefilter.detach().cpu().contiguous()
    expected_support_keys = {
        f"{arm}_k{budget}" for arm in call_contract.SCORE_ARMS for budget in BUDGETS
    } | {f"random_k{budget}" for budget in BUDGETS}
    if set(value["supports"]) != expected_support_keys:
        raise RuntimeError("family-F stored support registry changed")
    for arm, scores in value["scores"].items():
        if not torch.is_tensor(scores) or scores.shape != (PREFILTER,) or scores.dtype != (
            torch.float64
        ) or not bool(torch.isfinite(scores).all()) or float(scores.min()) < 0 or (
            float(scores.max()) > 1
        ) or abs(float(scores.sum()) - 512) > 1e-10:
            raise RuntimeError(f"family-F stored score vector is malformed: {arm}")
        if float((core.project_capped_simplex(scores, 512) - scores).abs().max()) > 2e-10:
            raise RuntimeError(f"family-F stored score projection does not replay: {arm}")
        reconstructed = core.stable_nested_supports(scores, prefilter, BUDGETS)
        for budget in BUDGETS:
            if not torch.equal(
                reconstructed[budget], value["supports"][f"{arm}_k{budget}"],
            ):
                raise RuntimeError("family-F support does not replay score ranking")
    programs = {
        name: _materialize_program(payload, device="cpu")
        for name, payload in value["programs"].items()
    }
    if set(programs) != set(call_contract.REPORT_STUDENT_ARMS) - {"continuous_teacher_F1"}:
        raise RuntimeError("family-F executable program arm set changed")
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    random_order = torch.randperm(PREFILTER, generator=generator)
    for budget in BUDGETS:
        real = programs[f"real_F_post_refit_k{budget}"]
        binary = programs[f"real_F_binary_native_down_k{budget}"]
        permuted = programs[f"same_support_permuted_cross_post_refit_k{budget}"]
        expected_indices = {
            "real": value["supports"][f"teacher_k{budget}"],
            "random": value["supports"][f"random_k{budget}"],
            "reversal": value["supports"][f"teacher_row_reversal_k{budget}"],
            "document": value["supports"][f"teacher_document_derangement_k{budget}"],
        }
        budget_programs = tuple(
            program for name, program in programs.items()
            if f"k{budget}" in name and not name.startswith("affine_")
        )
        if any(program.gates != budget or program.width != WIDTH for program in budget_programs):
            raise RuntimeError("family-F executable program dimensions changed")
        if not torch.equal(
            value["supports"][f"random_k{budget}"], prefilter[random_order[:budget]],
        ) or not torch.equal(real.indices, expected_indices["real"]) or not torch.equal(
            binary.indices, expected_indices["real"]
        ) or not torch.equal(permuted.indices, expected_indices["real"]) or not torch.equal(
            programs[f"random_post_refit_k{budget}"].indices, expected_indices["random"]
        ) or not torch.equal(
            programs[f"row_reversal_selector_post_refit_k{budget}"].indices,
            expected_indices["reversal"],
        ) or not torch.equal(
            programs[f"document_derangement_selector_post_refit_k{budget}"].indices,
            expected_indices["document"],
        ) or real.gates != budget:
            raise RuntimeError("family-F executable support provenance changed")
        uncalibrated = (
            real, binary, permuted, programs[f"random_post_refit_k{budget}"],
            programs[f"row_reversal_selector_post_refit_k{budget}"],
            programs[f"document_derangement_selector_post_refit_k{budget}"],
        )
        if any(not torch.equal(program.bias, real.bias) for program in uncalibrated):
            raise RuntimeError("family-F uncalibrated native bias changed")
        if family_a_supports is not None and budget in family_a_supports and budget == 512 and (
            not torch.equal(
                programs["family_A_uncalibrated_k512"].indices,
                family_a_supports[budget].detach().cpu(),
            )
        ):
            raise RuntimeError("family-F sealed family-A support changed")
    if value["promotive_programs"] != [
        "real_F_post_refit_k256", "real_F_post_refit_k512",
    ] or set(value["promotive_programs"]) & set(value["nonpromotive_programs"]):
        raise RuntimeError("family-F promotive/nonpromotive roles changed")
    if set(value["nonpromotive_programs"]) != set(programs) - set(
        value["promotive_programs"]
    ):
        raise RuntimeError("family-F nonpromotive program coverage changed")
    if set(value["affine_parameters"]) != set(call_contract.AFFINE_ARMS):
        raise RuntimeError("family-F affine parameter registry changed")
    for arm, parameters in value["affine_parameters"].items():
        correction = parameters.get("correction_float64")
        if set(parameters) != {"scale_float64", "correction_float64"} or not math.isfinite(
            parameters["scale_float64"]
        ) or not torch.is_tensor(correction) or correction.shape != (WIDTH,) or (
            correction.dtype != torch.float64
        ) or not bool(torch.isfinite(correction).all()):
            raise RuntimeError(f"family-F affine parameters are malformed: {arm}")
    if reconstructed_programs is not None:
        if set(reconstructed_programs) != set(value["programs"]):
            raise RuntimeError("family-F reconstructed program arm set changed")
        for arm, expected in reconstructed_programs.items():
            _require_exact_program_payload(value["programs"][arm], expected, arm=arm)
    for arm, program in programs.items():
        deployed_polarization_replay(program)


def execute_fit(
    *, frozen_authority: Mapping[str, Any], rows_binding: Mapping[str, Any],
    checkpoint: facade.CheckpointReceipt, calls: call_contract.FamilyFCallLedger,
    started: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Open fit-only tensors after authority and execute the exact frozen schedule."""

    require_resource_ceiling(started)
    rows = load_rows_after_authority(rows_binding)
    parent_payload, raw_a_programs = load_parent_tensors_after_authority(
        frozen_authority, started=started,
    )

    require_resource_ceiling(started)
    model, loaded_checkpoint = facade.load_bilin18(
        device=DEVICE, dtype=torch.float32, verify_weights_sha256=True,
    )
    require_resource_ceiling(started)
    if loaded_checkpoint != checkpoint:
        raise RuntimeError("family-F loaded checkpoint differs from authority")
    torch.cuda.reset_peak_memory_stats()
    model_before = model_state_sha256(model)

    block = model.transformer.h[LAYER]
    left = block.mlp.Left.weight.detach()
    right = block.mlp.Right.weight.detach()
    native_down = block.mlp.Down.weight.detach()
    native_bias = block.mlp.Down_bias.detach()
    balanced_left, balanced_right, _ = collector.balance_product_gauge(left, right)
    prefilter_indices_cpu = parent_payload["prefilter_indices"].long().contiguous()
    prefilter_indices = prefilter_indices_cpu.to(DEVICE)
    gram = parent_payload["prefilter_gram"].double()
    cross = parent_payload["prefilter_cross"].double()
    permuted_cross = parent_payload["prefilter_permuted_cross"].double()
    write_energy = parent_payload["native_typed_write_energy"].double()
    row_to_document = torch.tensor(
        rows_binding["row_to_document"], dtype=torch.long,
    )
    row_weights = core.source_document_row_weights(row_to_document)
    donor_rows = torch.tensor(
        rows_binding["document_deranged_donor_rows"], dtype=torch.long,
    )

    score_values: dict[str, torch.Tensor] = {}
    score_traces: dict[str, list[dict[str, Any]]] = {}
    known_answer: dict[str, float] | None = None
    for arm in call_contract.SCORE_ARMS:
        score, trace, replay = fit_score_arm(
            model=model, rows=rows, row_to_document=row_to_document,
            row_weights=row_weights, donor_rows=donor_rows,
            balanced_left=balanced_left, balanced_right=balanced_right,
            native_down=native_down, native_bias=native_bias,
            prefilter_indices=prefilter_indices, arm=arm, calls=calls,
            ordinary_replay=(arm == "teacher"), started=started,
        )
        score_values[arm] = score
        score_traces[arm] = trace
        if replay is not None:
            known_answer = replay
    if known_answer is None:
        raise RuntimeError("family-F ordinary-model known-answer replay did not run")

    programs, supports = build_programs(
        left=left, right=right, native_down=native_down, native_bias=native_bias,
        prefilter_indices=prefilter_indices_cpu, gram=gram, cross=cross,
        permuted_cross=permuted_cross, scores=score_values,
    )
    a_payload = raw_a_programs.get("programs", {}).get("activation_selected_k512")
    if not isinstance(a_payload, Mapping):
        raise RuntimeError("family-F sealed family-A K512 program is absent")
    programs["family_A_uncalibrated_k512"] = _materialize_program(a_payload)

    affine_parameters: dict[str, dict[str, Any]] = {}
    affine_traces: dict[str, list[dict[str, Any]]] = {}
    affine_bases = {
        "teacher_F_k512": "real_F_post_refit_k512",
        "family_A_k512": "family_A_uncalibrated_k512",
        "random_k512": "random_post_refit_k512",
        "same_support_permuted_cross_k512": (
            "same_support_permuted_cross_post_refit_k512"
        ),
    }
    for arm in call_contract.AFFINE_ARMS:
        folded, parameters, trace = fit_affine_arm(
            model=model, rows=rows, row_weights=row_weights, donor_rows=donor_rows,
            program=programs[affine_bases[arm]], arm=arm, calls=calls,
            started=started,
        )
        programs[f"affine_{arm}"] = folded
        affine_parameters[arm] = parameters
        affine_traces[arm] = trace

    report = report_fit_arms(
        model=model, rows=rows, row_weights=row_weights, calls=calls,
        balanced_left=balanced_left, balanced_right=balanced_right,
        native_down=native_down, native_bias=native_bias,
        prefilter_indices=prefilter_indices,
        teacher_scores=score_values["teacher"], programs=programs, started=started,
    )
    call_receipt = calls.validate_exact()
    model_after = model_state_sha256(model)
    if model_after != model_before:
        raise RuntimeError("family-F model tensors changed during fit")
    if collector.tensor_sha256(rows) != life.ROWS_RAW_SHA256:
        raise RuntimeError("family-F fit row tensor changed during execution")

    promotive = ["real_F_post_refit_k256", "real_F_post_refit_k512"]
    program_names = list(call_contract.REPORT_STUDENT_ARMS)
    program_names.remove("continuous_teacher_F1")
    nonpromotive = [name for name in program_names if name not in promotive]
    program_artifact = {
        "schema": "block3_consequence_family_f_v1_programs",
        "authority_sha256": frozen_authority["authority_sha256"],
        "scores": {name: value.clone() for name, value in score_values.items()},
        "supports": {name: value.cpu().clone() for name, value in supports.items()},
        "programs": {
            name: _program_payload(programs[name]) for name in program_names
        },
        "affine_parameters": affine_parameters,
        "promotive_programs": promotive,
        "nonpromotive_programs": nonpromotive,
    }
    reconstructed_programs, reconstructed_supports = (
        reconstruct_programs_from_sealed_parents(
            value=program_artifact, left=left, right=right,
            native_down=native_down, native_bias=native_bias,
            parent_payload=parent_payload, raw_a_programs=raw_a_programs,
        )
    )
    if set(reconstructed_supports) != set(supports) or any(
        not torch.equal(reconstructed_supports[name], supports[name])
        for name in supports
    ):
        raise RuntimeError("family-F independently reconstructed supports changed")
    family_a_supports = {
        budget: raw_a_programs["programs"][
            f"activation_selected_k{budget}"
        ]["indices"].long().contiguous()
        for budget in BUDGETS
    }
    semantic_validate_program_artifact(
        program_artifact,
        expected_authority_sha256=frozen_authority["authority_sha256"],
        prefilter=prefilter_indices_cpu,
        reconstructed_programs=reconstructed_programs,
        family_a_supports=family_a_supports,
    )

    stacked_nrmse = {
        name: _stacked_fit_nrmse(
            programs[name], prefilter_indices_cpu, gram, cross, write_energy,
        )
        for name in program_names if not name.startswith("affine_")
    }
    prices = {name: core.program_price(programs[name]) for name in program_names}
    polarization_replays = {
        name: deployed_polarization_replay(programs[name]) for name in program_names
    }
    projection_replays = {
        arm: float((
            core.project_capped_simplex(score, 512) - score
        ).abs().max())
        for arm, score in score_values.items()
    }
    if any(value > 2e-10 for value in projection_replays.values()):
        raise RuntimeError("family-F final score projection did not replay")
    postfit_transitions = {
        str(budget): {
            "continuous_F1_document_kl": report[
                "continuous_teacher_F1"
            ]["document_balanced_teacher_kl"],
            "binary_native_down_document_kl": report[
                f"real_F_binary_native_down_k{budget}"
            ]["document_balanced_teacher_kl"],
            "post_refit_document_kl": report[
                f"real_F_post_refit_k{budget}"
            ]["document_balanced_teacher_kl"],
            "binary_minus_continuous_document_kl": report[
                f"real_F_binary_native_down_k{budget}"
            ]["document_balanced_teacher_kl"] - report[
                "continuous_teacher_F1"
            ]["document_balanced_teacher_kl"],
            "refit_minus_binary_document_kl": report[
                f"real_F_post_refit_k{budget}"
            ]["document_balanced_teacher_kl"] - report[
                f"real_F_binary_native_down_k{budget}"
            ]["document_balanced_teacher_kl"],
        }
        for budget in BUDGETS
    }
    score_overlaps: dict[str, Any] = {}
    for budget in BUDGETS:
        real = set(supports[f"teacher_k{budget}"].tolist())
        for comparison in (
            "teacher_row_reversal", "teacher_document_derangement", "random",
        ):
            other = set(supports[f"{comparison}_k{budget}"].tolist())
            score_overlaps[f"teacher_vs_{comparison}_k{budget}"] = {
                "intersection": len(real & other),
                "jaccard": len(real & other) / len(real | other),
            }
        family_a = set(family_a_supports[budget].tolist())
        score_overlaps[f"teacher_vs_family_A_k{budget}"] = {
            "intersection": len(real & family_a),
            "jaccard": len(real & family_a) / len(real | family_a),
        }

    elapsed, max_cuda = require_resource_ceiling(started)
    result = {
        "schema": "block3_consequence_family_f_v1_fit_results",
        "status": "fit_complete_no_validation_or_final_opened",
        "authority_sha256": frozen_authority["authority_sha256"],
        "score_traces": score_traces,
        "affine_traces": affine_traces,
        "known_answer_replay": known_answer,
        "postfit_report": report,
        "postfit_stage_transitions": postfit_transitions,
        "stacked_typed_fit_nrmse": stacked_nrmse,
        "direct_polarization_replay": polarization_replays,
        "score_projection_replay_max_abs": projection_replays,
        "support_overlaps": score_overlaps,
        "program_prices": prices,
        "call_ledger": call_receipt,
        "model_state_before_sha256": model_before,
        "model_state_after_sha256": model_after,
        "fit_rows_loaded": life.ROW_COUNT,
        "validation_rows_loaded": 0,
        "final_rows_loaded": 0,
        "ground_truth_target_tokens_used": 0,
        "retained_teacher_logits": 0,
        "authorized_for_validation": False,
        "authorized_for_final": False,
        "authorized_for_global_ledger_credit": False,
        "elapsed_seconds": elapsed,
        "maximum_allocated_cuda_bytes": max_cuda,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    return program_artifact, result


def require_exact_tensor_tree(observed: Any, expected: Any, *, path: str = "root") -> None:
    """Exact recursive replay for a just-published tensor artifact."""

    if torch.is_tensor(expected):
        if not torch.is_tensor(observed) or not torch.equal(observed, expected):
            raise RuntimeError(f"family-F tensor artifact changed at {path}")
        return
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping) or set(observed) != set(expected):
            raise RuntimeError(f"family-F tensor artifact schema changed at {path}")
        for key in expected:
            require_exact_tensor_tree(observed[key], expected[key], path=f"{path}/{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(observed, type(expected)) or len(observed) != len(expected):
            raise RuntimeError(f"family-F tensor sequence changed at {path}")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            require_exact_tensor_tree(left, right, path=f"{path}/{index}")
        return
    if observed != expected:
        raise RuntimeError(f"family-F scalar artifact changed at {path}")


def _require_finite_tree(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite_tree(child, path=f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite_tree(child, path=f"{path}/{index}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"family-F nonfinite result at {path}")


def semantic_validate_result(
    value: Mapping[str, Any], *, expected_authority_sha256: str,
    expected_programs_file_sha256: str,
    program_artifact: Mapping[str, Any] | None = None,
    parent_payload: Mapping[str, Any] | None = None,
    family_a_supports: Mapping[int, torch.Tensor] | None = None,
) -> None:
    required = {
        "schema", "status", "authority_sha256", "score_traces", "affine_traces",
        "known_answer_replay", "postfit_report", "postfit_stage_transitions",
        "stacked_typed_fit_nrmse", "direct_polarization_replay",
        "score_projection_replay_max_abs", "support_overlaps", "program_prices",
        "call_ledger", "model_state_before_sha256", "model_state_after_sha256",
        "fit_rows_loaded", "validation_rows_loaded", "final_rows_loaded",
        "ground_truth_target_tokens_used", "retained_teacher_logits",
        "authorized_for_validation", "authorized_for_final",
        "authorized_for_global_ledger_credit", "elapsed_seconds",
        "maximum_allocated_cuda_bytes", "torch_version", "python_version",
        "programs_file_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get(
        "schema"
    ) != "block3_consequence_family_f_v1_fit_results" or value.get(
        "status"
    ) != "fit_complete_no_validation_or_final_opened" or value.get(
        "authority_sha256"
    ) != expected_authority_sha256 or value.get(
        "programs_file_sha256"
    ) != expected_programs_file_sha256:
        raise RuntimeError("family-F result schema or joins changed")
    _require_finite_tree(value)
    if set(value["score_traces"]) != set(call_contract.SCORE_ARMS) or any(
        len(trace) != 8 or [row.get("epoch") for row in trace] != list(range(1, 9))
        for trace in value["score_traces"].values()
    ):
        raise RuntimeError("family-F score trace schedule changed")
    score_trace_fields = {
        "epoch", "document_kl", "row_kl", "score_min", "score_max", "score_sum",
        "saturated_zero", "saturated_one", "gradient_norm_max",
    }
    if any(
        not isinstance(row, Mapping) or set(row) != score_trace_fields
        for trace in value["score_traces"].values() for row in trace
    ):
        raise RuntimeError("family-F score trace schema changed")
    if set(value["affine_traces"]) != set(call_contract.AFFINE_ARMS) or any(
        len(trace) != 4 or [row.get("epoch") for row in trace] != list(range(1, 5))
        for trace in value["affine_traces"].values()
    ):
        raise RuntimeError("family-F affine trace schedule changed")
    affine_trace_fields = {
        "epoch", "document_kl", "row_kl", "gradient_norm_max", "scale",
        "correction_rms", "correction_norm",
    }
    if any(
        not isinstance(row, Mapping) or set(row) != affine_trace_fields
        for trace in value["affine_traces"].values() for row in trace
    ):
        raise RuntimeError("family-F affine trace schema changed")
    known = value["known_answer_replay"]
    if not isinstance(known, Mapping) or set(known) != {
        "raw_max_absolute", "raw_max_relative", "max_absolute", "max_relative",
        "teacher_self_kl_max",
    } or any(number < 0 for number in known.values()) or known[
        "raw_max_relative"
    ] > core.REPLAY_RELATIVE_LIMIT or known[
        "max_relative"
    ] > core.REPLAY_RELATIVE_LIMIT or known["teacher_self_kl_max"] > 2e-7:
        raise RuntimeError("family-F known-answer replay changed")
    report_names = set(call_contract.REPORT_STUDENT_ARMS)
    if set(value["postfit_report"]) != report_names or any(
        set(metrics) != {
            "document_balanced_teacher_kl", "row_mean_teacher_kl",
            "summed_write_nrmse",
        }
        for metrics in value["postfit_report"].values()
    ):
        raise RuntimeError("family-F postfit report schema changed")
    if any(
        metrics["document_balanced_teacher_kl"] < -1e-7
        or metrics["row_mean_teacher_kl"] < -1e-7
        or metrics["summed_write_nrmse"] < 0
        for metrics in value["postfit_report"].values()
    ):
        raise RuntimeError("family-F postfit report contains impossible metrics")
    program_names = report_names - {"continuous_teacher_F1"}
    if set(value["direct_polarization_replay"]) != program_names or set(
        value["program_prices"]
    ) != program_names or set(value["stacked_typed_fit_nrmse"]) != {
        name for name in program_names if not name.startswith("affine_")
    }:
        raise RuntimeError("family-F executable diagnostic arm registry changed")
    if any(
        set(metrics) != {"max_absolute", "max_relative"} or metrics.get(
            "max_relative", math.inf
        ) > core.REPLAY_RELATIVE_LIMIT
        for metrics in value["direct_polarization_replay"].values()
    ) or set(value["score_projection_replay_max_abs"]) != set(
        call_contract.SCORE_ARMS
    ) or any(
        replay > 2e-10 for replay in value["score_projection_replay_max_abs"].values()
    ):
        raise RuntimeError("family-F program or projection replay changed")
    if any(number < 0 for number in value["stacked_typed_fit_nrmse"].values()):
        raise RuntimeError("family-F stacked NRMSE is negative")
    expected_overlap_keys = {
        f"teacher_vs_{comparison}_k{budget}"
        for comparison in (
            "teacher_row_reversal", "teacher_document_derangement", "random", "family_A",
        )
        for budget in BUDGETS
    }
    if set(value["support_overlaps"]) != expected_overlap_keys or any(
        set(metrics) != {"intersection", "jaccard"} or not (
            0 <= metrics["intersection"] <= budget and 0 <= metrics["jaccard"] <= 1
        )
        for key, metrics in value["support_overlaps"].items()
        for budget in [int(key.rsplit("k", 1)[1])]
    ):
        raise RuntimeError("family-F support-overlap registry changed")
    transition_fields = {
        "continuous_F1_document_kl", "binary_native_down_document_kl",
        "post_refit_document_kl", "binary_minus_continuous_document_kl",
        "refit_minus_binary_document_kl",
    }
    if set(value["postfit_stage_transitions"]) != {"256", "512"} or any(
        set(metrics) != transition_fields
        for metrics in value["postfit_stage_transitions"].values()
    ):
        raise RuntimeError("family-F postfit transition registry changed")
    for budget in BUDGETS:
        report = value["postfit_report"]
        expected_transition = {
            "continuous_F1_document_kl": report[
                "continuous_teacher_F1"
            ]["document_balanced_teacher_kl"],
            "binary_native_down_document_kl": report[
                f"real_F_binary_native_down_k{budget}"
            ]["document_balanced_teacher_kl"],
            "post_refit_document_kl": report[
                f"real_F_post_refit_k{budget}"
            ]["document_balanced_teacher_kl"],
            "binary_minus_continuous_document_kl": report[
                f"real_F_binary_native_down_k{budget}"
            ]["document_balanced_teacher_kl"] - report[
                "continuous_teacher_F1"
            ]["document_balanced_teacher_kl"],
            "refit_minus_binary_document_kl": report[
                f"real_F_post_refit_k{budget}"
            ]["document_balanced_teacher_kl"] - report[
                f"real_F_binary_native_down_k{budget}"
            ]["document_balanced_teacher_kl"],
        }
        if value["postfit_stage_transitions"][str(budget)] != expected_transition:
            raise RuntimeError("family-F postfit transition arithmetic changed")
    for price in value["program_prices"].values():
        if set(price) != {
            "float_values", "float_bytes", "index_bytes", "total_bytes",
            "products_per_token", "linear_multiplies_per_token",
        } or any(type(number) is not int or number <= 0 for number in price.values()):
            raise RuntimeError("family-F executable price changed")
    contexts = (program_artifact, parent_payload, family_a_supports)
    if any(context is not None for context in contexts):
        if any(context is None for context in contexts):
            raise RuntimeError("family-F result reconstruction context is incomplete")
        assert program_artifact is not None
        assert parent_payload is not None
        assert family_a_supports is not None
        programs = {
            name: _materialize_program(payload, device="cpu")
            for name, payload in program_artifact["programs"].items()
        }
        expected_prices = {
            name: core.program_price(program) for name, program in programs.items()
        }
        if value["program_prices"] != expected_prices:
            raise RuntimeError("family-F program prices do not reconstruct")
        prefilter = parent_payload["prefilter_indices"].long().contiguous()
        expected_stacked = {
            name: _stacked_fit_nrmse(
                program, prefilter, parent_payload["prefilter_gram"].double(),
                parent_payload["prefilter_cross"].double(),
                parent_payload["native_typed_write_energy"].double(),
            )
            for name, program in programs.items() if not name.startswith("affine_")
        }
        if any(
            not math.isclose(
                value["stacked_typed_fit_nrmse"][name], expected,
                rel_tol=1e-12, abs_tol=1e-12,
            )
            for name, expected in expected_stacked.items()
        ):
            raise RuntimeError("family-F stacked NRMSE does not reconstruct")
        expected_projection = {
            arm: float((
                core.project_capped_simplex(scores, 512) - scores
            ).abs().max())
            for arm, scores in program_artifact["scores"].items()
        }
        if value["score_projection_replay_max_abs"] != expected_projection:
            raise RuntimeError("family-F score projection diagnostic does not reconstruct")
        expected_overlaps: dict[str, Any] = {}
        for budget in BUDGETS:
            real = set(program_artifact["supports"][f"teacher_k{budget}"].tolist())
            comparisons = {
                "teacher_row_reversal": program_artifact["supports"][
                    f"teacher_row_reversal_k{budget}"
                ],
                "teacher_document_derangement": program_artifact["supports"][
                    f"teacher_document_derangement_k{budget}"
                ],
                "random": program_artifact["supports"][f"random_k{budget}"],
                "family_A": family_a_supports[budget],
            }
            for name, support in comparisons.items():
                other = set(support.tolist())
                expected_overlaps[f"teacher_vs_{name}_k{budget}"] = {
                    "intersection": len(real & other),
                    "jaccard": len(real & other) / len(real | other),
                }
        if value["support_overlaps"] != expected_overlaps:
            raise RuntimeError("family-F support overlaps do not reconstruct")
        for name, program in programs.items():
            expected_replay = deployed_polarization_replay(program)
            observed_replay = value["direct_polarization_replay"][name]
            if any(
                abs(observed_replay[key] - expected_replay[key]) > 2e-5
                for key in ("max_absolute", "max_relative")
            ):
                raise RuntimeError("family-F polarization diagnostic does not reconstruct")
    call_contract.FamilyFCallLedger.replay_complete_receipt(value["call_ledger"])
    before = value["model_state_before_sha256"]
    after = value["model_state_after_sha256"]
    if before != after or not isinstance(before, str) or len(before) != 64:
        raise RuntimeError("family-F model integrity result changed")
    if (
        value["fit_rows_loaded"] != life.ROW_COUNT
        or value["validation_rows_loaded"] != 0
        or value["final_rows_loaded"] != 0
        or value["ground_truth_target_tokens_used"] != 0
        or value["retained_teacher_logits"] != 0
        or value["authorized_for_validation"] is not False
        or value["authorized_for_final"] is not False
        or value["authorized_for_global_ledger_credit"] is not False
        or not 0 <= value["elapsed_seconds"] <= MAX_WALL_SECONDS
        or not 0 <= value["maximum_allocated_cuda_bytes"] <= MAX_ALLOCATED_CUDA_BYTES
    ):
        raise RuntimeError("family-F result permissions, rows, or resources changed")


def semantic_validate_receipt(
    value: Mapping[str, Any], *, expected_authority_sha256: str,
    authority_file_sha256: str, programs_file_sha256: str,
    results_file_sha256: str, source_sha256: str, prior_sha256: str,
    rows_sha256: str, checkpoint_weights_sha256: str,
    expected_call_ledger: Mapping[str, Any],
) -> None:
    required = {
        "schema", "status", "authority_sha256", "authority_file_sha256",
        "programs_file_sha256", "results_file_sha256", "source_closure_sha256",
        "prior_artifact_binding_sha256", "row_binding_sha256",
        "checkpoint_weights_sha256", "call_ledger", "validation_rows_loaded",
        "final_rows_loaded", "authorized_for_validation", "authorized_for_final",
        "authorized_for_global_ledger_credit", "elapsed_seconds",
    }
    expected = {
        "authority_sha256": expected_authority_sha256,
        "authority_file_sha256": authority_file_sha256,
        "programs_file_sha256": programs_file_sha256,
        "results_file_sha256": results_file_sha256,
        "source_closure_sha256": source_sha256,
        "prior_artifact_binding_sha256": prior_sha256,
        "row_binding_sha256": rows_sha256,
        "checkpoint_weights_sha256": checkpoint_weights_sha256,
    }
    if not isinstance(value, Mapping) or set(value) != required or value.get(
        "schema"
    ) != "block3_consequence_family_f_v1_receipt" or value.get(
        "status"
    ) != "fit_complete_receipt_last_no_evaluation_opened" or any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ) or value.get("call_ledger") != expected_call_ledger:
        raise RuntimeError("family-F receipt schema or joins changed")
    call_contract.FamilyFCallLedger.replay_complete_receipt(value["call_ledger"])
    if value["validation_rows_loaded"] != 0 or value["final_rows_loaded"] != 0 or (
        value["authorized_for_validation"] is not False
    ) or value["authorized_for_final"] is not False or value[
        "authorized_for_global_ledger_credit"
    ] is not False or not isinstance(value["elapsed_seconds"], (int, float)) or not (
        0 <= value["elapsed_seconds"] <= MAX_WALL_SECONDS
    ):
        raise RuntimeError("family-F receipt permissions or resources changed")


def run() -> dict[str, Any]:
    life.require_pristine_namespace()
    claim = collector.acquire_claim(LOCK)
    calls: call_contract.FamilyFCallLedger | None = None
    started = time.time()
    try:
        source = source_closure()
        prior = life.prior_artifact_binding()
        rows_binding = life.row_binding()
        checkpoint = facade.validate_snapshot(verify_weights_sha256=True)
        frozen_authority = authority(source, prior, rows_binding, checkpoint)
        verify_frozen_inputs(source, prior, rows_binding, checkpoint)
        claim.verify()
        collector.create_json(AUTHORITY, frozen_authority)
        if json.loads(AUTHORITY.read_text()) != frozen_authority:
            raise RuntimeError("family-F authority did not replay after publication")
        verify_frozen_inputs(source, prior, rows_binding, checkpoint)

        calls = call_contract.FamilyFCallLedger()
        program_artifact, result = execute_fit(
            frozen_authority=frozen_authority, rows_binding=rows_binding,
            checkpoint=checkpoint, calls=calls, started=started,
        )
        require_resource_ceiling(started)
        claim.verify()
        verify_frozen_inputs(source, prior, rows_binding, checkpoint)
        if json.loads(AUTHORITY.read_text()) != frozen_authority:
            raise RuntimeError("family-F authority drifted before program publication")
        validation_parent, validation_a = load_parent_tensors_after_authority(
            frozen_authority, started=started,
        )
        family_a_supports = {
            budget: validation_a["programs"][
                f"activation_selected_k{budget}"
            ]["indices"].long().contiguous()
            for budget in BUDGETS
        }
        semantic_validate_program_artifact(
            program_artifact,
            expected_authority_sha256=frozen_authority["authority_sha256"],
            prefilter=validation_parent["prefilter_indices"].long(),
            family_a_supports=family_a_supports,
        )
        collector.create_torch(PROGRAMS, program_artifact)
        reloaded_programs = torch.load(PROGRAMS, map_location="cpu", weights_only=True)
        require_exact_tensor_tree(reloaded_programs, program_artifact)
        semantic_validate_program_artifact(
            reloaded_programs,
            expected_authority_sha256=frozen_authority["authority_sha256"],
            prefilter=validation_parent["prefilter_indices"].long(),
            family_a_supports=family_a_supports,
        )
        program_hash = file_sha256(PROGRAMS)
        result = {**result, "programs_file_sha256": program_hash}
        semantic_validate_result(
            result, expected_authority_sha256=frozen_authority["authority_sha256"],
            expected_programs_file_sha256=program_hash,
            program_artifact=program_artifact, parent_payload=validation_parent,
            family_a_supports=family_a_supports,
        )
        collector.create_json(RESULTS, result)
        reloaded_result = json.loads(RESULTS.read_text())
        if reloaded_result != result:
            raise RuntimeError("family-F result changed after publication")
        semantic_validate_result(
            reloaded_result,
            expected_authority_sha256=frozen_authority["authority_sha256"],
            expected_programs_file_sha256=program_hash,
            program_artifact=reloaded_programs, parent_payload=validation_parent,
            family_a_supports=family_a_supports,
        )
        result_hash = file_sha256(RESULTS)

        authority_hash = file_sha256(AUTHORITY)
        receipt = {
            "schema": "block3_consequence_family_f_v1_receipt",
            "status": "fit_complete_receipt_last_no_evaluation_opened",
            "authority_sha256": frozen_authority["authority_sha256"],
            "authority_file_sha256": authority_hash,
            "programs_file_sha256": program_hash,
            "results_file_sha256": result_hash,
            "source_closure_sha256": source["sha256"],
            "prior_artifact_binding_sha256": prior["sha256"],
            "row_binding_sha256": rows_binding["sha256"],
            "checkpoint_weights_sha256": checkpoint.weights_sha256,
            "call_ledger": result["call_ledger"],
            "validation_rows_loaded": 0,
            "final_rows_loaded": 0,
            "authorized_for_validation": False,
            "authorized_for_final": False,
            "authorized_for_global_ledger_credit": False,
            "elapsed_seconds": time.time() - started,
        }
        semantic_validate_receipt(
            receipt,
            expected_authority_sha256=frozen_authority["authority_sha256"],
            authority_file_sha256=authority_hash,
            programs_file_sha256=program_hash,
            results_file_sha256=result_hash,
            source_sha256=source["sha256"], prior_sha256=prior["sha256"],
            rows_sha256=rows_binding["sha256"],
            checkpoint_weights_sha256=checkpoint.weights_sha256,
            expected_call_ledger=result["call_ledger"],
        )
        # All deserialization and semantic reconstruction is complete above.  The
        # terminal window contains only frozen-input/byte guards and receipt creation.
        claim.verify()
        verify_frozen_inputs(source, prior, rows_binding, checkpoint)
        if json.loads(AUTHORITY.read_text()) != frozen_authority or file_sha256(
            AUTHORITY
        ) != authority_hash or file_sha256(PROGRAMS) != program_hash or file_sha256(
            RESULTS
        ) != result_hash:
            raise RuntimeError("family-F terminal byte integrity replay failed")
        require_resource_ceiling(started)
        claim.verify()
        collector.create_json(RECEIPT, receipt)
        return result
    except BaseException as error:
        if not FAILURE.exists() and not RECEIPT.exists():
            try:
                claim.verify()
                collector.create_json(FAILURE, {
                    "schema": "block3_consequence_family_f_v1_failure",
                    "status": "terminal_failure_no_receipt",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "authority_exists": AUTHORITY.exists(),
                    "programs_exists": PROGRAMS.exists(),
                    "results_exists": RESULTS.exists(),
                    "receipt_exists": RECEIPT.exists(),
                    "partial_call_ledger": (
                        calls.partial_receipt() if calls is not None and not calls.closed
                        else calls.receipt() if calls is not None else None
                    ),
                    "elapsed_seconds": time.time() - started,
                })
            except BaseException:
                pass
        raise
    finally:
        claim.release()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
