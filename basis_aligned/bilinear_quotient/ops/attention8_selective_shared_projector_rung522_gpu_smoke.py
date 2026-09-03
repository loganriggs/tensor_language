#!/usr/bin/env python3
"""Managed, instrument-only GPU smoke for rung 522.

Prediction A: one no-gradient physical batch containing six recipient rows,
eight donor maps, and two swap directions (96 sequences total) fits memory,
executes attention8 exactly once, and applies a nonzero rank-4 edit to every
sequence.

Prediction B: one differentiable batch-six projected intervention executes
attention8 exactly once and yields a finite, nonzero frame gradient while all
frozen model parameters retain ``grad is None``.

Prediction C: all frozen dependency hashes match before model imports and the
receipt retains no CE, circuit, mask, or task metric.

Null: either registered batch shape is infeasible, the intervention is dead,
attention8 executes other than once, or gradients are absent/nonfinite/leak
into model parameters. Any null blocks rung-522 science; the smoke is not
permission to reduce the physical batch or alter the optimizer call count.

Price: CPU construction of the frozen FIT row/donor design, native capture of
the required rows in chunks of six, one no-gradient batch-96 forward, and one
batch-six forward/backward pair. No circuit mask, CE response, task score, or
scientific model outcome is computed or retained.
"""

# BQGATE: EXPERIMENT
# pred_a: no-grad batch96 is live and attention8 executes exactly once
# pred_b: batch6 projected intervention gives a finite nonzero frame-only gradient
# pred_c: frozen hashes match and no scientific metric is retained

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


REGISTERED_PREDICTIONS = {
    "pred_a": "no-grad batch96 is live and attention8 executes exactly once",
    "pred_b": "batch6 projected intervention gives a finite nonzero frame-only gradient",
    "pred_c": "frozen hashes match and no scientific metric is retained",
}


REPO = Path("/workspace/tensor_language")
ROOT = REPO / "basis_aligned/bilinear_quotient"
OPS = ROOT / "ops"
POLY = REPO / "basis_aligned/polynomial_causal"
DEFAULT_OUTPUT = ROOT / "attention8_selective_shared_projector_rung522_gpu_smoke.json"

FROZEN_HASHES = {
    POLY / "ATTENTION8_SELECTIVE_SHARED_PROJECTOR_RUNG522_PREREGISTRATION.md": (
        "27bc74c3e19ac310f0ed88f1527a1df44ff52d8990d980971415b32b503126f5"
    ),
    POLY / "ATTENTION8_SELECTIVE_SHARED_PROJECTOR_RUNG522_PREFLIGHT_ADDENDUM.md": (
        "d3343c0acc8233580cdb209d1652c7d30c839823b399f9d19e7ba923ffe53b22"
    ),
    OPS / "attention8_selective_shared_projector_rung522_math.py": (
        "6cff6f7726dd8f76e786d64abf913fc31adbdfec101a97741a1aa3396f8431c2"
    ),
    OPS / "attention8_selective_shared_projector_rung522_toy_preflight.py": (
        "5abbb09ec0871e0d7ad5b8cb63a3f6103027848700df36fcc3dc85ce21c42935"
    ),
    OPS / "test_attention8_selective_shared_projector_rung522_math.py": (
        "42b7b2f41fccf1c4f662f0b7dfeddc6f59836c9a9b26b611977007d8c00542c7"
    ),
    OPS / "attention8_selective_shared_projector_rung522_toy_preflight_results.json": (
        "398842217e729e743dc4b5fe4947dc7837a40e01a42b2c267faa2249a6ad0fe4"
    ),
    ROOT / "attention8_shared_private_das_rung521_stage_a_results.json": (
        "6a303e0e62ef3d2443ed6d667f74bc28c703a79ce5f462657bff212c1c5a676c"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_frozen_hashes() -> dict[str, str]:
    observed = {}
    for path, expected in FROZEN_HASHES.items():
        if not path.is_file():
            raise RuntimeError(f"frozen rung522 dependency is absent: {path}")
        actual = _sha256_file(path)
        observed[str(path.relative_to(REPO))] = actual
        if actual != expected:
            raise RuntimeError(
                f"frozen rung522 dependency changed: {path}; expected {expected}, got {actual}"
            )
    return observed


# This gate deliberately runs before torch, TT.GPT, the model facade, or any
# experiment module is imported.
_PREIMPORT_HASHES = _validate_frozen_hashes()
if os.environ.get("BQLIB_DRYRUN") == "1":
    print(
        "DRYRUN OK: rung522 instrument-only smoke; 7 frozen hashes; "
        "no-grad batch=6*8*2=96; gradient batch=6; no science metrics retained",
        flush=True,
    )
    raise SystemExit(0)


import torch  # noqa: E402

for _path in (OPS, POLY, ROOT, REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import attention8_shared_private_das_rung521 as stage_a  # noqa: E402
import attention8_selective_shared_projector_rung522_math as core  # noqa: E402
import bilin18_observed_model_facade as facade  # noqa: E402


D = 1152
TOKENS = 256
CAPTURE_BATCH = 6
RECIPIENT_ROWS = 6
DONOR_MAPS = 8
DIRECTIONS = 2
PHYSICAL_BATCH = RECIPIENT_ROWS * DONOR_MAPS * DIRECTIONS
RANK = 4


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite smoke receipt: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as sink:
        json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
        sink.write("\n")
        sink.flush()
        os.fsync(sink.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _execute(
    model,
    tokens: torch.Tensor,
    *,
    edit=None,
    capture: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, object]]:
    """Execute one arbitrary-batch model call and audit the attention8 boundary."""
    dispatch_calls = 0
    module_calls = 0
    captured = None
    per_sequence_edit_rms = None

    def module_counter(_module, _inputs, _output):
        nonlocal module_calls
        module_calls += 1

    def attention(event):
        nonlocal dispatch_calls, captured, per_sequence_edit_rms
        write, first_value = event.block.attn(event.state, event.first_value)
        if event.site == 8:
            dispatch_calls += 1
            if capture:
                if captured is not None:
                    raise RuntimeError("attention8 capture occurred more than once")
                captured = write.detach().float().cpu().clone()
            if edit is not None:
                changed = edit(write)
                if changed.shape != write.shape or changed.dtype != write.dtype or (
                    changed.device != write.device
                ):
                    raise RuntimeError("attention8 smoke edit changed tensor metadata")
                difference = (changed.detach().float() - write.detach().float()).reshape(
                    write.shape[0], -1
                )
                per_sequence_edit_rms = difference.square().mean(dim=1).sqrt().cpu()
                write = changed
        return write, first_value

    def mlp(event):
        return event.block.mlp(event.state)

    handle = model.transformer.h[8].attn.register_forward_hook(module_counter)
    try:
        logits = facade.forward_with_dispatch(
            model,
            tokens,
            attention,
            mlp,
            require_production=False,
        )
    finally:
        handle.remove()
    return logits, captured, {
        "attention8_dispatch_calls": dispatch_calls,
        "attention8_module_calls": module_calls,
        "per_sequence_edit_rms": per_sequence_edit_rms,
    }


@torch.no_grad()
def _capture_required_writes(
    model, rows: torch.Tensor, required_rows: torch.Tensor, device: torch.device
) -> tuple[dict[int, torch.Tensor], dict[str, object]]:
    captures: dict[int, torch.Tensor] = {}
    executions = 0
    calls_once = True
    outputs_without_grad = True
    for start in range(0, required_rows.numel(), CAPTURE_BATCH):
        selected = required_rows[start : start + CAPTURE_BATCH]
        tokens = rows[selected, :TOKENS].to(device)
        logits, write, diagnostics = _execute(model, tokens, capture=True)
        if write is None:
            raise RuntimeError("attention8 write was not captured")
        executions += 1
        calls_once &= (
            diagnostics["attention8_dispatch_calls"] == 1
            and diagnostics["attention8_module_calls"] == 1
        )
        outputs_without_grad &= not logits.requires_grad and not write.requires_grad
        for local, row in enumerate(selected.tolist()):
            captures[int(row)] = write[local]
        del logits, write
    if len(captures) != required_rows.numel():
        raise RuntimeError("native capture cache lost or duplicated a requested row")
    return captures, {
        "capture_executions": executions,
        "attention8_once_every_capture": calls_once,
        "capture_outputs_require_no_grad": outputs_without_grad,
    }


def _donor_rows_for_map(
    donor_map: torch.Tensor, selected_rows: torch.Tensor
) -> torch.Tensor:
    first_positions = selected_rows * TOKENS
    donors = donor_map[first_positions]
    if bool((donors < 0).any()) or bool((donors % TOKENS != 0).any()):
        raise RuntimeError("row-coherent donor map changed the first token position")
    return donors // TOKENS


def _stack_donor_writes(
    captures: dict[int, torch.Tensor], donor_rows: torch.Tensor
) -> torch.Tensor:
    return torch.stack([captures[int(row)] for row in donor_rows.tolist()])


@torch.no_grad()
def _no_grad_batch96(
    model,
    rows: torch.Tensor,
    selected_rows: torch.Tensor,
    maps: tuple[torch.Tensor, ...],
    inverse_maps: tuple[torch.Tensor, ...],
    captures: dict[int, torch.Tensor],
    device: torch.device,
) -> dict[str, object]:
    token_blocks = []
    donor_blocks = []
    for direction_maps in (maps, inverse_maps):
        for donor_map in direction_maps:
            token_blocks.append(rows[selected_rows, :TOKENS])
            donor_rows = _donor_rows_for_map(donor_map, selected_rows)
            donor_blocks.append(_stack_donor_writes(captures, donor_rows))
    tokens = torch.cat(token_blocks, dim=0).to(device)
    donor_writes = torch.cat(donor_blocks, dim=0).to(device=device, dtype=torch.float32)
    if tuple(tokens.shape) != (PHYSICAL_BATCH, TOKENS):
        raise RuntimeError(f"physical smoke token batch changed: {tuple(tokens.shape)}")
    if tuple(donor_writes.shape) != (PHYSICAL_BATCH, TOKENS, D):
        raise RuntimeError(f"physical smoke donor batch changed: {tuple(donor_writes.shape)}")
    frame = core.deterministic_haar_frame(
        D, RANK, 52200, dtype=torch.float32, device=device
    )

    def edit(write: torch.Tensor) -> torch.Tensor:
        return core.daslib.projection_interchange(
            write, donor_writes, frame, validate=False
        )

    logits, _, diagnostics = _execute(model, tokens, edit=edit)
    rms = diagnostics.pop("per_sequence_edit_rms")
    if rms is None or tuple(rms.shape) != (PHYSICAL_BATCH,):
        raise RuntimeError("batch96 did not report one edit RMS per sequence")
    finite_logits = bool(torch.isfinite(logits).all().detach().cpu())
    result = {
        "physical_shape": [RECIPIENT_ROWS, DONOR_MAPS, DIRECTIONS, TOKENS],
        "physical_batch_sequences": PHYSICAL_BATCH,
        "output_requires_grad": bool(logits.requires_grad),
        "logits_finite": finite_logits,
        "attention8_dispatch_calls": diagnostics["attention8_dispatch_calls"],
        "attention8_module_calls": diagnostics["attention8_module_calls"],
        "minimum_per_sequence_edit_rms": float(rms.min()),
        "maximum_per_sequence_edit_rms": float(rms.max()),
        "all_sequence_edits_live": bool((rms > 0).all()),
    }
    result["passed"] = bool(
        result["physical_batch_sequences"] == 96
        and not result["output_requires_grad"]
        and result["logits_finite"]
        and result["attention8_dispatch_calls"] == 1
        and result["attention8_module_calls"] == 1
        and result["all_sequence_edits_live"]
    )
    del logits, tokens, donor_writes, frame
    return result


def _gradient_batch6(
    model,
    rows: torch.Tensor,
    selected_rows: torch.Tensor,
    donor_map: torch.Tensor,
    captures: dict[int, torch.Tensor],
    device: torch.device,
) -> dict[str, object]:
    for parameter in model.parameters():
        parameter.grad = None
    tokens = rows[selected_rows, :TOKENS].to(device)
    donor_rows = _donor_rows_for_map(donor_map, selected_rows)
    donor_writes = _stack_donor_writes(captures, donor_rows).to(
        device=device, dtype=torch.float32
    )
    initial = core.deterministic_haar_frame(
        D, RANK, 52200, dtype=torch.float32, device=device
    )
    raw_frame = torch.nn.Parameter(initial.clone())
    frame = core.differentiable_qr_retraction(raw_frame)

    def edit(write: torch.Tensor) -> torch.Tensor:
        return core.daslib.projection_interchange(
            write, donor_writes, frame, validate=False
        )

    with torch.enable_grad():
        logits, _, diagnostics = _execute(model, tokens, edit=edit)
        # Fixed, task-free suffix checksum. It exists only to establish a
        # downstream gradient path and is deliberately not written to output.
        checksum = logits[..., :64].float().square().mean()
        checksum.backward()
    rms = diagnostics.pop("per_sequence_edit_rms")
    gradient = raw_frame.grad
    gradient_finite = gradient is not None and bool(torch.isfinite(gradient).all().detach().cpu())
    gradient_norm = float(gradient.float().norm().detach().cpu()) if gradient_finite else 0.0
    model_parameters_without_grad = all(parameter.grad is None for parameter in model.parameters())
    orthonormality = float(core.daslib.orthonormality_error(frame.detach()).cpu())
    result = {
        "physical_batch_sequences": int(tokens.shape[0]),
        "attention8_dispatch_calls": diagnostics["attention8_dispatch_calls"],
        "attention8_module_calls": diagnostics["attention8_module_calls"],
        "minimum_per_sequence_edit_rms": float(rms.min()) if rms is not None else 0.0,
        "all_sequence_edits_live": bool(rms is not None and (rms > 0).all()),
        "frame_gradient_present": gradient is not None,
        "frame_gradient_finite": gradient_finite,
        "frame_gradient_norm": gradient_norm,
        "frame_gradient_nonzero": gradient_norm > 0,
        "model_parameters_without_gradients": model_parameters_without_grad,
        "frame_orthonormality_error": orthonormality,
    }
    result["passed"] = bool(
        result["physical_batch_sequences"] == 6
        and result["attention8_dispatch_calls"] == 1
        and result["attention8_module_calls"] == 1
        and result["all_sequence_edits_live"]
        and result["frame_gradient_present"]
        and result["frame_gradient_finite"]
        and result["frame_gradient_nonzero"]
        and result["model_parameters_without_gradients"]
        and result["frame_orthonormality_error"] <= 1e-5
    )
    del logits, checksum, tokens, donor_writes, initial, raw_frame, frame
    return result


def _gpu_smoke() -> dict[str, object]:
    if os.environ.get("BQLIB_NO_MODEL") == "1":
        raise RuntimeError("BQLIB_NO_MODEL forbids the rung522 managed GPU smoke")
    data = stage_a._load_cpu_inputs()
    donor_design = stage_a._construct_donors_for_split(
        "fit", data["row_masks"]["fit"], data["base_ce"], data["docids"]
    )
    maps = tuple(donor_design["maps"])
    inverse_maps = tuple(donor_design["inverse_maps"])
    if len(maps) != DONOR_MAPS or len(inverse_maps) != DONOR_MAPS:
        raise RuntimeError("frozen smoke requires exactly eight forward and inverse donor maps")
    fit_rows = data["row_masks"]["fit"].nonzero().flatten()
    selected_rows = fit_rows[:RECIPIENT_ROWS]
    if selected_rows.numel() != RECIPIENT_ROWS:
        raise RuntimeError("FIT has fewer than six smoke recipient rows")
    required = set(selected_rows.tolist())
    for donor_map in maps + inverse_maps:
        required.update(_donor_rows_for_map(donor_map, selected_rows).tolist())
    required_rows = torch.tensor(sorted(required), dtype=torch.int64)

    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32)
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats(device)
    captures, capture_diagnostics = _capture_required_writes(
        model, data["rows"], required_rows, device
    )
    no_grad_result = _no_grad_batch96(
        model,
        data["rows"],
        selected_rows,
        maps,
        inverse_maps,
        captures,
        device,
    )
    gradient_result = _gradient_batch6(
        model,
        data["rows"],
        selected_rows,
        maps[0],
        captures,
        device,
    )
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    result = {
        "schema_version": 1,
        "rung": 522,
        "namespace": "rung522_managed_gpu_smoke_no_task_or_circuit_outcome",
        "frozen_dependency_sha256": _PREIMPORT_HASHES,
        "checkpoint": checkpoint.__dict__,
        "capture_instrument": capture_diagnostics,
        "no_grad_batch96": no_grad_result,
        "gradient_batch6": gradient_result,
        "peak_cuda_memory_bytes": peak_bytes,
        "scientific_metrics_retained": False,
        "model_science_opened": False,
    }
    result["passed"] = bool(
        capture_diagnostics["attention8_once_every_capture"]
        and capture_diagnostics["capture_outputs_require_no_grad"]
        and no_grad_result["passed"]
        and gradient_result["passed"]
    )
    del model, captures
    torch.cuda.empty_cache()
    return result


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = _gpu_smoke()
    _atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": result["passed"],
                "no_grad_batch96": result["no_grad_batch96"],
                "gradient_batch6": result["gradient_batch6"],
                "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
                "scientific_metrics_retained": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit("RUNG522 GPU SMOKE FAILED: model science remains closed")
    return result


if __name__ == "__main__":
    main()
