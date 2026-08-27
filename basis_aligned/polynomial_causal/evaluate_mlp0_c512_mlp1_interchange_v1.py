#!/usr/bin/env python3
"""Authority-bound evaluation of the physical C512/MLP1 interchange assay."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
PC = ROOT / "basis_aligned" / "polynomial_causal"
sys.path.insert(0, str(BQ))
sys.path.insert(0, str(PC))

from mlp0_c512_mlp1_interchange import (  # noqa: E402
    BACKGROUNDS, additive_interaction_prediction, capture_through_mlp1,
    document_derangement, norm_matched_native_write, physical_post_states,
    suffix_forward,
)
from mlp0_native_down_program import load_program  # noqa: E402
from prepare_mlp0_c512_mlp1_interchange_v1_rows import (  # noqa: E402
    RECEIPT as ROW_RECEIPT, load_frozen_rows,
)
from prepare_mlp0_quotient_stage0_v1_rows import load_frozen_role  # noqa: E402
from score_mlp0_c512_mlp1_interchange_v1 import (  # noqa: E402
    ARMS, CONTRASTS, MARGINS, integer_array_sha256, ordered_ids_sha256,
    score_result, validate_integrity,
)


D = 1152
V = 50257
T = 256
BATCH_WINDOWS = 4
FIT_BATCH = 4
AUTHORITY = BQ / "mlp0_c512_mlp1_interchange_v2_eval_authority.json"
OUT = BQ / "mlp0_c512_mlp1_interchange_v2_results.json"
FAILURE = BQ / "mlp0_c512_mlp1_interchange_v2_failure.json"
LOCK = Path("/workspace/runs/.bilin18_mlp0_c512_mlp1_interchange_v2.lock")
FIT_RECEIPT = BQ / "mlp0_native_down_hierarchy_v1_fit_receipt.json"
STAGE0_FIT_RECEIPT = BQ / "mlp0_quotient_stage0_v2_fit_receipt.json"
STAGE0_ROW_RECEIPT = BQ / "mlp0_quotient_stage0_v1_rows_receipt.json"
CODE_REGISTER = PC / "code_oracle_corpus_v2.pt"
PROGRAM_KEY = "C512_at_C512"
CELL_NAMES = [
    f"pos{pos}_freq{freq}_prev{prev}_dev{dev}"
    for pos in range(2) for freq in range(2) for prev in range(2) for dev in range(2)
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def closure_sha256(source_hashes: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_unit_identity(row_receipt: Mapping[str, object], code_manifest: Mapping[str, object]) -> dict[str, object]:
    records = row_receipt["document_provenance"]["sets"]["eval"]
    by_ordinal: dict[int, str] = {}
    row_units = []
    for record in records:
        ordinal = int(record["source_document_ordinal"])
        document = str(record["document_id"])
        if ordinal in by_ordinal and by_ordinal[ordinal] != document:
            raise RuntimeError("FineWeb ordinal maps to multiple source documents")
        by_ordinal[ordinal] = document
        row_units.append(ordinal)
    if sorted(by_ordinal) != list(range(384)):
        raise RuntimeError("FineWeb source-document ordinals changed")
    fineweb_mapping = row_units + row_units

    files = code_manifest["files"]["heldout"]
    ordered_files = [str(record["path"]) for record in files]
    if len(ordered_files) != 48 or len(set(ordered_files)) != 48:
        raise RuntimeError("code heldout file identity changed")
    file_index = {path: index for index, path in enumerate(ordered_files)}
    code_records = code_manifest["row_provenance"]["heldout"]
    code_mapping = [file_index[str(record["path"])] for record in code_records]
    identity = {
        "fineweb": {
            "unit_kind": "source_document",
            "ordered_ids": [by_ordinal[index] for index in range(384)],
            "row_to_unit": fineweb_mapping,
        },
        "code": {
            "unit_kind": "source_file",
            "ordered_ids": ordered_files,
            "row_to_unit": code_mapping,
        },
    }
    return identity


def unit_identity_hashes(identity: Mapping[str, object]) -> dict[str, object]:
    output = {}
    for domain, record in identity.items():
        mapping = np.asarray(record["row_to_unit"], dtype=np.int64)
        occupancy = np.bincount(mapping, minlength=len(record["ordered_ids"]))
        output[domain] = {
            "ordered_ids_sha256": ordered_ids_sha256(record["ordered_ids"]),
            "row_to_unit_sha256": integer_array_sha256(mapping),
            "occupancy_sha256": integer_array_sha256(occupancy),
        }
    return output


def expected_call_counts(n_fit_rows: int, domain_rows: Mapping[str, int]) -> dict[str, int]:
    evaluation_batches = sum(math.ceil(value / BATCH_WINDOWS) for value in domain_rows.values())
    return {
        "candidate_original_down_calls": 0,
        "poison_canary_calls": 1,
        "mlp1_teacher_calls": math.ceil(n_fit_rows / FIT_BATCH) + 8 * evaluation_batches,
        "c512_proxy_calls": 4 * evaluation_batches,
    }


def coverage_by_unit_partition(
    valid: torch.Tensor, unit_ids: torch.Tensor, split_unit: int,
) -> dict[str, float]:
    """Report preregistered FineWeb waves by source-document identity, not row order."""
    if valid.ndim != 2 or unit_ids.shape != (valid.shape[0],):
        raise ValueError("coverage tensors have incompatible shapes")
    first = unit_ids < split_unit
    second = ~first
    if not bool(first.any()) or not bool(second.any()):
        raise ValueError("both source-unit partitions must be represented")
    return {
        "wave_A": float(valid[first].float().mean()),
        "wave_B": float(valid[second].float().mean()),
        "pooled": float(valid.float().mean()),
    }


def derangement_groups(name: str, cells: torch.Tensor, unit_ids: torch.Tensor) -> torch.Tensor:
    """Keep FineWeb shuffle donors inside their preregistered replication wave."""
    if cells.ndim != 2 or unit_ids.shape != (cells.shape[0],):
        raise ValueError("derangement tensors have incompatible shapes")
    groups = cells.long()
    if name == "fineweb":
        wave = (unit_ids >= 192).long()[:, None]
        groups = groups + 16 * wave
    elif name != "code":
        raise ValueError("unknown derangement domain")
    return groups


def load_domains() -> tuple[dict[str, dict[str, Any]], dict[str, object], torch.Tensor, dict[str, object]]:
    row_receipt, rows = load_frozen_rows()
    records = row_receipt["document_provenance"]["sets"]["eval"]
    fineweb_windows = torch.cat([rows[:, :257], rows[:, 256:513]], dim=0).contiguous()
    code_artifact = torch.load(CODE_REGISTER, map_location="cpu", weights_only=False)
    code_rows = code_artifact["rows"][288:480].contiguous()
    identity = build_unit_identity(row_receipt, code_artifact["manifest"])
    domains = {
        "fineweb": {
            "rows": fineweb_windows,
            "unit_ids": torch.tensor(identity["fineweb"]["row_to_unit"], dtype=torch.long),
            "n_units": 384,
        },
        "code": {
            "rows": code_rows,
            "unit_ids": torch.tensor(identity["code"]["row_to_unit"], dtype=torch.long),
            "n_units": 48,
        },
    }
    if tuple(fineweb_windows.shape) != (1170, 257) or tuple(code_rows.shape) != (192, 257):
        raise RuntimeError("registered domain row shapes changed")
    return domains, identity, rows, code_artifact


class C512Down(nn.Module):
    def __init__(self, program: Mapping[str, object]):
        super().__init__()
        if int(program["rank"]) != 512 or program["centroids"].shape[0] != 0:
            raise RuntimeError("program is not the frozen continuous C512 arm")
        self.calls = 0
        self.register_buffer("intercept", program["intercept"].float())
        self.register_buffer("left", program["left"].float())
        self.register_buffer("right", program["right"].float())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return ((hidden @ self.right.to(hidden.dtype).T) @ self.left.to(hidden.dtype).T
                + self.intercept.to(hidden.dtype))


@torch.no_grad()
def full_forward(model: Any, blocks: Any, idx: torch.Tensor, background: str) -> tuple[torch.Tensor, torch.Tensor]:
    if background not in BACKGROUNDS:
        raise ValueError("unknown full-forward background")
    x = F.rms_norm(model.transformer.wte(idx), (D,))
    x0, v1 = x, None
    for layer, block in enumerate(blocks):
        if background == "mlp2_omit" and layer == 2:
            x = block.lambdas[0] * x + block.lambdas[1] * x0
            attention, v1 = block.attn(F.rms_norm(x, (D,)), v1)
            x = x + attention
        else:
            x, v1 = block(x, v1, x0)
    raw = model.lm_head(F.rms_norm(x, (D,))).float()
    return raw, 30.0 * torch.tanh(raw / 30.0)


def punctuation_table() -> torch.Tensor:
    import tiktoken
    encoder = tiktoken.get_encoding("gpt2")
    values = torch.zeros(V, dtype=torch.bool)
    for token in range(V):
        raw = encoder.decode([token])
        stripped = raw.strip()
        values[token] = bool("\n" in raw or stripped == "" or re.fullmatch(r"[^\w\s]+", stripped))
    return values


def cell_map(
    idx: torch.Tensor, pre_mlp0: torch.Tensor, token_count: torch.Tensor,
    punctuation: torch.Tensor, frequency_median: float, norm_median: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    position = torch.arange(T, device=idx.device).view(1, T).expand_as(idx)
    previous = torch.cat([torch.full_like(idx[:, :1], -1), idx[:, :-1]], dim=1)
    prev_kind = (previous < 0) | punctuation[previous.clamp_min(0)]
    frequency_kind = token_count[idx] > frequency_median
    deviation_kind = pre_mlp0.norm(dim=-1) > norm_median
    position_kind = position >= T // 2
    cells = (position_kind.long() * 8 + frequency_kind.long() * 4
             + prev_kind.long() * 2 + deviation_kind.long())
    return cells, token_count[idx] > 0


def empty_ledgers(n_units: int) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    return {
        arm: {
            consumer: {
                "sums": torch.zeros(n_units, 16, dtype=torch.float64),
                "counts": torch.zeros(n_units, 16, dtype=torch.float64),
            }
            for consumer in MARGINS
        }
        for arm in ARMS
    }


def add_unit_cells(
    ledger: Mapping[str, torch.Tensor], unit_ids: torch.Tensor, cells: torch.Tensor,
    valid: torch.Tensor, effects: torch.Tensor,
) -> None:
    for row in range(len(unit_ids)):
        unit = int(unit_ids[row])
        for cell_id in range(16):
            selected = valid[row] & (cells[row] == cell_id)
            count = int(selected.sum())
            if count:
                ledger["counts"][unit, cell_id] += count
                ledger["sums"][unit, cell_id] += float(effects[row][selected].sum())


def pair_effects(
    reference: torch.Tensor, candidate: torch.Tensor, target: torch.Tensor,
    logit_scale: float,
) -> dict[str, torch.Tensor]:
    ref_logp = F.log_softmax(reference, dim=-1)
    can_logp = F.log_softmax(candidate, dim=-1)
    kl = (ref_logp.exp() * (ref_logp - can_logp)).sum(-1)
    ref_ce = F.cross_entropy(reference.flatten(0, 1), target.flatten(), reduction="none").view_as(target)
    can_ce = F.cross_entropy(candidate.flatten(0, 1), target.flatten(), reduction="none").view_as(target)
    centered_difference = ((candidate - candidate.mean(-1, keepdim=True))
                           - (reference - reference.mean(-1, keepdim=True)))
    nrmse = centered_difference.square().mean(-1).sqrt() / logit_scale
    return {"kl": kl, "ce_abs": can_ce - ref_ce, "centered_logit_nrmse": nrmse}


def contrast_logits(logits: Mapping[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    additive = additive_interaction_prediction(logits)
    return {
        "observational_CC": (logits["OO"], logits["CC"]),
        "write_on_O": (logits["OO"], logits["OC"]),
        "write_on_C": (logits["CO"], logits["CC"]),
        "upstream_state": (logits["OO"], logits["CO"]),
        "interaction": (logits["CC"], additive),
        "shuffle": (logits["OO"], logits["shuffle"]),
        "native_write": (logits["OO"], logits["native_write"]),
    }


def verify_authority_file() -> dict[str, object]:
    if not AUTHORITY.is_file():
        raise RuntimeError("frozen evaluation authority is absent")
    relative = str(AUTHORITY.relative_to(ROOT))
    blob = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)
    if hashlib.sha256(blob).hexdigest() != file_sha256(AUTHORITY):
        raise RuntimeError("authority file is not byte-identical to committed HEAD")
    authority = json.loads(AUTHORITY.read_text())
    if (authority.get("status") != "frozen_before_any_c512_mlp1_evaluation_forward"
            or authority.get("output_path") != str(OUT)
            or authority.get("failure_path") != str(FAILURE)):
        raise RuntimeError("authority status or namespace changed")
    for raw, expected in authority.get("source_hashes", {}).items():
        if file_sha256(Path(raw)) != expected:
            raise RuntimeError(f"authority-bound source changed: {raw}")
    for raw, expected in authority.get("model_files", {}).items():
        if file_sha256(Path(raw)) != expected:
            raise RuntimeError(f"authority-bound model changed: {raw}")
    if OUT.exists() or FAILURE.exists():
        raise RuntimeError("evaluation namespace is already spent")
    return authority


def verify_preflight_artifacts(
    authority: Mapping[str, object], frozen_rows: torch.Tensor, fit_rows: torch.Tensor,
    program_path: Path,
) -> dict[str, str]:
    """Validate all outcome-bearing artifacts before the first evaluation forward."""
    roles = authority.get("model_file_roles", {})
    if set(roles) != {"config", "checkpoint"}:
        raise RuntimeError("authority must bind explicit config/checkpoint roles")
    if len(set(roles.values())) != 2:
        raise RuntimeError("authority model roles must be distinct")
    if any(raw not in authority["model_files"] for raw in roles.values()):
        raise RuntimeError("authority model role is outside model files")
    model_checkpoint = Path(roles["checkpoint"])
    source_hashes = {raw: file_sha256(Path(raw)) for raw in authority["source_hashes"]}
    observed = {
        "source_closure_sha256": closure_sha256(source_hashes),
        "row_receipt_sha256": file_sha256(ROW_RECEIPT),
        "row_tensor_sha256": tensor_sha256(frozen_rows),
        "c512_program_sha256": file_sha256(program_path),
        "model_checkpoint_sha256": file_sha256(model_checkpoint),
        "code_register_sha256": file_sha256(CODE_REGISTER),
    }
    if observed != authority["integrity_contract"]["bound_hashes"]:
        raise RuntimeError("preflight artifacts differ from frozen authority")
    fit_contract = authority.get("fit_authority", {})
    expected_fit = {
        "fit_receipt_sha256": file_sha256(FIT_RECEIPT),
        "stage0_row_receipt_sha256": file_sha256(STAGE0_ROW_RECEIPT),
        "stage0_fit_receipt_sha256": file_sha256(STAGE0_FIT_RECEIPT),
        "fit_rows_tensor_sha256": tensor_sha256(fit_rows),
        "fit_rows": int(len(fit_rows)),
    }
    if expected_fit != fit_contract:
        raise RuntimeError("fit-frozen scaling/cell authority changed")
    return observed


def load_authority_model(authority: Mapping[str, object]) -> Any:
    """Instantiate only from the exact config/checkpoint files hashed in authority."""
    roles = authority["model_file_roles"]
    config_payload = json.loads(Path(roles["config"]).read_text())
    config_payload.pop("step", None)
    import jacclust.tt_model as TT
    model = TT.GPT(TT.GPTConfig(**config_payload)).eval()
    state = torch.load(roles["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device="cuda", dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def install_native(mlp: Any, original: Any, original_forward: Any) -> None:
    original.forward = original_forward
    mlp.Down = original


def install_candidate(mlp: Any, proxy: Any, original: Any, poisoned: Any) -> None:
    original.forward = poisoned
    mlp.Down = proxy


@torch.no_grad()
def fit_logit_scale(model: Any, blocks: Any, fit_rows: torch.Tensor) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(fit_rows), FIT_BATCH):
        idx = fit_rows[start:start + FIT_BATCH, :-1].to("cuda").contiguous()
        _, logits = full_forward(model, blocks, idx, "live")
        centered = logits - logits.mean(-1, keepdim=True)
        total += float(centered.double().square().sum())
        count += centered.numel()
    scale = math.sqrt(total / count)
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("fit-frozen centered-logit RMS is invalid")
    return scale


@torch.no_grad()
def prepare_domain(
    name: str, domain: Mapping[str, Any], model: Any, blocks: Any, mlp0: Any,
    original: Any, original_forward: Any, proxy: C512Down, poisoned: Any,
    token_count: torch.Tensor, punctuation: torch.Tensor,
    frequency_median: float, norm_median: float,
) -> dict[str, Any]:
    rows = domain["rows"]
    unit_ids = domain["unit_ids"]
    delta = torch.empty(len(rows), T, D, dtype=torch.float32)
    cells_all = torch.empty(len(rows), T, dtype=torch.uint8)
    valid_all = torch.empty(len(rows), T, dtype=torch.bool)
    state_identity_max = {"x0": 0.0, "v1": 0.0}
    for start in range(0, len(rows), BATCH_WINDOWS):
        stop = min(start + BATCH_WINDOWS, len(rows))
        idx = rows[start:stop, :-1].to("cuda").contiguous()
        install_native(mlp0, original, original_forward)
        exact = capture_through_mlp1(model, blocks, idx)
        install_candidate(mlp0, proxy, original, poisoned)
        candidate = capture_through_mlp1(model, blocks, idx)
        cells, valid = cell_map(
            idx, exact["pre_mlp0"], token_count, punctuation, frequency_median, norm_median
        )
        delta[start:stop] = (candidate["m"] - exact["m"]).cpu()
        cells_all[start:stop] = cells.byte().cpu()
        valid_all[start:stop] = valid.cpu()
        for key in state_identity_max:
            state_identity_max[key] = max(
                state_identity_max[key], float((candidate[key] - exact[key]).abs().max())
            )
    flat_units = unit_ids[:, None].expand(-1, T).reshape(-1)
    groups = derangement_groups(name, cells_all.long(), unit_ids)
    permutation = document_derangement(flat_units, groups.reshape(-1))
    coverage = float(valid_all.sum() / valid_all.numel())
    print(f"prepared {name}: rows={len(rows)} coverage={coverage:.6f}", flush=True)
    return {
        "delta": delta, "cells": cells_all.long(), "valid": valid_all,
        "permutation": permutation, "coverage": coverage,
        "state_identity_max": state_identity_max,
    }


@torch.no_grad()
def evaluate_domain(
    name: str, domain: Mapping[str, Any], prepared: Mapping[str, Any],
    model: Any, blocks: Any, mlp0: Any, original: Any, original_forward: Any,
    proxy: C512Down, poisoned: Any, logit_scale: float,
    replay: dict[str, dict[str, float]],
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    rows, unit_ids = domain["rows"], domain["unit_ids"]
    ledgers = empty_ledgers(int(domain["n_units"]))
    flat_delta = prepared["delta"].reshape(-1, D)
    permutation = prepared["permutation"]
    for start in range(0, len(rows), BATCH_WINDOWS):
        stop = min(start + BATCH_WINDOWS, len(rows))
        idx = rows[start:stop, :-1].to("cuda").contiguous()
        target = rows[start:stop, 1:].to("cuda").contiguous()
        install_native(mlp0, original, original_forward)
        exact = capture_through_mlp1(model, blocks, idx)
        install_candidate(mlp0, proxy, original, poisoned)
        candidate = capture_through_mlp1(model, blocks, idx)
        states = physical_post_states(exact, candidate)
        position_index = torch.arange(start * T, stop * T)
        shuffled = flat_delta[permutation[position_index]].reshape(stop - start, T, D).to("cuda")
        native_control = norm_matched_native_write(candidate["m"] - exact["m"], exact["m"])
        controls = {
            "shuffle": states["OO"] + shuffled,
            "native_write": states["OO"] + native_control,
        }
        for background in BACKGROUNDS:
            outputs: dict[str, torch.Tensor] = {}
            raw_replay = {}
            for arm in ("OO", "OC"):
                raw, capped = suffix_forward(
                    model, blocks, states[arm], exact["v1"], exact["x0"],
                    background=background, return_raw=True,
                )
                raw_replay[arm], outputs[arm] = raw, capped
            for arm in ("CC", "CO"):
                raw, capped = suffix_forward(
                    model, blocks, states[arm], candidate["v1"], candidate["x0"],
                    background=background, return_raw=True,
                )
                raw_replay[arm], outputs[arm] = raw, capped
            for arm, post in controls.items():
                outputs[arm] = suffix_forward(
                    model, blocks, post, exact["v1"], exact["x0"], background=background
                )

            install_native(mlp0, original, original_forward)
            raw_o, full_o = full_forward(model, blocks, idx, background)
            install_candidate(mlp0, proxy, original, poisoned)
            raw_c, full_c = full_forward(model, blocks, idx, background)
            for arm, parent_raw, parent_capped in (
                ("OO", raw_o, full_o), ("CC", raw_c, full_c),
            ):
                replay[background]["raw_logits_max_abs"] = max(
                    replay[background]["raw_logits_max_abs"],
                    float((raw_replay[arm] - parent_raw).abs().max()),
                )
                replay[background]["capped_logits_max_abs"] = max(
                    replay[background]["capped_logits_max_abs"],
                    float((outputs[arm] - parent_capped).abs().max()),
                )
                replay_ce = F.cross_entropy(outputs[arm].flatten(0, 1), target.flatten())
                parent_ce = F.cross_entropy(parent_capped.flatten(0, 1), target.flatten())
                replay[background]["ce_abs"] = max(
                    replay[background]["ce_abs"], float((replay_ce - parent_ce).abs())
                )

            for contrast, (reference, arm_logits) in contrast_logits(outputs).items():
                effects = pair_effects(reference, arm_logits, target, logit_scale)
                ledger_arm = f"{background}/{contrast}"
                for consumer, values in effects.items():
                    add_unit_cells(
                        ledgers[ledger_arm][consumer], unit_ids[start:stop],
                        prepared["cells"][start:stop], prepared["valid"][start:stop],
                        values.cpu(),
                    )
        print(f"evaluated {name} rows {stop}/{len(rows)}", flush=True)
    return ledgers


def poison_canary(mlp0: Any, original: Any, original_forward: Any) -> int:
    calls = 0
    def poison(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("C512 interchange original Down poison canary")
    original.forward = poison
    mlp0.Down = original
    try:
        mlp0(torch.zeros(1, 1, D, device="cuda"))
    except RuntimeError as error:
        if "original Down poison canary" not in str(error):
            raise
    finally:
        install_native(mlp0, original, original_forward)
    if calls != 1:
        raise RuntimeError("poison canary did not raise exactly once")
    return calls


@torch.no_grad()
def main() -> None:
    started = time.time()
    authority = verify_authority_file()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    original = original_forward = None
    teacher_hook = None
    try:
        domains, identity, frozen_rows, code_artifact = load_domains()
        if unit_identity_hashes(identity) != authority["integrity_contract"]["unit_identity_hashes"]:
            raise RuntimeError("runtime unit identity differs from frozen authority")
        _, fit_full = load_frozen_role("fit")
        fit_rows = fit_full[:, :257].contiguous()
        fit_constants = json.loads(STAGE0_FIT_RECEIPT.read_text())["constants"]
        frequency_median = float(fit_constants["frequency_median"])
        norm_median = float(fit_constants["pre_mlp0_raw_residual_norm_median"])
        fit_tokens = fit_rows[:, :-1].reshape(-1)
        token_count = torch.bincount(fit_tokens, minlength=V).float().to("cuda")
        punctuation = punctuation_table().to("cuda")

        fit_receipt = json.loads(FIT_RECEIPT.read_text())
        program_receipt = fit_receipt["programs"][PROGRAM_KEY]
        program_path = Path(program_receipt["path"])
        if (file_sha256(program_path) != program_receipt["sha256"]
                or program_path.stat().st_size != program_receipt["bytes"]):
            raise RuntimeError("C512 bundle differs from frozen fit receipt")
        observed_hashes = verify_preflight_artifacts(authority, frozen_rows, fit_rows, program_path)

        model = load_authority_model(authority)
        blocks = model.transformer.h
        mlp0 = blocks[0].mlp
        original = mlp0.Down
        original_forward = original.forward
        proxy = C512Down(load_program(program_path)).to("cuda")

        original_poison_calls = 0
        def poisoned_original(*args, **kwargs):
            nonlocal original_poison_calls
            original_poison_calls += 1
            raise RuntimeError("candidate reached poisoned original Down")

        teacher_calls = 0
        def teacher_counter(module, args, output):
            nonlocal teacher_calls
            teacher_calls += 1
        teacher_hook = blocks[1].mlp.register_forward_hook(teacher_counter)

        install_native(mlp0, original, original_forward)
        logit_scale = fit_logit_scale(model, blocks, fit_rows)
        canary_calls = poison_canary(mlp0, original, original_forward)

        prepared = {}
        for name, domain in domains.items():
            prepared[name] = prepare_domain(
                name, domain, model, blocks, mlp0, original, original_forward, proxy,
                poisoned_original, token_count, punctuation, frequency_median, norm_median,
            )
        state_tolerance = float(authority["integrity_contract"]["state_identity_tolerance"])
        if (not math.isfinite(state_tolerance) or state_tolerance < 0
                or any(value > state_tolerance
                       for item in prepared.values() for value in item["state_identity_max"].values())):
            raise RuntimeError("factorial parent x0/v1 state identity check failed")
        replay = {
            background: {"raw_logits_max_abs": 0.0, "capped_logits_max_abs": 0.0, "ce_abs": 0.0}
            for background in BACKGROUNDS
        }
        ledgers = {}
        for name, domain in domains.items():
            ledgers[name] = evaluate_domain(
                name, domain, prepared[name], model, blocks, mlp0, original,
                original_forward, proxy, poisoned_original, logit_scale, replay,
            )
        install_native(mlp0, original, original_forward)

        tolerance = authority["integrity_contract"]["parent_replay_tolerances"]
        for background in BACKGROUNDS:
            replay[background]["passes"] = all(
                replay[background][key] <= tolerance[key]
                for key in ("raw_logits_max_abs", "capped_logits_max_abs", "ce_abs")
            )
        integrity = {
            "call_counts": {
                "candidate_original_down_calls": original_poison_calls,
                "poison_canary_calls": canary_calls,
                "mlp1_teacher_calls": teacher_calls,
                "c512_proxy_calls": proxy.calls,
            },
            "observed_hashes": observed_hashes,
            "parent_replay": replay,
        }
        if not validate_integrity(authority, integrity):
            raise RuntimeError(f"runtime integrity differs from frozen authority: {integrity}")

        coverage = {
            "fineweb": coverage_by_unit_partition(
                prepared["fineweb"]["valid"], domains["fineweb"]["unit_ids"], 192
            ),
            "code": prepared["code"]["coverage"],
        }
        raw = {
            "schema_version": 1,
            "experiment": "mlp0_c512_mlp1_interchange_v1",
            "authority": authority,
            "authority_file_sha256": file_sha256(AUTHORITY),
            "unit_identity": identity,
            "rows": {
                "fineweb_receipt_sha256": file_sha256(ROW_RECEIPT),
                "fineweb_tensor_sha256": tensor_sha256(frozen_rows),
                "code_register_sha256": file_sha256(CODE_REGISTER),
            },
            "program": program_receipt,
            "fit_frozen_centered_logit_rms": logit_scale,
            "cell_names": CELL_NAMES,
            "coverage": coverage,
            "integrity": integrity,
            "state_identity_max": {
                name: values["state_identity_max"] for name, values in prepared.items()
            },
            "sufficient_statistics": {
                domain: {
                    arm: {
                        consumer: {
                            "sums": values["sums"].tolist(),
                            "counts": values["counts"].tolist(),
                        }
                        for consumer, values in consumers.items()
                    }
                    for arm, consumers in domain_ledgers.items()
                }
                for domain, domain_ledgers in ledgers.items()
            },
        }
        raw["inference"] = score_result(raw)
        raw["runtime_s"] = time.time() - started
        write_json_atomic(raw, OUT)
        print(json.dumps({
            "coverage": coverage, "integrity": integrity,
            "decisions": raw["inference"]["decisions"], "runtime_s": raw["runtime_s"],
        }, indent=2), flush=True)
        print(f"wrote {OUT}", flush=True)
    finally:
        if teacher_hook is not None:
            teacher_hook.remove()
        if original is not None and original_forward is not None:
            try:
                install_native(mlp0, original, original_forward)
            except Exception:
                pass
        os.close(lock_fd)
        LOCK.unlink(missing_ok=True)


def authoritative_entry() -> None:
    try:
        main()
    except BaseException as error:
        if not OUT.exists() and not FAILURE.exists():
            write_json_atomic({
                "schema_version": 1,
                "status": "failed_closed_without_scientific_result",
                "error_type": type(error).__name__, "error": str(error),
                "authority_sha256": file_sha256(AUTHORITY) if AUTHORITY.exists() else None,
            }, FAILURE)
        LOCK.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    authoritative_entry()
