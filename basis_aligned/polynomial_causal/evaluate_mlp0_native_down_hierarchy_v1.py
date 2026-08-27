#!/usr/bin/env python3
"""Authority-bound executable evaluation of priced MLP0 native-Down programs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
PC = ROOT / "basis_aligned" / "polynomial_causal"
sys.path.insert(0, str(BQ))
sys.path.insert(0, str(PC))

from mlp0_native_down_program import load_program  # noqa: E402
from prepare_mlp0_native_down_hierarchy_v1_rows import (  # noqa: E402
    RECEIPT as ROW_RECEIPT,
    load_frozen_rows,
)
from prepare_mlp0_quotient_stage0_v1_rows import load_frozen_role  # noqa: E402
from score_mlp0_native_down_hierarchy_v1 import (  # noqa: E402
    MARGINS,
    score_result,
)


D = 1152
HIDDEN = 4608
V = 50257
T = 256
BATCH_CHUNKS = 2
AUTHORITY = BQ / "mlp0_native_down_hierarchy_v1_eval_authority.json"
FIT_RECEIPT = BQ / "mlp0_native_down_hierarchy_v1_fit_receipt.json"
STAGE0_ROW_RECEIPT = BQ / "mlp0_quotient_stage0_v1_rows_receipt.json"
STAGE0_FIT_RECEIPT = BQ / "mlp0_quotient_stage0_v2_fit_receipt.json"
OUT = BQ / "mlp0_native_down_hierarchy_v1_results.json"
FAILURE = BQ / "mlp0_native_down_hierarchy_v1_eval_failure.json"
LOCK = Path("/workspace/runs/.bilin18_mlp0_native_down_eval_v1.lock")
CELL_NAMES = [
    f"pos{pos}_freq{freq}_prev{prev}_dev{dev}"
    for pos in range(2) for freq in range(2) for prev in range(2) for dev in range(2)
]
PROGRAM_KEYS = {
    "C256": "C256_at_C256", "C512": "C512_at_C512",
    "Q248": "Q248_at_C256", "Q504": "Q504_at_C512",
    "Qnull248": "Qnull248_at_C256", "Qnull504": "Qnull504_at_C512",
    "A247": "A247_at_C256", "A503": "A503_at_C512",
    "Anull247": "Anull247_at_C256", "Anull503": "Anull503_at_C512",
}
STATE = {"idx": None, "caps": {}, "block0_stream": None}


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
        raise RuntimeError("evaluation authority absent")
    authority = json.loads(AUTHORITY.read_text())
    if (authority.get("status") != "frozen_before_any_native_down_evaluation_forward"
            or authority.get("output_path") != str(OUT)
            or authority.get("row_receipt_sha256") != file_sha256(ROW_RECEIPT)
            or authority.get("fit_receipt_sha256") != file_sha256(FIT_RECEIPT)
            or authority.get("stage0_row_receipt_sha256") != file_sha256(STAGE0_ROW_RECEIPT)
            or authority.get("stage0_fit_receipt_sha256") != file_sha256(STAGE0_FIT_RECEIPT)):
        raise RuntimeError("evaluation authority identity/status mismatch")
    for raw, expected in authority.get("source_hashes", {}).items():
        if file_sha256(Path(raw)) != expected:
            raise RuntimeError(f"authority-bound source changed: {raw}")
    for raw, expected in authority.get("model_files", {}).items():
        if file_sha256(Path(raw)) != expected:
            raise RuntimeError(f"authority-bound model file changed: {raw}")
    for raw, expected in authority.get("program_files", {}).items():
        if file_sha256(Path(raw)) != expected:
            raise RuntimeError(f"authority-bound program changed: {raw}")
    if OUT.exists() or FAILURE.exists():
        raise RuntimeError("evaluation namespace is already spent")
    return authority


class ProgramDown(nn.Module):
    def __init__(self, program: dict, arm: str):
        super().__init__()
        self.arm = arm
        self.calls = 0
        self.register_buffer("intercept", program["intercept"].float())
        self.register_buffer("left", program["left"].float())
        self.register_buffer("right", program["right"].float())
        self.register_buffer("centroids", program["centroids"].float())
        self.register_buffer("assignments", program["assignments"].long())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        low_rank = (h @ self.right.to(h.dtype).T) @ self.left.to(h.dtype).T
        output = low_rank + self.intercept.to(h.dtype)
        if self.centroids.shape[0]:
            idx = STATE.get("idx")
            if idx is None or tuple(idx.shape) != tuple(h.shape[:2]):
                raise RuntimeError("candidate token side-channel is absent or misaligned")
            codes = self.assignments[idx]
            sentinel = self.centroids.shape[0]
            safe = codes.clamp_max(sentinel - 1)
            baseline = self.centroids[safe]
            baseline = torch.where((codes == sentinel).unsqueeze(-1), 0.0, baseline)
            output = output + baseline.to(h.dtype)
        return output


class ClonedNativeDown(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())
        self.calls = 0

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return F.linear(h, self.weight.to(h.dtype))


def split_chunks(rows: torch.Tensor, records: list[dict], start: int, stop: int):
    chunk = rows[start:stop]
    windows = torch.cat([chunk[:, :257], chunk[:, 256:513]], dim=0)
    ordinals = torch.tensor(
        [records[index]["source_document_ordinal"] for index in range(start, stop)] * 2,
        dtype=torch.long,
    )
    return windows, ordinals


def block0_pre_hook(module, args):
    x, _, x0 = args
    STATE["block0_stream"] = (module.lambdas[0] * x + module.lambdas[1] * x0).detach().float()


def attn0_hook(module, args, output):
    value = output[0] if isinstance(output, tuple) else output
    stream = STATE.get("block0_stream")
    if stream is None:
        raise RuntimeError("block-0 stream capture missing")
    STATE["caps"]["pre_mlp0"] = stream + value.detach().float()


def m0_hook(module, args, output):
    STATE["caps"]["m0"] = output.detach().float()


def attn1_hook(module, args, output):
    value = output[0] if isinstance(output, tuple) else output
    STATE["caps"]["attn1"] = value.detach().float()


def mlp1_hook(module, args, output):
    STATE["caps"]["mlp1"] = output.detach().float()


def register_hooks(blocks):
    return [
        blocks[0].register_forward_pre_hook(block0_pre_hook),
        blocks[0].attn.register_forward_hook(attn0_hook),
        blocks[0].mlp.register_forward_hook(m0_hook),
        blocks[1].attn.register_forward_hook(attn1_hook),
        blocks[1].mlp.register_forward_hook(mlp1_hook),
    ]


@torch.no_grad()
def fwd(model, blocks, idx: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    STATE["idx"] = idx
    STATE["caps"] = {}
    x = F.rms_norm(model.transformer.wte(idx), (D,))
    x0, v1 = x, None
    for block in blocks:
        x, v1 = block(x, v1, x0)
    logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0)
    required = {"pre_mlp0", "m0", "attn1", "mlp1"}
    if set(STATE["caps"]) != required:
        raise RuntimeError(f"capture failure: {sorted(STATE['caps'])}")
    return logits.float(), {name: value.clone() for name, value in STATE["caps"].items()}


def punctuation_table() -> torch.Tensor:
    import tiktoken
    encoder = tiktoken.get_encoding("gpt2")
    values = torch.zeros(V, dtype=torch.bool)
    for token in range(V):
        raw = encoder.decode([token])
        stripped = raw.strip()
        values[token] = bool("\n" in raw or stripped == "" or re.fullmatch(r"[^\w\s]+", stripped))
    return values


def empty_ledgers() -> dict[str, dict[str, torch.Tensor]]:
    return {
        consumer: {
            "sums": torch.zeros(384, 16, dtype=torch.float64),
            "counts": torch.zeros(384, 16, dtype=torch.float64),
        }
        for consumer in MARGINS
    }


def add_document_cells(ledger, ordinals, cell, valid, effects):
    for row in range(valid.shape[0]):
        document = int(ordinals[row])
        for cell_id in range(16):
            selected = valid[row] & (cell[row] == cell_id)
            count = int(selected.sum())
            if count:
                ledger["counts"][document, cell_id] += count
                ledger["sums"][document, cell_id] += float(effects[row][selected].sum())


def compute_effects(reference, candidate, target, scales):
    ref_logits, ref_cap = reference
    can_logits, can_cap = candidate
    ref_logp = F.log_softmax(ref_logits, dim=-1)
    can_logp = F.log_softmax(can_logits, dim=-1)
    kl = (ref_logp.exp() * (ref_logp - can_logp)).sum(-1)
    ref_ce = F.cross_entropy(ref_logits.flatten(0, 1), target.flatten(), reduction="none").view_as(target)
    can_ce = F.cross_entropy(can_logits.flatten(0, 1), target.flatten(), reduction="none").view_as(target)
    attn = (can_cap["attn1"] - ref_cap["attn1"]).pow(2).mean(-1).sqrt() / scales["attn1"]
    mlp = (can_cap["mlp1"] - ref_cap["mlp1"]).pow(2).mean(-1).sqrt() / scales["mlp1"]
    return {"kl": kl, "ce": can_ce - ref_ce, "attn1_nrmse": attn, "mlp1_nrmse": mlp}


def integrity_report(original, clone, target) -> dict:
    original_logits, original_caps = original
    clone_logits, clone_caps = clone
    ce_original = F.cross_entropy(original_logits.flatten(0, 1), target.flatten())
    ce_clone = F.cross_entropy(clone_logits.flatten(0, 1), target.flatten())
    report = {
        "logits_max_abs": float((clone_logits - original_logits).abs().max()),
        "ce_abs": float((ce_clone - ce_original).abs()),
        "caps_max_abs": {
            name: float((clone_caps[name] - original_caps[name]).abs().max())
            for name in ("m0", "attn1", "mlp1")
        },
    }
    report["passes"] = bool(
        report["logits_max_abs"] <= 1e-6 and report["ce_abs"] <= 1e-7
        and max(report["caps_max_abs"].values()) <= 1e-6
    )
    return report


def poison_canary(mlp, original, original_forward) -> dict:
    calls = {"poison": 0}
    def poisoned(*args, **kwargs):
        calls["poison"] += 1
        raise RuntimeError("original Down poison canary")
    original.forward = poisoned
    mlp.Down = original
    raised = False
    try:
        mlp(torch.zeros(1, 1, D, device=next(mlp.parameters()).device))
    except RuntimeError as error:
        raised = "original Down poison canary" in str(error)
    finally:
        original.forward = original_forward
    if not raised or calls["poison"] != 1:
        raise RuntimeError("original Down poison canary failed")
    return {"raised": raised, "poison_calls": calls["poison"]}


@torch.no_grad()
def main() -> None:
    started = time.time()
    authority = validate_authority()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    hooks = []
    original = original_forward = None
    try:
        row_receipt, rows = load_frozen_rows()
        records = row_receipt["document_provenance"]["sets"]["eval"]
        _, fit_full = load_frozen_role("fit")
        fit_rows = fit_full[:, :257].contiguous()
        fit_receipt = json.loads(FIT_RECEIPT.read_text())
        if tensor_sha256(fit_rows) != fit_receipt["fit_rows"]["sha256"]:
            raise RuntimeError("Stage-0 fit row bytes differ from native fit authority")
        fit_tokens = fit_rows[:, :-1].reshape(-1)
        token_count = torch.bincount(fit_tokens, minlength=V).float().to("cuda")
        fit_constants = json.loads(STAGE0_FIT_RECEIPT.read_text())["constants"]
        frequency_median = float(fit_constants["frequency_median"])
        norm_median = float(fit_constants["pre_mlp0_raw_residual_norm_median"])
        scales = fit_constants["direct_scales"]

        from bilin18_joint_removal import m as model
        blocks = model.transformer.h
        mlp = blocks[0].mlp
        original = mlp.Down
        original_forward = original.forward
        original_weight = original.weight.detach().clone()
        model_hash_before = tensor_sha256(original_weight)
        hooks = register_hooks(blocks)

        programs = {}
        for arm, key in PROGRAM_KEYS.items():
            receipt = fit_receipt["programs"][key]
            path = Path(receipt["path"])
            if path.stat().st_size != receipt["bytes"] or file_sha256(path) != receipt["sha256"]:
                raise RuntimeError(f"program receipt failed for {arm}")
            programs[arm] = load_program(path)

        first_windows, _ = split_chunks(rows, records, 0, 1)
        first = first_windows.to("cuda")
        idx0, target0 = first[:, :-1].contiguous(), first[:, 1:].contiguous()
        mlp.Down = original
        original_result = fwd(model, blocks, idx0)
        clone = ClonedNativeDown(original_weight).to("cuda")
        mlp.Down = clone
        clone_result = fwd(model, blocks, idx0)
        integrity = integrity_report(original_result, clone_result, target0)
        if not integrity["passes"] or clone.calls != 1:
            raise RuntimeError(f"cloned-native integrity failed: {integrity}")
        canary = poison_canary(mlp, original, original_forward)

        punctuation = punctuation_table().to("cuda")
        ledgers = {arm: empty_ledgers() for arm in PROGRAM_KEYS}
        coverage_counts = {"wave_A": [0, 0], "wave_B": [0, 0]}
        candidate_calls = {arm: 0 for arm in PROGRAM_KEYS}
        original_poison_calls = 0

        def poisoned_original(*args, **kwargs):
            nonlocal original_poison_calls
            original_poison_calls += 1
            raise RuntimeError("candidate reached poisoned original Down")

        for start in range(0, len(rows), BATCH_CHUNKS):
            stop = min(start + BATCH_CHUNKS, len(rows))
            windows, ordinals = split_chunks(rows, records, start, stop)
            batch = windows.to("cuda")
            idx, target = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
            mlp.Down = original
            original.forward = original_forward
            reference = fwd(model, blocks, idx)

            position = torch.arange(T, device="cuda").view(1, T).expand_as(idx)
            previous = torch.cat([torch.full_like(idx[:, :1], -1), idx[:, :-1]], dim=1)
            prev_kind = (previous < 0) | punctuation[previous.clamp_min(0)]
            frequency_kind = token_count[idx] > frequency_median
            deviation_kind = reference[1]["pre_mlp0"].norm(dim=-1) > norm_median
            position_kind = position >= (T // 2)
            cell = (position_kind.long() * 8 + frequency_kind.long() * 4
                    + prev_kind.long() * 2 + deviation_kind.long())
            valid = token_count[idx] > 0
            for row, ordinal in enumerate(ordinals):
                wave = "wave_A" if int(ordinal) < 192 else "wave_B"
                coverage_counts[wave][0] += int(valid[row].sum())
                coverage_counts[wave][1] += valid.shape[1]

            original.forward = poisoned_original
            for arm, program in programs.items():
                proxy = ProgramDown(program, arm).to("cuda")
                mlp.Down = proxy
                candidate = fwd(model, blocks, idx)
                if proxy.calls != 1:
                    raise RuntimeError(f"{arm} proxy call count changed")
                candidate_calls[arm] += proxy.calls
                effects = compute_effects(reference, candidate, target, scales)
                for consumer, values in effects.items():
                    add_document_cells(ledgers[arm][consumer], ordinals, cell, valid, values)
                del proxy, candidate
            if original_poison_calls:
                raise RuntimeError("candidate called original Down")
            print(f"eval chunks {stop}/{len(rows)}", flush=True)

        original.forward = original_forward
        mlp.Down = original
        coverage = {
            wave: covered / total for wave, (covered, total) in coverage_counts.items()
        }
        coverage["pooled"] = sum(value[0] for value in coverage_counts.values()) / sum(
            value[1] for value in coverage_counts.values()
        )
        raw = {
            "schema_version": 1,
            "experiment": "mlp0_native_down_hierarchy_v1",
            "authority": authority,
            "rows": {"receipt_path": str(ROW_RECEIPT), "receipt_sha256": file_sha256(ROW_RECEIPT),
                     "shape": list(rows.shape), "sha256": tensor_sha256(rows),
                     "window_semantics": "[0:257] and [256:513] per chunk"},
            "construction": fit_receipt["construction"],
            "price_gates": fit_receipt["price_gates"],
            "program_receipts": {arm: fit_receipt["programs"][key] for arm, key in PROGRAM_KEYS.items()},
            "integrity": {"cloned_native": integrity, "poison_canary": canary,
                          "candidate_original_down_calls": original_poison_calls,
                          "candidate_proxy_calls": candidate_calls,
                          "down_weight_sha256_before": model_hash_before,
                          "down_weight_sha256_after": tensor_sha256(original.weight)},
            "coverage": coverage,
            "margins": MARGINS,
            "cell_names": CELL_NAMES,
            "sufficient_statistics": {
                arm: {
                    consumer: {"sums": values["sums"].tolist(), "counts": values["counts"].tolist()}
                    for consumer, values in consumers.items()
                }
                for arm, consumers in ledgers.items()
            },
        }
        raw["inference"] = score_result(raw)
        raw["runtime_s"] = time.time() - started
        write_json_atomic(raw, OUT)
        print(json.dumps({
            "coverage": coverage,
            "absolute_gates": raw["inference"]["absolute_gates"],
            "lexical_gates": raw["inference"]["lexical_gates"],
            "runtime_s": raw["runtime_s"],
        }, indent=2), flush=True)
        print(f"wrote {OUT}", flush=True)
    finally:
        if original is not None and original_forward is not None:
            original.forward = original_forward
            try:
                mlp.Down = original
            except Exception:
                pass
        for hook in hooks:
            hook.remove()
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
