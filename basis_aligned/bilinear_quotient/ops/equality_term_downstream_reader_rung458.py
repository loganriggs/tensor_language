#!/usr/bin/env python3
"""RUNG458 -- DOWNSTREAM-READER-DEFINED EQUALITY-TERM GROUPING."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_term_subset_factorial_stage1 as stage1
import interchange


PREREG = POLY / "EQUALITY_TERM_DOWNSTREAM_READER_RUNG458_PREREGISTRATION.md"
ROWS = ROOT / ".rowcache_induction_equality_tensor_final_ood_v2/final_natural.pt"
ROW_RECEIPT = ROOT / "induction_equality_tensor_final_ood_v2_rows_receipt.json"
OUT = ROOT / "equality_term_downstream_reader_rung458_results.json"
BUNDLE = ROOT / "equality_term_downstream_reader_rung458_sufficient_statistics.pt"
TERMS = stage1.TERMS
TERM_NAMES = tuple(item[0] for item in TERMS)
SINGLETON_MASKS = (1, 2, 4, 8)
SCREEN_MASKS = (*SINGLETON_MASKS, 3, 12)
COMMON_COMPONENTS = tuple(
    component for site in range(9, 18) for component in (f"a{site}", f"m{site}")
)
RESPONSE_CELLS = (
    "all_positive", "matched_negative", "off_target", "near_positive", "far_positive",
    "one_predecessor_positive", "multiple_predecessor_positive",
)
CE_CELLS = ("all_positive", "matched_negative", "off_target", "all")
CROSS_DEPTH_PAIRS = ((0, 2), (0, 3), (1, 2), (1, 3))
FIT_SLICE = slice(0, 96)
VALIDATION_SLICE = slice(96, 192)
DOCUMENTS_PER_HALF = 96
BATCH = 4
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = "equality-term-downstream-reader-rung458:bootstrap:0"
INTERCHANGE_SEED = 458
HASHES = {
    PREREG: "f5ef4be7da9a01c0e72a8ca6fd22949fe34b0dbb02202d8ec91ff7d392f8ef7f",
    ROW_RECEIPT: "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    ROWS: "5f2813eacc3ec66162c2ce695b978264137c66126fdc25e3d49b4efd44a9d759",
    ROOT / "ops/equality_term_subset_factorial_stage1.py":
        "3caa753cd856ec87899936fe71137ce28e893f86433558f40a815afff61824af",
    ROOT / "ops/interchange.py":
        "df4a8585dd6a557a71be991f12d0547023ae771bfccc591008cc0ab08f08fd29",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    payload, masks, metadata = stage1.validate_inputs()
    if payload.get("role") != "final_natural" or "ood_code" in str(ROWS):
        raise RuntimeError("rung458 loaded a forbidden row role")
    return payload, masks, metadata


def component_parts(key: str) -> tuple[str, int]:
    if len(key) < 2 or key[0] not in {"a", "m"} or not key[1:].isdigit():
        raise ValueError(f"malformed component key: {key}")
    kind, site = key[0], int(key[1:])
    if not 9 <= site < 18:
        raise ValueError("reader component is outside the common suffix")
    return kind, site


def term_name(index: int) -> str:
    return TERM_NAMES[index]


def _record_audit(
    totals: dict[str, dict[str, int]], label: str, audit: Mapping[str, int],
    *, analytical: bool, captures: int, patched: bool,
) -> None:
    expected = {
        "native_attention": 15 if analytical else 18,
        "replayed_attention": 3 if analytical else 0,
        "native_mlp": 18,
        "patch_overrides": 1 if patched else 0,
        "captures": captures,
    }
    if dict(audit) != expected:
        raise RuntimeError(f"forward audit changed for {label}: {dict(audit)} != {expected}")
    row = totals.setdefault(label, {"forwards": 0, **{key: 0 for key in expected}})
    row["forwards"] += 1
    for key, value in audit.items():
        row[key] += value


@torch.no_grad()
def run_forward(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    removal_mask: int | None,
    capture_keys: Sequence[str] = (),
    patch_key: str | None = None,
    patch_write: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int]]:
    capture_set = set(capture_keys)
    if len(capture_set) != len(capture_keys) or not capture_set <= set(COMMON_COMPONENTS):
        raise ValueError("capture component identity changed")
    if (patch_key is None) != (patch_write is None):
        raise ValueError("patch key and write must be supplied together")
    if patch_key is not None and patch_key not in COMMON_COMPONENTS:
        raise ValueError("patch component is outside the common suffix")
    captures: dict[str, torch.Tensor] = {}
    audit = {
        "native_attention": 0, "replayed_attention": 0,
        "native_mlp": 0, "patch_overrides": 0, "captures": 0,
    }

    def attention(event: facade.AttentionEvent):
        if removal_mask is not None and event.site in stage1.SITE_HEADS:
            write = stage1.replay_site_arm(
                event.state, event.first_value, event.block.attn, event.site,
                "remove", removal_mask, event.tokens,
            )
            next_value = event.first_value
            audit["replayed_attention"] += 1
        else:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
        key = f"a{event.site}"
        if key in capture_set:
            captures[key] = write.detach().clone()
            audit["captures"] += 1
        if key == patch_key:
            assert patch_write is not None
            if patch_write.shape != write.shape or patch_write.dtype != write.dtype \
                    or patch_write.device != write.device:
                raise RuntimeError("attention patch write has wrong identity")
            write = patch_write
            audit["patch_overrides"] += 1
        return write, next_value

    def mlp(event: facade.EarlyMLPEvent):
        write = event.block.mlp(event.state)
        audit["native_mlp"] += 1
        key = f"m{event.site}"
        if key in capture_set:
            captures[key] = write.detach().clone()
            audit["captures"] += 1
        if key == patch_key:
            assert patch_write is not None
            if patch_write.shape != write.shape or patch_write.dtype != write.dtype \
                    or patch_write.device != write.device:
                raise RuntimeError("MLP patch write has wrong identity")
            write = patch_write
            audit["patch_overrides"] += 1
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=True,
    )
    if set(captures) != capture_set:
        raise RuntimeError("component capture set changed during forward")
    return logits, captures, audit


def empty_response_stats(component_count: int = len(COMMON_COMPONENTS)):
    return {
        "gram": torch.zeros(len(RESPONSE_CELLS), component_count, 4, 4, dtype=torch.float64),
        "native_square": torch.zeros(len(RESPONSE_CELLS), component_count, dtype=torch.float64),
        "counts": torch.zeros(len(RESPONSE_CELLS), dtype=torch.float64),
        "early_block_residual_square": torch.zeros(
            len(RESPONSE_CELLS), component_count, dtype=torch.float64,
        ),
        "layer8_block_residual_square": torch.zeros(
            len(RESPONSE_CELLS), component_count, dtype=torch.float64,
        ),
    }


@torch.no_grad()
def accumulate_responses(
    stats: dict[str, torch.Tensor],
    native: Mapping[str, torch.Tensor],
    singleton: Sequence[Mapping[str, torch.Tensor]],
    masks: Mapping[str, torch.Tensor],
    document_start: int,
    *,
    components: Sequence[str] = COMMON_COMPONENTS,
    blocks: Sequence[Mapping[str, torch.Tensor]] | None = None,
) -> None:
    if len(singleton) != 4 or len(components) != stats["gram"].shape[1]:
        raise ValueError("response accumulator arm/component identity changed")
    for cell_index, cell in enumerate(RESPONSE_CELLS):
        batch_mask = masks[cell][document_start:document_start + BATCH]
        count = int(batch_mask.sum())
        stats["counts"][cell_index] += count
        if count == 0:
            continue
        for component_index, component in enumerate(components):
            native_values = native[component][batch_mask].float()
            deltas = torch.stack([
                (arm[component] - native[component])[batch_mask].float()
                for arm in singleton
            ])
            stats["gram"][cell_index, component_index] += torch.einsum(
                "tnd,snd->ts", deltas, deltas,
            ).double().cpu()
            stats["native_square"][cell_index, component_index] += (
                native_values.square().sum().double().cpu()
            )
            if blocks is not None:
                early_delta = (blocks[0][component] - native[component])[batch_mask].float()
                layer_delta = (blocks[1][component] - native[component])[batch_mask].float()
                early_residual = early_delta - deltas[0] - deltas[1]
                layer_residual = layer_delta - deltas[2] - deltas[3]
                stats["early_block_residual_square"][cell_index, component_index] += (
                    early_residual.square().sum().double().cpu()
                )
                stats["layer8_block_residual_square"][cell_index, component_index] += (
                    layer_residual.square().sum().double().cpu()
                )


def response_reports(stats: Mapping[str, torch.Tensor], components=COMMON_COMPONENTS):
    output = {}
    for cell_index, cell in enumerate(RESPONSE_CELLS):
        output[cell] = {}
        for component_index, component in enumerate(components):
            gram = stats["gram"][cell_index, component_index]
            diagonal = gram.diag().clamp_min(0)
            denominator = torch.sqrt(diagonal[:, None] * diagonal[None, :]).clamp_min(1e-30)
            cosine = gram / denominator
            native_square = float(stats["native_square"][cell_index, component_index])
            rms_relative = torch.sqrt(diagonal / max(native_square, 1e-30))
            output[cell][component] = {
                "cosine": cosine.tolist(),
                "response_rms_over_native_write_rms": rms_relative.tolist(),
                "response_square": diagonal.tolist(),
                "native_write_square": native_square,
                "tokens": int(stats["counts"][cell_index]),
            }
    return output


def select_candidate(reports: Mapping[str, object]):
    candidates = []
    for component in COMMON_COMPONENTS:
        positive = reports["all_positive"][component]
        negative = reports["matched_negative"][component]
        off = reports["off_target"][component]
        for early, late in CROSS_DEPTH_PAIRS:
            positive_cosine = positive["cosine"][early][late]
            task_margin = positive_cosine - max(
                negative["cosine"][early][late], off["cosine"][early][late],
            )
            live = min(
                positive["response_rms_over_native_write_rms"][early],
                positive["response_rms_over_native_write_rms"][late],
            )
            row = {
                "component": component,
                "pair_indices": [early, late],
                "pair_terms": [term_name(early), term_name(late)],
                "positive_cosine": positive_cosine,
                "matched_negative_cosine": negative["cosine"][early][late],
                "off_target_cosine": off["cosine"][early][late],
                "task_margin": task_margin,
                "minimum_relative_response_rms": live,
                "qualified": bool(live >= 1e-4 and positive_cosine >= .70 and task_margin >= .15),
            }
            candidates.append(row)
    qualified = [row for row in candidates if row["qualified"]]
    selected = sorted(
        qualified,
        key=lambda row: (
            -row["task_margin"], -row["positive_cosine"],
            row["component"], tuple(row["pair_terms"]),
        ),
    )[0] if qualified else None
    return candidates, selected


def _ce_sums(
    logits: torch.Tensor,
    rows: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    global_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = rows[:, 1:].to(logits.device)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).view(len(rows), -1)
    sums = torch.zeros(len(rows), len(CE_CELLS), dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for local in range(len(rows)):
        for cell_index, cell in enumerate(CE_CELLS):
            selected = masks[cell][global_start + local]
            sums[local, cell_index] = nll[local, selected].double().sum().cpu()
            counts[local, cell_index] = int(selected.sum())
    return sums, counts


def patch_bootstrap(losses: Mapping[str, torch.Tensor], counts: torch.Tensor,
                    pair: Sequence[int]):
    cell = CE_CELLS.index("all_positive")
    off = CE_CELLS.index("off_target")
    native = losses["native"][:, cell]
    denominator_counts = counts[:, cell]
    point_rows = []
    for term in pair:
        removal = losses[f"remove:{term_name(term)}"][:, cell]
        patched = losses[f"patch:{term_name(term)}"][:, cell]
        stake = (removal.sum() - native.sum()) / denominator_counts.sum()
        repair = (removal.sum() - patched.sum()) / denominator_counts.sum()
        off_change = (
            losses[f"patch:{term_name(term)}"][:, off].sum()
            - losses[f"remove:{term_name(term)}"][:, off].sum()
        ) / counts[:, off].sum()
        point_rows.append({
            "term": term_name(term), "removal_stake_nat": float(stake),
            "repair_effect_nat": float(repair),
            "patch_recovery": float(repair / stake) if float(stake) > 0 else None,
            "off_target_change_from_removal_nat": float(off_change),
        })
    generator = torch.Generator().manual_seed(
        int.from_bytes(hashlib.sha256(BOOTSTRAP_SEED.encode()).digest()[:8], "little")
    )
    replicates = []
    every_stake_positive = True
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        n = min(500, BOOTSTRAP_DRAWS - start)
        draws = torch.randint(DOCUMENTS_PER_HALF, (n, DOCUMENTS_PER_HALF), generator=generator)
        weights = torch.zeros(n, DOCUMENTS_PER_HALF, dtype=torch.float64)
        weights.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
        denominator = weights @ denominator_counts
        chunk = []
        for term in pair:
            removal = losses[f"remove:{term_name(term)}"][:, cell]
            patched = losses[f"patch:{term_name(term)}"][:, cell]
            stake = (weights @ (removal - native)) / denominator
            effect = (weights @ (removal - patched)) / denominator
            if bool((stake <= 0).any()):
                every_stake_positive = False
            chunk.append(torch.where(stake > 0, effect / stake, torch.zeros_like(stake)))
        replicates.append(torch.stack(chunk, dim=1))
    draws = torch.cat(replicates)
    point = torch.tensor([
        row["patch_recovery"] if row["patch_recovery"] is not None else 0.0
        for row in point_rows
    ], dtype=torch.float64)
    if every_stake_positive and all(row["patch_recovery"] is not None for row in point_rows):
        maximum_shortfall = (point.unsqueeze(0) - draws).max(1).values.sort().values
        critical = float(maximum_shortfall[math.ceil(.95 * BOOTSTRAP_DRAWS) - 1])
        lower = (point - critical).tolist()
    else:
        critical, lower = None, [None, None]
    for row, low in zip(point_rows, lower, strict=True):
        row["simultaneous_lower"] = low
    return {
        "terms": point_rows,
        "every_bootstrap_removal_stake_positive": every_stake_positive,
        "simultaneous_lower_critical": critical,
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
    }


def interchange_report(losses: Mapping[str, torch.Tensor], counts: torch.Tensor,
                       pair: Sequence[int], swaps: Sequence[tuple[str, int, int]]):
    cell = CE_CELLS.index("all_positive")
    supported = counts[:, cell] > 0
    within, between, details = [], [], []
    stake_values = []
    for target in pair:
        delta = (
            losses[f"remove:{term_name(target)}"][:, cell] - losses["native"][:, cell]
        )[supported] / counts[:, cell][supported]
        stake_values.extend(delta.abs().tolist())
    for group, target, source in swaps:
        label = f"{group}:{term_name(target)}<-{term_name(source)}"
        delta = (
            losses[label][:, cell] - losses[f"remove:{term_name(target)}"][:, cell]
        )[supported] / counts[:, cell][supported]
        values = delta.abs().tolist()
        (within if group == "within" else between).extend(values)
        details.append({
            "label": label, "target": term_name(target), "source": term_name(source),
            "mean_absolute_all_positive_change_nat": float(delta.abs().mean()),
        })
    statistic = interchange.commutation(
        within, between, seed=INTERCHANGE_SEED, permutations=10_000,
    )
    stake_mean = sum(stake_values) / len(stake_values)
    statistic["mean_absolute_per_document_removal_stake_nat"] = stake_mean
    statistic["within_mean_over_removal_stake"] = statistic["within_mean"] / max(stake_mean, 1e-12)
    statistic["seed"] = INTERCHANGE_SEED
    statistic["permutations"] = 10_000
    statistic["swaps"] = details
    for cell_name in ("matched_negative", "off_target"):
        cell_index = CE_CELLS.index(cell_name)
        report = {}
        for group, target, source in swaps:
            label = f"{group}:{term_name(target)}<-{term_name(source)}"
            count = counts[:, cell_index].sum()
            report[label] = float(
                (losses[label][:, cell_index] - losses[f"remove:{term_name(target)}"][:, cell_index]).sum()
                / count
            )
        statistic[f"{cell_name}_swap_changes_nat"] = report
    return statistic


def validation_response_gate(fit_reports, validation_reports, selected):
    component = selected["component"]
    early, late = selected["pair_indices"]

    def values(reports):
        positive = reports["all_positive"][component]
        negative = reports["matched_negative"][component]
        off = reports["off_target"][component]
        cosine = positive["cosine"][early][late]
        margin = cosine - max(
            negative["cosine"][early][late], off["cosine"][early][late],
        )
        live = min(
            positive["response_rms_over_native_write_rms"][early],
            positive["response_rms_over_native_write_rms"][late],
        )
        context = {
            "near_minus_far": reports["near_positive"][component]["cosine"][early][late]
                - reports["far_positive"][component]["cosine"][early][late],
            "one_minus_multiple": reports["one_predecessor_positive"][component]["cosine"][early][late]
                - reports["multiple_predecessor_positive"][component]["cosine"][early][late],
        }
        return {"positive_cosine": cosine, "task_margin": margin,
                "minimum_relative_response_rms": live, "context_differences": context}

    fit, validation = values(fit_reports), values(validation_reports)
    context_signs = {}
    for key, fit_value in fit["context_differences"].items():
        validation_value = validation["context_differences"][key]
        required = abs(fit_value) >= .05
        held = not required or (
            validation_value != 0 and math.copysign(1, validation_value) == math.copysign(1, fit_value)
        )
        context_signs[key] = {"fit": fit_value, "validation": validation_value,
                              "required": required, "held": held}
    passed = (
        validation["positive_cosine"] >= .60 and validation["task_margin"] >= .10
        and validation["minimum_relative_response_rms"] >= 1e-4
        and all(row["held"] for row in context_signs.values())
    )
    return {"fit": fit, "validation": validation, "context_signs": context_signs, "passed": passed}


@torch.no_grad()
def collect_fit(model, payload, masks, audit_totals):
    rows = payload["rows"]
    stats = empty_response_stats()
    replay_max, replay_relative = 0.0, 0.0
    device = next(model.parameters()).device
    for start in range(FIT_SLICE.start, FIT_SLICE.stop, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native_logits, native_captures, audit = run_forward(
            model, tokens, removal_mask=None, capture_keys=COMMON_COMPONENTS,
        )
        _record_audit(audit_totals, "fit:native", audit, analytical=False,
                      captures=len(COMMON_COMPONENTS), patched=False)
        replay_logits, _, audit = run_forward(model, tokens, removal_mask=0)
        _record_audit(audit_totals, "fit:replay", audit, analytical=True,
                      captures=0, patched=False)
        difference = replay_logits - native_logits
        replay_max = max(replay_max, float(difference.abs().max()))
        replay_relative = max(
            replay_relative,
            float(difference.square().sum()) / max(float(native_logits.square().sum()), 1e-30),
        )
        del native_logits, replay_logits, difference
        singleton_captures = []
        for term_index, mask in enumerate(SINGLETON_MASKS):
            logits, captures, audit = run_forward(
                model, tokens, removal_mask=mask, capture_keys=COMMON_COMPONENTS,
            )
            _record_audit(audit_totals, f"fit:remove:{term_name(term_index)}", audit,
                          analytical=True, captures=len(COMMON_COMPONENTS), patched=False)
            singleton_captures.append(captures)
            del logits
        block_captures = []
        for name, mask in (("early", 3), ("layer8", 12)):
            logits, captures, audit = run_forward(
                model, tokens, removal_mask=mask, capture_keys=COMMON_COMPONENTS,
            )
            _record_audit(audit_totals, f"fit:remove:{name}_block", audit,
                          analytical=True, captures=len(COMMON_COMPONENTS), patched=False)
            block_captures.append(captures)
            del logits
        accumulate_responses(
            stats, native_captures, singleton_captures, masks, start,
            blocks=block_captures,
        )
        del native_captures, singleton_captures, block_captures
    return stats, {"max_abs": replay_max, "relative_squared": replay_relative}


@torch.no_grad()
def collect_validation(model, payload, masks, selected, audit_totals):
    rows = payload["rows"]
    component = selected["component"]
    pair = tuple(selected["pair_indices"])
    remaining_early = next(index for index in (0, 1) if index not in pair)
    remaining_late = next(index for index in (2, 3) if index not in pair)
    early = next(index for index in pair if index in (0, 1))
    late = next(index for index in pair if index in (2, 3))
    swaps = (
        ("within", early, late), ("within", late, early),
        ("between", early, remaining_late), ("between", late, remaining_early),
    )
    validation_stats = empty_response_stats(component_count=1)
    losses = {"native": torch.zeros(DOCUMENTS_PER_HALF, len(CE_CELLS), dtype=torch.float64)}
    for index in range(4):
        losses[f"remove:{term_name(index)}"] = torch.zeros_like(losses["native"])
    for index in pair:
        losses[f"patch:{term_name(index)}"] = torch.zeros_like(losses["native"])
    for group, target, source in swaps:
        losses[f"{group}:{term_name(target)}<-{term_name(source)}"] = torch.zeros_like(losses["native"])
    counts = torch.zeros_like(losses["native"])
    device = next(model.parameters()).device
    for start in range(VALIDATION_SLICE.start, VALIDATION_SLICE.stop, BATCH):
        local_start = start - VALIDATION_SLICE.start
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native_logits, native_capture, audit = run_forward(
            model, tokens, removal_mask=None, capture_keys=(component,),
        )
        _record_audit(audit_totals, "validation:native", audit, analytical=False,
                      captures=1, patched=False)
        sums, batch_counts = _ce_sums(native_logits, batch_rows, masks, start)
        losses["native"][local_start:local_start + BATCH] = sums
        counts[local_start:local_start + BATCH] = batch_counts
        del native_logits
        removal_captures = []
        for term_index, mask in enumerate(SINGLETON_MASKS):
            logits, capture, audit = run_forward(
                model, tokens, removal_mask=mask, capture_keys=(component,),
            )
            _record_audit(audit_totals, f"validation:remove:{term_name(term_index)}", audit,
                          analytical=True, captures=1, patched=False)
            sums, observed_counts = _ce_sums(logits, batch_rows, masks, start)
            if not torch.equal(observed_counts, batch_counts):
                raise RuntimeError("validation CE supports changed across arms")
            losses[f"remove:{term_name(term_index)}"][local_start:local_start + BATCH] = sums
            removal_captures.append(capture)
            del logits
        accumulate_responses(
            validation_stats, native_capture, removal_captures, masks, start,
            components=(component,), blocks=None,
        )
        for target in pair:
            label = f"patch:{term_name(target)}"
            logits, _, audit = run_forward(
                model, tokens, removal_mask=SINGLETON_MASKS[target],
                patch_key=component, patch_write=native_capture[component],
            )
            _record_audit(audit_totals, f"validation:{label}", audit, analytical=True,
                          captures=0, patched=True)
            sums, _ = _ce_sums(logits, batch_rows, masks, start)
            losses[label][local_start:local_start + BATCH] = sums
            del logits
        for group, target, source in swaps:
            label = f"{group}:{term_name(target)}<-{term_name(source)}"
            logits, _, audit = run_forward(
                model, tokens, removal_mask=SINGLETON_MASKS[target],
                patch_key=component, patch_write=removal_captures[source][component],
            )
            _record_audit(audit_totals, f"validation:{label}", audit, analytical=True,
                          captures=0, patched=True)
            sums, _ = _ce_sums(logits, batch_rows, masks, start)
            losses[label][local_start:local_start + BATCH] = sums
            del logits
        del native_capture, removal_captures
    return validation_stats, losses, counts, swaps


def main() -> None:
    started = time.time()
    payload, masks, input_metadata = validate_inputs()
    if len(COMMON_COMPONENTS) != 18 or len(CROSS_DEPTH_PAIRS) * len(COMMON_COMPONENTS) != 72:
        raise RuntimeError("reader search family changed")
    if os.environ.get("BQLIB_DRYRUN") == "1":
        planted = {cell: {} for cell in RESPONSE_CELLS}
        for cell in RESPONSE_CELLS:
            for component in COMMON_COMPONENTS:
                cosine = torch.eye(4).tolist()
                cosine[0][2] = cosine[2][0] = .2
                if component == "m12" and cell == "all_positive":
                    cosine[0][2] = cosine[2][0] = .9
                planted[cell][component] = {
                    "cosine": cosine,
                    "response_rms_over_native_write_rms": [.01] * 4,
                }
        candidates, selected = select_candidate(planted)
        if len(candidates) != 72 or selected is None or selected["component"] != "m12" \
                or selected["pair_indices"] != [0, 2]:
            raise RuntimeError("deterministic reader selection failed synthetic check")
        print(json.dumps({
            "status": "dry_run_passed", "rung": 458, "model_loaded": False,
            "code_ood_loaded": False, "sealed_opened": False,
            "fit_documents": DOCUMENTS_PER_HALF, "validation_documents": DOCUMENTS_PER_HALF,
            "screen_arms_plus_replay": 8, "candidate_pair_readers": 72,
            "components": list(COMMON_COMPONENTS), "input_metadata": input_metadata,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung458 output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    audit_totals: dict[str, dict[str, int]] = {}
    fit_stats, replay = collect_fit(model, payload, masks, audit_totals)
    fit_reports = response_reports(fit_stats)
    candidates, selected = select_candidate(fit_reports)
    pred_a = bool(replay["relative_squared"] <= 1e-12 and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
    pred_b = selected is not None
    validation_stats = losses = counts = swaps = None
    response_validation = patch = interchange_result = None
    pred_c = pred_d = pred_e = False
    if selected is not None:
        validation_stats, losses, counts, swaps = collect_validation(
            model, payload, masks, selected, audit_totals,
        )
        validation_reports = response_reports(validation_stats, components=(selected["component"],))
        response_validation = validation_response_gate(fit_reports, validation_reports, selected)
        pred_c = response_validation["passed"]
        patch = patch_bootstrap(losses, counts, selected["pair_indices"])
        pred_d = bool(
            patch["every_bootstrap_removal_stake_positive"]
            and all(row["patch_recovery"] is not None and row["patch_recovery"] >= .15
                    and row["simultaneous_lower"] is not None and row["simultaneous_lower"] > 0
                    and abs(row["off_target_change_from_removal_nat"]) <= .01
                    for row in patch["terms"])
        )
        interchange_result = interchange_report(losses, counts, selected["pair_indices"], swaps)
        pred_e = bool(
            interchange_result["separation"] >= 2.0
            and interchange_result["p_value"] <= .05
            and interchange_result["within_mean_over_removal_stake"] <= .25
        )
    strong_null = bool(
        not pred_a or selected is None
        or (response_validation is not None and response_validation["validation"]["positive_cosine"] < .30)
        or (patch is not None and all(
            row["patch_recovery"] is None or row["patch_recovery"] <= .05 for row in patch["terms"]
        ))
        or (interchange_result is not None and interchange_result["separation"] <= 1.2)
    )
    bundle = {
        "schema": "equality_term_downstream_reader_rung458_sufficient_statistics_v1",
        "fit_response_stats": fit_stats,
        "validation_response_stats": validation_stats,
        "validation_loss_sums": losses,
        "validation_counts": counts,
        "raw_rows_tokens_logits_or_hidden_states_included": False,
        "code_ood_loaded": False,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 458,
        "claim_level": "natural_text_reader_screen_and_heldout_patch_interchange_not_ood_or_adoption",
        "input_identity": input_metadata,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "code_ood_loaded": False, "sealed_attention0_confirmation_opened": False,
        "terms": TERMS, "common_suffix_components": COMMON_COMPONENTS,
        "fit_screen": {
            "candidate_count": len(candidates), "qualified_count": sum(row["qualified"] for row in candidates),
            "selected": selected, "candidates": candidates, "response_reports": fit_reports,
        },
        "native_replay": replay,
        "response_validation": response_validation,
        "patch": patch,
        "interchange": interchange_result,
        "audit_totals": audit_totals,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {
            "outer_forwards": sum(row["forwards"] for row in audit_totals.values()),
            "searched_pair_reader_candidates": 72,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0,
        },
        'pred_a_instrument': pred_a,
        'pred_b_fit_task_conditioned_pair_reader': pred_b,
        'pred_c_heldout_response_transfer': pred_c,
        'pred_d_heldout_causal_patch': pred_d,
        'pred_e_heldout_interchange': pred_e,
        "strong_null": strong_null,
        "runtime_s": time.time() - started,
        "next_step": (
            "freeze_code_ood_group_confirmation"
            if all((pred_a, pred_b, pred_c, pred_d, pred_e)) and not strong_null
            else "split_equality_terms_into_qk_score_and_value_output_pieces"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 458,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "selected": selected,
        "response_validation": response_validation, "patch": patch,
        "interchange": interchange_result,
        "execution_price": result["execution_price"],
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
