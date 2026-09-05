#!/usr/bin/env python3
# BQGATE: exact T3/T7 causal factorial; SELECT only, FINAL_TEST/OOD closed.
"""Managed create-only runner for the exact numbered-list T3/T7 removal factorial."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import circuit_candidate_numbered_list_cached_term_head_source_factorial as candidate
import circuit_fast_screen_managed_runner as managed
import numbered_list_cached_value_weight_removal_rung576 as r576
import numbered_list_factor_localization_rung573 as r573


ROOT = Path(__file__).resolve().parent.parent
RESULT = ROOT / "circuits/fast_screens/numbered_list_cached_term_head_source_factorial_v1_result.json"
CHECKPOINT_SHA256 = candidate.CHECKPOINT_SHA256


class RunError(ValueError):
    pass


def _verify_checkpoint_files() -> None:
    import fastload
    _config, blob, source = fastload._paths()
    if hashlib.sha256((source.SNAP / "config.json").read_bytes()).hexdigest() != candidate.CONFIG_SHA256:
        raise RunError("checkpoint config hash changed")
    with open(blob, "rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != CHECKPOINT_SHA256:
            raise RunError("checkpoint weights hash changed")


def _relative_squared(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).square().sum()) / max(float(b.float().square().sum()), 1e-30)


def _head_terms(tensors: Mapping[str, torch.Tensor], cached: torch.Tensor, finals: torch.Tensor,
                sources: torch.Tensor, weight: torch.Tensor) -> dict[int, torch.Tensor]:
    arange = torch.arange(cached.size(0), device=cached.device)
    output = {}
    for head in (3, 7):
        score = tensors["pattern"][arange, head, finals, sources]
        head_weight = weight[:, head * r573.HEAD_D:(head + 1) * r573.HEAD_D]
        output[head] = score[:, None] * F.linear(cached[arange, sources, head], head_weight.to(cached.dtype))
    return output


@torch.no_grad()
def _factor_forward(model: torch.nn.Module, tokens: torch.Tensor, finals: torch.Tensor,
                    sources: torch.Tensor, zero_heads: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, dict]:
    cached = r576.compiled_cached(model, tokens)
    term_norm = None
    diagnostics = {"joint_term_relative_squared_error": 0.0,
                   "cached_bus_relative_squared_error": 0.0,
                   "head_source_sum_relative_squared_error": 0.0,
                   "value_split_relative_squared_error": 0.0}

    def attention(event):
        nonlocal term_norm
        if event.site != r576.LAYER:
            return event.block.attn(event.state, event.first_value)
        write, tensors, errors = r573.replay_attention(event.state, event.first_value, event.block.attn, finals)
        for key, value in errors.items():
            diagnostics[key] = max(diagnostics.get(key, 0.0), float(value))
        diagnostics["cached_bus_relative_squared_error"] = _relative_squared(cached, tensors["cached"])
        terms = _head_terms(tensors, cached, finals, sources, event.block.attn.c_proj.weight)
        frozen_joint = r576.projected_terms(tensors, cached, finals, sources, event.block.attn.c_proj.weight)
        diagnostics["joint_term_relative_squared_error"] = _relative_squared(terms[3] + terms[7], frozen_joint)
        removed = sum((terms[head] for head in zero_heads), torch.zeros_like(frozen_joint))
        term_norm = removed.float().norm(dim=-1)
        arange = torch.arange(tokens.size(0), device=tokens.device)
        write = write.clone()
        write[arange, finals] -= removed.to(write.dtype)
        return write, event.first_value

    logits = facade.forward_with_dispatch(model, tokens, attention,
        lambda event: event.block.mlp(event.state), require_production=False)
    if term_norm is None:
        raise RunError("layer-8 intervention hook did not fire")
    arange = torch.arange(tokens.size(0), device=tokens.device)
    return logits[arange, finals].float(), term_norm, diagnostics


def _margin(logits: torch.Tensor, answer_id: int, answer: str) -> float:
    pool = r576.candidate_ids(answer).to(logits.device)
    alternatives = pool[pool != answer_id]
    return float(logits[answer_id] - logits[alternatives].max())


def _ce(logits: torch.Tensor, answer_id: int) -> float:
    return float(torch.logsumexp(logits.float(), -1) - logits[answer_id].float())


def _bootstrap_interaction(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        if record["family"] not in candidate.TARGET_FAMILIES:
            continue
        grouped[str(record["group_id"])].append(record)
    values = np.asarray([statistics.fmean(float(item["interaction_margin_damage"]) for item in group)
                         for group in grouped.values()], dtype=np.float64)
    rng = np.random.default_rng(candidate.SEED)
    indices = rng.integers(0, len(values), size=(candidate.BOOTSTRAPS, len(values)))
    draws = values[indices].mean(1)
    return {"mean": float(values.mean()), "ci95_low": float(np.quantile(draws, .025)),
            "ci95_high": float(np.quantile(draws, .975)), "group_count": len(values)}


def score_records(records: Sequence[Mapping[str, object]], replay_rse: float,
                  joint_term_rse: float) -> dict[str, object]:
    by = defaultdict(list)
    for record in records:
        for condition in ("zero_T3", "zero_T7", "zero_T3_T7"):
            by[(condition, record["family"], record["endpoint"])].append(record)
    reports = {}
    selective = {}
    target_necessary = {}
    for condition in ("zero_T3", "zero_T7", "zero_T3_T7"):
        reports[condition] = {}
        target_ok = True
        control_ok = True
        for family in candidate.FAMILIES:
            reports[condition][family] = {}
            for endpoint in candidate.ENDPOINTS:
                cells = by[(condition, family, endpoint)]
                damages = [float(row[f"{condition}_margin_damage"]) for row in cells]
                ces = [float(row[f"{condition}_ce_increase"]) for row in cells]
                item = {"n": len(cells), "mean_margin_damage": statistics.fmean(damages),
                        "positive_fraction": sum(x > 0 for x in damages) / len(damages),
                        "mean_ce_increase": statistics.fmean(ces),
                        "median_absolute_margin_fraction": statistics.median(abs(x) for x in damages) /
                            candidate.FIT_SCALES["margin_damage"],
                        "median_logit_rms_fraction": statistics.median(float(row[f"{condition}_logit_rms"])
                            for row in cells) / candidate.FIT_SCALES["logit_rms"],
                        "median_term_norm_fraction": statistics.median(float(row[f"{condition}_term_norm"])
                            for row in cells) / candidate.FIT_SCALES["term_norm"],
                        "answer_preserved_fraction": statistics.fmean(bool(row[f"{condition}_answer_preserved"])
                            for row in cells)}
                if family in candidate.TARGET_FAMILIES:
                    lows = [r576.lower(damages, candidate.SEED), r576.lower(ces, candidate.SEED + 1)]
                    item["passed"] = item["positive_fraction"] >= candidate.MIN_POSITIVE_FRACTION and min(lows) > 0
                    item["bootstrap_lower_margin_damage"], item["bootstrap_lower_ce_increase"] = lows
                    target_ok &= bool(item["passed"])
                else:
                    item["passed"] = (item["median_term_norm_fraction"] >= candidate.MIN_TERM_SCALE_FRACTION
                        and item["answer_preserved_fraction"] >= candidate.MIN_ANSWER_PRESERVED
                        and item["mean_ce_increase"] <= candidate.MAX_CONTROL_CE
                        and item["median_absolute_margin_fraction"] <= candidate.MAX_CONTROL_SCALE_FRACTION
                        and item["median_logit_rms_fraction"] <= candidate.MAX_CONTROL_SCALE_FRACTION)
                    control_ok &= bool(item["passed"])
                reports[condition][family][endpoint] = item
        target_necessary[condition] = target_ok
        selective[condition] = target_ok and control_ok
    interaction = _bootstrap_interaction(records)
    instrument = replay_rse <= candidate.MAX_NATIVE_REPLAY_RSE and joint_term_rse <= candidate.MAX_JOINT_TERM_RSE
    singleton_selective = selective["zero_T3"] or selective["zero_T7"]
    cooperative = (not singleton_selective and target_necessary["zero_T3_T7"]
                   and interaction["ci95_low"] > 0)
    redundant = interaction["ci95_high"] < 0
    return {"instrument_passed": instrument, "target_necessary": target_necessary,
            "selectively_necessary": selective, "interaction": interaction,
            "predictions": {"pred_a_instrument_exact": instrument,
                            "pred_b_individual_source_selective": singleton_selective,
                            "pred_c_cooperative_service": cooperative,
                            "pred_d_redundant_service": redundant}, "reports": reports}


@torch.no_grad()
def run_science(device: str = "cuda") -> dict[str, object]:
    rows = candidate.build_rows()
    plan = candidate.compile_plan(rows)
    _verify_checkpoint_files()
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32, verify_weights_sha256=True)
    if checkpoint.weights_sha256 != CHECKPOINT_SHA256:
        raise RunError("checkpoint hash changed")
    outputs, norms = {}, {}
    replay_num = replay_den = 0.0
    joint_term_rse = 0.0
    forwards = evaluations = 0
    for chunk in candidate._chunks(rows):
        tokens = torch.tensor([row["ids"] for row in chunk], dtype=torch.long, device=device)
        finals = torch.tensor([row["query_position"] for row in chunk], dtype=torch.long, device=device)
        sources = torch.tensor([row["source_position"] for row in chunk], dtype=torch.long, device=device)
        direct = r573.native_logits(model, tokens)[torch.arange(len(chunk), device=device), finals].float()
        replay, replay_norm, replay_diag = _factor_forward(model, tokens, finals, sources, ())
        forwards += 2; evaluations += 2 * len(chunk)
        replay_num += float((direct - replay).square().sum()); replay_den += float(direct.square().sum())
        joint_term_rse = max(joint_term_rse, float(replay_diag["joint_term_relative_squared_error"]))
        outputs["native_direct", tuple(row["row_id"] for row in chunk)] = direct.cpu()
        for condition, heads in candidate.HEADS_BY_CONDITION.items():
            logits, term_norm, diag = _factor_forward(model, tokens, finals, sources, heads)
            forwards += 1; evaluations += len(chunk)
            joint_term_rse = max(joint_term_rse, float(diag["joint_term_relative_squared_error"]))
            outputs[condition, tuple(row["row_id"] for row in chunk)] = logits.cpu()
            norms[condition, tuple(row["row_id"] for row in chunk)] = term_norm.cpu()
    records = []
    for chunk in candidate._chunks(rows):
        key = tuple(row["row_id"] for row in chunk)
        native = outputs["native_direct", key]
        item_logits = {condition: outputs[condition, key] for condition in candidate.HEADS_BY_CONDITION}
        for i, row in enumerate(chunk):
            answer_id, answer = int(row["answer_id"]), str(row["answer"])
            native_margin = _margin(native[i], answer_id, answer)
            record = {"row_id": row["row_id"], "group_id": row["group_id"], "family": row["family"],
                      "endpoint": row["endpoint"], "native_margin": native_margin}
            for condition, logits in item_logits.items():
                after_margin = _margin(logits[i], answer_id, answer)
                record[f"{condition}_margin_damage"] = native_margin - after_margin
                record[f"{condition}_ce_increase"] = _ce(logits[i], answer_id) - _ce(native[i], answer_id)
                record[f"{condition}_logit_rms"] = float((logits[i] - native[i]).square().mean().sqrt())
                record[f"{condition}_term_norm"] = float(norms[condition, key][i])
                record[f"{condition}_answer_preserved"] = after_margin > 0
            record["interaction_margin_damage"] = (record["zero_T3_T7_margin_damage"]
                - record["zero_T3_margin_damage"] - record["zero_T7_margin_damage"])
            records.append(record)
    replay_rse = replay_num / max(replay_den, 1e-30)
    scored = score_records(records, replay_rse, joint_term_rse)
    if not scored["instrument_passed"]:
        terminal, reason = "invalid", "exactness_gate_failed"
    elif scored["predictions"]["pred_b_individual_source_selective"]:
        terminal, reason = "screen", "individual_cached_head_source_selective"
    else:
        terminal, reason = "null", "no_individual_cached_head_source_selective"
    return {"schema": "numbered_list_cached_term_head_source_factorial_result_v1",
            "task_id": candidate.TASK_ID, "phase": candidate.PHASE,
            "opened_splits": [candidate.PHASE], "closed_splits": ["FINAL_TEST", "OOD"],
            "authority_sha256": candidate.validate_rows(rows), "plan_sha256": plan["compiled_sha256"],
            "checkpoint_weights_sha256": checkpoint.weights_sha256,
            "native_replay_relative_squared_error": replay_rse,
            "joint_term_relative_squared_error": joint_term_rse,
            "terminal": terminal, "reason": reason, **scored, "raw": records,
            "price": {"forward_calls": forwards, "example_evaluations": evaluations,
                      "backward_calls": 0, "model_updates": 0,
                      "raw_numeric_evidence_bytes": plan["price"]["raw_numeric_evidence_bytes"]}}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(candidate.compile_plan(), sort_keys=True)); return
    if RESULT.exists():
        raise RunError("create-only result already exists")
    result = run_science()
    managed.atomic_create_json(RESULT, result)
    print(json.dumps({key: result[key] for key in ("terminal", "reason", "predictions", "price")}, indent=2))


if __name__ == "__main__":
    main()
