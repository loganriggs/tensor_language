#!/usr/bin/env python3
"""Fit-free L13H8 causal ceiling for the newest bracket native baseline."""

# BQGATE: EXPERIMENT pred_a_exact_live_instrument pred_b_semantic_opener_carries_material_native_baseline pred_c_six_pair_recurrence pred_d_fixed_next_action
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import run_bracket_l13h8_source_region_payload_factorial as exact

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/bracket_native_baseline_l13h8_causal_ceiling_newest_v1.json"
ROWS = ROOT / "circuits/prior_art/bracket_native_baseline_fresh_corpus_v1_rows.json"
PROSPECTIVE = ROOT / "circuits/followups/task14_bracket_native_baseline_semantic_linear_prospective_v2_result.json"
FACTOR_RUNNER = ROOT / "ops/run_bracket_l13h8_source_region_payload_factorial.py"
ZERO_RUNNER = ROOT / "ops/run_bracket_l13h8_semantic_open_zero_removal.py"
OUT = ROOT / "circuits/followups/bracket_native_baseline_l13h8_causal_ceiling_newest_v1_result.json"
CANDIDATE_ID = "bracket.pending_opener.native_baseline_l13h8_causal_ceiling_newest_v1"
EXPECTED = {
    ROWS: "ad246a0ab2affd0a351b971c100c27c2ad09597d0d9e7b84b636e1eb4c8fb399",
    PROSPECTIVE: "1d2f99a6c965ed0d6794cb83a6fb0c8953d11e9a599e769b02d4a0f612d89ea4",
    FACTOR_RUNNER: "b4897139e6bc8451909c09c038bf46dae80ccdca5931e62fd8b7d7ed66c7f53e",
    ZERO_RUNNER: "9de2994652a3cc86efad64ebb48cdd9ac7ee605c0c49c82b5aaa8aefe02dc840",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> tuple[dict, list[dict]]:
    prior = json.loads(PRIOR.read_text())
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"immutable authority changed: {path}")
    if prior["candidate_id"] != CANDIDATE_ID or prior["authority"] != {
        path.name: digest for path, digest in EXPECTED.items()
    }:
        raise ValueError("prior/authority mismatch")
    corpus = json.loads(ROWS.read_text())
    if len(corpus["rows"]) != 36 or corpus["endpoint_count"] != 72:
        raise ValueError("unexpected frozen corpus dimensions")
    return prior, corpus["rows"]


def compile_plan() -> dict:
    prior, rows = load()
    return {
        "schema": "bracket_native_baseline_l13h8_causal_ceiling_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": sha(PRIOR),
        "rows": len(rows),
        "endpoints": 2 * len(rows),
        "arms": ["native", "exact_replay", "semantic_open_zero", "complete_head_zero"],
        "bars": prior["bars"],
        "price": prior["price"],
    }


def l2(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right)) / (l2(left) * l2(right))


def vector_metrics(native: list[float], zeroed: list[float]) -> dict:
    damage = [base - residual for base, residual in zip(native, zeroed)]
    return {
        "count": len(native),
        "damage_positive_fraction": sum(value > 0 for value in damage) / len(damage),
        "explained_norm_ratio": l2(damage) / l2(native),
        "residual_norm_ratio": l2(zeroed) / l2(native),
        "damage_native_cosine": cosine(damage, native),
        "median_fraction_of_native": statistics.median(value / base for value, base in zip(damage, native)),
    }


def collect(model, rows: list[dict], torch, F, facade) -> tuple[list[dict], float, float]:
    device = next(model.parameters()).device
    endpoint_rows, sides = [], []
    for row in rows:
        for side in ("base", "donor"):
            endpoint_rows.append(row)
            sides.append(side)
    length = max(len(row[f"{side}_ids"]) for row, side in zip(endpoint_rows, sides))
    tokens = torch.full((len(endpoint_rows), length), 50256, dtype=torch.long, device=device)
    finals, sources = [], []
    for index, (row, side) in enumerate(zip(endpoint_rows, sides)):
        ids = row[f"{side}_ids"]
        other = row["donor_ids" if side == "base" else "base_ids"]
        differences = [position for position, (a, b) in enumerate(zip(ids, other)) if a != b]
        if len(ids) != len(other) or len(differences) != 1:
            raise ValueError(f"row {row['row_id']} is not a single aligned opener substitution")
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        finals.append(len(ids) - 1)
        sources.append(differences[0])
    finals = torch.tensor(finals, dtype=torch.long, device=device)
    sources = torch.tensor(sources, dtype=torch.long, device=device)
    native = exact.native_logits(model, tokens, torch, F)
    replay, factors = exact.factor_forward(model, tokens, finals, {}, torch, F, facade)
    zeros = torch.zeros_like(factors["u"][torch.arange(len(endpoint_rows), device=device), sources])
    semantic_zero = exact.factor_forward(
        model, tokens, finals, {}, torch, F, facade,
        replacement_terms=zeros, source_positions=sources,
    )[0]
    zero_donor = {"u": torch.zeros_like(factors["u"]), "head": torch.zeros_like(factors["head"])}
    head_zero = exact.factor_forward(
        model, tokens, finals, {}, torch, F, facade,
        donor=zero_donor, complete=True,
    )[0]
    replay_error = float((native - replay).abs().max())
    term_norms = (factors["p"][torch.arange(len(endpoint_rows), device=device), sources].unsqueeze(-1) *
                  factors["u"][torch.arange(len(endpoint_rows), device=device), sources]).norm(dim=-1)
    records = []
    for index, (row, side) in enumerate(zip(endpoint_rows, sides)):
        other = "donor" if side == "base" else "base"
        recipient = row[f"{side}_answer_id"]
        donor = row[f"{other}_answer_id"]
        q = int(finals[index])
        values = {
            "native_margin": exact.closer_margin(native[index, q], recipient),
            "replay_margin": exact.closer_margin(replay[index, q], recipient),
            "semantic_open_zero_margin": exact.closer_margin(semantic_zero[index, q], recipient),
            "complete_head_zero_margin": exact.closer_margin(head_zero[index, q], recipient),
        }
        values["semantic_open_damage"] = values["replay_margin"] - values["semantic_open_zero_margin"]
        values["complete_head_damage"] = values["replay_margin"] - values["complete_head_zero_margin"]
        records.append({
            "row_id": row["row_id"], "side": side,
            "ordered_pair": f"{recipient}->{donor}",
            "recipient_closer_id": recipient, "donor_closer_id": donor,
            "semantic_open_position": int(sources[index]),
            "semantic_open_term_norm": float(term_norms[index]),
            **values,
        })
    return records, replay_error, float(term_norms.min())


def score(records: list[dict], replay_error: float, minimum_term_norm: float, bars: dict) -> dict:
    native = [row["replay_margin"] for row in records]
    semantic_zero = [row["semantic_open_zero_margin"] for row in records]
    head_zero = [row["complete_head_zero_margin"] for row in records]
    semantic = vector_metrics(native, semantic_zero)
    complete = vector_metrics(native, head_zero)
    semantic_damage = [row["semantic_open_damage"] for row in records]
    complete_damage = [row["complete_head_damage"] for row in records]
    semantic["damage_complete_cosine"] = cosine(semantic_damage, complete_damage)
    semantic["damage_to_complete_norm_ratio"] = l2(semantic_damage) / l2(complete_damage)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["ordered_pair"]].append(row)
    recurrence = {}
    for pair, cells in sorted(grouped.items()):
        fractions = [row["semantic_open_damage"] / row["replay_margin"] for row in cells]
        recurrence[pair] = {
            "count": len(cells),
            "positive_damage_fraction": sum(row["semantic_open_damage"] > 0 for row in cells) / len(cells),
            "median_fraction_of_native": statistics.median(fractions),
        }
    instrument = {
        "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
        "native_capability": sum(value > 0 for value in native) / len(native) >= bars["native_positive_fraction_min"],
        "live_semantic_term": minimum_term_norm >= bars["semantic_term_minimum_norm_min"],
        "complete_head_positive_ceiling": complete["damage_positive_fraction"] >= bars["complete_head_damage_positive_fraction_min"],
        "complete_head_material_ceiling": complete["explained_norm_ratio"] >= bars["complete_head_explained_norm_ratio_min"],
    }
    material = (
        semantic["damage_positive_fraction"] >= bars["semantic_damage_positive_fraction_min"] and
        semantic["explained_norm_ratio"] >= bars["semantic_explained_norm_ratio_min"] and
        semantic["damage_native_cosine"] >= bars["semantic_damage_native_cosine_min"] and
        semantic["residual_norm_ratio"] <= bars["semantic_residual_norm_ratio_max"] and
        semantic["damage_complete_cosine"] >= bars["semantic_to_complete_damage_cosine_min"] and
        semantic["damage_to_complete_norm_ratio"] >= bars["semantic_to_complete_damage_norm_ratio_min"]
    )
    recurring = all(
        value["positive_damage_fraction"] >= bars["ordered_pair_semantic_positive_fraction_min"] and
        value["median_fraction_of_native"] >= bars["ordered_pair_semantic_median_fraction_of_native_min"]
        for value in recurrence.values()
    ) and len(recurrence) == 6
    predictions = {
        "pred_a_exact_live_instrument": all(instrument.values()),
        "pred_b_semantic_opener_carries_material_native_baseline": material,
        "pred_c_six_pair_recurrence": recurring,
        "pred_d_fixed_next_action": True,
    }
    terminal = "screen" if all(instrument.values()) and material and recurring else "null" if all(instrument.values()) else "invalid"
    return {
        "instrument_checks": instrument,
        "native_replay_max_absolute_logit_error": replay_error,
        "minimum_semantic_open_term_norm": minimum_term_norm,
        "native_positive_fraction": sum(value > 0 for value in native) / len(native),
        "complete_head": complete,
        "semantic_open": semantic,
        "ordered_pair_recurrence": recurrence,
        "predictions": predictions,
        "licensed_next_action": "one_fixed_direct_readout_compression" if terminal == "screen" else "retain_native_baseline_and_close_local_l13h8_generator" if terminal == "null" else "repair_instrument_only",
        "terminal": terminal,
    }


def main() -> None:
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise ValueError(f"refusing to overwrite {OUT}")
    prior, rows = load()
    torch, F, facade = exact._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        records, replay_error, minimum_term_norm = collect(model, rows, torch, F, facade)
    result = score(records, replay_error, minimum_term_norm, prior["bars"])
    payload = managed.atomic_create_json(OUT, {
        "schema": "bracket_native_baseline_l13h8_causal_ceiling_result_v1",
        "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": result,
        "evidence": records,
        "terminal": result["terminal"],
    })
    print(json.dumps({"terminal": result["terminal"], "predictions": result["predictions"], "licensed_next_action": result["licensed_next_action"], "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
