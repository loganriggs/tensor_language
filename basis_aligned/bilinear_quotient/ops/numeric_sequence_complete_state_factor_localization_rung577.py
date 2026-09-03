#!/usr/bin/env python3
"""R577: numeric-sequence complete-state and exact attention-factor localization.

No rank is fitted. FIT selects one complete final-query site and, conditionally,
one exact semantic L8H7/H3 factor arm. SELECT can validate but cannot replace
either choice. FINAL_TEST/OOD remain closed.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bilin18_observed_model_facade as facade  # noqa: E402
import numbered_list_factor_localization_rung573 as r573  # noqa: E402


ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
RECEIPT = ROOT / "increment_two_hypothesis_rows_rung567_receipt.json"
CAPABILITY = ROOT / "numeric_two_hypothesis_capability_rung569_570_results.json"
CAPABILITY_AUDIT = ROOT / "numeric_two_hypothesis_capability_rung571_audit.json"
R575 = ROOT / "numeric_factor_removal_positions_rung575.json"
POSITIONS = ROOT / "numeric_sequence_semantic_positions_rung577.json"
PREREG = POLY / "NUMERIC_SEQUENCE_COMPLETE_STATE_FACTOR_LOCALIZATION_RUNG577_PREREGISTRATION.md"
OUT = ROOT / "numeric_sequence_complete_state_factor_localization_rung577_results.json"
HASHES = {
    ROWS: "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
    RECEIPT: "02b2c37cc23434138accd63e920f417cda10f1c86a4c08c174537149ec2b1072",
    CAPABILITY: "7cc56f22def334673e0035fad7c6a7d1fc58ab8edd3a99744bebd9fb4e6af7e7",
    CAPABILITY_AUDIT: "c5453ddaa4aa46806cbfcb9a9b0941fe8ddbb21c61e5e22d00c1d1cea6dd74bb",
    R575: "3663ebc48e5dca1ff336cb0627fc43c6db8d7d6e1666b81d7631ab150168dd4b",
    POSITIONS: "a6a98715617cf91971655c252553f42d45b59937ecfbf46722b518333721de1d",
    PREREG: "a35ac6dbf4ce2ee85e4e047157f0778d33bf066dee9883b94065149ae3252c98",
}
HYPOTHESIS = "numeric_sequence_continuation"
TARGETS = (
    "sequence_digit_state_shift",
    "sequence_word_state_shift",
    "sequence_cross_format_shift",
)
RELATION = "sequence_middle_value_break"
CONTROLS = (
    "sequence_digit_surface_preserved",
    "sequence_word_surface_preserved",
    "sequence_digit_copy_control",
    "sequence_word_copy_control",
    "sequence_step_two_conflict",
)
FAMILIES = TARGETS + (RELATION,) + CONTROLS
DIRECTIONS = ("base_to_donor", "donor_to_base")
SITE_ARMS = (
    "a8_h73_complete",
    "a8_all_heads_complete",
    "post_attention8_state",
    "post_mlp8_state",
    "post_mlp10_state",
    "post_mlp12_state",
    "post_mlp14_state",
)
FACTOR_ARMS = (
    "semantic_final_score",
    "semantic_nonfinal_score",
    "semantic_all_score",
    "semantic_nonfinal_cached_value",
    "semantic_final_own_value",
    "semantic_nonfinal_own_value",
    "semantic_all_own_value",
    "semantic_final_joint",
    "semantic_nonfinal_joint",
    "semantic_all_joint",
)
STATE_LAYERS = (8, 10, 12, 14)
HEADS = (7, 3)
LAYER = 8
BATCH = 24
BOOTSTRAPS = 2000
SEED = 577
MAXIMUM_FORWARDS = 652
ENC = tiktoken.get_encoding("gpt2")
WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen", "twenty",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower(values: list[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[indices].mean(1), .025))


def candidate_ids(answer: str) -> torch.Tensor:
    if not answer.startswith(" "):
        strings = [str(value) for value in range(101)]
    elif answer.strip().isdigit():
        strings = [" " + str(value) for value in range(101)]
    else:
        strings = [" " + word for word in WORDS]
    return torch.tensor(sorted({ids[0] for text in strings if len(ids := ENC.encode(text)) == 1}),
                        dtype=torch.long)


def numeric_margin(logits: torch.Tensor, answer_id: int, answer: str) -> float:
    pool = candidate_ids(answer).to(logits.device)
    alternatives = pool[pool != answer_id]
    return float(logits[answer_id] - logits[alternatives].max())


def answer_is_best(logits: torch.Tensor, answer_id: int, answer: str) -> bool:
    pool = candidate_ids(answer).to(logits.device)
    return int(pool[logits[pool].argmax()]) == int(answer_id)


def load_authority() -> tuple[list[dict], dict[str, dict]]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input changed: {path}")
    document = json.loads(ROWS.read_text())
    positions_document = json.loads(POSITIONS.read_text())
    capability = json.loads(CAPABILITY.read_text())
    audit = json.loads(CAPABILITY_AUDIT.read_text())
    assert document["model_loaded"] is False and document["outcomes_opened"] == []
    assert capability["sequence_all_gates_pass"] is True
    assert capability["evaluated_splits"][HYPOTHESIS] == ["FIT", "SELECT"]
    assert audit["input_hashes_match"] and audit["conditional_split_opening_match"]
    assert audit["terminal_decisions_reproduced"]
    assert positions_document["model_loaded"] is False
    assert positions_document["outcomes_opened"] == []
    assert positions_document["r575_endpoint_mappings_reproduced"] == 480
    rows = [row for row in document["rows"] if row["hypothesis_id"] == HYPOTHESIS
            and row["split"] in {"FIT", "SELECT"}]
    assert len(rows) == 432 and all(row["family_id"] in FAMILIES for row in rows)
    positions = {item["row_id"]: item for item in positions_document["records"]}
    assert len(positions) == len(rows) and set(positions) == {row["row_id"] for row in rows}
    for row in rows:
        mapping = positions[row["row_id"]]
        assert mapping["split"] == row["split"] and mapping["family_id"] == row["family_id"]
        for endpoint in ("base", "donor"):
            item = mapping["endpoints"][endpoint]
            assert item["sequence_length"] == len(row[f"{endpoint}_ids"])
            assert item["query_position"] == len(row[f"{endpoint}_ids"]) - 1
            assert len(item["value_positions"]) == 3
    return rows, positions


def native_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    x = F.rms_norm(model.transformer.wte(tokens), (r573.D,))
    x0, first_value = x, None
    for block in model.transformer.h:
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        write, first_value = block.attn(F.rms_norm(x, (r573.D,)), first_value)
        x = x + write
        x = x + block.mlp(F.rms_norm(x, (r573.D,)))
    return (30 * torch.tanh(model.lm_head(F.rms_norm(x, (r573.D,))) / 30)).float()


def _semantic_factors(tensors: dict, finals: torch.Tensor,
                      positions: torch.Tensor) -> dict[str, torch.Tensor]:
    batch = positions.size(0)
    output = {key: [] for key in ("score", "value", "cached", "own")}
    for index in range(batch):
        output["score"].append(torch.stack([
            torch.stack([tensors["pattern"][index, head, finals[index], position]
                         for position in positions[index]]) for head in HEADS]))
        for key in ("value", "cached", "own"):
            output[key].append(torch.stack([
                torch.stack([tensors[key][index, position, head]
                             for position in positions[index]]) for head in HEADS]))
    return {key: torch.stack(value) for key, value in output.items()}


@torch.no_grad()
def capture_forward(model: torch.nn.Module, tokens: torch.Tensor, finals: torch.Tensor,
                    positions: torch.Tensor) -> tuple[torch.Tensor, dict, dict]:
    x = F.rms_norm(model.transformer.wte(tokens), (r573.D,))
    x0, first_value = x, None
    capture: dict = {"states": {}}
    diagnostics = {"head_source_sum_relative_squared_error": 0.0,
                   "value_split_relative_squared_error": 0.0}
    arange = torch.arange(tokens.size(0), device=tokens.device)
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        state = F.rms_norm(x, (r573.D,))
        if site == LAYER:
            write, tensors, errors = r573.replay_attention(state, first_value, block.attn, finals)
            capture["a8_head_output"] = tensors["head_output"][arange, finals]
            capture["a8_factors"] = _semantic_factors(tensors, finals, positions)
            diagnostics.update({key: max(diagnostics[key], value) for key, value in errors.items()})
        else:
            write, first_value = block.attn(state, first_value)
        x = x + write
        if site == LAYER:
            capture["states"]["post_attention8_state"] = x[arange, finals]
        x = x + block.mlp(F.rms_norm(x, (r573.D,)))
        if site in STATE_LAYERS:
            capture["states"][f"post_mlp{site}_state"] = x[arange, finals]
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (r573.D,))) / 30)).float()
    return logits[arange, finals], capture, diagnostics


def _donor_batch(captures: list[dict], device: torch.device) -> dict:
    return {
        "states": {key: torch.stack([item["states"][key] for item in captures]).to(device)
                   for key in captures[0]["states"]},
        "a8_head_output": torch.stack([item["a8_head_output"] for item in captures]).to(device),
        "a8_factors": {key: torch.stack([item["a8_factors"][key] for item in captures]).to(device)
                       for key in captures[0]["a8_factors"]},
    }


def _factor_ordinals(arm: str) -> tuple[int, ...]:
    if "nonfinal" in arm:
        return (0, 1)
    if "final" in arm:
        return (2,)
    return (0, 1, 2)


def modify_a8_head_output(head_output: torch.Tensor, tensors: dict, finals: torch.Tensor,
                          positions: torch.Tensor, donor: dict, arm: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return modified head outputs and the c_proj-input delta at the final query."""
    arange = torch.arange(head_output.size(0), device=head_output.device)
    native = head_output[arange, finals].clone()
    changed = native.clone()
    if arm == "a8_all_heads_complete":
        changed = donor["a8_head_output"].to(changed)
    elif arm == "a8_h73_complete":
        for head in HEADS:
            changed[:, head] = donor["a8_head_output"][:, head].to(changed)
    else:
        ordinals = _factor_ordinals(arm)
        factors = donor["a8_factors"]
        for index in range(head_output.size(0)):
            for head_slot, head in enumerate(HEADS):
                for ordinal in ordinals:
                    position = int(positions[index, ordinal])
                    base_score = tensors["pattern"][index, head, finals[index], position]
                    base_value = tensors["value"][index, position, head]
                    native_term = base_score * base_value
                    donor_score = factors["score"][index, head_slot, ordinal].to(base_value)
                    if arm.endswith("joint"):
                        replacement = donor_score * factors["value"][index, head_slot, ordinal].to(base_value)
                    elif arm.endswith("score"):
                        replacement = donor_score * base_value
                    elif arm.endswith("cached_value"):
                        donor_cached = factors["cached"][index, head_slot, ordinal].to(base_value)
                        replacement = base_score * (
                            tensors["own"][index, position, head] + donor_cached)
                    elif arm.endswith("own_value"):
                        donor_own = factors["own"][index, head_slot, ordinal].to(base_value)
                        replacement = base_score * (donor_own + tensors["cached"][index, position, head])
                    else:
                        raise ValueError(arm)
                    changed[index, head] += replacement - native_term
    result = head_output.clone()
    result[arange, finals] = changed
    return result, changed - native


@torch.no_grad()
def intervention_forward(model: torch.nn.Module, tokens: torch.Tensor, finals: torch.Tensor,
                         positions: torch.Tensor, donor: dict, arm: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if arm not in SITE_ARMS + FACTOR_ARMS:
        raise ValueError(arm)
    x = F.rms_norm(model.transformer.wte(tokens), (r573.D,))
    x0, first_value = x, None
    diagnostics = {"head_source_sum_relative_squared_error": 0.0,
                   "value_split_relative_squared_error": 0.0}
    intervention_norm = None
    arange = torch.arange(tokens.size(0), device=tokens.device)
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        state = F.rms_norm(x, (r573.D,))
        if site == LAYER:
            write, tensors, errors = r573.replay_attention(state, first_value, block.attn, finals)
            diagnostics.update({key: max(diagnostics[key], value) for key, value in errors.items()})
            if arm.startswith("a8_") or arm.startswith("semantic_"):
                changed, head_delta = modify_a8_head_output(
                    tensors["head_output"], tensors, finals, positions, donor, arm)
                write = r573.linear(changed.reshape(tokens.size(0), tokens.size(1), r573.D),
                                    block.attn.c_proj.weight)
                projected = r573.linear(head_delta.reshape(tokens.size(0), r573.D),
                                        block.attn.c_proj.weight)
                intervention_norm = projected.float().norm(dim=-1)
        else:
            write, first_value = block.attn(state, first_value)
        x = x + write
        if arm == "post_attention8_state" and site == LAYER:
            replacement = donor["states"][arm].to(x)
            intervention_norm = (replacement - x[arange, finals]).float().norm(dim=-1)
            x = x.clone(); x[arange, finals] = replacement
        x = x + block.mlp(F.rms_norm(x, (r573.D,)))
        state_arm = f"post_mlp{site}_state"
        if arm == state_arm:
            replacement = donor["states"][arm].to(x)
            intervention_norm = (replacement - x[arange, finals]).float().norm(dim=-1)
            x = x.clone(); x[arange, finals] = replacement
    if intervention_norm is None:
        raise RuntimeError(f"arm never executed: {arm}")
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (r573.D,))) / 30)).float()
    return logits[arange, finals], intervention_norm, diagnostics


def endpoint_positions(position: dict, endpoint: str) -> tuple[int, list[int]]:
    item = position["endpoints"][endpoint]
    return item["query_position"], [value["token_position"] for value in item["value_positions"]]


def chunk_by_length(items: list, length_fn) -> list[list]:
    ordered = sorted(items, key=lambda item: (length_fn(item), str(item)))
    output, cursor = [], 0
    while cursor < len(ordered):
        length = length_fn(ordered[cursor])
        chunk = []
        while cursor < len(ordered) and length_fn(ordered[cursor]) == length and len(chunk) < BATCH:
            chunk.append(ordered[cursor]); cursor += 1
        output.append(chunk)
    return output


def unique_sequences(rows: list[dict], positions: dict[str, dict], split: str) -> dict[tuple[int, ...], list[int]]:
    output = {}
    for row in rows:
        if row["split"] != split:
            continue
        for endpoint in ("base", "donor"):
            ids = tuple(row[f"{endpoint}_ids"])
            _, semantic = endpoint_positions(positions[row["row_id"]], endpoint)
            if ids in output:
                assert output[ids] == semantic
            output[ids] = semantic
    return output


@torch.no_grad()
def capture_split(model: torch.nn.Module, rows: list[dict], positions: dict[str, dict], split: str):
    sequences = unique_sequences(rows, positions, split)
    cache, calls, replay_error = {}, 0, 0.0
    diagnostics = {"head_source_sum_relative_squared_error": 0.0,
                   "value_split_relative_squared_error": 0.0}
    device = next(model.parameters()).device
    for chunk_index, chunk in enumerate(chunk_by_length(list(sequences), len)):
        tokens = torch.tensor(chunk, dtype=torch.long, device=device)
        finals = torch.tensor([len(ids) - 1 for ids in chunk], dtype=torch.long, device=device)
        semantic = torch.tensor([sequences[ids] for ids in chunk], dtype=torch.long, device=device)
        logits, captures, errors = capture_forward(model, tokens, finals, semantic)
        calls += 1
        if chunk_index == 0:
            direct = native_logits(model, tokens)[torch.arange(len(chunk), device=device), finals].float()
            calls += 1
            replay_error = float((direct - logits).square().sum()) / max(float(direct.square().sum()), 1e-30)
        for key, value in errors.items():
            diagnostics[key] = max(diagnostics[key], value)
        for index, ids in enumerate(chunk):
            cache[ids] = {
                "logits": logits[index].detach().cpu(),
                "states": {key: value[index].detach().cpu() for key, value in captures["states"].items()},
                "a8_head_output": captures["a8_head_output"][index].detach().cpu(),
                "a8_factors": {key: value[index].detach().cpu()
                               for key, value in captures["a8_factors"].items()},
            }
    return cache, calls, {"native_replay_relative_squared_error": replay_error, **diagnostics}


def _target_cell(row: dict, direction: str, before: torch.Tensor,
                 source: torch.Tensor, after: torch.Tensor) -> dict:
    if direction == "base_to_donor":
        positive, negative = row["donor_answer_id"], row["base_answer_id"]
        answer, answer_id = row["donor_answer"], row["donor_answer_id"]
    else:
        positive, negative = row["base_answer_id"], row["donor_answer_id"]
        answer, answer_id = row["base_answer"], row["base_answer_id"]
    natural = float((source[positive] - source[negative]) - (before[positive] - before[negative]))
    effect = float((after[positive] - after[negative]) - (before[positive] - before[negative]))
    return {"natural_effect": natural, "effect": effect,
            "target_answer_best": answer_is_best(after, answer_id, answer)}


def _relation_cell(row: dict, direction: str, before: torch.Tensor,
                   source: torch.Tensor, after: torch.Tensor) -> dict:
    coherent = row["base_answer"]
    answer_id = row["base_answer_id"]
    if direction == "base_to_donor":
        base_margin = numeric_margin(before, answer_id, coherent)
        donor_margin = numeric_margin(source, answer_id, coherent)
        after_margin = numeric_margin(after, answer_id, coherent)
        natural, effect = base_margin - donor_margin, base_margin - after_margin
    else:
        donor_margin = numeric_margin(before, answer_id, coherent)
        base_margin = numeric_margin(source, answer_id, coherent)
        after_margin = numeric_margin(after, answer_id, coherent)
        natural, effect = base_margin - donor_margin, after_margin - donor_margin
    return {"natural_effect": natural, "effect": effect}


def _control_cell(row: dict, direction: str, before: torch.Tensor,
                  after: torch.Tensor) -> dict:
    cell = {"full_vocabulary_logit_rms": float((after - before).square().mean().sqrt())}
    if row["family_id"] == "sequence_step_two_conflict":
        arithmetic = ENC.encode(" " + str(row["semantic_details"]["arithmetic_step_two_answer"]))
        successor = ENC.encode(" " + str(row["semantic_details"]["last_value_successor_answer"]))
        assert len(arithmetic) == len(successor) == 1
        old = float(before[arithmetic[0]] - before[successor[0]])
        new = float(after[arithmetic[0]] - after[successor[0]])
        cell.update({"registered_margin_change": new - old,
                     "preference_sign_preserved": (new >= 0) == (old >= 0),
                     "registered_answer_best": None, "ce_increase": None})
    else:
        endpoint = "base" if direction == "base_to_donor" else "donor"
        answer, answer_id = row[f"{endpoint}_answer"], row[f"{endpoint}_answer_id"]
        old = numeric_margin(before, answer_id, answer)
        new = numeric_margin(after, answer_id, answer)
        cell.update({"registered_margin_change": new - old,
                     "preference_sign_preserved": None,
                     "registered_answer_best": answer_is_best(after, answer_id, answer),
                     "ce_increase": float(F.cross_entropy(after[None], torch.tensor([answer_id],
                         device=after.device)) - F.cross_entropy(before[None], torch.tensor([answer_id],
                         device=before.device)))})
    return cell


@torch.no_grad()
def evaluate_arms(model: torch.nn.Module, rows: list[dict], positions: dict[str, dict], split: str,
                  cache: dict, arms: tuple[str, ...]) -> tuple[dict, int, dict]:
    oriented = [(row, direction) for row in rows if row["split"] == split for direction in DIRECTIONS]
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    diagnostics = {"head_source_sum_relative_squared_error": 0.0,
                   "value_split_relative_squared_error": 0.0}
    device = next(model.parameters()).device
    calls = 0
    for arm in arms:
        for chunk in chunk_by_length(oriented, lambda item: len(
                item[0]["base_ids" if item[1] == "base_to_donor" else "donor_ids"])):
            tokens, finals, semantic, donors = [], [], [], []
            for row, direction in chunk:
                recipient = "base" if direction == "base_to_donor" else "donor"
                source = "donor" if direction == "base_to_donor" else "base"
                tokens.append(row[f"{recipient}_ids"])
                final, values = endpoint_positions(positions[row["row_id"]], recipient)
                finals.append(final); semantic.append(values)
                donors.append(cache[tuple(row[f"{source}_ids"])])
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            final_tensor = torch.tensor(finals, dtype=torch.long, device=device)
            semantic_tensor = torch.tensor(semantic, dtype=torch.long, device=device)
            after, norms, errors = intervention_forward(
                model, token_tensor, final_tensor, semantic_tensor, _donor_batch(donors, device), arm)
            calls += 1
            for key, value in errors.items():
                diagnostics[key] = max(diagnostics[key], value)
            for index, (row, direction) in enumerate(chunk):
                recipient = "base" if direction == "base_to_donor" else "donor"
                source = "donor" if direction == "base_to_donor" else "base"
                before = cache[tuple(row[f"{recipient}_ids"])]["logits"].to(device)
                source_logits = cache[tuple(row[f"{source}_ids"])]["logits"].to(device)
                if row["family_id"] in TARGETS:
                    cell = _target_cell(row, direction, before, source_logits, after[index])
                    cell["full_vocabulary_logit_rms"] = float(
                        (after[index] - before).square().mean().sqrt())
                elif row["family_id"] == RELATION:
                    cell = _relation_cell(row, direction, before, source_logits, after[index])
                    cell["full_vocabulary_logit_rms"] = float(
                        (after[index] - before).square().mean().sqrt())
                else:
                    cell = _control_cell(row, direction, before, after[index])
                cell.update({"row_id": row["row_id"], "group_id": row["group_id"],
                             "intervention_vector_norm": float(norms[index])})
                raw[arm][row["family_id"]][direction].append(cell)
    return json.loads(json.dumps(raw)), calls, diagnostics


def _effect_report(cells: list[dict], seed: int, positive_bar: float) -> dict:
    effects = [item["effect"] for item in cells]
    natural = [item["natural_effect"] for item in cells]
    mean_den, median_den = float(np.mean(natural)), float(np.median(natural))
    item = {"n": len(cells), "mean_effect": float(np.mean(effects)),
            "median_effect": float(np.median(effects)), "mean_natural_effect": mean_den,
            "median_natural_effect": median_den,
            "mean_recovery": float(np.mean(effects)) / mean_den if mean_den > 0 else None,
            "median_recovery": float(np.median(effects)) / median_den if median_den > 0 else None,
            "positive_fraction": float(np.mean(np.asarray(effects) > 0)),
            "bootstrap95_lower_mean_effect": lower(effects, seed)}
    item["passed"] = bool(item["mean_recovery"] is not None and item["median_recovery"] is not None
                          and item["mean_recovery"] >= .5 and item["median_recovery"] >= .5
                          and item["positive_fraction"] >= positive_bar
                          and item["bootstrap95_lower_mean_effect"] > 0)
    return item


def arm_report(raw: dict, arm: str, seed: int,
               reference_scales: dict[str, float] | None = None) -> dict:
    targets, target_pass = {}, True
    for family in TARGETS:
        targets[family] = {}
        for direction in DIRECTIONS:
            cells = raw[arm][family][direction]
            item = _effect_report(cells, seed, .75); seed += 1
            item["target_answer_best_fraction"] = float(np.mean([
                cell["target_answer_best"] for cell in cells]))
            item["passed"] &= item["target_answer_best_fraction"] >= .5
            targets[family][direction] = item
            target_pass &= item["passed"]
    relation, relation_pass = {}, True
    for direction in DIRECTIONS:
        item = _effect_report(raw[arm][RELATION][direction], seed, .65); seed += 1
        relation[direction] = item; relation_pass &= item["passed"]
    target_cells = [cell for family in TARGETS for direction in DIRECTIONS
                    for cell in raw[arm][family][direction]]
    observed_scales = {
        "answer_effect": float(np.median([abs(cell["effect"]) for cell in target_cells])),
        "logit_rms": float(np.median([cell["full_vocabulary_logit_rms"] for cell in target_cells])),
        "intervention_norm": float(np.median([cell["intervention_vector_norm"] for cell in target_cells])),
    }
    scales = observed_scales if reference_scales is None else reference_scales
    controls, control_pass = {}, min(scales.values()) > 0
    def relative(value: float, scale: float) -> float:
        return float(value / scale) if scale > 0 else float("inf")
    for family in CONTROLS:
        controls[family] = {}
        for direction in DIRECTIONS:
            cells = raw[arm][family][direction]
            item = {
                "n": len(cells),
                "median_intervention_norm_fraction": relative(float(np.median([
                    cell["intervention_vector_norm"] for cell in cells])), scales["intervention_norm"]),
                "median_absolute_margin_change_fraction": relative(float(np.median([
                    abs(cell["registered_margin_change"]) for cell in cells])), scales["answer_effect"]),
                "median_logit_rms_fraction": relative(float(np.median([
                    cell["full_vocabulary_logit_rms"] for cell in cells])), scales["logit_rms"]),
            }
            if family == "sequence_step_two_conflict":
                item["preference_sign_preserved_fraction"] = float(np.mean([
                    cell["preference_sign_preserved"] for cell in cells]))
                behavioral = item["preference_sign_preserved_fraction"] >= .75
            else:
                item["registered_answer_preserved_fraction"] = float(np.mean([
                    cell["registered_answer_best"] for cell in cells]))
                item["mean_ce_increase"] = float(np.mean([cell["ce_increase"] for cell in cells]))
                behavioral = (item["registered_answer_preserved_fraction"] >= .75
                              and item["mean_ce_increase"] <= .10)
            item["passed"] = bool(item["median_intervention_norm_fraction"] >= .10
                                  and item["median_absolute_margin_change_fraction"] <= .25
                                  and item["median_logit_rms_fraction"] <= .25 and behavioral)
            controls[family][direction] = item; control_pass &= item["passed"]
    return {"targets": targets, "relation": relation, "controls": controls,
            "observed_target_scales": observed_scales,
            "control_reference_scales": scales,
            "target_pass": bool(target_pass), "relation_pass": bool(relation_pass),
            "controls_pass": bool(control_pass),
            "passed": bool(target_pass and relation_pass and control_pass)}


def choose(reports: dict, order: tuple[str, ...]) -> dict:
    eligible = [arm for arm in order if reports.get(arm, {}).get("passed")]
    return {"fixed_order": list(order), "eligible_arms": eligible,
            "selected_arm": eligible[0] if eligible else None}


def price(rows: list[dict], positions: dict[str, dict]) -> dict:
    output = {}
    for split in ("FIT", "SELECT"):
        split_rows = [row for row in rows if row["split"] == split]
        capture_chunks = len(chunk_by_length(list(unique_sequences(rows, positions, split)), len))
        oriented = [(row, direction) for row in split_rows for direction in DIRECTIONS]
        intervention_chunks = len(chunk_by_length(oriented, lambda item: len(
            item[0]["base_ids" if item[1] == "base_to_donor" else "donor_ids"])))
        output[split] = {"rows": len(split_rows), "unique_endpoint_capture_chunks": capture_chunks,
                         "oriented_intervention_chunks": intervention_chunks}
    maximum = (output["FIT"]["unique_endpoint_capture_chunks"] + 1
               + len(SITE_ARMS) * output["FIT"]["oriented_intervention_chunks"]
               + len(FACTOR_ARMS) * output["FIT"]["oriented_intervention_chunks"]
               + output["SELECT"]["unique_endpoint_capture_chunks"] + 1
               + (1 + len(FACTOR_ARMS)) * output["SELECT"]["oriented_intervention_chunks"])
    output["maximum_forwards_if_all_conditionals_open"] = maximum
    return output


def main() -> None:
    started = time.time()
    rows, positions = load_authority()
    declared_price = price(rows, positions)
    assert declared_price["maximum_forwards_if_all_conditionals_open"] == MAXIMUM_FORWARDS
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({"status": "dryrun_passed", "rung": 577,
                          "rows": len(rows), "price": declared_price,
                          "site_arms": list(SITE_ARMS), "factor_arms": list(FACTOR_ARMS),
                          "r576_final_cached_value_comparator_duplicated": False,
                          "model_loaded": False, "model_forwards": 0, "model_backwards": 0,
                          "FINAL_TEST_or_OOD_opened": False}, indent=2))
        return
    if OUT.exists():
        raise RuntimeError("R577 result namespace already exists")
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    fit_cache, fit_capture_calls, fit_exact = capture_split(model, rows, positions, "FIT")
    fit_site_raw, fit_site_calls, fit_site_exact = evaluate_arms(
        model, rows, positions, "FIT", fit_cache, SITE_ARMS)
    fit_site_reports = {arm: arm_report(fit_site_raw, arm, SEED + 100 * index)
                        for index, arm in enumerate(SITE_ARMS)}
    site_choice = choose(fit_site_reports, SITE_ARMS)
    fit_factor_raw = fit_factor_reports = factor_choice = None
    fit_factor_calls = 0
    fit_factor_exact = {"head_source_sum_relative_squared_error": 0.0,
                        "value_split_relative_squared_error": 0.0}
    if fit_site_reports["a8_h73_complete"]["passed"]:
        fit_factor_raw, fit_factor_calls, fit_factor_exact = evaluate_arms(
            model, rows, positions, "FIT", fit_cache, FACTOR_ARMS)
        fit_factor_reports = {arm: arm_report(fit_factor_raw, arm, SEED + 1000 + 100 * index)
                              for index, arm in enumerate(FACTOR_ARMS)}
        factor_choice = choose(fit_factor_reports, FACTOR_ARMS)
    selected_factor = factor_choice["selected_arm"] if factor_choice else None
    select_cache = select_site_raw = select_site_report = select_factor_raw = None
    select_factor_reports = select_factor_report = None
    select_capture_calls = select_site_calls = select_factor_calls = 0
    select_exact = {"native_replay_relative_squared_error": 0.0,
                    "head_source_sum_relative_squared_error": 0.0,
                    "value_split_relative_squared_error": 0.0}
    opened = ["FIT"]
    if site_choice["selected_arm"] is not None:
        select_cache, select_capture_calls, select_exact = capture_split(
            model, rows, positions, "SELECT")
        select_site_raw, select_site_calls, site_errors = evaluate_arms(
            model, rows, positions, "SELECT", select_cache, (site_choice["selected_arm"],))
        selected_site_fit_scales = fit_site_reports[site_choice["selected_arm"]][
            "control_reference_scales"]
        select_site_report = arm_report(
            select_site_raw, site_choice["selected_arm"], SEED + 5000,
            reference_scales=selected_site_fit_scales)
        for key, value in site_errors.items(): select_exact[key] = max(select_exact[key], value)
        eligible_factors = tuple(factor_choice["eligible_arms"]) if factor_choice else ()
        if eligible_factors:
            select_factor_raw, select_factor_calls, factor_errors = evaluate_arms(
                model, rows, positions, "SELECT", select_cache, eligible_factors)
            select_factor_reports = {
                arm: arm_report(
                    select_factor_raw, arm, SEED + 6000 + 100 * FACTOR_ARMS.index(arm),
                    reference_scales=fit_factor_reports[arm]["control_reference_scales"])
                for arm in eligible_factors
            }
            select_factor_report = select_factor_reports[selected_factor]
            for key, value in factor_errors.items(): select_exact[key] = max(select_exact[key], value)
        opened.append("SELECT")
    exact_values = [fit_exact, fit_site_exact, fit_factor_exact, select_exact]
    exact = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                 and max(item.get("native_replay_relative_squared_error", 0.0)
                         for item in exact_values) <= 1e-10
                 and max(item["head_source_sum_relative_squared_error"] for item in exact_values) <= 1e-10
                 and max(item["value_split_relative_squared_error"] for item in exact_values) <= 1e-10)
    total_forwards = sum((fit_capture_calls, fit_site_calls, fit_factor_calls,
                          select_capture_calls, select_site_calls, select_factor_calls))
    if total_forwards > MAXIMUM_FORWARDS:
        raise RuntimeError(f"forward price exceeded: {total_forwards} > {MAXIMUM_FORWARDS}")
    selected_site_held = bool(exact and select_site_report and select_site_report["passed"])
    a8_selected = site_choice["selected_arm"] == "a8_h73_complete"
    selected_factor_held = bool(exact and select_factor_report and select_factor_report["passed"])
    result = {
        "rung": 577, "stage": "numeric_sequence_complete_state_and_factor_localization",
        "pred_a_exact_replay_and_semantic_factor_algebra": exact,
        "pred_b_complete_state_site_holds_fit_and_select": selected_site_held,
        "pred_c_a8_h73_shared_sequence_carrier": bool(selected_site_held and a8_selected),
        "pred_d_semantic_factor_holds_fit_and_select": selected_factor_held,
        "site_choice": site_choice, "factor_choice": factor_choice,
        "fit_site_reports": fit_site_reports, "fit_factor_reports": fit_factor_reports,
        "select_site_report": select_site_report, "select_factor_report": select_factor_report,
        "select_factor_reports": select_factor_reports,
        "fit_site_raw": fit_site_raw, "fit_factor_raw": fit_factor_raw,
        "select_site_raw": select_site_raw, "select_factor_raw": select_factor_raw,
        "exactness": {"fit_capture": fit_exact, "fit_sites": fit_site_exact,
                      "fit_factors": fit_factor_exact, "select": select_exact},
        "execution_price": {"declared": declared_price, "observed_forwards": total_forwards,
                            "model_backwards": 0, "fitted_vectors": 0, "weights_updated": False},
        "r576_external_comparator": "final-source L0 cached-value deletion; not duplicated in R577",
        "evaluated_splits": opened, "forbidden_splits_opened": [],
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_sha256": {str(path): sha256(path) for path in HASHES},
        "elapsed_seconds": time.time() - started,
        "decision": ("exact_semantic_factor" if selected_factor_held else
                     "a8_complete_but_registered_factors_fail" if selected_site_held and a8_selected else
                     "later_complete_state_site" if selected_site_held else "complete_state_site_null"),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key.startswith("pred_")
                      or key in {"site_choice", "factor_choice", "execution_price",
                                 "evaluated_splits", "decision"}}, indent=2))


if __name__ == "__main__":
    main()
