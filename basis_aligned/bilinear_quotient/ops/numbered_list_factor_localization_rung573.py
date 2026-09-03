#!/usr/bin/env python3
"""R573: exact layer-8 list-label score/value factor localization.

Pred A: the custom attention replay is exact and the two-head complete-output
ceiling works in every FIT target cell.
Pred B: the first preregistered exact factor arm passing every FIT target and
control cell is selected.
Pred C: that one arm, without replacement, and the complete-output ceiling hold
on SELECT. FINAL_TEST/OOD remain closed; no model weight is updated.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
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


ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
RECEIPT = ROOT / "increment_two_hypothesis_rows_rung567_receipt.json"
OVERLAY = ROOT / "increment_rung568_semantic_role_overlay.json"
POSITIONS = ROOT / "numbered_list_semantic_positions_rung573.json"
CAPABILITY = ROOT / "numeric_two_hypothesis_capability_rung569_570_results.json"
CONFLICT = ROOT / "numbered_list_conflict_confirmation_rung572_results.json"
PREREG = POLY / "NUMBERED_LIST_FACTOR_LOCALIZATION_RUNG573_PREREGISTRATION.md"
OUT = ROOT / "numbered_list_factor_localization_rung573_results.json"
HASHES = {
    ROWS: "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
    RECEIPT: "02b2c37cc23434138accd63e920f417cda10f1c86a4c08c174537149ec2b1072",
    OVERLAY: "90c03d026b4daaae4794b02399967cbd3f9daf8b5412a24e13e594b4ba659765",
    POSITIONS: "b4f4f8e58c03deb3a015141656572dcf1fd0fe0c12027c358b441459c245c16b",
    CAPABILITY: "7cc56f22def334673e0035fad7c6a7d1fc58ab8edd3a99744bebd9fb4e6af7e7",
    CONFLICT: "3df046bdcc4fa4387a2dbef084ed732c5f6a05232b7fa64072af3cd4939daea1",
    PREREG: "12e06f11a0865396a85fb3b58c6b2fa23bb90ea28f80d94de88796a0f13d9365",
}
HYPOTHESIS = "numbered_list_index_successor"
TARGETS = ("list_two_line_state_shift", "list_three_line_state_shift")
CONTROLS = ("list_surface_preserved", "list_middle_index_break",
            "list_repeated_index_control", "list_step_two_conflict")
FAMILIES = TARGETS + CONTROLS
DIRECTIONS = ("base_to_donor", "donor_to_base")
HEADS = (7, 3)
LAYER = 8
D, N_HEAD, HEAD_D = 1152, 9, 128
BATCH = 24
BOOTSTRAPS = 2000
SEED = 573
ALL_ARMS = (
    "complete_heads",
    "all_label_joint",
    "final_label_joint",
    "final_label_score",
    "final_label_value",
    "final_label_cached_value",
    "final_label_own_value",
    "all_label_cached_value",
)
SELECTION_ORDER = (
    "final_label_cached_value",
    "final_label_value",
    "all_label_cached_value",
    "final_label_joint",
    "all_label_joint",
    "final_label_score",
    "final_label_own_value",
)
ENC = tiktoken.get_encoding("gpt2")
DIGIT_IDS = torch.tensor(sorted({ENC.encode(str(value))[0] for value in range(101)
                                 if len(ENC.encode(str(value))) == 1}), dtype=torch.long)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower(values: list[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(1), .025))


def load_authority() -> tuple[list[dict], dict[str, dict]]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    document = json.loads(ROWS.read_text())
    position_document = json.loads(POSITIONS.read_text())
    capability = json.loads(CAPABILITY.read_text())
    conflict = json.loads(CONFLICT.read_text())
    assert all(capability["hypothesis_results"][HYPOTHESIS][split]["all_pass"] is True
               for split in ("FIT", "SELECT"))
    assert conflict["all_gates_pass"] is True
    assert position_document["all_labels_are_single_semantic_tokens"] is True
    assert position_document["all_queries_are_final_newlines"] is True
    rows = [row for row in document["rows"] if row["hypothesis_id"] == HYPOTHESIS
            and row["split"] in {"FIT", "SELECT"}]
    assert len(rows) == 288 and all(row["family_id"] in FAMILIES for row in rows)
    positions = {item["row_id"]: item for item in position_document["mappings"]}
    assert len(positions) == len(rows) and set(positions) == {row["row_id"] for row in rows}
    for row in rows:
        mapping = positions[row["row_id"]]
        assert row["split"] == mapping["split"] and row["family_id"] == mapping["family_id"]
        assert len(row["base_ids"]) == len(row["donor_ids"])
        base = mapping["endpoints"]["base"]
        donor = mapping["endpoints"]["donor"]
        assert base["sequence_length"] == donor["sequence_length"] == len(row["base_ids"])
        assert len(base["label_positions"]) == len(donor["label_positions"])
        assert [item["line_index"] for item in base["label_positions"]] \
            == [item["line_index"] for item in donor["label_positions"]]
    return rows, positions


def native_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
    x0, first_value = x, None
    for block in model.transformer.h:
        x, first_value = block(x, first_value, x0)
    return (30 * torch.tanh(model.lm_head(F.rms_norm(x, (x.size(-1),))) / 30)).float()


def linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def replay_attention(state: torch.Tensor, first_value: torch.Tensor, attention: torch.nn.Module,
                     finals: torch.Tensor) -> tuple[torch.Tensor, dict, dict]:
    """Reproduce native attention and expose exact source terms for H7/H3."""
    batch, length, width = state.shape
    assert width == D and first_value.shape == (batch, length, N_HEAD, HEAD_D)
    q = linear(state, attention.c_q.weight).view(batch, length, N_HEAD, HEAD_D)
    k = linear(state, attention.c_k.weight).view(batch, length, N_HEAD, HEAD_D)
    q2 = linear(state, attention.c_q2.weight).view(batch, length, N_HEAD, HEAD_D)
    k2 = linear(state, attention.c_k2.weight).view(batch, length, N_HEAD, HEAD_D)
    current = linear(state, attention.c_v.weight).view(batch, length, N_HEAD, HEAD_D)
    own = (1 - attention.lamb) * current
    cached = attention.lamb * first_value.view_as(current)
    value = own + cached
    cos, sin = attention.rotary(q)
    module = sys.modules[type(attention).__module__]
    q = module.apply_rotary_emb(F.rms_norm(q, (HEAD_D,)), cos, sin)
    k = module.apply_rotary_emb(F.rms_norm(k, (HEAD_D,)), cos, sin)
    q2 = module.apply_rotary_emb(F.rms_norm(q2, (HEAD_D,)), cos, sin)
    k2 = module.apply_rotary_emb(F.rms_norm(k2, (HEAD_D,)), cos, sin)
    score1 = torch.einsum("bqhd,bkhd->bhqk", q, k) / HEAD_D
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HEAD_D
    pattern = score1 * score2
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device))
    pattern = pattern.masked_fill(~causal, 0)
    head_output = torch.einsum("bhqk,bkhd->bqhd", pattern, value)
    write = linear(head_output.reshape(batch, length, width), attention.c_proj.weight)
    arange = torch.arange(batch, device=state.device)
    reconstructed = torch.stack([
        (pattern[arange, head, finals, :, None] * value[:, :, head]).sum(1)
        for head in HEADS
    ], dim=1)
    observed = torch.stack([head_output[arange, finals, head] for head in HEADS], dim=1)
    head_error = float((reconstructed - observed).square().sum()) / max(
        float(observed.square().sum()), 1e-30)
    value_error = float((value - own - cached).square().sum()) / max(float(value.square().sum()), 1e-30)
    tensors = {"pattern": pattern, "value": value, "own": own, "cached": cached,
               "head_output": head_output}
    return write, tensors, {"head_source_sum_relative_squared_error": head_error,
                            "value_split_relative_squared_error": value_error}


def capture_rows(tensors: dict, finals: torch.Tensor, line_positions: list[list[int]]) -> list[dict]:
    output = []
    for index, positions in enumerate(line_positions):
        final = int(finals[index])
        item = {"complete": torch.stack([
                    tensors["head_output"][index, final, head] for head in HEADS
                ]).detach().cpu(), "labels": []}
        for position in positions:
            item["labels"].append({
                "score": torch.stack([tensors["pattern"][index, head, final, position]
                                      for head in HEADS]).detach().cpu(),
                "value": torch.stack([tensors["value"][index, position, head]
                                      for head in HEADS]).detach().cpu(),
                "cached": torch.stack([tensors["cached"][index, position, head]
                                       for head in HEADS]).detach().cpu(),
                "own": torch.stack([tensors["own"][index, position, head]
                                    for head in HEADS]).detach().cpu(),
            })
        output.append(item)
    return output


def modify_head_output(head_output: torch.Tensor, tensors: dict, finals: torch.Tensor,
                       line_positions: list[list[int]], donor: list[dict], arm: str) -> torch.Tensor:
    result = head_output.clone()
    for index, positions in enumerate(line_positions):
        final = int(finals[index])
        if arm == "complete_heads":
            for head_slot, head in enumerate(HEADS):
                result[index, final, head] = donor[index]["complete"][head_slot].to(result.device)
            continue
        selected = range(len(positions)) if arm.startswith("all_label") else (len(positions) - 1,)
        for line_index in selected:
            source = positions[line_index]
            for head_slot, head in enumerate(HEADS):
                base_score = tensors["pattern"][index, head, final, source]
                base_value = tensors["value"][index, source, head]
                native_term = base_score * base_value
                donor_score = donor[index]["labels"][line_index]["score"][head_slot].to(result.device)
                if arm in {"all_label_joint", "final_label_joint"}:
                    replacement = donor_score * donor[index]["labels"][line_index]["value"][head_slot].to(result.device)
                elif arm == "final_label_score":
                    replacement = donor_score * base_value
                elif arm == "final_label_value":
                    replacement = base_score * donor[index]["labels"][line_index]["value"][head_slot].to(result.device)
                elif arm in {"final_label_cached_value", "all_label_cached_value"}:
                    donor_cached = donor[index]["labels"][line_index]["cached"][head_slot].to(result.device)
                    replacement = base_score * (tensors["own"][index, source, head] + donor_cached)
                elif arm == "final_label_own_value":
                    donor_own = donor[index]["labels"][line_index]["own"][head_slot].to(result.device)
                    replacement = base_score * (donor_own + tensors["cached"][index, source, head])
                else:
                    raise ValueError(arm)
                result[index, final, head] += replacement - native_term
    return result


@torch.no_grad()
def factor_forward(model: torch.nn.Module, tokens: torch.Tensor, finals: torch.Tensor,
                   line_positions: list[list[int]], donor: list[dict] | None = None,
                   arm: str = "replay") -> tuple[torch.Tensor, list[dict], dict]:
    captured: list[dict] = []
    diagnostics = {"head_source_sum_relative_squared_error": 0.0,
                   "value_split_relative_squared_error": 0.0}

    def attention(event):
        nonlocal captured
        if event.site != LAYER:
            return event.block.attn(event.state, event.first_value)
        write, tensors, errors = replay_attention(event.state, event.first_value, event.block.attn, finals)
        for key, value in errors.items():
            diagnostics[key] = max(diagnostics[key], value)
        if arm == "replay":
            captured = capture_rows(tensors, finals, line_positions)
            return write, event.first_value
        if donor is None:
            raise RuntimeError("factor intervention has no donor factors")
        changed = modify_head_output(tensors["head_output"], tensors, finals, line_positions, donor, arm)
        write = linear(changed.reshape(tokens.size(0), tokens.size(1), D), event.block.attn.c_proj.weight)
        return write, event.first_value

    logits = facade.forward_with_dispatch(
        model, tokens, attention, lambda event: event.block.mlp(event.state), require_production=False)
    arange = torch.arange(tokens.size(0), device=tokens.device)
    return logits[arange, finals].float(), captured, diagnostics


def make_batch(rows: list[dict], positions: dict[str, dict], endpoint: str,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    length = len(rows[0][f"{endpoint}_ids"])
    assert all(len(row[f"{endpoint}_ids"]) == length for row in rows)
    tokens = torch.tensor([row[f"{endpoint}_ids"] for row in rows], dtype=torch.long, device=device)
    finals, labels = [], []
    for row in rows:
        item = positions[row["row_id"]]["endpoints"][endpoint]
        finals.append(item["final_query_position"])
        labels.append([source["token_position"] for source in item["label_positions"]])
    return tokens, torch.tensor(finals, dtype=torch.long, device=device), labels


def target_effect(before: torch.Tensor, after: torch.Tensor, row: dict, direction: str) -> float:
    if direction == "base_to_donor":
        positive, negative = row["donor_answer_id"], row["base_answer_id"]
    else:
        positive, negative = row["base_answer_id"], row["donor_answer_id"]
    return float((after[positive] - after[negative]) - (before[positive] - before[negative]))


def digit_margin(logits: torch.Tensor, answer: int) -> float:
    pool = DIGIT_IDS.to(logits.device)
    alternatives = pool[pool != answer]
    return float(logits[answer] - logits[alternatives].max())


def record(raw: dict, row: dict, direction: str, arm: str,
           before: torch.Tensor, after: torch.Tensor) -> None:
    family = row["family_id"]
    cell = {"row_id": row["row_id"], "group_id": row["group_id"],
            "full_vocabulary_logit_rms": float((after - before).square().mean().sqrt())}
    if family in TARGETS:
        cell["donor_direction_answer_margin_effect"] = target_effect(before, after, row, direction)
    else:
        answer = row["base_answer_id"] if direction == "base_to_donor" else row["donor_answer_id"]
        cell["registered_answer_margin_change"] = digit_margin(after, answer) - digit_margin(before, answer)
        pool = DIGIT_IDS.to(after.device)
        cell["registered_answer_remains_best"] = bool(int(pool[after[pool].argmax()]) == answer)
    raw[family][direction][arm].append(cell)


@torch.no_grad()
def evaluate(model: torch.nn.Module, rows: list[dict], positions: dict[str, dict],
             split: str, arms: tuple[str, ...]) -> tuple[dict, dict]:
    selected = sorted([row for row in rows if row["split"] == split],
                      key=lambda row: (len(row["base_ids"]), row["family_id"], row["row_id"]))
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    device = next(model.parameters()).device
    calls, cursor = 0, 0
    replay_error = 0.0
    factor_errors = {"head_source_sum_relative_squared_error": 0.0,
                     "value_split_relative_squared_error": 0.0}
    while cursor < len(selected):
        length = len(selected[cursor]["base_ids"])
        chunk = []
        while cursor < len(selected) and len(selected[cursor]["base_ids"]) == length and len(chunk) < BATCH:
            chunk.append(selected[cursor])
            cursor += 1
        base_tokens, base_finals, base_positions = make_batch(chunk, positions, "base", device)
        donor_tokens, donor_finals, donor_positions = make_batch(chunk, positions, "donor", device)
        base_logits, base_factors, base_diag = factor_forward(
            model, base_tokens, base_finals, base_positions)
        donor_logits, donor_factors, donor_diag = factor_forward(
            model, donor_tokens, donor_finals, donor_positions)
        calls += 2
        for diag in (base_diag, donor_diag):
            for key, value in diag.items():
                factor_errors[key] = max(factor_errors[key], value)
        if calls == 2:
            for tokens, finals, replay in ((base_tokens, base_finals, base_logits),
                                           (donor_tokens, donor_finals, donor_logits)):
                native = native_logits(model, tokens)
                native = native[torch.arange(tokens.size(0), device=device), finals].float()
                calls += 1
                replay_error = max(replay_error, float((native - replay).square().sum()) /
                                   max(float(native.square().sum()), 1e-30))
        for arm in arms:
            base_after, _, base_arm_diag = factor_forward(
                model, base_tokens, base_finals, base_positions, donor_factors, arm)
            donor_after, _, donor_arm_diag = factor_forward(
                model, donor_tokens, donor_finals, donor_positions, base_factors, arm)
            calls += 2
            for diag in (base_arm_diag, donor_arm_diag):
                for key, value in diag.items():
                    factor_errors[key] = max(factor_errors[key], value)
            for index, row in enumerate(chunk):
                record(raw, row, "base_to_donor", arm, base_logits[index], base_after[index])
                record(raw, row, "donor_to_base", arm, donor_logits[index], donor_after[index])
    return json.loads(json.dumps(raw)), {"model_forwards": calls,
        "native_replay_relative_squared_error": replay_error, **factor_errors}


def fit_scales(raw: dict) -> dict:
    complete = [cell for family in TARGETS for direction in DIRECTIONS
                for cell in raw[family][direction]["complete_heads"]]
    answer = [abs(cell["donor_direction_answer_margin_effect"]) for cell in complete]
    rms = [cell["full_vocabulary_logit_rms"] for cell in complete]
    scales = {"answer_margin": float(np.median(answer)), "full_vocabulary_logit_rms": float(np.median(rms))}
    if min(scales.values()) <= 0:
        raise RuntimeError(f"nonpositive FIT control scale: {scales}")
    return scales


def ceiling_report(raw: dict, seed: int) -> tuple[dict, bool]:
    report, passed = {}, True
    for family in TARGETS:
        report[family] = {}
        for direction in DIRECTIONS:
            values = [cell["donor_direction_answer_margin_effect"]
                      for cell in raw[family][direction]["complete_heads"]]
            item = {"n": len(values), "mean_effect": float(np.mean(values)),
                    "median_effect": float(np.median(values)),
                    "positive_fraction": float(np.mean(np.asarray(values) > 0)),
                    "bootstrap95_lower_mean_effect": lower(values, seed), "values": values}
            seed += 1
            item["passed"] = bool(item["positive_fraction"] >= .75
                                  and item["bootstrap95_lower_mean_effect"] > 0)
            report[family][direction] = item
            passed &= item["passed"]
    return report, bool(passed)


def arm_report(raw: dict, arm: str, scales: dict, seed: int) -> tuple[dict, bool]:
    targets, controls, passed = {}, {}, True
    for family in TARGETS:
        targets[family] = {}
        for direction in DIRECTIONS:
            values = [cell["donor_direction_answer_margin_effect"]
                      for cell in raw[family][direction][arm]]
            complete = [cell["donor_direction_answer_margin_effect"]
                        for cell in raw[family][direction]["complete_heads"]]
            mean_den, median_den = float(np.mean(complete)), float(np.median(complete))
            item = {"n": len(values), "mean_effect": float(np.mean(values)),
                    "median_effect": float(np.median(values)),
                    "complete_head_mean_effect": mean_den, "complete_head_median_effect": median_den,
                    "mean_recovery": float(np.mean(values)) / mean_den if mean_den > 0 else None,
                    "median_recovery": float(np.median(values)) / median_den if median_den > 0 else None,
                    "positive_fraction": float(np.mean(np.asarray(values) > 0)),
                    "bootstrap95_lower_mean_effect": lower(values, seed), "values": values}
            seed += 1
            item["passed"] = bool(item["mean_recovery"] is not None and item["median_recovery"] is not None
                                  and item["mean_recovery"] >= .5 and item["median_recovery"] >= .5
                                  and item["positive_fraction"] >= .75
                                  and item["bootstrap95_lower_mean_effect"] > 0)
            targets[family][direction] = item
            passed &= item["passed"]
    for family in CONTROLS:
        controls[family] = {}
        for direction in DIRECTIONS:
            cells = raw[family][direction][arm]
            answer = [abs(cell["registered_answer_margin_change"]) for cell in cells]
            rms = [cell["full_vocabulary_logit_rms"] for cell in cells]
            item = {"n": len(cells), "median_absolute_answer_margin_change": float(np.median(answer)),
                    "median_full_vocabulary_logit_rms": float(np.median(rms)),
                    "fraction_of_fit_target_answer_scale": float(np.median(answer)) / scales["answer_margin"],
                    "fraction_of_fit_target_logit_rms_scale": float(np.median(rms)) / scales["full_vocabulary_logit_rms"],
                    "registered_answer_preserved_fraction": float(np.mean([
                        cell["registered_answer_remains_best"] for cell in cells]))}
            item["passed"] = bool(item["fraction_of_fit_target_answer_scale"] <= .25
                                  and item["fraction_of_fit_target_logit_rms_scale"] <= .25
                                  and item["registered_answer_preserved_fraction"] >= .75)
            controls[family][direction] = item
            passed &= item["passed"]
    return {"targets": targets, "controls": controls, "passed": bool(passed)}, bool(passed)


def score(raw: dict, arms: tuple[str, ...], scales: dict, seed: int) -> tuple[dict, dict, bool]:
    ceiling, ceiling_pass = ceiling_report(raw, seed)
    reports = {}
    for index, arm in enumerate(arms):
        if arm == "complete_heads":
            continue
        reports[arm], _ = arm_report(raw, arm, scales, seed + 100 + 20 * index)
    return ceiling, reports, ceiling_pass


def choose(reports: dict) -> dict:
    passing = [arm for arm in SELECTION_ORDER if reports.get(arm, {}).get("passed")]
    return {"fixed_order": list(SELECTION_ORDER), "eligible_arms": passing,
            "selected_arm": passing[0] if passing else None}


def count_chunks(rows: list[dict], split: str) -> int:
    lengths = [len(row["base_ids"]) for row in rows if row["split"] == split]
    return sum((lengths.count(length) + BATCH - 1) // BATCH for length in set(lengths))


def synthetic_choice_test() -> None:
    reports = {arm: {"passed": arm in {"all_label_joint", "final_label_value"}}
               for arm in SELECTION_ORDER}
    assert choose(reports)["selected_arm"] == "final_label_value"


def main() -> None:
    started = time.time()
    rows, positions = load_authority()
    fit_chunks, select_chunks = count_chunks(rows, "FIT"), count_chunks(rows, "SELECT")
    # FIT: two capture calls plus two calls per eight arms in every chunk, and
    # two one-time native-replay checks. SELECT: capture plus complete+chosen.
    maximum_forwards = fit_chunks * (2 + 2 * len(ALL_ARMS)) + 2 + select_chunks * 6
    if os.environ.get("BQLIB_DRYRUN") == "1":
        synthetic_choice_test()
        print(json.dumps({"status": "dryrun_passed", "rung": 573, "rows": len(rows),
                          "fit_chunks": fit_chunks, "select_chunks": select_chunks,
                          "maximum_forwards": maximum_forwards, "model_backwards": 0,
                          "model_loaded": False, "FINAL_TEST_or_OOD_opened": False}, indent=2))
        return
    if OUT.exists():
        raise RuntimeError("R573 result namespace already exists")
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    observed_lambda = float(model.transformer.h[LAYER].attn.lamb.detach().cpu())
    if abs(observed_lambda - 4.0) > 1e-7:
        raise RuntimeError(f"layer-8 value mixing coefficient changed: {observed_lambda}")
    fit_raw, fit_execution = evaluate(model, rows, positions, "FIT", ALL_ARMS)
    scales = fit_scales(fit_raw)
    fit_ceiling, fit_reports, fit_ceiling_pass = score(fit_raw, ALL_ARMS, scales, SEED)
    choice = choose(fit_reports) if fit_ceiling_pass else choose({})
    select_raw = select_ceiling = select_reports = select_execution = None
    select_ceiling_pass = selected_held = False
    opened = ["FIT"]
    if choice["selected_arm"] is not None:
        select_arms = ("complete_heads", choice["selected_arm"])
        select_raw, select_execution = evaluate(model, rows, positions, "SELECT", select_arms)
        select_ceiling, select_reports, select_ceiling_pass = score(
            select_raw, select_arms, scales, SEED + 1000)
        selected_held = bool(select_ceiling_pass and select_reports[choice["selected_arm"]]["passed"])
        opened.append("SELECT")
    exact = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                 and fit_execution["native_replay_relative_squared_error"] <= 1e-12
                 and fit_execution["head_source_sum_relative_squared_error"] <= 1e-10
                 and fit_execution["value_split_relative_squared_error"] <= 1e-10
                 and (select_execution is None or (
                     select_execution["native_replay_relative_squared_error"] <= 1e-12
                     and select_execution["head_source_sum_relative_squared_error"] <= 1e-10
                     and select_execution["value_split_relative_squared_error"] <= 1e-10)))
    total_forwards = fit_execution["model_forwards"] + (
        select_execution["model_forwards"] if select_execution else 0)
    pred_a = bool(exact and fit_ceiling_pass)
    pred_b = bool(pred_a and choice["selected_arm"] is not None)
    pred_c = bool(pred_b and selected_held)
    result = {"rung": 573, "stage": "numbered_list_exact_label_factor_localization",
              "pred_a_exact_replay_and_fit_complete_head_ceiling": pred_a,
              "pred_b_fit_exact_factor_selected": pred_b,
              "pred_c_selected_factor_holds_on_select": pred_c,
              "all_gates_pass": bool(pred_a and pred_b and pred_c),
              "heads": [f"L{LAYER}H{head}" for head in HEADS], "layer8_value_lambda": observed_lambda,
              "fit_control_scales": scales, "fit_ceiling": fit_ceiling,
              "fit_factor_reports": fit_reports, "fit_choice": choice,
              "select_ceiling": select_ceiling, "select_factor_reports": select_reports,
              "selected_factor_held": selected_held, "fit_raw": fit_raw, "select_raw": select_raw,
              "execution": {"fit": fit_execution, "select": select_execution,
                            "maximum_forwards": maximum_forwards},
              "model_forwards": total_forwards, "model_backwards": 0, "model_weights_updated": False,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "input_sha256": {str(path): sha256(path) for path in HASHES},
              "evaluated_splits": opened, "forbidden_splits_opened": [],
              "elapsed_seconds": time.time() - started,
              "decision": "held_exact_label_factor" if pred_c else (
                  "complete_head_site_null" if not pred_a else "exact_label_factor_null"),
              "next_step": "compile_selected_factor_to_weights_and_downstream_consumers" if pred_c else
                           "retain_behavior_circuit_and_test_complete_state_cross_format_sites"}
    if total_forwards > maximum_forwards:
        raise RuntimeError(f"forward price exceeded: {total_forwards} > {maximum_forwards}")
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key.startswith("pred_")
                      or key in {"fit_choice", "selected_factor_held", "model_forwards",
                                 "evaluated_splits", "decision", "next_step"}}, indent=2))


if __name__ == "__main__":
    main()
