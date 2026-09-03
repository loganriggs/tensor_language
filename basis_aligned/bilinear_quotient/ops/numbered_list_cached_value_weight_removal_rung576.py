#!/usr/bin/env python3
"""R576: exact weight compilation and active removal of the list cached-value factor.

Pred A: compiled L0-value/L8-score terms exactly reproduce the R573 activation patch.
Pred B: deleting the term damages every registered numbered-list successor cell.
Pred C: the same active deletion preserves repeated-list/digit/word copy cells.
Pred D: digit, word, and cross-format sequence effects characterize shared reuse.
Null: A, B, or C fails. SELECT opens only after all required FIT gates pass.
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
POSITIONS = ROOT / "numeric_factor_removal_positions_rung575.json"
R573_RESULT = ROOT / "numbered_list_factor_localization_rung573_v2_results.json"
R574_AUDIT = ROOT / "numbered_list_factor_localization_rung574_audit.json"
PREREG = POLY / "NUMBERED_LIST_CACHED_VALUE_WEIGHT_REMOVAL_RUNG576_PREREGISTRATION.md"
OUT = ROOT / "numbered_list_cached_value_weight_removal_rung576_results.json"
HASHES = {
    ROWS: "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
    POSITIONS: "3663ebc48e5dca1ff336cb0627fc43c6db8d7d6e1666b81d7631ab150168dd4b",
    R573_RESULT: "052930b8b9086e8b7606e3d05929f521f468c04427be8d1182720f1772ee43ec",
    R574_AUDIT: "3d6580ee1a4f1bb77c07e4ee2b404bc23dc70f733db31425bc5da2a11a25a04e",
    PREREG: "a776ebc1df29a6f3193d3315e190ec9494c95905596e450461c002378f8f59b6",
}
LIST_TARGETS = ("list_two_line_state_shift", "list_three_line_state_shift",
                "list_surface_preserved", "list_middle_index_break", "list_step_two_conflict")
COPY_CONTROLS = ("list_repeated_index_control", "sequence_digit_copy_control",
                 "sequence_word_copy_control")
SEQUENCE_TARGETS = ("sequence_digit_state_shift", "sequence_word_state_shift",
                    "sequence_cross_format_shift")
FAMILIES = LIST_TARGETS + COPY_CONTROLS + SEQUENCE_TARGETS
ENDPOINTS = ("base", "donor")
HEADS = (7, 3)
LAYER = 8
BATCH = 24
BOOTSTRAPS = 2000
SEED = 576
ENC = tiktoken.get_encoding("gpt2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower(values: list[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[indices].mean(1), .025))


def candidate_ids(answer: str) -> torch.Tensor:
    if not answer.startswith(" "):
        strings = [str(value) for value in range(101)]
    elif answer.strip().isdigit():
        strings = [" " + str(value) for value in range(101)]
    else:
        strings = [" " + word for word in (
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen", "twenty")]
    return torch.tensor(sorted({ids[0] for text in strings if len(ids := ENC.encode(text)) == 1}),
                        dtype=torch.long)


def load_authority() -> tuple[list[dict], dict[str, dict]]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen authority changed: {path}")
    r573_result = json.loads(R573_RESULT.read_text())
    audit = json.loads(R574_AUDIT.read_text())
    assert r573_result["selected_factor_held"] is True
    assert r573_result["fit_choice"]["selected_arm"] == "final_label_cached_value"
    assert audit["all_checks_pass"] is True and audit["selected_arm"] == "final_label_cached_value"
    rows = [row for row in json.loads(ROWS.read_text())["rows"]
            if row["split"] in {"FIT", "SELECT"} and row["family_id"] in FAMILIES]
    positions_document = json.loads(POSITIONS.read_text())
    assert positions_document["model_loaded"] is False and positions_document["outcomes_opened"] == []
    positions = {item["row_id"]: item for item in positions_document["records"]}
    assert len(rows) == 528 and set(positions) == {row["row_id"] for row in rows}
    return rows, positions


def compiled_cached(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Compute lambda8 * W_V0 * z0 directly from embeddings and weights."""
    x0 = F.rms_norm(model.transformer.wte(tokens), (r573.D,))
    block0 = model.transformer.h[0]
    pre0 = block0.lambdas[0] * x0 + block0.lambdas[1] * x0
    z0 = F.rms_norm(pre0, (r573.D,))
    value0 = r573.linear(z0, block0.attn.c_v.weight).view(
        tokens.size(0), tokens.size(1), r573.N_HEAD, r573.HEAD_D)
    return model.transformer.h[LAYER].attn.lamb * value0


def projected_terms(tensors: dict, cached: torch.Tensor, finals: torch.Tensor,
                    score_sources: torch.Tensor, c_proj_weight: torch.Tensor,
                    value_sources: torch.Tensor | None = None) -> torch.Tensor:
    if value_sources is None:
        value_sources = score_sources
    arange = torch.arange(cached.size(0), device=cached.device)
    output = torch.zeros(cached.size(0), r573.D, dtype=cached.dtype, device=cached.device)
    for head in HEADS:
        score = tensors["pattern"][arange, head, finals, score_sources]
        weight = c_proj_weight[:, head * r573.HEAD_D:(head + 1) * r573.HEAD_D]
        value = cached[arange, value_sources, head]
        output += score[:, None] * F.linear(value, weight.to(value.dtype))
    return output


@torch.no_grad()
def compiled_forward(model: torch.nn.Module, tokens: torch.Tensor, finals: torch.Tensor,
                     sources: torch.Tensor, *, mode: str,
                     donor_cached: torch.Tensor | None = None,
                     donor_sources: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if mode not in {"remove", "donor_patch"}:
        raise ValueError(mode)
    direct_cached = compiled_cached(model, tokens)
    diagnostics = {"cached_bus_relative_squared_error": 0.0,
                   "projected_term_relative_squared_error": 0.0,
                   "head_source_sum_relative_squared_error": 0.0,
                   "value_split_relative_squared_error": 0.0}
    term_norm = None

    def attention(event):
        nonlocal term_norm
        if event.site != LAYER:
            return event.block.attn(event.state, event.first_value)
        write, tensors, errors = r573.replay_attention(
            event.state, event.first_value, event.block.attn, finals)
        for key, value in errors.items():
            diagnostics[key] = max(diagnostics[key], value)
        bus_error = float((direct_cached - tensors["cached"]).square().sum()) / max(
            float(tensors["cached"].square().sum()), 1e-30)
        diagnostics["cached_bus_relative_squared_error"] = bus_error
        native_term = projected_terms(tensors, direct_cached, finals, sources, event.block.attn.c_proj.weight)
        # Independent reconstruction via one flattened head-space source term.
        arange = torch.arange(tokens.size(0), device=tokens.device)
        head_delta = torch.zeros(tokens.size(0), r573.N_HEAD, r573.HEAD_D,
                                 dtype=event.state.dtype, device=tokens.device)
        for head in HEADS:
            score = tensors["pattern"][arange, head, finals, sources]
            head_delta[:, head] = score[:, None] * direct_cached[arange, sources, head]
        flattened = F.linear(head_delta.reshape(tokens.size(0), r573.D),
                             event.block.attn.c_proj.weight.to(head_delta.dtype))
        diagnostics["projected_term_relative_squared_error"] = float(
            (flattened - native_term).square().sum()) / max(float(native_term.square().sum()), 1e-30)
        term_norm = native_term.float().norm(dim=-1)
        if mode == "remove":
            replacement = torch.zeros_like(native_term)
        else:
            if donor_cached is None or donor_sources is None:
                raise RuntimeError("compiled donor patch lacks donor values")
            replacement = projected_terms(
                tensors, donor_cached, finals, sources, event.block.attn.c_proj.weight,
                value_sources=donor_sources)
        arange = torch.arange(tokens.size(0), device=tokens.device)
        write = write.clone()
        write[arange, finals] += (replacement - native_term).to(write.dtype)
        return write, event.first_value

    logits = facade.forward_with_dispatch(
        model, tokens, attention, lambda event: event.block.mlp(event.state), require_production=False)
    arange = torch.arange(tokens.size(0), device=tokens.device)
    assert term_norm is not None
    return logits[arange, finals].float(), term_norm, diagnostics


def batch_endpoint(items: list[tuple[dict, str]], positions: dict[str, dict], device: torch.device):
    length = len(items[0][0][f"{items[0][1]}_ids"])
    tokens, finals, sources = [], [], []
    for row, endpoint in items:
        ids = row[f"{endpoint}_ids"]
        assert len(ids) == length
        mapping = positions[row["row_id"]]["endpoints"][endpoint]
        tokens.append(ids); finals.append(mapping["query_position"]); sources.append(mapping["source_position"])
    return (torch.tensor(tokens, dtype=torch.long, device=device),
            torch.tensor(finals, dtype=torch.long, device=device),
            torch.tensor(sources, dtype=torch.long, device=device))


def chunks(items: list, length_fn) -> list[list]:
    ordered = sorted(items, key=lambda item: (length_fn(item), str(item)))
    output, cursor = [], 0
    while cursor < len(ordered):
        length = length_fn(ordered[cursor])
        chunk = []
        while cursor < len(ordered) and length_fn(ordered[cursor]) == length and len(chunk) < BATCH:
            chunk.append(ordered[cursor]); cursor += 1
        output.append(chunk)
    return output


@torch.no_grad()
def equivalence(model: torch.nn.Module, rows: list[dict], positions: dict[str, dict], split: str):
    selected = [row for row in rows if row["split"] == split
                and row["family_id"] in {"list_two_line_state_shift", "list_three_line_state_shift"}]
    device = next(model.parameters()).device
    calls, raw = 0, []
    maxima = defaultdict(float)
    for group in chunks(selected, lambda row: len(row["base_ids"])):
        base_items = [(row, "base") for row in group]
        donor_items = [(row, "donor") for row in group]
        bt, bf, bs = batch_endpoint(base_items, positions, device)
        dt, df, ds = batch_endpoint(donor_items, positions, device)
        base_lines = [[int(source)] for source in bs]
        donor_lines = [[int(source)] for source in ds]
        base_native, base_factors, base_diag = r573.factor_forward(model, bt, bf, base_lines)
        donor_native, donor_factors, donor_diag = r573.factor_forward(model, dt, df, donor_lines)
        base_activation, _, _ = r573.factor_forward(
            model, bt, bf, base_lines, donor_factors, "final_label_cached_value")
        donor_activation, _, _ = r573.factor_forward(
            model, dt, df, donor_lines, base_factors, "final_label_cached_value")
        direct_base, direct_donor = compiled_cached(model, bt), compiled_cached(model, dt)
        base_weight, _, base_weight_diag = compiled_forward(
            model, bt, bf, bs, mode="donor_patch", donor_cached=direct_donor, donor_sources=ds)
        donor_weight, _, donor_weight_diag = compiled_forward(
            model, dt, df, ds, mode="donor_patch", donor_cached=direct_base, donor_sources=bs)
        calls += 6
        for diag in (base_diag, donor_diag, base_weight_diag, donor_weight_diag):
            for key, value in diag.items():
                maxima[key] = max(maxima[key], float(value))
        for direction, activation, weight, native in (
            ("base_to_donor", base_activation, base_weight, base_native),
            ("donor_to_base", donor_activation, donor_weight, donor_native),
        ):
            difference = (activation - weight).float()
            relative = float(difference.square().sum()) / max(float(activation.float().square().sum()), 1e-30)
            maxima["activation_vs_weight_logits_relative_squared_error"] = max(
                maxima["activation_vs_weight_logits_relative_squared_error"], relative)
            for index, row in enumerate(group):
                raw.append({"row_id": row["row_id"], "family_id": row["family_id"],
                            "direction": direction,
                            "maximum_absolute_logit_difference": float(difference[index].abs().max()),
                            "relative_squared_logit_error": float(difference[index].square().sum()) /
                                max(float(activation[index].float().square().sum()), 1e-30),
                            "compiled_logit_rms_from_native": float(
                                (weight[index] - native[index]).square().mean().sqrt())})
    passed = (maxima["cached_bus_relative_squared_error"] <= 1e-10
              and maxima["projected_term_relative_squared_error"] <= 1e-10
              and maxima["head_source_sum_relative_squared_error"] <= 1e-10
              and maxima["value_split_relative_squared_error"] <= 1e-10
              and maxima["activation_vs_weight_logits_relative_squared_error"] <= 1e-10)
    return {"passed": bool(passed), "max_errors": dict(maxima), "raw": raw}, calls


def margin(logits: torch.Tensor, answer_id: int, answer: str) -> float:
    pool = candidate_ids(answer).to(logits.device)
    return float(logits[answer_id] - logits[pool[pool != answer_id]].max())


def ce(logits: torch.Tensor, answer_id: int) -> float:
    return float(torch.logsumexp(logits.float(), dim=-1) - logits[answer_id].float())


@torch.no_grad()
def removal(model: torch.nn.Module, rows: list[dict], positions: dict[str, dict], split: str):
    items = [(row, endpoint) for row in rows if row["split"] == split for endpoint in ENDPOINTS]
    device = next(model.parameters()).device
    raw = defaultdict(lambda: defaultdict(list))
    calls = 0
    replay_error = 0.0
    factor_errors = defaultdict(float)
    for group in chunks(items, lambda item: len(item[0][f"{item[1]}_ids"])):
        tokens, finals, sources = batch_endpoint(group, positions, device)
        native, _, native_diag = r573.factor_forward(
            model, tokens, finals, [[int(source)] for source in sources])
        removed, term_norm, remove_diag = compiled_forward(model, tokens, finals, sources, mode="remove")
        calls += 2
        for diag in (native_diag, remove_diag):
            for key, value in diag.items():
                factor_errors[key] = max(factor_errors[key], float(value))
        if calls == 2:
            direct_native = r573.native_logits(model, tokens)
            direct_native = direct_native[torch.arange(tokens.size(0), device=device), finals].float()
            calls += 1
            replay_error = max(replay_error, float((direct_native - native).square().sum()) /
                               max(float(direct_native.square().sum()), 1e-30))
        for index, (row, endpoint) in enumerate(group):
            answer, answer_id = row[f"{endpoint}_answer"], row[f"{endpoint}_answer_id"]
            before, after = native[index], removed[index]
            before_margin, after_margin = margin(before, answer_id, answer), margin(after, answer_id, answer)
            pool = candidate_ids(answer).to(after.device)
            raw[row["family_id"]][endpoint].append({
                "row_id": row["row_id"], "group_id": row["group_id"],
                "answer_id": answer_id, "native_margin": before_margin, "removed_margin": after_margin,
                "margin_damage": before_margin - after_margin,
                "native_ce": ce(before, answer_id), "removed_ce": ce(after, answer_id),
                "ce_increase": ce(after, answer_id) - ce(before, answer_id),
                "full_vocabulary_logit_rms": float((after - before).square().mean().sqrt()),
                "compiled_residual_term_norm": float(term_norm[index]),
                "answer_remains_best": bool(int(pool[after[pool].argmax()]) == answer_id),
            })
    return json.loads(json.dumps(raw)), {"model_forwards": calls,
        "native_replay_relative_squared_error": replay_error, **factor_errors}


def target_report(raw: dict, families: tuple[str, ...], seed: int):
    report, passed = {}, True
    for family in families:
        report[family] = {}
        for endpoint in ENDPOINTS:
            cells = raw[family][endpoint]
            damage = [cell["margin_damage"] for cell in cells]
            ce_values = [cell["ce_increase"] for cell in cells]
            item = {"n": len(cells), "mean_margin_damage": float(np.mean(damage)),
                    "median_margin_damage": float(np.median(damage)),
                    "positive_margin_damage_fraction": float(np.mean(np.asarray(damage) > 0)),
                    "bootstrap95_lower_mean_margin_damage": lower(damage, seed),
                    "mean_ce_increase": float(np.mean(ce_values)),
                    "bootstrap95_lower_mean_ce_increase": lower(ce_values, seed + 1),
                    "median_logit_rms": float(np.median([cell["full_vocabulary_logit_rms"] for cell in cells])),
                    "median_term_norm": float(np.median([cell["compiled_residual_term_norm"] for cell in cells]))}
            seed += 2
            item["passed"] = bool(item["positive_margin_damage_fraction"] >= .75
                                  and item["bootstrap95_lower_mean_margin_damage"] > 0
                                  and item["bootstrap95_lower_mean_ce_increase"] > 0)
            report[family][endpoint] = item
            passed &= item["passed"]
    return report, bool(passed), seed


def fit_scales(raw: dict) -> dict:
    cells = [cell for family in LIST_TARGETS for endpoint in ENDPOINTS for cell in raw[family][endpoint]]
    return {"margin_damage": float(np.median([abs(cell["margin_damage"]) for cell in cells])),
            "logit_rms": float(np.median([cell["full_vocabulary_logit_rms"] for cell in cells])),
            "term_norm": float(np.median([cell["compiled_residual_term_norm"] for cell in cells]))}


def relative_to_fit(value: float, scale: float) -> float:
    """Return a fail-closed ratio without turning a scientific null into a crash."""
    return float(value / scale) if scale > 0 else float("inf")


def control_report(raw: dict, scales: dict):
    report, passed = {}, True
    for family in COPY_CONTROLS:
        report[family] = {}
        for endpoint in ENDPOINTS:
            cells = raw[family][endpoint]
            mean_ce = float(np.mean([cell["ce_increase"] for cell in cells]))
            answer_fraction = float(np.mean([cell["answer_remains_best"] for cell in cells]))
            term_fraction = relative_to_fit(
                float(np.median([cell["compiled_residual_term_norm"] for cell in cells])), scales["term_norm"])
            margin_fraction = relative_to_fit(
                float(np.median([abs(cell["margin_damage"]) for cell in cells])), scales["margin_damage"])
            rms_fraction = relative_to_fit(
                float(np.median([cell["full_vocabulary_logit_rms"] for cell in cells])), scales["logit_rms"])
            item = {"n": len(cells), "median_term_norm_fraction_of_fit_list": term_fraction,
                    "answer_preserved_fraction": answer_fraction, "mean_ce_increase": mean_ce,
                    "median_absolute_margin_change_fraction_of_fit_list": margin_fraction,
                    "median_logit_rms_fraction_of_fit_list": rms_fraction}
            item["passed"] = bool(term_fraction >= .10 and answer_fraction >= .75 and mean_ce <= .10
                                  and margin_fraction <= .25 and rms_fraction <= .25)
            report[family][endpoint] = item
            passed &= item["passed"]
    return report, bool(passed)


def score(raw: dict, scales: dict | None, seed: int):
    list_report, list_pass, seed = target_report(raw, LIST_TARGETS, seed)
    if scales is None:
        scales = fit_scales(raw)
    copy_report, copy_pass = control_report(raw, scales)
    sequence_report, sequence_pass, _ = target_report(raw, SEQUENCE_TARGETS, seed)
    return {"list_necessity": list_report, "list_necessity_pass": list_pass,
            "active_copy_controls": copy_report, "active_copy_controls_pass": copy_pass,
            "sequence_shared_characterization": sequence_report,
            "all_sequence_successor_cells_pass": sequence_pass,
            "required_pass": bool(list_pass and copy_pass)}, scales


def count_chunks(rows: list[dict], split: str):
    pair_rows = [row for row in rows if row["split"] == split
                 and row["family_id"] in {"list_two_line_state_shift", "list_three_line_state_shift"}]
    endpoint_items = [(row, endpoint) for row in rows if row["split"] == split for endpoint in ENDPOINTS]
    return (len(chunks(pair_rows, lambda row: len(row["base_ids"]))),
            len(chunks(endpoint_items, lambda item: len(item[0][f"{item[1]}_ids"]))))


def main() -> None:
    started = time.time()
    rows, positions = load_authority()
    fit_pair, fit_remove = count_chunks(rows, "FIT")
    select_pair, select_remove = count_chunks(rows, "SELECT")
    # Equivalence costs six calls/chunk. Removal costs two calls/chunk plus one replay identity call/split.
    maximum_forwards = 6 * (fit_pair + select_pair) + 2 * (fit_remove + select_remove) + 2
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({"status": "dryrun_passed", "rung": 576, "rows": len(rows),
                          "fit_pair_chunks": fit_pair, "fit_removal_chunks": fit_remove,
                          "select_pair_chunks": select_pair, "select_removal_chunks": select_remove,
                          "maximum_forwards_if_SELECT_opens": maximum_forwards,
                          "model_backwards": 0, "model_loaded": False,
                          "FINAL_TEST_or_OOD_opened": False}, indent=2))
        return
    if OUT.exists():
        raise RuntimeError("R576 result namespace already exists")
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    fit_equivalence, fit_eq_calls = equivalence(model, rows, positions, "FIT")
    fit_raw, fit_execution = removal(model, rows, positions, "FIT")
    fit_report, scales = score(fit_raw, None, SEED)
    fit_required = bool(fit_equivalence["passed"] and fit_report["required_pass"])
    opened = ["FIT"]
    select_equivalence = select_raw = select_report = select_execution = None
    select_required = False
    select_calls = 0
    if fit_required:
        select_equivalence, select_eq_calls = equivalence(model, rows, positions, "SELECT")
        select_raw, select_execution = removal(model, rows, positions, "SELECT")
        select_report, _ = score(select_raw, scales, SEED + 1000)
        select_required = bool(select_equivalence["passed"] and select_report["required_pass"])
        select_calls = select_eq_calls + select_execution["model_forwards"]
        opened.append("SELECT")
    total_forwards = fit_eq_calls + fit_execution["model_forwards"] + select_calls
    exact = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                 and fit_equivalence["passed"]
                 and fit_execution["native_replay_relative_squared_error"] <= 1e-12
                 and (select_execution is None or select_execution["native_replay_relative_squared_error"] <= 1e-12))
    result = {"rung": 576, "stage": "numbered_list_cached_value_weight_compilation_and_active_removal",
              "pred_a_exact_weight_compilation": exact,
              "pred_b_list_necessity": bool(fit_report["list_necessity_pass"] and select_report is not None
                                             and select_report["list_necessity_pass"]),
              "pred_c_active_copy_preservation": bool(fit_report["active_copy_controls_pass"]
                                                       and select_report is not None
                                                       and select_report["active_copy_controls_pass"]),
              "pred_d_shared_sequence_successor": bool(fit_report["all_sequence_successor_cells_pass"]
                                                        and select_report is not None
                                                        and select_report["all_sequence_successor_cells_pass"]),
              "all_required_gates_pass": bool(exact and fit_required and select_required),
              "fit_equivalence": fit_equivalence, "fit_report": fit_report,
              "fit_scales": scales, "select_equivalence": select_equivalence,
              "select_report": select_report, "fit_raw": fit_raw, "select_raw": select_raw,
              "execution": {"fit": fit_execution, "select": select_execution,
                            "maximum_forwards": maximum_forwards},
              "model_forwards": total_forwards, "model_backwards": 0, "model_weights_updated": False,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "input_sha256": {str(path): sha256(path) for path in HASHES},
              "evaluated_splits": opened, "forbidden_splits_opened": [],
              "elapsed_seconds": time.time() - started,
              "decision": "weight_factor_selectively_removable" if exact and fit_required and select_required else
                          ("invalid_compilation" if not exact else "removal_or_selectivity_null"),
              "next_step": "register_weight_translation_and_open_OOD_only_after_audit" if exact and fit_required and select_required
                           else "retain_activation_identification_without_adoption_and_audit_failure"}
    if total_forwards > maximum_forwards:
        raise RuntimeError(f"forward price exceeded: {total_forwards} > {maximum_forwards}")
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: result[key] for key in result if key.startswith("pred_")
                      or key in {"all_required_gates_pass", "model_forwards", "evaluated_splits",
                                 "decision", "next_step"}}, indent=2))


if __name__ == "__main__":
    main()
