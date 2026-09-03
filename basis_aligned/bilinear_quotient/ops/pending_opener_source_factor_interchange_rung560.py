#!/usr/bin/env python3
"""R560: exact L13H8 semantic-source score/payload causal interchange."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bilin18_observed_model_facade as facade  # noqa: E402


ROWS = ROOT / "pending_opener_three_value_fresh_rows_rung545.json"
RECEIPT = ROOT / "pending_opener_three_value_fresh_rows_rung545_receipt.json"
CEILING = ROOT / "pending_opener_three_value_confirmation_rung546_results.json"
CEILING_AUDIT = ROOT / "pending_opener_three_value_confirmation_rung548_audit.json"
SOURCE_AUDIT = ROOT / "pending_opener_source_positions_rung560_audit.json"
PREREG = POLY / "PENDING_OPENER_SOURCE_FACTOR_INTERCHANGE_RUNG560_PREREGISTRATION.md"
OUT = ROOT / "pending_opener_source_factor_interchange_rung560_results.json"
HASHES = {
    ROWS: "07b64d2e48a6ca67685c81d3475a064daba612d6fe7ff233efd5b6c157b940a9",
    RECEIPT: "a6b3e7468f510277b247cb78148b619625ecdde07f9ba264e5358f7bb5138609",
    CEILING: "209b9bfcc20bff13bb37d822137003d6878506e66b0d9321ba0a0f7e9d8f2c5c",
    CEILING_AUDIT: "25acb35355f457163c1ed1183aeb55aea0c08a224992d688250ba5e272564875",
    SOURCE_AUDIT: "4bcd51ff290aed85bb18852a11a229a4720326c1d27cdd13d147f76f7ac894d7",
    PREREG: "1c0ebf715eb1c75335b182dfe79e11c86eaf19b8571208e9fe01f8216cf1cee3",
}
TARGETS = ("direct_three_value_type_substitution", "completed_then_reopened_three_value_order")
CONTROLS = (
    "pending_type_preserved_surface_rewrite",
    "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
)
FAMILIES = TARGETS + CONTROLS
DIRECTIONS = ("base_to_donor", "donor_to_base")
ARMS = ("score", "payload", "joint")
ARM_COST = {"score": 1, "payload": 1, "joint": 2}
CLOSERS = (8, 60, 1)
PATCH_LAYER, PATCH_HEAD = 13, 8
D, HEADS, HEAD_D = 1152, 9, 128
BATCH = 18
BOOTSTRAPS, SEED = 2000, 560


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_authority() -> tuple[dict, dict, dict]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    document = json.loads(ROWS.read_text())
    ceiling = json.loads(CEILING.read_text())
    source = json.loads(SOURCE_AUDIT.read_text())
    assert ceiling["all_gates_pass"] is True
    assert source["all_checks_pass"] is True and source["row_count"] == 540
    assert source["model_loaded"] is False and source["outcomes_opened"] == []
    rows = [row for row in document["rows"] if row["split"] in {"FIT", "SELECT"}]
    assert len(rows) == 540
    sources = {row["row_id"]: row for row in source["records"]}
    assert len(sources) == 540 and {row["row_id"] for row in rows} == set(sources)
    return document, ceiling, sources


def native_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    x = model.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    logits = model.lm_head(F.rms_norm(x, (x.size(-1),)))
    return (30.0 * torch.tanh(logits / 30.0)).float()


def linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def replay_head(
    state: torch.Tensor,
    first_value: torch.Tensor,
    attention: torch.nn.Module,
    finals: torch.Tensor,
    sources: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    batch, length, width = state.shape
    assert width == D and first_value.shape == (batch, length, HEADS, HEAD_D)
    q = linear(state, attention.c_q.weight).view(batch, length, HEADS, HEAD_D)
    k = linear(state, attention.c_k.weight).view(batch, length, HEADS, HEAD_D)
    q2 = linear(state, attention.c_q2.weight).view(batch, length, HEADS, HEAD_D)
    k2 = linear(state, attention.c_k2.weight).view(batch, length, HEADS, HEAD_D)
    raw_value = linear(state, attention.c_v.weight).view(batch, length, HEADS, HEAD_D)
    value = (1 - attention.lamb) * raw_value + attention.lamb * first_value.view_as(raw_value)
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
    heads = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
    flattened = heads.transpose(1, 2).contiguous().view(batch, length, width)
    write = linear(flattened, attention.c_proj.weight)
    arange = torch.arange(batch, device=state.device)
    score = pattern[arange, PATCH_HEAD, finals, sources].float()
    head_value = value[arange, sources, PATCH_HEAD]
    weight = attention.c_proj.weight[:, PATCH_HEAD * HEAD_D:(PATCH_HEAD + 1) * HEAD_D]
    u = F.linear(head_value.float(), weight.float())
    native_term = F.linear((score.to(head_value.dtype).unsqueeze(-1) * head_value), weight.to(head_value.dtype))
    factor_term = score.unsqueeze(-1) * u
    error = float((factor_term - native_term.float()).square().sum()) / max(
        float(native_term.float().square().sum()), 1e-30,
    )
    return write, {"score": score, "u": u, "native_term": native_term}, error


@torch.no_grad()
def factor_forward(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    finals: torch.Tensor,
    semantic_sources: torch.Tensor,
    wrong_sources: torch.Tensor,
    *,
    donor: dict[str, dict[str, torch.Tensor]] | None = None,
    arm: str = "replay",
    donor_kind: str = "semantic",
    capture: bool = False,
) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]], float]:
    if arm not in {"replay", *ARMS} or donor_kind not in {"semantic", "wrong"}:
        raise ValueError((arm, donor_kind))
    if arm != "replay" and donor is None:
        raise ValueError("intervention needs donor factors")
    captured = {}
    reconstruction = 0.0

    def attention(event):
        nonlocal reconstruction
        if event.site != PATCH_LAYER:
            return event.block.attn(event.state, event.first_value)
        write, semantic, error = replay_head(
            event.state, event.first_value, event.block.attn, finals, semantic_sources,
        )
        _, wrong, wrong_error = replay_head(
            event.state, event.first_value, event.block.attn, finals, wrong_sources,
        )
        reconstruction = max(reconstruction, error, wrong_error)
        if capture:
            captured.update({
                "semantic": {"score": semantic["score"].detach().clone(), "u": semantic["u"].detach().clone()},
                "wrong": {"score": wrong["score"].detach().clone(), "u": wrong["u"].detach().clone()},
            })
        if arm != "replay":
            assert donor is not None
            selected = donor[donor_kind]
            score = selected["score"] if arm in {"score", "joint"} else semantic["score"]
            u = selected["u"] if arm in {"payload", "joint"} else semantic["u"]
            hybrid = score.unsqueeze(-1) * u
            arange = torch.arange(tokens.size(0), device=tokens.device)
            write[arange, finals] += (hybrid.to(write.dtype) - semantic["native_term"])
        return write, event.first_value

    def mlp(event):
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=False).float()
    if capture and set(captured) != {"semantic", "wrong"}:
        raise RuntimeError("L13H8 factor capture failed")
    return logits, captured, reconstruction


def pad_endpoint(rows: list[dict], endpoint: str, source_map: dict, device: torch.device):
    length = max(len(row[f"{endpoint}_ids"]) for row in rows)
    tokens = torch.full((len(rows), length), 50256, dtype=torch.long, device=device)
    finals, sources, wrong = [], [], []
    for index, row in enumerate(rows):
        ids = row[f"{endpoint}_ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, device=device)
        finals.append(len(ids) - 1)
        sources.append(source_map[row["row_id"]][f"{endpoint}_source_position"])
        wrong.append(source_map[row["row_id"]][f"{endpoint}_wrong_source_position"])
    return (tokens, torch.tensor(finals, device=device), torch.tensor(sources, device=device),
            torch.tensor(wrong, device=device))


def closer_margin(logits: torch.Tensor, answer: int) -> float:
    alternatives = [token for token in CLOSERS if token != answer]
    return float(logits[answer] - logits[alternatives].mean())


def target_change(before: torch.Tensor, after: torch.Tensor, row: dict, direction: str) -> float:
    if direction == "base_to_donor":
        positive, negative = row["donor_answer_id"], row["base_answer_id"]
    else:
        positive, negative = row["base_answer_id"], row["donor_answer_id"]
    return float((after[positive] - after[negative]) - (before[positive] - before[negative]))


def ceiling_map(ceiling: dict) -> dict[tuple[str, str, str, str], dict]:
    output = {}
    for split in ("FIT", "SELECT"):
        for family in FAMILIES:
            for direction in DIRECTIONS:
                for row in ceiling["raw_site_effects"][split][family][direction]:
                    output[(split, family, direction, row["row_id"])] = row
    return output


def append_measurement(raw: dict, *, split: str, family: str, direction: str, arm: str,
                       kind: str, row: dict, before: torch.Tensor, after: torch.Tensor,
                       reference: dict) -> None:
    if family in TARGETS:
        change = target_change(before, after, row, direction)
        full = reference["endpoint_change"]
        recovery = change / full if full > 1e-12 else None
        cell = {"row_id": row["row_id"], "group_id": row["group_id"],
                "endpoint_change": change, "complete_head_endpoint_change": full,
                "recovery": recovery}
    else:
        answer = row["base_answer_id"] if direction == "base_to_donor" else row["donor_answer_id"]
        change = closer_margin(after, answer) - closer_margin(before, answer)
        rms = float((after - before).square().mean().sqrt())
        cell = {"row_id": row["row_id"], "group_id": row["group_id"],
                "closer_margin_change": change, "full_vocabulary_logit_rms": rms,
                "complete_head_endpoint_change": reference["endpoint_change"],
                "complete_head_full_vocabulary_rms": reference["full_logit_rms"]}
    raw[split][family][direction][arm][kind].append(cell)


@torch.no_grad()
def evaluate(model, document: dict, ceiling: dict, source_map: dict, split: str, arms: tuple[str, ...]):
    rows = sorted(
        [row for row in document["rows"] if row["split"] == split],
        key=lambda row: (len(row["base_ids"]), len(row["donor_ids"]), row["family_id"], row["row_id"]),
    )
    references = ceiling_map(ceiling)
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))))
    device = next(model.parameters()).device
    calls = 0
    replay_error = 0.0
    factor_error = 0.0
    cursor = 0
    while cursor < len(rows):
        base_len, donor_len = len(rows[cursor]["base_ids"]), len(rows[cursor]["donor_ids"])
        chunk = []
        while cursor < len(rows) and len(rows[cursor]["base_ids"]) == base_len \
                and len(rows[cursor]["donor_ids"]) == donor_len and len(chunk) < BATCH:
            chunk.append(rows[cursor])
            cursor += 1
        base_tokens, base_finals, base_sources, base_wrong = pad_endpoint(chunk, "base", source_map, device)
        donor_tokens, donor_finals, donor_sources, donor_wrong = pad_endpoint(chunk, "donor", source_map, device)
        base_logits, base_factors, base_error = factor_forward(
            model, base_tokens, base_finals, base_sources, base_wrong, capture=True,
        )
        donor_logits, donor_factors, donor_error = factor_forward(
            model, donor_tokens, donor_finals, donor_sources, donor_wrong, capture=True,
        )
        calls += 2
        factor_error = max(factor_error, base_error, donor_error)
        if calls == 2:
            for tokens, logits in ((base_tokens, base_logits), (donor_tokens, donor_logits)):
                native = native_logits(model, tokens)
                calls += 1
                replay_error = max(replay_error, float((native - logits).square().sum()) /
                                   max(float(native.square().sum()), 1e-30))
        for arm in arms:
            for kind in ("semantic", "wrong"):
                base_patch, _, error = factor_forward(
                    model, base_tokens, base_finals, base_sources, base_wrong,
                    donor=donor_factors, arm=arm, donor_kind=kind,
                )
                donor_patch, _, reverse_error = factor_forward(
                    model, donor_tokens, donor_finals, donor_sources, donor_wrong,
                    donor=base_factors, arm=arm, donor_kind=kind,
                )
                calls += 2
                factor_error = max(factor_error, error, reverse_error)
                for index, row in enumerate(chunk):
                    family = row["family_id"]
                    for direction, before_all, after_all, finals in (
                        ("base_to_donor", base_logits, base_patch, base_finals),
                        ("donor_to_base", donor_logits, donor_patch, donor_finals),
                    ):
                        reference = references[(split, family, direction, row["row_id"])]
                        append_measurement(
                            raw, split=split, family=family, direction=direction, arm=arm, kind=kind,
                            row=row, before=before_all[index, finals[index]],
                            after=after_all[index, finals[index]], reference=reference,
                        )
    plain = json.loads(json.dumps(raw))
    return plain, {"model_forwards": calls, "native_replay_relative_squared_error": replay_error,
                   "max_source_factor_relative_squared_reconstruction_error": factor_error}


def bootstrap_lower(values: list[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(1), .025))


def score(raw: dict, arms: tuple[str, ...], seed: int) -> dict:
    reports = {}
    for arm in arms:
        target_reports, control_reports, wrong_reports = {}, {}, {}
        passed = True
        for family in TARGETS:
            target_reports[family], wrong_reports[family] = {}, {}
            for direction in DIRECTIONS:
                cells = raw[family][direction][arm]["semantic"]
                recoveries = [cell["recovery"] for cell in cells]
                assert all(value is not None for value in recoveries)
                report = {"n": len(recoveries), "mean": float(np.mean(recoveries)),
                          "median": float(np.median(recoveries)),
                          "bootstrap95_lower_mean": bootstrap_lower(recoveries, seed),
                          "positive_fraction": float(np.mean(np.asarray(recoveries) > 0)),
                          "values": recoveries}
                seed += 1
                report["passed"] = bool(report["median"] >= .50 and report["bootstrap95_lower_mean"] > 0
                                        and report["positive_fraction"] >= .75)
                target_reports[family][direction] = report
                wrong = [cell["recovery"] for cell in raw[family][direction][arm]["wrong"]]
                wrong_report = {"n": len(wrong), "mean_absolute_recovery": float(np.mean(np.abs(wrong)))}
                wrong_report["passed"] = wrong_report["mean_absolute_recovery"] <= .25
                wrong_reports[family][direction] = wrong_report
                passed &= report["passed"] and wrong_report["passed"]
        for family in CONTROLS:
            control_reports[family] = {}
            for direction in DIRECTIONS:
                cells = raw[family][direction][arm]["semantic"]
                endpoint = np.asarray([cell["closer_margin_change"] for cell in cells])
                rms = np.asarray([cell["full_vocabulary_logit_rms"] for cell in cells])
                full_endpoint = np.asarray([cell["complete_head_endpoint_change"] for cell in cells])
                full_rms = np.asarray([cell["complete_head_full_vocabulary_rms"] for cell in cells])
                mean_abs = float(np.mean(np.abs(endpoint)))
                report = {"n": len(cells), "mean_absolute_closer_margin_change": mean_abs,
                          "fraction_of_complete_head_margin_change": mean_abs / float(np.mean(np.abs(full_endpoint))),
                          "mean_full_vocabulary_logit_rms": float(np.mean(rms)),
                          "fraction_of_complete_head_full_vocabulary_rms": float(np.mean(rms)) / float(np.mean(full_rms))}
                report["passed"] = bool(mean_abs <= .10
                                        and report["fraction_of_complete_head_margin_change"] <= .25
                                        and report["fraction_of_complete_head_full_vocabulary_rms"] <= .25)
                control_reports[family][direction] = report
                passed &= report["passed"]
        reports[arm] = {"targets": target_reports, "controls": control_reports,
                        "wrong_source_controls": wrong_reports, "passed": bool(passed)}
    return reports


def choose(reports: dict) -> dict:
    passing = [arm for arm, report in reports.items() if report["passed"]]
    def worst_lower(arm: str) -> float:
        return min(cell["bootstrap95_lower_mean"]
                   for family in reports[arm]["targets"].values() for cell in family.values())
    passing.sort(key=lambda arm: (ARM_COST[arm], -worst_lower(arm), arm))
    return {"eligible_arms": passing, "selected_arm": passing[0] if passing else None,
            "selected_cost": ARM_COST[passing[0]] if passing else None}


def interaction_report(raw: dict) -> dict:
    output = {}
    for family in TARGETS:
        output[family] = {}
        for direction in DIRECTIONS:
            by_arm = {
                arm: {cell["row_id"]: cell["endpoint_change"]
                      for cell in raw[family][direction][arm]["semantic"]}
                for arm in ARMS
            }
            row_ids = sorted(by_arm["joint"])
            values = [by_arm["joint"][row] - by_arm["score"][row] - by_arm["payload"][row]
                      for row in row_ids]
            output[family][direction] = {"n": len(values), "mean_interaction_logit": float(np.mean(values)),
                                         "mean_absolute_interaction_logit": float(np.mean(np.abs(values))),
                                         "values": values}
    return output


def synthetic_reports() -> dict:
    reports = {}
    for arm in ARMS:
        target = {family: {direction: {"bootstrap95_lower_mean": .4} for direction in DIRECTIONS}
                  for family in TARGETS}
        reports[arm] = {"targets": target, "passed": arm in {"payload", "joint"}}
    return reports


def main() -> None:
    started = time.time()
    document, ceiling, source_map = load_authority()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        choice = choose(synthetic_reports())
        assert choice["selected_arm"] == "payload"
        print(json.dumps({"status": "dryrun_passed", "rung": 560, "selected_arm": "payload",
                          "semantic_rows": len(source_map), "model_loaded": False,
                          "final_or_ood_opened": False}, indent=2))
        return
    if OUT.exists():
        raise RuntimeError("R560 result namespace already exists")
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    fit_raw, fit_execution = evaluate(model, document, ceiling, source_map, "FIT", ARMS)
    fit_reports = score(fit_raw, ARMS, SEED)
    choice = choose(fit_reports)
    select_raw = select_reports = select_execution = None
    select_held = False
    opened = ["FIT"]
    if choice["selected_arm"] is not None:
        arm = choice["selected_arm"]
        select_raw, select_execution = evaluate(model, document, ceiling, source_map, "SELECT", (arm,))
        select_reports = score(select_raw, (arm,), SEED + 100)
        select_held = select_reports[arm]["passed"]
        opened.append("SELECT")
    exact = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                 and fit_execution["native_replay_relative_squared_error"] <= 1e-12
                 and fit_execution["max_source_factor_relative_squared_reconstruction_error"] <= 1e-10
                 and (select_execution is None or (
                     select_execution["native_replay_relative_squared_error"] <= 1e-12
                     and select_execution["max_source_factor_relative_squared_reconstruction_error"] <= 1e-10)))
    result = {"rung": 560, "stage": "pending_opener_l13h8_source_factor_interchange",
              "pred_a_exact_instrument": exact,
              "pred_b_fit_selective_source_factor_exists": choice["selected_arm"] is not None,
              "pred_c_selected_source_factor_holds": bool(exact and select_held),
              "fit_reports": fit_reports, "fit_choice": choice,
              "fit_score_payload_interactions": interaction_report(fit_raw),
              "select_reports": select_reports, "selected_factor_held": bool(exact and select_held),
              "fit_raw": fit_raw, "select_raw": select_raw,
              "execution": {"fit": fit_execution, "select": select_execution},
              "model_forwards": fit_execution["model_forwards"] + (select_execution["model_forwards"] if select_execution else 0),
              "model_backwards": 0, "model_weights_updated": False,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "input_sha256": {str(path): sha256(path) for path in HASHES},
              "evaluated_splits": opened, "forbidden_splits_opened": [],
              "elapsed_seconds": time.time() - started,
              "decision": "held_source_factor" if exact and select_held else "source_factor_null"}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key.startswith("pred_") or key in {
                          "fit_choice", "selected_factor_held", "model_forwards",
                          "evaluated_splits", "decision",
                      }}, indent=2))


if __name__ == "__main__":
    main()
