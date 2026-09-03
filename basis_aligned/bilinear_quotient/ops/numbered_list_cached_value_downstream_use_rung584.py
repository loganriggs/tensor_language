#!/usr/bin/env python3
"""R584: execute the frozen R582 cached-value downstream-use decomposition.

FIT selects one exact MLP response component.  Active nulls must pass before
SELECT can open.  FINAL_TEST and OOD are never loaded into an execution batch.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

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
import numbered_list_cached_value_weight_removal_rung576 as r576  # noqa: E402
import numbered_list_cached_value_downstream_use_rung582 as r582  # noqa: E402


ROWS = ROOT / "numbered_list_cached_value_downstream_use_rows_rung582.json"
RECEIPT = ROOT / "numbered_list_cached_value_downstream_use_rows_rung582_receipt.json"
R582_PREREG = POLY / "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG582_PREREGISTRATION.md"
R582_HELPER = ROOT / "ops" / "numbered_list_cached_value_downstream_use_rung582.py"
R584_PREREG = POLY / "NUMBERED_LIST_CACHED_VALUE_DOWNSTREAM_USE_RUNG584_IMPLEMENTATION.md"
OUT = ROOT / "numbered_list_cached_value_downstream_use_rung584_results.json"
DRYRUN_OUT = ROOT / "numbered_list_cached_value_downstream_use_rung584_dryrun.json"
HASHES = {
    ROWS: "84c6a78882a33c266b3875285f63ceaed746dac7810fce16b591f7b57763cf3b",
    RECEIPT: "1511cfd7fcfe729edf4427f9f88f8552c32230e013d01a0661767713fdc29148",
    R582_PREREG: "e7832dc77cabe7a1afba61c759188a0aca73802163cef1abe013ffaff5c987b3",
    R582_HELPER: "b0d99eeeef834091cf9ddfe77b58682f0e9a7e101e143a18570808dacb57bc1c",
    R584_PREREG: "d8e62cda2f6721499c5de2b0bc4c09411b201da743b1be3df0c2c976d343bea0",
}
SITES = r582.SITES
COMPONENTS = r582.COMPONENT_ARMS
NULLS = r582.NULL_ARMS
SELECTION = tuple((site, component) for site in SITES for component in COMPONENTS)
BATCH = r582.BATCH
EXACT_BAR = 1e-10
ENC = tiktoken.get_encoding("gpt2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_authority() -> list[dict]:
    r582.validate_authorities()
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen R582/R584 authority changed: {path}")
    document = json.loads(ROWS.read_text())
    receipt = json.loads(RECEIPT.read_text())
    if not (document["model_loaded"] is False and document["model_forwards"] == 0
            and document["outcomes_opened"] == []):
        raise RuntimeError("R582 rows are not outcome-blind")
    rows = document["rows"]
    validation = r582.validate_rows(rows)
    if receipt["rows_sha256"] != sha256(ROWS) or receipt["rows"] != validation["rows"]:
        raise RuntimeError("R582 receipt does not certify the exact rows")
    if any(row["split"] not in {"FIT", "SELECT", "FINAL_TEST", "OOD"} for row in rows):
        raise RuntimeError("unknown split in R582 rows")
    return rows


def chunks(items: Sequence, length_fn) -> list[list]:
    ordered = sorted(items, key=lambda item: (length_fn(item), str(item)))
    output, cursor = [], 0
    while cursor < len(ordered):
        length = length_fn(ordered[cursor])
        group = []
        while cursor < len(ordered) and length_fn(ordered[cursor]) == length and len(group) < BATCH:
            group.append(ordered[cursor]); cursor += 1
        output.append(group)
    return output


def eligible_null_rows(rows: Sequence[dict], split: str) -> list[dict]:
    return [row for row in rows if row["split"] == split and row["condition"] in {
        "factorial_copy", "factorial_successor", "surface_copy", "surface_successor"}]


def price(rows: Sequence[dict]) -> dict:
    split_batches = {split: len(chunks([row for row in rows if row["split"] == split],
                                      lambda row: len(row["ids"])))
                     for split in ("FIT", "SELECT")}
    null_batches = {split: len(chunks(eligible_null_rows(rows, split), lambda row: len(row["ids"])))
                    for split in ("FIT", "SELECT")}
    fit = 2 * split_batches["FIT"] + len(SELECTION) * split_batches["FIT"] + 1
    fit += len(NULLS) * null_batches["FIT"]
    select = 2 * split_batches["SELECT"] + len(COMPONENTS) * split_batches["SELECT"] + 1
    select += len(NULLS) * null_batches["SELECT"]
    return {
        "split_batches": split_batches,
        "null_eligible_batches": null_batches,
        "fit_maximum_forwards": fit,
        "conditional_select_maximum_forwards": select,
        "literal_executable_maximum_forwards": fit + select,
        "r582_conservative_maximum_forwards": 530,
    }


def _candidate_ids(representation: str) -> torch.Tensor:
    if representation == "list":
        strings = [str(value) for value in range(101)]
    elif representation == "digit":
        strings = [f" {value}" for value in range(101)]
    elif representation == "word":
        strings = [" " + r582.NUMBER_WORD[value] for value in sorted(r582.NUMBER_WORD)]
    else:
        raise ValueError(representation)
    return torch.tensor(sorted({ids[0] for text in strings if len(ids := ENC.encode(text)) == 1}),
                        dtype=torch.long)


def endpoint_stats(logits: torch.Tensor, row: Mapping[str, object]) -> dict:
    lse = float(torch.logsumexp(logits.float(), dim=-1))
    if row["condition"] == "step_two":
        structural = int(row["structural_answer_id"]); arithmetic = int(row["arithmetic_answer_id"])
        return {
            "logsumexp": lse,
            "structural_logit": float(logits[structural]),
            "arithmetic_logit": float(logits[arithmetic]),
            "arithmetic_minus_structural": float(logits[arithmetic] - logits[structural]),
        }
    answer_id = int(row["answer_id"])
    pool = _candidate_ids(str(row["representation"])).to(logits.device)
    alternatives = pool[pool != answer_id]
    answer_logit = float(logits[answer_id])
    max_other = float(logits[alternatives].max())
    return {
        "logsumexp": lse, "answer_logit": answer_logit,
        "max_other_candidate_logit": max_other,
        "margin": answer_logit - max_other, "ce": lse - answer_logit,
        "answer_best": bool(answer_logit >= float(logits[pool].max())),
    }


def torch_bilinear_response(mlp: torch.nn.Module, state_without: torch.Tensor,
                            state_with: torch.Tensor) -> dict[str, torch.Tensor | float]:
    delta = state_with - state_without
    left0 = F.linear(state_without, mlp.Left.weight.to(state_without.dtype))
    right0 = F.linear(state_without, mlp.Right.weight.to(state_without.dtype))
    leftd = F.linear(delta, mlp.Left.weight.to(delta.dtype))
    rightd = F.linear(delta, mlp.Right.weight.to(delta.dtype))
    cross = F.linear(leftd * right0 + left0 * rightd, mlp.Down.weight.to(delta.dtype))
    self_term = F.linear(leftd * rightd, mlp.Down.weight.to(delta.dtype))
    joint = cross + self_term
    direct = F.linear(
        F.linear(state_with, mlp.Left.weight.to(state_with.dtype))
        * F.linear(state_with, mlp.Right.weight.to(state_with.dtype))
        - left0 * right0,
        mlp.Down.weight.to(state_with.dtype),
    )
    error = float((joint.float() - direct.float()).square().sum().detach()) / max(
        float(direct.float().square().sum().detach()), 1e-30)
    return {"background_cross": cross, "contrast_self": self_term,
            "joint_response": joint, "direct_response": direct,
            "relative_squared_error": error}


@torch.no_grad()
def trajectory(model: torch.nn.Module, tokens: torch.Tensor, finals: torch.Tensor,
               sources: torch.Tensor, *, delete_cached_term: bool) -> tuple[torch.Tensor, dict, dict]:
    """Run one real prefix trajectory and optionally delete the exact R576 term."""
    x = F.rms_norm(model.transformer.wte(tokens), (r576.r573.D,))
    x0, first_value = x, None
    arange = torch.arange(tokens.size(0), device=tokens.device)
    direct_cached = r576.compiled_cached(model, tokens)
    capture = {"states": {}, "term_norm": None}
    diagnostics = {"head_source_sum_relative_squared_error": 0.0,
                   "value_split_relative_squared_error": 0.0,
                   "cached_bus_relative_squared_error": 0.0,
                   "projected_term_relative_squared_error": 0.0}
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_state = F.rms_norm(x, (r576.r573.D,))
        if site == r576.LAYER:
            write, tensors, errors = r576.r573.replay_attention(
                attention_state, first_value, block.attn, finals)
            diagnostics.update({key: max(float(diagnostics[key]), float(value))
                                for key, value in errors.items()})
            bus_error = float((direct_cached - tensors["cached"]).float().square().sum()) / max(
                float(tensors["cached"].float().square().sum()), 1e-30)
            diagnostics["cached_bus_relative_squared_error"] = max(
                diagnostics["cached_bus_relative_squared_error"], bus_error)
            term = r576.projected_terms(
                tensors, direct_cached, finals, sources, block.attn.c_proj.weight)
            head_delta = torch.zeros(tokens.size(0), r576.r573.N_HEAD, r576.r573.HEAD_D,
                                     dtype=attention_state.dtype, device=tokens.device)
            for head in r576.HEADS:
                score = tensors["pattern"][arange, head, finals, sources]
                head_delta[:, head] = score[:, None] * direct_cached[arange, sources, head]
            flat = F.linear(head_delta.reshape(tokens.size(0), r576.r573.D),
                            block.attn.c_proj.weight.to(head_delta.dtype))
            project_error = float((flat.float() - term.float()).square().sum()) / max(
                float(term.float().square().sum()), 1e-30)
            diagnostics["projected_term_relative_squared_error"] = max(
                diagnostics["projected_term_relative_squared_error"], project_error)
            capture["term_norm"] = term.float().norm(dim=-1)
            if delete_cached_term:
                write = write.clone(); write[arange, finals] -= term.to(write.dtype)
        else:
            write, first_value = block.attn(attention_state, first_value)
        x = x + write
        mlp_state = F.rms_norm(x, (r576.r573.D,))
        if site in SITES:
            capture["states"][site] = mlp_state[arange, finals].detach()
        x = x + block.mlp(mlp_state)
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (r576.r573.D,))) / 30)).float()
    if capture["term_norm"] is None:
        raise RuntimeError("R576 deletion site never executed")
    return logits[arange, finals], capture, diagnostics


@torch.no_grad()
def component_forward(model: torch.nn.Module, tokens: torch.Tensor, finals: torch.Tensor,
                      *, site_target: int, vectors: torch.Tensor) -> torch.Tensor:
    """Subtract a frozen exact response vector from one native MLP write."""
    if site_target not in SITES:
        raise ValueError(site_target)
    x = F.rms_norm(model.transformer.wte(tokens), (r576.r573.D,))
    x0, first_value = x, None
    arange = torch.arange(tokens.size(0), device=tokens.device)
    fired = False
    for site, block in enumerate(model.transformer.h):
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        write, first_value = block.attn(F.rms_norm(x, (r576.r573.D,)), first_value)
        x = x + write
        mlp_write = block.mlp(F.rms_norm(x, (r576.r573.D,)))
        if site == site_target:
            if vectors.shape != (tokens.size(0), r576.r573.D):
                raise RuntimeError("component vector batch has the wrong shape")
            mlp_write = mlp_write.clone()
            mlp_write[arange, finals] -= vectors.to(mlp_write.dtype)
            fired = True
        x = x + mlp_write
    if not fired:
        raise RuntimeError("component intervention never fired")
    logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (r576.r573.D,))) / 30)).float()
    return logits[arange, finals]


@torch.no_grad()
def capture_split(model: torch.nn.Module, rows: Sequence[dict], split: str) -> tuple[dict, list, int, dict]:
    selected = [row for row in rows if row["split"] == split]
    device = next(model.parameters()).device
    cache, raw, calls = {}, [], 0
    maxima = collections.defaultdict(float)
    replay_error = 0.0
    for batch_index, group in enumerate(chunks(selected, lambda row: len(row["ids"]))):
        tokens = torch.tensor([row["ids"] for row in group], dtype=torch.long, device=device)
        finals = torch.tensor([row["query_position"] for row in group], dtype=torch.long, device=device)
        sources = torch.tensor([row["source_position"] for row in group], dtype=torch.long, device=device)
        native, present, present_diag = trajectory(
            model, tokens, finals, sources, delete_cached_term=False)
        deleted, absent, absent_diag = trajectory(
            model, tokens, finals, sources, delete_cached_term=True)
        calls += 2
        if batch_index == 0:
            direct = r576.r573.native_logits(model, tokens)[
                torch.arange(tokens.size(0), device=device), finals].float()
            calls += 1
            replay_error = float((native - direct).square().sum()) / max(float(direct.square().sum()), 1e-30)
        for diagnostics in (present_diag, absent_diag):
            for key, value in diagnostics.items():
                maxima[key] = max(maxima[key], float(value))
        for index, row in enumerate(group):
            components = {}
            row_error = 0.0
            norms = {}
            for site in SITES:
                response = torch_bilinear_response(
                    model.transformer.h[site].mlp,
                    absent["states"][site][index], present["states"][site][index])
                row_error = max(row_error, float(response["relative_squared_error"]))
                components[site] = {name: response[name].detach().cpu() for name in COMPONENTS}
                norms[site] = {name: float(response[name].float().norm()) for name in COMPONENTS}
            row_id = row["row_id"]
            cache[row_id] = {"native_logits": native[index].detach().cpu(),
                             "deleted_logits": deleted[index].detach().cpu(),
                             "components": components, "component_norms": norms,
                             "term_norm": float(present["term_norm"][index]),
                             "response_error": row_error}
            raw.append({
                "row_id": row_id, "group_id": row["group_id"], "split": split,
                "representation": row["representation"], "source_level": row["source_level"],
                "condition": row["condition"], "action": row["action"],
                "native": endpoint_stats(native[index], row),
                "source_deleted": endpoint_stats(deleted[index], row),
                "r576_term_norm": float(present["term_norm"][index]),
                "component_norms": {str(site): values for site, values in norms.items()},
                "bilinear_response_relative_squared_error": row_error,
            })
    maxima["native_replay_relative_squared_error"] = replay_error
    maxima["bilinear_response_relative_squared_error"] = max(
        (item["response_error"] for item in cache.values()), default=float("inf"))
    return cache, raw, calls, dict(maxima)


def intervention_record(row: dict, before: torch.Tensor, after: torch.Tensor,
                        vector_norm: float, *, donor_row_id: str | None = None) -> dict:
    native, changed = endpoint_stats(before, row), endpoint_stats(after, row)
    record = {
        "row_id": row["row_id"], "group_id": row["group_id"], "split": row["split"],
        "representation": row["representation"], "source_level": row["source_level"],
        "condition": row["condition"], "action": row["action"],
        "native": native, "intervened": changed, "intervention_vector_norm": vector_norm,
        "logit_difference_squared_sum": float((after.float() - before.float()).square().sum()),
        "logit_vocabulary_count": int(after.numel()),
        "full_vocabulary_logit_rms": float((after.float() - before.float()).square().mean().sqrt()),
        "null_donor_row_id": donor_row_id,
    }
    if row["condition"] == "step_two":
        old, new = native["arithmetic_minus_structural"], changed["arithmetic_minus_structural"]
        record["preference_sign_preserved"] = bool((old >= 0) == (new >= 0))
    else:
        record["margin_damage"] = native["margin"] - changed["margin"]
        record["ce_increase"] = changed["ce"] - native["ce"]
    return record


@torch.no_grad()
def evaluate_component(model: torch.nn.Module, rows: Sequence[dict], split: str,
                       cache: Mapping[str, dict], site: int, component: str,
                       donor_map: Mapping[str, str] | None = None) -> tuple[list[dict], int]:
    selected = ([row for row in rows if row["split"] == split] if donor_map is None else
                [row for row in eligible_null_rows(rows, split) if row["row_id"] in donor_map])
    device = next(model.parameters()).device
    raw, calls = [], 0
    for group in chunks(selected, lambda row: len(row["ids"])):
        tokens = torch.tensor([row["ids"] for row in group], dtype=torch.long, device=device)
        finals = torch.tensor([row["query_position"] for row in group], dtype=torch.long, device=device)
        donor_ids = [row["row_id"] if donor_map is None else donor_map[row["row_id"]] for row in group]
        vectors = torch.stack([cache[donor_id]["components"][site][component] for donor_id in donor_ids]).to(device)
        after = component_forward(model, tokens, finals, site_target=site, vectors=vectors)
        calls += 1
        for index, (row, donor_id) in enumerate(zip(group, donor_ids, strict=True)):
            before = cache[row["row_id"]]["native_logits"].to(device)
            raw.append(intervention_record(
                row, before, after[index], float(vectors[index].float().norm()),
                donor_row_id=None if donor_map is None else donor_id))
    return raw, calls


def _bootstrap_lower(cells: Sequence[dict], key: str, cell_id: str) -> float:
    values = {str(cell["group_id"]): float(cell[key]) for cell in cells}
    if len(values) != len(cells):
        raise RuntimeError("a bootstrap cell contains repeated semantic groups")
    return float(np.quantile(r582.deterministic_group_bootstrap(values, cell_id=cell_id), .025))


def score_candidate(raw: Sequence[dict], *, cell_prefix: str,
                    frozen_scales: Mapping[str, Mapping[str, float]] | None = None) -> dict:
    lookup = {(row["group_id"], row["representation"], row["source_level"], row["condition"]): row
              for row in raw}
    target_report, copy_report, action_report = {}, {}, {}
    exact_cells = []
    scales = {} if frozen_scales is None else {key: dict(value) for key, value in frozen_scales.items()}
    all_pass = True
    for representation in ("list", "digit", "word"):
        for source in (0, 1):
            for surface, successor_condition, copy_condition in (
                ("factorial", "factorial_successor", "factorial_copy"),
                ("surface", "surface_successor", "surface_copy"),
            ):
                target = [row for row in raw if row["representation"] == representation
                          and row["source_level"] == source and row["condition"] == successor_condition]
                copy = [row for row in raw if row["representation"] == representation
                        and row["source_level"] == source and row["condition"] == copy_condition]
                if not target or not copy:
                    raise RuntimeError("candidate report lacks a frozen action cell")
                key = f"{representation}:source{source}:{surface}"
                positive = float(np.mean([row["margin_damage"] > 0 for row in target]))
                lower_margin = _bootstrap_lower(target, "margin_damage", f"{cell_prefix}:{key}:margin")
                lower_ce = _bootstrap_lower(target, "ce_increase", f"{cell_prefix}:{key}:ce")
                target_norm = float(np.median([row["intervention_vector_norm"] for row in target]))
                copy_norm = float(np.median([row["intervention_vector_norm"] for row in copy]))
                target_pass = bool(positive >= .75 and lower_margin > 0 and lower_ce > 0 and target_norm > 0)
                target_report[key] = {"n": len(target), "positive_margin_damage_fraction": positive,
                                      "bootstrap95_lower_mean_margin_damage": lower_margin,
                                      "bootstrap95_lower_mean_ce_increase": lower_ce,
                                      "median_intervention_norm": target_norm, "passed": target_pass}
                if frozen_scales is None:
                    scales[key] = {
                        "margin_damage": float(np.median([abs(row["margin_damage"]) for row in target])),
                        "logit_rms": float(np.median([row["full_vocabulary_logit_rms"] for row in target])),
                    }
                margin_scale, rms_scale = scales[key]["margin_damage"], scales[key]["logit_rms"]
                answer_fraction = float(np.mean([row["intervened"]["answer_best"] for row in copy]))
                mean_ce = float(np.mean([row["ce_increase"] for row in copy]))
                margin_fraction = (float(np.median([abs(row["margin_damage"]) for row in copy])) / margin_scale
                                   if margin_scale > 0 else float("inf"))
                rms_fraction = (float(np.median([row["full_vocabulary_logit_rms"] for row in copy])) / rms_scale
                                if rms_scale > 0 else float("inf"))
                norm_fraction = copy_norm / target_norm if target_norm > 0 else 0.0
                copy_pass = bool(norm_fraction >= .10 and answer_fraction >= .75 and mean_ce <= .10
                                 and margin_fraction <= .25 and rms_fraction <= .25)
                copy_report[key] = {"n": len(copy), "median_norm_fraction": norm_fraction,
                                    "answer_best_fraction": answer_fraction, "mean_ce_increase": mean_ce,
                                    "median_absolute_margin_fraction": margin_fraction,
                                    "median_logit_rms_fraction": rms_fraction, "passed": copy_pass}
                gaps = []
                for target_row in target:
                    copy_row = lookup[(target_row["group_id"], representation, source, copy_condition)]
                    gaps.append({"group_id": target_row["group_id"],
                                 "gap": target_row["margin_damage"] - abs(copy_row["margin_damage"])})
                gap_lower = _bootstrap_lower(gaps, "gap", f"{cell_prefix}:{key}:action_gap")
                gap_pass = bool(gap_lower > 0)
                action_report[key] = {"n": len(gaps), "mean_gap": float(np.mean([x["gap"] for x in gaps])),
                                      "bootstrap95_lower_mean_gap": gap_lower, "passed": gap_pass}
                all_pass &= target_pass and copy_pass and gap_pass
                exact_cells.extend(target + copy)
    conflict_report = {}
    activity_report = {}
    for representation in ("list", "digit", "word"):
        for source in (0, 1):
            target_norm = target_report[f"{representation}:source{source}:factorial"]["median_intervention_norm"]
            for condition in ("relation_break", "step_two"):
                activity_cells = [row for row in raw if row["representation"] == representation
                                  and row["source_level"] == source and row["condition"] == condition]
                median_norm = float(np.median([row["intervention_vector_norm"] for row in activity_cells]))
                norm_fraction = median_norm / target_norm if target_norm > 0 else 0.0
                activity_pass = bool(norm_fraction >= .10)
                activity_report[f"{representation}:source{source}:{condition}"] = {
                    "n": len(activity_cells), "median_norm_fraction_of_successor": norm_fraction,
                    "passed": activity_pass}
                all_pass &= activity_pass
            cells = [row for row in raw if row["representation"] == representation
                     and row["source_level"] == source and row["condition"] == "step_two"]
            fraction = float(np.mean([row["preference_sign_preserved"] for row in cells]))
            passed = bool(fraction >= .75)
            conflict_report[f"{representation}:source{source}"] = {
                "n": len(cells), "preference_sign_preserved_fraction": fraction, "passed": passed}
            all_pass &= passed
    stability_report = {"source_sign": {}, "surface_recovery": {}}
    for representation in ("list", "digit", "word"):
        for surface in ("factorial", "surface"):
            a = action_report[f"{representation}:source0:{surface}"]
            b = action_report[f"{representation}:source1:{surface}"]
            groups = sorted({row["group_id"] for row in raw if row["representation"] == representation})
            signs = []
            copy_condition = f"{surface}_copy"; succ_condition = f"{surface}_successor"
            for group in groups:
                gap = []
                for source in (0, 1):
                    target = lookup[(group, representation, source, succ_condition)]
                    copy = lookup[(group, representation, source, copy_condition)]
                    gap.append(target["margin_damage"] - abs(copy["margin_damage"]))
                signs.append(gap[0] * gap[1] > 0)
            fraction = float(np.mean(signs))
            passed = bool(fraction >= .75 and a["bootstrap95_lower_mean_gap"] > 0
                          and b["bootstrap95_lower_mean_gap"] > 0)
            stability_report["source_sign"][f"{representation}:{surface}"] = {
                "agreement_fraction": fraction, "passed": passed}
            all_pass &= passed
        for source in (0, 1):
            ordinary = action_report[f"{representation}:source{source}:factorial"]["mean_gap"]
            surface_item = action_report[f"{representation}:source{source}:surface"]
            ratio = surface_item["mean_gap"] / ordinary if ordinary > 0 else float("-inf")
            passed = bool(ratio >= .50 and surface_item["bootstrap95_lower_mean_gap"] > 0)
            stability_report["surface_recovery"][f"{representation}:source{source}"] = {
                "mean_gap_ratio": ratio, "passed": passed}
            all_pass &= passed
    relation_report = {}
    for representation in ("list", "digit", "word"):
        for source in (0, 1):
            coherent = [row["margin_damage"] for row in raw if row["representation"] == representation
                        and row["source_level"] == source and row["condition"] == "factorial_successor"]
            broken = [row["margin_damage"] for row in raw if row["representation"] == representation
                      and row["source_level"] == source and row["condition"] == "relation_break"]
            denominator = float(np.mean(coherent))
            relation_report[f"{representation}:source{source}"] = {
                "mean_broken_damage": float(np.mean(broken)),
                "mean_broken_to_coherent_ratio": float(np.mean(broken)) / denominator if denominator != 0 else None,
                "gate": "characterization_only",
            }
    representation_pass = {representation: all(
        item["passed"] for key, item in target_report.items() if key.startswith(representation + ":"))
        and all(item["passed"] for key, item in copy_report.items() if key.startswith(representation + ":"))
        and all(item["passed"] for key, item in action_report.items() if key.startswith(representation + ":"))
        for representation in ("list", "digit", "word")}
    return {"passed_without_nulls": bool(all_pass), "all_representations_pass": bool(all(representation_pass.values())),
            "representation_pass": representation_pass, "targets": target_report, "copies": copy_report,
            "action_gaps": action_report, "conflicts": conflict_report, "stability": stability_report,
            "active_relation_and_conflict_controls": activity_report,
            "relation_characterization": relation_report, "fit_scales": scales}


def score_null(real_raw: Sequence[dict], null_raw: Sequence[dict], *, cell_prefix: str) -> dict:
    real = score_candidate(real_raw, cell_prefix=cell_prefix + ":real")
    # Null rows omit relation/conflict cells, so compute only matched action-gap and norm comparisons here.
    real_lookup = {(row["representation"], row["source_level"], row["condition"]): [] for row in real_raw}
    null_lookup = {(row["representation"], row["source_level"], row["condition"]): [] for row in null_raw}
    for row in real_raw:
        key = (row["representation"], row["source_level"], row["condition"])
        if row["condition"] in {"factorial_copy", "factorial_successor", "surface_copy", "surface_successor"}:
            real_lookup.setdefault(key, []).append(row)
    for row in null_raw:
        key = (row["representation"], row["source_level"], row["condition"])
        null_lookup.setdefault(key, []).append(row)
    cells, passed = {}, True
    for representation in ("list", "digit", "word"):
        for source in (0, 1):
            for surface in ("factorial", "surface"):
                sc, cc = f"{surface}_successor", f"{surface}_copy"
                rt, rc = real_lookup[(representation, source, sc)], real_lookup[(representation, source, cc)]
                nt, nc = null_lookup[(representation, source, sc)], null_lookup[(representation, source, cc)]
                def gaps(target, copy):
                    by_group = {row["group_id"]: row for row in copy}
                    return [{"group_id": row["group_id"], "gap": row["margin_damage"]
                             - abs(by_group[row["group_id"]]["margin_damage"])} for row in target]
                rg, ng = gaps(rt, rc), gaps(nt, nc)
                key = f"{representation}:source{source}:{surface}"
                real_lower = _bootstrap_lower(rg, "gap", f"{cell_prefix}:{key}:real")
                null_lower = _bootstrap_lower(ng, "gap", f"{cell_prefix}:{key}:null")
                ratios = []
                real_by_row = {row["row_id"]: row for row in rt + rc}
                for row in nt + nc:
                    denominator = real_by_row[row["row_id"]]["intervention_vector_norm"]
                    ratios.append(row["intervention_vector_norm"] / denominator if denominator > 0 else float("inf"))
                median_ratio = float(np.median(ratios))
                cell_pass = bool(.8 <= median_ratio <= 1.25 and real_lower > null_lower)
                cells[key] = {"real_gap_lower95": real_lower, "null_gap_lower95": null_lower,
                              "median_null_to_real_norm_ratio": median_ratio, "passed": cell_pass}
                passed &= cell_pass
    return {"passed": bool(passed), "cells": cells,
            "real_minimum_action_gap_lower95": min(item["bootstrap95_lower_mean_gap"]
                                                    for item in real["action_gaps"].values())}


def interaction_records(raw_by_name: Mapping[str, Sequence[dict]], site: int) -> list[dict]:
    names = {component: f"mlp{site}_{component}" for component in COMPONENTS}
    maps = {component: {row["row_id"]: row for row in raw_by_name[names[component]]}
            for component in COMPONENTS}
    output = []
    for row_id in maps["joint_response"]:
        c, q, joint = (maps[name][row_id] for name in COMPONENTS)
        if c["condition"] == "step_two":
            continue
        native = c["native"]["margin"]
        terms = r582.two_factor_mobius(native, c["intervened"]["margin"],
                                       q["intervened"]["margin"], joint["intervened"]["margin"])
        output.append({"row_id": row_id, "group_id": c["group_id"], **terms})
    return output


def dryrun(rows: Sequence[dict]) -> dict:
    pricing = price(rows)
    return {"status": "dryrun_passed", "rung": 584, "rows": len(rows),
            "fit_rows": sum(row["split"] == "FIT" for row in rows),
            "select_rows": sum(row["split"] == "SELECT" for row in rows),
            **pricing, "selection_order": [f"mlp{site}_{component}" for site, component in SELECTION],
            "model_loaded": False, "model_forwards": 0, "model_backwards": 0,
            "model_weights_updated": False, "opened_splits": [],
            "FINAL_TEST_or_OOD_opened": False,
            "input_sha256": {str(path): sha256(path) for path in HASHES}}


def main() -> None:
    started = time.time()
    rows = load_authority()
    plan = dryrun(rows)
    if plan["literal_executable_maximum_forwards"] > plan["r582_conservative_maximum_forwards"]:
        raise RuntimeError("executable implementation exceeds the frozen R582 ceiling")
    if os.environ.get("BQLIB_DRYRUN") == "1":
        DRYRUN_OUT.write_text(json.dumps(plan, indent=1) + "\n")
        print(json.dumps(plan, indent=2))
        return
    if OUT.exists():
        raise RuntimeError("R584 result namespace already exists")
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    fit_cache, fit_capture_raw, fit_calls, fit_exact = capture_split(model, rows, "FIT")
    fit_raw, fit_reports = {}, {}
    for site, component in SELECTION:
        name = f"mlp{site}_{component}"
        fit_raw[name], calls = evaluate_component(model, rows, "FIT", fit_cache, site, component)
        fit_calls += calls
        fit_reports[name] = score_candidate(fit_raw[name], cell_prefix=f"FIT:{name}")
    fit_exact_pass = bool(max(fit_exact.values(), default=float("inf")) <= EXACT_BAR
                          and all(item["r576_term_norm"] > 0 for item in fit_capture_raw))
    provisional = (next((name for name in [f"mlp{site}_{component}" for site, component in SELECTION]
                         if fit_reports[name]["passed_without_nulls"]), None)
                   if fit_exact_pass else None)
    selected = None
    fit_null_raw = fit_null_reports = None
    if provisional is not None:
        site_string, component = provisional[3:].split("_", 1)
        site = int(site_string)
        null_maps = r582.deterministic_null_maps(rows, "FIT")
        fit_null_raw, fit_null_reports = {}, {}
        for null_name in NULLS:
            key = f"{provisional}:null:{null_name}"
            fit_null_raw[key], calls = evaluate_component(
                model, rows, "FIT", fit_cache, site, component, null_maps[null_name])
            fit_calls += calls
            fit_null_reports[key] = score_null(
                fit_raw[provisional], fit_null_raw[key], cell_prefix=f"FIT:{key}")
        if all(report["passed"] for report in fit_null_reports.values()):
            selected = provisional
    opened = ["FIT"]
    select_cache = select_capture_raw = select_exact = select_raw = select_reports = None
    select_null_raw = select_null_reports = None
    select_calls = 0
    if selected is not None:
        select_cache, select_capture_raw, select_calls, select_exact = capture_split(model, rows, "SELECT")
        site_text, selected_component = selected[3:].split("_", 1)
        selected_site = int(site_text)
        select_raw, select_reports = {}, {}
        for component in COMPONENTS:
            name = f"mlp{selected_site}_{component}"
            select_raw[name], calls = evaluate_component(
                model, rows, "SELECT", select_cache, selected_site, component)
            select_calls += calls
            select_reports[name] = score_candidate(
                select_raw[name], cell_prefix=f"SELECT:{name}",
                frozen_scales=fit_reports[selected]["fit_scales"])
        null_maps = r582.deterministic_null_maps(rows, "SELECT")
        select_null_raw, select_null_reports = {}, {}
        for null_name in NULLS:
            key = f"{selected}:null:{null_name}"
            select_null_raw[key], calls = evaluate_component(
                model, rows, "SELECT", select_cache, selected_site, selected_component,
                null_maps[null_name])
            select_calls += calls
            select_null_reports[key] = score_null(
                select_raw[selected], select_null_raw[key], cell_prefix=f"SELECT:{key}")
        opened.append("SELECT")
    total_calls = fit_calls + select_calls
    exact = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                 and max(fit_exact.values(), default=float("inf")) <= EXACT_BAR
                 and all(item["r576_term_norm"] > 0 for item in fit_capture_raw)
                 and (select_exact is None or (max(select_exact.values(), default=float("inf")) <= EXACT_BAR
                                               and all(item["r576_term_norm"] > 0
                                                       for item in select_capture_raw))))
    fit_selected_pass = bool(selected is not None)
    select_selected_pass = bool(selected is not None and select_reports[selected]["passed_without_nulls"]
                                and all(report["passed"] for report in select_null_reports.values()))
    reuse = bool(fit_selected_pass and fit_reports[selected]["all_representations_pass"]
                 and select_selected_pass and select_reports[selected]["all_representations_pass"])
    all_pass = bool(exact and fit_selected_pass and select_selected_pass and reuse)
    interactions = {"fit": None, "select": None}
    if selected is not None:
        selected_site = int(selected[3:].split("_", 1)[0])
        interactions["fit"] = interaction_records(fit_raw, selected_site)
        interactions["select"] = interaction_records(select_raw, selected_site)
    result = {
        "rung": 584, "stage": "cached_value_downstream_bilinear_use",
        "pred_a_exact_prefix_and_bilinear_decomposition": exact,
        "pred_b_selective_downstream_action_component": bool(fit_selected_pass and select_selected_pass),
        "pred_c_cross_representation_reuse": reuse,
        "all_required_gates_pass": all_pass,
        "provisional_fit_selection": provisional, "selected_component": selected,
        "fit_exactness": fit_exact, "select_exactness": select_exact,
        "fit_capture_raw": fit_capture_raw, "select_capture_raw": select_capture_raw,
        "fit_raw": fit_raw, "fit_reports": fit_reports,
        "fit_null_raw": fit_null_raw, "fit_null_reports": fit_null_reports,
        "select_raw": select_raw, "select_reports": select_reports,
        "select_null_raw": select_null_raw, "select_null_reports": select_null_reports,
        "component_interactions": interactions,
        "execution": plan, "model_forwards": total_calls, "model_backwards": 0,
        "model_weights_updated": False, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "evaluated_splits": opened, "forbidden_splits_opened": [],
        "input_sha256": {str(path): sha256(path) for path in HASHES},
        "elapsed_seconds": time.time() - started,
        "decision": "downstream_use_component_held" if all_pass else "downstream_use_decomposition_null",
        "next_step": ("independent_cpu_audit_then_FINAL_TEST_remains_separately_preregistered" if all_pass else
                      "retain_R576_broad_carrier_and_do_not_promote_R582_component"),
    }
    if total_calls > plan["literal_executable_maximum_forwards"]:
        raise RuntimeError(f"literal forward count exceeded: {total_calls} > {plan['literal_executable_maximum_forwards']}")
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: result[key] for key in result if key.startswith("pred_") or key in {
        "all_required_gates_pass", "selected_component", "model_forwards", "evaluated_splits",
        "decision", "next_step"}}, indent=2))


if __name__ == "__main__":
    main()
