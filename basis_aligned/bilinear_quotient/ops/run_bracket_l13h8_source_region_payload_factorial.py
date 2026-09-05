#!/usr/bin/env python3
"""Exact L13H8 payload-region factorial. Dry-run is CPU-only; science is explicit."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_source_regions as candidate


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_source_region_payload_factorial_v1_result.json"
DIRECTIONS = ("base_to_donor", "donor_to_base")


def corner_name(corner: tuple[str, ...]) -> str:
    return "+".join(corner) if corner else "NONE"


def region_masks(rows: list[dict], endpoint: str, length: int, torch, device) -> dict:
    masks = {}
    for region in candidate.REGIONS:
        mask = torch.zeros((len(rows), length), dtype=torch.bool, device=device)
        for index, row in enumerate(rows):
            # Equal aligned lengths and opener positions make one frozen partition valid at both endpoints.
            assert len(row["base_ids"]) == len(row["donor_ids"])
            for position in row["regions"][region]:
                mask[index, position] = True
        masks[region] = mask
    return masks


def _dependencies():
    import torch
    import torch.nn.functional as F
    os.environ.setdefault("BQLIB_NO_MODEL", "1")
    poly = ROOT.parent / "polynomial_causal"
    for path in (ROOT, ROOT / "ops", poly):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import bilin18_observed_model_facade as facade
    return torch, F, facade


def native_logits(model, tokens, torch, F):
    x = model.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    logits = model.lm_head(F.rms_norm(x, (x.size(-1),)))
    return (30.0 * torch.tanh(logits / 30.0)).float()


def _linear(value, weight, F):
    return F.linear(value, weight.to(device=value.device, dtype=value.dtype))


def payload_hybrid(pattern, recipient_payload, donor_payload, selected, torch):
    """Retain recipient attention scores and swap only selected projected values."""
    payload = torch.where(selected.unsqueeze(-1), donor_payload, recipient_payload)
    return torch.einsum("bk,bkd->bd", pattern, payload)


def replay_head(state, first_value, attention, finals, torch, F):
    batch, length, width = state.shape
    heads, head_d = 9, width // 9
    q = _linear(state, attention.c_q.weight, F).view(batch, length, heads, head_d)
    k = _linear(state, attention.c_k.weight, F).view(batch, length, heads, head_d)
    q2 = _linear(state, attention.c_q2.weight, F).view(batch, length, heads, head_d)
    k2 = _linear(state, attention.c_k2.weight, F).view(batch, length, heads, head_d)
    raw = _linear(state, attention.c_v.weight, F).view(batch, length, heads, head_d)
    value = (1 - attention.lamb) * raw + attention.lamb * first_value.view_as(raw)
    cos, sin = attention.rotary(q)
    rotary = sys.modules[type(attention).__module__].apply_rotary_emb
    q = rotary(F.rms_norm(q, (head_d,)), cos, sin)
    k = rotary(F.rms_norm(k, (head_d,)), cos, sin)
    q2 = rotary(F.rms_norm(q2, (head_d,)), cos, sin)
    k2 = rotary(F.rms_norm(k2, (head_d,)), cos, sin)
    pattern = torch.einsum("bqhd,bkhd->bhqk", q, k) / head_d
    pattern *= torch.einsum("bqhd,bkhd->bhqk", q2, k2) / head_d
    pattern = pattern.masked_fill(~torch.tril(torch.ones(length, length, dtype=torch.bool, device=state.device)), 0)
    all_heads = torch.einsum("bhqk,bkhd->bhqd", pattern, value)
    write = _linear(all_heads.transpose(1, 2).contiguous().view(batch, length, width), attention.c_proj.weight, F)
    arange = torch.arange(batch, device=state.device)
    p = pattern[arange, candidate.PATCH_HEAD, finals]
    weight = attention.c_proj.weight[:, candidate.PATCH_HEAD * head_d:(candidate.PATCH_HEAD + 1) * head_d]
    u = _linear(value[:, :, candidate.PATCH_HEAD].float(), weight.float(), F)
    native_head = torch.einsum("bk,bkd->bd", p.float(), u)
    return write, {"p": p.float(), "u": u, "head": native_head}


def factor_forward(model, tokens, finals, masks, torch, F, facade, *, donor=None, corner=None,
                   complete=False, replacement_terms=None, source_positions=None):
    if replacement_terms is not None and (donor is not None or source_positions is None):
        raise ValueError("exact term replacement needs source_positions and excludes donor mode")
    captured = {}
    def attention(event):
        if event.site != candidate.PATCH_LAYER:
            return event.block.attn(event.state, event.first_value)
        write, factors = replay_head(event.state, event.first_value, event.block.attn, finals, torch, F)
        captured.update({key: value.detach().clone() for key, value in factors.items()})
        if replacement_terms is not None:
            arange = torch.arange(tokens.size(0), device=tokens.device)
            native_terms = factors["p"][arange, source_positions].unsqueeze(-1) \
                * factors["u"][arange, source_positions]
            write[arange, finals] += (replacement_terms - native_terms).to(write.dtype)
        elif donor is not None:
            arange = torch.arange(tokens.size(0), device=tokens.device)
            if complete:
                hybrid = donor["head"]
            else:
                selected = torch.zeros_like(factors["p"], dtype=torch.bool)
                for region in corner or ():
                    selected |= masks[region]
                hybrid = payload_hybrid(factors["p"], factors["u"], donor["u"], selected, torch)
            write[arange, finals] += (hybrid - factors["head"]).to(write.dtype)
        return write, event.first_value
    logits = facade.forward_with_dispatch(
        model, tokens, attention, lambda event: event.block.mlp(event.state), require_production=False,
    ).float()
    assert set(captured) == {"p", "u", "head"}
    return logits, captured


def pad(rows, endpoint, torch, device):
    length = max(len(row[f"{endpoint}_ids"]) for row in rows)
    tokens = torch.full((len(rows), length), 50256, dtype=torch.long, device=device)
    finals = []
    for index, row in enumerate(rows):
        ids = row[f"{endpoint}_ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, device=device)
        finals.append(len(ids) - 1)
    return tokens, torch.tensor(finals, device=device)


def endpoint_change(before, after, row, direction) -> float:
    positive, negative = ((row["donor_answer_id"], row["base_answer_id"])
                          if direction == "base_to_donor" else
                          (row["base_answer_id"], row["donor_answer_id"]))
    return float((after[positive] - after[negative]) - (before[positive] - before[negative]))


def closer_margin(logits, answer) -> float:
    closers = (8, 60, 1)
    return float(logits[answer] - sum(logits[item] for item in closers if item != answer) / 2)


def summarize(raw: list[dict]) -> dict:
    by_cell = defaultdict(list)
    for cell in raw:
        by_cell[(cell["family_id"], cell["direction"], cell["condition"])].append(cell)
    output = {}
    for key, cells in sorted(by_cell.items()):
        family, direction, condition = key
        values = [cell["normalized_effect"] for cell in cells]
        output["|".join(key)] = {"n": len(values), "mean": sum(values) / len(values),
                                  "positive_fraction": sum(value > 0 for value in values) / len(values)}
    return output


def mobius_interactions(raw: list[dict]) -> dict:
    """Exact Boolean-lattice coefficients of each row's normalized payload response."""
    corner_sets = {"payload_" + corner_name(corner): frozenset(corner) for corner in candidate.CORNERS}
    grouped = defaultdict(dict)
    metadata = {}
    for cell in raw:
        if cell["condition"] not in corner_sets:
            continue
        key = (cell["row_id"], cell["direction"])
        grouped[key][corner_sets[cell["condition"]]] = cell["normalized_effect"]
        metadata[key] = (cell["family_id"], cell["group_id"])
    records = []
    for key, values in grouped.items():
        assert len(values) == 8
        family, group = metadata[key]
        coefficients = {}
        for subset in values:
            coefficient = 0.0
            for inner, value in values.items():
                if inner <= subset:
                    coefficient += (-1.0 if (len(subset) - len(inner)) % 2 else 1.0) * value
            coefficients[corner_name(tuple(region for region in candidate.REGIONS if region in subset))] = coefficient
        records.append({"row_id": key[0], "direction": key[1], "family_id": family,
                        "group_id": group, "coefficients": coefficients})
    return records


def score_basic_screen(raw: list[dict], replay_error: float, native_capability: list[dict]) -> dict:
    bars = candidate.compile_plan()["bars"]
    def cells(role, condition):
        return [cell for cell in raw if cell["role"] == role and cell["condition"] == condition]
    complete = cells("target", "complete_head")
    open_post = cells("target", "payload_OPEN+POST")
    prefix = cells("target", "payload_PREFIX")
    all_payload = cells("target", "payload_PREFIX+OPEN+POST")
    controls = cells("control", "payload_OPEN+POST")
    complete_positive = sum(cell["effect"] > 0 for cell in complete) / len(complete)
    open_values = [cell["normalized_effect"] for cell in open_post]
    prefix_abs = [abs(cell["normalized_effect"]) for cell in prefix]
    all_values = [cell["normalized_effect"] for cell in all_payload]
    control_abs = [abs(cell["effect"]) for cell in controls]
    control_ratio = [abs(cell["normalized_effect"]) for cell in controls]
    native_by_cell = defaultdict(list)
    for cell in native_capability:
        native_by_cell[(cell["family_id"], cell["direction"])].append(cell["answer_margin"])
    native_positive_by_cell = {
        "|".join(key): sum(value > 0 for value in values) / len(values)
        for key, values in sorted(native_by_cell.items())
    }
    minimum_native_positive = min(native_positive_by_cell.values())
    checks = {
        "native_capability": minimum_native_positive >= bars["native_answer_positive_fraction_min"],
        "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
        "complete_head_ceiling": complete_positive >= bars["complete_head_target_positive_fraction_min"],
        "open_post_target": (statistics.median(open_values) >= bars["open_post_target_median_recovery_min"]
                             and sum(value > 0 for value in open_values) / len(open_values)
                             >= bars["open_post_target_positive_fraction_min"]),
        "prefix_localization": sum(prefix_abs) / len(prefix_abs) <= bars["prefix_target_mean_absolute_recovery_max"],
        "same_state_controls": (sum(control_abs) / len(control_abs)
                                <= bars["control_mean_absolute_closer_margin_change_max"]
                                and sum(control_ratio) / len(control_ratio)
                                <= bars["control_mean_absolute_fraction_of_complete_head_max"]),
    }
    pred_a = checks["native_capability"] and checks["native_replay"] and checks["complete_head_ceiling"]
    pred_b = pred_a and checks["open_post_target"] and checks["prefix_localization"] \
        and checks["same_state_controls"]
    pred_c = pred_a and not pred_b and (
        sum(prefix_abs) / len(prefix_abs) >= 0.50
        or (statistics.median(all_values) >= 0.50 and statistics.median(open_values) < 0.50)
    )
    return {"checks": checks, "passed": all(checks.values()),
            "predictions": {"pred_a_instrument_live": pred_a,
                            "pred_b_open_post_localized": pred_b,
                            "pred_c_broad_or_distributed": pred_c},
            "native_positive_fraction_by_family_direction": native_positive_by_cell,
            "minimum_native_positive_fraction": minimum_native_positive,
            "complete_head_target_positive_fraction": complete_positive,
            "open_post_target_median_recovery": statistics.median(open_values),
            "prefix_target_mean_absolute_recovery": sum(prefix_abs) / len(prefix_abs),
            "control_mean_absolute_closer_margin_change": sum(control_abs) / len(control_abs)}


def evaluate(model, torch, F, facade):
    rows = candidate.ROWS
    device = next(model.parameters()).device
    base_tokens, base_finals = pad(rows, "base", torch, device)
    donor_tokens, donor_finals = pad(rows, "donor", torch, device)
    masks = region_masks(rows, "base", base_tokens.size(1), torch, device)
    native_base, native_donor = native_logits(model, base_tokens, torch, F), native_logits(model, donor_tokens, torch, F)
    replay_base, base = factor_forward(model, base_tokens, base_finals, masks, torch, F, facade)
    replay_donor, donor = factor_forward(model, donor_tokens, donor_finals, masks, torch, F, facade)
    replay_error = max(float((native_base - replay_base).abs().max()), float((native_donor - replay_donor).abs().max()))
    conditions = {}
    conditions["complete_head"] = (
        factor_forward(model, base_tokens, base_finals, masks, torch, F, facade, donor=donor, complete=True)[0],
        factor_forward(model, donor_tokens, donor_finals, masks, torch, F, facade, donor=base, complete=True)[0],
    )
    for corner in candidate.CORNERS:
        name = "payload_" + corner_name(corner)
        conditions[name] = (
            factor_forward(model, base_tokens, base_finals, masks, torch, F, facade, donor=donor, corner=corner)[0],
            factor_forward(model, donor_tokens, donor_finals, masks, torch, F, facade, donor=base, corner=corner)[0],
        )
    raw = []
    native_capability = []
    for index, row in enumerate(rows):
        for direction, logits, finals, endpoint in (
            ("base_to_donor", native_base, base_finals, "base"),
            ("donor_to_base", native_donor, donor_finals, "donor"),
        ):
            q = int(finals[index])
            answer = row[f"{endpoint}_answer_id"]
            native_capability.append({
                "row_id": row["row_id"], "group_id": row["group_id"],
                "family_id": row["family_id"], "role": row["role"], "direction": direction,
                "answer_margin": closer_margin(logits[index, q], answer),
            })
        full = {}
        for direction, before, after, finals in (
            ("base_to_donor", replay_base, conditions["complete_head"][0], base_finals),
            ("donor_to_base", replay_donor, conditions["complete_head"][1], donor_finals),
        ):
            q = int(finals[index])
            if row["role"] == "target":
                full[direction] = endpoint_change(before[index, q], after[index, q], row, direction)
            else:
                answer = row["base_answer_id"] if direction == "base_to_donor" else row["donor_answer_id"]
                full[direction] = closer_margin(after[index, q], answer) - closer_margin(before[index, q], answer)
        for condition, (base_after, donor_after) in conditions.items():
            for direction, before, after, finals in (
                ("base_to_donor", replay_base, base_after, base_finals),
                ("donor_to_base", replay_donor, donor_after, donor_finals),
            ):
                q = int(finals[index])
                if row["role"] == "target":
                    effect = endpoint_change(before[index, q], after[index, q], row, direction)
                else:
                    answer = row["base_answer_id"] if direction == "base_to_donor" else row["donor_answer_id"]
                    effect = closer_margin(after[index, q], answer) - closer_margin(before[index, q], answer)
                denominator = full[direction]
                raw.append({"row_id": row["row_id"], "group_id": row["group_id"],
                            "family_id": row["family_id"], "role": row["role"], "direction": direction,
                            "condition": condition, "effect": effect, "complete_head_effect": denominator,
                            "normalized_effect": effect / denominator if abs(denominator) > 1e-9 else 0.0})
    return raw, replay_error, native_capability


def main() -> None:
    plan = candidate.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = _dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        raw, replay_error, native_capability = evaluate(model, torch, F, facade)
    result = {"schema": "bracket_source_region_payload_result_v1", "plan": plan,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "native_replay_max_absolute_logit_error": replay_error,
              "native_capability": native_capability,
              "raw": raw, "summary": summarize(raw), "mobius_interactions": mobius_interactions(raw),
              "screen": score_basic_screen(raw, replay_error, native_capability),
              "evaluated_splits": ["BASIC_SCREEN"],
              "forbidden_splits_opened": [], "model_forwards": plan["price"]["model_forwards"]}
    predictions = result["screen"]["predictions"]
    if not predictions["pred_a_instrument_live"]:
        result.update(terminal="invalid", reason="native_capability_replay_or_ceiling_failed")
    elif predictions["pred_b_open_post_localized"]:
        result.update(terminal="screen", reason="open_post_payload_localized")
    else:
        result.update(terminal="null", reason="open_post_payload_not_localized")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "replay_error": replay_error, "model_forwards": result["model_forwards"]}, indent=2))


if __name__ == "__main__":
    main()
