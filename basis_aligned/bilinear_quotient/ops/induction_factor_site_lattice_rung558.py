#!/usr/bin/env python3
"""R558: causal selector-score versus payload-value subset lattice.

Conditional on held R554/R555 capability, exhaust the 2^4 lattice of the four
registered equality terms on FIT, select the smallest causally sufficient subset,
and evaluate that subset plus the full set once on SELECT.
"""

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
import equality_term_score_payload_rung459 as factor_lib  # noqa: E402


ROWS = ROOT / "induction_selector_payload_factorial_rows_rung552.json"
ROW_AUDIT = ROOT / "induction_selector_payload_factorial_rows_rung553_audit.json"
R554 = ROOT / "induction_selector_payload_capability_rung554_results.json"
R555 = ROOT / "induction_selector_payload_capability_rung555_audit.json"
PREREG = POLY / "INDUCTION_FACTOR_SITE_LATTICE_RUNG558_PREREGISTRATION.md"
OUT = ROOT / "induction_factor_site_lattice_rung558_results.json"
STATIC_HASHES = {
    ROWS: "6a0a6d2c8a3891ae5d6f787527b35e71c17518548b3b1836042afe730b13c460",
    ROW_AUDIT: "9fc0376fade6fb204686e164f293f8991caf7bc45c67eedd064f330dffd5d1ea",
    PREREG: "9f44aa1adb04470e283b47f3d5d59a7debaac33ceedf31c6bc2b5e4014815b33",
    ROOT / "ops/equality_term_score_payload_rung459.py":
        "9f9e66f689452cbcb14d741792b66eb9ff526dff5472a5938c58a2a4c82620d8",
}
TERMS = factor_lib.TERMS
TERM_NAMES = tuple(term[0] for term in TERMS)
SITE_HEADS = factor_lib.stage1.SITE_HEADS
FAMILIES = {
    "selector": "two_valid_sources_selector_swap",
    "payload": "payload_swap_match_preserved",
    "joint": "selector_payload_joint_answer_preserved",
    "matchbreak": "match_break_payload_preserved",
    "irrelevant": "irrelevant_source_edit",
}
ARMS = {
    "selector": ("score", "payload"),
    "payload": ("payload", "score"),
    "joint": ("score", "payload", "joint"),
    "matchbreak": ("score", "payload"),
    "irrelevant": ("score",),
}
FIT_MASKS = tuple(range(1, 16))
BATCH = 18
BOOTSTRAPS = 2000
SEED = 558


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_static_inputs() -> dict:
    for path, expected in STATIC_HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    rows = json.loads(ROWS.read_text())
    audit = json.loads(ROW_AUDIT.read_text())
    assert rows["model_loaded"] is False and rows["outcomes_opened"] == []
    assert audit["all_token_level_factorial_checks_pass"] is True
    assert tuple(name for name, _, _ in TERMS) == TERM_NAMES
    return rows


def validate_dependency() -> tuple[dict, dict]:
    if not R554.is_file() or not R555.is_file():
        raise RuntimeError("R558 is conditional: R554 result and R555 audit must exist")
    capability = json.loads(R554.read_text())
    audit = json.loads(R555.read_text())
    if capability.get("all_gates_pass") is not True or audit.get("all_gates_pass") is not True:
        raise RuntimeError("R558 is forbidden because R554/R555 capability did not hold")
    if audit.get("result_sha256") != sha256(R554):
        raise RuntimeError("R555 does not bind the present R554 result")
    return capability, audit


def native_logits(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    x = model.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    logits = model.lm_head(F.rms_norm(x, (x.size(-1),)))
    return (30.0 * torch.tanh(logits / 30.0)).float()


@torch.no_grad()
def factor_forward(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    donor: dict[int, dict[str, torch.Tensor]] | None = None,
    mask: int = 0,
    arm: str = "replay",
    capture: bool = False,
) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]], float]:
    if arm not in {"replay", "score", "payload", "joint"}:
        raise ValueError(arm)
    if mask and donor is None:
        raise ValueError("intervention requires donor factors")
    captured: dict[int, dict[str, torch.Tensor]] = {}
    reconstruction = 0.0

    def attention(event):
        nonlocal reconstruction
        if event.site not in SITE_HEADS:
            return event.block.attn(event.state, event.first_value)
        write, factors, support, error = factor_lib._factor_site(
            event.state, event.first_value, event.block.attn, event.site, event.tokens,
        )
        reconstruction = max(reconstruction, error)
        for index, factor in factors.items():
            edge = factor["p"] * support
            if capture:
                captured[index] = {"edge": edge.detach().clone(), "u": factor["u"].detach().clone()}
            if mask & (1 << index):
                assert donor is not None and index in donor
                write = write - factor["native_term"]
                selected_edge = donor[index]["edge"] if arm in {"score", "joint"} else edge
                selected_u = donor[index]["u"] if arm in {"payload", "joint"} else factor["u"]
                hybrid = torch.bmm(selected_edge.float(), selected_u.float()).to(write.dtype)
                write = write + hybrid
        return write, event.first_value

    def mlp(event):
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=False,
    ).float()
    if capture and set(captured) != set(range(4)):
        raise RuntimeError(f"factor capture changed: {sorted(captured)}")
    return logits, captured, reconstruction


def pad(rows: list[dict], endpoint: str, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    sequences = [row[f"{endpoint}_ids"] for row in rows]
    length = max(map(len, sequences))
    tokens = torch.full((len(rows), length), 50256, dtype=torch.long, device=device)
    finals = []
    for index, ids in enumerate(sequences):
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        finals.append(len(ids) - 1)
    return tokens, finals


def other_payload(row: dict, groups: dict[str, dict]) -> int:
    group = groups[row["group_id"]]
    choices = (group["variable_token_ids"]["B"], group["variable_token_ids"]["D"])
    answer = row["base_answer_id"]
    assert answer in choices
    return choices[1] if answer == choices[0] else choices[0]


def row_margins(logits: torch.Tensor, finals: list[int], rows: list[dict], groups: dict[str, dict]) -> list[float]:
    values = []
    for index, (final, row) in enumerate(zip(finals, rows, strict=True)):
        if row["base_answer_id"] != row["donor_answer_id"]:
            positive, negative = row["donor_answer_id"], row["base_answer_id"]
        else:
            positive, negative = row["base_answer_id"], other_payload(row, groups)
        values.append(float(logits[index, final, positive] - logits[index, final, negative]))
    return values


def grouped_rows(document: dict, split: str, family_id: str) -> list[dict]:
    rows = [row for row in document["rows"] if row["split"] == split and row["family_id"] == family_id]
    assert rows and all(len(row["base_ids"]) == len(row["donor_ids"]) for row in rows)
    return sorted(rows, key=lambda row: (len(row["base_ids"]), row["group_id"], row["row_id"]))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    document: dict,
    split: str,
    masks: tuple[int, ...],
) -> tuple[dict, dict]:
    groups = {group["group_id"]: group for group in document["groups"]}
    raw: dict[str, dict[str, list[dict]]] = {}
    calls = 0
    max_reconstruction = 0.0
    max_replay_error = 0.0
    device = next(model.parameters()).device
    for short_name, family_id in FAMILIES.items():
        family_rows = grouped_rows(document, split, family_id)
        raw[short_name] = defaultdict(list)
        cursor = 0
        while cursor < len(family_rows):
            length = len(family_rows[cursor]["base_ids"])
            same_length = []
            while cursor < len(family_rows) and len(family_rows[cursor]["base_ids"]) == length \
                    and len(same_length) < BATCH:
                same_length.append(family_rows[cursor])
                cursor += 1
            base_tokens, base_finals = pad(same_length, "base", device)
            donor_tokens, donor_finals = pad(same_length, "donor", device)
            base_logits, _, error = factor_forward(model, base_tokens)
            donor_logits, donor_factors, donor_error = factor_forward(model, donor_tokens, capture=True)
            calls += 2
            max_reconstruction = max(max_reconstruction, error, donor_error)
            if calls == 2:
                native = native_logits(model, base_tokens)
                calls += 1
                relative = float((native - base_logits).square().sum()) / max(float(native.square().sum()), 1e-30)
                max_replay_error = max(max_replay_error, relative)
            base_margin = row_margins(base_logits, base_finals, same_length, groups)
            donor_margin = row_margins(donor_logits, donor_finals, same_length, groups)
            for mask in masks:
                for arm in ARMS[short_name]:
                    logits, _, intervention_error = factor_forward(
                        model, base_tokens, donor=donor_factors, mask=mask, arm=arm,
                    )
                    calls += 1
                    max_reconstruction = max(max_reconstruction, intervention_error)
                    arm_margin = row_margins(logits, base_finals, same_length, groups)
                    key = f"{mask:04b}:{arm}"
                    for row, base, donor_value, value in zip(
                        same_length, base_margin, donor_margin, arm_margin, strict=True,
                    ):
                        raw[short_name][key].append({
                            "group_id": row["group_id"],
                            "row_id": row["row_id"],
                            "base": base,
                            "donor": donor_value,
                            "arm": value,
                        })
            del base_logits, donor_logits, donor_factors
    return {family: dict(values) for family, values in raw.items()}, {
        "model_forwards": calls,
        "max_factor_relative_squared_reconstruction_error": max_reconstruction,
        "native_replay_relative_squared_error": max_replay_error,
    }


def group_means(rows: list[dict]) -> dict[str, dict[str, float]]:
    accum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for key in ("base", "donor", "arm"):
            accum[row["group_id"]][key].append(row[key])
    return {
        group: {key: float(np.mean(values)) for key, values in fields.items()}
        for group, fields in accum.items()
    }


def ratio_report(numerators: list[float], denominators: list[float], seed: int) -> dict:
    numerator = np.asarray(numerators, dtype=np.float64)
    denominator = np.asarray(denominators, dtype=np.float64)
    point_denom = float(denominator.mean())
    point = float(numerator.mean() / point_denom) if point_denom > 1e-12 else None
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(numerator), size=(BOOTSTRAPS, len(numerator)))
    num_boot = numerator[choices].mean(1)
    den_boot = denominator[choices].mean(1)
    ratios = np.full(BOOTSTRAPS, -np.inf)
    valid = den_boot > 1e-12
    ratios[valid] = num_boot[valid] / den_boot[valid]
    return {
        "n_groups": len(numerator),
        "mean_numerator": float(numerator.mean()),
        "mean_native_denominator": point_denom,
        "recovery": point,
        "bootstrap95_lower_recovery": float(np.quantile(ratios, .025)),
        "bootstrap_draws": BOOTSTRAPS,
    }


def direct_report(raw: dict, family: str, mask: int, arm: str, seed: int, *, reverse: bool = False) -> dict:
    rows = group_means(raw[family][f"{mask:04b}:{arm}"])
    if reverse:
        numerator = [cell["base"] - cell["arm"] for cell in rows.values()]
        denominator = [cell["base"] - cell["donor"] for cell in rows.values()]
    else:
        numerator = [cell["arm"] - cell["base"] for cell in rows.values()]
        denominator = [cell["donor"] - cell["base"] for cell in rows.values()]
    return ratio_report(numerator, denominator, seed)


def joint_report(raw: dict, mask: int, seed: int) -> dict:
    cells = {
        arm: group_means(raw["joint"][f"{mask:04b}:{arm}"])
        for arm in ("score", "payload", "joint")
    }
    groups = sorted(cells["joint"])
    numerator, denominator = [], []
    for group in groups:
        base = cells["joint"][group]["base"]
        single = min(cells["score"][group]["arm"], cells["payload"][group]["arm"])
        numerator.append(cells["joint"][group]["arm"] - single)
        denominator.append(base - single)
    return ratio_report(numerator, denominator, seed)


def reports(raw: dict, masks: tuple[int, ...]) -> dict[str, dict]:
    output = {}
    seed = SEED
    for mask in masks:
        target = {
            "selector_score": direct_report(raw, "selector", mask, "score", seed),
            "payload_value": direct_report(raw, "payload", mask, "payload", seed + 1),
            "matchbreak_score": direct_report(raw, "matchbreak", mask, "score", seed + 2, reverse=True),
            "joint_restoration": joint_report(raw, mask, seed + 3),
        }
        controls = {
            "selector_payload_cross": direct_report(raw, "selector", mask, "payload", seed + 4)["recovery"],
            "payload_score_cross": direct_report(raw, "payload", mask, "score", seed + 5)["recovery"],
            "matchbreak_payload_cross": direct_report(raw, "matchbreak", mask, "payload", seed + 6, reverse=True)["recovery"],
        }
        irrelevant = group_means(raw["irrelevant"][f"{mask:04b}:score"])
        matchbreak = group_means(raw["matchbreak"][f"{mask:04b}:score"])
        irrelevant_effect = float(np.mean([abs(cell["arm"] - cell["base"]) for cell in irrelevant.values()]))
        native_matchbreak = float(np.mean([cell["base"] - cell["donor"] for cell in matchbreak.values()]))
        controls["irrelevant_to_native_matchbreak_ratio"] = (
            irrelevant_effect / native_matchbreak if native_matchbreak > 1e-12 else None
        )
        output[f"{mask:04b}"] = {"mask": mask, "channels": [TERM_NAMES[i] for i in range(4) if mask & (1 << i)],
                                  "target": target, "controls": controls}
        seed += 11
    return output


def controls_pass(cell: dict) -> bool:
    values = cell["controls"]
    return bool(
        all(values[name] is not None and abs(values[name]) <= .25 for name in (
            "selector_payload_cross", "payload_score_cross", "matchbreak_payload_cross",
        ))
        and values["irrelevant_to_native_matchbreak_ratio"] is not None
        and values["irrelevant_to_native_matchbreak_ratio"] <= .25
    )


def target_pass(cell: dict, point_bar: float, lower_bar: float) -> bool:
    return all(
        report["recovery"] is not None and report["recovery"] >= point_bar
        and report["bootstrap95_lower_recovery"] > lower_bar
        for report in cell["target"].values()
    )


def choose_fit(fit: dict[str, dict]) -> dict:
    full = fit["1111"]
    full_pass = bool(target_pass(full, .30, 0.0) and controls_pass(full))
    eligible = [
        cell for cell in fit.values()
        if target_pass(cell, .40, .10) and controls_pass(cell)
    ] if full_pass else []
    eligible.sort(key=lambda cell: (
        len(cell["channels"]),
        -min(report["bootstrap95_lower_recovery"] for report in cell["target"].values()),
        cell["channels"],
    ))
    return {
        "full_set_passed": full_pass,
        "eligible_masks": [f"{cell['mask']:04b}" for cell in eligible],
        "selected_mask": f"{eligible[0]['mask']:04b}" if eligible else None,
        "selected_channels": eligible[0]["channels"] if eligible else [],
    }


def select_holds(cell: dict) -> bool:
    return bool(target_pass(cell, .30, 0.0) and controls_pass(cell))


def mobius(values: dict[int, float]) -> dict[str, float]:
    coefficients: dict[int, float] = {}
    for mask in range(16):
        subtotal = sum(value for subset, value in coefficients.items() if subset != mask and (subset & mask) == subset)
        coefficients[mask] = float(values[mask] - subtotal)
    return {f"{mask:04b}": value for mask, value in coefficients.items()}


def interaction_report(fit: dict[str, dict]) -> dict:
    output = {}
    for target in ("selector_score", "payload_value", "matchbreak_score", "joint_restoration"):
        values = {0: 0.0}
        for key, cell in fit.items():
            recovery = cell["target"][target]["recovery"]
            values[int(key, 2)] = float(recovery) if recovery is not None else float("nan")
        output[target] = mobius(values)
    return output


def synthetic_fit() -> dict[str, dict]:
    output = {}
    for mask in FIT_MASKS:
        recovery = .55 if mask & 0b0011 == 0b0011 else .20
        lower = .20 if mask & 0b0011 == 0b0011 else -.05
        output[f"{mask:04b}"] = {
            "mask": mask,
            "channels": [TERM_NAMES[i] for i in range(4) if mask & (1 << i)],
            "target": {name: {"recovery": recovery, "bootstrap95_lower_recovery": lower}
                       for name in ("selector_score", "payload_value", "matchbreak_score", "joint_restoration")},
            "controls": {
                "selector_payload_cross": .05,
                "payload_score_cross": .05,
                "matchbreak_payload_cross": .05,
                "irrelevant_to_native_matchbreak_ratio": .05,
            },
        }
    return output


def main() -> None:
    started = time.time()
    document = validate_static_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        fit = synthetic_fit()
        choice = choose_fit(fit)
        interactions = interaction_report(fit)
        assert choice["selected_mask"] == "0011"
        assert all(len(values) == 16 for values in interactions.values())
        print(json.dumps({
            "status": "dryrun_passed",
            "rung": 558,
            "selected_mask": choice["selected_mask"],
            "selected_channels": choice["selected_channels"],
            "fit_masks": len(fit),
            "mobius_coefficients_per_target": 16,
            "model_loaded": False,
            "r554_result_required_only_at_execution": True,
        }, indent=2))
        return
    capability, audit = validate_dependency()
    if OUT.exists():
        raise RuntimeError("R558 result namespace already exists")
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    fit_raw, fit_execution = evaluate(model, document, "FIT", FIT_MASKS)
    fit = reports(fit_raw, FIT_MASKS)
    choice = choose_fit(fit)
    select_raw = select = select_execution = None
    selected_holds = False
    opened_splits = ["FIT"]
    if choice["selected_mask"] is not None:
        masks = tuple(sorted({15, int(choice["selected_mask"], 2)}))
        select_raw, select_execution = evaluate(model, document, "SELECT", masks)
        select = reports(select_raw, masks)
        selected_holds = select_holds(select[choice["selected_mask"]])
        opened_splits.append("SELECT")
    exact = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and fit_execution["native_replay_relative_squared_error"] <= 1e-12
        and fit_execution["max_factor_relative_squared_reconstruction_error"] <= 1e-10
        and (select_execution is None or (
            select_execution["native_replay_relative_squared_error"] <= 1e-12
            and select_execution["max_factor_relative_squared_reconstruction_error"] <= 1e-10
        ))
    )
    result = {
        "rung": 558,
        "stage": "induction_factor_site_subset_lattice",
        "pred_0_exact_instrument": exact,
        "fit_choice": choice,
        "fit_reports": fit,
        "fit_mobius_interactions": interaction_report(fit),
        "select_reports": select,
        "selected_subset_held": bool(exact and selected_holds),
        "fit_raw_group_statistics": fit_raw,
        "select_raw_group_statistics": select_raw,
        "execution": {"fit": fit_execution, "select": select_execution},
        "model_forwards": fit_execution["model_forwards"] + (
            select_execution["model_forwards"] if select_execution else 0
        ),
        "model_backwards": 0,
        "model_weights_updated": False,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_sha256": {str(path): sha256(path) for path in STATIC_HASHES},
        "r554_sha256": sha256(R554),
        "r555_sha256": sha256(R555),
        "r554_all_gates_pass": capability["all_gates_pass"],
        "r555_all_gates_pass": audit["all_gates_pass"],
        "evaluated_splits": opened_splits,
        "forbidden_splits_opened": [],
        "elapsed_seconds": time.time() - started,
        "decision": (
            "held_factor_level_site_subset" if exact and selected_holds
            else "factor_level_site_capacity_or_selectivity_null"
        ),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({
        "pred_0_exact_instrument": exact,
        "fit_choice": choice,
        "selected_subset_held": result["selected_subset_held"],
        "model_forwards": result["model_forwards"],
        "evaluated_splits": opened_splits,
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
