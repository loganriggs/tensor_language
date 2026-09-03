#!/usr/bin/env python3
"""Full-state common-site interchange screen for pending-opener families."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bqlib as B  # noqa: E402
import bilin18_observed_model_facade as facade  # noqa: E402

ROWS = ROOT / "pending_opener_multifamily_rows_rung537.json"
ROWS_RECEIPT = ROOT / "pending_opener_multifamily_rows_rung537_receipt.json"
CONTROLS = ROOT / "pending_opener_controls_rung537.json"
CONTROLS_RECEIPT = ROOT / "pending_opener_controls_rung537_receipt.json"
CAPABILITY = ROOT / "pending_opener_capability_rung537_results.json"
PREREG = POLY / "PENDING_OPENER_COMMON_SITE_RUNG538_PREREGISTRATION.md"
OUT = ROOT / "pending_opener_common_site_rung538_results.json"

HASHES = {
    ROWS: "c62cdf3929231e06de6883d74f3ab2c86bd524e02474bb2259267d6976e9e7d9",
    ROWS_RECEIPT: "d50528aa355ba89ab43edd43491c672a6aed88bd8a805ffda936afbfa4cc4816",
    CONTROLS: "f2693b9b78a9266619afc45ceb6f70e4f2339aa1980263ca22d3ea4453145494",
    CONTROLS_RECEIPT: "1ad594b4fc19abc3dab761f7cd09f3c9df764f7390ada09dc334f4b7268e626c",
    CAPABILITY: "9e76d2c7dba8ea1cfeaf640f9d80508bda3c5df1b151af20f407b535c0dbcb0c",
    PREREG: "5fb04f7e345c7e9ed0fe062e36f97b3ccd6d1f366ecbaaf277b8e1528bd396f8",
}
FAMILIES = ("opener_type_substitution", "closed_then_reopened_type")
SPLITS = ("FIT", "SELECT")
SITE_ORDER = tuple(
    site
    for layer in range(8, 15)
    for site in (
        f"resid{layer}",
        *(("attn13h8",) if layer == 13 else ()),
        f"mlp_product{layer}",
    )
)
PAIR_BATCH = 8
EXPECTED_PAIRS = 128
EXPECTED_BASELINE_FORWARDS = math.ceil(EXPECTED_PAIRS / PAIR_BATCH)
EXPECTED_PATCHED_FORWARDS = 2 * len(SITE_ORDER) * EXPECTED_BASELINE_FORWARDS
EXPECTED_FORWARDS = EXPECTED_BASELINE_FORWARDS + EXPECTED_PATCHED_FORWARDS
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 538


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_rows() -> list[dict]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    capability = json.loads(CAPABILITY.read_text())
    if capability.get("pred_e_dataset_authorized_for_site_screen") is not True:
        raise RuntimeError("capability gate did not authorize the site screen")
    rows = json.loads(ROWS.read_text())["rows"]
    selected = [row for row in rows if row["split"] in SPLITS and row["family_id"] in FAMILIES]
    if len(selected) != EXPECTED_PAIRS:
        raise RuntimeError("FIT/SELECT answer-changing row count changed")
    if any(len(row["base_ids"]) != len(row["donor_ids"]) for row in selected):
        raise RuntimeError("site interchange requires equal-length pairs")
    return selected


def _capture_hook(store: dict, site: str, finals: torch.Tensor):
    def hook(_module, args):
        value = args[0]
        picked = value[torch.arange(value.shape[0], device=value.device), finals]
        if site == "attn13h8":
            picked = picked[:, 8 * 128:9 * 128]
        store[site] = picked.detach().clone()
        return None
    return hook


def _patch_hook(site: str, replacement: torch.Tensor, finals: torch.Tensor):
    def hook(_module, args):
        value = args[0].clone()
        arange = torch.arange(value.shape[0], device=value.device)
        if site == "attn13h8":
            value[arange, finals, 8 * 128:9 * 128] = replacement.to(value.dtype)
        else:
            value[arange, finals] = replacement.to(value.dtype)
        return (value,) + tuple(args[1:])
    return hook


def _module_for(model, site: str):
    if site.startswith("resid"):
        return model.transformer.h[int(site.removeprefix("resid"))]
    if site == "attn13h8":
        return model.transformer.h[13].attn.c_proj
    if site.startswith("mlp_product"):
        return model.transformer.h[int(site.removeprefix("mlp_product"))].mlp.Down
    raise KeyError(site)


@torch.no_grad()
def forward_native(model, tokens: torch.Tensor) -> torch.Tensor:
    x = F.rms_norm(model.transformer.wte(tokens), (1152,))
    x0, v1 = x, None
    for block in model.transformer.h:
        x, v1 = block(x, v1, x0)
    return (30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (1152,))) / 30.0)).float()


def capture_all(model, tokens: torch.Tensor, finals: torch.Tensor) -> tuple[torch.Tensor, dict]:
    store, handles = {}, []
    try:
        for site in SITE_ORDER:
            handles.append(_module_for(model, site).register_forward_pre_hook(
                _capture_hook(store, site, finals)))
        logits = forward_native(model, tokens)
    finally:
        for handle in handles:
            handle.remove()
    if set(store) != set(SITE_ORDER):
        raise RuntimeError("not every frozen site was captured")
    return logits, store


def patched(model, tokens: torch.Tensor, finals: torch.Tensor, site: str, source: torch.Tensor) -> torch.Tensor:
    handle = _module_for(model, site).register_forward_pre_hook(_patch_hook(site, source, finals))
    try:
        return forward_native(model, tokens)
    finally:
        handle.remove()


def bootstrap_lower(values: list[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(axis=1), 0.025))


def summarize(values: list[float], seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    fraction = float((array > 0).mean())
    lower = bootstrap_lower(values, seed)
    return {
        "n": len(values), "mean_donorward_movement": mean,
        "positive_movement_fraction": fraction,
        "bootstrap95_lower_mean": lower,
        "passed": bool(mean > 0 and fraction >= 0.70 and lower > 0),
    }


def score(raw: dict) -> tuple[dict, list[str]]:
    report = {}
    seed = BOOTSTRAP_SEED
    for site in SITE_ORDER:
        report[site] = {}
        for split in SPLITS:
            report[site][split] = {}
            for family in FAMILIES:
                directions = {}
                for direction in ("base_to_donor", "donor_to_base"):
                    values = raw[site][split][family][direction]
                    directions[direction] = summarize(values, seed)
                    seed += 1
                directions["passed"] = all(directions[d]["passed"] for d in ("base_to_donor", "donor_to_base"))
                report[site][split][family] = directions
        report[site]["common_live"] = all(
            report[site][split][family]["passed"]
            for split in SPLITS for family in FAMILIES
        )
    passing = [site for site in SITE_ORDER if report[site]["common_live"]]
    return report, passing


def main() -> None:
    started = time.time()
    rows = selected_rows()
    if B.DRYRUN:
        print(json.dumps({
            "status": "dryrun_passed", "pairs": len(rows),
            "sites": list(SITE_ORDER), "expected_forwards": EXPECTED_FORWARDS,
            "splits": list(SPLITS), "families": list(FAMILIES),
        }, indent=2))
        return

    if B.m is None:
        raise RuntimeError("model was disabled outside dry-run")
    model = B.m
    raw = {
        site: {
            split: {family: {"base_to_donor": [], "donor_to_base": []} for family in FAMILIES}
            for split in SPLITS
        } for site in SITE_ORDER
    }
    calls, min_edit_rms = 0, float("inf")
    for start in range(0, len(rows), PAIR_BATCH):
        chunk = rows[start:start + PAIR_BATCH]
        length = max(len(row["base_ids"]) for row in chunk)
        base = torch.full((len(chunk), length), 50256, dtype=torch.long, device=B.DEV)
        donor = base.clone()
        finals = torch.tensor([len(row["base_ids"]) - 1 for row in chunk], device=B.DEV)
        for index, row in enumerate(chunk):
            base[index, :len(row["base_ids"])] = torch.tensor(row["base_ids"], device=B.DEV)
            donor[index, :len(row["donor_ids"])] = torch.tensor(row["donor_ids"], device=B.DEV)
        both = torch.cat((base, donor))
        both_finals = torch.cat((finals, finals))
        native, states = capture_all(model, both, both_finals)
        calls += 1
        arange = torch.arange(len(chunk), device=B.DEV)
        base_native = native[arange, finals]
        donor_native = native[arange + len(chunk), finals]

        for site in SITE_ORDER:
            base_state, donor_state = states[site].chunk(2)
            edit_rms = (donor_state.float() - base_state.float()).square().mean(-1).sqrt()
            min_edit_rms = min(min_edit_rms, float(edit_rms.min()))
            base_patch = patched(model, base, finals, site, donor_state)[arange, finals]
            donor_patch = patched(model, donor, finals, site, base_state)[arange, finals]
            calls += 2
            for index, row in enumerate(chunk):
                yb, yd = row["base_answer_id"], row["donor_answer_id"]
                b0 = float(base_native[index, yd] - base_native[index, yb])
                bp = float(base_patch[index, yd] - base_patch[index, yb])
                d0 = float(donor_native[index, yb] - donor_native[index, yd])
                dp = float(donor_patch[index, yb] - donor_patch[index, yd])
                cell = raw[site][row["split"]][row["family_id"]]
                cell["base_to_donor"].append(bp - b0)
                cell["donor_to_base"].append(dp - d0)
        del base, donor, both, native, states

    reports, passing = score(raw)
    instrument_valid = bool(
        calls == EXPECTED_FORWARDS and min_edit_rms > 0
        and facade.WEIGHTS_SHA256 == "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
    )
    selected = passing[0] if instrument_valid and passing else None
    frozen_selection_valid = bool(
        selected is None or selected == next(site for site in SITE_ORDER if site in passing)
    )
    result = {
        "rung": 538, "stage": "common_site_full_state_interchange",
        "pred_a_exact_instrument": instrument_valid,
        "pred_b_common_live_site": selected is not None,
        "pred_c_frozen_causal_order_selection": frozen_selection_valid,
        "strong_null": bool(instrument_valid and frozen_selection_valid and selected is None),
        "selected_site": selected, "passing_sites_in_frozen_order": passing,
        "reports": reports,
        "evaluated_splits": list(SPLITS), "forbidden_splits_opened": [],
        "evaluated_families": list(FAMILIES),
        "model_forwards": calls, "model_backwards": 0,
        "minimum_source_target_activation_rms": min_edit_rms,
        "checkpoint_weights_sha256": facade.WEIGHTS_SHA256,
        "input_sha256": {str(path): expected for path, expected in HASHES.items()},
        "implementation_price": {
            "pairs": len(rows), "sites": len(SITE_ORDER),
            "baseline_forwards": EXPECTED_BASELINE_FORWARDS,
            "patched_forwards": EXPECTED_PATCHED_FORWARDS,
            "total_forwards": EXPECTED_FORWARDS, "backwards": 0,
        },
        "elapsed_seconds": time.time() - started,
        "next_step": (
            "fit_shared_and_family_specific_projectors_at_selected_site"
            if selected else "redesign_causal_site_vocabulary_without_rank_search"
        ),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
