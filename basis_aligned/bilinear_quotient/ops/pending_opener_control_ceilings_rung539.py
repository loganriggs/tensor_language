#!/usr/bin/env python3
"""Selected-site full-state ceilings for answer-preserving opener controls."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bilin18_observed_model_facade as facade  # noqa: E402
import pending_opener_common_site_rung538 as site_core  # noqa: E402

ROWS = ROOT / "pending_opener_multifamily_rows_rung537.json"
CONTROLS = ROOT / "pending_opener_controls_rung537.json"
SITE_RESULT = ROOT / "pending_opener_common_site_rung538_results.json"
SITE_AUDIT = ROOT / "pending_opener_common_site_rung538_terminal_audit.json"
PREREG = POLY / "PENDING_OPENER_CONTROL_CEILINGS_RUNG539_PREREGISTRATION.md"
OUT = ROOT / "pending_opener_control_ceilings_rung539_results.json"
HASHES = {
    ROWS: "c62cdf3929231e06de6883d74f3ab2c86bd524e02474bb2259267d6976e9e7d9",
    CONTROLS: "f2693b9b78a9266619afc45ceb6f70e4f2339aa1980263ca22d3ea4453145494",
    SITE_RESULT: "f011399614953c958faf2a12ef15e938dcc2f5e3f52ea868763de2a82443a205",
    SITE_AUDIT: "e1ec113d48f942105d233bd4ac3a2e1bd619c5f94bd364c869351da33324f285",
    PREREG: "87abcdefc0f5432d957f1e06c46c98e06f79a83b42e5f92badfc1d6246e78c0a",
}
FAMILIES = ("pending_state_preserved_surface_edit", "nonopener_punctuation_substitution")
SPLITS = ("FIT", "SELECT")
PAIR_BATCH = 8
EXPECTED_PAIRS = 128
EXPECTED_FORWARDS = 48
BOOTSTRAPS = 2000
SEED = 539
SITE = "resid8"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows() -> list[dict]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    site = json.loads(SITE_RESULT.read_text())
    audit = json.loads(SITE_AUDIT.read_text())
    if site.get("selected_site") != SITE or audit.get("all_checks_pass") is not True:
        raise RuntimeError("verified R538 site authority is missing")
    main = [row for row in json.loads(ROWS.read_text())["rows"]
            if row["split"] in SPLITS and row["family_id"] == FAMILIES[0]]
    controls = [row for row in json.loads(CONTROLS.read_text())["rows"]
                if row["split"] in SPLITS and row["family_id"] == FAMILIES[1]]
    rows = main + controls
    if len(main) != 64 or len(controls) != 64:
        raise RuntimeError("control row count changed")
    return rows


def capture_resid8(model, tokens: torch.Tensor, finals: torch.Tensor):
    store = {}

    def capture(_module, args):
        x = args[0]
        store["state"] = x[torch.arange(x.shape[0], device=x.device), finals].detach().clone()

    handle = model.transformer.h[8].register_forward_pre_hook(capture)
    try:
        logits = site_core.forward_native(model, tokens)
    finally:
        handle.remove()
    return logits, store["state"]


def patch_resid8(model, tokens: torch.Tensor, finals: torch.Tensor, source: torch.Tensor):
    def patch(_module, args):
        x = args[0].clone()
        x[torch.arange(x.shape[0], device=x.device), finals] = source.to(x.dtype)
        return (x,) + tuple(args[1:])

    handle = model.transformer.h[8].register_forward_pre_hook(patch)
    try:
        return site_core.forward_native(model, tokens)
    finally:
        handle.remove()


def lower_abs(values: list[float], seed: int) -> float:
    array = np.abs(np.asarray(values, dtype=np.float64))
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(axis=1), 0.025))


def summarize(endpoint: list[float], logit_rms: list[float], seed: int) -> dict:
    array = np.asarray(endpoint, dtype=np.float64)
    return {
        "n": len(endpoint), "mean_signed_endpoint_change": float(array.mean()),
        "mean_absolute_endpoint_change": float(np.abs(array).mean()),
        "bootstrap95_lower_mean_absolute": lower_abs(endpoint, seed),
        "mean_full_vocabulary_logit_rms": float(np.mean(logit_rms)),
    }


def main() -> None:
    started = time.time()
    rows = load_rows()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dryrun_passed", "pairs": len(rows), "site": SITE,
            "families": list(FAMILIES), "splits": list(SPLITS),
            "expected_forwards": EXPECTED_FORWARDS,
        }, indent=2))
        return

    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    raw = {
        split: {family: {direction: {"endpoint_change": [], "logit_rms": []}
                         for direction in ("base_to_donor", "donor_to_base")}
                for family in FAMILIES}
        for split in SPLITS
    }
    calls, min_activation_rms = 0, float("inf")
    for start in range(0, len(rows), PAIR_BATCH):
        chunk = rows[start:start + PAIR_BATCH]
        length = max(max(len(row["base_ids"]), len(row["donor_ids"])) for row in chunk)
        base = torch.full((len(chunk), length), 50256, dtype=torch.long, device="cuda")
        donor = base.clone()
        base_finals = torch.tensor([len(row["base_ids"]) - 1 for row in chunk], device="cuda")
        donor_finals = torch.tensor([len(row["donor_ids"]) - 1 for row in chunk], device="cuda")
        for index, row in enumerate(chunk):
            base[index, :len(row["base_ids"])] = torch.tensor(row["base_ids"], device="cuda")
            donor[index, :len(row["donor_ids"])] = torch.tensor(row["donor_ids"], device="cuda")
        both = torch.cat((base, donor))
        both_finals = torch.cat((base_finals, donor_finals))
        native, states = capture_resid8(model, both, both_finals)
        calls += 1
        base_state, donor_state = states.chunk(2)
        min_activation_rms = min(min_activation_rms, float(
            (donor_state.float() - base_state.float()).square().mean(-1).sqrt().min()))
        base_patch = patch_resid8(model, base, base_finals, donor_state)
        donor_patch = patch_resid8(model, donor, donor_finals, base_state)
        calls += 2
        arange = torch.arange(len(chunk), device="cuda")
        base_native = native[arange, base_finals]
        donor_native = native[arange + len(chunk), donor_finals]
        base_patched = base_patch[arange, base_finals]
        donor_patched = donor_patch[arange, donor_finals]
        for index, row in enumerate(chunk):
            for direction, original, changed in (
                ("base_to_donor", base_native[index], base_patched[index]),
                ("donor_to_base", donor_native[index], donor_patched[index]),
            ):
                original_margin = float(original[8] - original[1])
                changed_margin = float(changed[8] - changed[1])
                cell = raw[row["split"]][row["family_id"]][direction]
                cell["endpoint_change"].append(changed_margin - original_margin)
                cell["logit_rms"].append(float((changed - original).square().mean().sqrt()))

    reports, seed = {}, SEED
    for split in SPLITS:
        reports[split] = {}
        for family in FAMILIES:
            reports[split][family] = {}
            for direction in ("base_to_donor", "donor_to_base"):
                cell = raw[split][family][direction]
                reports[split][family][direction] = summarize(
                    cell["endpoint_change"], cell["logit_rms"], seed)
                seed += 1
            reports[split][family]["causally_testable"] = all(
                reports[split][family][direction]["bootstrap95_lower_mean_absolute"] > 0.05
                and reports[split][family][direction]["mean_full_vocabulary_logit_rms"] > 0.01
                for direction in ("base_to_donor", "donor_to_base")
            )

    instrument = bool(calls == EXPECTED_FORWARDS and min_activation_rms > 0
                      and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
    pred_b = all(reports[split][FAMILIES[0]]["causally_testable"] for split in SPLITS)
    pred_c = all(reports[split][FAMILIES[1]]["causally_testable"] for split in SPLITS)
    result = {
        "rung": 539, "stage": "selected_site_invariance_control_ceilings",
        "pred_a_exact_instrument": instrument,
        "pred_b_surface_invariance_causally_testable": pred_b,
        "pred_c_nonopener_control_causally_testable": pred_c,
        "strong_null": bool(instrument and not pred_b and not pred_c),
        "site": SITE, "reports": reports, "raw_sufficient_statistics": raw,
        "minimum_source_target_activation_rms": min_activation_rms,
        "model_forwards": calls, "model_backwards": 0,
        "evaluated_splits": list(SPLITS), "forbidden_splits_opened": [],
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_sha256": {str(path): expected for path, expected in HASHES.items()},
        "elapsed_seconds": time.time() - started,
        "next_step": "freeze_projector_invariance_bars_using_live_control_ceilings",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = {key: value for key, value in result.items() if key.startswith("pred_")}
    summary.update({key: result[key] for key in (
        "strong_null", "site", "minimum_source_target_activation_rms",
        "model_forwards", "model_backwards", "next_step")})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
