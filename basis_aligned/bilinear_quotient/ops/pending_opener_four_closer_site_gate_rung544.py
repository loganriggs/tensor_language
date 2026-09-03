#!/usr/bin/env python3
"""Four-closer capability and full-state causal-site gate."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
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
import pending_opener_common_site_rung538 as core  # noqa: E402

ROWS = ROOT / "pending_opener_unique_rows_rung543_v2.json"
RECEIPT = ROOT / "pending_opener_unique_rows_rung543_v2_receipt.json"
PREREG = POLY / "PENDING_OPENER_FOUR_CLOSER_SITE_GATE_RUNG544_PREREGISTRATION.md"
OUT = ROOT / "pending_opener_four_closer_site_gate_rung544_results.json"
HASHES = {
    ROWS: "894f882c2c8f039eda8320c855ca899ce3d7da4617c935b6b6c2bbf8be0cfd69",
    RECEIPT: "6eeabbe802b21d41b7a79ce03a523db6ad6a569a8ee2a80ea0b419c2d1017ad3",
    PREREG: "4b8e449bd546ab351f80701ae595178806b9e81aff0fa8a3f08634d111ca1f13",
}
SPLITS = ("FIT", "SELECT")
TARGETS = ("direct_type_substitution", "completed_then_reopened_order")
CONTROLS = ("surface_paraphrase", "distance_shift", "nonopener_punctuation_substitution")
SITES = ("resid8", "attn13h8")
CLOSERS = (8, 60, 92, 1)
BATCH = 8
BOOTSTRAPS = 2000
SEED = 544
EXPECTED_PAIRS = 720
EXPECTED_FORWARDS = math.ceil(EXPECTED_PAIRS / BATCH) * (1 + 2 * len(SITES))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_lower(values: list[float], seed: int, *, absolute: bool = False) -> float:
    array = np.asarray(values, dtype=np.float64)
    if absolute:
        array = np.abs(array)
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(1), .025))


def closer_margin(logits: torch.Tensor, answer_id: int) -> float:
    alternatives = [token for token in CLOSERS if token != answer_id]
    return float(logits[answer_id] - logits[alternatives].mean())


def load_rows() -> list[dict]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    rows = [row for row in json.loads(ROWS.read_text())["rows"] if row["split"] in SPLITS]
    if len(rows) != EXPECTED_PAIRS or len({row["row_id"] for row in rows}) != EXPECTED_PAIRS:
        raise RuntimeError("R543v2 FIT/SELECT identity changed")
    if any(row["base_answer_id"] not in CLOSERS or row["donor_answer_id"] not in CLOSERS for row in rows):
        raise RuntimeError("unexpected closer token")
    return rows


def pad(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(max(len(row["base_ids"]), len(row["donor_ids"])) for row in rows)
    base = torch.full((len(rows), length), 50256, dtype=torch.long, device="cuda")
    donor = base.clone()
    base_finals, donor_finals = [], []
    for index, row in enumerate(rows):
        base[index, :len(row["base_ids"])] = torch.tensor(row["base_ids"], device="cuda")
        donor[index, :len(row["donor_ids"])] = torch.tensor(row["donor_ids"], device="cuda")
        base_finals.append(len(row["base_ids"]) - 1)
        donor_finals.append(len(row["donor_ids"]) - 1)
    return base, donor, torch.tensor(base_finals, device="cuda"), torch.tensor(donor_finals, device="cuda")


def score_capability(raw: dict) -> tuple[dict, bool]:
    reports, seed, all_pass = {}, SEED, True
    for split in SPLITS:
        reports[split] = {}
        for family in TARGETS + CONTROLS:
            rows = raw[split][family]
            if family in TARGETS:
                by_pair = {}
                for answer_pair in sorted({(row["base_answer_id"], row["donor_answer_id"]) for row in rows}):
                    cell = [row for row in rows if (row["base_answer_id"], row["donor_answer_id"]) == answer_pair]
                    base_fraction = float(np.mean([row["base_margin"] > 0 for row in cell]))
                    donor_fraction = float(np.mean([row["donor_margin"] > 0 for row in cell]))
                    by_pair[f"{answer_pair[0]}->{answer_pair[1]}"] = {
                        "n": len(cell), "base_correct_fraction": base_fraction,
                        "donor_correct_fraction": donor_fraction,
                        "passed": bool(base_fraction >= .75 and donor_fraction >= .75),
                    }
                margins = [row["base_margin"] for row in rows] + [row["donor_margin"] for row in rows]
                report = {
                    "n": len(rows), "ordered_pairs": by_pair,
                    "mean_symmetric_margin": float(np.mean(margins)),
                    "bootstrap95_lower_symmetric_margin": bootstrap_lower(margins, seed),
                }
                seed += 1
                report["passed"] = bool(all(cell["passed"] for cell in by_pair.values())
                                        and report["mean_symmetric_margin"] > 0
                                        and report["bootstrap95_lower_symmetric_margin"] > 0)
            else:
                base_fraction = float(np.mean([row["base_margin"] > 0 for row in rows]))
                donor_fraction = float(np.mean([row["donor_margin"] > 0 for row in rows]))
                report = {"n": len(rows), "base_correct_fraction": base_fraction,
                          "donor_correct_fraction": donor_fraction,
                          "passed": bool(base_fraction >= .75 and donor_fraction >= .75)}
            all_pass &= report["passed"]
            reports[split][family] = report
    return reports, bool(all_pass)


def score_sites(raw: dict) -> tuple[dict, list[str]]:
    reports, passing, seed = {}, [], SEED + 1000
    for site in SITES:
        reports[site] = {}
        site_pass = True
        for split in SPLITS:
            reports[site][split] = {}
            for family in TARGETS:
                reports[site][split][family] = {}
                family_pass = True
                for direction in ("base_to_donor", "donor_to_base"):
                    rows = raw[site][split][family][direction]
                    values = [row["endpoint_change"] for row in rows]
                    pair_reports = {}
                    for pair in sorted({row["ordered_pair"] for row in rows}):
                        cell = [row["endpoint_change"] for row in rows if row["ordered_pair"] == pair]
                        pair_reports[pair] = {"n": len(cell), "mean": float(np.mean(cell)),
                                              "positive_fraction": float(np.mean(np.asarray(cell) > 0))}
                        pair_reports[pair]["passed"] = bool(
                            pair_reports[pair]["mean"] > 0 and pair_reports[pair]["positive_fraction"] >= .5)
                    report = {"n": len(values), "mean": float(np.mean(values)),
                              "bootstrap95_lower_mean": bootstrap_lower(values, seed),
                              "positive_fraction": float(np.mean(np.asarray(values) > 0)),
                              "ordered_pairs": pair_reports}
                    seed += 1
                    report["passed"] = bool(report["mean"] > 0 and report["bootstrap95_lower_mean"] > 0
                                            and report["positive_fraction"] >= .70
                                            and all(item["passed"] for item in pair_reports.values()))
                    family_pass &= report["passed"]
                    reports[site][split][family][direction] = report
                reports[site][split][family]["passed"] = bool(family_pass)
                site_pass &= family_pass
            for family in CONTROLS:
                reports[site][split][family] = {}
                family_pass = True
                for direction in ("base_to_donor", "donor_to_base"):
                    rows = raw[site][split][family][direction]
                    endpoint = [row["endpoint_change"] for row in rows]
                    rms = [row["full_logit_rms"] for row in rows]
                    report = {"n": len(endpoint), "mean_absolute_endpoint_change": float(np.mean(np.abs(endpoint))),
                              "bootstrap95_lower_mean_absolute": bootstrap_lower(endpoint, seed, absolute=True),
                              "mean_full_vocabulary_logit_rms": float(np.mean(rms))}
                    seed += 1
                    report["causally_live"] = bool(report["bootstrap95_lower_mean_absolute"] > .03
                                                    and report["mean_full_vocabulary_logit_rms"] > .01)
                    family_pass &= report["causally_live"]
                    reports[site][split][family][direction] = report
                reports[site][split][family]["causally_live"] = bool(family_pass)
                site_pass &= family_pass
        reports[site]["passed"] = bool(site_pass)
        if site_pass:
            passing.append(site)
    return reports, passing


def main() -> None:
    started = time.time()
    rows = load_rows()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({"status": "dryrun_passed", "pairs": len(rows), "sites": SITES,
                          "expected_forwards": EXPECTED_FORWARDS, "final_or_ood_opened": False}, indent=2))
        return
    if sha256(PREREG) != HASHES[PREREG]:
        raise RuntimeError("preregistration hash mismatch")
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    capability_raw = {split: {family: [] for family in TARGETS + CONTROLS} for split in SPLITS}
    raw = {site: {split: {family: {direction: [] for direction in ("base_to_donor", "donor_to_base")}
                                  for family in TARGETS + CONTROLS} for split in SPLITS} for site in SITES}
    calls, minimum_edit_rms = 0, float("inf")
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start+BATCH]
        base, donor, base_finals, donor_finals = pad(chunk)
        both, finals = torch.cat((base, donor)), torch.cat((base_finals, donor_finals))
        native, states = core.capture_all(model, both, finals)
        calls += 1
        arange = torch.arange(len(chunk), device="cuda")
        base_native = native[arange, base_finals]
        donor_native = native[arange + len(chunk), donor_finals]
        for index, row in enumerate(chunk):
            capability_raw[row["split"]][row["family_id"]].append({
                "row_id": row["row_id"], "group_id": row["group_id"],
                "base_answer_id": row["base_answer_id"], "donor_answer_id": row["donor_answer_id"],
                "base_margin": closer_margin(base_native[index], row["base_answer_id"]),
                "donor_margin": closer_margin(donor_native[index], row["donor_answer_id"]),
            })
        for site in SITES:
            base_state, donor_state = states[site].chunk(2)
            minimum_edit_rms = min(minimum_edit_rms, float(
                (donor_state.float() - base_state.float()).square().mean(-1).sqrt().min()))
            base_patch = core.patched(model, base, base_finals, site, donor_state)[arange, base_finals]
            donor_patch = core.patched(model, donor, donor_finals, site, base_state)[arange, donor_finals]
            calls += 2
            for index, row in enumerate(chunk):
                family, split = row["family_id"], row["split"]
                if family in TARGETS:
                    b0 = float(base_native[index, row["donor_answer_id"]] - base_native[index, row["base_answer_id"]])
                    bp = float(base_patch[index, row["donor_answer_id"]] - base_patch[index, row["base_answer_id"]])
                    d0 = float(donor_native[index, row["base_answer_id"]] - donor_native[index, row["donor_answer_id"]])
                    dp = float(donor_patch[index, row["base_answer_id"]] - donor_patch[index, row["donor_answer_id"]])
                    pair = f"{row['base_answer_id']}->{row['donor_answer_id']}"
                    raw[site][split][family]["base_to_donor"].append(
                        {"row_id": row["row_id"], "group_id": row["group_id"], "ordered_pair": pair,
                         "endpoint_change": bp - b0, "full_logit_rms": float((base_patch[index]-base_native[index]).square().mean().sqrt())})
                    raw[site][split][family]["donor_to_base"].append(
                        {"row_id": row["row_id"], "group_id": row["group_id"], "ordered_pair": pair,
                         "endpoint_change": dp - d0, "full_logit_rms": float((donor_patch[index]-donor_native[index]).square().mean().sqrt())})
                else:
                    b0, bp = closer_margin(base_native[index], row["base_answer_id"]), closer_margin(base_patch[index], row["base_answer_id"])
                    d0, dp = closer_margin(donor_native[index], row["donor_answer_id"]), closer_margin(donor_patch[index], row["donor_answer_id"])
                    raw[site][split][family]["base_to_donor"].append(
                        {"row_id": row["row_id"], "group_id": row["group_id"], "endpoint_change": bp-b0,
                         "full_logit_rms": float((base_patch[index]-base_native[index]).square().mean().sqrt())})
                    raw[site][split][family]["donor_to_base"].append(
                        {"row_id": row["row_id"], "group_id": row["group_id"], "endpoint_change": dp-d0,
                         "full_logit_rms": float((donor_patch[index]-donor_native[index]).square().mean().sqrt())})
    capability, capability_pass = score_capability(capability_raw)
    sites, passing_sites = score_sites(raw)
    selected = passing_sites[0] if capability_pass and passing_sites else None
    result = {
        "rung": 544, "stage": "four_closer_capability_and_site_gate",
        "pred_a_exact_instrument": bool(calls == EXPECTED_FORWARDS and minimum_edit_rms > 0
                                        and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256),
        "pred_b_four_closer_capability": capability_pass,
        "pred_c_common_target_and_control_live_site": selected is not None,
        "strong_null": bool(not capability_pass or selected is None),
        "selected_site": selected, "passing_sites_in_frozen_order": passing_sites,
        "capability": capability, "site_reports": sites,
        "raw_capability": capability_raw, "raw_site_effects": raw,
        "minimum_source_target_activation_rms": minimum_edit_rms,
        "model_forwards": calls, "model_backwards": 0, "model_weights_updated": False,
        "evaluated_splits": list(SPLITS), "forbidden_splits_opened": [],
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_sha256": {str(path): expected for path, expected in HASHES.items()},
        "elapsed_seconds": time.time()-started,
        "next_step": ("preregister_contrastive_readout_deflated_das" if selected
                      else "redesign_behavior_or_site_without_rank_search"),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key.startswith("pred_") or key in {"strong_null", "selected_site", "passing_sites_in_frozen_order", "model_forwards", "model_backwards", "forbidden_splits_opened", "next_step"}}, indent=2))


if __name__ == "__main__":
    main()
