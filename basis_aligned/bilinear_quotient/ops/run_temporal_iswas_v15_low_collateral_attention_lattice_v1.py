#!/usr/bin/env python3
"""Complete five-piece low-collateral attention lattice on v15."""
# BQGATE: EXPERIMENT pred_a_authority_population_lattice_closure_finiteness_and_exact_price pred_b_target_sufficient_selective_subset_exists pred_c_selected_subset_confirms_without_reselection pred_d_selected_subset_is_proper pred_e_every_selected_piece_is_necessary pred_f_final_residual_tracks_selected_behavior
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v15_cross_boundary_adaptive_greedy_v1 as greedy


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_low_collateral_attention_lattice_v1.json"
GREEDY_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_cross_boundary_adaptive_greedy_v1_result.json"
GREEDY_RUNNER = ROOT / "ops/run_temporal_iswas_v15_cross_boundary_adaptive_greedy_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_low_collateral_attention_lattice_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_low_collateral_attention_lattice_v1"
EXPECTED = {
    "prior": "a04c8bd73b750f6eaadcfeaed0cdd8c8a83a0745d3ff03e062476aebadd9c125",
    "greedy_result": "62d1d16422c756da4838507b467b1a7d72f51981b9dc9a851f8b8f4485c59ece",
    "greedy_runner": "de7906f02a1739c0ef01d08210a68140bfcac3a496a3d62780bfc024423f6954",
}
COMPONENTS = ("L8H1", "L9H1", "L9H4", "L11H3", "attn:15")
TARGET_PANELS = ("A1", "A2")
CONTROL_PANELS = ("P", "C")
EXACT_FORWARDS = 35


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def main():
    paths = {"prior": PRIOR, "greedy_result": GREEDY_RESULT, "greedy_runner": GREEDY_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    prior, parent = json.loads(PRIOR.read_text()), json.loads(GREEDY_RESULT.read_text())
    rows = fresh.build_rows()
    panel_counts = {panel: sum(row["transform_id"] == panel for row in rows)
                    for panel in TARGET_PANELS + CONTROL_PANELS}
    stage_one = parent["stages"][0]["candidates"]
    derived = tuple(item["added"] for item in stage_one
                    if item["report"]["controls"]["median_kl"] <= .02
                    and item["report"]["controls"]["top1_flip_count"] <= 2)
    authority_ok = bool(observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
                        and derived == COMPONENTS
                        and panel_counts == {panel: 16 for panel in TARGET_PANELS + CONTROL_PANELS}
                        and len(rows) == len({row["row_id"] for row in rows}) == 64)
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "components": list(COMPONENTS),
        "rows": len(rows), "subset_arms": 32, "model_forwards_exact": EXACT_FORWARDS,
        "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, F = backend.torch, backend.F
    native, forwards = backend.native, 0

    def counted(*args, **kwargs):
        nonlocal forwards
        forwards += 1
        return native(*args, **kwargs)

    backend.native = counted
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, base_cache = greedy.capture(backend, base_batch)
    donor_output, donor_cache = greedy.capture(backend, donor_batch)
    aligned = set(base_cache) == set(donor_cache) and all(
        base_cache[key].shape == donor_cache[key].shape for key in base_cache)
    if not aligned:
        raise RuntimeError("base/donor response shapes changed")
    base_state = greedy.module_impl.states(torch, backend, base_output, rows)
    donor_state = greedy.module_impl.states(torch, backend, donor_output, rows)
    base_logits = das.head_logits(backend, base_state).float()
    donor_logits = das.head_logits(backend, donor_state).float()
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    all_index = torch.arange(len(rows), device=backend.device)
    panel_indices = {panel: torch.as_tensor(
        [i for i, row in enumerate(rows) if row["transform_id"] == panel], device=backend.device)
        for panel in TARGET_PANELS + CONTROL_PANELS}
    control_ids = [i for i, row in enumerate(rows) if row["transform_id"] in CONTROL_PANELS]
    control_index = torch.as_tensor(control_ids, device=backend.device)

    def margins(logits):
        return logits[all_index, answer] - logits[all_index, foil]

    base_margin, donor_margin = margins(base_logits), margins(donor_logits)
    target_margin, target_state = donor_margin - base_margin, donor_state - base_state

    def report(output):
        state = greedy.module_impl.states(torch, backend, output, rows)
        logits = das.head_logits(backend, state).float()
        delta_margin, delta_state = margins(logits) - base_margin, state - base_state
        targets = {}
        for panel in TARGET_PANELS:
            ix = panel_indices[panel]
            behavior = greedy.module_impl.vector_metrics(torch, delta_margin[ix], target_margin[ix])
            behavior["direction_fraction"] = float(
                ((delta_margin[ix] * target_margin[ix]) > 0).float().mean())
            targets[panel] = {"behavior": behavior, "final_residual":
                greedy.module_impl.vector_metrics(torch, delta_state[ix], target_state[ix])}
        lb, lp = base_logits[control_index], logits[control_index]
        logb, logp = F.log_softmax(lb, -1), F.log_softmax(lp, -1)
        kl = (logb.exp() * (logb - logp)).sum(-1)
        flips = lb.argmax(-1) != lp.argmax(-1)
        controls = {"median_kl": float(kl.median()), "max_kl": float(kl.max()),
                    "top1_flip_fraction": float(flips.float().mean()),
                    "top1_flip_count": int(flips.sum()),
                    "flipped_row_ids": [rows[control_ids[i]]["row_id"]
                                        for i in range(len(control_ids)) if bool(flips[i])]}
        return {"targets": targets, "controls": controls}

    def eligible(value):
        a1, controls = value["targets"]["A1"]["behavior"], value["controls"]
        return bool(a1["signed_projection"] >= .75 and a1["direction_fraction"] >= .875
                    and controls["top1_flip_count"] == 0 and controls["median_kl"] <= .02)

    self_output = greedy.run_patch(backend, base_batch, base_cache, COMPONENTS)
    self_state = greedy.module_impl.states(torch, backend, self_output, rows)
    self_logits = das.head_logits(backend, self_state).float()
    self_error = max(float((self_state - base_state).abs().max()),
                     float((self_logits - base_logits).abs().max()))
    reports = {}
    for mask in range(32):
        support = tuple(site for bit, site in enumerate(COMPONENTS) if mask & (1 << bit))
        value = report(greedy.run_patch(backend, base_batch, donor_cache, support))
        reports[str(mask)] = {"support": list(support), "eligible": eligible(value), "report": value}

    eligible_masks = [mask for mask in range(32) if reports[str(mask)]["eligible"]]
    selected_mask = min(eligible_masks, key=lambda mask: (
        mask.bit_count(), -reports[str(mask)]["report"]["targets"]["A1"]["behavior"]["signed_projection"], mask
    )) if eligible_masks else None
    selected = reports[str(selected_mask)] if selected_mask is not None else None
    necessity = {}
    if selected_mask is not None:
        selected_a1 = selected["report"]["targets"]["A1"]["behavior"]["signed_projection"]
        for bit, site in enumerate(COMPONENTS):
            if not selected_mask & (1 << bit):
                continue
            removed_mask = selected_mask & ~(1 << bit)
            removed = reports[str(removed_mask)]
            drop = selected_a1 - removed["report"]["targets"]["A1"]["behavior"]["signed_projection"]
            necessity[site] = {"removed_mask": removed_mask, "a1_projection_drop": drop,
                               "removed_eligible": removed["eligible"],
                               "necessary": bool(drop >= .05 or not removed["eligible"])}
    pred_a = bool(authority_ok and aligned and self_error <= 1e-4 and len(reports) == 32
                  and finite(reports) and forwards == EXACT_FORWARDS)
    pred_b = selected is not None
    pred_c = bool(selected and selected["report"]["targets"]["A2"]["behavior"]["signed_projection"] >= .75
                  and selected["report"]["targets"]["A2"]["behavior"]["direction_fraction"] >= .875)
    pred_d = bool(selected_mask is not None and selected_mask.bit_count() <= 4)
    pred_e = bool(necessity and all(value["necessary"] for value in necessity.values()))
    pred_f = bool(selected and all(
        selected["report"]["targets"][panel]["final_residual"]["signed_projection"] >= .50
        for panel in TARGET_PANELS))
    predictions = {
        "pred_a_authority_population_lattice_closure_finiteness_and_exact_price": pred_a,
        "pred_b_target_sufficient_selective_subset_exists": pred_b,
        "pred_c_selected_subset_confirms_without_reselection": pred_c,
        "pred_d_selected_subset_is_proper": pred_d,
        "pred_e_every_selected_piece_is_necessary": pred_e,
        "pred_f_final_residual_tracks_selected_behavior": pred_f,
    }
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "selective_attention_circuit"
    elif all(predictions[key] for key in list(predictions)[:5]) and not pred_f:
        terminal = "selective_attention_behavior_only"
    elif pred_b and not pred_c:
        terminal = "attention_construction_failure"
    elif pred_b and pred_c and not pred_d:
        terminal = "attention_requires_all_five"
    elif pred_b and pred_c and pred_d and not pred_e:
        terminal = "attention_subset_redundant"
    elif not pred_b:
        terminal = "no_selective_attention_subset"
    else:
        terminal = "partial"
    result = {
        "schema": "temporal_iswas_v15_low_collateral_attention_lattice_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "population": {"panel_counts": panel_counts, "row_ids": [row["row_id"] for row in rows]},
        "components": list(COMPONENTS), "instrument": {"aligned_shapes": aligned,
            "base_self_max_abs_error": self_error}, "reports": reports,
        "eligible_masks": eligible_masks, "selected_mask": selected_mask,
        "selected": selected, "necessity": necessity,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards_exact": EXACT_FORWARDS, "model_forwards_observed": forwards,
                  "example_evaluations": forwards * len(rows), "fit_updates": 0,
                  "model_updates": 0, "transformer_backwards": 0}, "dryrun": dryrun,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "eligible_masks", "selected_mask", "selected", "necessity", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
