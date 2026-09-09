#!/usr/bin/env python3
"""Binding-ready, unfiltered v24 OOD confirmation of the fixed four-head circuit."""

# BQGATE: EXPERIMENT
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v24 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1 as reference

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v24_unfiltered_four_head_ood_confirmation_v1.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v24_capability_v1_result.json"
BINDING = ROOT / "circuits/bindings/temporal_iswas_v24_unfiltered_four_head_ood_confirmation_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v24.py"
CAPABILITY_PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v24_capability_v1.json"
CAPABILITY_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v24_capability_v1.py"
V23_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
REFERENCE = ROOT / "ops/run_temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v24_unfiltered_four_head_ood_confirmation_v1_result.json"
EXPECTED = {
    "prior": "efa80859663097b0d6cece09b3e52290f07be30f0d84bb1a7d7ac7c598646999",
    "builder": "240a1685d8e7190ee6803268b1b2e7c910891c29135583f2ae4b14eb3d19f1a0",
    "capability_prior": "ffa818d105793a63d8a6f2bedf3e240ae3751e584ad5098d8dfa7e8685e2604a",
    "capability_runner": "6c11f38c6cec2c4f9cfae7180b39e0b1782ee0c10f60d94c9e03dcdfa20db210",
    "v23_result": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
    "reference": "d1e8bf0237b33e42bda11dc877cf6ef7315cd24ae00a93e31c0d62821f152d5a",
}
ROWS_SHA256 = "870c829290e1791351d0b2b67985aa5700780920b14423906e7b8fd4d35ed2de"
CANDIDATE_ID = "cross_task.temporal_iswas.v24_unfiltered_four_head_ood_confirmation_v1"
ROUTES = ("L9H1", "L9H4", "L8H1", "L11H3")
PANELS = ("A1", "A2", "P", "C")
BARS = {
    "closure": 1e-4, "hidden_closure": 1e-4,
    "union_recovery": .50, "union_cosine": .90, "union_direction": .90,
    "union_control": .15, "singleton_behavior": .15, "singleton_q": .05,
    "singleton_direction": .75, "singleton_control": .10,
    "singleton_count": 3, "half_recovery": .35, "half_direction": .75,
}
PRICE = {"checkpoint_loads": 1, "model_forwards_exact": 9,
         "sequence_evaluations_exact": 576, "transformer_backwards": 1,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_capability_alignment_closures_finiteness_and_exact_price",
    "pred_b_union_selectively_transfers_to_new_constructions",
    "pred_c_at_least_three_singletons_transfer_selectively",
    "pred_d_transfer_is_stable_across_frozen_reporter_halves",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def eligibility(binding, capability):
    return bool(
        binding.get("schema") == "temporal_iswas_v24_ood_confirmation_binding_v1"
        and binding.get("capability_result_sha256") == sha(CAPABILITY)
        and binding.get("capability_runner_sha256") == EXPECTED["capability_runner"]
        and binding.get("confirmation_runner_sha256") == sha(SELF)
        and capability.get("terminal") == "screen"
        and all(capability.get("predictions", {}).values())
        and capability.get("causal_outcomes_opened") is False
        and capability.get("rows_sha256") == ROWS_SHA256
        and set(capability.get("jointly_capable_row_ids", {})) == set(PANELS)
    )


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "capability_prior": CAPABILITY_PRIOR,
             "capability_runner": CAPABILITY_RUNNER, "v23_result": V23_RESULT,
             "reference": REFERENCE}
    observed = {name: sha(path) for name, path in paths.items()}
    rows = fresh.build_rows()
    counts = {panel: sum(row["family"] == panel for row in rows) for panel in PANELS}
    aligned = all(row["base_semantic_position"] == row["donor_semantic_position"]
                  and len(row["base_ids"]) == len(row["donor_ids"]) for row in rows)
    prebound_authority = bool(observed == EXPECTED and fresh.authority_sha256() == ROWS_SHA256
                              and counts == {panel: 16 for panel in PANELS} and aligned)
    awaiting = not CAPABILITY.exists() or not BINDING.exists()
    capability = json.loads(CAPABILITY.read_text()) if CAPABILITY.exists() else {}
    bound = bool(not awaiting and eligibility(
        json.loads(BINDING.read_text()), capability
    ))
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False,
        "prebound_authority_ok": prebound_authority,
        "status": "bound" if bound else "awaiting_hash_bound_v24_capability_result",
        "rows": counts, "target_rows_unfiltered": 32, "capability_row_filter_used": False,
        "routes": list(ROUTES), "arms": ["self", "all_sites_donor", "union", *ROUTES],
        "bars": BARS, "price": PRICE,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if not prebound_authority:
        raise RuntimeError("v24 OOD prebound authority changed")
    if awaiting or not bound:
        raise RuntimeError("v24 capability result is not hash-bound; GPU access forbidden")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    shared = reference.shared
    base_logits, lengths, base_final, base_cache, base_saved, _ = shared.capture_native(
        backend, base_batch, True
    )
    base_margin = shared.margin(torch, base_logits, lengths, rows, backend.device)
    base_margin.sum().backward()
    if base_saved["leaf"].grad is None:
        raise RuntimeError("frozen native reader gradient missing")
    reader = (base_saved["leaf"].grad.detach().float()
              @ backend.model.transformer.h[11].mlp.Down.weight.detach().float())
    with torch.no_grad():
        donor_logits, donor_lengths, donor_final, donor_cache, donor_saved, _ = shared.capture_native(
            backend, donor_batch, False
        )
        donor_margin = shared.margin(torch, donor_logits, donor_lengths, rows, backend.device)
        base_h = shared.hidden(backend.model, base_saved["x"])
        donor_h = shared.hidden(backend.model, donor_saved["x"])
        mask = torch.zeros(base_h.shape[:2], dtype=torch.bool, device=backend.device)
        for index, query in enumerate(base_batch.semantic_positions):
            mask[index, :int(query) + 1] = True
        native_q = ((donor_h - base_h) * reader * mask.unsqueeze(-1)).sum(dim=(1, 2))
        native_behavior = donor_margin - base_margin.detach()
        target_rows = [index for index, row in enumerate(rows)
                       if row["family"] in ("A1", "A2")]
        target_idx = torch.as_tensor(target_rows, device=backend.device)
        target_behavior, target_q = native_behavior[target_idx], native_q[target_idx]
        control_idx = {
            panel: torch.as_tensor([index for index, row in enumerate(rows)
                                    if row["family"] == panel], device=backend.device)
            for panel in ("P", "C")
        }
        half_idx = {
            name: torch.as_tensor([index for index in target_rows
                                   if int(rows[index]["group_number"]) % 4 in residues],
                                  device=backend.device)
            for name, residues in (("first", (0, 1)), ("second", (2, 3)))
        }

        def report(logits, run_lengths, x):
            behavior = (shared.margin(torch, logits, run_lengths, rows, backend.device)
                        - base_margin.detach())
            q = ((shared.hidden(backend.model, x) - base_h)
                 * reader * mask.unsqueeze(-1)).sum(dim=(1, 2))
            value = {
                "target": {"behavior": shared.metrics(torch, behavior[target_idx], target_behavior),
                           "q": shared.metrics(torch, q[target_idx], target_q)},
                "controls": {}, "halves": {},
            }
            for panel, selected in control_idx.items():
                value["controls"][panel] = {
                    "behavior_leak_ratio": shared.rms_ratio(behavior[selected], target_behavior),
                    "q_leak_ratio": shared.rms_ratio(q[selected], target_q),
                }
            for name, selected in half_idx.items():
                value["halves"][name] = {
                    "behavior": shared.metrics(torch, behavior[selected], native_behavior[selected]),
                    "q": shared.metrics(torch, q[selected], native_q[selected]),
                }
            return value

        self_logits, self_lengths, self_final, self_x = shared.run_patch(
            backend, base_batch, base_cache, all_sites=True
        )
        donor_patch_logits, donor_patch_lengths, donor_patch_final, donor_patch_x = shared.run_patch(
            backend, base_batch, donor_cache, all_sites=True
        )
        self_error = max(
            float((self_final - base_final.detach()).abs().max()),
            float((shared.margin(torch, self_logits, self_lengths, rows, backend.device)
                   - base_margin.detach()).abs().max()),
            float((self_x - base_saved["x"]).abs().max()),
        )
        donor_error = max(
            float((donor_patch_final - donor_final).abs().max()),
            float((shared.margin(torch, donor_patch_logits, donor_patch_lengths, rows, backend.device)
                   - donor_margin).abs().max()),
            float((donor_patch_x - donor_saved["x"]).abs().max()),
        )
        hidden_error = float(
            torch.linalg.vector_norm(shared.hidden(backend.model, donor_patch_x) - donor_h)
            / torch.linalg.vector_norm(donor_h).clamp_min(1e-30)
        )
        reports = {}
        for label, routes in (("union", ROUTES), *((route, (route,)) for route in ROUTES)):
            logits, run_lengths, _, x = shared.run_patch(
                backend, base_batch, donor_cache, routes, False
            )
            reports[label] = report(logits, run_lengths, x)

    union = reports["union"]
    singletons = {route: reports[route] for route in ROUTES}
    singleton_pass = {
        route: bool(value["target"]["behavior"]["signed_projection"] >= BARS["singleton_behavior"]
                    and value["target"]["q"]["signed_projection"] >= BARS["singleton_q"]
                    and value["target"]["behavior"]["direction_fraction"] >= BARS["singleton_direction"]
                    and value["target"]["q"]["direction_fraction"] >= BARS["singleton_direction"]
                    and all(value["controls"][panel][metric] <= BARS["singleton_control"]
                            for panel in ("P", "C")
                            for metric in ("behavior_leak_ratio", "q_leak_ratio")))
        for route, value in singletons.items()
    }
    A = bool(float(reader.abs().max()) > 0 and self_error <= BARS["closure"]
             and donor_error <= BARS["closure"] and hidden_error <= BARS["hidden_closure"]
             and shared.finite(reports) and len(target_rows) == 32)
    B = bool(
        all(union["target"][kind]["signed_projection"] >= BARS["union_recovery"]
            and union["target"][kind]["cosine"] >= BARS["union_cosine"]
            and union["target"][kind]["direction_fraction"] >= BARS["union_direction"]
            for kind in ("behavior", "q"))
        and all(union["controls"][panel][metric] <= BARS["union_control"]
                for panel in ("P", "C")
                for metric in ("behavior_leak_ratio", "q_leak_ratio"))
    )
    C = sum(singleton_pass.values()) >= BARS["singleton_count"]
    D = all(union["halves"][half][kind]["signed_projection"] >= BARS["half_recovery"]
            and union["halves"][half][kind]["direction_fraction"] >= BARS["half_direction"]
            for half in ("first", "second") for kind in ("behavior", "q"))
    forwards = 9
    A = bool(A and forwards == PRICE["model_forwards_exact"]
             and forwards * len(rows) == PRICE["sequence_evaluations_exact"])
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A else
                "confirmed_new_construction_ood_four_head_program" if all((B, C, D)) else
                "valid_new_construction_ood_transfer_failure")
    result = {
        "schema": "temporal_iswas_v24_unfiltered_four_head_ood_confirmation_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": {**observed, "capability_result": sha(CAPABILITY),
                              "binding": sha(BINDING)},
        "population": {"counts": counts, "target_rows_unfiltered": len(target_rows),
                       "capability_row_filter_used": False},
        "alignment_all_rows": aligned, "routes": list(ROUTES),
        "reader_gradient_max_abs": float(reader.abs().max()),
        "closures": {"self_max_abs": self_error, "donor_max_abs": donor_error,
                     "donor_hidden_relative": hidden_error},
        "union_report": union, "singleton_reports": singletons,
        "singleton_pass": singleton_pass, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE,
        "model_forwards": forwards, "searched_route_rank_or_dose": False,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "population", "closures", "union_report", "singleton_pass",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
