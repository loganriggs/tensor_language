#!/usr/bin/env python3
"""Exact grouped downstream-response factorial under MLP8-complement actuation."""

# BQGATE: EXPERIMENT pred_a_authority_replay_self_clamp_finiteness_and_price pred_b_four_group_program_is_near_complete pred_c_attention_core_is_dominant pred_d_late_bank_is_an_opposing_correction pred_e_zero_fit_exact_group_inventory
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_complement_downstream_converter_atlas_v1 as atlas
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_complement_downstream_group_factorial_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_complement_downstream_converter_atlas_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_complement_downstream_converter_atlas_v1.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_complement_downstream_group_factorial_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_complement_downstream_group_factorial_v1"
EXPECTED = {
    "prior": "ae6fbcc7fd85c239c18f859f5b412797d5ff320a8a268ff22b2119781e03742c",
    "parent": "2ccfb7116e45665820a546ebd5edc2bf4e2616b49cf46723064891640835ac5b",
    "parent_runner": "d757d4c54b53d5cda8d877bfe9a01e636f4459582be6cb379b28f0e8823be218",
}
GROUPS = {
    "core_attention": ("attn:09",),
    "auxiliary_attention": ("attn:11", "attn:15"),
    "positive_mlp_bank": tuple(f"mlp:{layer:02d}" for layer in range(9, 16)),
    "late_mlp_correction": ("mlp:16", "mlp:17"),
}
GROUP_NAMES = tuple(GROUPS)
MAX_FORWARDS, MAX_EVALUATIONS = 22, 638


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subset_name(names) -> str:
    return "+".join(names) if names else "empty"


def group_subsets():
    for width in range(len(GROUP_NAMES) + 1):
        yield from itertools.combinations(GROUP_NAMES, width)


def cosine(x, y) -> float:
    denominator = float(x.norm() * y.norm())
    return float((x * y).sum()) / denominator if denominator else 0.0


def main() -> None:
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("group-factorial authority changed")
    inherited_paths = {
        "prior": atlas.PRIOR, "weight_v2": atlas.WEIGHT_V2,
        "weight_v2_runner": atlas.WEIGHT_V2_RUNNER,
        "weight_instrument": atlas.WEIGHT_INSTRUMENT, "source": atlas.SOURCE,
        "capability": weight.CAPABILITY, "iswas": weight.ISWAS,
        "subspace": weight.SUBSPACE, "builder": weight.BUILDER,
        "family_runner": weight.FAMILY_RUNNER, "overlap_runner": weight.OVERLAP_RUNNER,
    }
    if {name: sha(path) for name, path in inherited_paths.items()} != atlas.EXPECTED:
        raise RuntimeError("inherited converter authority changed")
    prior, parent = json.loads(PRIOR.read_text()), json.loads(PARENT.read_text())
    capability = json.loads(weight.CAPABILITY.read_text())
    subspace = json.loads(weight.SUBSPACE.read_text())
    capable = {}
    for record in capability["native_records"]:
        capable.setdefault(record["row_id"], {})[record["side"]] = bool(record["correct"])
    allowed = {row_id for row_id, sides in capable.items()
               if sides == {"base": True, "donor": True}}
    rows = [row for row in weight.candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions = [weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen"
            or len(rows) != 29 or len(tuple(group_subsets())) != 16
            or any(site not in atlas.SITES for sites in GROUPS.values() for site in sites)):
        raise RuntimeError("parent terminal, population, or group inventory changed")
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "groups": GROUPS, "subsets": [subset_name(value) for value in group_subsets()],
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, family[8], gain)
    s = torch.linalg.qr(modes, mode="reduced").Q
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, base_capture = weight.capture_mlp8(backend, base_batch)
    _donor_output, donor_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T @ down, full_matrices=False)
    full_delta = donor_capture["hidden"].float() - base_capture["hidden"].float()
    complement = full_delta - weight.project(full_delta, vh, vh.shape[0])
    base_modules_output, base_modules = atlas.capture_modules(backend, base_batch)
    live_output, _live_modules = atlas.capture_modules(
        backend, base_batch, base_capture["hidden"], complement, positions)

    outputs = {}
    for subset in group_subsets():
        sites = tuple(site for name in subset for site in GROUPS[name])
        outputs[subset_name(subset)] = atlas.run_clamped(
            backend, base_batch, base_capture["hidden"], complement, positions,
            base_modules, sites)
    outputs["complete_18_sites"] = atlas.run_clamped(
        backend, base_batch, base_capture["hidden"], complement, positions,
        base_modules, atlas.SITES)
    self_output = atlas.run_clamped(
        backend, base_batch, base_capture["hidden"], complement, positions,
        base_modules, atlas.SITES, actuate=False)
    forwards, evaluations = 22, 22 * len(rows)
    state = lambda output: atlas.state(output, rows, torch, backend.device)
    base18, base_modules18, live18, self18 = map(
        state, (base_output, base_modules_output, live_output, self_output))
    states = {name: state(output) for name, output in outputs.items()}
    index = torch.arange(len(rows), device=backend.device)
    answers = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foils = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    margin = lambda value: das.head_logits(backend, value)[index, answers] - das.head_logits(backend, value)[index, foils]
    base_margin, live_margin = margin(base18), margin(live18)
    clamped_margins = {name: margin(value) for name, value in states.items()}
    live_effect, live_coord = live_margin - base_margin, (live18 - base18) @ s
    masks = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)
             for panel in ("A1", "A2")}
    metrics = {panel: {} for panel in masks}
    for panel, mask in masks.items():
        for name, value in states.items():
            removed_effect = (live_margin - clamped_margins[name])[mask]
            removed_coord = ((live18 - value) @ s)[mask]
            metrics[panel][name] = {
                "signed_behavior_fraction": float(removed_effect.mean() / live_effect[mask].mean()),
                "absolute_behavior_fraction": float(removed_effect.abs().mean() / live_effect[mask].abs().mean()),
                "behavior_cosine": cosine(removed_effect, live_effect[mask]),
                "q8_norm_fraction": float(removed_coord.norm() / live_coord[mask].norm()),
                "q8_cosine": cosine(removed_coord.reshape(-1), live_coord[mask].reshape(-1)),
            }
        full_name = subset_name(GROUP_NAMES)
        prelate_name = subset_name(GROUP_NAMES[:-1])
        late_effect = (clamped_margins[prelate_name] - clamped_margins[full_name])[mask]
        late_coord = ((states[prelate_name] - states[full_name]) @ s)[mask]
        singleton_effects = sum((live_margin - clamped_margins[name])[mask] for name in GROUP_NAMES)
        union_effect = (live_margin - clamped_margins[full_name])[mask]
        metrics[panel]["late_conditional_marginal"] = {
            "signed_behavior_fraction": float(late_effect.mean() / live_effect[mask].mean()),
            "absolute_behavior_fraction": float(late_effect.abs().mean() / live_effect[mask].abs().mean()),
            "q8_norm_fraction": float(late_coord.norm() / live_coord[mask].norm()),
        }
        metrics[panel]["group_nonadditivity"] = {
            "behavior_relative_rms": float((union_effect - singleton_effects).norm() / live_effect[mask].norm())
        }

    full_name = subset_name(GROUP_NAMES)
    core_name = "core_attention"
    identity_error = max(float((base_modules18 - base18).abs().max()),
                         float((self18 - base18).abs().max()))
    empty_error = float((states["empty"] - live18).abs().max())
    complete_replay = max(abs(metrics[p]["complete_18_sites"][key] - parent["metrics"][p]["all"][key])
                          for p in ("A1", "A2")
                          for key in ("absolute_behavior_fraction", "q8_norm_fraction"))
    finite = all(math.isfinite(value) for panel in metrics.values()
                 for arm in panel.values() for value in arm.values())
    pred_a = bool(orientation_error <= 1e-6 and identity_error <= 1e-4
                  and empty_error <= 1e-4 and complete_replay <= 1e-5 and finite
                  and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS)
    pred_b = all(metrics[p][full_name][key] >= threshold for p in ("A1", "A2")
                 for key, threshold in (("absolute_behavior_fraction", .95),
                     ("q8_norm_fraction", .90), ("behavior_cosine", .95), ("q8_cosine", .95)))
    pred_c = all(metrics[p][core_name]["absolute_behavior_fraction"] >= .75
                 and metrics[p][core_name]["q8_norm_fraction"] >= .75 for p in ("A1", "A2"))
    late = [metrics[p]["late_conditional_marginal"] for p in ("A1", "A2")]
    pred_d = all(value["signed_behavior_fraction"] < 0 for value in late) and any(
        value["absolute_behavior_fraction"] >= .03 for value in late)
    pred_e = len(outputs) == 17
    predictions = {
        "pred_a_authority_replay_self_clamp_finiteness_and_price": pred_a,
        "pred_b_four_group_program_is_near_complete": pred_b,
        "pred_c_attention_core_is_dominant": pred_c,
        "pred_d_late_bank_is_an_opposing_correction": pred_d,
        "pred_e_zero_fit_exact_group_inventory": pred_e,
    }
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {
        "schema": "iswas_mlp8_complement_downstream_group_factorial_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": {**EXPECTED, **{f"inherited_{name}": value for name, value in atlas.EXPECTED.items()}},
        "dryrun": dryrun,
        "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_base_self_clamp_resid18_max_abs": identity_error,
            "empty_subset_live_replay_max_abs": empty_error,
            "complete_atlas_metric_replay_max_abs": complete_replay,
            "rows": len(rows)},
        "groups": GROUPS, "metrics": metrics, "predictions": predictions,
        "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "groups",
          "metrics", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
