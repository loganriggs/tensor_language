#!/usr/bin/env python3
"""Frozen-axis target audit for the fresh temporal block11H3 DAS result."""

# BQGATE: EXPERIMENT pred_a_authority_axis_and_full_rank_closure pred_b_parent_scalar_aggregates_reproduce pred_c_cdas_scalar_objective_dominates_dim pred_d_cdas_full_vocabulary_objective_dominates_dim pred_e_no_signed_cancellation
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_unit_greedy as g


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_block11h3_cdas_target_redteam_v1.json"
PARENT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_single_component_cdas_v1_result.json"
AXIS = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_cdas_axis_necessity_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
UNIT_LIB = ROOT / "ops/circuit_unit_greedy.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_cdas_target_redteam_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_block11h3_cdas_target_redteam_v1"
UNIT = ("attn:11:head:03",)
EXPECTED = {
    "prior": "2b8c773433d07fdd46582666ee8437448d14dd04af3490f8afba1f706e2a9427",
    "parent": "5fba2b88bd7a6ec9883ee624e05495600e31258defcc355a22ab219ba9734b78",
    "axis": "4f345907c41222ebeec33c3a860052a0a8f39166655da0354f69e79a1c577fc5",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "unit_lib": "b7bc7def571e04cadce38c90f978219bc95c0f8ddaa3ef941c3309a4b913e1a7",
}
MODEL_FORWARDS, EXAMPLE_EVALUATIONS = 20, 464


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "parent": PARENT, "axis": AXIS,
             "builder": BUILDER, "unit_lib": UNIT_LIB}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    prior, parent, axis = [json.loads(path.read_text()) for path in (PRIOR, PARENT, AXIS)]
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen"
            or axis.get("terminal") != "screen" or not all(parent["predictions"].values())):
        raise ExperimentError("authority terminal changed")
    coordinates = axis["axis_artifact"]["coordinates"]
    if len(coordinates) != 128:
        raise ExperimentError("stored axis dimension changed")
    rows = candidate.build_rows()
    counts = {family: sum(row["transform_id"] == family for row in rows)
              for family in ("A1", "A2", "P", "C")}
    if counts != {"A1": 32, "A2": 32, "P": 32, "C": 32}:
        raise ExperimentError("population changed")
    return rows, parent, coordinates


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "rank": 1,
            "fit_updates": 0, "model_updates": 0, "model_forwards": MODEL_FORWARDS,
            "example_evaluations": EXAMPLE_EVALUATIONS,
            "arms": ["base", "exact", "cdas", "cdas_complement", "dim", "dim_complement"]}


def full_logits(backend, prep, *, q=None, complement=False, exact=False):
    """Capture the already-computed full vocabulary logits without changing the forward."""
    torch = backend.torch
    captured = []

    def hook(_module, _arguments, output):
        captured.append(output)

    handle = backend.model.lm_head.register_forward_hook(hook)
    try:
        kwargs = {}
        if exact or q is not None:
            kwargs = {"units": UNIT, "donor_cache": prep.donor_cache,
                      "base_cache": prep.base_cache, "q": q, "complement": complement}
        answer_foil = g.forward_units(backend, prep.base_batch, **kwargs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ExperimentError("lm_head hook did not fire exactly once")
    _tokens, lengths = backend._tensor_batch(prep.base_batch)
    index = torch.arange(len(lengths), device=backend.device)
    position = torch.tensor([length - 1 for length in lengths], device=backend.device)
    logits = 30.0 * torch.tanh(captured[0][index, position].float() / 30.0)
    return answer_foil.detach(), logits.detach()


def summarize(values):
    torch = __import__("torch")
    values = values.float()
    absolute = values.abs()
    return {"mean_signed": float(values.mean()), "mean_absolute": float(absolute.mean()),
            "rms": float(values.square().mean().sqrt()),
            "p90_absolute": float(torch.quantile(absolute, 0.9)),
            "maximum_absolute": float(absolute.max())}


def scalar_axis(answer_foil):
    return -(answer_foil[:, 0] - answer_foil[:, 1])


def axis_report(prep, outputs, name):
    torch = __import__("torch")
    base = torch.tensor(prep.base_axis, device=outputs["base"][0].device)
    donor = torch.tensor(prep.donor_axis, device=base.device)
    denominator = donor - base
    exact = scalar_axis(outputs["exact"][0])
    sub = scalar_axis(outputs[name][0])
    comp = scalar_axis(outputs[name + "_complement"][0])
    exact_effect = (exact - base) / denominator
    sub_effect = (sub - base) / denominator
    match_error = (sub - exact) / denominator
    comp_effect = (comp - base) / denominator
    signed_exact = float(exact_effect.mean())

    base_logits = outputs["base"][1]
    exact_logits = outputs["exact"][1]
    sub_logits = outputs[name][1]
    comp_logits = outputs[name + "_complement"][1]
    exact_delta = exact_logits - base_logits
    match_delta = sub_logits - exact_logits
    comp_delta = comp_logits - base_logits
    exact_delta = exact_delta - exact_delta.mean(1, keepdim=True)
    match_delta = match_delta - match_delta.mean(1, keepdim=True)
    comp_delta = comp_delta - comp_delta.mean(1, keepdim=True)
    scale = exact_delta.square().mean(1).sqrt()
    if bool((scale <= 1e-8).any()):
        raise ExperimentError("exact H3 full-vocabulary effect is zero")
    vector_match = match_delta.square().mean(1).sqrt() / scale
    vector_comp = comp_delta.square().mean(1).sqrt() / scale
    sub_delta = sub_logits - base_logits
    sub_delta = sub_delta - sub_delta.mean(1, keepdim=True)
    cosine = torch.nn.functional.cosine_similarity(sub_delta, exact_delta, dim=1)
    return {
        "scalar": {
            "exact_effect": summarize(exact_effect),
            "subspace_effect": summarize(sub_effect),
            "match_error": summarize(match_error),
            "complement_effect": summarize(comp_effect),
            "signed_subspace_fraction": float(sub_effect.mean() / exact_effect.mean()),
            "signed_complement_fraction": float(comp_effect.mean() / exact_effect.mean()),
            "joint_squared_objective": float(match_error.square().mean() + comp_effect.square().mean()),
        },
        "full_vocabulary": {
            "exact_effect_rms": summarize(scale),
            "normalized_match_error": summarize(vector_match),
            "normalized_complement_effect": summarize(vector_comp),
            "subspace_exact_cosine": summarize(cosine),
            "joint_squared_objective": float(vector_match.square().mean() + vector_comp.square().mean()),
        },
        "n": int(len(base)),
        "exact_signed_mean": signed_exact,
    }


def evaluate(backend, prep, q_cdas, q_dim):
    identity = backend.torch.eye(128, device=backend.device)
    outputs = {
        "base": full_logits(backend, prep),
        "exact": full_logits(backend, prep, exact=True),
        "identity": full_logits(backend, prep, q=identity),
        "cdas": full_logits(backend, prep, q=q_cdas),
        "cdas_complement": full_logits(backend, prep, q=q_cdas, complement=True),
        "dim": full_logits(backend, prep, q=q_dim),
        "dim_complement": full_logits(backend, prep, q=q_dim, complement=True),
    }
    closure = {
        "max_abs_answer_foil_error": float((outputs["identity"][0] - outputs["exact"][0]).abs().max()),
        "max_abs_full_vocabulary_logit_error": float((outputs["identity"][1] - outputs["exact"][1]).abs().max()),
    }
    return {name: axis_report(prep, outputs, name) for name in ("cdas", "dim")}, closure


def main():
    rows, parent, coordinates = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    family = {name: [row for row in rows if row["transform_id"] == name]
              for name in ("A1", "A2")}
    preps = {"fit": g.prepare(backend, family["A1"][0::2]),
             "heldout": g.prepare(backend, family["A1"][1::2]),
             "a2": g.prepare(backend, family["A2"])}
    q_cdas = torch.tensor(coordinates, device=backend.device).float().unsqueeze(1)
    q_dim = g.diff_in_means_direction(backend, preps["fit"], UNIT)
    axis_norm = float(q_cdas.norm())
    evaluated = {name: evaluate(backend, prep, q_cdas, q_dim)
                 for name, prep in preps.items() if name != "fit"}
    reports = {name: value[0] for name, value in evaluated.items()}
    closure = {name: value[1] for name, value in evaluated.items()}
    parent_metric = {"signed_subspace_fraction": "subspace_fraction",
                     "signed_complement_fraction": "complement_fraction"}
    reproduction_error = max(
        abs(reports[split]["cdas"]["scalar"][metric]
            - parent["reports"]["cdas"][split][parent_metric[metric]])
        for split in ("heldout", "a2")
        for metric in ("signed_subspace_fraction", "signed_complement_fraction")
    )
    pred_a = bool(abs(axis_norm - 1.0) <= 1e-5 and all(
        max(cell.values()) <= 1e-4 for cell in closure.values()))
    pred_b = reproduction_error <= 0.01
    pred_c = all(reports[split]["cdas"]["scalar"]["joint_squared_objective"]
                 <= reports[split]["dim"]["scalar"]["joint_squared_objective"]
                 for split in ("heldout", "a2"))
    pred_d = all(reports[split]["cdas"]["full_vocabulary"]["joint_squared_objective"]
                 <= reports[split]["dim"]["full_vocabulary"]["joint_squared_objective"]
                 for split in ("heldout", "a2"))
    pred_e = all(
        (cell := reports[split]["cdas"]["scalar"]["complement_effect"])["mean_absolute"] <= 0.30
        and cell["mean_absolute"] <= 4.0 * abs(cell["mean_signed"]) + 0.02
        for split in ("heldout", "a2"))
    predictions = {
        "pred_a_authority_axis_and_full_rank_closure": pred_a,
        "pred_b_parent_scalar_aggregates_reproduce": pred_b,
        "pred_c_cdas_scalar_objective_dominates_dim": pred_c,
        "pred_d_cdas_full_vocabulary_objective_dominates_dim": pred_d,
        "pred_e_no_signed_cancellation": pred_e,
    }
    if not pred_a or not pred_b:
        terminal = "invalid"
    elif pred_c and pred_d and pred_e:
        terminal = "screen"
    elif not pred_c:
        terminal = "ordinary_overfit"
    else:
        terminal = "target_specific"
    result = {
        "schema": "temporal_auxiliary_fresh_block11h3_cdas_target_redteam_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "axis_norm": axis_norm, "identity_closure": closure,
        "parent_reproduction_max_abs_error": reproduction_error,
        "reports": reports, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": MODEL_FORWARDS,
                  "example_evaluations": EXAMPLE_EVALUATIONS,
                  "model_updates": 0, "fit_updates": 0},
    }
    finite = [axis_norm, reproduction_error]
    for split in reports.values():
        for method in split.values():
            finite.extend((method["scalar"]["joint_squared_objective"],
                           method["full_vocabulary"]["joint_squared_objective"]))
    if not all(math.isfinite(value) for value in finite):
        raise ExperimentError("nonfinite audit metric")
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal,
                      "predictions": predictions, "reports": reports,
                      "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
