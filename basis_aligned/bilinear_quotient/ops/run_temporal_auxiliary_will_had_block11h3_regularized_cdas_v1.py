#!/usr/bin/env python3
"""Separate KL target regularization from tangent-noise regularization for cDAS."""

# BQGATE: EXPERIMENT pred_a_authority_closure_baseline_reproduction_and_price pred_b_kl_noise_repairs_full_vocabulary_fidelity pred_c_kl_noise_retains_scalar_advantage pred_d_noise_alone_reduces_readout_overfit pred_e_regularized_restarts_are_identifiable
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
import run_temporal_auxiliary_will_had_fresh_block11h3_cdas_target_redteam_v1 as redteam
import single_component_das_eval as single


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_block11h3_regularized_cdas_v1.json"
REDTEAM = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_cdas_target_redteam_v1_result.json"
AXIS = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_block11h3_cdas_axis_necessity_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
UNIT_LIB = ROOT / "ops/circuit_unit_greedy.py"
SINGLE_LIB = ROOT / "ops/single_component_das_eval.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_regularized_cdas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.block11h3_regularized_cdas_v1"
UNIT = ("attn:11:head:03",)
EXPECTED = {
    "prior": "c2d5c598d3854630cfcfd5f0e4f586b4a733388ef995ae11de2a765d11a6b35f",
    "redteam": "7382cf43bc4564b5d71bbb2bed70856bd1d44e84efc11c87ecaccf70c818c19d",
    "axis": "4f345907c41222ebeec33c3a860052a0a8f39166655da0354f69e79a1c577fc5",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "unit_lib": "b7bc7def571e04cadce38c90f978219bc95c0f8ddaa3ef941c3309a4b913e1a7",
    "single_lib": "363569c1b1cf20e4f31a4569d2467b4c86a6563405a49766af968037e12028b8",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
VARIANTS = {
    "kl": {"kl_weight": 1.0, "noise_sigma": 0.0},
    "noise": {"kl_weight": 0.0, "noise_sigma": 0.05},
    "kl_noise": {"kl_weight": 1.0, "noise_sigma": 0.05},
}
STEPS, LR, SEEDS = 100, 0.03, (1, 2)
FORWARDS, EVALUATIONS = 3233, 51968
BACKWARD_FORWARDS, UPDATES = 3000, 900


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "redteam": REDTEAM, "axis": AXIS, "builder": BUILDER,
             "unit_lib": UNIT_LIB, "single_lib": SINGLE_LIB, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    prior, old, axis = [json.loads(path.read_text()) for path in (PRIOR, REDTEAM, AXIS)]
    rows = candidate.build_rows()
    counts = {family: sum(row["transform_id"] == family for row in rows)
              for family in ("A1", "A2", "P", "C")}
    if (prior.get("candidate_id") != CANDIDATE_ID or old.get("terminal") != "target_specific"
            or axis.get("terminal") != "screen" or len(axis["axis_artifact"]["coordinates"]) != 128
            or counts != {"A1": 32, "A2": 32, "P": 32, "C": 32}
            or single.validate_units(UNIT) != (11, "head")):
        raise ExperimentError("candidate, authority terminal, axis, or population changed")
    return rows, old, axis["axis_artifact"]["coordinates"]


def full_forward(backend, prep, *, q=None, complement=False, exact=False, grad=False):
    """Return answer/foil and full soft-capped logits, retaining graph when requested."""
    torch = backend.torch
    captured = []

    def hook(_module, _arguments, output):
        captured.append(output)

    handle = backend.model.lm_head.register_forward_hook(hook)
    try:
        kwargs = {"grad": grad}
        if exact or q is not None:
            kwargs.update({"units": UNIT, "donor_cache": prep.donor_cache,
                           "base_cache": prep.base_cache, "q": q,
                           "complement": complement})
        answer_foil = g.forward_units(backend, prep.base_batch, **kwargs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ExperimentError("lm_head hook coverage changed")
    _tokens, lengths = backend._tensor_batch(prep.base_batch)
    index = torch.arange(len(lengths), device=backend.device)
    position = torch.tensor([length - 1 for length in lengths], device=backend.device)
    logits = 30.0 * torch.tanh(captured[0][index, position].float() / 30.0)
    if not grad:
        answer_foil, logits = answer_foil.detach(), logits.detach()
    return answer_foil, logits


def scalar_axis(answer_foil):
    return -(answer_foil[:, 0] - answer_foil[:, 1])


def fit_targets(backend, prep):
    base = full_forward(backend, prep)
    exact = full_forward(backend, prep, exact=True)
    identity = full_forward(backend, prep, q=backend.torch.eye(128, device=backend.device))
    base_axis = backend.torch.tensor(prep.base_axis, device=backend.device)
    donor_axis = backend.torch.tensor(prep.donor_axis, device=backend.device)
    denominator = donor_axis - base_axis
    if bool((denominator <= 1e-6).any()):
        raise ExperimentError("fit denominator is nonpositive")
    F = backend.F
    log_base = F.log_softmax(base[1], dim=-1)
    log_exact = F.log_softmax(exact[1], dim=-1)
    reference_kl = F.kl_div(log_base, log_exact, log_target=True, reduction="batchmean")
    if not math.isfinite(float(reference_kl)) or float(reference_kl) <= 1e-8:
        raise ExperimentError("full-vocabulary reference KL is ill-conditioned")
    closure = {"answer_foil_max_abs": float((identity[0] - exact[0]).abs().max()),
               "full_vocabulary_max_abs": float((identity[1] - exact[1]).abs().max())}
    return {"base": base, "exact": exact, "base_axis": base_axis,
            "exact_axis": scalar_axis(exact[0]), "denominator": denominator,
            "log_base": log_base.detach(), "log_exact": log_exact.detach(),
            "reference_kl": reference_kl.detach(), "closure": closure}


def objective(backend, prep, targets, q, kl_weight):
    F = backend.F
    sub = full_forward(backend, prep, q=q, grad=True)
    comp = full_forward(backend, prep, q=q, complement=True, grad=True)
    match = ((((scalar_axis(sub[0]) - targets["exact_axis"])
               / targets["denominator"]) ** 2).mean())
    inert = ((((scalar_axis(comp[0]) - targets["base_axis"])
               / targets["denominator"]) ** 2).mean())
    scalar = match + inert
    log_sub = F.log_softmax(sub[1], dim=-1)
    log_comp = F.log_softmax(comp[1], dim=-1)
    kl_match = F.kl_div(log_sub, targets["log_exact"], log_target=True,
                        reduction="batchmean") / targets["reference_kl"]
    kl_inert = F.kl_div(log_comp, targets["log_base"], log_target=True,
                        reduction="batchmean") / targets["reference_kl"]
    full = kl_match + kl_inert
    return scalar + kl_weight * full, {"scalar": scalar, "match": match, "inert": inert,
                                      "full_kl": full, "kl_match": kl_match,
                                      "kl_inert": kl_inert}


def tangent_pair(torch, coefficients, sigma, noise):
    coordinate = coefficients / coefficients.norm().clamp_min(1e-30)
    tangent = noise - coordinate * (coordinate.T @ noise)
    tangent = tangent / tangent.norm().clamp_min(1e-30)
    return tuple((coordinate + sign * sigma * tangent)
                 / (coordinate + sign * sigma * tangent).norm().clamp_min(1e-30)
                 for sign in (1.0, -1.0))


def fit_variant(backend, prep, targets, span, q_dim, *, name, kl_weight, noise_sigma):
    torch = backend.torch
    starts = [("dim", span @ q_dim), *[(f"random_seed_{seed}", None) for seed in SEEDS]]
    global_best = None
    restart_reports = []
    restart_axes = []
    for start_name, initial in starts:
        seed = 0 if start_name == "dim" else int(start_name.rsplit("_", 1)[1])
        generator = torch.Generator(device="cpu").manual_seed(7300 + seed)
        if initial is None:
            initial = torch.randn(span.shape[0], 1, generator=generator).to(backend.device)
        raw = initial.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([raw], lr=LR)
        best = None
        trace = []

        def checkpoint(step):
            nonlocal best
            with torch.no_grad():
                coordinate = raw / raw.norm().clamp_min(1e-30)
                q = span.T @ coordinate
                loss, pieces = objective(backend, prep, targets, q, kl_weight)
            report = {"step": step, "joint": float(loss),
                      **{key: float(value) for key, value in pieces.items()}}
            trace.append(report)
            if best is None or report["joint"] < best[0]["joint"]:
                best = (report, q.detach().clone())

        checkpoint(0)
        for update in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            coordinate = raw / raw.norm().clamp_min(1e-30)
            if noise_sigma > 0:
                noise = torch.randn(raw.shape, generator=generator).to(backend.device)
                perturbed = tangent_pair(torch, coordinate, noise_sigma, noise)
                losses = [objective(backend, prep, targets, span.T @ value, kl_weight)[0]
                          for value in perturbed]
                loss = sum(losses) / len(losses)
            else:
                loss = objective(backend, prep, targets, span.T @ coordinate, kl_weight)[0]
            loss.backward()
            optimizer.step()
            if (update + 1) % 10 == 0 or update + 1 == STEPS:
                checkpoint(update + 1)
        restart_reports.append({"start": start_name, "best": best[0], "trace": trace})
        restart_axes.append(best[1])
        if global_best is None or best[0]["joint"] < global_best[0]["joint"]:
            global_best = (best[0], best[1], start_name)
    pairwise = [float((restart_axes[i][:, 0] @ restart_axes[j][:, 0]).abs())
                for i in range(len(restart_axes)) for j in range(i + 1, len(restart_axes))]
    return {"name": name, "q": global_best[1], "best": global_best[0],
            "selected_start": global_best[2], "restarts": restart_reports,
            "restart_min_pairwise_cosine": min(pairwise),
            "coordinates": global_best[1][:, 0].detach().cpu().tolist()}


def evaluate_axes(backend, prep, axes):
    identity = backend.torch.eye(128, device=backend.device)
    outputs = {"base": full_forward(backend, prep),
               "exact": full_forward(backend, prep, exact=True),
               "identity": full_forward(backend, prep, q=identity)}
    for name, q in axes.items():
        outputs[name] = full_forward(backend, prep, q=q)
        outputs[name + "_complement"] = full_forward(backend, prep, q=q, complement=True)
    closure = {"answer_foil_max_abs": float((outputs["identity"][0] - outputs["exact"][0]).abs().max()),
               "full_vocabulary_max_abs": float((outputs["identity"][1] - outputs["exact"][1]).abs().max())}
    return {name: redteam.axis_report(prep, outputs, name) for name in axes}, closure


def main():
    rows, old, old_coordinates = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "rank": 1,
              "variants": VARIANTS, "starts": 3, "updates_per_start": STEPS,
              "model_forwards": FORWARDS, "example_evaluations": EVALUATIONS,
              "transformer_backward_forwards": BACKWARD_FORWARDS,
              "model_updates": UPDATES, "fit_parameters_max": 16}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    family = {name: [row for row in rows if row["transform_id"] == name]
              for name in ("A1", "A2")}
    preps = {"fit": g.prepare(backend, family["A1"][0::2]),
             "heldout": g.prepare(backend, family["A1"][1::2]),
             "a2": g.prepare(backend, family["A2"])}
    targets = fit_targets(backend, preps["fit"])
    delta = single.cached_delta_matrix(backend, preps["fit"], UNIT)
    span, singular, span_rank = single.empirical_span(delta)
    q_dim = g.diff_in_means_direction(backend, preps["fit"], UNIT)
    fits = {name: fit_variant(backend, preps["fit"], targets, span, q_dim,
                             name=name, **configuration)
            for name, configuration in VARIANTS.items()}
    q_old = backend.torch.tensor(old_coordinates, device=backend.device).float().unsqueeze(1)
    axes = {"dim": q_dim, "unregularized": q_old,
            **{name: fit["q"] for name, fit in fits.items()}}
    evaluated = {split: evaluate_axes(backend, preps[split], axes)
                 for split in ("heldout", "a2")}
    reports = {split: value[0] for split, value in evaluated.items()}
    closures = {"fit": targets["closure"],
                **{split: value[1] for split, value in evaluated.items()}}
    reproduction = max(abs(reports[split][name][scope]["joint_squared_objective"]
                           - old["reports"][split][old_name][scope]["joint_squared_objective"])
                       for split in ("heldout", "a2")
                       for name, old_name in (("dim", "dim"), ("unregularized", "cdas"))
                       for scope in ("scalar", "full_vocabulary"))
    all_fit_values = [value for fit in fits.values() for restart in fit["restarts"]
                      for point in restart["trace"] for value in point.values()
                      if isinstance(value, (int, float))]
    price = {"model_forwards": FORWARDS, "example_evaluations": EVALUATIONS,
             "transformer_backward_forwards": BACKWARD_FORWARDS,
             "model_updates": UPDATES, "fit_parameters": span_rank}
    pred_a = bool(max(value for closure in closures.values() for value in closure.values()) <= 1e-4
                  and reproduction <= 0.01 and all(math.isfinite(value) for value in all_fit_values)
                  and price == {"model_forwards": 3233, "example_evaluations": 51968,
                                "transformer_backward_forwards": 3000,
                                "model_updates": 900, "fit_parameters": span_rank})
    pred_b = all(reports[split]["kl_noise"]["full_vocabulary"]["joint_squared_objective"]
                 < min(reports[split][baseline]["full_vocabulary"]["joint_squared_objective"]
                       for baseline in ("unregularized", "dim")) for split in ("heldout", "a2"))
    pred_c = all(reports[split]["kl_noise"]["scalar"]["joint_squared_objective"]
                 < reports[split]["dim"]["scalar"]["joint_squared_objective"]
                 for split in ("heldout", "a2"))
    pred_d = all(reports[split]["noise"]["full_vocabulary"]["joint_squared_objective"]
                 <= 0.90 * reports[split]["unregularized"]["full_vocabulary"]["joint_squared_objective"]
                 for split in ("heldout", "a2"))
    pred_e = all(fit["restart_min_pairwise_cosine"] >= 0.80 for fit in fits.values())
    predictions = {
        "pred_a_authority_closure_baseline_reproduction_and_price": pred_a,
        "pred_b_kl_noise_repairs_full_vocabulary_fidelity": pred_b,
        "pred_c_kl_noise_retains_scalar_advantage": pred_c,
        "pred_d_noise_alone_reduces_readout_overfit": pred_d,
        "pred_e_regularized_restarts_are_identifiable": pred_e,
    }
    if not pred_a:
        terminal = "invalid"
    elif all(predictions.values()):
        terminal = "screen"
    elif all((pred_a, pred_b, pred_c, pred_e)) and not pred_d:
        terminal = "objective_regularized"
    elif all((pred_a, pred_c, pred_d, pred_e)) and not pred_b:
        terminal = "stability_regularized"
    else:
        terminal = "null"
    result = {
        "schema": "temporal_auxiliary_block11h3_regularized_cdas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "empirical_span": {"rank": span_rank,
            "singular_values": [float(value) for value in singular.detach().cpu()]},
        "identity_closure": closures, "baseline_reproduction_max_abs_error": reproduction,
        "fits": {name: {key: value for key, value in fit.items() if key != "q"}
                 for name, fit in fits.items()},
        "reports": reports, "predictions": predictions, "price": price,
        "terminal": terminal,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "identity_closure",
          "baseline_reproduction_max_abs_error", "fits", "reports", "predictions",
          "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
