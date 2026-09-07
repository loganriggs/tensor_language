#!/usr/bin/env python3
"""Construction-specific rank-one oracle axes and exact projective bisectors."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_parent_replay_gradient_geometry_closure_finiteness_and_price pred_b_each_construction_has_a_stable_effective_rank1_oracle pred_c_construction_axes_are_distinct pred_d_cross_use_is_construction_specific pred_e_analytic_bisector_recovers_a_fixed_invariant pred_f_bisector_beats_failed_joint_fit
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from aligned_full_sequence_patch_contract import derive_full_sequence_alignment_contract
import circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1 as v15
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16 as v16
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import construction_oracle_projector_fit as oracle
from projective_bisector_contract import projective_minimax_center
import run_temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1 as factor_parent
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_construction_oracle_projective_bisector_v1.json"
FIT_CORE = ROOT / "ops/construction_oracle_projector_fit.py"
BISECTOR_CORE = ROOT / "ops/projective_bisector_contract.py"
MULTI_RESULT = ROOT / "circuits/followups/temporal_iswas_v15_multi_environment_rank1_das_v1_result.json"
MULTI_RUNNER = ROOT / "ops/run_temporal_iswas_v15_multi_environment_rank1_das_v1.py"
V15_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"
V16_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16.py"
V16_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2_audit_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py"
PROJECTOR_CONTRACT = ROOT / "ops/head_response_projector_contract.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_construction_oracle_projective_bisector_v1"
V15_ROWS_SHA256 = "3f1d28abb658040493284b307cc27ba76f422dddb08ee9c53686c557d49f283c"
V16_ROWS_SHA256 = "4c5dfaee126c04ac7ea6ef5f53d6ad62a24806fa7133a0c773471a90e4d2e468"
EXPECTED = {
    "prior": "2a1409f44d5438bfe995677a04cbca3b910d5ec137a7aa33960a2aa6463a0e64",
    "fit_core": "2519ddf5dd1768a79b84219dd22c888bad55218b9249cbfee2ec6fa6d92f8a24",
    "bisector_core": "db972f4c81947377dc2cb16072d9c95bcf42b3acf1538ae1a8345fe80e064f2c",
    "multi_result": "0eba172c8ada3e3cfcaa2826afa7a348480a6b61bde819c1a0daad9ed8dcb1a4",
    "multi_runner": "8c14405674b4773303e197be04668f29ca97d2662dbeccc0f9653c119716c556",
    "v15_builder": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "v16_builder": "5b1cb38cc62b5682c03505a5090e962d4773015efb44f8fad145515f5066b549",
    "v16_capability": "a1c2baf0bd9548e189ccc4ba4d11c4905f85af1ff7013d408c143f6f2a6434e3",
    "parent_runner": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "projector_contract": "23dbdd6380b745683f502ab0678b6e65e9168b5485eb488d759fa6eebbdc23ef",
}
TARGETS, CONTROLS = ("A1", "A2"), ("P", "C")
PRICE_MAX = {
    "native_capture_forwards": 20, "differentiable_transformer_forwards": 140,
    "transformer_backward_forwards": 32, "model_updates": 32,
    "example_evaluations": 7000, "fit_parameters": 1024,
}


class ExperimentError(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def panel_initialization(backend, bank, training_parity, panel):
    """Factor-SVD initialization using only one target construction and parity."""
    torch, rows = backend.torch, bank["rows"]
    indices = parent.row_indices(rows, panels=(panel,), parity=training_parity)
    bases, receipts = {}, {}
    for layer, head in parent.HEADS:
        site, key = f"L{layer}H{head}", f"head_layer:{layer}"
        full = (bank["donor_cache"][key].view(len(rows), -1, 9, 128)
                - bank["base_cache"][key].view(len(rows), -1, 9, 128))
        matrix = parent.stack_head_prefix(torch, full, rows, indices, head)
        basis, singular = parent._dim_task_basis(torch, matrix, 1)
        gap = float(singular[0] - singular[1]) if len(singular) > 1 else float(singular[0])
        bases[site] = basis.detach().clone()
        receipts[site] = {
            "singular_values_first_12": singular[:12],
            "relative_retained_gap": gap / max(float(singular[0]), 1e-30),
        }
    return bases, receipts


def bases_for_evaluation(fits, panel):
    return {evaluation: fits[panel][1 - evaluation]["best_bases"] for evaluation in (0, 1)}


def evaluate_tuple(backend, contexts, bases_by_evaluation, counters):
    outputs, by_parity = [], {}
    for parity, context in sorted(contexts.items()):
        output = parent.manual_forward(
            backend, context["base_batch"], counters, context=context,
            raw_by_site=bases_by_evaluation[parity], complete15=True)
        outputs.append((context, output))
        by_parity[str(parity)] = parent.output_metrics(backend, context, output)
    return {"by_evaluation_parity": by_parity,
            "combined": parent.combine_reports(backend, outputs)}


def report_replay(observed, expected):
    return factor_parent.replay_comparison(
        parent.registered_report(observed), parent.registered_report(expected))


def replay_pass(report):
    return bool(report["numeric_schema_match"] and report["categorical_match"]
                and report["numeric_max_abs_error"] <= 1e-4)


def target_projection(report, parity, panel):
    return report["by_evaluation_parity"][str(parity)]["targets"][panel]["behavior"]["signed_projection"]


def target_direction(report, parity, panel):
    return report["by_evaluation_parity"][str(parity)]["targets"][panel]["behavior"]["direction_fraction"]


def control_flips(report, parity, panel):
    return report["by_evaluation_parity"][str(parity)]["controls"][panel]["top1_flip_count"]


def main():
    paths = {
        "prior": PRIOR, "fit_core": FIT_CORE, "bisector_core": BISECTOR_CORE,
        "multi_result": MULTI_RESULT, "multi_runner": MULTI_RUNNER,
        "v15_builder": V15_BUILDER, "v16_builder": V16_BUILDER,
        "v16_capability": V16_CAPABILITY, "parent_runner": PARENT_RUNNER,
        "projector_contract": PROJECTOR_CONTRACT,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise ExperimentError(f"missing dependencies: {missing}")
    observed = {name: sha256(path) for name, path in paths.items()}
    prior, multi = json.loads(PRIOR.read_text()), json.loads(MULTI_RESULT.read_text())
    rows15, rows16_all = v15.build_rows(), v16.build_rows()
    rows16 = [row for row in rows16_all if row["transform_id"] in ("A1", "A2", "P")]
    alignment15 = derive_full_sequence_alignment_contract(rows15, required_panels=TARGETS + CONTROLS)
    alignment16 = derive_full_sequence_alignment_contract(rows16, required_panels=("A1", "A2", "P"))
    authority_ok = bool(
        observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
        and v15.validate_rows(rows15) == V15_ROWS_SHA256
        and v16.validate_rows(rows16_all) == V16_ROWS_SHA256
        and alignment15["panel_counts"] == {panel: 16 for panel in TARGETS + CONTROLS}
        and alignment16["panel_counts"] == {panel: 16 for panel in ("A1", "A2", "P")}
        and multi.get("terminal") == "fixed_projector_infeasible_on_observed_constructions"
        and multi.get("predictions", {}).get(
            "pred_a_authority_alignment_absolute_clamp_gradient_closure_finiteness_and_price")
        and not multi.get("predictions", {}).get(
            "pred_b_joint_v15_target_feasible_moved_checkpoint_exists"))
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "v15_rows": len(rows15),
        "v16_patched_rows": len(rows16), "target_oracle_fits": 4,
        "steps_each": oracle.STEPS, "checkpoints": list(oracle.CHECKPOINTS),
        "arms": ["A1_oracle", "A2_oracle", "bisector", "failed_joint_parent"],
        "price_max": PRICE_MAX, "v16_C_excluded": True,
    }
    if not authority_ok:
        raise ExperimentError(f"authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters():
        parameter.requires_grad_(False)
    counters = {key: 0 for key in PRICE_MAX}
    native = backend.native

    def counted_native(batch, *, capture):
        counters["native_capture_forwards"] += 1
        counters["example_evaluations"] += len(batch.row_ids)
        return native(batch, capture=capture)

    backend.native = counted_native
    bank15 = parent.capture_bank(backend, rows15, counters, factors=True)
    contexts15 = {parity: parent.attach_references(
        backend, parent.subset_context(bank15, parent.row_indices(rows15, parity=parity)), counters)
        for parity in (0, 1)}
    manual_errors = [context["manual_native_max_abs_error"] for context in contexts15.values()]
    identity = {site: backend.torch.eye(128, device=backend.device) for site in parent.HEAD_NAMES}
    identity15 = evaluate_tuple(backend, contexts15, {0: identity, 1: identity}, counters)
    identity_replay = report_replay(
        identity15["combined"], multi["reports"]["identity_v15_parent"])

    fits, receipts = {panel: {} for panel in TARGETS}, {panel: {} for panel in TARGETS}
    for panel in TARGETS:
        for training_parity in (0, 1):
            initial, receipt = panel_initialization(backend, bank15, training_parity, panel)
            receipts[panel][training_parity] = receipt
            fits[panel][training_parity] = oracle.fit_one(
                backend, contexts15[training_parity], contexts15[1 - training_parity],
                initial, panel, training_parity, counters)

    stability = {panel: parent.stability_report(
        backend, fits[panel][0]["best_bases"], fits[panel][1]["best_bases"],
        [receipts[panel][0], receipts[panel][1]], 1) for panel in TARGETS}
    between, bisectors_by_training = {}, {}
    geometry_closure = 0.0
    for training_parity in (0, 1):
        between[str(training_parity)], bisectors_by_training[training_parity] = {}, {}
        for site in parent.HEAD_NAMES:
            geometry = projective_minimax_center(
                backend.torch, fits["A1"][training_parity]["best_bases"][site],
                fits["A2"][training_parity]["best_bases"][site])
            bisectors_by_training[training_parity][site] = geometry["center"]
            between[str(training_parity)][site] = {
                key: value for key, value in geometry.items() if key != "center"}
            geometry_closure = max(geometry_closure, geometry["closure_abs_error"])

    bases = {
        "A1_oracle": bases_for_evaluation(fits, "A1"),
        "A2_oracle": bases_for_evaluation(fits, "A2"),
        "bisector": {evaluation: bisectors_by_training[1 - evaluation]
                     for evaluation in (0, 1)},
        "failed_joint_parent": {evaluation: {
            site: backend.torch.tensor(values, device=backend.device)
            for site, values in multi["selected"]["projectors"][str(1 - evaluation)].items()
        } for evaluation in (0, 1)},
    }
    reports15 = {name: evaluate_tuple(backend, contexts15, value, counters)
                 for name, value in bases.items()}
    joint15_replay = report_replay(
        reports15["failed_joint_parent"]["combined"],
        multi["reports"]["selected_v15_crossfit"])

    # V16 is constructed only after all v15 fits, checkpoints, and bisectors are frozen.
    selection_finished_utc = utc_now()
    bank16 = parent.capture_bank(backend, rows16, counters, factors=False)
    contexts16 = {parity: parent.attach_references(
        backend, parent.subset_context(bank16, parent.row_indices(rows16, parity=parity)), counters)
        for parity in (0, 1)}
    manual_errors.extend(context["manual_native_max_abs_error"] for context in contexts16.values())
    reports16 = {name: evaluate_tuple(backend, contexts16, value, counters)
                 for name, value in bases.items()}
    joint16_replay = report_replay(
        reports16["failed_joint_parent"]["combined"],
        multi["reports"]["selected_v16_sealed_crossfit"])

    orthogonality = max(
        float((basis.T @ basis - backend.torch.eye(1, device=backend.device)).abs().max())
        for family in bases.values() for fold_bases in family.values()
        for basis in fold_bases.values())
    gradient_min = min(fit["gradient_max_abs"] for panel in TARGETS for fit in fits[panel].values())
    counters["fit_parameters"] = 2 * 4 * 128
    price_ok = all(counters[key] <= PRICE_MAX[key] for key in PRICE_MAX)

    pred_b = bool(all(fits[panel][fold]["best"]["score"]["feasible"]
                      for panel in TARGETS for fold in (0, 1))
                  and all(stability[panel]["minimum_principal_cosine"] >= .75
                          for panel in TARGETS))
    pred_c = any(all(between[str(fold)][site]["absolute_axis_cosine"] < .75
                     for fold in (0, 1)) for site in parent.HEAD_NAMES)
    pred_d = all(
        target_projection(reports15[panel + "_oracle"], parity, panel)
        >= target_projection(reports15[("A2" if panel == "A1" else "A1") + "_oracle"], parity, panel) + .08
        for parity in (0, 1) for panel in TARGETS)
    pred_e = bool(all(
        target_projection(reports15["bisector"], parity, panel) >= .75
        and target_direction(reports15["bisector"], parity, panel) >= .875
        for parity in (0, 1) for panel in TARGETS)
        and all(control_flips(reports15["bisector"], parity, panel) == 0
                for parity in (0, 1) for panel in CONTROLS)
        and all(target_projection(reports16["bisector"], parity, panel) >= .65
                and target_direction(reports16["bisector"], parity, panel) >= .875
                for parity in (0, 1) for panel in TARGETS)
        and all(control_flips(reports16["bisector"], parity, "P") == 0
                for parity in (0, 1)))
    pred_f = all(
        min(target_projection(reports15["bisector"], parity, panel) for panel in TARGETS)
        >= min(target_projection(reports15["failed_joint_parent"], parity, panel)
               for panel in TARGETS) + .03 for parity in (0, 1))
    pred_a = bool(
        authority_ok and max(manual_errors) <= 1e-4 and replay_pass(identity_replay)
        and replay_pass(joint15_replay) and replay_pass(joint16_replay)
        and bank15["attention_reconstruction_max_abs_error"] <= 1e-4
        and bank15["factor_closure_max_abs_error"] <= 1e-4
        and gradient_min > 1e-10 and geometry_closure <= 1e-6
        and orthogonality <= 1e-4 and price_ok
        and finite({"stability": stability, "between": between,
                    "reports15": reports15, "reports16": reports16}))
    predictions = {
        "pred_a_authority_alignment_parent_replay_gradient_geometry_closure_finiteness_and_price": pred_a,
        "pred_b_each_construction_has_a_stable_effective_rank1_oracle": pred_b,
        "pred_c_construction_axes_are_distinct": pred_c,
        "pred_d_cross_use_is_construction_specific": pred_d,
        "pred_e_analytic_bisector_recovers_a_fixed_invariant": pred_e,
        "pred_f_bisector_beats_failed_joint_fit": pred_f,
    }
    if not pred_a:
        terminal = "invalid"
    elif not pred_b:
        terminal = "no_stable_effective_single_construction_rank1"
    elif pred_e and pred_f:
        terminal = "fixed_invariant_bisector_candidate"
    elif pred_c and pred_d:
        terminal = "construction_conditioned_coordinate"
    else:
        terminal = "interaction_conditioned_or_nonlinear_geometry"

    result = {
        "schema": "temporal_iswas_v15_construction_oracle_projective_bisector_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "selection_finished_utc": selection_finished_utc,
        "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "v15_rows_sha256": V15_ROWS_SHA256,
        "v16_rows_sha256": V16_ROWS_SHA256,
        "alignment_contracts": {"v15": alignment15, "v16_without_c": alignment16},
        "dryrun": dryrun,
        "fit_spec": {"rank": 1, "targets": list(TARGETS), "controls": list(CONTROLS),
                     "steps": oracle.STEPS, "checkpoints": list(oracle.CHECKPOINTS),
                     "learning_rate": oracle.LEARNING_RATE},
        "fits": {panel: {str(fold): oracle.strip_fit(fit)
                          for fold, fit in panel_fits.items()}
                 for panel, panel_fits in fits.items()},
        "projectors": {panel: {str(fold): {site: basis.detach().cpu().tolist()
                                            for site, basis in fit["best_bases"].items()}
                               for fold, fit in panel_fits.items()}
                       for panel, panel_fits in fits.items()},
        "within_construction_stability": stability,
        "between_construction_geometry": between,
        "bisectors": {str(fold): {site: basis.detach().cpu().tolist()
                                   for site, basis in site_bases.items()}
                      for fold, site_bases in bisectors_by_training.items()},
        "reports": {"v15": reports15, "v16": reports16},
        "instrument": {
            "manual_native_max_abs_error": max(manual_errors),
            "identity_v15_parent_replay": identity_replay,
            "failed_joint_v15_replay": joint15_replay,
            "failed_joint_v16_replay": joint16_replay,
            "attention_reconstruction_max_abs_error": bank15["attention_reconstruction_max_abs_error"],
            "factor_closure_max_abs_error": bank15["factor_closure_max_abs_error"],
            "gradient_min_fit_max_abs": gradient_min,
            "bisector_geometry_closure_max_abs_error": geometry_closure,
            "projector_orthogonality_max_abs_error": orthogonality,
        },
        "predictions": predictions, "terminal": terminal,
        "price": {**counters, "maxima": PRICE_MAX},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "stability": stability, "between": between,
        "v15": reports15, "v16": reports16, "predictions": predictions,
        "terminal": terminal, "price": result["price"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
