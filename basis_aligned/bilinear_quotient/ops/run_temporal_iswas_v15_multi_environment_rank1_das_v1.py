#!/usr/bin/env python3
"""Fixed-rank multi-construction DAS with v16 sealed until v15 selection."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_absolute_clamp_gradient_closure_finiteness_and_price pred_b_joint_v15_target_feasible_moved_checkpoint_exists pred_c_joint_v15_improves_a2_selectively pred_d_joint_projector_is_fold_stable pred_e_sealed_v16_identification_transfers pred_f_sealed_v16_p_is_selective
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
import multi_construction_head_projector_fit as joint
import run_temporal_iswas_v15_aligned_attention_pattern_value_factorial_v1 as factor_parent
import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_multi_environment_rank1_das_v1.json"
DESIGN = ROOT.parent / "polynomial_causal/explanations/TEMPORAL_ISWAS_MULTI_CONSTRUCTION_DAS_DESIGN_2026-09-07.md"
FIT_CORE = ROOT / "ops/multi_construction_head_projector_fit.py"
SELECTOR = ROOT / "ops/multi_environment_projector_contract.py"
V15_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_v15_aligned_controls_v1.py"
V16_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16.py"
V16_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2_audit_result.json"
V15_DAS = ROOT / "circuits/followups/temporal_iswas_v15_head_response_target_feasible_regularized_das_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1.py"
PROJECTOR_CONTRACT = ROOT / "ops/head_response_projector_contract.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_multi_environment_rank1_das_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_multi_environment_rank1_das_v1"
V15_ROWS_SHA256 = "3f1d28abb658040493284b307cc27ba76f422dddb08ee9c53686c557d49f283c"
V16_ROWS_SHA256 = "4c5dfaee126c04ac7ea6ef5f53d6ad62a24806fa7133a0c773471a90e4d2e468"
EXPECTED = {
    "prior": "252e604f133ce68f473d7b16c7676dd4b592889b1d4b413d0d796d3ab3bf2eb4",
    "design": "8a808f03078f1c93ca30fce9dbb4eab7e67d9079b9e4768132e47a56bc61694d",
    "fit_core": "db0f87d1cd30b189303ac53500e44868bfeafbf483a47885b8498b5cbf250215",
    "selector": "12f18ed73401e54014553f049fd91ffcc09f3fffda0d2c6fb343ad857a170311",
    "v15_builder": "7027e128ea3e68a35e92f451d0f62453a937e64a8dafda24d841aa1f1d29e645",
    "v16_builder": "5b1cb38cc62b5682c03505a5090e962d4773015efb44f8fad145515f5066b549",
    "v16_capability": "a1c2baf0bd9548e189ccc4ba4d11c4905f85af1ff7013d408c143f6f2a6434e3",
    "v15_das": "0e3ee2641a8f7c38ebad5cb0b73cf71461e2cd9bd96cdcb30a581e2f8d7bd30b",
    "parent_runner": "0b7d92038283cad0fd48e398739474c4004de0da8e361d5aaaf0f68a4db6595e",
    "projector_contract": "23dbdd6380b745683f502ab0678b6e65e9168b5485eb488d759fa6eebbdc23ef",
}
INITIALIZATIONS = ("factor_svd_a1", "joint_construction_dim", "frozen_a1_only_projector")
STABILITY_MIN_COSINE = 0.75
V16_PROJECTION_MIN, V16_DIRECTION_MIN, V16_IMPROVEMENT_MIN = 0.65, 0.875, 0.08
PRICE_MAX = {
    "native_capture_forwards": 20,
    "differentiable_transformer_forwards": 200,
    "transformer_backward_forwards": 48,
    "model_updates": 48,
    "example_evaluations": 8000,
    "fit_parameters": 512,
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


def _joint_dim_initialization(backend, bank, training_parity):
    torch, rows = backend.torch, bank["rows"]
    indices = parent.row_indices(rows, panels=("A1", "A2"), parity=training_parity)
    bases, receipts = {}, {}
    for layer, head in parent.HEADS:
        site, key = f"L{layer}H{head}", f"head_layer:{layer}"
        full = (bank["donor_cache"][key].view(len(rows), -1, 9, 128)
                - bank["base_cache"][key].view(len(rows), -1, 9, 128))
        matrix = parent.stack_head_prefix(torch, full, rows, indices, head)
        basis, singular = parent._dim_task_basis(torch, matrix, 1)
        gap = float(singular[0] - singular[1]) if len(singular) > 1 else float(singular[0])
        bases[site] = basis.detach().clone()
        receipts[site] = {"singular_values_first_12": singular[:12],
                          "relative_retained_gap": gap / max(float(singular[0]), 1e-30)}
    return bases, receipts


def initialization_for(backend, bank, context, training_parity, name, frozen_result):
    if name == "factor_svd_a1":
        return parent.initialization_for(
            backend, bank, context, training_parity, 1, "factor_svd")
    if name == "joint_construction_dim":
        return _joint_dim_initialization(backend, bank, training_parity)
    if name == "frozen_a1_only_projector":
        bases = {
            site: backend.torch.tensor(values, device=backend.device)
            for site, values in frozen_result["selected"]["projectors"][str(training_parity)].items()
        }
        return bases, {site: {"singular_values_first_12": [], "relative_retained_gap": 0.0}
                       for site in parent.HEAD_NAMES}
    raise ExperimentError(f"unknown initialization {name}")


def pair_key(pair):
    scores = [pair["fits"][fold]["best"]["score"] for fold in (0, 1)]
    return (
        max(score["worst_target_violation"] for score in scores),
        max(score["control_objective"] for score in scores),
        max(score["worst_control_max_kl"] for score in scores),
        sum(score["total_control_flips"] for score in scores),
        INITIALIZATIONS.index(pair["initialization"]),
    )


def run_crossfit(backend, contexts, bases_by_evaluation_parity, counters):
    return parent.run_crossfit(backend, contexts, bases_by_evaluation_parity, counters)


def main():
    paths = {
        "prior": PRIOR, "design": DESIGN, "fit_core": FIT_CORE, "selector": SELECTOR,
        "v15_builder": V15_BUILDER, "v16_builder": V16_BUILDER,
        "v16_capability": V16_CAPABILITY, "v15_das": V15_DAS,
        "parent_runner": PARENT_RUNNER, "projector_contract": PROJECTOR_CONTRACT,
    }
    missing = [key for key, path in paths.items() if not path.exists()]
    if missing:
        raise ExperimentError(f"missing sealed dependencies: {missing}")
    observed = {key: sha256(path) for key, path in paths.items()}
    prior = json.loads(PRIOR.read_text())
    capability = json.loads(V16_CAPABILITY.read_text())
    frozen = json.loads(V15_DAS.read_text())
    rows15, rows16_all = v15.build_rows(), v16.build_rows()
    rows16 = [row for row in rows16_all if row["transform_id"] in ("A1", "A2", "P")]
    alignment15 = derive_full_sequence_alignment_contract(
        rows15, required_panels=("A1", "A2", "P", "C"))
    alignment16 = derive_full_sequence_alignment_contract(
        rows16, required_panels=("A1", "A2", "P"))
    authority_ok = bool(
        observed == EXPECTED and prior.get("candidate_id") == CANDIDATE_ID
        and v15.validate_rows(rows15) == V15_ROWS_SHA256
        and v16.validate_rows(rows16_all) == V16_ROWS_SHA256
        and alignment15["panel_counts"] == {panel: 16 for panel in ("A1", "A2", "P", "C")}
        and alignment16["panel_counts"] == {panel: 16 for panel in ("A1", "A2", "P")}
        and capability.get("terminal") == "manifest"
        and all(capability.get("predictions", {}).values())
        and capability.get("causal_outcomes_opened") is False
        and frozen.get("predictions", {}).get(
            "pred_a_authority_alignment_absolute_clamp_gradient_closure_finiteness_and_price")
        and len(INITIALIZATIONS) == 3)
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "v15_rows": len(rows15),
        "v16_patched_rows": len(rows16), "rank": 1,
        "initializations": list(INITIALIZATIONS), "steps": joint.STEPS,
        "checkpoints": list(joint.CHECKPOINTS), "price_max": PRICE_MAX,
        "sealed_until_v15_selection": ["v16:A1", "v16:A2", "v16:P"],
        "explicitly_excluded": ["v16:C"],
    }
    if not authority_ok:
        raise ExperimentError(f"authority or capability changed: {observed}")
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
    contexts15 = {
        parity: parent.attach_references(
            backend, parent.subset_context(
                bank15, parent.row_indices(rows15, parity=parity)), counters)
        for parity in (0, 1)
    }
    if max(context["manual_native_max_abs_error"] for context in contexts15.values()) > 1e-4:
        raise ExperimentError("manual reader does not reproduce native v15 logits")
    identity = {site: backend.torch.eye(128, device=backend.device) for site in parent.HEAD_NAMES}
    identity_outputs15, identity_report15 = run_crossfit(
        backend, contexts15, {0: identity, 1: identity}, counters)
    del identity_outputs15
    parent_reference = parent.registered_report(frozen["reports"]["identity_full_parent"])
    replay15 = factor_parent.replay_comparison(
        parent.registered_report(identity_report15), parent_reference)
    if (not replay15["numeric_schema_match"] or not replay15["categorical_match"]
            or replay15["numeric_max_abs_error"] > 1e-4
            or bank15["attention_reconstruction_max_abs_error"] > 1e-4
            or bank15["factor_closure_max_abs_error"] > 1e-4):
        raise ExperimentError("v15 identity replay or factor closure failed")

    pairs = []
    for initialization in INITIALIZATIONS:
        fits, receipts = {}, {}
        for training_parity in (0, 1):
            initial, receipt = initialization_for(
                backend, bank15, contexts15[training_parity], training_parity,
                initialization, frozen)
            receipts[training_parity] = receipt
            fits[training_parity] = joint.fit_one(
                backend, contexts15[training_parity], contexts15[1 - training_parity],
                initial, initialization, training_parity, counters)
        stability = parent.stability_report(
            backend, fits[0]["best_bases"], fits[1]["best_bases"],
            [receipts[0], receipts[1]], 1)
        pairs.append({
            "initialization": initialization, "fits": fits, "stability": stability,
            "both_feasible": all(fits[fold]["best"]["score"]["feasible"] for fold in (0, 1)),
        })
    feasible_pairs = [pair for pair in pairs if pair["both_feasible"]]
    selected = min(feasible_pairs or pairs, key=pair_key)
    selection_finished_utc = utc_now()

    selected_bases15 = {0: selected["fits"][1]["best_bases"],
                        1: selected["fits"][0]["best_bases"]}
    _selected_outputs15, selected_report15 = run_crossfit(
        backend, contexts15, selected_bases15, counters)
    frozen_bases15 = {
        evaluation: {site: backend.torch.tensor(values, device=backend.device)
                     for site, values in frozen["selected"]["projectors"][str(1 - evaluation)].items()}
        for evaluation in (0, 1)
    }
    _frozen_outputs15, frozen_report15 = run_crossfit(
        backend, contexts15, frozen_bases15, counters)

    # The v16 bank is first touched causally only after all v15 selection is frozen above.
    bank16 = parent.capture_bank(backend, rows16, counters, factors=False)
    contexts16 = {
        parity: parent.attach_references(
            backend, parent.subset_context(
                bank16, parent.row_indices(rows16, parity=parity)), counters)
        for parity in (0, 1)
    }
    selected_bases16 = {0: selected["fits"][1]["best_bases"],
                        1: selected["fits"][0]["best_bases"]}
    _selected_outputs16, selected_report16 = run_crossfit(
        backend, contexts16, selected_bases16, counters)
    _frozen_outputs16, frozen_report16 = run_crossfit(
        backend, contexts16, frozen_bases15, counters)

    closure = max(context["manual_native_max_abs_error"]
                  for context in tuple(contexts15.values()) + tuple(contexts16.values()))
    orthogonality = max(
        float((basis.T @ basis - backend.torch.eye(1, device=backend.device)).abs().max())
        for pair in pairs for fit in pair["fits"].values()
        for basis in fit["best_bases"].values())
    gradient_min = min(fit["gradient_max_abs"] for pair in pairs for fit in pair["fits"].values())
    counters["fit_parameters"] = 4 * 128
    price_ok = all(counters[key] <= PRICE_MAX[key] for key in PRICE_MAX)
    pred_a = bool(
        authority_ok and closure <= 1e-4 and replay15["numeric_schema_match"]
        and replay15["categorical_match"] and replay15["numeric_max_abs_error"] <= 1e-4
        and bank15["attention_reconstruction_max_abs_error"] <= 1e-4
        and bank15["factor_closure_max_abs_error"] <= 1e-4
        and gradient_min > 1e-10 and orthogonality <= 1e-4 and price_ok
        and finite({"pairs": [{"key": pair_key(pair), "stability": pair["stability"]}
                              for pair in pairs],
                    "v15": selected_report15, "v16": selected_report16}))
    pred_b = bool(selected["both_feasible"] and all(
        selected["fits"][fold]["best"]["report"]["step"] > 0 for fold in (0, 1)))
    pred_c = bool(
        selected_report15["targets"]["A2"]["behavior"]["signed_projection"]
            >= frozen_report15["targets"]["A2"]["behavior"]["signed_projection"] + 0.08
        and selected_report15["targets"]["A1"]["behavior"]["signed_projection"] >= 0.75
        and all(selected_report15["controls"][panel]["top1_flip_count"] == 0
                for panel in ("P", "C")))
    pred_d = selected["stability"]["minimum_principal_cosine"] >= STABILITY_MIN_COSINE
    v16_worst_selected = min(
        selected_report16["targets"][panel]["behavior"]["signed_projection"]
        for panel in ("A1", "A2"))
    v16_worst_frozen = min(
        frozen_report16["targets"][panel]["behavior"]["signed_projection"]
        for panel in ("A1", "A2"))
    pred_e = bool(all(
        selected_report16["targets"][panel]["behavior"]["signed_projection"] >= V16_PROJECTION_MIN
        and selected_report16["targets"][panel]["behavior"]["direction_fraction"] >= V16_DIRECTION_MIN
        for panel in ("A1", "A2"))
        and v16_worst_selected >= v16_worst_frozen + V16_IMPROVEMENT_MIN)
    pred_f = selected_report16["controls"]["P"]["top1_flip_count"] == 0
    predictions = {
        "pred_a_authority_alignment_absolute_clamp_gradient_closure_finiteness_and_price": pred_a,
        "pred_b_joint_v15_target_feasible_moved_checkpoint_exists": pred_b,
        "pred_c_joint_v15_improves_a2_selectively": pred_c,
        "pred_d_joint_projector_is_fold_stable": pred_d,
        "pred_e_sealed_v16_identification_transfers": pred_e,
        "pred_f_sealed_v16_p_is_selective": pred_f,
    }
    if not pred_a:
        terminal = "invalid"
    elif not pred_b:
        terminal = "fixed_projector_infeasible_on_observed_constructions"
    elif not pred_c:
        terminal = "observed_construction_or_control_failure"
    elif not pred_d:
        terminal = "projector_instability"
    elif not pred_e:
        terminal = "construction_conditioned_geometry"
    elif not pred_f:
        terminal = "ood_control_nonselective"
    else:
        terminal = "fixed_invariant_causal_candidate"

    result = {
        "schema": "temporal_iswas_v15_multi_environment_rank1_das_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "selection_finished_utc": selection_finished_utc,
        "v16_opened_after_v15_selection": True, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "v15_rows_sha256": V15_ROWS_SHA256,
        "v16_rows_sha256": V16_ROWS_SHA256,
        "alignment_contracts": {"v15": alignment15, "v16_without_c": alignment16},
        "dryrun": dryrun,
        "fit_spec": {
            "heads": list(parent.HEAD_NAMES), "rank": 1,
            "initializations": list(INITIALIZATIONS), "steps": joint.STEPS,
            "checkpoints": list(joint.CHECKPOINTS), "learning_rate": joint.LEARNING_RATE,
            "target_projection_bar": joint.PROJECTION_MIN,
            "direction_fraction_bar": joint.DIRECTION_MIN,
        },
        "fit_pairs": [{
            "initialization": pair["initialization"], "both_feasible": pair["both_feasible"],
            "selection_key": pair_key(pair), "stability": pair["stability"],
            "fits": {str(fold): joint.strip_fit(fit) for fold, fit in pair["fits"].items()},
        } for pair in pairs],
        "selected": {
            "initialization": selected["initialization"], "selection_key": pair_key(selected),
            "both_feasible": selected["both_feasible"], "stability": selected["stability"],
            "folds": {str(fold): joint.strip_fit(fit) for fold, fit in selected["fits"].items()},
            "projectors": {
                str(fold): {site: basis.detach().cpu().tolist()
                            for site, basis in selected["fits"][fold]["best_bases"].items()}
                for fold in (0, 1)
            },
        },
        "reports": {
            "selected_v15_crossfit": selected_report15,
            "frozen_a1_only_v15_crossfit": frozen_report15,
            "selected_v16_sealed_crossfit": selected_report16,
            "frozen_a1_only_v16_sealed_crossfit": frozen_report16,
            "identity_v15_parent": identity_report15,
        },
        "instrument": {
            "manual_native_max_abs_error": closure, "v15_parent_replay": replay15,
            "attention_reconstruction_max_abs_error": bank15["attention_reconstruction_max_abs_error"],
            "factor_closure_max_abs_error": bank15["factor_closure_max_abs_error"],
            "projector_orthogonality_max_abs_error": orthogonality,
            "gradient_min_fit_max_abs": gradient_min,
        },
        "predictions": predictions, "terminal": terminal,
        "price": {**counters, "maxima": PRICE_MAX},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "selected_initialization": selected["initialization"],
        "selection_key": pair_key(selected), "stability": selected["stability"],
        "v15": selected_report15, "v16": selected_report16,
        "predictions": predictions, "terminal": terminal, "price": result["price"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
