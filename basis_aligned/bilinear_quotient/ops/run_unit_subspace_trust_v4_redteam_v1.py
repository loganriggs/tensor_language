#!/usr/bin/env python3
# BQGATE: frozen closure and optimizer-baseline audit; no fitting or post-outcome repair.
"""Red-team v4's concatenated DAS/complement construction.

Registered receipt: circuits/prior_art/unit_subspace_trust_v4_redteam_v1.json.
The decisive algebraic check is q=I: a purported subspace family over an exact unit-set
intervention must reproduce that exact intervention when its subspace spans the complete
concatenated space.  The second check evaluates difference-in-means under the *same raw training
loss* used by constrained DAS and compares it with v4's recorded final loss.  No refit is done.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_candidate_control_choice as m_list
import circuit_fast_screen_candidate_correlative_pair as m_corr
import circuit_fast_screen_candidate_modal_remoteness as m_modal
import circuit_fast_screen_candidate_possessive_adjacent as m_poss
import circuit_fast_screen_candidate_polarity_state as m_pol
import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g

ROOT = Path(__file__).resolve().parent.parent
V4 = ROOT / "circuits/followups/unit_subspace_trust_v4_result.json"
OUT = ROOT / "circuits/followups/unit_subspace_trust_v4_redteam_v1_result.json"
TOL = 1e-4
SETS = {
    "correlative_pair.both_vs_neither": (m_corr, "heads", [
        "attn:08:head:01", "attn:07:head:08", "attn:14:head:08"]),
    "modal_remoteness.would_vs_will": (m_modal, "heads", [
        "attn:09:head:04", "attn:11:head:03"]),
    "numbered_list.control_choice_discriminator": (m_list, "heads", [
        "attn:08:head:03", "attn:08:head:07"]),
    "polarity_state.negative_vs_positive.heads": (m_pol, "heads", [
        "attn:07:head:08", "attn:08:head:01", "attn:04:head:07", "attn:05:head:08"]),
    "possessive_number.adjacent_antecedent.heads": (m_poss, "heads", [
        "attn:04:head:05", "attn:03:head:04", "attn:09:head:06", "attn:10:head:05"]),
    "polarity_state.negative_vs_positive.with_mlp04": (m_pol, "mlp", [
        "attn:07:head:08", "attn:08:head:01", "mlp:04", "attn:10:head:05"]),
    "possessive_number.adjacent_antecedent.with_mlp08": (m_poss, "mlp", [
        "attn:04:head:05", "mlp:08", "attn:09:head:06", "attn:10:head:05"]),
}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 80, 2560


def _plan():
    return {
        "candidate_id": "corpus.unit_subspace_trust_v4_redteam_v1",
        "sets": {key: value[2] for key, value in SETS.items()},
        "checks": ["q_identity_exact_patch_closure", "dim_on_cdas_training_objective"],
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0,
        "model_updates": 0,
        "fit_parameters": 0,
        "gpu_accessed": False,
        "model_loaded": False,
        "execution_policy": "managed_queue_only",
    }


def _axis(out):
    return -(out[:, 0] - out[:, 1])


def main():
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True))
        return
    torch = __import__("torch")
    backend = producer.Bilin18TorchBackend.load("cuda")
    started = time.perf_counter()
    old = json.loads(V4.read_text())
    instrument = g.verify_against_producer(
        backend, g.rows_of(m_pol, "A1"), layer=7, heads=(8,), mlp_layer=4)
    reports = {}
    for label, (module, kind, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1")[0::2])
        exact_out = g.forward_units(
            backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
            base_cache=prep.base_cache)
        dim = sum(g.unit_dim(unit) for unit in units)
        identity = torch.eye(dim, device=backend.device)
        identity_out = g.forward_units(
            backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
            base_cache=prep.base_cache, q=identity)
        closure_max = float((identity_out - exact_out).abs().max())
        closure_mean = float((identity_out - exact_out).abs().mean())
        exact_axis = _axis(exact_out)
        identity_axis = _axis(identity_out)
        exact_recovery = g.recovery(prep, exact_axis.detach().cpu().tolist())
        identity_recovery = g.recovery(prep, identity_axis.detach().cpu().tolist())

        q_dim = g.diff_in_means_direction(backend, prep, units)
        sub_out = g.forward_units(
            backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
            base_cache=prep.base_cache, q=q_dim)
        comp_out = g.forward_units(
            backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
            base_cache=prep.base_cache, q=q_dim, complement=True)
        target = exact_axis.detach()
        base_target = torch.tensor(prep.base_axis, device=backend.device)
        dim_match = float(((_axis(sub_out) - target) ** 2).mean())
        dim_inert = float(((_axis(comp_out) - base_target) ** 2).mean())
        dim_joint = dim_match + dim_inert
        old_tail = old["sets"][label]["directions"]["cdas"]["loss_history"][-1]
        old_match, old_inert = float(old_tail[1]), float(old_tail[2])
        old_joint = old_match + old_inert
        layers = [g.unit_layer(unit) for unit in units]
        reports[label] = {
            "kind": kind,
            "units": units,
            "layers": layers,
            "same_layer": len(set(layers)) == 1,
            "dimension": dim,
            "identity_vs_exact": {
                "max_abs_logit_error": closure_max,
                "mean_abs_logit_error": closure_mean,
                "exact_recovery": exact_recovery,
                "identity_recovery": identity_recovery,
                "recovery_gap": identity_recovery - exact_recovery,
                "passed": closure_max <= TOL,
            },
            "difference_in_means_training_objective": {
                "match_mse": dim_match,
                "complement_inert_mse": dim_inert,
                "joint_lambda1": dim_joint,
            },
            "v4_cdas_recorded_final_training_objective": {
                "match_mse": old_match,
                "complement_inert_mse": old_inert,
                "joint_lambda1": old_joint,
            },
            "cdas_beats_dim_on_own_loss": old_joint <= dim_joint,
            "dim_to_cdas_joint_loss_ratio": dim_joint / old_joint if old_joint else None,
        }
        print(label, json.dumps({
            "same_layer": reports[label]["same_layer"],
            "closure_max": closure_max,
            "dim_loss": dim_joint,
            "cdas_loss": old_joint,
        }))

    same = reports["numbered_list.control_choice_discriminator"]
    multi = [value for value in reports.values() if not value["same_layer"]]
    mlp = [value for value in reports.values() if value["kind"] == "mlp"]
    predictions = {
        "pred_a_same_layer_identity_closes": same["identity_vs_exact"]["passed"],
        "pred_b_all_multilayer_identity_closes": all(
            value["identity_vs_exact"]["passed"] for value in multi),
        "pred_c_cdas_beat_dim_on_own_loss": all(
            value["cdas_beats_dim_on_own_loss"] for value in mlp),
    }
    if not instrument["passed"] or not predictions["pred_a_same_layer_identity_closes"]:
        terminal = "instrument_invalid"
    elif not predictions["pred_b_all_multilayer_identity_closes"]:
        terminal = "v4_partition_invalid"
    elif not predictions["pred_c_cdas_beat_dim_on_own_loss"]:
        terminal = "optimizer_failure"
    else:
        terminal = "target_generalization_mismatch"
    result = {
        "schema": "circuit_unit_subspace_trust_v4_redteam_result_v1",
        "candidate_id": "corpus.unit_subspace_trust_v4_redteam_v1",
        "instrument": instrument,
        "tolerance": TOL,
        "predictions": {key: bool(value) for key, value in predictions.items()},
        "terminal": terminal,
        "sets": reports,
        "price": {
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0,
            "model_updates": 0,
        },
        "serial_seconds": time.perf_counter() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": result["predictions"], "terminal": terminal}, indent=2))


if __name__ == "__main__":
    main()
