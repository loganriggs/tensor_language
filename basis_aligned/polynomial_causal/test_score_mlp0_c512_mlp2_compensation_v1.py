import numpy as np

import score_mlp0_c512_mlp2_compensation_v1 as score


def synthetic_ledgers(n_units=384, effects=None):
    effects = effects or {}
    output = {}
    for contrast in score.CONTRASTS:
        output[contrast] = {}
        value = effects.get(contrast, .2)
        for metric, margin in score.MARGINS.items():
            # Synthetic identity has two 256-token windows per unit.
            counts = np.full((n_units, 16), 32, dtype=np.int64)
            output[contrast][metric] = {
                "sums": (counts * margin * value).tolist(),
                "counts": counts.tolist(),
            }
    return output


def synthetic_identity():
    occupancy = [2] * 384
    mapping = [unit for unit, count in enumerate(occupancy) for _ in range(count)]
    return {
        "unit_kind": "source_document",
        "ordered_ids": [f"doc-{index}" for index in range(384)],
        "row_to_unit": mapping,
        "wave_labels": ["A"] * 192 + ["B"] * 192,
    }


def authority_binding(identity):
    mapping = np.asarray(identity["row_to_unit"], dtype=np.int64)
    occupancy = np.bincount(mapping, minlength=384)
    return {
        "status": "frozen_before_any_v2_c512_mlp2_compensation_evaluation_forward",
        "inference_contract": score.frozen_inference_contract(),
        "integrity_contract": {
            "n_eval_windows": 768,
            "exact_call_counts": {
                "candidate_original_down_calls": 0,
                "poison_canary_calls": 1,
                "c512_proxy_calls": 768,
            },
            "exact_phase_site_call_counts": {
                "mlp1_teacher_capture": {"1": 768},
                "mlp2_teacher_capture": {"2": 768},
                "crossed_suffix_replay": {str(i): 1536 for i in range(3, 18)},
                "parent_replay_mlp_sites": {
                    **{str(i): 768 for i in range(18)},
                    "2": 384,
                },
                "crossed_forbidden_teacher": {"1": 0, "2": 0},
            },
            "bound_hashes": {
                "source_closure_sha256": "a" * 64,
                "row_receipt_sha256": "b" * 64,
                "row_tensor_sha256": "c" * 64,
                "c512_program_sha256": "d" * 64,
                "model_checkpoint_sha256": "e" * 64,
                "model_config_sha256": "f" * 64,
                "inherited_currency_sha256": "1" * 64,
                "control_contract_sha256": "2" * 64,
                "repair_amendment_sha256": "4" * 64,
            },
            "parent_replay_tolerances": {
                "raw_logits_max_abs": 1e-6,
                "capped_logits_max_abs": 1e-6,
                "ce_abs": 1e-7,
            },
            "same_realization_delta_tolerance": 1e-6,
            "carried_state_identity_tolerance": 1e-6,
            "native_control_norm_contract": dict(score.NATIVE_CONTROL_NORM_CONTRACT),
            "inherited_centered_capped_logit_rms": 2.5,
            "unit_identity_hashes": {
                "ordered_ids_sha256": score.ordered_ids_sha256(identity["ordered_ids"]),
                "row_to_unit_sha256": score.integer_array_sha256(mapping),
                "occupancy_sha256": score.integer_array_sha256(occupancy),
                "wave_labels_sha256": score.ordered_ids_sha256(identity["wave_labels"]),
            },
        },
    }


def passing_integrity(authority):
    contract = authority["integrity_contract"]
    return {
        "call_counts": dict(contract["exact_call_counts"]),
        "phase_site_call_counts": {
            phase: dict(values)
            for phase, values in contract["exact_phase_site_call_counts"].items()
        },
        "observed_hashes": dict(contract["bound_hashes"]),
        "parent_replay": {
            parent: {
                "raw_logits_max_abs": 0.0,
                "capped_logits_max_abs": 0.0,
                "ce_abs": 0.0,
                "passes": True,
            }
            for parent in (
                "exact_live", "candidate_live", "exact_mlp2_omit", "candidate_mlp2_omit"
            )
        },
        "same_realization_delta": {"max_abs": 0.0, "passes": True},
        "carried_state_identity": {"x0_max_abs": 0.0, "v1_max_abs": 0.0, "passes": True},
        "control_checks": {
            "derangement_bijection": True,
            "donor_arrays_indexed_by_permutation": True,
            "derangement_no_same_document": True,
            "derangement_wave_cell_preserving": True,
            "control_realization_sha256": "3" * 64,
            "native_control_norm_max_abs_error": 0.0,
            "native_control_norm_max_allowance_ratio": 0.0,
            "native_control_norm_all_positions_within_bound": True,
            "passes": True,
        },
        "scoring_currency": {
            "centered_capped_logit_rms": contract[
                "inherited_centered_capped_logit_rms"
            ],
            "matches_authority": True,
        },
    }


def payload(effects):
    identity = synthetic_identity()
    authority = authority_binding(identity)
    return {
        "sufficient_statistics": synthetic_ledgers(effects=effects),
        "unit_identity": identity,
        "coverage": {"wave_A": 1.0, "wave_B": 1.0, "pooled": 1.0},
        "authority": authority,
        "integrity": passing_integrity(authority),
    }


def test_signed_ce_sign_flip_uses_two_sided_centered_error():
    coordinates = np.zeros((1, len(score.MARGINS) * 16))
    bootstrap = np.zeros((3, 1, len(score.MARGINS) * 16))
    ce_offset = list(score.MARGINS).index("ce_abs") * 16
    coordinates[0, ce_offset] = .2
    bootstrap[:, 0, ce_offset] = -.2
    upper, lower, up, low, joint = score.familywise_bounds(coordinates, bootstrap)
    assert np.isclose(up, .4) and np.isclose(low, .4) and np.isclose(joint, .4)
    assert np.isclose(upper[0], .6) and np.isclose(lower[0], -.2)


def test_one_common_family_contains_all_eight_contrasts():
    scope = score.score_scope(
        synthetic_ledgers(20), np.arange(20), minimum_support=1,
        n_bootstrap=20, seed=3,
    )
    assert set(scope["contrasts"]) == set(score.CONTRASTS)
    assert scope["bootstrap"]["replicates"] == 20


def test_suppression_alignment_and_component_equivalence_require_controls(monkeypatch):
    monkeypatch.setattr(score, "MIN_DOCUMENTS_PER_CELL", 1)
    monkeypatch.setattr(score, "N_BOOTSTRAP", 20)
    monkeypatch.setattr(score, "SEED", 4)
    result = score.score_result(payload({
        "observational": .2,
        "omission_exposure": 2.0,
        "alignment_null": 1.8,
        "sensitivity": 2.0,
    }), n_bootstrap=20, seed=4)
    assert result["decisions"]["mlp2_suppression_replicates"]
    assert result["decisions"]["complete_compensation"]
    assert result["decisions"]["aligned_mlp2_write_compensates"]
    assert set(result["decisions"]["component_status"].values()) == {"equivalent"}


def test_powered_component_gets_non_null_status(monkeypatch):
    monkeypatch.setattr(score, "MIN_DOCUMENTS_PER_CELL", 1)
    monkeypatch.setattr(score, "N_BOOTSTRAP", 20)
    monkeypatch.setattr(score, "SEED", 5)
    result = score.score_result(payload({
        "prewrite_state": 2.0,
        "omission_exposure": 2.0,
        "sensitivity": 2.0,
    }), n_bootstrap=20, seed=5)
    assert result["decisions"]["component_status"]["prewrite_state"] == "powered_non_null"


def test_integrity_failure_closes_all_promotive_labels(monkeypatch):
    monkeypatch.setattr(score, "MIN_DOCUMENTS_PER_CELL", 1)
    candidate = payload({
        "observational": .2,
        "omission_exposure": 2.0,
        "alignment_null": 1.8,
        "sensitivity": 2.0,
    })
    candidate["integrity"]["phase_site_call_counts"]["crossed_forbidden_teacher"]["2"] = 1
    result = score.score_result(candidate, n_bootstrap=20, seed=6, authoritative=False)
    assert not result["integrity_passes"]
    assert not result["decisions"]["mlp2_suppression_replicates"]
    assert not result["decisions"]["complete_compensation"]
    assert not result["decisions"]["aligned_mlp2_write_compensates"]


def test_moving_arm_max_cannot_promote_reduction():
    coordinates = np.zeros((len(score.CONTRASTS), len(score.MARGINS) * 16))
    baseline = score.CONTRASTS.index("omission_exposure")
    candidate = score.CONTRASTS.index("observational")
    coordinates[baseline, 0] = .5
    coordinates[candidate, 1] = .4
    report = score.comparison_bounds(
        coordinates, .2,
        {"suppression": ("omission_exposure", "observational")},
    )["comparisons"]["suppression"]
    assert np.isclose(report["point_max_reduction"], .1)
    assert report["familywise_95pct_lcb_reduction"] < 0
    assert not report["candidate_pointwise_no_worse"]


def test_authoritative_inference_rejects_bootstrap_or_seed_override():
    candidate = payload({})
    for kwargs in ({"n_bootstrap": 20}, {"seed": score.SEED + 1}):
        try:
            score.score_result(candidate, **kwargs)
        except ValueError as error:
            assert "frozen" in str(error)
        else:
            raise AssertionError("authoritative inference accepted an override")


def test_orientation_wave_and_control_contracts_fail_closed():
    candidate = payload({})
    candidate["authority"]["inference_contract"]["contrast_orientations"][
        "observational"
    ] = ["CC", "OO"]
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])

    candidate = payload({})
    candidate["unit_identity"]["wave_labels"][0] = "B"
    try:
        score.validate_unit_identity(
            candidate["unit_identity"],
            candidate["authority"]["integrity_contract"]["unit_identity_hashes"],
        )
    except ValueError as error:
        assert "wave" in str(error)
    else:
        raise AssertionError("wave-label mutation passed")

    candidate = payload({})
    candidate["integrity"]["control_checks"]["derangement_no_same_document"] = False
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])


def test_scale_aware_native_control_gate_is_exact_and_fail_closed():
    candidate = payload({})
    controls = candidate["integrity"]["control_checks"]
    controls["native_control_norm_max_abs_error"] = 9.75e-4
    controls["native_control_norm_max_allowance_ratio"] = .97
    assert score.validate_integrity(candidate["authority"], candidate["integrity"])

    controls["native_control_norm_max_allowance_ratio"] = 1.0000001
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])

    candidate = payload({})
    candidate["authority"]["integrity_contract"]["native_control_norm_contract"][
        "rtol"
    ] = 1.1e-5
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])


def test_nan_tolerance_float_count_and_config_hash_fail_closed():
    candidate = payload({})
    candidate["authority"]["integrity_contract"]["same_realization_delta_tolerance"] = float("nan")
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])

    candidate = payload({})
    candidate["authority"]["integrity_contract"]["exact_call_counts"][
        "c512_proxy_calls"
    ] = 100.0
    candidate["integrity"]["call_counts"]["c512_proxy_calls"] = 100.0
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])

    candidate = payload({})
    del candidate["authority"]["integrity_contract"]["bound_hashes"]["model_config_sha256"]
    del candidate["integrity"]["observed_hashes"]["model_config_sha256"]
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])


def test_alignment_cannot_claim_compensation_without_exposed_suppression(monkeypatch):
    monkeypatch.setattr(score, "MIN_DOCUMENTS_PER_CELL", 1)
    monkeypatch.setattr(score, "N_BOOTSTRAP", 20)
    monkeypatch.setattr(score, "SEED", 7)
    result = score.score_result(payload({
        "observational": .2,
        "alignment_null": 1.8,
        "sensitivity": 2.0,
        "omission_exposure": .2,
    }), n_bootstrap=20, seed=7)
    assert not result["decisions"]["mlp2_suppression_replicates"]
    assert not result["decisions"]["aligned_mlp2_write_compensates"]


def test_nonauthoritative_fast_path_cannot_emit_promotive_decisions(monkeypatch):
    monkeypatch.setattr(score, "MIN_DOCUMENTS_PER_CELL", 1)
    result = score.score_result(payload({
        "observational": .2,
        "omission_exposure": 2.0,
        "alignment_null": 1.8,
        "sensitivity": 2.0,
    }), n_bootstrap=20, seed=8, authoritative=False)
    assert not result["authoritative_inference"]
    assert not result["common_gates"]["authoritative_inference"]
    assert not result["decisions"]["mlp2_suppression_replicates"]
    assert not result["decisions"]["complete_compensation"]
    assert not result["decisions"]["aligned_mlp2_write_compensates"]
    assert set(result["decisions"]["component_status"].values()) == {"inconclusive"}


def test_nonuniform_suffix_site_counts_fail_closed():
    candidate = payload({})
    candidate["authority"]["integrity_contract"]["exact_phase_site_call_counts"][
        "crossed_suffix_replay"
    ]["17"] += 1
    candidate["integrity"]["phase_site_call_counts"]["crossed_suffix_replay"]["17"] += 1
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])


def test_self_consistent_but_algebraically_wrong_call_counts_fail_closed():
    candidate = payload({})
    phases = candidate["authority"]["integrity_contract"][
        "exact_phase_site_call_counts"
    ]
    phases["mlp1_teacher_capture"]["1"] += 4
    candidate["integrity"]["phase_site_call_counts"]["mlp1_teacher_capture"]["1"] += 4
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])


def test_batch_count_is_bound_to_evaluation_window_identity(monkeypatch):
    monkeypatch.setattr(score, "MIN_DOCUMENTS_PER_CELL", 1)
    candidate = payload({})
    candidate["authority"]["integrity_contract"]["n_eval_windows"] = 100
    # Make all forward counts self-consistent with the false 100-window claim.
    fake = {
        "mlp1_teacher_capture": {"1": 100},
        "mlp2_teacher_capture": {"2": 100},
        "parent_replay_mlp_sites": {
            **{str(i): 100 for i in range(18)}, "2": 50,
        },
        "crossed_suffix_replay": {str(i): 200 for i in range(3, 18)},
        "crossed_forbidden_teacher": {"1": 0, "2": 0},
    }
    candidate["authority"]["integrity_contract"]["exact_call_counts"][
        "c512_proxy_calls"
    ] = 100
    candidate["integrity"]["call_counts"]["c512_proxy_calls"] = 100
    candidate["authority"]["integrity_contract"]["exact_phase_site_call_counts"] = fake
    candidate["integrity"]["phase_site_call_counts"] = {
        phase: dict(values) for phase, values in fake.items()
    }
    result = score.score_result(
        candidate, n_bootstrap=20, seed=8, authoritative=False
    )
    assert not result["common_gates"]["evaluation_windows_match_unit_identity"]


def test_reported_coverage_must_equal_common_ledger_support(monkeypatch):
    monkeypatch.setattr(score, "MIN_DOCUMENTS_PER_CELL", 1)
    candidate = payload({})
    candidate["coverage"]["wave_A"] = .99
    result = score.score_result(
        candidate, n_bootstrap=20, seed=8, authoritative=False
    )
    assert not result["common_gates"]["reported_coverage_matches_common_ledger"]


def test_scoring_currency_must_equal_inherited_numeric():
    candidate = payload({})
    candidate["integrity"]["scoring_currency"]["centered_capped_logit_rms"] = 2.6
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])


def test_actual_only_float_or_bool_call_counts_fail_closed():
    candidate = payload({})
    candidate["integrity"]["call_counts"]["c512_proxy_calls"] = 100.0
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])

    candidate = payload({})
    candidate["integrity"]["phase_site_call_counts"]["crossed_forbidden_teacher"][
        "2"
    ] = False
    assert not score.validate_integrity(candidate["authority"], candidate["integrity"])
