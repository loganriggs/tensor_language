import numpy as np

import score_mlp0_c512_mlp1_interchange_v1 as score


def synthetic_ledgers(n_units, effects=None, n_cells=16):
    effects = effects or {}
    ledgers = {}
    for arm in score.ARMS:
        ledgers[arm] = {}
        value = effects.get(arm, 0.2)
        for consumer, margin in score.MARGINS.items():
            counts = np.ones((n_units, n_cells))
            sums = counts * margin * value
            ledgers[arm][consumer] = {"sums": sums.tolist(), "counts": counts.tolist()}
    return ledgers


def synthetic_identity():
    fineweb_counts = [2] * 248 + [4] * 71 + [6] * 65
    fineweb_mapping = [unit for unit, count in enumerate(fineweb_counts) for _ in range(count)]
    return {
        "fineweb": {
            "unit_kind": "source_document",
            "ordered_ids": [f"doc-{index}" for index in range(384)],
            "row_to_unit": fineweb_mapping,
        },
        "code": {
            "unit_kind": "source_file",
            "ordered_ids": [f"file-{index}" for index in range(48)],
            "row_to_unit": [unit for unit in range(48) for _ in range(4)],
        },
    }


def authority_binding(identity):
    return {
        "status": "frozen_before_any_c512_mlp1_evaluation_forward",
        "integrity_contract": {
            "exact_call_counts": {
                "candidate_original_down_calls": 0,
                "poison_canary_calls": 1,
                "mlp1_teacher_calls": 100,
                "c512_proxy_calls": 50,
            },
            "bound_hashes": {
                "source_closure_sha256": "a" * 64,
                "row_receipt_sha256": "b" * 64,
                "row_tensor_sha256": "c" * 64,
                "c512_program_sha256": "d" * 64,
                "model_checkpoint_sha256": "e" * 64,
                "code_register_sha256": "f" * 64,
            },
            "parent_replay_tolerances": {
                "raw_logits_max_abs": 1e-6,
                "capped_logits_max_abs": 1e-6,
                "ce_abs": 1e-7,
            },
            "unit_identity_hashes": {
                domain: {
                    "ordered_ids_sha256": score.ordered_ids_sha256(record["ordered_ids"]),
                    "row_to_unit_sha256": score.integer_array_sha256(record["row_to_unit"]),
                    "occupancy_sha256": score.integer_array_sha256(
                        np.bincount(record["row_to_unit"], minlength=len(record["ordered_ids"]))
                    ),
                }
                for domain, record in identity.items()
            },
        },
    }


def passing_integrity(authority):
    return {
        "call_counts": dict(authority["integrity_contract"]["exact_call_counts"]),
        "observed_hashes": dict(authority["integrity_contract"]["bound_hashes"]),
        "parent_replay": {
            background: {
                "raw_logits_max_abs": 0.0, "capped_logits_max_abs": 0.0,
                "ce_abs": 0.0, "passes": True,
            }
            for background in score.BACKGROUNDS
        },
    }


def test_signed_ce_sign_flip_gets_two_sided_centered_error():
    coordinates = np.zeros((1, len(score.MARGINS) * 16))
    bootstrap = np.zeros((3, 1, len(score.MARGINS) * 16))
    ce_offset = list(score.MARGINS).index("ce_abs") * 16
    coordinates[0, ce_offset] = .2
    bootstrap[:, 0, ce_offset] = -.2
    upper, lower, correction, lower_correction, joint = score.familywise_bounds(coordinates, bootstrap)
    assert np.isclose(correction, .4) and np.isclose(lower_correction, .4) and np.isclose(joint, .4)
    assert np.isclose(upper[0], .6) and np.isclose(lower[0], -.2)


def test_familywise_bounds_center_coordinates_before_global_max():
    coordinates = np.zeros((1, len(score.MARGINS) * 16))
    coordinates[0, :2] = [.50, .49]
    bootstrap = np.repeat(coordinates[None], 3, axis=0)
    bootstrap[:, 0, 1] = .58
    upper, _, correction, _, joint = score.familywise_bounds(coordinates, bootstrap)
    assert np.isclose(correction, .09)
    assert np.isclose(joint, .09)
    assert np.isclose(upper[0], .59)


def test_one_common_bootstrap_family_contains_every_registered_arm(monkeypatch):
    monkeypatch.setattr(score, "MIN_FINEWEB_DOCUMENTS_PER_CELL", 1)
    ledger = synthetic_ledgers(12)
    scope = score.score_scope(ledger, np.arange(12), minimum_support=1, n_bootstrap=20, seed=3)
    assert set(scope["arms"]) == set(score.ARMS)
    assert scope["bootstrap"]["replicates"] == 20


def test_rescue_comparison_is_positive_when_upstream_state_is_better(monkeypatch):
    effects = {
        "live/observational_CC": .8,
        "live/upstream_state": .2,
        "mlp2_omit/observational_CC": .7,
        "mlp2_omit/upstream_state": .3,
    }
    scope = score.score_scope(
        synthetic_ledgers(20, effects), np.arange(20), minimum_support=1,
        n_bootstrap=50, seed=4,
    )
    assert scope["rescue"]["comparisons"]["rescue_live"]["familywise_95pct_lcb_reduction"] > 0


def test_rescue_band_cannot_promote_switching_near_tie():
    coordinates = np.zeros((len(score.ARMS), len(score.MARGINS) * 16))
    baseline = score.ARMS.index("live/observational_CC")
    candidate = score.ARMS.index("live/upstream_state")
    coordinates[baseline, 0] = .50
    coordinates[candidate, 1] = .40
    report = score.comparison_bounds(
        coordinates, .20,
        {"rescue_live": ("live/observational_CC", "live/upstream_state")},
    )["comparisons"]["rescue_live"]
    assert np.isclose(report["point_max_reduction"], .10)
    assert report["familywise_95pct_lcb_reduction"] < 0
    assert not report["candidate_pointwise_no_worse"]


def test_joint_correction_uses_replicatewise_two_sided_family_event():
    coordinates = np.zeros((1, len(score.MARGINS) * 16))
    bootstrap = np.zeros((20, 1, len(score.MARGINS) * 16))
    bootstrap[:10, 0, 0] = .7
    bootstrap[10:, 0, 0] = -.6
    _, _, upper, lower, joint = score.familywise_bounds(coordinates, bootstrap)
    assert np.isclose(upper, .7)
    assert np.isclose(lower, .6)
    assert np.isclose(joint, .7)
    report = score.comparison_bounds(
        np.zeros((len(score.ARMS), len(score.MARGINS) * 16)), joint,
        {"rescue_live": ("live/observational_CC", "live/upstream_state")},
    )["comparisons"]["rescue_live"]
    assert np.isclose(report["familywise_95pct_lcb_reduction"], -1.4)


def test_full_decision_requires_sensitive_positive_controls(monkeypatch):
    monkeypatch.setattr(score, "MIN_FINEWEB_DOCUMENTS_PER_CELL", 1)
    monkeypatch.setattr(score, "MIN_CODE_FILES_PER_CELL", 1)
    effects = {
        "live/native_write": 2.0,
        "mlp2_omit/native_write": 2.0,
    }
    identity = synthetic_identity()
    authority = authority_binding(identity)
    payload = {
        "sufficient_statistics": {
            "fineweb": synthetic_ledgers(384, effects),
            "code": synthetic_ledgers(48, effects),
        },
        "coverage": {"fineweb": {"wave_A": 1.0, "wave_B": 1.0, "pooled": 1.0}, "code": 1.0},
        "unit_identity": identity,
        "authority": authority,
        "integrity": passing_integrity(authority),
    }
    result = score.score_result(payload, n_bootstrap=30, seed=5)
    assert result["decisions"]["downstream_null_on_registered_fineweb_backgrounds"]
    assert result["decisions"]["positive_control_each_background"] == {
        "live": True, "mlp2_omit": True
    }


def test_promotive_decisions_fail_closed_without_integrity(monkeypatch):
    monkeypatch.setattr(score, "MIN_FINEWEB_DOCUMENTS_PER_CELL", 1)
    monkeypatch.setattr(score, "MIN_CODE_FILES_PER_CELL", 1)
    effects = {"live/native_write": 2.0, "mlp2_omit/native_write": 2.0}
    identity = synthetic_identity()
    authority = authority_binding(identity)
    integrity = passing_integrity(authority)
    integrity["call_counts"]["candidate_original_down_calls"] = 1
    payload = {
        "sufficient_statistics": {
            "fineweb": synthetic_ledgers(384, effects),
            "code": synthetic_ledgers(48, effects),
        },
        "coverage": {"fineweb": {"wave_A": 1.0, "wave_B": 1.0, "pooled": 1.0}, "code": 1.0},
        "unit_identity": identity,
        "authority": authority,
        "integrity": integrity,
    }
    result = score.score_result(payload, n_bootstrap=20, seed=9)
    assert not result["integrity_passes"]
    assert not result["decisions"]["downstream_null_on_registered_fineweb_backgrounds"]
    assert not result["decisions"]["mlp1_repair_license_live"]


def test_integrity_rejects_nan_replay_and_empty_hash_contract():
    identity = synthetic_identity()
    authority = authority_binding(identity)
    integrity = passing_integrity(authority)
    integrity["parent_replay"]["live"]["raw_logits_max_abs"] = float("nan")
    assert not score.validate_integrity(authority, integrity)
    integrity = passing_integrity(authority)
    authority["integrity_contract"]["bound_hashes"] = {}
    integrity["observed_hashes"] = {}
    assert not score.validate_integrity(authority, integrity)
