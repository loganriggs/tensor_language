import torch

import mlp2_error_rayleigh_collector_core as core
import mlp2_error_rayleigh_predictor as predictor
import run_mlp2_error_rayleigh_v1_score_design as score


def test_serialized_predictor_bundle_has_exact_deployable_schema():
    models = {}
    for family, names in predictor.FAMILIES.items():
        width = len(names)
        models[family] = {
            "ridge": {"selected": 0.01,
                      "clustered_lodo_mse": {penalty: 1.0 for penalty in predictor.RIDGE_GRID}},
            "mean": torch.zeros(width, dtype=torch.float64),
            "scale": torch.ones(width, dtype=torch.float64),
            "coefficients": torch.ones(width + 1, dtype=torch.float64),
            "design_prediction": torch.zeros(32, 3, dtype=torch.float64),
        }
    fit = {
        "target": torch.zeros(32, 3, dtype=torch.float64), "models": models,
        "null_predictions": {control: {
            family: torch.zeros(32, 3, dtype=torch.float64) for family in predictor.FAMILIES
        } for control in ("DERANGED", "COV_RANDOM")},
    }
    value = score.serialize_fit(fit)
    assert score.validate_bundle(value) is value
    assert value["program_identity_feature"] is False
    assert value["directional_amplitude_reduction"] == "arithmetic_mean_h16_h8"


def test_source_closure_contains_scorer_predictor_and_collector_contracts():
    for path in (score.RUNNER, score.TEST, score.HERE / "mlp2_error_rayleigh_predictor.py",
                 score.HERE / "test_mlp2_error_rayleigh_predictor.py",
                 score.collector.RUNNER, score.collector.TEST, score.collector.ADDENDUM):
        assert score.SOURCE_PATHS.count(path) == 1
    assert set(score.collector.SOURCE_PATHS).issubset(score.SOURCE_PATHS)


def test_receipt_shape_matches_heldout_unlock_exactly():
    required = {
        "schema", "status", "design_ledger_sha256", "design_receipt_sha256",
        "predictor_authority_sha256", "scorer_audit_sha256",
        "predictor_bundle_sha256", "heldout_unlocked",
    }
    assert required == {
        "schema", "status", "design_ledger_sha256", "design_receipt_sha256",
        "predictor_authority_sha256", "scorer_audit_sha256",
        "predictor_bundle_sha256", "heldout_unlocked",
    }
    assert score.RECEIPT == score.collector.PREDICTOR_RECEIPT
