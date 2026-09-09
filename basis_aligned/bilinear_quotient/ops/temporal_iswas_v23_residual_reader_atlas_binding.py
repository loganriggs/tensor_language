"""Model-free result binding for the v23 residual-carrier downstream reader atlas."""

# BQGATE: LIBRARY
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
FACTORIAL_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_block11_residual_mlp_factorial_rescue_v1_result.json"
FACTORIAL_RUNNER = ROOT / "ops/run_temporal_iswas_v23_block11_residual_mlp_factorial_rescue_v1.py"
PRECISION_AUDIT = ROOT / "circuits/audits/temporal_iswas_v23_block11_factorial_precision_audit_v1.json"
ATLAS_RUNNER = ROOT / "ops/run_temporal_iswas_v23_residual_carrier_downstream_module_reader_atlas_v1.py"
BINDING = ROOT / "circuits/bindings/temporal_iswas_v23_residual_carrier_downstream_module_reader_atlas_v1.json"
EXPECTED_FACTORIAL_RUNNER_SHA256 = "7aac1b15c2090e4812bd4de6f0adb78a58ee34e4c7598ecc77ba60ebe012bff7"
EXPECTED_ATLAS_RUNNER_SHA256 = "421bbf43d5a93df00785820a5efc02e7ce00d05e65868a6de70b729a0f2ceacb"
EXPECTED_PRECISION_AUDIT_SHA256 = "9b8ddd4ea12c03dd91f3affbe1e343f5de2f16e2c2f1d1ab3302e082cfa0faa3"
PREDICTION_KEYS = (
    "pred_a_authority_hooks_closures_self_patch_finiteness_and_exact_price",
    "pred_b_residual_carrier_is_the_dominant_selective_branch",
    "pred_c_m11_is_a_real_but_minor_branch",
    "pred_d_exact_joint_rescue_closes_the_block11_mediation",
)
REQUIRED = (PREDICTION_KEYS[0], PREDICTION_KEYS[1])


class ResidualReaderBindingError(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_hash(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ResidualReaderBindingError(f"{label} is not a lowercase SHA-256")


def binding_payload(result, audit, *, result_sha256, audit_sha256,
                    factorial_runner_sha256, atlas_runner_sha256):
    """Validate one immutable factorial receipt and return the atlas license."""
    for value, label in (
        (result_sha256, "factorial result hash"),
        (audit_sha256, "precision audit hash"),
        (factorial_runner_sha256, "factorial runner hash"),
        (atlas_runner_sha256, "atlas runner hash"),
    ):
        _validate_hash(value, label)
    if result.get("schema") != "temporal_iswas_v23_block11_residual_mlp_factorial_rescue_result_v1":
        raise ResidualReaderBindingError("factorial result schema changed")
    predictions = result.get("predictions")
    if (not isinstance(predictions, dict) or tuple(predictions) != PREDICTION_KEYS
            or any(not isinstance(value, bool) for value in predictions.values())):
        raise ResidualReaderBindingError("factorial prediction inventory changed")
    if not all(predictions[key] for key in REQUIRED):
        raise ResidualReaderBindingError("factorial does not license residual reader localization")
    if (audit_sha256 != EXPECTED_PRECISION_AUDIT_SHA256
            or audit.get("schema") != "temporal_iswas_v23_block11_factorial_precision_audit_v1"
            or audit.get("factorial_result_sha256") != result_sha256
            or audit.get("original_terminal") != "invalid_instrument"
            or audit.get("original_factorial_relabelled") is not False
            or audit.get("downstream_reader_license") is not True
            or audit.get("license_scope") !=
                "block11_residual_correction_to_A12_M17_reader_atlas_only"
            or not all(audit.get("predictions", {}).values())):
        raise ResidualReaderBindingError("scoped precision audit does not license reader localization")
    if result.get("selected_component_or_threshold_after_outcome") is not None:
        raise ResidualReaderBindingError("factorial reports post-outcome selection")
    price = result.get("price", {})
    if (price.get("model_forwards_exact") != 7
            or price.get("sequence_evaluations_exact") != 448):
        raise ResidualReaderBindingError("factorial execution price changed")
    return {
        "schema": "temporal_iswas_v23_residual_reader_atlas_binding_v2",
        "factorial_result_sha256": result_sha256,
        "precision_audit_sha256": audit_sha256,
        "factorial_runner_sha256": factorial_runner_sha256,
        "atlas_runner_sha256": atlas_runner_sha256,
        "required_predictions": {key: predictions[key] for key in REQUIRED},
    }


def create_binding(*, factorial_result=FACTORIAL_RESULT,
                   precision_audit=PRECISION_AUDIT,
                   factorial_runner=FACTORIAL_RUNNER, atlas_runner=ATLAS_RUNNER,
                   binding=BINDING, dry_run=False):
    """Create the unique eligible binding, or report that the result is pending."""
    factorial_result = Path(factorial_result)
    factorial_runner = Path(factorial_runner)
    precision_audit = Path(precision_audit)
    atlas_runner = Path(atlas_runner)
    binding = Path(binding)
    if not factorial_result.exists() or not precision_audit.exists():
        return {"status": "awaiting_result", "created": False}
    factorial_runner_hash = sha256(factorial_runner)
    atlas_runner_hash = sha256(atlas_runner)
    if factorial_runner_hash != EXPECTED_FACTORIAL_RUNNER_SHA256:
        raise ResidualReaderBindingError("factorial runner hash changed")
    if atlas_runner_hash != EXPECTED_ATLAS_RUNNER_SHA256:
        raise ResidualReaderBindingError("atlas runner hash changed")
    result_hash = sha256(factorial_result)
    audit_hash = sha256(precision_audit)
    payload = binding_payload(
        json.loads(factorial_result.read_text()), json.loads(precision_audit.read_text()),
        result_sha256=result_hash, audit_sha256=audit_hash,
        factorial_runner_sha256=factorial_runner_hash,
        atlas_runner_sha256=atlas_runner_hash,
    )
    if dry_run:
        return {"status": "eligible", "created": False, "payload": payload}
    if binding.exists():
        if json.loads(binding.read_text()) != payload:
            raise ResidualReaderBindingError("binding exists with different authority")
        return {"status": "bound", "created": False, "payload": payload}
    atomic_create_json(binding, payload)
    return {"status": "bound", "created": True, "payload": payload}
