"""Model-free capability binding for the unfiltered v24 four-head OOD confirmation."""

# BQGATE: LIBRARY
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
CAPABILITY_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v24_capability_v1_result.json"
CAPABILITY_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v24_capability_v1.py"
CONFIRMATION_RUNNER = ROOT / "ops/run_temporal_iswas_v24_unfiltered_four_head_ood_confirmation_v1.py"
BINDING = ROOT / "circuits/bindings/temporal_iswas_v24_unfiltered_four_head_ood_confirmation_v1.json"
EXPECTED_CAPABILITY_RUNNER_SHA256 = "6c11f38c6cec2c4f9cfae7180b39e0b1782ee0c10f60d94c9e03dcdfa20db210"
EXPECTED_CONFIRMATION_RUNNER_SHA256 = "52071e28b8b196242f5f058d3b06cfb39178dea76607d7820517aeaec71367df"
ROWS_SHA256 = "870c829290e1791351d0b2b67985aa5700780920b14423906e7b8fd4d35ed2de"
FAMILIES = ("A1", "A2", "P", "C")
PREDICTION_KEYS = (
    "pred_a_authority_novelty_and_exact_population",
    "pred_b_native_target_capability",
    "pred_c_native_control_capability",
    "pred_d_no_causal_outcome_access_and_exact_price",
)


class V24ConfirmationBindingError(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_hash(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise V24ConfirmationBindingError(f"{label} is not a lowercase SHA-256")


def binding_payload(result, *, result_sha256, capability_runner_sha256,
                    confirmation_runner_sha256):
    for value, label in (
        (result_sha256, "capability result hash"),
        (capability_runner_sha256, "capability runner hash"),
        (confirmation_runner_sha256, "confirmation runner hash"),
    ):
        _validate_hash(value, label)
    if result.get("schema") != "tense_auxiliary_is_was_fresh_lexicon_v24_capability_result_v1":
        raise V24ConfirmationBindingError("capability result schema changed")
    predictions = result.get("predictions")
    if (not isinstance(predictions, dict) or tuple(predictions) != PREDICTION_KEYS
            or any(value is not True for value in predictions.values())):
        raise V24ConfirmationBindingError("capability predictions do not license confirmation")
    if (result.get("terminal") != "screen"
            or result.get("causal_outcomes_opened") is not False
            or result.get("rows_sha256") != ROWS_SHA256
            or set(result.get("jointly_capable_row_ids", {})) != set(FAMILIES)
            or result.get("family_capable") != {family: True for family in FAMILIES}):
        raise V24ConfirmationBindingError("capability population or outcome firewall changed")
    price = result.get("price", {})
    if (price.get("model_forwards") != 2 or price.get("example_evaluations") != 128
            or price.get("interventions") != 0 or price.get("transformer_backwards") != 0
            or price.get("model_updates") != 0):
        raise V24ConfirmationBindingError("capability execution price changed")
    return {
        "schema": "temporal_iswas_v24_ood_confirmation_binding_v1",
        "capability_result_sha256": result_sha256,
        "capability_runner_sha256": capability_runner_sha256,
        "confirmation_runner_sha256": confirmation_runner_sha256,
        "rows_sha256": ROWS_SHA256,
        "all_families_capable": True,
        "causal_outcomes_opened": False,
    }


def create_binding(*, capability_result=CAPABILITY_RESULT,
                   capability_runner=CAPABILITY_RUNNER,
                   confirmation_runner=CONFIRMATION_RUNNER,
                   binding=BINDING, dry_run=False):
    capability_result = Path(capability_result)
    capability_runner = Path(capability_runner)
    confirmation_runner = Path(confirmation_runner)
    binding = Path(binding)
    if not capability_result.exists():
        return {"status": "awaiting_result", "created": False}
    capability_runner_hash = sha256(capability_runner)
    confirmation_runner_hash = sha256(confirmation_runner)
    if capability_runner_hash != EXPECTED_CAPABILITY_RUNNER_SHA256:
        raise V24ConfirmationBindingError("capability runner hash changed")
    if confirmation_runner_hash != EXPECTED_CONFIRMATION_RUNNER_SHA256:
        raise V24ConfirmationBindingError("confirmation runner hash changed")
    result_hash = sha256(capability_result)
    payload = binding_payload(
        json.loads(capability_result.read_text()), result_sha256=result_hash,
        capability_runner_sha256=capability_runner_hash,
        confirmation_runner_sha256=confirmation_runner_hash,
    )
    if dry_run:
        return {"status": "eligible", "created": False, "payload": payload}
    if binding.exists():
        if json.loads(binding.read_text()) != payload:
            raise V24ConfirmationBindingError("binding exists with different authority")
        return {"status": "bound", "created": False, "payload": payload}
    atomic_create_json(binding, payload)
    return {"status": "bound", "created": True, "payload": payload}
