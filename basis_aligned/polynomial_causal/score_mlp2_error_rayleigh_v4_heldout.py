"""Receipt-backed evaluation of the frozen MLP2 error-Rayleigh HELDOUT assay.

The scientific gates and fitted predictor were frozen before HELDOUT access.  This
reporter makes no fit and exposes every reduction used to turn the receipt-complete
HELDOUT ledger into the preregistered pass/fail decision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

import mlp2_error_rayleigh_collector_core as core
import mlp2_error_rayleigh_metrics as metrics
import mlp2_error_rayleigh_predictor as predictor


HERE = Path(__file__).resolve().parent
LEDGER = HERE / "mlp2_error_rayleigh_v4_heldout_ledger.pt"
COLLECTION_RECEIPT = HERE / "mlp2_error_rayleigh_v4_heldout_receipt.json"
BUNDLE = HERE / "mlp2_error_rayleigh_v4_design_predictor_bundle.pt"
PREDICTOR_RECEIPT = HERE / "mlp2_error_rayleigh_v4_design_predictor_receipt.json"
OUTPUT = HERE / "mlp2_error_rayleigh_v4_heldout_score_receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _predict(
    contrasts: dict[str, torch.Tensor], bundle: dict[str, Any], family: str,
) -> torch.Tensor:
    model = bundle["models"][family]
    matrix = predictor.family_matrix(contrasts, family)
    return predictor.predict(
        predictor.normalize(matrix, model["mean"], model["scale"]),
        model["coefficients"],
    )


def evaluate(ledger: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen gates without fitting, threshold selection, or resampling."""
    if ledger.get("schema") != "mlp2_error_rayleigh_v1_role_ledger" \
            or ledger.get("role") != "HELDOUT":
        raise RuntimeError("not the frozen HELDOUT ledger")
    predictor.validate_frozen_bundle(bundle)
    features, finite = ledger["features"], ledger["finite"]
    expected_features = (3, 2, 3, 32, len(core.FEATURE_NAMES))
    expected_finite = (3, 2, 32, len(core.FINITE_NAMES))
    if features.dtype != torch.float64 or features.shape != expected_features \
            or finite.dtype != torch.float64 or finite.shape != expected_finite \
            or not torch.isfinite(features).all() or not torch.isfinite(finite).all():
        raise RuntimeError("HELDOUT sufficient-statistic layout changed")

    fi = {name: index for index, name in enumerate(core.FEATURE_NAMES)}
    ni = {name: index for index, name in enumerate(core.FINITE_NAMES)}

    # The endpoint replay was a prerequisite for admitting any scientific metric.
    replay = {
        "max_abs_logits": float(finite[..., ni["logits_max_abs"]].max()),
        "max_abs_attention5": float(finite[..., ni["attention5_max_abs"]].max()),
        "max_abs_attention6": float(finite[..., ni["attention6_max_abs"]].max()),
        "all_logits_exact": bool(finite[..., ni["logits_exact"]].bool().all()),
        "all_attention5_exact": bool(finite[..., ni["attention5_exact"]].bool().all()),
        "all_attention6_exact": bool(finite[..., ni["attention6_exact"]].bool().all()),
        "max_abs_direct_vs_injected_dce": float((
            finite[..., ni["direct_dce"]] - finite[..., ni["injected_dce"]]
        ).abs().max()),
    }
    replay["passes"] = all((
        replay["max_abs_logits"] == 0.0,
        replay["max_abs_attention5"] == 0.0,
        replay["max_abs_attention6"] == 0.0,
        replay["all_logits_exact"],
        replay["all_attention5_exact"],
        replay["all_attention6_exact"],
        replay["max_abs_direct_vs_injected_dce"] == 0.0,
    ))
    if not replay["passes"]:
        raise RuntimeError("alpha=1 injection does not exactly replay the physical program")

    # One value per independent source document: average over the six frozen
    # program/background cells, retaining no token as an independent observation.
    q16 = features[:, :, 0, :, fi["qlogit_h16"]].mean(dim=(0, 1))
    q8 = features[:, :, 0, :, fi["qlogit_h8"]].mean(dim=(0, 1))
    tangent = metrics.tangent_scale_gate(q16, q8)
    kl8 = 0.5 * (
        features[:, :, 0, :, fi["kl_minus_h8"]]
        + features[:, :, 0, :, fi["kl_plus_h8"]]
    ).mean(dim=(0, 1))
    fisher = metrics.fisher_kl_gate(kl8, q8, 1.0 / 8.0)

    actual, target = predictor.regression_units(features, finite, 0)
    predictions = {
        family: _predict(actual, bundle, family) for family in predictor.FAMILIES
    }
    predictive = metrics.predictor_gate(
        target.flatten(), predictions["LOCAL"].flatten(),
        predictions["FINAL"].flatten(), predictions["FULL"].flatten(),
    )
    interaction = metrics.finite_interaction_gate(
        predictions["FULL"].mean(dim=0), target.mean(dim=0),
    )
    interaction.update({
        "programs": list(ledger["axes"]["programs"]),
        "predicted_mean_by_program": predictions["FULL"].mean(dim=0).tolist(),
        "observed_mean_by_program": target.mean(dim=0).tolist(),
    })

    controls: dict[str, Any] = {}
    for control_index, control in enumerate(core.CONTROL_NAMES[1:], start=1):
        contrasts, null_target = predictor.regression_units(features, finite, control_index)
        null_predictions = {
            family: _predict(contrasts, bundle, family) for family in predictor.FAMILIES
        }
        null_predictive = metrics.predictor_gate(
            null_target.flatten(), null_predictions["LOCAL"].flatten(),
            null_predictions["FINAL"].flatten(), null_predictions["FULL"].flatten(),
        )
        null_interaction = metrics.finite_interaction_gate(
            null_predictions["FULL"].mean(dim=0), null_target.mean(dim=0),
        )
        controls[control] = {
            "predictor_gate": null_predictive,
            "finite_interaction_gate": null_interaction,
            "fails_corresponding_predictive_gates": not (
                null_predictive["passes"] and null_interaction["passes"]
            ),
        }

    primary = {
        "gate_1_tangent_scale": tangent,
        "gate_2_fisher_teacher_kl": fisher,
        "gate_3_frozen_predictor": predictive,
        "gate_4_finite_interaction": interaction,
    }
    controls_pass = all(
        value["fails_corresponding_predictive_gates"] for value in controls.values()
    )
    return {
        "schema": "mlp2_error_rayleigh_v4_heldout_score_receipt",
        "status": "PASS" if replay["passes"] and all(
            value["passes"] for value in primary.values()
        ) and controls_pass else "FAIL",
        "frozen_predictor_only_no_heldout_fit": True,
        "endpoint_replay": replay,
        "primary_gates": primary,
        "gate_5_negative_controls": {"controls": controls, "passes": controls_pass},
        "decision": (
            "authorize_equal_price_consequence_weighted_MLP2_fit"
            if all(value["passes"] for value in primary.values()) and controls_pass
            else "reject_current_adjoint_Fisher_metric_and_do_not_train_from_it"
        ),
        "aggregation": {
            "tangent_and_Fisher": "mean programs/backgrounds, then 32 source documents",
            "predictor_MSE_and_Spearman": "96 document-program rows; no token pseudo-replication",
            "finite_interaction": "mean over 32 documents separately for each program",
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("HELDOUT score receipt already exists")
    collection_receipt = json.loads(COLLECTION_RECEIPT.read_text())
    predictor_receipt = json.loads(PREDICTOR_RECEIPT.read_text())
    if sha256(LEDGER) != collection_receipt.get("ledger_sha256") \
            or sha256(BUNDLE) != predictor_receipt.get("predictor_bundle_sha256"):
        raise RuntimeError("receipt-bound HELDOUT input changed")
    ledger = torch.load(LEDGER, map_location="cpu", weights_only=True)
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=True)
    report = evaluate(ledger, bundle)
    report["inputs"] = {
        "heldout_ledger_sha256": sha256(LEDGER),
        "heldout_collection_receipt_sha256": sha256(COLLECTION_RECEIPT),
        "frozen_predictor_bundle_sha256": sha256(BUNDLE),
        "frozen_predictor_receipt_sha256": sha256(PREDICTOR_RECEIPT),
    }
    payload = (json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n").encode()
    descriptor = OUTPUT.open("xb")
    with descriptor:
        descriptor.write(payload)
        descriptor.flush()
    print(payload.decode(), end="")


if __name__ == "__main__":
    main()
