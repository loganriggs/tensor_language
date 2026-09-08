"""Model-free authority binding for result-conditional P7 split executors."""

# BQGATE: LIBRARY
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
NECESSITY_RESULT = ROOT / "circuits/followups/temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1_result.json"
NECESSITY_RUNNER = ROOT / "ops/run_temporal_iswas_p7_identity_background_module_leave_one_out_ood_v1.py"
EXPECTED_NECESSITY_RUNNER_SHA256 = "384a79a38294416204d5bde0e24fcee7008e391a6712de82ec59c627f7f1318a"
P7 = ("A11", "M11", "M12", "M15", "M13", "M16", "M10")
PRED_A = "pred_a_authority_capture_full_replay_finiteness_and_exact_price"
PRED_B = "pred_b_at_least_two_p7_modules_are_stably_necessary"
PRED_C = "pred_c_attention11_is_stably_necessary_and_licenses_head_splitting"
PRED_D = "pred_d_at_least_one_p7_mlp_is_stably_necessary_and_licenses_weight_factor_splitting"
PRED_E = "pred_e_every_leave_one_out_arm_is_temporally_selective"
BRANCHES = {
    "a11": {
        "member": "A11", "required_predictions": (PRED_A, PRED_C, PRED_E),
        "runner": ROOT / "ops/run_temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1.py",
        "expected_runner_sha256": "417aaa017c06116f69141dbaaa683c511b955c5f80f59f7ccda0508cabbbf3e7",
        "binding": ROOT / "circuits/bindings/temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1.json",
    },
    "m11": {
        "member": "M11", "required_predictions": (PRED_A, PRED_D, PRED_E),
        "runner": ROOT / "ops/run_temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1.py",
        "expected_runner_sha256": "6551339eeec03b3121ed23fb274d24adc02a8654cd5dc6555dfe3a5158f98b43",
        "binding": ROOT / "circuits/bindings/temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1.json",
    },
}


class P7SplitBindingError(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_hash(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise P7SplitBindingError(f"{label} is not a lowercase SHA-256")


def binding_plan(result, *, result_sha256, necessity_runner_sha256, branches=BRANCHES):
    """Validate the immutable receipt and return eligible branch binding payloads."""
    _validate_hash(result_sha256, "necessity result hash")
    _validate_hash(necessity_runner_sha256, "necessity runner hash")
    if result.get("schema") != "temporal_iswas_p7_identity_background_module_leave_one_out_ood_result_v1":
        raise P7SplitBindingError("necessity result schema changed")
    if tuple(result.get("P7", ())) != P7:
        raise P7SplitBindingError("necessity result P7 changed")
    predictions = result.get("predictions")
    stable = tuple(result.get("stable_necessary_modules", ()))
    if (not isinstance(predictions, dict) or set(predictions) != {
            PRED_A, PRED_B, PRED_C, PRED_D, PRED_E}
            or any(not isinstance(value, bool) for value in predictions.values())
            or len(stable) != len(set(stable)) or not set(stable).issubset(P7)):
        raise P7SplitBindingError("necessity predictions or stable-module set changed")
    derived = {
        PRED_B: len(stable) >= 2,
        PRED_C: "A11" in stable,
        PRED_D: any(member.startswith("M") for member in stable),
    }
    if any(predictions[key] is not value for key, value in derived.items()):
        raise P7SplitBindingError("necessity predictions disagree with stable modules")

    payloads = {}
    for name, branch in branches.items():
        runner_hash = branch["expected_runner_sha256"]
        _validate_hash(runner_hash, f"{name} expected runner hash")
        eligible = (branch["member"] in stable
                    and all(predictions[key] for key in branch["required_predictions"]))
        if eligible:
            payloads[name] = {
                "schema": "temporal_iswas_p7_identity_split_binding_v1",
                "branch": name,
                "necessity_result_sha256": result_sha256,
                "necessity_runner_sha256": necessity_runner_sha256,
                "branch_runner_sha256": runner_hash,
                "stable_necessary_modules": list(stable),
                "required_predictions": {
                    key: predictions[key] for key in branch["required_predictions"]
                },
            }
    return payloads


def create_bindings(*, necessity_result=NECESSITY_RESULT,
                    necessity_runner=NECESSITY_RUNNER,
                    expected_necessity_runner_sha256=EXPECTED_NECESSITY_RUNNER_SHA256,
                    branches=BRANCHES, dry_run=False):
    """Create only eligible bindings, refusing changed runners and stale files."""
    necessity_result, necessity_runner = Path(necessity_result), Path(necessity_runner)
    if not necessity_result.exists():
        return {"status": "awaiting_result", "created": [], "existing": [], "eligible": []}
    observed_necessity_runner = sha256(necessity_runner)
    if observed_necessity_runner != expected_necessity_runner_sha256:
        raise P7SplitBindingError("necessity runner hash changed")
    for name, branch in branches.items():
        if sha256(branch["runner"]) != branch["expected_runner_sha256"]:
            raise P7SplitBindingError(f"{name} branch runner hash changed")
    result_hash = sha256(necessity_result)
    result = json.loads(necessity_result.read_text())
    payloads = binding_plan(
        result, result_sha256=result_hash,
        necessity_runner_sha256=observed_necessity_runner, branches=branches,
    )
    if dry_run:
        return {"status": "dry_run", "created": [], "existing": [],
                "eligible": sorted(payloads), "payloads": payloads}

    created, existing = [], []
    for name, payload in payloads.items():
        path = Path(branches[name]["binding"])
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise P7SplitBindingError(f"{name} binding exists with different authority")
            existing.append(name)
        else:
            atomic_create_json(path, payload)
            created.append(name)
    return {"status": "bound", "created": created, "existing": existing,
            "eligible": sorted(payloads), "necessity_result_sha256": result_hash}
