#!/usr/bin/env python3
"""Independently audit rung 526's fail-closed circuit-response grouping result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Mapping

import torch


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
REPO = ROOT.parent.parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

import mlp0_circuit_response_operator_quotient_rung526_math as qm  # noqa: E402


DEFAULT_RESULT = ROOT / "mlp0_circuit_response_operator_quotient_rung526_results.json"
DEFAULT_ARTIFACT = ROOT / "mlp0_circuit_response_operator_quotient_rung526_pairs.pt"
DEFAULT_OUTPUT = ROOT / "mlp0_circuit_response_operator_quotient_rung526_terminal_audit.json"
R525_PAIRS = ROOT / "mlp0_token_context_operator_quotient_rung525_pairs.pt"
RUNNER = OPS / "mlp0_circuit_response_operator_quotient_rung526_run.py"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _phase_valid(phase: Mapping[str, object]) -> bool:
    return bool(
        phase.get("identity_leaf_logit_max_abs") == 0.0
        and phase.get("all_circuit_gradients_nonzero") is True
        and float(phase.get("member_weight_sum_max_abs_error", 1)) <= 1e-5
        and float(phase.get("control_weight_sum_max_abs_error", 1)) <= 1e-5
        and float(phase.get("aggregate_contraction_relative_squared_error", 1)) <= 1e-5
        and phase.get("signature_finite") is True
        and phase.get("signature_nonconstant") is True
    )


def audit_terminal_result(
    result: Mapping[str, object], artifact: Mapping[str, object],
    rung525_artifact: Mapping[str, object], *, artifact_sha256: str,
) -> dict[str, object]:
    if result.get("rung") != 526 or result.get("status") != "complete":
        raise ValueError("unexpected rung or terminal status")
    if result.get("claim_level") != "downstream_circuit_conditioned_tangent_grouping_screen_not_finite_circuit_evidence":
        raise ValueError("claim level changed")
    receipt = result.get("pair_artifact")
    if not isinstance(receipt, Mapping) or receipt.get("sha256") != artifact_sha256:
        raise ValueError("pair artifact receipt differs")
    expected_keys = {
        "schema", "runner_sha256", "receivers_ids", "candidate_ids", "raw_ids",
        "scrambled_ids", "random_ids", "first_half_ids", "second_half_ids",
        "d0_candidate_distance", "d1_candidate_distance", "d1_raw_distance",
        "d1_scrambled_distance", "d1_random_distance", "d1_first_half_distance",
        "d1_second_half_distance",
    }
    if set(artifact) != expected_keys:
        raise ValueError("pair artifact key census changed")
    if artifact["schema"] != "mlp0-circuit-response-operator-quotient-rung526-pairs-v1":
        raise ValueError("pair artifact schema changed")
    if artifact["runner_sha256"] != result.get("runner_sha256"):
        raise ValueError("artifact runner hash changed")
    n = 10_052
    vector_keys = expected_keys - {"schema", "runner_sha256", "random_ids", "d1_random_distance"}
    for key in vector_keys:
        value = artifact[key]
        if not isinstance(value, torch.Tensor) or value.shape != (n,):
            raise ValueError(f"invalid vector {key}")
    for key in ("random_ids", "d1_random_distance"):
        value = artifact[key]
        if not isinstance(value, torch.Tensor) or value.shape != (n, 16):
            raise ValueError(f"invalid random matrix {key}")
    receivers = artifact["receivers_ids"]
    if not torch.equal(receivers, torch.arange(0, 50_257, 5, dtype=torch.int64)):
        raise ValueError("receiver split changed")
    donor_keys = ("candidate_ids", "raw_ids", "scrambled_ids", "first_half_ids", "second_half_ids")
    if any(bool((artifact[key].remainder(5) == 0).any()) for key in donor_keys) \
            or bool((artifact["random_ids"].remainder(5) == 0).any()):
        raise ValueError("receiver entered donor population")
    old_donors = rung525_artifact.get("candidate_donor_ids")
    if not isinstance(old_donors, torch.Tensor) or old_donors.shape != (n,):
        raise ValueError("rung 525 donor authority changed")

    recomputed = qm.score_discovery(
        distance_d0=artifact["d0_candidate_distance"],
        distance_d1=artifact["d1_candidate_distance"],
        raw_d1=artifact["d1_raw_distance"],
        random_d1=artifact["d1_random_distance"],
        scrambled_d1=artifact["d1_scrambled_distance"],
        candidate_donors=artifact["candidate_ids"],
        circuit_half_d1=(artifact["d1_first_half_distance"], artifact["d1_second_half_distance"]),
        rung525_donors=old_donors,
    )
    recorded = result.get("discovery_score")
    if not isinstance(recorded, Mapping):
        raise ValueError("discovery score absent")
    for key, value in recomputed.items():
        if _json(recorded.get(key)) != _json(value):
            raise ValueError(f"recomputed score differs for {key}")

    phases = result.get("phase_instruments")
    if not isinstance(phases, Mapping) or set(phases) != {"D0", "D1"}:
        raise ValueError("phase census changed")
    if not all(_phase_valid(phase) for phase in phases.values()):
        raise ValueError("one or more phase instruments are invalid")
    if result.get("d1_opened") is not True:
        raise ValueError("D1 open receipt changed")
    if result.get("validation_circuits_opened") is not False or result.get("validation_score") is not None:
        raise ValueError("validation circuits were opened after discovery failure")
    expected_verdicts = {
        "pred_a_exact_live_leakage_free_instrument": True,
        "pred_b_same_circuit_new_document_transfer": False,
        "pred_c_heldout_circuit_transfer": False,
        "pred_d_reusable_changed_groups": True,
        "strong_null": True,
        "physical_successor_licensed": False,
    }
    for key, expected in expected_verdicts.items():
        if result.get(key) is not expected:
            raise ValueError(f"terminal verdict changed for {key}")
    if not recomputed["strong_null"] or recomputed["prediction_b_document_transfer"]:
        raise ValueError("recomputed discovery decision changed")

    price = result.get("execution_price")
    if not isinstance(price, Mapping):
        raise ValueError("execution price absent")
    summed = {
        key: sum(int(phase["calls"][key]) for phase in phases.values())
        for key in ("forwards", "native_replays", "batched_backwards", "gradient_objectives")
    }
    for key, expected in summed.items():
        if price.get(key) != expected:
            raise ValueError(f"execution count differs for {key}")
    if price.get("finite_intervention_forwards") != 0:
        raise ValueError("finite interventions were run")
    if price.get("deployed_values_added") != 0 or price.get("deployed_values_saved") != 0:
        raise ValueError("screen was priced as a deployment")
    return {
        "schema": "rung526-terminal-audit-v1", "passes": True,
        "result_status": result["status"], "artifact_sha256": artifact_sha256,
        "receiver_count": n, "predictions": {"A": True, "B": False, "C": False, "D": True},
        "candidate_over_raw": recomputed["candidate_over_raw"],
        "candidate_over_random": recomputed["candidate_over_random"],
        "candidate_over_scrambled": recomputed["candidate_over_scrambled"],
        "d0_d1_spearman": recomputed["d0_d1_candidate_distance_spearman"],
        "validation_circuits_opened": False, "finite_intervention_forwards": 0,
        "strong_null": True, "physical_successor_licensed": False,
        "exact_execution_counts": summed,
        "interpretation": (
            "circuit-conditioned tangent neighbors are document-specific and much worse than raw-token "
            "neighbors on held-out documents; close token grouping under this metric"
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    for relative, expected in result["dependency_sha256"].items():
        if _file_sha256(REPO / relative) != expected:
            raise ValueError(f"dependency changed: {relative}")
    if _file_sha256(RUNNER) != result["runner_sha256"]:
        raise ValueError("runner changed")
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    old = torch.load(R525_PAIRS, map_location="cpu", weights_only=True)
    audit = {
        "result_sha256": _file_sha256(args.result),
        **audit_terminal_result(result, artifact, old, artifact_sha256=_file_sha256(args.artifact)),
    }
    _atomic_json(args.output, audit)
    print(json.dumps({
        "output": str(args.output), "passes": True, "strong_null": True,
        "validation_circuits_opened": False,
        "next_action": result["next_action"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
