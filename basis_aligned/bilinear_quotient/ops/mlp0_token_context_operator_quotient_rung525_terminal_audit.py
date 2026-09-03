#!/usr/bin/env python3
"""Independently audit rung 525's exact-operator grouping result."""

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

import mlp0_token_context_operator_quotient_rung525_math as oq  # noqa: E402


DEFAULT_RESULT = ROOT / "mlp0_token_context_operator_quotient_rung525_results.json"
DEFAULT_ARTIFACT = ROOT / "mlp0_token_context_operator_quotient_rung525_pairs.pt"
DEFAULT_OUTPUT = ROOT / "mlp0_token_context_operator_quotient_rung525_terminal_audit.json"
RUNNER = OPS / "mlp0_token_context_operator_quotient_rung525_run.py"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: object) -> object:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def audit_terminal_result(
    result: Mapping[str, object],
    artifact: Mapping[str, object],
    *,
    artifact_file_sha256: str,
) -> dict[str, object]:
    if result.get("rung") != 525 or result.get("status") != "complete":
        raise ValueError("unexpected rung or terminal status")
    if result.get("claim_level") != "exact_weight_function_operator_grouping_screen_not_circuit_evidence":
        raise ValueError("claim level changed")
    receipt = result.get("pair_artifact")
    if not isinstance(receipt, Mapping) or receipt.get("sha256") != artifact_file_sha256:
        raise ValueError("pair-artifact receipt differs from file")

    expected_keys = {
        "schema", "role_hashes", "runner_sha256",
        "receiver_ids", "candidate_donor_ids", "raw_control_donor_ids",
        "deranged_control_donor_ids", "far_random_donor_ids",
        "bank_a_candidate_distance", "bank_b_candidate_distance",
        "bank_b_raw_control_distance", "bank_b_deranged_control_distance",
        "bank_b_far_random_distances", "half0_candidate_donor_ids",
        "half1_candidate_donor_ids", "half0_bank_b_distance",
        "half1_bank_b_distance",
    }
    if set(artifact) != expected_keys:
        raise ValueError("pair-artifact key census changed")
    if artifact["schema"] != "mlp0-token-context-operator-quotient-rung525-pairs-v1":
        raise ValueError("pair-artifact schema changed")
    if artifact["role_hashes"] != result["probe_banks"]["role_hashes"]:
        raise ValueError("pair-artifact role hashes changed")
    if artifact["runner_sha256"] != result["runner_sha256"]:
        raise ValueError("pair-artifact runner hash changed")
    receivers = artifact["receiver_ids"]
    candidates = artifact["candidate_donor_ids"]
    n = 10_052
    one_dimensional = expected_keys - {
        "schema", "role_hashes", "runner_sha256",
        "far_random_donor_ids", "bank_b_far_random_distances"
    }
    for key in one_dimensional:
        value = artifact[key]
        if not isinstance(value, torch.Tensor) or value.shape != (n,):
            raise ValueError(f"invalid pair vector {key}")
    for key in ("far_random_donor_ids", "bank_b_far_random_distances"):
        value = artifact[key]
        if not isinstance(value, torch.Tensor) or value.shape != (n, 16):
            raise ValueError(f"invalid random-control matrix {key}")
    if receivers.dtype != torch.int64 or candidates.dtype != torch.int64:
        raise ValueError("token identifiers changed dtype")
    if not torch.equal(receivers, torch.arange(0, 50_257, 5, dtype=torch.int64)):
        raise ValueError("receiver split changed")
    donor_vectors = (
        candidates, artifact["raw_control_donor_ids"], artifact["deranged_control_donor_ids"],
        artifact["half0_candidate_donor_ids"], artifact["half1_candidate_donor_ids"],
        artifact["far_random_donor_ids"].reshape(-1),
    )
    if any(bool((value.remainder(5) == 0).any()) for value in donor_vectors):
        raise ValueError("receiver token entered the donor population")

    recomputed = oq.score_real(
        candidate_a_distance=artifact["bank_a_candidate_distance"],
        candidate_b_distance=artifact["bank_b_candidate_distance"],
        raw_b_distance=artifact["bank_b_raw_control_distance"],
        random_b_distances=artifact["bank_b_far_random_distances"],
        deranged_b_distance=artifact["bank_b_deranged_control_distance"],
        candidate_donors=candidates,
        half_a_b_distances=(
            artifact["half0_bank_b_distance"], artifact["half1_bank_b_distance"]
        ),
    )
    recorded_score = result.get("score")
    if not isinstance(recorded_score, Mapping):
        raise ValueError("score is absent")
    for key, value in recomputed.items():
        if _json_value(recorded_score.get(key)) != _json_value(value):
            raise ValueError(f"recomputed score differs for {key}")

    if not result.get("pred_a_exact_lawful_instrument"):
        raise ValueError("exact-instrument prediction was not satisfied")
    if result.get("pred_b_operator_grouping_transfers") or not result.get("pred_c_repeated_groups_not_isolated_pairs"):
        raise ValueError("registered B/C verdict changed")
    if not result.get("strong_null") or not recomputed["strong_null"]:
        raise ValueError("registered strong null did not fire")
    if result.get("physical_downstream_successor_licensed") or recomputed["physical_successor_licensed"]:
        raise ValueError("physical successor was incorrectly licensed")
    if result.get("next_action") != "context_only_or_downstream_conditioned_operator_metric":
        raise ValueError("registered next action changed")

    instrument = result.get("instrument")
    price = result.get("execution_price")
    if not isinstance(instrument, Mapping) or not isinstance(price, Mapping):
        raise ValueError("instrument or execution receipt is absent")
    if instrument.get("FINAL_or_sealed_opened") is not False:
        raise ValueError("sealed outcomes were opened")
    if instrument.get("no_downstream_model_or_circuit_calls") is not True:
        raise ValueError("downstream-call seal changed")
    if price.get("downstream_model_forwards") != 0 or price.get("circuit_evaluations") != 0:
        raise ValueError("unregistered downstream computation was reported")
    if price.get("deployed_values_added") != 0 or price.get("deployed_values_saved") != 0:
        raise ValueError("screen was incorrectly priced as a deployment")

    return {
        "schema": "rung525-terminal-audit-v1",
        "passes": True,
        "status": result["status"],
        "pair_artifact_file_sha256": artifact_file_sha256,
        "receiver_count": n,
        "candidate_over_raw": recomputed["candidate_over_raw"],
        "candidate_over_random": recomputed["candidate_over_random"],
        "candidate_over_deranged": recomputed["candidate_over_deranged"],
        "exceptional_vs_far_random_fraction": recomputed["exceptional_vs_far_random_fraction"],
        "prediction_a": True,
        "prediction_b": False,
        "prediction_c": True,
        "strong_null": True,
        "physical_successor_licensed": False,
        "sealed_outcomes_opened": False,
        "downstream_calls": 0,
        "interpretation": (
            "the exact token-by-context operator metric is stable but does not group "
            "far tokens better than ordinary token similarity; close task-free grouping "
            "under this metric and change to a downstream-conditioned causal object"
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


def main() -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with args.result.open(encoding="utf-8") as source:
        result = json.load(source)
    for relative, expected in result["dependency_sha256"].items():
        if _file_sha256(REPO / relative) != expected:
            raise ValueError(f"dependency changed: {relative}")
    if _file_sha256(RUNNER) != result["runner_sha256"]:
        raise ValueError("runner changed")
    artifact_sha = _file_sha256(args.artifact)
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    audit = {
        "result_file_sha256": _file_sha256(args.result),
        **audit_terminal_result(result, artifact, artifact_file_sha256=artifact_sha),
    }
    _atomic_json(args.output, audit)
    print(json.dumps({
        "output": str(args.output), "passes": True,
        "strong_null": True, "physical_successor_licensed": False,
        "next_action": result["next_action"],
    }, indent=2, sort_keys=True), flush=True)
    return audit


if __name__ == "__main__":
    main()
