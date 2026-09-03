#!/usr/bin/env python3
"""Post-result response-space diagnosis for rung 528's shared/private route."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import equality_distributed_finite_transition_quotient_rung528_math as qm
import equality_distributed_finite_transition_quotient_rung528_run as r528


BQ = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
RESULT = BQ / "equality_distributed_finite_transition_quotient_rung528_results.json"
BUNDLE = BQ / "equality_distributed_finite_transition_quotient_rung528_bundle.pt"
OUT = BQ / "rung528_leave_one_action_out_consensus_diagnosis_results.json"
EXPECTED = {
    RESULT: "f931e5fb6f618b002203ce1e870a8ad4442ed3a38a7475809754ab2de91554b6",
    BUNDLE: "c17db82832a76daba23f74e57e75abc258093c6820c79c93a62d8d29b6143d38",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run():
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise RuntimeError(f"rung528 artifact changed: {path}")
    result = json.loads(RESULT.read_text())
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=True)
    unit = r528.phase_views(bundle["phases"]["discovery"], "unit")
    betas = {"N": 1.0, **{
        source: result["discovery"]["checks"][source]["beta"]
        for source in r528.CANDIDATE_SOURCES
    }}
    aligned = torch.stack([
        betas[source] * unit["circuit_halves"][r528.SOURCES.index(source)]
        for source in r528.SOURCES
    ])
    reports = {}
    for target_index, target in enumerate(r528.SOURCES):
        consensus = aligned[[index for index in range(len(r528.SOURCES))
                             if index != target_index]].mean(0)
        target_response = aligned[target_index]
        private = target_response - consensus
        interaction = qm.factorial_interaction(unit["circuit_halves"][target_index])
        reports[target] = {
            "D0": qm.relation_metrics(target_response[0], consensus[0], 1.0),
            "D1": qm.relation_metrics(target_response[1], consensus[1], 1.0),
            "private_cross_half_cosine": qm.relation_metrics(private[0], private[1], 1.0)["cosine"],
            "continuation_factorial_interaction_over_native": [
                float(interaction[half].norm()
                      / unit["circuit_halves"][target_index, half, 0].norm().clamp_min(1e-30))
                for half in range(2)
            ],
        }
    return {
        "status": "complete",
        "claim_level": "post_result_response_space_diagnosis_not_physical_state_evidence",
        "source_hashes": {str(path): expected for path, expected in EXPECTED.items()},
        "alignment_scales_to_N_units": betas,
        "leave_one_action_out_consensus": reports,
        "best_registered_signal": "Z7",
        "physical_consensus_insertion_opened": False,
        "next_step": "preregister_all_action_leave_one_out_physical_state_consensus",
    }


if __name__ == "__main__":
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite result: {OUT}")
    report = run()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
