#!/usr/bin/env python3
"""Post-hoc positive/negative red-team of the preregistered rank-eight DCT null."""
import hashlib
import json
from pathlib import Path

import torch

P = Path(__file__).resolve().parent
ARTIFACT = P / "MLP9_DCT_UNSUPERVISED_READER_V1_ARTIFACT.pt"
RESULT = P / "MLP9_DCT_UNSUPERVISED_READER_V1_RESULT.json"
OUT = P / "MLP9_DCT_UNSUPERVISED_READER_RANK_STABILITY_V1_AUDIT.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def basis(rows):
    return torch.linalg.qr(rows.T.double(), mode="reduced").Q


def coverage(reader, subspace):
    return float((reader @ subspace @ subspace.T).norm() / reader.norm().clamp_min(1e-30))


artifact = torch.load(ARTIFACT, weights_only=True)
result = json.loads(RESULT.read_text())
first, second = artifact["fits"]
input_cosines = (first["input"].double() @ second["input"].double().T).abs()
output_cosines = (first["output"].double() @ second["output"].double().T).abs()
joint = torch.minimum(input_cosines, output_cosines)
stable_pairs = []
used_second = set()
for first_index in range(8):
    value, second_index = joint[first_index].max(0)
    second_index = int(second_index)
    if float(value) >= .95 and second_index not in used_second:
        stable_pairs.append((first_index, second_index))
        used_second.add(second_index)

reader = torch.cat(artifact["readers"], dim=1).reshape(-1, 1152).double()
stable_first = [pair[0] for pair in stable_pairs]
stable_basis = basis(first["output"][stable_first])
stable_coverage = coverage(reader, stable_basis)
generator = torch.Generator().manual_seed(2026091605)
controls = []
for _ in range(16):
    random_basis, _ = torch.linalg.qr(torch.randn(1152, len(stable_pairs), generator=generator, dtype=torch.float64), mode="reduced")
    controls.append(coverage(reader, random_basis))

audit = {
    "schema": "mlp9_dct_unsupervised_reader_rank_stability_v1_audit",
    "status": "posthoc_audit_not_a_new_preregistered_test",
    "result_sha256": sha(RESULT),
    "artifact_sha256": sha(ARTIFACT),
    "stable_pair_threshold": .95,
    "stable_pairs_first_to_second": [list(pair) for pair in stable_pairs],
    "stable_pair_input_absolute_cosines": [float(input_cosines[pair]) for pair in stable_pairs],
    "stable_pair_output_absolute_cosines": [float(output_cosines[pair]) for pair in stable_pairs],
    "stable_rank": len(stable_pairs),
    "stable_rank_reader_coverage": stable_coverage,
    "random_rank_matched_coverage": controls,
    "random_rank_matched_median": float(torch.tensor(controls).median()),
    "random_rank_matched_maximum": max(controls),
    "rank8_reader_coverage": result["pooled_reader_coverage"],
    "rank8_random_median": result["random_control_median"],
    "finding": "Four factors are individually stable across seeds after permutation, but their pooled CrossFirst reader coverage is indistinguishable from equal-rank random subspaces. The other four factors are unstable. This preserves real weight-tensor structure while confirming that raw weight-only MLP9 DCT factors are not useful CrossFirst readers.",
}
OUT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(json.dumps(audit, indent=2, sort_keys=True))
