#!/usr/bin/env python3
"""CPU audit of matched-bit MLP4 candidates as symmetric quadratic tensors."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from . import bilinear_tensor_invariants as invariants
from . import mlp4_bilinear_residual_codec as native_codec
from . import mlp4_seeded_random_bilinear_codec as random_codec

HERE = Path(__file__).resolve().parent
BYTES = HERE / "mlp4_z4_candidate_bytes.pt"
INVENTORY = HERE / "mlp4_z4_candidate_inventory.json"
OUTPUT = HERE / "mlp4_candidate_tensor_invariants.json"
THREADS = 4


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factors(candidate_id, encoded):
    if candidate_id.startswith("native_"):
        decoded = native_codec.decode(encoded)
        return decoded["A"], decoded["B"], decoded["C"]
    decoded = random_codec.decode(encoded)
    A, B = random_codec.feature_factors(
        decoded["semantic"]["seed"], decoded["din"], decoded["components"])
    return A, B, decoded["C"]


def summarize(candidate_id, encoded):
    started = time.time()
    A, B, C = factors(candidate_id, encoded)
    spectrum = invariants.output_mode_spectrum(A, B, C)
    singular = spectrum.pop("singular_values")
    energy = singular.square()
    total = float(energy.sum())
    return {
        "candidate_id": candidate_id,
        "components": A.shape[1],
        "input_width": A.shape[0],
        "output_width": C.shape[1],
        **spectrum,
        "largest_mode_energy_fraction": float(energy[0]/total) if total else 0.0,
        "top8_mode_energy_fraction": float(energy[:8].sum()/total) if total else 0.0,
        "top32_mode_energy_fraction": float(energy[:32].sum()/total) if total else 0.0,
        "energy_rank_90": invariants.energy_rank(singular, .90),
        "energy_rank_95": invariants.energy_rank(singular, .95),
        "energy_rank_99": invariants.energy_rank(singular, .99),
        "best_rank8_relative_frobenius_error":
            invariants.best_rank_relative_frobenius_error(singular, min(8, singular.numel())),
        "best_rank32_relative_frobenius_error":
            invariants.best_rank_relative_frobenius_error(singular, min(32, singular.numel())),
        "singular_values_top32": singular[:32].tolist(),
        "runtime_s": time.time()-started,
    }


def main():
    started = time.time()
    torch.set_num_threads(THREADS)
    artifact = torch.load(BYTES, map_location="cpu", weights_only=False)
    inventory = json.loads(INVENTORY.read_text())
    inventory_by_id = {row["candidate_id"]: row for row in inventory["candidates"]}
    roster = []
    for pair in inventory["native_random_actual_bit_pairings"]:
        roster.extend((pair["native_candidate_id"], pair["random_candidate_id"]))
    if len(roster) != len(set(roster)):
        raise ValueError("matched-pair roster unexpectedly repeats a candidate")
    rows = []
    for candidate_id in roster:
        print(f"tensor invariant {candidate_id}", flush=True)
        row = summarize(candidate_id, artifact["encoded"][candidate_id])
        source = inventory_by_id[candidate_id]
        row["serialized_bits"] = source.get(
            "conditional_known_gauge_bits", source.get("canonical_program_bits"))
        row["stable_rank_per_mbit"] = row["stable_rank"]/(row["serialized_bits"]/1e6)
        rows.append(row)
    by_id = {row["candidate_id"]: row for row in rows}
    pair_comparisons = []
    for pair in inventory["native_random_actual_bit_pairings"]:
        native = by_id[pair["native_candidate_id"]]
        random = by_id[pair["random_candidate_id"]]
        pair_comparisons.append({
            "native": native["candidate_id"], "random": random["candidate_id"],
            "native_stable_rank": native["stable_rank"],
            "random_stable_rank": random["stable_rank"],
            "native_to_random_stable_rank_ratio": (
                native["stable_rank"]/random["stable_rank"]),
            "native_stable_rank_per_mbit": native["stable_rank_per_mbit"],
            "random_stable_rank_per_mbit": random["stable_rank_per_mbit"],
        })
    result = {
        "schema_version": 1,
        "audit_id": "bilin18.mlp4.matched-bit-symmetric-tensor-invariants.v1",
        "candidate_bytes_sha256": sha(BYTES),
        "inventory_sha256": sha(INVENTORY),
        "formula": "G_jk=.5[(a_j.a_k)(b_j.b_k)+(a_j.b_k)(b_j.a_k)]; output_gram=C^T G C",
        "interpretation": {
            "invariant_under": ["factor scale/sign", "input-leg swap",
                                "component permutation", "equivalent exact factorization"],
            "not_behavioral_evidence": True,
            "not_a_description_length": True,
            "rank_object": "matrix rank after grouping the two symmetric input legs",
            "energy_rank_lower_bound": "Eckart-Young: any K-term bilinear program has grouped rank <= K",
            "no_bit_lower_bound_claim": True,
        },
        "cpu_threads": THREADS,
        "rows": rows,
        "pair_comparisons": pair_comparisons,
        "runtime_s": time.time()-started,
    }
    OUTPUT.write_text(json.dumps(result, indent=2)+"\n")
    print(f"wrote {OUTPUT.name} in {result['runtime_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
