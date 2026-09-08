#!/usr/bin/env python3
"""Contract the four-head weight tensor with its frozen occupied reader mode."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_shapes_contraction_replay_orthogonality_finiteness_and_zero_forward_price pred_b_occupied_reader_contracts_heads_into_one_shared_signed_map pred_c_heads_share_both_context_and_physical_coordinate_interfaces pred_d_each_head_has_a_compact_internal_interface
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_occupied_reader_weight_writer_grouping_v1.json"
WEIGHT_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1_result.json"
WEIGHT_NPZ = ROOT / "circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1.npz"
OCCUPANCY = ROOT / "circuits/followups/temporal_iswas_v23_four_head_weight_mode_occupancy_transport_v1_result.json"
CONFIRMATION = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_occupied_reader_weight_writer_grouping_v1_result.json"
ARRAYS = ROOT / "circuits/followups/temporal_iswas_v23_occupied_reader_weight_writer_grouping_v1.npz"
EXPECTED = {
    "prior": "21b978cc4f552a78a620cc81279204af21013698806f63bbc60b215d6019aadf",
    "weight_result": "8065c2403bd583b71adca4adcc5e41db9ce4247e1360fd2a75ca6a34980f5393",
    "weight_npz": "9b553f42e708f1612fd90a934e0d26783b935457156f4856bc4b69f60df37892",
    "occupancy": "2aae7e4bb5eb5092c64cc0248f5fa789a418ff2d78549e4d2acf438b8eef589b",
    "confirmation": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
}
BARS = {"replay": 1e-5, "head_energy": .70, "map_cosine": .60,
        "row_overlap": .50, "column_overlap": .50, "top16_energy": .50}
PRICE = {"checkpoint_loads": 0, "model_forwards": 0, "sequence_evaluations": 0,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_shapes_contraction_replay_orthogonality_finiteness_and_zero_forward_price",
    "pred_b_occupied_reader_contracts_heads_into_one_shared_signed_map",
    "pred_c_heads_share_both_context_and_physical_coordinate_interfaces",
    "pred_d_each_head_has_a_compact_internal_interface",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def projector_overlap(basis_a, basis_b):
    rank = basis_a.shape[1]
    if basis_a.shape[0] != basis_b.shape[0] or basis_b.shape[1] != rank or rank == 0:
        raise ValueError("projector bases have incompatible shapes")
    return float(np.square(basis_a.T @ basis_b).sum() / rank)


def pairwise(values, score):
    matrix = np.eye(len(values), dtype=np.float64)
    pairs = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            value = float(score(values[left], values[right]))
            matrix[left, right] = matrix[right, left] = value; pairs.append(value)
    return matrix, pairs


def main():
    paths = {"prior": PRIOR, "weight_result": WEIGHT_RESULT, "weight_npz": WEIGHT_NPZ,
             "occupancy": OCCUPANCY, "confirmation": CONFIRMATION}
    observed = {name: sha(path) for name, path in paths.items()}
    dry = {"candidate_id": "cross_task.temporal_iswas.v23_occupied_reader_weight_writer_grouping_v1",
           "dryrun": True, "gpu_accessed": False, "model_loaded": False,
           "queue_touched": False, "authority_ok": observed == EXPECTED, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True)); return
    weight_result = json.loads(WEIGHT_RESULT.read_text())
    occupancy = json.loads(OCCUPANCY.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    authority_ok = bool(observed == EXPECTED and weight_result.get("terminal") == "head_private_weight_capability"
        and occupancy.get("predictions", {}).get("pred_d_realized_target_use_is_sparser_and_split_stable") is True
        and confirmation.get("terminal") == "confirmed_selective_four_head_writer_program")
    if not authority_ok: raise RuntimeError("occupied-reader writer grouping authority changed")
    if OUT.exists() or ARRAYS.exists(): raise FileExistsError("refusing to overwrite grouping result")
    started_utc, started = now(), time.perf_counter()
    frozen = np.load(WEIGHT_NPZ); cross = frozen["cross"].astype(np.float64)
    routes = frozen["routes"].tolist()
    if cross.shape != (4, 4, 1152, 128) or len(routes) != 4:
        raise RuntimeError("frozen cross tensor shape changed")
    reader_unfolding = np.moveaxis(cross, 1, 0).reshape(4, -1)
    reader_u, reader_s, _ = np.linalg.svd(reader_unfolding, full_matrices=False)
    reader_mode = reader_u[:, 0]
    contracted = np.einsum("a,haik->hik", reader_mode, cross, optimize=True)
    replay = np.tensordot(cross, reader_mode, axes=([1], [0]))
    replay_max_abs = float(np.max(np.abs(contracted-replay)))
    head_u, head_s, _ = np.linalg.svd(contracted.reshape(4, -1), full_matrices=False)
    head_energy = np.square(head_s) / np.square(head_s).sum()
    flat = [value.reshape(-1) for value in contracted]
    cosine_matrix, cosine_pairs = pairwise(flat, lambda a, b: (a @ b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-300))
    absolute_cosine_pairs = [abs(value) for value in cosine_pairs]
    singular_values, left16, right16, top8_energy, top16_energy, orthogonality = [], [], [], [], [], []
    for tensor in contracted:
        u, s, vh = np.linalg.svd(tensor, full_matrices=False)
        energy = np.square(s) / np.square(s).sum()
        singular_values.append(s); left16.append(u[:, :16]); right16.append(vh[:16].T)
        top8_energy.append(float(energy[:8].sum())); top16_energy.append(float(energy[:16].sum()))
        orthogonality.append(max(float(np.max(np.abs(u[:, :16].T@u[:, :16]-np.eye(16)))),
                                 float(np.max(np.abs(vh[:16]@vh[:16].T-np.eye(16))))))
    row_matrix, row_pairs = pairwise([value[:, :8] for value in right16], projector_overlap)
    column_matrix, column_pairs = pairwise([value[:, :8] for value in left16], projector_overlap)
    summary = {"reader_mode_singular_values": reader_s.tolist(),
        "contracted_head_mode_energy_fraction": head_energy.tolist(),
        "pairwise_signed_map_cosine": cosine_matrix.tolist(),
        "median_pair_absolute_map_cosine": float(np.median(absolute_cosine_pairs)),
        "top8_energy_fraction_by_head": dict(zip(routes, top8_energy)),
        "top16_energy_fraction_by_head": dict(zip(routes, top16_energy)),
        "pairwise_top8_row_projector_overlap": row_matrix.tolist(),
        "median_pair_top8_row_projector_overlap": float(np.median(row_pairs)),
        "pairwise_top8_column_projector_overlap": column_matrix.tolist(),
        "median_pair_top8_column_projector_overlap": float(np.median(column_pairs))}
    finite = bool(all(np.isfinite(value).all() for value in (contracted, reader_s, head_s,
        cosine_matrix, row_matrix, column_matrix, *singular_values)))
    A = bool(authority_ok and replay_max_abs <= BARS["replay"] and max(orthogonality) <= BARS["replay"]
             and finite and PRICE == {k: 0 for k in PRICE})
    B = bool(head_energy[0] >= BARS["head_energy"]
             and summary["median_pair_absolute_map_cosine"] >= BARS["map_cosine"])
    C = bool(summary["median_pair_top8_row_projector_overlap"] >= BARS["row_overlap"]
             and summary["median_pair_top8_column_projector_overlap"] >= BARS["column_overlap"])
    D = bool(min(top16_energy) >= BARS["top16_energy"])
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A else "cross_head_group_candidate" if B and C
                else "convergent_private_writers" if D else "broad_private_writer_interfaces")
    np.savez_compressed(ARRAYS, routes=np.asarray(routes), reader_mode=reader_mode.astype(np.float32),
        contracted=contracted.astype(np.float32), reader_singular_values=reader_s.astype(np.float32),
        head_singular_values=head_s.astype(np.float32), per_head_singular_values=np.asarray(singular_values, dtype=np.float32),
        left16=np.asarray(left16, dtype=np.float32), right16=np.asarray(right16, dtype=np.float32))
    result = {"schema": "temporal_iswas_v23_occupied_reader_weight_writer_grouping_result_v1",
        "candidate_id": dry["candidate_id"], "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": observed,
        "routes": routes, "contraction_replay_max_abs": replay_max_abs,
        "svd_orthogonality_max_abs": max(orthogonality), "summary": summary,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE,
        "arrays_sha256": sha(ARRAYS), "selected_rank_or_group_after_outcome": None}
    atomic_create_json(OUT, result)
    print(json.dumps({"contraction_replay_max_abs": replay_max_abs, "summary": summary,
        "predictions": predictions, "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__": main()
