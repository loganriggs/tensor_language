#!/usr/bin/env python3
"""Gauge-invariant audit of occupied-reader-contracted head maps."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_formula_rotation_invariance_finiteness_and_zero_forward_price pred_b_head_maps_remain_private_up_to_orthogonal_gauge pred_c_heads_share_only_a_broad_spectral_profile pred_d_no_cross_head_weight_group_is_licensed
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_occupied_reader_weight_writer_gauge_audit_v1.json"
PARENT = ROOT / "circuits/followups/temporal_iswas_v23_occupied_reader_weight_writer_grouping_v1_result.json"
ARRAYS = ROOT / "circuits/followups/temporal_iswas_v23_occupied_reader_weight_writer_grouping_v1.npz"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_occupied_reader_weight_writer_gauge_audit_v1_result.json"
EXPECTED = {"prior": "c349f2b6dc2cd2db84bda04a95c85320ed3f1f9e3bd0b0f711b3b5b6a374f97d", "parent": "f9b5c7206c6c67d5bb1096b1f65c230be4327efd5c0372a75d3577c5e6114717",
            "arrays": "e8dde195577c4ff8778ef471eee744469e8f95a346611376f5f964dae6ba215c"}
BARS = {"invariance": 1e-8, "procrustes": .80, "gram": .70, "spectrum": .95, "rank32": .60}
PRICE = {"checkpoint_loads": 0, "model_forwards": 0, "sequence_evaluations": 0,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = ("pred_a_authority_formula_rotation_invariance_finiteness_and_zero_forward_price",
    "pred_b_head_maps_remain_private_up_to_orthogonal_gauge",
    "pred_c_heads_share_only_a_broad_spectral_profile", "pred_d_no_cross_head_weight_group_is_licensed")
RANKS = (1, 2, 4, 8, 16, 32, 64, 128)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

def procrustes(a, b):
    return float(np.linalg.svd(a.T @ b, compute_uv=False).sum() / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-300))

def gram_cosine(a, b):
    numerator = np.square(a.T @ b).sum()
    denominator = np.sqrt(np.square(a.T @ a).sum() * np.square(b.T @ b).sum())
    return float(numerator / max(denominator, 1e-300))

def projector_overlap(a, b, rank):
    ua = np.linalg.svd(a, full_matrices=False)[0][:, :rank]
    ub = np.linalg.svd(b, full_matrices=False)[0][:, :rank]
    return float(np.square(ua.T @ ub).sum()/rank)

def pair_values(maps, score):
    matrix = np.eye(len(maps)); values = []
    for i in range(len(maps)):
        for j in range(i+1, len(maps)):
            value = float(score(maps[i], maps[j])); matrix[i,j] = matrix[j,i] = value; values.append(value)
    return matrix, values

def main():
    observed = {"prior": sha(PRIOR), "parent": sha(PARENT), "arrays": sha(ARRAYS)}
    dry = {"candidate_id": "cross_task.temporal_iswas.v23_occupied_reader_weight_writer_gauge_audit_v1",
           "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "authority_ok": observed == EXPECTED, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True)); return
    if observed != EXPECTED or json.loads(PARENT.read_text()).get("terminal") not in {
            "broad_private_writer_interfaces", "convergent_private_writers"}:
        raise RuntimeError("gauge-audit authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = now(), time.perf_counter(); frozen = np.load(ARRAYS)
    maps = frozen["contracted"].astype(np.float64); spectra = frozen["per_head_singular_values"].astype(np.float64)
    proc_matrix, proc = pair_values(maps, procrustes); gram_matrix, gram = pair_values(maps, gram_cosine)
    spectrum_matrix, spectrum = pair_values(spectra, lambda a,b: (a@b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    ladder = {}; ladder_values = {}
    for rank in RANKS:
        matrix, values = pair_values(maps, lambda a,b,rank=rank: projector_overlap(a,b,rank))
        ladder[str(rank)] = matrix.tolist(); ladder_values[rank] = values
    rng = np.random.default_rng(1729); rotation, _ = np.linalg.qr(rng.standard_normal((128,128)))
    rotated = maps[1] @ rotation
    invariance_error = max(abs(procrustes(maps[0], maps[1])-procrustes(maps[0], rotated)),
        abs(gram_cosine(maps[0], maps[1])-gram_cosine(maps[0], rotated)))
    summary = {"pairwise_procrustes_similarity": proc_matrix.tolist(),
        "median_pair_procrustes_similarity": float(np.median(proc)),
        "pairwise_context_gram_similarity": gram_matrix.tolist(),
        "median_pair_context_gram_similarity": float(np.median(gram)),
        "pairwise_spectrum_cosine": spectrum_matrix.tolist(),
        "median_pair_spectrum_cosine": float(np.median(spectrum)),
        "context_projector_overlap_ladder": ladder,
        "median_pair_rank32_context_projector_overlap": float(np.median(ladder_values[32]))}
    finite = all(np.isfinite(value).all() for value in (proc_matrix, gram_matrix, spectrum_matrix))
    A = bool(invariance_error <= BARS["invariance"] and finite and set(PRICE.values()) == {0})
    B = bool(summary["median_pair_procrustes_similarity"] < BARS["procrustes"] or
             summary["median_pair_context_gram_similarity"] < BARS["gram"])
    C = bool(summary["median_pair_spectrum_cosine"] >= BARS["spectrum"] and
             summary["median_pair_rank32_context_projector_overlap"] < BARS["rank32"])
    D = bool(B)
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = "invalid_instrument" if not A else "gauge_robust_private_writers" if B else "gauge_hidden_cross_head_group"
    result = {"schema": "temporal_iswas_v23_occupied_reader_weight_writer_gauge_audit_result_v1",
        "candidate_id": dry["candidate_id"], "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": observed,
        "synthetic_rotation_invariance_max_abs": invariance_error, "summary": summary,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE,
        "selected_gauge_or_rank_after_outcome": None}
    atomic_create_json(OUT, result)
    print(json.dumps({"synthetic_rotation_invariance_max_abs": invariance_error, "summary": summary,
        "predictions": predictions, "terminal": terminal}, sort_keys=True))

if __name__ == "__main__": main()
