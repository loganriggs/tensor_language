#!/usr/bin/env python3
"""Exact zero-forward weight tensor for the confirmed four-head is/was program."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_basis_formula_reconstruction_finiteness_and_exact_price pred_b_restricted_capability_has_a_shared_head_mode pred_c_literal_factor_support_is_shared_across_heads pred_d_restricted_capability_is_concentrated_on_few_literal_factors
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

import fastload
from circuit_fast_screen_managed_runner import atomic_create_json
import subspace_weight_atlas as atlas

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1.json"
UMBRELLA = ROOT / "circuits/prior_art/temporal_iswas_v20_hr_weight_capability_occupancy_causal_atlas_v1.json"
HR_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_writer_reader_factor_split_v1_result.json"
HR_NPZ = ROOT / "circuits/followups/temporal_iswas_v20_m11_writer_reader_factor_split_v1.npz"
V23_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
ATLAS = ROOT / "ops/subspace_weight_atlas.py"
FASTLOAD = ROOT / "ops/fastload.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1_result.json"
TENSOR_OUT = ROOT / "circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1.npz"
CANDIDATE_ID = "cross_task.temporal_iswas.v23_four_head_m11_restricted_weight_capability_tensor_v1"
EXPECTED = {
    "prior": "b04ea5a13452fe480b66a0bb6593614d3a2b667a43afa30a865d10feaa2899ad",
    "umbrella": "4467613bd38d62e3ecccd56390054ef6f7b3eb18ef262fcec28e6c247d97fa47",
    "hr_result": "e334a580b72806a3976408cc16efb36aa05d066c3a7db28cb74da8f96c5ccc71",
    "hr_npz": "0af2a5e263b0a227e1f88a4d91d7b1ac2191e89734d4882458ba5f46b492a5eb",
    "v23_result": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
    "atlas": "076c63208ed10658702e1779ce336a3d85ba89191542408f45eb3bc9e0c6cf68",
    "fastload": "5803de7f127d1f556470107b559c06daecf7fbc2bccf4574aeb1c347b6225d90",
    "checkpoint": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
}
ROUTES = ((9, 1), (9, 4), (8, 1), (11, 3))
LABELS = tuple(f"L{layer}H{head}" for layer, head in ROUTES)
BARS = {"basis_orthonormal": 1e-5, "formula_relative_error": 1e-5,
        "shared_head_energy": .70, "factor_score_cosine": .80,
        "top256_squared_mass": .50}
PRICE = {"checkpoint_loads": 1, "model_forwards": 0, "sequence_evaluations": 0,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0,
         "saved_tensor_families": 2}
PREDICTION_KEYS = (
    "pred_a_authority_basis_formula_reconstruction_finiteness_and_exact_price",
    "pred_b_restricted_capability_has_a_shared_head_mode",
    "pred_c_literal_factor_support_is_shared_across_heads",
    "pred_d_restricted_capability_is_concentrated_on_few_literal_factors",
)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(float(value))


def spectra(tensor):
    return [{"mode": item["mode"], "shape": list(item["shape"]),
             "stable_rank": item["stable_rank"],
             "singular_values": item["singular_values"].tolist()}
            for item in atlas.tensor_unfolding_spectra(tensor)]


def reader_basis_from_states(states, valid_mask):
    """Return the complete span of four target construction-by-direction reader means."""
    states, valid_mask = np.asarray(states, dtype=np.float64), np.asarray(valid_mask, dtype=bool)
    if states.ndim != 4 or valid_mask.shape != states.shape[:3] or states.shape[0] < 2 or states.shape[1] != 16:
        raise ValueError("reader state/mask shapes changed")
    cell_vectors = []
    for panel_index in (0, 1):
        for parity in (0, 1):
            chosen = np.arange(16) % 2 == parity
            cell_vectors.append(states[panel_index, chosen][valid_mask[panel_index, chosen]].mean(axis=0))
    cell_vectors = np.stack(cell_vectors)
    _, singular, vh = np.linalg.svd(cell_vectors, full_matrices=False)
    tolerance = max(cell_vectors.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int((singular > tolerance).sum())
    return vh[:rank].astype(np.float32), cell_vectors, singular


def main():
    paths = {"prior": PRIOR, "umbrella": UMBRELLA, "hr_result": HR_RESULT,
             "hr_npz": HR_NPZ, "v23_result": V23_RESULT, "atlas": ATLAS,
             "fastload": FASTLOAD}
    observed = {name: sha(path) for name, path in paths.items()}
    hr_result, v23 = json.loads(HR_RESULT.read_text()), json.loads(V23_RESULT.read_text())
    authority_ok = bool(observed == {name: EXPECTED[name] for name in paths}
        and hr_result.get("terminal") == "writer_or_interaction_dominated_context_sign"
        and hr_result.get("tensor_artifact", {}).get("sha256") == EXPECTED["hr_npz"]
        and v23.get("terminal") == "confirmed_selective_four_head_writer_program"
        and v23.get("routes") == list(LABELS) and all(v23.get("predictions", {}).values()))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "routes": LABELS, "model_forwards": 0, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("restricted weight tensor authority changed")
    if OUT.exists() or TENSOR_OUT.exists(): raise FileExistsError("refusing restricted tensor overwrite")
    config, checkpoint, _ = fastload._paths()
    if sha(checkpoint) != EXPECTED["checkpoint"]: raise RuntimeError("checkpoint weights changed")
    if config.get("n_embd") != 1152 or config.get("n_head") != 9 or not config.get("bilinear"):
        raise RuntimeError("model architecture changed")
    started_utc, started = now(), time.perf_counter()
    artifact = np.load(HR_NPZ); Rstates, mask = artifact["R"].astype(np.float64), artifact["mask"]
    reader_basis, cell_vectors, cell_singular = reader_basis_from_states(Rstates, mask)
    reader_rank = reader_basis.shape[0]

    import torch
    torch.set_num_threads(4)
    model = fastload.load_model_fast().eval(); m11 = model.transformer.h[11].mlp
    basis = torch.from_numpy(reader_basis)
    left, right = m11.Left.weight.detach().float(), m11.Right.weight.detach().float()
    cross_tensors, self_tensors, cross_scores, self_scores, formula_errors = [], [], [], [], []
    x = torch.linspace(-1, 1, 1152); x = x / x.norm()
    with torch.no_grad():
        for route_index, (layer, head) in enumerate(ROUTES):
            attention = model.transformer.h[layer].attn; width = int(attention.head_dim)
            writer = attention.c_proj.weight.detach().float()[:, head*width:(head+1)*width]
            left_write, right_write = left @ writer, right @ writer
            cross = (torch.einsum("an,ni,nk->aik", basis, left, right_write)
                     + torch.einsum("an,ni,nk->aik", basis, right, left_write))
            self_term = torch.einsum("an,nk,nl->akl", basis, left_write, right_write)
            cross_tensors.append(cross); self_tensors.append(self_term)
            basis_norm = torch.linalg.vector_norm(basis, dim=0)
            cross_score = basis_norm * (torch.linalg.vector_norm(left, dim=1)*torch.linalg.vector_norm(right_write, dim=1)
                                      + torch.linalg.vector_norm(right, dim=1)*torch.linalg.vector_norm(left_write, dim=1))
            self_score = basis_norm * torch.linalg.vector_norm(left_write, dim=1)*torch.linalg.vector_norm(right_write, dim=1)
            cross_scores.append(cross_score); self_scores.append(self_score)
            z = torch.linspace(-.5 + .03*route_index, .5 + .03*route_index, width); z = z/z.norm()
            changed = x + writer @ z
            direct = basis @ ((left @ changed)*(right @ changed) - (left @ x)*(right @ x))
            replay = torch.einsum("aik,i,k->a", cross, x, z) + torch.einsum("akl,k,l->a", self_term, z, z)
            formula_errors.append(float(torch.linalg.vector_norm(direct-replay)
                                        / torch.linalg.vector_norm(direct).clamp_min(1e-30)))
    cross = torch.stack(cross_tensors); self_term = torch.stack(self_tensors)
    cross_score = torch.stack(cross_scores); self_score = torch.stack(self_scores)
    cross_spectra, self_spectra = spectra(cross), spectra(self_term)
    head_singular = torch.as_tensor(cross_spectra[0]["singular_values"])
    head_energy = float(head_singular[0].square()/head_singular.square().sum().clamp_min(1e-30))
    pair_cosines = {}
    for i, j in itertools.combinations(range(len(LABELS)), 2):
        value = float((cross_score[i] @ cross_score[j])
                      / (cross_score[i].norm()*cross_score[j].norm()).clamp_min(1e-30))
        pair_cosines[f"{LABELS[i]}|{LABELS[j]}"] = value
    median_cosine = float(np.median(list(pair_cosines.values())))
    pooled = torch.linalg.vector_norm(cross_score, dim=0); order = torch.argsort(pooled, descending=True, stable=True)
    squared = pooled[order].square(); total = squared.sum().clamp_min(1e-30)
    cutoffs = (1, 4, 16, 64, 256, 1024)
    concentration = {str(k): float(squared[:k].sum()/total) for k in cutoffs}
    participation = cross_score.sum(dim=0).square()/cross_score.square().sum(dim=0).clamp_min(1e-30)
    basis_error = float((basis @ basis.T-torch.eye(reader_rank)).abs().max())
    max_formula_error = max(formula_errors)
    fd, temp_name = tempfile.mkstemp(prefix=TENSOR_OUT.stem+".", suffix=".npz", dir=TENSOR_OUT.parent); os.close(fd)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(temp_path, reader_basis=reader_basis, reader_cell_vectors=cell_vectors.astype(np.float32),
            reader_cell_singular_values=cell_singular.astype(np.float32), cross=cross.numpy(), self=self_term.numpy(),
            cross_factor_scores=cross_score.numpy(), self_factor_scores=self_score.numpy(),
            pooled_factor_order=order.numpy(), factor_head_participation=participation.numpy(), routes=np.asarray(LABELS))
        os.replace(temp_path, TENSOR_OUT)
    finally:
        if temp_path.exists(): temp_path.unlink()
    tensor_sha = sha(TENSOR_OUT)
    A = bool(authority_ok and reader_rank == 4 and basis_error <= BARS["basis_orthonormal"]
        and max_formula_error <= BARS["formula_relative_error"]
        and finite([cross_spectra, self_spectra, pair_cosines, concentration, formula_errors])
        and PRICE == {"checkpoint_loads": 1, "model_forwards": 0, "sequence_evaluations": 0,
                      "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0,
                      "saved_tensor_families": 2})
    B = head_energy >= BARS["shared_head_energy"]
    C = median_cosine >= BARS["factor_score_cosine"]
    D = concentration["256"] >= BARS["top256_squared_mass"]
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A else "shared_and_concentrated_weight_capability" if B and C and D
                else "shared_but_diffuse_weight_capability" if B and C else "head_private_weight_capability")
    result = {"schema": "temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_cpu_queue_only",
        "started_utc": started_utc, "finished_utc": now(), "serial_seconds": time.perf_counter()-started,
        "authority_sha256": observed, "reader_subspace": {"source_cells": ["A1:0", "A1:1", "A2:0", "A2:1"],
            "rank": reader_rank, "cell_singular_values": cell_singular.tolist(),
            "orthonormal_max_abs_error": basis_error, "semantic_object": "row_span_projector_not_basis_orientation"},
        "tensor_artifact": {"path": str(TENSOR_OUT.relative_to(ROOT)), "sha256": tensor_sha,
            "cross_shape": list(cross.shape), "self_shape": list(self_term.shape)},
        "formula_relative_errors": dict(zip(LABELS, formula_errors)), "max_formula_relative_error": max_formula_error,
        "cross_unfolding_spectra": cross_spectra, "self_unfolding_spectra": self_spectra,
        "shared_head_top_mode_energy_fraction": head_energy, "factor_score_pair_cosines": pair_cosines,
        "factor_score_median_pair_cosine": median_cosine, "pooled_cross_factor_squared_mass": concentration,
        "factor_head_participation_quantiles": {str(q): float(torch.quantile(participation, q)) for q in (0., .25, .5, .75, 1.)},
        "activation_occupancy_opened": False, "new_causal_edge_selected": None,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("reader_subspace", "tensor_artifact",
        "max_formula_relative_error", "shared_head_top_mode_energy_fraction",
        "factor_score_median_pair_cosine", "pooled_cross_factor_squared_mass",
        "factor_head_participation_quantiles", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
