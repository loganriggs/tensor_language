#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_energy_identity pred_b_full_core_beats_pruned pred_c_reduced_replay
"""Weights-first joint tensor screen, no trained model forwards or fits.

Repair the v614 instrument by discovering from C,A,B rather than fitting
normalized outputs with an unnormalized predictor. Keep every residual source
in z exactly once; E=[I,lambda17[0]*Down16,O17]. QR removes only exact ambient
nullspaces, retaining the Euclidean metric in the declared z coordinates.

Compare joint symmetric HOSVD + exact sparse core with independent matrix SVD.
HOSVD is a closed-form initializer, NOT an optimized sparse Tucker fit or HT.
No rank is selected using behavioral outcomes. All ranks and prices reported.
Predictions: reduced energy agrees with both mode-Gram traces to 2e-5;
full core is no worse than each pruned core (1e-6 tolerance); reduced CP replay
agrees with original factored path to 2e-4. These are instrument gates only.
PRICE: 0 model forwards, 0 model backwards/updates, 0 learned fits.
"""
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

PREDICTIONS = {'pred_a_energy_identity': 'relative <=2e-5',
               'pred_b_full_core_beats_pruned': 'monotone by budget',
               'pred_c_reduced_replay': 'relative <=2e-4'}
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'basis_aligned/bilinear_quotient/circuits/followups/joint_folded_tucker_weights_v615_result.json'
SNAP = Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240')


def main():
    plan = dict(ranks=[16, 32, 64], fractions=[.01, .1, 1.0], forwards_max=0,
                model_backwards=0, model_updates=0, fit_parameters=0,
                execution_policy='managed_queue_only', discovery='weights_only')
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(plan)); return
    import torch
    import disk_guard
    sys.path.insert(0, str(ROOT / 'basis_aligned/polynomial_causal'))
    from joint_folded_tucker import (mode_grams, tensor_inner, project_core,
        sparse_core, executable_price, projection_error_squared, require_orthonormal)
    t0 = time.perf_counter()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(615)
    state = torch.load(SNAP / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    def weight(name):
        return state[name].to(device='cuda', dtype=torch.float32)
    U = weight('lm_head.weight')
    D = weight('transformer.h.17.mlp.Down.weight')
    L = weight('transformer.h.17.mlp.Left.weight')
    R = weight('transformer.h.17.mlp.Right.weight')
    Dprev = weight('transformer.h.16.mlp.Down.weight') * weight('transformer.h.17.lambdas')[0]
    O = weight('transformer.h.17.attn.c_proj.weight')
    E = torch.cat([torch.eye(L.shape[1], device='cuda'), Dprev, O], 1)
    # Exact thin ambient frames; do not construct C=U@D or A=L@E in full.
    Qu, Ru = torch.linalg.qr(U)
    Qe, Re = torch.linalg.qr(E.T)
    C, A, B = Ru @ D, L @ Re.T, R @ Re.T
    z = torch.randn(16, E.shape[1], device='cuda')
    h = z @ E.T
    native = ((h @ L.T) * (h @ R.T)) @ D.T @ U.T
    reduced = (((z @ Qe @ A.T) * (z @ Qe @ B.T)) @ C.T) @ Qu.T
    replay = float((native - reduced).norm() / native.norm())
    print('exact ambient reduction', E.shape, 'replay', replay, flush=True)
    go, gi = mode_grams(C, A, B)
    energy = tensor_inner(C, A, B, C, A, B)
    disagreement = float(torch.maximum((go.trace()-energy).abs(), (gi.trace()-energy).abs()) / energy)
    eo, wo = torch.linalg.eigh(go)
    ei, pi = torch.linalg.eigh(gi)
    # SVD baseline must use the same tensor metric and ambient coordinates.
    cu, cs, cv = torch.linalg.svd(C, full_matrices=False)
    au, ass, av = torch.linalg.svd(A, full_matrices=False)
    bu, bs, bv = torch.linalg.svd(B, full_matrices=False)
    rows = []
    monotone = True
    for k in plan['ranks']:
        W, P = wo[:, -k:], pi[:, -k:]
        require_orthonormal(W); require_orthonormal(P)
        core = project_core(C, A, B, W, P)
        prices = []
        last = float('inf')
        for fraction in plan['fractions']:
            budget = max(1, int(k * k * (k+1) // 2 * fraction))
            kept = sparse_core(core, budget)
            error = float((projection_error_squared(energy, core, kept) / energy).sqrt())
            monotone &= error <= last + 1e-6
            last = error
            price = executable_price(Qu @ W, Qe @ P, kept)
            prices.append(dict(fraction=fraction, budget=budget, relative_tensor_error=error, **price))
        Ck = (cu[:, :k] * cs[:k]) @ cv[:k]
        Ak = (au[:, :k] * ass[:k]) @ av[:k]
        Bk = (bu[:, :k] * bs[:k]) @ bv[:k]
        base_energy = tensor_inner(Ck, Ak, Bk, Ck, Ak, Bk)
        cross = tensor_inner(C, A, B, Ck, Ak, Bk)
        base_err = float(((energy + base_energy - 2*cross).clamp_min(0) / energy).sqrt())
        # Count deployable low-rank matrix factors, not dense reconstructed matrices.
        baseline_values = k * (U.shape[0] + D.shape[1] + 2*(L.shape[0] + E.shape[1]))
        row = dict(rank=k, joint_hosvd_sparse=prices,
                   any_rank_k_tucker_error_lower_bound=float(torch.maximum(
                       eo[:-k].sum(), ei[:-k].sum()).clamp_min(0).div(energy).sqrt()),
                   independent_matrix_relative_tensor_error=base_err,
                   independent_matrix_stored_values=baseline_values)
        rows.append(row)
        print(json.dumps(row), flush=True)
    payload = dict(schema='joint_folded_tucker_weights_v615', plan=plan, rows=rows,
                   input_dim=E.shape[1], output_dim=U.shape[0],
                   reduced_replay_error=replay, energy_trace_relative_disagreement=disagreement,
                   mode_spectra={'output_eigenvalues': eo.cpu().tolist(),
                                 'input_eigenvalues': ei.cpu().tolist()},
                   predictions={'pred_a_energy_identity': disagreement <= 2e-5,
                       'pred_b_full_core_beats_pruned': bool(monotone),
                       'pred_c_reduced_replay': replay <= 2e-4},
                   scope='weight numerator only; no OOD/removal/reuse/circuit claim',
                   seconds=time.perf_counter()-t0, finished_utc=datetime.now(timezone.utc).isoformat())
    disk_guard.guard_write(100000, label='v615 JSON')
    OUT.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
