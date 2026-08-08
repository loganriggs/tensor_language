"""Rung 1-2 machinery: exact folds, materialized tables, spectra, hard gates.

RUNG 1 (exact fold).  Every layer is written as its tensor and machine-checked
against the model's own forward inside exact_math() (tf32 off SYMMETRICALLY on
both sides).  Gates, all HARD:
    MLP tensor identity                   rel < 1e-6
    RMSNorm gauge  MLP(rms x) = T(x,x)D/|x|^2   rel < 1e-6
    layer-0 attention table identity      rel < 1e-6
    fold_forward vs forward               max logit diff < 1e-5
    planted known-answer table test       < 1e-10 (fp64)

RUNG 2 (materialized tables + spectra).  The layer-0 score table at relative
distance delta is
    S_h(delta) = Q_h R(delta) K_h^T / hd ,      Q_h, K_h in R^{V x hd}
so it is EXACTLY rank <= hd = 16 and its singular values can be computed from
the (V x 16) factors without ever forming the V x V matrix:
    Q_h R = Qa Ra  (QR),  K_h = Qb Rb  (QR)  ->  sv(S) = sv(Ra Rb^T) / hd
That identity is itself controlled here against a direct SVD of the materialized
table for one head, so the cheap path can never silently drift.

Artifacts written per checkpoint (stem = the cell stem):
    {stem}_fold.npz     Q1/K1/Q2/K2/Vv factors, MLP tensors, spectra, top pairs
    {stem}_tables_d{delta}.npy   the materialized V x V tables (only with
                                 --materialize; 67 MB per head-branch at V=4096)
and a 'fold' block merged into {stem}.json.

Usage
    python tf_fold.py --stem tf_vanilla_d1_w32_v4096_s0 [--materialize]
                      [--deltas 0,1,2] [--topk 20]
"""
import argparse
import json
import os

import numpy as np
import torch

import tf_corpus
import tf_model as M

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_checkpoint(stem, device=DEV):
    ck = torch.load(f'{HERE}/{stem}.pt', map_location=device, weights_only=False)
    cfg = M.TFConfig(**{k: v for k, v in ck['cfg'].items()
                        if k in M.TFConfig.__dataclass_fields__})
    model = M.make_model(cfg, device)
    model.load_state_dict(ck['state_dict'])
    model.eval().float()
    return model, cfg, ck


@torch.no_grad()
def table_spectrum(Qh, Kh, R, hd):
    """Singular values of the V x V table Q R K^T / hd computed from the
    (V x hd) FACTORS alone -- exact, and never forms the V x V matrix.
    A = Q R = Qa Ra, B = K = Qb Rb (thin QR, orthonormal Qa/Qb), so
    A B^T = Qa (Ra Rb^T) Qb^T and sv(A B^T) = sv(Ra Rb^T)."""
    A = Qh.double() @ R.double()
    Ra = torch.linalg.qr(A, mode='r').R
    Rb = torch.linalg.qr(Kh.double(), mode='r').R
    return torch.linalg.svdvals(Ra @ Rb.t()) / hd


@torch.no_grad()
def table_spectrum_eig(Qh, Kh, R, hd):
    """INDEPENDENT second path for the same singular values, sharing no code
    with the QR route: for A (V x hd), B (V x hd), the nonzero eigenvalues of
    (A B^T)(A B^T)^T = A (B^T B) A^T equal those of (B^T B)(A^T A), an hd x hd
    product.  sv = sqrt(eig), sorted descending."""
    A = Qh.double() @ R.double()
    B = Kh.double()
    ev = torch.linalg.eigvals(B.t() @ B @ (A.t() @ A)).real
    return ev.clamp_min(0).sqrt().sort(descending=True).values / hd


def eff_rank(sv, tol=1e-6):
    """Numerical rank (sigma > tol * sigma_max), participation ratio, and the
    spectral-entropy effective rank exp(H(p)) with p = sigma / sum sigma."""
    sv = np.asarray(sv, dtype=np.float64)
    s = sv.sum()
    if s <= 0:
        return {'numerical_rank': 0, 'participation_ratio': 0.0,
                'entropy_rank': 0.0, 'top1_share': 0.0}
    p = sv / s
    nz = p[p > 0]
    return {'numerical_rank': int((sv > tol * sv.max()).sum()),
            'participation_ratio': float(s ** 2 / (sv ** 2).sum()),
            'entropy_rank': float(np.exp(-(nz * np.log(nz)).sum())),
            'top1_share': float(sv.max() / s)}


@torch.no_grad()
def fold_report(stem, deltas=(0, 1, 2), materialize=False, topk=20,
                gate_batch=4, device=DEV, direct_svd=False):
    model, cfg, ck = load_checkpoint(stem, device)
    V, H, hd = cfg.vocab, cfg.n_heads, cfg.head_dim
    vocab = tf_corpus.load_vocab(cfg.vocab)
    rep = {'stem': stem, 'variant': cfg.variant, 'depth': cfg.depth,
           'width': cfg.width, 'stream_width': cfg.stream_width,
           'n_heads': H, 'head_dim': hd, 'vocab': V, 'deltas': list(deltas)}

    # ---------------- HARD GATES ----------------
    held = torch.from_numpy(
        tf_corpus.load_split(cfg.vocab, 'held', gate_batch)).to(device)
    gate = M.check_fold_identities(model, held[:, :min(cfg.T, 128)],
                                   verbose=True)
    rep['identity_gate'] = gate
    planted = M.planted_qk_test(device=device)
    rep['planted_known_answer'] = planted
    assert gate['pass'], f'FOLD IDENTITY GATE FAILED for {stem}: {gate}'
    assert planted['pass'], f'PLANTED TABLE TEST FAILED: {planted}'

    # ---------------- factors + spectra ----------------
    fold = model.fold_layer0_qk(deltas=deltas, materialize=False, device=device)
    arts = {k: fold[k].cpu().numpy() for k in ('Q1', 'K1', 'Q2', 'K2', 'Vv')}
    spec = {}
    for d in deltas:
        R = M.rot_matrix(d, hd, device)
        for br, (Qn, Kn) in (('s1', ('Q1', 'K1')), ('s2', ('Q2', 'K2'))):
            svs = []
            for h in range(H):
                sv = table_spectrum(fold[Qn][h], fold[Kn][h], R, hd).cpu().numpy()
                svs.append(sv)
            svs = np.stack(svs)
            arts[f'sv_{br}_d{d}'] = svs
            spec[f'{br}_d{d}'] = {
                'per_head': [eff_rank(s) for s in svs],
                'rank_bound': hd,
                'mean_numerical_rank': float(np.mean(
                    [eff_rank(s)['numerical_rank'] for s in svs])),
                'mean_entropy_rank': float(np.mean(
                    [eff_rank(s)['entropy_rank'] for s in svs]))}
    rep['spectra'] = spec

    # CONTROL (mandatory, cheap): the QR-factor spectrum must agree with the
    # independent eigenvalue route on every head -- otherwise the cheap path
    # could drift silently.
    R0 = M.rot_matrix(deltas[0], hd, device)
    worst = 0.0
    for h in range(H):
        sv_e = table_spectrum_eig(fold['Q1'][h], fold['K1'][h], R0, hd)
        sv_q = torch.as_tensor(arts[f'sv_s1_d{deltas[0]}'][h],
                               device=sv_e.device, dtype=sv_e.dtype)
        worst = max(worst, float((sv_e - sv_q).abs().max()
                                 / sv_q.max().clamp_min(1e-30)))
    rep['control_spectrum_qr_vs_eig_rel'] = worst
    assert worst < 1e-8, f'spectrum control failed: {worst}'
    # OPTIONAL second control: a dense SVD of the actually-materialized V x V
    # table (head 0).  Run on CPU -- V x V in fp64 is 134 MB at V=4096 and the
    # GPU may be shared.
    if direct_svd:
        tab0 = ((fold['Q1'][0].cpu().double() @ R0.cpu().double())
                @ fold['K1'][0].cpu().double().t() / hd)
        sv_d = torch.linalg.svdvals(tab0).numpy()
        sv_f = arts[f'sv_s1_d{deltas[0]}'][0]
        n = min(len(sv_d), len(sv_f))
        rep['control_spectrum_factor_vs_dense_svd_rel'] = float(
            np.abs(sv_d[:n] - sv_f[:n]).max() / max(sv_d.max(), 1e-30))
        rep['control_dense_svd_tail_over_rank_bound'] = float(
            np.abs(sv_d[hd:]).max()) if len(sv_d) > hd else 0.0
        del tab0

    # ---------------- MLP tensors ----------------
    mlp_stats = []
    for li in range(cfg.depth):
        T = model.fold_mlp(li, device=device)
        arts[f'mlp_T_l{li}'] = T.numpy()
        Tf = T.reshape(T.shape[0], -1).double()
        sv = torch.linalg.svdvals(Tf).numpy()
        mlp_stats.append({'shape': list(T.shape), 'fro': float(T.norm()),
                          'unfolding_spectrum': eff_rank(sv),
                          'sv_top8': [float(x) for x in sv[:8]]})
        arts[f'mlp_sv_l{li}'] = sv
        del T
    rep['mlp'] = mlp_stats

    # ---------------- top token pairs (rung-2 labelling) ----------------
    d0 = deltas[0]
    R = M.rot_matrix(d0, hd, device)
    tops = []
    for h in range(H):
        s1 = (fold['Q1'][h] @ R) @ fold['K1'][h].t() / hd
        s2 = (fold['Q2'][h] @ R) @ fold['K2'][h].t() / hd
        pat = s1 * s2
        flat = pat.reshape(-1)
        hi = torch.topk(flat, topk)
        lo = torch.topk(-flat, topk)
        tops.append({
            'head': h,
            'max': [[tf_corpus.label(vocab, int(i) // V),
                     tf_corpus.label(vocab, int(i) % V), float(v)]
                    for i, v in zip(hi.indices.cpu(), hi.values.cpu())],
            'min': [[tf_corpus.label(vocab, int(i) // V),
                     tf_corpus.label(vocab, int(i) % V), float(-v)]
                    for i, v in zip(lo.indices.cpu(), lo.values.cpu())],
            'abs_mean': float(pat.abs().mean()),
            'std': float(pat.std())})
        if materialize:
            np.save(f'{HERE}/{stem}_tables_d{d0}_h{h}_s1.npy',
                    s1.cpu().numpy())
            np.save(f'{HERE}/{stem}_tables_d{d0}_h{h}_s2.npy',
                    s2.cpu().numpy())
        del s1, s2, pat, flat
    rep[f'top_pairs_delta{d0}'] = tops
    if model.pred_on:
        rep['predicate'] = {
            'prof_absmax': float(model.pred_prof.abs().max()),
            'b': [float(x) for x in model.pred_b[0]],
            'c': [float(x) for x in model.pred_c[0]],
            'index_note': fold['predicate']['index_sets']}

    np.savez_compressed(f'{HERE}/{stem}_fold.npz', **arts)
    rep['artifact'] = f'{stem}_fold.npz'
    rep['artifact_mib'] = round(
        os.path.getsize(f'{HERE}/{stem}_fold.npz') / 2 ** 20, 2)

    jp = f'{HERE}/{stem}.json'
    out = json.load(open(jp)) if os.path.exists(jp) else {}
    out['fold'] = rep
    json.dump(out, open(jp, 'w'), indent=2)
    print(f'== fold OK for {stem}: gate pass, spectra saved '
          f'({rep["artifact_mib"]} MiB)', flush=True)
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', required=True)
    ap.add_argument('--deltas', default='0,1,2')
    ap.add_argument('--materialize', action='store_true')
    ap.add_argument('--topk', type=int, default=20)
    ap.add_argument('--direct-svd', action='store_true',
                    help='additionally dense-SVD the materialized V x V table '
                         '(CPU, ~134 MB in fp64 at V=4096)')
    a = ap.parse_args()
    r = fold_report(a.stem, tuple(int(x) for x in a.deltas.split(',')),
                    materialize=a.materialize, topk=a.topk,
                    direct_svd=a.direct_svd)
    print(json.dumps({k: v for k, v in r.items()
                      if k in ('identity_gate', 'planted_known_answer',
                               'control_spectrum_qr_vs_eig_rel',
                               'control_spectrum_factor_vs_dense_svd_rel')},
                     indent=2))
