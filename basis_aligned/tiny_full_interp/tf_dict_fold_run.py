"""DESCRIPTION A -- compress the EXACT FOLDED per-head-branch object.

This is the literal port of ../qk_mdl RESULTS_l0_mdl.md section 3/3b/3c to the
depth-1 width-128 seed-0 cell: the object is the exact layer-0 fold (four
(V x head_dim) factor tables per head), the coder is a sparse dictionary with
OMP/least-squares codes, the objective is either plain MSE or the ported
context-expected OV objective (eq. dagger of ov_metric_explainer.md), and the
anchor hybrid (exact rows for the top-B tokens by attribution + dictionary for
the tail) is retested against THIS object rather than against the embedding.

Everything is scored by HELD CROSS-ENTROPY (primary; against the DATA, per
Logan's redirection) and KL from the model (secondary).  Hyperparameters that
are not part of a scheme's definition (the objective blend, the T knob) are
selected on the ESTIMATION split; held is only ever read out.

Run:  python tf_dict_fold_run.py --stem tf_vanilla_d1_w128_b8192_s0
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

import tf_compress as CC
import tf_corpus
import tf_dict_lib as L
from tf_compress import Bits, bits_dense, bits_index

HERE = os.path.dirname(os.path.abspath(__file__))
log = L.log

REGISTERED = {
    "written_before_any_measurement": True,
    "R1_fold_is_an_expansion": "ARITHMETIC, stated first because it frames "
        "everything: the folded object is 4 x (V x hd) x H = 4.19M numbers, "
        "3.1x the whole 1.34M-parameter model and 4x the embedding it is "
        "derived from, because the fold trades 4 x 128 x 128 = 65k weights for "
        "a V-row table. So a description that STORES the fold cannot beat the "
        "model unless the fold is coded below 2.10 Mbit (the four projection "
        "matrices it replaces). Prediction: no dictionary point below 2.10 "
        "Mbit is within 0.10 nats of the model, so Description A LOSES as a "
        "description while still being the right object for the method test.",
    "R2_objective_ports": "The context-expected OV objective will beat plain "
        "MSE at matched bits at the LOW end of the budget range (the parent's "
        "crossover was ~12% of raw bits), by >=15% of the CE gap to the model.",
    "R3_grouping_flips_with_vocabulary": "The parent's per-head-branch "
        "grouping (rows of dim 2*hd) will be DOMINATED here by a joint "
        "dictionary over the token's whole folded signature, because the "
        "index+coefficient cost per row is amortised over 2*hd=32 numbers "
        "instead of 256 at V=50304; per-head-branch structure was a quality "
        "finding at their scale and is a bits finding at ours.",
    "R4_anchors_port": "Exact anchor rows for the top-B tokens by attribution "
        "plus a dictionary tail will beat the pure dictionary at matched bits "
        "somewhere on the curve, by >=1.3x in bits at matched CE. This is a "
        "RETEST: the earlier negative in FINDING 12 was measured against the "
        "embedding table with no context objective.",
    "R5_nothing_beats_the_model_CE": "No point of Description A reaches the "
        "model's own held CE (4.7114). If any does it will be at the high-bit "
        "end and will be a regularisation effect, not compression.",
    "R6_quantisation_still_wins_the_bits": "As a FULL description, every "
        "Description-A point is dominated by the existing quantisation "
        "frontier (7.59 Mbit at KL 0.004), because the fold is an expansion.",
}


def save(out, path):
    json.dump(out, open(path, 'w'), indent=1)


# ---------------------------------------------------------------------------
def fold_bits(mode, n, k, V, H, hd, b_atom=32, b_coef=32):
    gs = L.groups(mode, H)
    bt = Bits()
    for gi, g in enumerate(gs):
        d = len(g) * hd
        bt.add(**{f'g{gi}_atoms': bits_dense(n * d, b_atom),
                  f'g{gi}_idx': bits_index(V * k, n),
                  f'g{gi}_coef': bits_dense(V * k, b_coef)})
    return bt


def total_bits(D, fold_b):
    """The FULL description: the fold's bits replace the four projection
    matrices; everything else (embedding table, Wv, Wproj, MLP, bias) is still
    needed and is charged at fp32."""
    proj = 4 * 128 * 128 * 32 if D.Ws == 128 else 4 * D.Ws * D.Ws * 32
    return D.n_params_model * 32 - proj + fold_b


# ---------------------------------------------------------------------------
def code_groups(X, Mfull, mode, n, k, H, iters, seed=0, anchors=None,
                order=None):
    """Apply the dictionary (or the anchor hybrid) group by group."""
    R = X.clone()
    info = []
    for g in L.groups(mode, H):
        Xg = X[:, g].contiguous()
        Mg = None if Mfull is None else Mfull[:, g].contiguous()
        if anchors:
            Rg, _, meta = L.anchor_dict(Xg, Mg, n, k, anchors, order,
                                        iters=iters, seed=seed)
            info.append(meta)
        else:
            Dic, idx, coef = L.dict_learn(Xg, Mg, n, k, iters=iters, seed=seed)
            Rg = L.sparse_recon(Dic, idx, coef)
        R[:, g] = Rg
    return R, info


def fvu(X, R):
    return float(((X - R) ** 2).sum() / (X ** 2).sum())


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default='tf_vanilla_d1_w128_b8192_s0')
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--held_seq', type=int, default=256)
    ap.add_argument('--est_seq', type=int, default=128)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    path = f'{HERE}/{a.stem}_dict_fold.json'
    out = {'stem': a.stem, 'registered_predictions': REGISTERED,
           'config': vars(a)}

    D = L.FoldDesc(a.stem)
    V, H, hd = D.V, D.H, D.hd
    SC = dict(n_seq=a.held_seq, T=256)
    SCE = dict(split='est', n_seq=a.est_seq, T=256)

    MREF = [None]      # the reference context metric, filled in below

    def score(R, renorm=True):
        FT = L.X_to_FT(D, R, renorm=renorm)
        s = D.score_fold(FT, **SC)
        s['ce_est'] = D.score_fold(FT, **SCE)['ce']
        if MREF[0] is not None:
            e = X - R
            s['m_ctx'] = float(torch.einsum('vbd,vbde,vbe->', e, MREF[0], e))
        return s

    # ---------------------------------------------------- positive controls
    log('positive controls')
    ctl = D.gate(**SC)
    X = L.build_X(D)
    ctl['raw_fold_numbers'] = int(X.numel())
    ctl['raw_fold_bits'] = int(X.numel() * 32)
    ctl['model_bits'] = int(D.n_params_model * 32)
    # identity dictionary: one atom per token, k=1, coefficient 1 -> EXACT
    idI = torch.arange(V, device=X.device)[:, None]
    cfI = torch.ones(V, 1, device=X.device)
    RI = L.sparse_recon(X, idI, cfI)
    ctl['identity_dictionary'] = {
        'max_abs_err': float((RI - X).abs().max()),
        **score(RI, renorm=False),
        'bits': fold_bits('joint', V, 1, V, H, hd).total}
    log('  identity dictionary', ctl['identity_dictionary']['ce'])
    out['controls'] = ctl
    save(out, path)

    # ------------------------------------------------------------- metrics
    log('building the context-expected OV metric (eq. dagger)')
    q, cnt = L.unigram_q(D)
    t0 = time.time()
    Mv, Ms = L.ctx_metrics(D, q, verbose=False)
    out['controls']['metric_seconds'] = time.time() - t0
    out['controls']['metric_mass_top50_share'] = float(
        (torch.sort(Mv.sum((1, 2, 3)) * 256 + Ms.sum((1, 2, 3)) * 256 ** 2,
                    descending=True).values[:50]).sum()
        / (Mv.sum() * 256 + Ms.sum() * 256 ** 2))

    MREF[0] = L.metric_at(Mv, Ms, 256, blend=1.0)

    # ------------------------------------------------- objective selection
    # blend/T are hyperparameters -> chosen on est, never on held.
    log('objective grid (selected on est)')
    grid = []
    Ts = (1, 16, 256)
    blends = (0.0, 0.25, 0.5, 0.8, 1.0)
    for n, k in ((256, 2), (1024, 8)):
        for T in Ts:
            for b in blends:
                if b == 0.0 and T != Ts[0]:
                    continue                     # b=0 is MSE, T is irrelevant
                Mf = None if b == 0.0 else L.metric_at(Mv, Ms, T, blend=b)
                R, _ = code_groups(X, Mf, 'joint', n, k, H, a.iters)
                s = score(R)
                grid.append({'n': n, 'k': k, 'T': T, 'blend': b,
                             'fvu': fvu(X, R), **s})
                log(f'  n={n} k={k} T={T} blend={b} ce_est '
                    f'{s["ce_est"]:.4f} ce_held {s["ce"]:.4f}')
    out['objective_grid'] = grid
    save(out, path)
    best = {}
    for n, k in ((256, 2), (1024, 8)):
        rows = [r for r in grid if r['n'] == n and r['k'] == k]
        bb = min(rows, key=lambda r: r['ce_est'])
        best[f'{n}_{k}'] = {'T': bb['T'], 'blend': bb['blend']}
    # one setting for the whole sweep: the est-argmin at the low-bit anchor
    T_sel = best['256_2']['T']
    b_sel = best['256_2']['blend']
    b_sel_hi = best['1024_8']['blend']
    out['objective_selected'] = {'T': T_sel, 'blend_low': b_sel,
                                 'blend_high': b_sel_hi, 'per_budget': best}
    log('selected', out['objective_selected'])

    # -------------------------------------------------------- the main sweep
    log('main sweep')
    Msel = L.metric_at(Mv, Ms, T_sel, blend=b_sel)
    rows = []
    budgets = [(64, 1), (128, 2), (256, 2), (256, 4), (512, 4), (1024, 4),
               (1024, 8), (2048, 8), (4096, 8)]
    if a.quick:
        budgets = [(64, 1), (256, 2), (1024, 8)]
    for mode in ('joint', 'perhb'):
        for n, k in budgets:
            if mode == 'perhb' and n > 1024:
                continue
            for obj, Mf in (('mse', None), ('ctx', Msel)):
                t0 = time.time()
                R, _ = code_groups(X, Mf, mode, n, k, H, a.iters)
                s = score(R)
                bt = fold_bits(mode, n, k, V, H, hd)
                rows.append({'family': f'dict_{mode}', 'obj': obj, 'n': n,
                             'k': k, 'mode': mode, 'fold_bits': bt.total,
                             'pct_raw': bt.total / (X.numel() * 32),
                             'bits_total': total_bits(D, bt.total),
                             'fvu': fvu(X, R), 'bill': bt.to_json(),
                             'secs': time.time() - t0, **s})
                log(f'  {mode} {obj} n={n} k={k} '
                    f'{bt.total / 1e6:.2f} Mbit ce {s["ce"]:.4f} '
                    f'kl {s["kl"]:.4f}')
                save({**out, 'sweep': rows}, path)
    out['sweep'] = rows
    save(out, path)

    # ------------------------------------------------ reference: low rank
    log('low-rank reference on the same object')
    lr = []
    for r in (2, 4, 8, 16, 32, 64, 128):
        R, bt = L.lowrank(X, r)
        s = score(R)
        lr.append({'family': 'svd_joint', 'rank': r, 'fold_bits': bt.total,
                   'pct_raw': bt.total / (X.numel() * 32),
                   'bits_total': total_bits(D, bt.total), 'fvu': fvu(X, R),
                   'bill': bt.to_json(), **s})
        log(f'  rank {r} {bt.total / 1e6:.2f} Mbit ce {s["ce"]:.4f}')
    out['lowrank'] = lr
    save(out, path)

    # --------------------------------------------------------- null control
    log('random-dictionary null at matched bits')
    nulls = []
    for n, k in ((256, 2), (1024, 8)):
        g = torch.Generator(device='cpu').manual_seed(7)
        Dic = X[torch.randperm(V, generator=g)[:n].to(X.device)].clone()
        Dic = Dic / Dic.reshape(n, -1).norm(dim=1)[:, None, None]
        for obj, Mf in (('mse', None), ('ctx', Msel)):
            idx, coef = L.omp_metric(Dic, X, Mf, k)
            R = L.sparse_recon(Dic, idx, coef)
            s = score(R)
            nulls.append({'family': 'random_dict_null', 'obj': obj, 'n': n,
                          'k': k, 'fold_bits': fold_bits('joint', n, k, V, H,
                                                         hd).total,
                          'fvu': fvu(X, R), **s})
            log(f'  null n={n} k={k} {obj} ce {s["ce"]:.4f}')
    out['nulls'] = nulls
    save(out, path)

    # ------------------------------------------------------------- anchors
    log('anchor hybrid (RETEST of the parent 3c result on the FOLDED object)')
    # attribution orders, all estimation-side / weight-side only
    Mref = L.metric_at(Mv, Ms, T_sel, blend=1.0)
    Rref, _ = code_groups(X, Msel, 'joint', 256, 2, H, a.iters)
    err = X - Rref
    attr_ctx = torch.einsum('vbd,vbde,vbe->v', err, Mref, err)
    expo = Mref.diagonal(dim1=-2, dim2=-1).sum((1, 2))
    gg = torch.Generator(device='cpu').manual_seed(11)
    orders = {
        'ctx_error': torch.argsort(attr_ctx, descending=True),
        'exposure': torch.argsort(expo, descending=True),
        'frequency': torch.argsort(cnt, descending=True),
        'random': torch.randperm(V, generator=gg).to(X.device),
    }
    out['attribution_top20'] = {
        k: [int(i) for i in v[:20]] for k, v in orders.items()}
    anc = []
    cases = [('joint', 64, 1), ('joint', 256, 2), ('perhb', 64, 1)]
    if a.quick:
        cases = [('joint', 256, 2)]
    for name, order in orders.items():
        for B in (32, 128, 512, 1024):
            for mode, n, k in cases:
                R, meta = code_groups(X, Msel, mode, n, k, H, a.iters,
                                      anchors=B, order=order)
                s = score(R)
                bt = Bits(anchor_rows=bits_dense(B * 4 * H * hd, 32),
                          anchor_ids=bits_index(B, V))
                for gi, g in enumerate(L.groups(mode, H)):
                    d = len(g) * hd
                    bt.add(**{f'g{gi}_atoms': bits_dense(n * d, 32),
                              f'g{gi}_idx': bits_index((V - B) * k, n),
                              f'g{gi}_coef': bits_dense((V - B) * k, 32)})
                anc.append({'family': 'anchor_hybrid', 'attr': name, 'B': B,
                            'mode': mode, 'n': n, 'k': k,
                            'fold_bits': bt.total,
                            'pct_raw': bt.total / (X.numel() * 32),
                            'bits_total': total_bits(D, bt.total),
                            'fvu': fvu(X, R), 'bill': bt.to_json(), **s})
                log(f'  anchors {name} {mode} B={B} n={n} k={k} '
                    f'{bt.total / 1e6:.2f} Mbit ce {s["ce"]:.4f}')
                save({**out, 'anchors': anc}, path)
    out['anchors'] = anc

    # ------------------------------------- does the metric predict CE at all?
    log('metric-vs-CE validation')
    pts = [r for r in rows if r['mode'] == 'joint'] + lr
    mval = {}
    try:
        from scipy.stats import spearmanr
        ce = [r['ce'] for r in pts]
        mval = {'spearman_fvu_vs_ce':
                float(spearmanr([r['fvu'] for r in pts], ce).statistic),
                'spearman_mctx_vs_ce':
                float(spearmanr([r['m_ctx'] for r in pts], ce).statistic),
                'n_points': len(pts)}
    except Exception as e:
        mval = {'error': str(e)}
    # exact eq.(dagger) on a few arms: validates the block-diagonal shortcut
    ex = []
    for r_ in (4, 32):
        R, _ = L.lowrank(X, r_)
        num, rel = L.ctx_cost_exact(D, L.X_to_FT(D, R), q)
        e = X - R
        ex.append({'arm': f'svd{r_}', 'exact_rel': rel,
                   'blockdiag': float(torch.einsum('vbd,vbde,vbe->', e,
                                                   MREF[0], e))})
    for n, k in ((64, 1), (1024, 8)):
        R, _ = code_groups(X, Msel, 'joint', n, k, H, a.iters)
        num, rel = L.ctx_cost_exact(D, L.X_to_FT(D, R), q)
        e = X - R
        ex.append({'arm': f'dict{n}_{k}', 'exact_rel': rel,
                   'blockdiag': float(torch.einsum('vbd,vbde,vbe->', e,
                                                   MREF[0], e))})
    mval['exact_vs_blockdiag'] = ex
    out['metric_validation'] = mval
    save(out, path)
    log('done ->', path)


if __name__ == '__main__':
    main()
