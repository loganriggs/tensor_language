"""DESCRIPTION B -- the same ported method applied where the bits actually are.

Description A (tf_dict_fold_run.py) compresses the exact fold.  That is the
right object for testing the METHOD, but at this scale the fold is a 4x
EXPANSION of the embedding it is derived from, so no fold-storing description
can be short.  Description B keeps the model's own parameterisation and codes
the one table that holds 78% of its bits -- the tied embedding -- but chooses
the dictionary and the codes in the geometry the FOLD imposes:

    metric on an embedding row = J_t^T  M_ctx(t)  J_t

with J_t the exact Jacobian of the folded query/key rows with respect to that
embedding row (through both RMS norms) and M_ctx the ported context-expected OV
metric of ov_metric_explainer.md eq. (dagger).  This is the sense in which
"fold the embedding into attention, then learn a sparse dictionary" is a method
and not just a pair of words: the fold supplies the metric, the dictionary
supplies the code.

BECAUSE THE EMBEDDING IS TIED, a second term is needed that the parent program
never had to model: the WRITE role (the unembedding).  Its Gauss-Newton
(Fisher) metric is measured on the estimation split.  It is clearly labelled as
an addition to the port, and its weight is a hyperparameter selected on est.

Run:  python tf_dict_emb_run.py --stem tf_vanilla_d1_w128_b8192_s0
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import tf_compress as CC
import tf_corpus
import tf_dict_lib as L
from tf_compress import Bits, bits_dense, bits_index

HERE = os.path.dirname(os.path.abspath(__file__))
log = L.log

REGISTERED = {
    "written_before_any_measurement": True,
    "S1_fold_metric_beats_plain_MSE": "A dictionary on the embedding learned "
        "under the fold-derived context metric will beat the same dictionary "
        "learned under plain embedding MSE at matched bits, by >=0.02 nats of "
        "held CE somewhere in the 1-7 Mbit range.",
    "S2_write_role_dominates": "Because the table is tied, the write-role "
        "term will matter MORE than the read-role term: the est-selected "
        "write weight will be non-zero, and a read-only metric will lose to "
        "the blend by more than a write-only metric does.",
    "S3_dictionary_loses_to_quantisation": "Stated as the adversarial null: "
        "sparse dictionary coding of the embedding will NOT beat plain scalar "
        "quantisation at matched bits, because 1024 atoms of 128 numbers "
        "already cost an eighth of the table before a single code is written. "
        "If it does win it will only be below ~2 Mbit.",
    "S4_anchors_port_here_if_anywhere": "Exact anchor rows for the top-B "
        "tokens by fold-metric attribution plus a dictionary tail will beat "
        "the pure dictionary at matched bits; this is the retest of the "
        "FINDING-12 negative with the context objective in place.",
    "S5_nothing_beats_the_model_CE_without_refitting": "No code-only "
        "description reaches held CE 4.7114. The only arm with a chance is "
        "the one whose coefficients are refit against estimation-split cross-"
        "entropy (bits unchanged), and even that is predicted to land within "
        "0.01 nats above the model rather than below it.",
}


def save(out, path):
    json.dump(out, open(path, 'w'), indent=1)


# ===========================================================================
# THE PULLBACK: fold-space metric -> embedding-row metric
# ===========================================================================
@torch.no_grad()
def emb_jacobian_metric(D, Mfold, chunk=256):
    """M_emb[t] = sum_blocks J_b(t)^T Mfold[t,b] J_b(t), exact through both
    RMS norms.  Mfold: (V, 4H, hd, hd) -> returns (V, 128, 128)."""
    V, Ws, H, hd = D.V, D.Ws, D.H, D.hd
    W = D.base['wte_read']
    mats = []
    for h in range(H):
        for b in (1, 2):
            mats.append(D.base[f'Wq{"" if b == 1 else "2"}'][h * hd:(h + 1) * hd])
            mats.append(D.base[f'Wk{"" if b == 1 else "2"}'][h * hd:(h + 1) * hd])
    Wblk = torch.stack(mats, 0)                       # (4H, hd, Ws)
    out = torch.zeros(V, Ws, Ws, device=W.device)
    I_w = torch.eye(Ws, device=W.device)
    I_h = torch.eye(hd, device=W.device)
    for a in range(0, V, chunk):
        sl = slice(a, min(a + chunk, V))
        w = W[sl]
        sw = w.norm(dim=1, keepdim=True) / math.sqrt(Ws)
        e = w / sw
        Je = (I_w[None] - e[:, :, None] * e[:, None, :] / Ws) / sw[:, :, None]
        z = torch.einsum('bhw,cw->cbh', Wblk, e)      # (B, 4H, hd)
        sz = z.norm(dim=2, keepdim=True) / math.sqrt(hd)
        Qn = z / sz
        Jz = (I_h[None, None] - Qn[:, :, :, None] * Qn[:, :, None, :] / hd) \
            / sz[:, :, :, None]
        J = torch.einsum('cbij,bjw,cwv->cbiv', Jz, Wblk, Je)   # (B,4H,hd,Ws)
        out[sl] = torch.einsum('cbiv,cbij,cbjw->cvw', J, Mfold[sl], J)
    return 0.5 * (out + out.transpose(1, 2))


@torch.no_grad()
def write_metric(D, r=24, n_seq=64, T=256, batch=8):
    """Gauss-Newton (Fisher) metric of estimation-split cross-entropy with
    respect to the UNEMBEDDING row of each token:
        M_write[t] = (1/N) sum_n p_nt (1 - p_nt) g'(lg_nt)^2  z_n z_n^T,
    with z = rms(x) the pre-readout activation and g' the derivative of the
    30*tanh(./30) logit squash.  Computed in the top-`r` principal subspace of
    z (the full V x 128 x 128 einsum is 2 PFLOP; this is 0.1 TFLOP), and the
    captured variance share is reported so the approximation is auditable."""
    x = D.eval_seqs('est', n_seq, T)[:, :-1]
    zs, ps = [], []
    for a in range(0, x.shape[0], batch):
        xi = x[a:a + batch]
        B, Tq = xi.shape
        Ws, H, hd = D.Ws, D.H, D.hd
        cos, sin = D.cos[None, :Tq, None, :], D.sin[None, :Tq, None, :]
        e = L.rms(D.base['wte_read'][xi], Ws)
        import tf_model as M

        def qk(Wm):
            zz = (e @ Wm.t()).view(B, Tq, H, hd)
            return M.apply_rot(L.rms(zz, hd), cos, sin)
        s1 = torch.einsum('bqhd,bkhd->bhqk', qk(D.base['Wq']),
                          qk(D.base['Wk'])) / hd
        s2 = torch.einsum('bqhd,bkhd->bhqk', qk(D.base['Wq2']),
                          qk(D.base['Wk2'])) / hd
        pat = (s1 * s2).masked_fill(~D.mask[:Tq, :Tq], 0.0)
        v = (e @ D.base['Wv'].t()).view(B, Tq, H, hd)
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, Tq, D.Dc)
        xx = e + y @ D.base['Wproj'].t()
        xn = L.rms(xx, Ws)
        hh = (xn @ D.base['Left'].t()) * (xn @ D.base['Right'].t())
        xx = xx + hh @ D.base['Down'].t() + D.base['Down_bias']
        z = L.rms(xx, Ws)
        lg = z @ D.base['wte_out'].t()
        p = F.softmax(30 * torch.tanh(lg / 30), -1)
        gp = (1.0 / torch.cosh(lg / 30)) ** 2
        zs.append(z.reshape(-1, Ws))
        ps.append((p * (1 - p) * gp * gp).reshape(-1, D.V))
    Zc = torch.cat(zs)
    C = torch.cat(ps)
    mu = Zc.mean(0, keepdim=True)
    Zc0 = Zc - mu
    cov = Zc0.t() @ Zc0 / Zc.shape[0]
    ev, Ev = torch.linalg.eigh(cov.double())
    P = Ev.float().flip(1)[:, :r].t()                   # (r, Ws)
    share = float(ev.flip(0)[:r].sum() / ev.sum())
    Zp = Zc @ P.t()                                     # (N, r)
    A = torch.einsum('nt,nr,ns->trs', C, Zp, Zp) / Zc.shape[0]
    Mw = torch.einsum('rv,trs,sw->tvw', P, A, P)
    return 0.5 * (Mw + Mw.transpose(1, 2)), {'z_var_share': share, 'r': r}


def norm_metric(M):
    d = M.shape[-1]
    sc = M.diagonal(dim1=-2, dim2=-1).sum() / (M.shape[0] * d)
    return M / sc.clamp_min(1e-30)


# ===========================================================================
def emb_bits(n, k, V, d=128, b_atom=32, b_coef=32):
    return Bits(atoms=bits_dense(n * d, b_atom),
                indices=bits_index(V * k, n),
                coefs=bits_dense(V * k, b_coef))


def refit_codes(D, Dic, idx, coef, steps=400, lr=3e-3, n_seq=64, T=256,
                batch=8, tune_atoms=True):
    """Fit the SAME description (same atoms, same supports, same bit count)
    against ESTIMATION-split cross-entropy instead of a weight-space proxy.
    Bits are unchanged; only the values the bits encode move."""
    Dp = Dic.clone().requires_grad_(tune_atoms)
    cp = coef.clone().requires_grad_(True)
    ps = [cp] + ([Dp] if tune_atoms else [])
    opt = torch.optim.Adam(ps, lr=lr)
    x = D.eval_seqs('est', n_seq, T)
    nb = x.shape[0]
    for s in range(steps):
        a = (s * batch) % nb
        bb = x[a:a + batch]
        if bb.shape[0] < 2:
            continue
        xi, y = bb[:, :-1], bb[:, 1:]
        wte = L.sparse_recon(Dp, idx, cp)[:, 0]
        lg = D.forward(xi, {'wte_read': wte, 'wte_out': wte})
        loss = F.cross_entropy(lg.reshape(-1, D.V).float(), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return Dp.detach(), cp.detach(), float(loss)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default='tf_vanilla_d1_w128_b8192_s0')
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--held_seq', type=int, default=256)
    ap.add_argument('--est_seq', type=int, default=128)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    path = f'{HERE}/{a.stem}_dict_emb.json'
    out = {'stem': a.stem, 'registered_predictions': REGISTERED,
           'config': vars(a)}

    D = L.FoldDesc(a.stem)
    V, Ws = D.V, D.Ws
    SC = dict(n_seq=a.held_seq, T=256)
    SCE = dict(split='est', n_seq=a.est_seq, T=256)
    EMB_BITS = V * Ws * 32
    BODY_BITS = D.n_params_model * 32 - EMB_BITS
    W0 = D.base['wte_read'].clone()

    def score(wte):
        s = D.score_emb(wte, **SC)
        s['ce_est'] = D.score_emb(wte, **SCE)['ce']
        return s

    log('positive controls')
    ctl = D.gate(**SC)
    ctl['model_bits'] = int(D.n_params_model * 32)
    ctl['embedding_bits'] = int(EMB_BITS)
    ctl['body_bits'] = int(BODY_BITS)
    idI = torch.arange(V, device=W0.device)[:, None]
    RI = L.sparse_recon(W0[:, None], idI, torch.ones(V, 1, device=W0.device))
    ctl['identity_dictionary'] = {'max_abs_err': float((RI[:, 0] - W0).abs().max()),
                                  **score(RI[:, 0])}
    out['controls'] = ctl
    save(out, path)

    # ------------------------------------------------------------ metrics
    log('fold metric -> embedding pullback')
    q, cnt = L.unigram_q(D)
    Mv, Ms = L.ctx_metrics(D, q, verbose=False)
    Mctx_fold = L.metric_at(Mv, Ms, 256, blend=1.0)
    del Mv, Ms
    torch.cuda.empty_cache()
    M_read = norm_metric(emb_jacobian_metric(D, Mctx_fold))
    I_fold = torch.eye(D.hd, device=W0.device)[None, None].expand(
        V, 4 * D.H, D.hd, D.hd).contiguous()
    M_foldmse = norm_metric(emb_jacobian_metric(D, I_fold))
    del I_fold, Mctx_fold
    torch.cuda.empty_cache()
    log('write-role Fisher metric')
    M_wr, wmeta = write_metric(D)
    M_write = norm_metric(M_wr)
    out['controls']['write_metric'] = wmeta
    I128 = torch.eye(Ws, device=W0.device)[None].expand(V, Ws, Ws)

    def metric(kind, beta=1.0):
        """kind selects the READ-side term (none / fold-space MSE / the ported
        context-expected OV metric); `beta` is the weight of the write-role
        Fisher term.  Both terms are trace-normalised, so beta is a pure
        trade-off knob and is selected on the estimation split."""
        if kind == 'mse':
            return None
        if kind == 'write':
            M = beta * M_write
        else:
            M = {'ctx': M_read, 'foldmse': M_foldmse}[kind] + beta * M_write
        return (M + 1e-6 * I128)[:, None].contiguous()

    # --------------------------------------------- metric choice (on est)
    log('metric selection grid (est)')
    grid = []
    for kind in ('mse', 'foldmse', 'ctx', 'write'):
        for beta in ((0.0, 0.3, 1.0, 3.0) if kind in ('ctx', 'foldmse')
                     else (1.0,)):
            Mm = metric(kind, beta)
            for n, k in ((256, 2), (1024, 8)):
                Dic, idx, coef = L.dict_learn(W0[:, None], Mm, n, k,
                                              iters=a.iters)
                s = score(L.sparse_recon(Dic, idx, coef)[:, 0])
                grid.append({'metric': kind, 'beta': beta, 'n': n, 'k': k,
                             'bits_emb': emb_bits(n, k, V).total, **s})
                log(f'  {kind} beta={beta} n={n} k={k} '
                    f'ce_est {s["ce_est"]:.4f} ce {s["ce"]:.4f}')
                save({**out, 'metric_grid': grid}, path)
    out['metric_grid'] = grid
    sel = min([g for g in grid if g['n'] == 256],
              key=lambda g: g['ce_est'])
    out['metric_selected'] = {'metric': sel['metric'], 'beta': sel['beta']}
    log('selected metric', out['metric_selected'])
    Msel = metric(sel['metric'], sel['beta'])

    # ------------------------------------------------------- the main sweep
    log('main sweep')
    budgets = [(64, 2), (128, 2), (256, 2), (256, 4), (512, 4), (1024, 4),
               (1024, 8), (2048, 8)]
    if a.quick:
        budgets = [(256, 2), (1024, 8)]
    rows = []
    for n, k in budgets:
        for obj, Mm in (('mse', None), ('ctx', Msel)):
            t0 = time.time()
            Dic, idx, coef = L.dict_learn(W0[:, None], Mm, n, k, iters=a.iters)
            R = L.sparse_recon(Dic, idx, coef)[:, 0]
            s = score(R)
            bt = emb_bits(n, k, V)
            rows.append({'family': 'emb_dict', 'obj': obj, 'n': n, 'k': k,
                         'bits_emb': bt.total,
                         'bits_total': BODY_BITS + bt.total,
                         'bill': bt.to_json(), 'secs': time.time() - t0,
                         'fvu': float(((W0 - R) ** 2).sum() / (W0 ** 2).sum()),
                         **s})
            log(f'  {obj} n={n} k={k} {bt.total / 1e6:.2f} Mbit '
                f'ce {s["ce"]:.4f} kl {s["kl"]:.4f}')
            save({**out, 'sweep': rows}, path)
    out['sweep'] = rows
    save(out, path)

    # ------------------------------------------------------------ anchors
    log('anchor hybrid on the embedding, attribution from the FOLD metric')
    Dic, idx, coef = L.dict_learn(W0[:, None], Msel, 256, 2, iters=a.iters)
    err = (W0 - L.sparse_recon(Dic, idx, coef)[:, 0])[:, None]
    Mattr = Msel if Msel is not None else torch.eye(
        Ws, device=W0.device)[None, None].expand(V, 1, Ws, Ws)
    attr = torch.einsum('vbd,vbde,vbe->v', err, Mattr, err)
    gg = torch.Generator(device='cpu').manual_seed(11)
    orders = {'ctx_error': torch.argsort(attr, descending=True),
              'exposure': torch.argsort(
                  Mattr.diagonal(dim1=-2, dim2=-1).sum((1, 2)),
                  descending=True),
              'frequency': torch.argsort(cnt, descending=True),
              'random': torch.randperm(V, generator=gg).to(W0.device)}
    out['attribution_top20'] = {k: [int(i) for i in v[:20]]
                                for k, v in orders.items()}
    anc = []
    for name, order in orders.items():
        for B in (64, 256, 1024):
            for n, k in ((128, 2), (512, 4)):
                R, bt, meta = L.anchor_dict(W0[:, None], Msel, n, k, B, order,
                                            iters=a.iters)
                s = score(R[:, 0])
                anc.append({'family': 'emb_anchor', 'attr': name, 'B': B,
                            'n': n, 'k': k, 'bits_emb': bt.total,
                            'bits_total': BODY_BITS + bt.total,
                            'bill': bt.to_json(), **s})
                log(f'  anchors {name} B={B} n={n} k={k} '
                    f'{bt.total / 1e6:.2f} Mbit ce {s["ce"]:.4f}')
                save({**out, 'anchors': anc}, path)
    out['anchors'] = anc
    save(out, path)

    # ------------------------------------- reference line: RECODING, not
    # explanation.  Same harness, same held set, so the comparison is exact.
    log('quantisation reference line (recoding, not explanation)')
    ref = []
    for b in (2, 3, 4, 5, 6, 8):
        R, bt = CC.q_scalar(W0, b)
        s = score(R)
        ref.append({'family': 'quant_scalar', 'b': b, 'bits_emb': bt.total,
                    'bits_total': BODY_BITS + bt.total, **s})
        R, bt = CC.q_scalar_entropy(W0, b)
        s = score(R)
        ref.append({'family': 'quant_scalar_entropy', 'b': b,
                    'bits_emb': bt.total, 'bits_total': BODY_BITS + bt.total,
                    **s})
    for bpr in (128, 256, 384, 512, 640):
        R, bt = CC.q_transform(W0, bpr, rot='pca', entropy=True)
        s = score(R)
        ref.append({'family': 'quant_transform', 'bits_per_row': bpr,
                    'bits_emb': bt.total, 'bits_total': BODY_BITS + bt.total,
                    **s})
    for r in (8, 16, 32, 64):
        R, bt = CC.q_lowrank(W0, r)
        s = score(R)
        ref.append({'family': 'lowrank', 'rank': r, 'bits_emb': bt.total,
                    'bits_total': BODY_BITS + bt.total, **s})
    out['reference'] = ref
    save(out, path)

    # ------------------------------------ codes refit against est CROSS-ENTROPY
    log('refit against estimation-split CE (bits unchanged)')
    rf = []
    for n, k in (((256, 2), (1024, 8)) if not a.quick else ((256, 2),)):
        Dic, idx, coef = L.dict_learn(W0[:, None], Msel, n, k, iters=a.iters)
        s0 = score(L.sparse_recon(Dic, idx, coef)[:, 0])
        D2, c2, tl = refit_codes(D, Dic, idx, coef)
        s1 = score(L.sparse_recon(D2, idx, c2)[:, 0])
        bt = emb_bits(n, k, V)
        rf.append({'family': 'emb_dict_ce_refit', 'n': n, 'k': k,
                   'bits_emb': bt.total, 'bits_total': BODY_BITS + bt.total,
                   'before': s0, 'train_loss': tl, **s1})
        log(f'  refit n={n} k={k} ce {s0["ce"]:.4f} -> {s1["ce"]:.4f}')
        save({**out, 'ce_refit': rf}, path)
    out['ce_refit'] = rf
    save(out, path)
    log('done ->', path)


if __name__ == '__main__':
    main()
