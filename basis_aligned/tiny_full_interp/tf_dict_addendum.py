"""SELF-RED-TEAM ADDENDUM.

Two objections to the anchor result that the main runs do not answer, and the
measurements that settle them.

OBJECTION 1 -- "the anchor hybrid is not about EXACT rows, it is just more bits
spent on frequent tokens."  Control: STRATIFIED-k.  Same dictionary, same
attribution order, but the top-B tokens get k_hi active atoms and the tail gets
k_lo, with the extra indices and coefficients charged.  If the smooth version
matches the exact-row version at matched bits, "anchors" is a precision
allocation, not an exact-row phenomenon.

OBJECTION 2 -- "the ported objective's win is really just frequency weighting;
the OV geometry contributes nothing."  Control: a metric that keeps the
exposure weights q_i but throws away the OV directions, i.e. the same per-token
scalar weight times the IDENTITY.  If that recovers the whole gain, the port
reduces to weighting rows by frequency and the OV/cancellation structure of
eq. (dagger) is decoration.
"""
import argparse
import json
import os

import torch

import tf_dict_lib as L
from tf_compress import Bits, bits_dense, bits_index

HERE = os.path.dirname(os.path.abspath(__file__))
log = L.log


@torch.no_grad()
def stratified_k(X, Mb, n, k_hi, k_lo, B, order, iters=6, seed=0):
    """One dictionary, two code lengths."""
    V, nb, d = X.shape
    Dic, idx, coef = L.dict_learn(X, Mb, n, k_hi, iters=iters, seed=seed)
    hi = order[:B]
    lo_mask = torch.ones(V, dtype=torch.bool, device=X.device)
    lo_mask[hi] = False
    lo = torch.nonzero(lo_mask).squeeze(1)
    R = L.sparse_recon(Dic, idx, coef)
    il, cl = L.omp_metric(Dic, X[lo].contiguous(),
                          None if Mb is None else Mb[lo].contiguous(), k_lo)
    R[lo] = L.sparse_recon(Dic, il, cl)
    bt = Bits(atoms=bits_dense(n * nb * d, 32),
              hi_idx=bits_index(B * k_hi, n), hi_coef=bits_dense(B * k_hi, 32),
              lo_idx=bits_index(len(lo) * k_lo, n),
              lo_coef=bits_dense(len(lo) * k_lo, 32),
              strata_ids=bits_index(B, V))
    return R, bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stem', default='tf_vanilla_d1_w128_b8192_s0')
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--held_seq', type=int, default=256)
    a = ap.parse_args()
    path = f'{HERE}/{a.stem}_dict_addendum.json'
    out = {'stem': a.stem}
    D = L.FoldDesc(a.stem)
    V, H, hd = D.V, D.H, D.hd
    X = L.build_X(D)
    q, cnt = L.unigram_q(D)
    Mv, Ms = L.ctx_metrics(D, q, verbose=False)
    Msel = L.metric_at(Mv, Ms, 16, blend=0.8)
    SC = dict(n_seq=a.held_seq, T=256)

    def score(R):
        return D.score_fold(L.X_to_FT(D, R), **SC)

    order = torch.argsort(cnt, descending=True)

    # ---- OBJECTION 1
    rows = []
    for B in (128, 512, 1024):
        for n, k_hi, k_lo in ((256, 8, 2), (256, 4, 2), (256, 16, 2)):
            R, bt = stratified_k(X, Msel, n, k_hi, k_lo, B, order, a.iters)
            s = score(R)
            rows.append({'family': 'stratified_k', 'B': B, 'n': n,
                         'k_hi': k_hi, 'k_lo': k_lo, 'fold_bits': bt.total,
                         'bill': bt.to_json(), **s})
            log(f'  stratified B={B} n={n} k {k_hi}/{k_lo} '
                f'{bt.total / 1e6:.2f} Mbit ce {s["ce"]:.4f}')
            json.dump({**out, 'stratified': rows}, open(path, 'w'), indent=1)
    out['stratified'] = rows

    # ---- OBJECTION 2
    rows2 = []
    I = torch.eye(hd, device=X.device)[None, None]
    for tag, Mm in (
            ('mse', None),
            ('exposure_scalar_only',
             (Msel.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)[..., None]
              / hd) * I),
            ('full_ctx', Msel)):
        for n, k in ((256, 2), (1024, 8)):
            Dic, idx, coef = L.dict_learn(X, Mm, n, k, iters=a.iters)
            s = score(L.sparse_recon(Dic, idx, coef))
            rows2.append({'family': 'metric_ablation', 'metric': tag, 'n': n,
                          'k': k, **s})
            log(f'  metric {tag} n={n} k={k} ce {s["ce"]:.4f}')
            json.dump({**out, 'metric_ablation': rows2}, open(path, 'w'),
                      indent=1)
    out['metric_ablation'] = rows2
    json.dump(out, open(path, 'w'), indent=1)
    log('done ->', path)


if __name__ == '__main__':
    main()
