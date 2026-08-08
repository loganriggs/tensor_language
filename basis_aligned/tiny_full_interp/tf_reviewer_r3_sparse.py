"""R7 addendum -- the strongest UNTRIED structural family.

The finding tested prototypes (one atom per token), low rank (a dense
subspace), and product quantisation (per-subspace atoms).  It never tested the
family this program actually has a prior on: **each token's row as a SPARSE
combination of a shared overcomplete dictionary** -- the feature hypothesis.
Asking "is class (b) empty?" without it would be unfair to the structural side.

The description is: a dictionary of m atoms, and for each token s (atom index,
coefficient) pairs.  Everything charged: atoms at b_dict bits with per-atom
fp16 scales, s indices at ceil(log2 m) bits per token, s coefficients at
b_coef bits with a per-token fp16 scale.

Fitted with alternating batched orthogonal matching pursuit and a least-squares
dictionary update.  Positive control: at s = 128 = d with m >= d the code is
exact, so the reconstruction error must fall to numerical zero.
"""
import json
import math
import os

import torch

import tf_compress as CC
import tf_compress_run as RR
from tf_compress import Bits, bits_dense, bits_index

HERE = os.path.dirname(os.path.abspath(__file__))
P = f'{HERE}/tf_reviewer_round_3_compression.json'


@torch.no_grad()
def omp(X, Dct, s):
    """Batched orthogonal matching pursuit: for every row of X pick s atoms of
    the (unit-norm) dictionary greedily and least-squares fit them."""
    n, d = X.shape
    m = Dct.shape[0]
    R = X.clone()
    idx = torch.zeros(n, s, dtype=torch.long, device=X.device)
    for t in range(s):
        c = (R @ Dct.t()).abs()
        if t:
            c.scatter_(1, idx[:, :t], -1.0)
        idx[:, t] = c.argmax(1)
        A = Dct[idx[:, :t + 1]]                       # (n, t+1, d)
        G = A @ A.transpose(1, 2)
        G = G + 1e-6 * torch.eye(t + 1, device=X.device)[None]
        rhs = torch.einsum('ntd,nd->nt', A, X)
        w = torch.linalg.solve(G, rhs.unsqueeze(-1)).squeeze(-1)
        R = X - torch.einsum('nt,ntd->nd', w, A)
    return idx, w


@torch.no_grad()
def learn_dict(X, m, s, iters=12, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    n, d = X.shape
    perm = torch.randperm(n, generator=g)[:m].to(X.device)
    Dct = X[perm].clone()
    Dct = Dct / Dct.norm(dim=1, keepdim=True).clamp_min(1e-9)
    for _ in range(iters):
        idx, w = omp(X, Dct, s)
        # least-squares dictionary update given the support
        C = torch.zeros(n, m, device=X.device)
        C.scatter_(1, idx, w)
        G = C.t() @ C + 1e-4 * torch.eye(m, device=X.device)
        Dct = torch.linalg.solve(G, C.t() @ X)
        nrm = Dct.norm(dim=1, keepdim=True)
        dead = (nrm.squeeze(1) < 1e-8)
        if dead.any():
            Dct[dead] = X[torch.randint(n, (int(dead.sum()),),
                                        generator=g).to(X.device)]
            nrm = Dct.norm(dim=1, keepdim=True)
        Dct = Dct / nrm.clamp_min(1e-9)
    idx, w = omp(X, Dct, s)
    return Dct, idx, w


def main():
    D = CC.D1Desc('tf_vanilla_d1_w128_b8192_s0')
    W = D.base['wte_out']
    V, d = W.shape
    mu = W.mean(0, keepdim=True)
    X = W - mu
    rows = []

    # ---- positive control: s = d atoms out of an m >= d dictionary must be
    # able to reproduce X exactly, so the reconstruction error must vanish.
    Dct, idx, w = learn_dict(X, 128, 128, iters=2)
    C = torch.zeros(V, 128, device=X.device).scatter_(1, idx, w)
    ctrl = float((X - C @ Dct).abs().max())
    print('positive control (s = d = 128, full-rank dictionary): max abs err',
          ctrl, flush=True)

    for m in (512, 1024, 2048):
        for s in (4, 8, 16, 32):
            Dct, idx, w = learn_dict(X, m, s)
            for bd, bc in ((8, 6), (6, 5)):
                Dq, bdict = CC.q_scalar_entropy(Dct, bd)
                lo = w.min(1, keepdim=True).values.half().float()
                hi = w.max(1, keepdim=True).values.half().float()
                step = ((hi - lo) / (2 ** bc - 1)).clamp_min(1e-30)
                wq = (((w - lo) / step).round().clamp(0, 2 ** bc - 1))
                bits_coef = CC.entropy_bits(wq, 2 ** bc)
                wr = wq * step + lo
                C = torch.zeros(V, m, device=X.device).scatter_(1, idx, wr)
                Wc = C @ Dq + mu
                bits = Bits(indices=bits_index(V * s, m),
                            coefficients=bits_coef,
                            coef_scales=2 * V * 16,
                            mean=bits_dense(d, 32)).merge(bdict, 'dict_')
                sc = D.score({'wte_read': Wc, 'wte_out': Wc})
                rows.append({'scheme': f'sparsedict_m{m}_s{s}_d{bd}_c{bc}',
                             'm': m, 's': s, 'bits': bits.total,
                             'bill': bits.to_json(), **sc})
                print(f'   m={m} s={s} dict{bd}/coef{bc}: '
                      f'{bits.total/1e6:6.3f} Mbit  KL {sc["kl"]:.5f}  '
                      f'CE {sc["ce"]:.5f}', flush=True)

    # matched-bits comparison against the best pure recoding of the same table
    base = []
    for bpr in (256, 384, 512, 640, 768):
        Wt, bt = CC.q_transform(W, bpr, rot='none')
        s2 = D.score({'wte_read': Wt, 'wte_out': Wt})
        base.append({'scheme': f'transform_{bpr}', 'bits': bt.total, **s2})
    for b in (2, 3, 4, 5, 6):
        Wq, bt = CC.q_scalar_entropy(W, b)
        s2 = D.score({'wte_read': Wq, 'wte_out': Wq})
        base.append({'scheme': f'scalar_q{b}e', 'bits': bt.total, **s2})

    def kl_at(bits):
        c = sorted([(r['bits'], r['kl']) for r in base])
        for i in range(len(c) - 1):
            if c[i][0] <= bits <= c[i + 1][0]:
                (b0, k0), (b1, k1) = c[i], c[i + 1]
                t = (math.log(bits) - math.log(b0)) / (math.log(b1) - math.log(b0))
                return math.exp(math.log(k0) + t * (math.log(k1) - math.log(k0)))
        return None
    for r in rows:
        r['best_recoding_kl_at_same_bits'] = kl_at(r['bits'])
        r['kl_penalty_x'] = (r['kl'] / r['best_recoding_kl_at_same_bits']
                             if r['best_recoding_kl_at_same_bits'] else None)
    ok = [r for r in rows if r['kl_penalty_x']]
    best = min(ok, key=lambda q: q['kl_penalty_x']) if ok else None
    o = json.load(open(P))
    o['R7b_sparse_dictionary_the_untried_structural_family'] = {
        'positive_control_max_abs_err_at_s_equals_d': ctrl,
        'rows': rows, 'recoding_baselines': base,
        'best_penalty_vs_recoding': best,
        'note': ('Each token\'s embedding row as a sparse combination of a '
                 'shared overcomplete dictionary -- the feature hypothesis, '
                 'and the one structural family the finding never tried. '
                 'Charged: atoms, per-token atom indices, per-token '
                 'coefficients with an fp16 scale, and the column mean.')}
    json.dump(o, open(P, 'w'), indent=1)
    if best:
        print('BEST sparse-dictionary point vs recoding at matched bits: '
              f'{best["scheme"]} {best["bits"]/1e6:.3f} Mbit KL {best["kl"]:.5f}'
              f' vs {best["best_recoding_kl_at_same_bits"]:.5f} -> '
              f'{best["kl_penalty_x"]:.2f}x')


if __name__ == '__main__':
    main()
