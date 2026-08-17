"""The weights+S formula's first blind test: predict the leaders of unprofiled layers.

§23 validated the Λ-weighted output Gram — G_lam = Down[(LSLᵀ)∘(RSRᵀ)]Downᵀ, weights
plus the input second moment — but only on layers whose causal leaders were already
measured (0, 1, 16, 17), so it has never predicted anything it could get wrong. The
standalone report lists this as the program's most pointed open item.

Layers 7, 9, 11, 13 have never been profiled. Protocol, in this order, so the
prediction cannot be contaminated:

  STEP 1 (prediction, weights + one data matrix only): for each layer, compute G_lam
     and record its top-8 eigenvectors, and separately its #1 eigenvector, BEFORE any
     causal measurement exists. These are the registered predictions: the causal
     leader will lie in the top-8 span (energy >> the ~0.007 random baseline), and
     the #1 eigenvector will be its best single guess.

  STEP 2 (measurement): empirical top-32 output PCA basis, then a Shapley attribution
     over it (8 permutations — enough to identify the leader, not to fine-rank the
     tail) with mean-ablation values on held-out CE. The measured leader is the
     direction with the largest Shapley value.

  STEP 3 (score): energy of the measured leader in the predicted top-8; |cos| with
     the predicted #1; random-span baseline. Success criterion, registered now:
     leader energy in predicted top-8 above 0.5 at every layer (the validated layers
     scored 0.90-0.996; tail layers are smaller and noisier, so the bar is lower but
     still 70x random).

These are middle layers, where §29 found high-rank structure and §27 found small fair
shares -- the hostile regime for the formula, which is the point of a blind test.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, held, orth, m, FW, DEV, PATCH

LAYERS = (7, 9, 11, 13)
NDIR = 32
N_PERM = 8
D = 1152
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_blind_prediction_results.json')


@torch.no_grad()
def collect(li, n_seq=96, what='out'):
    ins, outs = [], []

    def hook(mod, inp, outp):
        if what in ('in', 'both'):
            ins.append(inp[0].detach().reshape(-1, D).float())
        if what in ('out', 'both'):
            outs.append(outp.detach().reshape(-1, D).float())

    h = m.transformer.h[li].mlp.register_forward_hook(hook)
    for i in range(0, n_seq, 6):
        b = FW[i:i + 6, :513].to(DEV)
        m(b[:, :-1].contiguous(), b[:, 1:].contiguous())
    h.remove()
    if what == 'both':
        return torch.cat(ins), torch.cat(outs)
    return torch.cat(outs if what == 'out' else ins)


def g_lam_top(li, S, k=8):
    mlp = m.transformer.h[li].mlp
    L = mlp.Left.weight.detach().float()
    R = mlp.Right.weight.detach().float()
    Dw = mlp.Down.weight.detach().float()
    LS, RS = L @ S, R @ S
    G = Dw @ ((LS @ L.T) * (RS @ R.T)) @ Dw.T
    ev, U = torch.linalg.eigh(G)
    idx = ev.argsort(descending=True)
    return orth(U[:, idx[:k]]), U[:, idx[0]]


def main():
    t0 = time.time()
    base = held()
    BASE = float(base.mean())
    out = {'base_ce': BASE, 'layers': {}}
    print(f'base CE {BASE:.4f}\n')

    # ===== STEP 1: all predictions first, and freeze them =====
    preds = {}
    for li in LAYERS:
        X = collect(li, what='in')
        S = X.T @ X / X.shape[0]
        preds[li] = g_lam_top(li, S)
        del X
    print(f'STEP 1 done: predictions frozen for layers {LAYERS} '
          f'(weights + input second moment only)\n')

    # ===== STEP 2/3: measure and score =====
    g = torch.Generator().manual_seed(0)
    for li in LAYERS:
        Y = collect(li, what='out')
        Ybar = Y.mean(0)
        _, _, Vh = torch.linalg.svd((Y - Ybar).float(), full_matrices=False)
        Q = orth(Vh[:NDIR].T)

        def value(cols):
            if not cols:
                return 0.0
            Qs = Q[:, cols]
            PATCH[li] = (Qs, Ybar @ Qs)
            try:
                return float((held() - base).mean())
            finally:
                PATCH.pop(li)

        v_all = value(list(range(NDIR)))
        phi = torch.zeros(NDIR, N_PERM, dtype=torch.float64)
        for p in range(N_PERM):
            perm = torch.randperm(NDIR, generator=g).tolist()
            prev = 0.0
            cur = []
            for pos, i in enumerate(perm):
                cur.append(i)
                v = v_all if pos == NDIR - 1 else value(cur)
                phi[i, p] = v - prev
                prev = v
        est = phi.mean(1)
        lead = int(est.argmax())
        d0 = Q[:, lead]
        W8, w1 = preds[li]
        e8 = float((W8.T @ d0).pow(2).sum())
        c1 = float((w1 @ d0).abs())
        gr = torch.Generator(device=DEV).manual_seed(li)
        Qr = orth(torch.randn(D, 8, device=DEV, generator=gr))
        er = float((Qr.T @ d0).pow(2).sum())
        hit = e8 > 0.5
        out['layers'][li] = {'v_all': v_all,
                             'leader': lead,
                             'leader_share': float(est[lead] / est.sum()),
                             'pr': float(est.sum() ** 2 / (est ** 2).sum()),
                             'leader_energy_pred8': e8,
                             'cos_pred1': c1, 'random8': er, 'hit': bool(hit)}
        print(f'layer {li:2d}: span effect {v_all:+.4f} | leader dir {lead} '
              f'({100*float(est[lead]/est.sum()):.0f}%, PR '
              f'{float(est.sum()**2/(est**2).sum()):.1f})')
        print(f'          predicted-top8 energy {e8:.3f} (random {er:.3f}) | '
              f'|cos| with predicted #1: {c1:.3f}  -> '
              f'{"HIT" if hit else "MISS"}', flush=True)

    hits = sum(r['hit'] for r in out['layers'].values())
    out['n_hits'] = hits
    es = [r['leader_energy_pred8'] for r in out['layers'].values()]
    print(f'\nBLIND SCORE: {hits}/{len(LAYERS)} layers above the registered 0.5 bar; '
          f'energies {[round(e, 2) for e in es]}')
    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
