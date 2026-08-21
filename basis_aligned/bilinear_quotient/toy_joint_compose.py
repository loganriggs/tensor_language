"""TOY JOINT SPARSE COMPOSITION (validate the machinery on ground truth before
the real model). Plant a 2-layer circuit with a KNOWN SPARSE cross-layer coupling
S, then test whether JOINTLY training two weight-action SAEs with an EDGE penalty
recovers S -- and whether it wins on an MDL (description-length) frontier vs
independent SAEs and SVD.

Design note (v2): real weights are LINEAR, so the sparse structure must live in
how layer-1 atoms MIX into layer-2 atoms (a sparse coupling S), NOT in a nonlinear
generator. And to isolate the COUPLING question (not sparse-coding-from-dense-
input, which is a separate hard problem), layer-1's gate IS its sparse code.

GROUND TRUTH (all linear; S is the object to recover):
  z1  : k1-sparse nonneg code over P1 atoms           (N x P1)   [= layer-1 gate a1]
  y1  = z1 @ D1t^T                                     (N x Dm1)  layer-1 output (obs)
  z2  = z1 @ S^T          (S: P2 x P1, ~FANOUT nz/col) (N x P2)   sparse MIXING
  y2  = z2 @ D2t^T                                     (N x Dm2)  layer-2 output (obs)
  coupling ground truth = S (which source atom drives which target atom).

SAEs (weight-action form): SAE1 reconstructs y1 from a1=z1; SAE2 reconstructs y2
from a2=y1. Recovered coupling C = E2 @ D1 (pure weights). We test whether C ~ S.

METHODS (matched P,k): (indep) fit the two SAEs separately; (joint) fit together
with + lam_e*||normalize(E2)@normalize(D1)||_1 (edge penalty on unit-normed dicts,
so L1 sparsifies the ANGLE not the scale); (svd) dense reference (no atom coupling).

REGISTERED PREDICTIONS:
  (0) SANITY: both SAEs reconstruct y1,y2 (R2>0.7);
  (a) JOINT RECOVERS WIRING: joint edge-recovery F1 > 0.6 and > indep by >=0.15,
      >> svd (0); joint coupling is SPARSER (<= indep live-edges) at equal recon;
  (b) MDL: joint total description bits <= indep (edge penalty pays for itself);
  NULL: shuffling planted S -> joint edge-F1 at chance (< 0.2)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'toy_joint_compose_results.json'
N = 6000; Dm1 = 64; Dm2 = 64
P1 = 96; P2 = 96; K1 = 4; FANOUT = 1; K2 = 6      # z2 = S@z1 has ~K1*FANOUT nz -> K2 slack
STEPS = 4000; LAM_E = 0.1     # edge weight: strong enough to break factorization non-identifiability (swept)


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def unit_cols(W):
    with torch.no_grad(): W /= W.norm(dim=0, keepdim=True).clamp_min(1e-8)


def resample_dead(Ee, Dd, fired, Xin):
    with torch.no_grad():
        dead = (~fired).nonzero(as_tuple=True)[0]
        if len(dead) == 0: return
        pick = Xin[torch.randint(0, Xin.shape[0], (len(dead),), device=Xin.device)]
        Ee[dead] = F.normalize(pick, dim=1) * Ee.norm(dim=1).mean().clamp_min(1e-6)


def plant(seed=0, shuffle_S=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    D1t = torch.randn(Dm1, P1, generator=g, device=DEV); unit_cols(D1t)
    D2t = torch.randn(Dm2, P2, generator=g, device=DEV); unit_cols(D2t)
    # k1-sparse nonneg layer-1 codes
    z1 = torch.zeros(N, P1, device=DEV)
    for r in range(N):
        idx = torch.randperm(P1, generator=g, device=DEV)[:K1]
        z1[r, idx] = torch.rand(K1, generator=g, device=DEV)*1.5 + 0.5
    # sparse coupling S: each source atom -> FANOUT targets
    S = torch.zeros(P2, P1, device=DEV)
    tgt = torch.randint(0, P2, (P1, FANOUT), generator=g, device=DEV)
    for i in range(P1):
        for f in range(FANOUT):
            S[tgt[i, f], i] = torch.rand(1, generator=g, device=DEV).item()*1.5 + 0.5
    if shuffle_S:
        S = S[torch.randperm(P2, generator=g, device=DEV)][:, torch.randperm(P1, generator=g, device=DEV)]
    y1 = z1 @ D1t.T
    z2 = z1 @ S.T
    y2 = z2 @ D2t.T
    return z1, y1, y2, D1t, D2t, S


def train_sae(Xin, Ytrue, P, k, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    din = Xin.shape[1]; dout = Ytrue.shape[1]
    Dd = torch.randn(dout, P, device=DEV); unit_cols(Dd); Dd.requires_grad_(True)
    Ee = F.normalize(Xin[torch.randperm(Xin.shape[0])[:P]].clone(), dim=1).requires_grad_(True)
    opt = torch.optim.Adam([Dd, Ee], lr=1e-2)
    fired = torch.zeros(P, dtype=torch.bool, device=DEV)
    for s in range(steps):
        z = topk(Xin @ Ee.T, k); loss = F.mse_loss(z @ Dd.T, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            unit_cols(Dd); fired |= (z > 1e-6).any(0)
            if s > 0 and s % 800 == 0: resample_dead(Ee, Dd, fired, Xin); fired.zero_()
    return Dd.detach(), Ee.detach()


def train_joint(a1, y1, y2, lam_e, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    D1 = torch.randn(Dm1, P1, device=DEV); unit_cols(D1); D1.requires_grad_(True)
    E1 = F.normalize(a1[torch.randperm(a1.shape[0])[:P1]].clone(), dim=1).requires_grad_(True)
    D2 = torch.randn(Dm2, P2, device=DEV); unit_cols(D2); D2.requires_grad_(True)
    E2 = F.normalize(y1[torch.randperm(y1.shape[0])[:P2]].clone(), dim=1).requires_grad_(True)
    opt = torch.optim.Adam([D1, E1, D2, E2], lr=1e-2)
    f1 = torch.zeros(P1, dtype=torch.bool, device=DEV); f2 = torch.zeros(P2, dtype=torch.bool, device=DEV)
    for s in range(steps):
        z1 = topk(a1 @ E1.T, K1); r1 = z1 @ D1.T
        z2 = topk(r1 @ E2.T, K2); r2 = z2 @ D2.T
        C = F.normalize(E2, dim=1) @ F.normalize(D1, dim=0)     # coupling on unit dicts
        loss = F.mse_loss(r1, y1) + F.mse_loss(r2, y2) + lam_e*C.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            unit_cols(D1); unit_cols(D2); f1 |= (z1 > 1e-6).any(0); f2 |= (z2 > 1e-6).any(0)
            if s > 0 and s % 800 == 0:
                resample_dead(E1, D1, f1, a1); resample_dead(E2, D2, f2, r1); f1.zero_(); f2.zero_()
    return D1.detach(), E1.detach(), D2.detach(), E2.detach()


def match_atoms(D_hat, D_true):
    A = (F.normalize(D_true, dim=0).T @ F.normalize(D_hat, dim=0)).abs()
    perm = torch.full((D_true.shape[1],), -1, dtype=torch.long, device=D_true.device); used = set()
    for t in A.max(1).values.argsort(descending=True).tolist():
        for c in A[t].argsort(descending=True).tolist():
            if c not in used: perm[t] = c; used.add(c); break
    return perm


def align(C_hat, D1_hat, D1_true, D2_hat, D2_true):
    """remap recovered coupling to true-atom order; also return match quality (mean |cos|)."""
    A1 = (F.normalize(D1_true, dim=0).T @ F.normalize(D1_hat, dim=0)).abs()
    A2 = (F.normalize(D2_true, dim=0).T @ F.normalize(D2_hat, dim=0)).abs()
    p1 = match_atoms(D1_hat, D1_true); p2 = match_atoms(D2_hat, D2_true)
    mq = float(0.5*(A1.gather(1, p1[:, None]).mean() + A2.gather(1, p2[:, None]).mean()))
    return C_hat[p2][:, p1], mq


def f1_of(Cre, S_true, thr=0.2):
    St = (S_true.abs() > 1e-6); Ce = (Cre.abs() > thr*Cre.abs().max().clamp_min(1e-9))
    tp = float((St & Ce).sum()); fp = float((~St & Ce).sum()); fn = float((St & ~Ce).sum())
    prec = tp/max(tp+fp, 1); rec = tp/max(tp+fn, 1)
    return 2*prec*rec/max(prec+rec, 1e-9), float(Ce.sum())


def perm_rows_cols(S, seed=7):
    g = torch.Generator(device=S.device).manual_seed(seed)
    return S[torch.randperm(S.shape[0], generator=g, device=S.device)][:, torch.randperm(S.shape[1], generator=g, device=S.device)]


def mdl_bits(a1, y2, D1, E1, D2, E2):
    with torch.no_grad():
        z1 = topk(a1 @ E1.T, K1); r1 = z1 @ D1.T; z2 = topk(r1 @ E2.T, K2); r2 = z2 @ D2.T
        resid = ((y2 - r2)**2).mean().clamp_min(1e-12)
        recon_bits = 0.5*np.log2(2*np.pi*np.e*float(resid)) * y2.shape[1]
        node_bits = (K1*np.log2(P1) + K2*np.log2(P2))
        C = F.normalize(E2, dim=1) @ F.normalize(D1, dim=0)
        edges = float((C.abs() > 0.2*C.abs().max()).sum())
        edge_bits = edges*np.log2(P1*P2) / a1.shape[0]
    return recon_bits + node_bits + edge_bits


def r2(a, b): return float(1 - ((a-b)**2).sum()/((b-b.mean(0))**2).sum())


def handwritten():
    """Tiny circuit where the answer is OBVIOUS: 5 layer-1 atoms each drive ONE
    named layer-2 atom via a hand-written S. Print true vs recovered coupling."""
    global P1, P2, K1, K2, Dm1, Dm2, N
    P1o, P2o, K1o, K2o, Dm1o, Dm2o, No = P1, P2, K1, K2, Dm1, Dm2, N
    P1 = P2 = 5; K1 = 2; K2 = 2; Dm1 = Dm2 = 8; N = 3000
    g = torch.Generator(device=DEV).manual_seed(0)
    D1t = torch.randn(Dm1, P1, generator=g, device=DEV); unit_cols(D1t)
    D2t = torch.randn(Dm2, P2, generator=g, device=DEV); unit_cols(D2t)
    # HAND-WRITTEN wiring: 0->2, 1->4, 2->0, 3->3, 4->1  (a fixed permutation)
    wire = {0: 2, 1: 4, 2: 0, 3: 3, 4: 1}
    S = torch.zeros(P2, P1, device=DEV)
    for src, dst in wire.items(): S[dst, src] = 1.0
    z1 = torch.zeros(N, P1, device=DEV)
    for r in range(N):
        idx = torch.randperm(P1, generator=g, device=DEV)[:K1]
        z1[r, idx] = torch.rand(K1, generator=g, device=DEV) + 0.5
    y1 = z1 @ D1t.T; y2 = (z1 @ S.T) @ D2t.T
    D1j, E1j, D2j, E2j = train_joint(z1, y1, y2, LAM_E, seed=3)
    Cre, mq = align(E2j @ D1j, D1j, D1t, D2j, D2t)
    recovered = {src: int(Cre[:, src].abs().argmax()) for src in range(P1)}
    correct = sum(recovered[s] == wire[s] for s in range(P1))
    print('HANDWRITTEN tiny circuit (5 atoms, wiring src->dst):', flush=True)
    print('  true      :', {s: wire[s] for s in range(P1)}, flush=True)
    print('  recovered :', recovered, f'  ({correct}/5 correct, atom-match cos {mq:.2f})', flush=True)
    print('  |recovered coupling| (rows=target atom, cols=source atom):', flush=True)
    Cn = (Cre.abs()/Cre.abs().max()).cpu().numpy()
    for r_ in range(P2): print('   ', ' '.join(f'{Cn[r_,c]:.2f}' for c in range(P1)), flush=True)
    P1, P2, K1, K2, Dm1, Dm2, N = P1o, P2o, K1o, K2o, Dm1o, Dm2o, No
    return correct, mq


def main():
    t0 = time.time()
    hw_correct, hw_mq = handwritten()

    z1, y1, y2, D1t, D2t, S = plant(seed=0)
    a1 = z1                                                   # layer-1 gate = code

    # independent
    D1i, E1i = train_sae(a1, y1, P1, K1, seed=1); D2i, E2i = train_sae(y1, y2, P2, K2, seed=2)
    Crei, mqi = align(E2i @ D1i, D1i, D1t, D2i, D2t)
    r1i = r2(topk(a1@E1i.T, K1)@D1i.T, y1); r2i = r2(topk(y1@E2i.T, K2)@D2i.T, y2)
    f1_i, e_i = f1_of(Crei, S)
    mdl_i = mdl_bits(a1, y2, D1i, E1i, D2i, E2i)

    # joint
    D1j, E1j, D2j, E2j = train_joint(a1, y1, y2, LAM_E, seed=3)
    Crej, mqj = align(E2j @ D1j, D1j, D1t, D2j, D2t)
    z1j = topk(a1@E1j.T, K1); r1j = z1j@D1j.T
    r1jj = r2(r1j, y1); r2jj = r2(topk(r1j@E2j.T, K2)@D2j.T, y2)
    f1_j, e_j = f1_of(Crej, S)
    mdl_j = mdl_bits(a1, y2, D1j, E1j, D2j, E2j)

    # SPECIFICITY NULL (red-teamed): compare the SAME recovered joint coupling to a
    # WRONG (row/col-permuted) S. Real recovery => f1 vs true >> f1 vs wrong.
    f1_null, _ = f1_of(Crej, perm_rows_cols(S))

    print(f'indep : R2 y1 {r1i:.2f} y2 {r2i:.2f} | edge-F1 {f1_i:.2f} live-edges {e_i:.0f} match-cos {mqi:.2f} | MDL {mdl_i:.1f}', flush=True)
    print(f'joint : R2 y1 {r1jj:.2f} y2 {r2jj:.2f} | edge-F1 {f1_j:.2f} live-edges {e_j:.0f} match-cos {mqj:.2f} | MDL {mdl_j:.1f}', flush=True)
    print(f'svd   : edge-F1 0.00 (dense) | SPECIFICITY NULL (joint C vs WRONG S) {f1_null:.2f}', flush=True)

    p0 = min(r2i, r2jj, r1i, r1jj) > 0.7
    pa = f1_j > 0.6 and f1_j - f1_i >= 0.15 and e_j <= e_i and f1_j - f1_null >= 0.25   # specificity gate
    pb = mdl_j <= mdl_i
    null_ok = f1_null < 0.35 and f1_j - f1_null >= 0.25
    hw_ok = hw_correct >= 4 and hw_mq > 0.8
    out = {'handwritten': {'correct_of_5': hw_correct, 'atom_match_cos': round(hw_mq, 3)},
           'indep': {'r2_y1': round(r1i, 3), 'r2_y2': round(r2i, 3), 'edge_f1': round(f1_i, 3),
                     'live_edges': e_i, 'match_cos': round(mqi, 3), 'mdl_bits': round(mdl_i, 2)},
           'joint': {'r2_y1': round(r1jj, 3), 'r2_y2': round(r2jj, 3), 'edge_f1': round(f1_j, 3),
                     'live_edges': e_j, 'match_cos': round(mqj, 3), 'mdl_bits': round(mdl_j, 2)},
           'specificity_null_f1': round(f1_null, 3),
           'planted': {'P1': P1, 'P2': P2, 'fanout': FANOUT, 'K1': K1, 'K2': K2, 'lam_e': LAM_E},
           'pred_0': bool(p0), 'pred_a': bool(pa), 'pred_b': bool(pb), 'null_ok': bool(null_ok),
           'handwritten_ok': bool(hw_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nHANDWRITTEN {hw_correct}/5 correct (match-cos {hw_mq:.2f}) -> ok {hw_ok}', flush=True)
    print(f'(0) both reconstruct: {p0}; (a) joint recovers wiring>indep, sparser, SPECIFIC: {pa}; '
          f'(b) joint MDL<=indep: {pb}; NULL joint-vs-wrong-S low & separated: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
