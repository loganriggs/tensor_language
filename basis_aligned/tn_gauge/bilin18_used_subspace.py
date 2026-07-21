"""Data-driven USED-subspace QK compression: does it generalize F22/F23? (Logan 2026-07-21).
F22: at layer 1, projecting QK onto the bilinear-output subspace beats generic low-rank ~6x.
F23: that single-source projection is layer-1-specific (deep selection distributed). This tests
the GENERAL version: the activation-aware optimal rank-r INPUT projection that preserves the
query/key reads over data -- min_P E||R(I-P)x||^2, R=[Wq;Wk;Wq2;Wk2]. Whitened solution: with
C=Cov(x), the used subspace = top-r eigenvectors of C^{-1/2} R^T R C^{-1/2}, giving oblique
projection P = C^{-1/2} Wz Wz^T C^{1/2}; W'=W P factors as (D x r)(r x D) shared basis = 5rD bits,
same budget as F22. If used-subspace beats generic (weight-only) low-rank at ALL depths -> a
general activation-aware compression method. Compare used vs generic-low-rank vs source-subspace
(F23), r=128, across layers. ΔCE via reference_forward.
"""
import json, sys
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL = len(m.transformer.h)
Vsz = cfg['vocab_size']
AUD = build_eval_tokens(n_chunks=10, seq_len=513)[:8]
IDX = AUD[:, :-1].to(DEV); TGT = AUD[:, 1:].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
Rr = 128
TARGETS = [1, 3, 6, 9, 12]


@torch.no_grad()
def capture():
    """per layer: QK input activations rms(xin_L); plus each block's mlp output (for source ctrl)."""
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
    qkin, mlps = {}, {}
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn; h = rms(xin)
        qkin[li] = h.reshape(-1, D)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        mo = blk.mlp(rms(x)); x = x + mo
        mlps[li] = mo.reshape(-1, D)
    return qkin, mlps


@torch.no_grad()
def ce():
    return F.cross_entropy(reference_forward(m, IDX, 'bf16').float().reshape(-1, Vsz), TGT.reshape(-1)).item()


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double())
    ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T


def used_subspace_P(X, R, r):
    """optimal rank-r input projection preserving reads R x over data: max read-variance
    captured. W = top-r eigvecs of C^{1/2} G C^{1/2}; P = C^{1/2} W W^T C^{-1/2}."""
    C = (X.double().T @ X.double()) / X.shape[0]
    Cis = sym_pow(C, -0.5); Cs = sym_pow(C, 0.5)
    G = (R.double().T @ R.double())
    Mm = Cs @ G @ Cs
    ev, U = torch.linalg.eigh(Mm)
    Wz = U[:, -r:]                                        # top-r read-variance directions
    P = Cs @ Wz @ Wz.T @ Cis
    return P.float()


def lowrank(W, r):
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vh[:r]).float()


def pca_P(X, r):
    _, _, Vh = torch.linalg.svd(X.double() - X.double().mean(0), full_matrices=False)
    U = Vh[:r].float(); return U.T @ U


QKIN, MLPS = capture()
CE0 = ce()
res = {'baseline_ce': round(CE0, 4), 'r': Rr, 'by_layer': {}}
print(f'baseline CE {CE0:.4f}; r={Rr}. USED-subspace vs generic-lowrank vs source(F23) vs input-PCA:', flush=True)
print('  L | USED   | lowrank | source | input-PCA', flush=True)
for L in TARGETS:
    A = m.transformer.h[L].attn
    W0 = {n: getattr(A, n).weight.data.clone() for n in NAMES}
    R = torch.cat([W0[n].float() for n in NAMES], 0)
    Pused = used_subspace_P(QKIN[L], R, Rr)
    Psrc = pca_P(MLPS[L - 1], Rr)

    def apply_and_ce(transform):
        for n in NAMES:
            getattr(A, n).weight.data.copy_(transform(W0[n].float()).to(W0[n].dtype))
        d = ce() - CE0
        for n in NAMES:
            getattr(A, n).weight.data.copy_(W0[n])
        return d
    d_used = apply_and_ce(lambda W: W @ Pused)
    d_lr = apply_and_ce(lambda W: lowrank(W, Rr))
    d_src = apply_and_ce(lambda W: W @ Psrc)
    d_ipca = apply_and_ce(lambda W: W @ pca_P(QKIN[L], Rr))
    res['by_layer'][L] = {'used': round(d_used, 4), 'lowrank': round(d_lr, 4),
                          'source': round(d_src, 4), 'input_pca': round(d_ipca, 4)}
    print(f'  {L:2d} | {d_used:+.4f}| {d_lr:+.4f} | {d_src:+.4f}| {d_ipca:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_used_subspace.json', 'w'), indent=2)
wins = sum(1 for L in TARGETS if res['by_layer'][L]['used'] <= res['by_layer'][L]['lowrank'] + 1e-4)
res['used_beats_lowrank_at'] = f'{wins}/{len(TARGETS)}'
json.dump(res, open(f'{OUT}/bilin18_used_subspace.json', 'w'), indent=2)
print(f'\nUSED-subspace beats/ties generic low-rank at {wins}/{len(TARGETS)} layers (r={Rr}) '
      f'-> {"GENERAL METHOD" if wins >= len(TARGETS)-1 else "not general"}', flush=True)
print('bilin18 used subspace done', flush=True)
