"""Equivalence classes for layer-1 QK, INSIDE the used-subspace (Logan 2026-07-21).
The used-subspace (F24/F25) removes input directions QK doesn't read -- a continuous linear
reduction. Logan's question: is there FURTHER compression of the 'some inputs are the same for
QK' kind (all colors attend to each other)? Test: cluster the layer-1 QK input INSIDE the
used-subspace into K classes and measure how few equivalence classes attention needs.
  used-subspace r=128 -> per-position code z (r-dim) -> KMeans K classes -> replace each
  position's z with its centroid -> feed QK -> ΔCE. Low K near-free => few equivalence classes.
Held-out: fit subspace+centroids on TRAIN, assign+measure on disjoint TEST. Continuous
used-subspace (no VQ) is the floor. This is rank-then-VQ (QCR-2 shape) scoped to the QK-relevant
basis of the bilinear output. ΔCE binding; layer 1 (single-source) as the clean case.
"""
import json, sys, math
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
Vsz = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=24, seq_len=513)
TRAIN, TEST = ALL[:12], ALL[12:24]
rms = lambda x: F.rms_norm(x, (D,))
L = 1
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
R = 128


@torch.no_grad()
def forward(tok, patch=None):
    """inline forward; patch = (layer, tensor) replaces that layer's QK input h_qk. returns logits."""
    idx = tok[:, :-1].to(DEV); B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(idx)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn
        h_qk = rms(xin)
        if patch is not None and li == patch[0]:
            h_qk = patch[1]
        h_v = rms(xin)
        qkf = lambda lin, hh: apply_rot(F.rms_norm(lin(hh).view(B, T, NH, HD), (HD,)), cosr, sinr)
        q, k = qkf(a.c_q, h_qk), qkf(a.c_k, h_qk)
        q2, k2 = qkf(a.c_q2, h_qk), qkf(a.c_k2, h_qk)
        v = a.c_v(h_v).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', q, k) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(B, T, D))
        x = x + blk.mlp(rms(x))
    return 30 * torch.tanh(m.lm_head(rms(x)) / 30)


@torch.no_grad()
def ce(tok, patch=None):
    tgt = tok[:, 1:].to(DEV)
    return F.cross_entropy(forward(tok, patch).float().reshape(-1, Vsz), tgt.reshape(-1)).item()


@torch.no_grad()
def qk_input(tok):
    idx = tok[:, :-1].to(DEV); B, T = idx.shape
    x = rms(m.transformer.wte(idx)); x0 = x; v1 = None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x
        if li == L:
            return rms(xin), (B, T)
        a = blk.attn; h = rms(xin)
        qkf = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qkf(a.c_q), qkf(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qkf(a.c_q2), qkf(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(B, T, D))
        x = x + blk.mlp(rms(x))


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double()); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T


def kmeans(X, K, iters=15, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        a = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 8192):
            xx = X[i:i + 8192]
            a[i:i + 8192] = ((xx * xx).sum(1, True) - 2 * xx @ C.T + (C * C).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C); cnt = torch.zeros(K, device=X.device)
        Cn.index_add_(0, a, X); cnt.index_add_(0, a, torch.ones(len(X), device=X.device))
        nz = cnt > 0; C[nz] = Cn[nz] / cnt[nz][:, None]
    return C


def assign(X, C):
    a = torch.empty(len(X), dtype=torch.long, device=X.device)
    for i in range(0, len(X), 8192):
        xx = X[i:i + 8192]
        a[i:i + 8192] = ((xx * xx).sum(1, True) - 2 * xx @ C.T + (C * C).sum(1)[None]).argmin(1)
    return a


# fit used-subspace on TRAIN
Htr, _ = qk_input(TRAIN)
Htr = Htr.reshape(-1, D)
Wq = torch.cat([getattr(m.transformer.h[L].attn, n).weight.data.float() for n in NAMES], 0)
C = (Htr.double().T @ Htr.double()) / Htr.shape[0]
Cs = sym_pow(C, 0.5); Cis = sym_pow(C, -0.5)
ev, U = torch.linalg.eigh(Cs @ (Wq.double().T @ Wq.double()) @ Cs)
Wz = U[:, -R:].float()                                    # (D, r) whitened read-var directions
ENC = (Cis.float() @ Wz)                                  # (D, r): code z = h @ ENC
DEC = (Cs.float() @ Wz).T                                 # (r, D): reconstruct h_used = z @ DEC
ztr = Htr @ ENC                                           # TRAIN codes (used-subspace coords)

CE0 = ce(TEST)
# continuous used-subspace floor on TEST (patch h_qk = reconstruction, no VQ)
Hte, (Bte, Tte) = qk_input(TEST)
Hte_flat = Hte.reshape(-1, D)
zte = Hte_flat @ ENC
h_used = (zte @ DEC).reshape(Bte, Tte, D)
CE_used = ce(TEST, patch=(L, h_used))
res = {'baseline_ce_test': round(CE0, 4), 'r': R,
       'used_subspace_floor_dce': round(CE_used - CE0, 4), 'vq': {}}
print(f'baseline CE(TEST) {CE0:.4f}; continuous used-subspace floor ΔCE {CE_used-CE0:+.4f}', flush=True)
print('VQ inside used-subspace (K equivalence classes), held-out ΔCE:', flush=True)
for K in [16, 64, 256, 1024, 4096]:
    Cent = kmeans(ztr, K, seed=0)                         # centroids in code space, fit on TRAIN
    a_te = assign(zte, Cent)                              # assign TEST positions
    h_vq = (Cent[a_te] @ DEC).reshape(Bte, Tte, D)        # reconstruct from centroid
    d = ce(TEST, patch=(L, h_vq)) - CE0
    res['vq'][K] = round(d, 4)
    print(f'  K={K:5d} classes: ΔCE {d:+.4f}  (log2K={math.log2(K):.0f} bits/token)', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_qk1_vq.json', 'w'), indent=2)
print('\ninterpretation: ΔCE at small K = how few equivalence classes layer-1 attention needs', flush=True)
print('bilin18 qk1 vq done', flush=True)
