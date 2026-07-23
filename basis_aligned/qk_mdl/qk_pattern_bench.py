"""BENCHMARK (Logan): what does training directly on the full V x V pattern table
P = (Q1 K1^T) o (Q2 K2^T) actually cost? Real numbers, one head, fp32.

  1. Materialization check: bytes for the full table (never allocated — arithmetic only).
  2. Full-table FORWARD pass, chunked: compute P over all V^2 pairs + a Frobenius reduction.
  3. One NAIVE full-table training step: dictionary recon -> chunked P-hat vs P weighted MSE
     -> backward with gradient accumulation. Extrapolate to 1500 steps x 9 heads.
  4. The structural shortcut: P has EXACT rank <= 128^2 = 16384 via the row-wise Kronecker
     (Khatri-Rao) factors  P = A B^T,  A_t = q1_t (x) q2_t,  B_t = k1_t (x) k2_t.
     So ||P_hat - P||_F^2 (optionally with separable row/col frequency weights, absorbed as
     sqrt(q) row scaling) reduces to traces of 16384 x 16384 Gram products — NO V x V object.
     Time one Gram and the full 6-Gram exact loss.
  5. Current sampled (M=1024) ctx-style step for comparison.
"""
import sys
import time
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs
from tier2_folding import branch_factors
from qk_sae_lib import train_dict, encode_token

torch.manual_seed(0)
DEV = 'cuda'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
h = 0
q1, k1 = branch_factors(m, 1)
q2, k2 = branch_factors(m, 2)
Q1, K1 = q1[:, h].float().to(DEV), k1[:, h].float().to(DEV)
Q2, K2 = q2[:, h].float().to(DEV), k2[:, h].float().to(DEV)
del q1, k1, q2, k2


def sync():
    torch.cuda.synchronize()


print(f'--- 1. materialization: full P is V^2 = {V*V:,} floats = {V*V*4/1e9:.1f} GB fp32 '
      f'per head ({V*V*4*9/1e9:.0f} GB all heads); S1+S2+P resident would be '
      f'{3*V*V*4/1e9:.0f} GB -> streaming mandatory on 16 GB', flush=True)

# --- 2. full-table forward pass, chunked ---
CH = 4096
sync(); t0 = time.time()
fro = 0.0
with torch.no_grad():
    for c0 in range(0, V, CH):
        sl = slice(c0, min(c0 + CH, V))
        P = (Q1[sl] @ K1.T / HD) * (Q2[sl] @ K2.T / HD)
        fro += float(P.pow(2).sum())
        del P
sync()
t_fwd = time.time() - t0
print(f'--- 2. full-table forward (1 head, chunk {CH}): {t_fwd:.1f} s '
      f'(x9 heads = {9*t_fwd:.0f} s); ||P||_F^2 = {fro:.3e}', flush=True)

# --- 3. one naive full-table training step ---
n, k = 256, 4
X = torch.cat([Q1, K1], 1)
Dn0, b0, We0 = train_dict(X, n, k, steps=200, seed=0)      # quick fit, timing only
Dn2, b2, We2 = train_dict(torch.cat([Q2, K2], 1), n, k, steps=200, seed=0)
QF = (torch.rand(V, device=DEV) + 0.1)
QF = QF / QF.sum()                                          # stand-in frequency weights
params = []
fits = []
for (Dn_, b_, We_) in ((Dn0, b0, We0), (Dn2, b2, We2)):
    t = [Dn_.clone().requires_grad_(True), b_.clone().requires_grad_(True),
         We_.clone().requires_grad_(True)]
    fits.append(t)
    params += t


def recon(Xin, Dm, b, We):
    Dnn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    z = (Xin - b) @ We.T
    vals, idx = z.abs().topk(k, dim=1)
    coeff = torch.gather(z, 1, idx)
    return b + (coeff.unsqueeze(-1) * Dnn[idx]).sum(1)


def unit_rms(t):
    return t * (t.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt())


sync(); t0 = time.time()
r1 = recon(torch.cat([Q1, K1], 1), *fits[0])
r2 = recon(torch.cat([Q2, K2], 1), *fits[1])
q1h, k1h = unit_rms(r1[:, :HD]), unit_rms(r1[:, HD:])
q2h, k2h = unit_rms(r2[:, :HD]), unit_rms(r2[:, HD:])
CHB = 2048
nch = 0
for c0 in range(0, V, CHB):
    sl = slice(c0, min(c0 + CHB, V))
    with torch.no_grad():
        P = (Q1[sl] @ K1.T / HD) * (Q2[sl] @ K2.T / HD)
    Ph = (q1h[sl] @ k1h.T / HD) * (q2h[sl] @ k2h.T / HD)
    loss = ((Ph - P).pow(2) * (QF[sl][:, None] * QF[None, :])).sum()
    loss.backward(retain_graph=True)                        # accumulate into dict grads
    del P, Ph, loss
    nch += 1
sync()
t_step = time.time() - t0
total_naive = t_step * 1500 * 9
print(f'--- 3. ONE naive full-table training step (fwd+bwd, chunk {CHB}, {nch} chunks): '
      f'{t_step:.1f} s -> 1500 steps x 9 heads = {total_naive/3600:.1f} h', flush=True)
for p in params:
    p.grad = None

# --- 4. Khatri-Rao Gram exact Frobenius (no V x V) ---
d2 = HD * HD
CHG = 2048


def gram(Aq, Ak, Bq=None, Bk=None):
    """G = sum_t (Aq_t (x) Ak_t)(Bq_t (x) Bk_t)^T accumulated in chunks -> (d2, d2)."""
    Bq = Aq if Bq is None else Bq
    Bk = Ak if Bk is None else Bk
    G = torch.zeros(d2, d2, device=DEV)
    for c0 in range(0, V, CHG):
        sl = slice(c0, min(c0 + CHG, V))
        Fa = torch.einsum('ti,tj->tij', Aq[sl], Ak[sl]).reshape(-1, d2)
        Fb = (Fa if (Bq is Aq and Bk is Ak) else
              torch.einsum('ti,tj->tij', Bq[sl], Bk[sl]).reshape(-1, d2))
        G += Fa.T @ Fb
        del Fa
    return G


with torch.no_grad():
    sync(); t0 = time.time()
    Ga = gram(Q1 * QF.sqrt()[:, None], Q2)                  # weighted query-side Gram
    sync()
    t_gram = time.time() - t0
    sync(); t0 = time.time()
    Gb = gram(K1 * QF.sqrt()[:, None], K2)
    tr = float((Ga * Gb.T).sum())                            # tr(Ga Gb) term pattern
    sync()
    t_pair = time.time() - t0
print(f'--- 4. Khatri-Rao Gram (d^2 = {d2}, {d2*d2*4/1e9:.1f} GB each): one Gram {t_gram:.1f} s; '
      f'exact full-table weighted Frobenius = 6 Grams + traces ~ {6*t_gram:.0f} s/eval; '
      f'a 50-step Gram-exact polish x 9 heads ~ {6*t_gram*3*50*9/3600:.1f} h '
      f'(x3 for backward)', flush=True)
print(f'    sanity: tr(Ga Gb) computed in {t_pair - t_gram:.2f} s extra', flush=True)

# --- 5. current sampled step for scale ---
M = 1024
g = torch.Generator(device='cpu').manual_seed(0)
sync(); t0 = time.time()
for _ in range(10):
    sample = torch.randperm(V, generator=g)[:M].to(DEV)
    r1s = recon(torch.cat([Q1[sample], K1[sample]], 1), *fits[0])
    r2s = recon(torch.cat([Q2[sample], K2[sample]], 1), *fits[1])
    Ph = (unit_rms(r1s[:, :HD]) @ unit_rms(r1s[:, HD:]).T / HD) * \
         (unit_rms(r2s[:, :HD]) @ unit_rms(r2s[:, HD:]).T / HD)
    with torch.no_grad():
        P = (Q1[sample] @ K1[sample].T / HD) * (Q2[sample] @ K2[sample].T / HD)
    loss = (Ph - P).pow(2).sum()
    loss.backward()
    for p in params:
        p.grad = None
sync()
t_samp = (time.time() - t0) / 10
print(f'--- 5. current sampled step (M={M}, fwd+bwd): {t_samp*1000:.0f} ms '
      f'-> 1500 x 9 = {t_samp*1500*9/60:.1f} min. Coverage per step: '
      f'{M*M/(V*V)*100:.3f}% of pairs (importance-sampled by frequency in the real loss)',
      flush=True)
print('BENCH DONE', flush=True)
