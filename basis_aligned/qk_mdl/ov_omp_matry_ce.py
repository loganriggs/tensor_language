"""Binding-metric (cross-entropy) confirmation of OVD-7/8 (Logan 2026-07-21).
The least-squares/OMP and Matryoshka results were reconstruction-FVU only, on
head 0. Confirm the orderings hold at the real ΔCE audit, ALL heads:
  A. linear-encoder per-token top-k (k=8) -- reference (the cheap encoder)
  B. OMP per-token (LS, k=8) -- proper coefficients
  C. OMP batch (LS, marginal-error global budget, avg k=8) -- Logan's scheme
  D. Matryoshka dict, per-token top-k over full 512 (k=8)
Reconstruct each head's value table with each scheme, patch layer 0, audit ΔCE."""
import json
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
AUDIT = build_eval_tokens(n_chunks=20, seq_len=513)[4:20]
E_hat = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
VT = (E_hat @ m.transformer.h[0].attn.c_v.weight.detach().float().T).view(V, NH, HD)
K, KMAX, N = 8, 28, 512
PREFIXES = [32, 128, 512]


def train_dict(X, nested=False, steps=2500):
    g = torch.Generator(device='cpu'); g.manual_seed(0)
    Dm = X[torch.randperm(V, generator=g)[:N]].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone(); b = X.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=3e-3)
    for _ in range(steps):
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (X - b) @ We.T
        loss = 0.0
        for P in (PREFIXES if nested else [N]):
            zp = z[:, :P]
            _, idx = zp.abs().topk(K, 1); coeff = torch.gather(zp, 1, idx)
            loss = loss + ((b + (coeff.unsqueeze(-1) * Dn[:P][idx]).sum(1) - X) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach(), We.detach(), b.detach()


@torch.no_grad()
def recon_topk(X, Dn, We, b):
    z = (X - b) @ We.T
    _, idx = z.abs().topk(K, 1); coeff = torch.gather(z, 1, idx)
    return b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)


@torch.no_grad()
def omp(X, Dn, nsteps):
    Y = X - X.mean(0)
    r = Y.clone()
    res_sse = [(r ** 2).sum(1)]
    sup = torch.full((V, nsteps), -1, device=DEV, dtype=torch.long)
    recon_at = [torch.zeros_like(Y)]
    chosen = torch.zeros(V, N, dtype=torch.bool, device=DEV)
    for s in range(nsteps):
        corr = (r @ Dn.T).abs(); corr[chosen] = -1
        a = corr.argmax(1); sup[:, s] = a
        chosen[torch.arange(V, device=DEV), a] = True
        Ds = Dn[sup[:, :s + 1]]
        G = torch.bmm(Ds, Ds.transpose(1, 2))
        rhs = torch.bmm(Ds, Y.unsqueeze(-1)).squeeze(-1)
        c = torch.linalg.solve(G + 1e-6 * torch.eye(s + 1, device=DEV), rhs)
        recon = torch.bmm(c.unsqueeze(1), Ds).squeeze(1)
        r = Y - recon
        res_sse.append((r ** 2).sum(1)); recon_at.append(recon)
    return torch.stack(res_sse, 1), torch.stack(recon_at, 1)   # (V,ns+1),(V,ns+1,D)


@torch.no_grad()
def ce(v_tab):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        bt = AUDIT[i:i + 4].to(DEV)
        x = m.transformer.wte(bt[:, :-1]); x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        B, T = x.shape[0], x.shape[1]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
            v = v_tab[bt[:, :-1]].to(x.dtype) if (li == 0 and v_tab is not None) else a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            q, k = qk(a.c_q), qk(a.c_k); q2, k2 = qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        x = F.rms_norm(x, (x.size(-1),))
        lg = 30 * torch.tanh(m.lm_head(x) / 30)
        tot += F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), bt[:, 1:].reshape(-1)).item() * bt[:, 1:].numel()
        n += bt[:, 1:].numel()
    return tot / n


CE0 = ce(None)
res = {'baseline_ce': CE0, 'arms': {}}
print(f'baseline {CE0:.4f}', flush=True)

vt_lin = torch.empty_like(VT)
vt_ompT = torch.empty_like(VT)
vt_ompB = torch.empty_like(VT)
vt_mat = torch.empty_like(VT)
for hh in range(NH):
    X = VT[:, hh].to(DEV)
    Dn, We, b = train_dict(X, nested=False)
    vt_lin[:, hh] = recon_topk(X, Dn, We, b)
    # OMP per-token and batch (marginal-error) on the SAME plain dict atoms
    sse, recon = omp(X, Dn, K)
    vt_ompT[:, hh] = b + recon[:, K]
    sseM, reconM = omp(X, Dn, KMAX)
    gains = sseM[:, :-1] - sseM[:, 1:]
    thr = gains.reshape(-1).topk(K * V).values.min()
    p = (gains >= thr).sum(1)                          # per-word count
    vt_ompB[:, hh] = b + reconM[torch.arange(V, device=DEV), p]
    # matryoshka
    Dn2, We2, b2 = train_dict(X, nested=True)
    vt_mat[:, hh] = recon_topk(X, Dn2, We2, b2)
    print(f'  head {hh} reconstructed', flush=True)

for tag, vt in [('A linear-encoder per-token k=8', vt_lin),
                ('B OMP per-token (LS) k=8', vt_ompT),
                ('C OMP batch (LS, marginal-error) avg-k=8', vt_ompB),
                ('D Matryoshka per-token k=8', vt_mat)]:
    d = ce(vt) - CE0
    res['arms'][tag] = round(d, 4)
    print(f'{tag}: dCE {d:+.4f}', flush=True)
    json.dump(res, open(f'{QK}/ov_omp_matry_ce.json', 'w'), indent=2)
print('ov omp matry ce done', flush=True)
