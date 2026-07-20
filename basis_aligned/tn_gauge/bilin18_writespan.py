"""Gate 2b on the FLAGSHIP (Logan 2026-07-20). bilin18's residual stream IS low-rank
(rank@90% ~150-260 of 1152), so unlike the toy it can actually test atom-birth. Train
a shared dictionary on bilin18 bond activations and ask, per bond, whether the coding
residual lies in the UPSTREAM block-write span (=> atom-birth, regime survives, births
from write directions) or is isotropic/random (=> capacity). Middle bonds are most
compressible; test a spread of depths. Dictionary on rms-normed block-input activations,
m=2048, LS-refit k=32, K=64 projection subspaces."""
import json, sys
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL = len(m.transformer.h)
TOK = build_eval_tokens(n_chunks=40, seq_len=513)[:40]
PROBE_BONDS = [3, 6, 10, 17]


@torch.no_grad()
def collect():
    H = [None] * NL
    DELTA = [None] * NL
    x = m.transformer.wte(TOK[:, :-1].to(DEV))
    x0 = x
    B, T = x.shape[0], x.shape[1]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xb = x
        H[li] = F.rms_norm(x, (D,)).reshape(-1, D).float()
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(B, T, NH, HD)
        q, k = qk(a.c_q), qk(a.c_k); q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        DELTA[li] = (x - xb).reshape(-1, D).float()       # full block write into the stream
    return H, DELTA


H, DELTA = collect()
NT = H[0].shape[0]
print(f'bilin18 d={D}, {NT} tokens, probing bonds {PROBE_BONDS}', flush=True)


def train_dict(X, mm=2048, steps=2500):
    g = torch.Generator(device='cpu').manual_seed(0)
    Phi = X[torch.randperm(X.shape[0], generator=g)[:mm]].clone().T
    Phi = Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)
    We = Phi.clone(); b = X.mean(0).clone()
    Phi.requires_grad_(True); We.requires_grad_(True); b.requires_grad_(True)
    opt = torch.optim.Adam([Phi, We, b], lr=3e-3)
    for _ in range(steps):
        Pn = Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)
        z = (X - b) @ We
        _, idx = z.abs().topk(16, 1); coeff = torch.gather(z, 1, idx)
        rec = b + torch.einsum('nk,nkd->nd', coeff, Pn.T[idx])
        (((rec - X) ** 2).mean()).backward(); opt.step(); opt.zero_grad()
    return (Phi / Phi.norm(dim=0, keepdim=True).clamp_min(1e-8)).detach(), We.detach(), b.detach()


@torch.no_grad()
def encode(h, Phi, We, b, k=32):
    z = (h - b) @ We
    _, idx = z.abs().topk(k, 1)
    Psup = Phi[:, idx].permute(1, 2, 0)
    G = torch.bmm(Psup, Psup.transpose(1, 2))
    rhs = torch.bmm(Psup, (h - b).unsqueeze(-1))
    c = torch.linalg.solve(G + 1e-4 * torch.eye(k, device=DEV), rhs)
    return b + torch.bmm(c.transpose(1, 2), Psup).squeeze(1)


def top_basis(X, K):
    _, _, Vh = torch.linalg.svd(X.double(), full_matrices=False)
    return Vh[:K].T.float()


def captured(r, U):
    return ((r @ U) ** 2).sum().item() / (r ** 2).sum().item()


def eff_rank(X):
    s = torch.linalg.svdvals(X.double()); p = s / s.sum()
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


K = 64
res = {'d_model': D, 'm': 2048, 'k': 32, 'K_proj': K, 'bonds': {}}
print('  bond | resid-FVU | write-span | random | self-ceiling | resid eff-rank', flush=True)
for ell in PROBE_BONDS:
    Phi, We, b = train_dict(H[ell])
    r = H[ell] - encode(H[ell], Phi, We, b)
    Wup = torch.cat([DELTA[j] for j in range(ell)], 0)
    idxsub = torch.randperm(Wup.shape[0])[:20000]      # cap for SVD cost
    Uwrite = top_basis(Wup[idxsub], K)
    wcap = captured(r, Uwrite)
    self_cap = captured(r, top_basis(r, K))
    rnd = K / D
    fvu = (r ** 2).sum().item() / ((H[ell] - H[ell].mean(0)) ** 2).sum().item()
    er = eff_rank(r)
    res['bonds'][f'bond{ell}'] = {'resid_fvu': round(fvu, 4), 'write_span': round(wcap, 4),
                                  'random': round(rnd, 4), 'self_ceiling': round(self_cap, 4),
                                  'resid_eff_rank': round(er, 1)}
    print(f'  {ell:2d}   |  {fvu:.3f}   |   {wcap:.3f}   |  {rnd:.3f} |    {self_cap:.3f}    |  {er:.1f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_writespan.json', 'w'), indent=2)
print('bilin18 writespan done -> bilin18_writespan.json', flush=True)
