"""Flagship un-confounded births test (Logan 2026-07-20) — F9 on bilin18, where the
low-rank premise holds (F5). Same design: fixed weight-derived SEED atoms (never trained),
compare WRITE (upstream block write deltas) vs TOKEN (embedding) vs RANDOM by fixed-dict
sparse-code FVU (corr top-k + LS refit), 5 subsamples, at a spread of depths."""
import json, sys
import numpy as np, torch, torch.nn.functional as F
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
TOK = build_eval_tokens(n_chunks=32, seq_len=513)[:32]
PROBE = [3, 6, 10, 17]
M, K, NSEED = 512, 16, 5


@torch.no_grad()
def collect():
    H = [None] * NL; DELTA = [None] * NL
    x = m.transformer.wte(TOK[:, :-1].to(DEV)); x0 = x
    B, T = x.shape[0], x.shape[1]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xb = x
        H[li] = F.rms_norm(x, (D,)).reshape(-1, D).float()
        a = blk.attn; h = F.rms_norm(x, (D,))
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(B, T, NH, HD)
        q, k = qk(a.c_q), qk(a.c_k); q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + blk.mlp(F.rms_norm(x, (D,)))
        DELTA[li] = (x - xb).reshape(-1, D).float()
    return H, DELTA


H, DELTA = collect()
Etok = F.rms_norm(m.transformer.wte.weight.data.float(), (D,))
print(f'bilin18 births test: {H[0].shape[0]} tokens, probing {PROBE}', flush=True)


def unit(A):
    return A / A.norm(dim=1, keepdim=True).clamp_min(1e-8)


def sample(A, seed):
    gg = torch.Generator(device='cpu').manual_seed(seed)
    return unit(A[torch.randperm(A.shape[0], generator=gg)[:M]].clone())


@torch.no_grad()
def fvu(Hc, Dict):
    Y = Hc - Hc.mean(0)
    z = Y @ Dict.T
    _, idx = z.abs().topk(K, 1)
    Psup = Dict[idx].transpose(1, 2)
    G = torch.bmm(Psup.transpose(1, 2), Psup)
    rhs = torch.bmm(Psup.transpose(1, 2), Y.unsqueeze(-1))
    c = torch.linalg.solve(G + 1e-4 * torch.eye(K, device=DEV), rhs)
    return ((torch.bmm(Psup, c).squeeze(-1) - Y) ** 2).sum().item() / (Y ** 2).sum().item()


res = {'m': M, 'k': K, 'per_bond': {}}
print('  bond | WRITE | TOKEN | RANDOM', flush=True)
for ell in PROBE:
    Wup = torch.cat([DELTA[j] for j in range(ell)], 0)
    wv = [fvu(H[ell], sample(Wup, 200 + s)) for s in range(NSEED)]
    tv = [fvu(H[ell], sample(Etok, 100 + s)) for s in range(NSEED)]
    rv = [fvu(H[ell], unit(torch.randn(M, D, generator=torch.Generator(device='cpu').manual_seed(300 + s)).to(DEV))) for s in range(NSEED)]
    res['per_bond'][f'bond{ell}'] = {'write': [round(np.mean(wv), 4), round(np.std(wv), 4)],
                                     'token': [round(np.mean(tv), 4), round(np.std(tv), 4)],
                                     'random': [round(np.mean(rv), 4), round(np.std(rv), 4)]}
    print(f'  {ell:2d}   | {np.mean(wv):.3f}±{np.std(wv):.3f} | {np.mean(tv):.3f}±{np.std(tv):.3f} | {np.mean(rv):.3f}±{np.std(rv):.3f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_births_seed_test.json', 'w'), indent=2)
wm = np.mean([res['per_bond'][f'bond{e}']['write'][0] for e in PROBE])
tm = np.mean([res['per_bond'][f'bond{e}']['token'][0] for e in PROBE])
rm = np.mean([res['per_bond'][f'bond{e}']['random'][0] for e in PROBE])
res['means'] = {'write': round(float(wm), 4), 'token': round(float(tm), 4), 'random': round(float(rm), 4)}
json.dump(res, open(f'{OUT}/bilin18_births_seed_test.json', 'w'), indent=2)
print(f'\nmean: WRITE {wm:.3f} | TOKEN {tm:.3f} | RANDOM {rm:.3f}  '
      f'(write beats token: {wm<tm}, beats random: {wm<rm})', flush=True)
print('bilin18 births seed test done', flush=True)
