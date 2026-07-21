"""Decode the composed pair-features (Logan 2026-07-21): F33/F34 showed the (current,attended)
PAIRS carry joint structure that beats individual token classes for layer-1 selection. WHAT are
those composed classes? Cluster the real co-occurring pairs by their joint layer-1 QK code and
decode example (current -> attended) pairs per class -- the "features in the folded basis".
Data-validated (real attention argmax). Writes composed_pair_features.md.
"""
import sys
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
from transformers import AutoTokenizer
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
Vsz = cfg['vocab_size']
tk = AutoTokenizer.from_pretrained('gpt2')
TOK = build_eval_tokens(n_chunks=48, seq_len=513)[:48]
IDX = TOK[:, :-1].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
L = 1
R_qk = torch.cat([getattr(m.transformer.h[L].attn, n).weight.data.float() for n in ['c_q', 'c_k', 'c_q2', 'c_k2']], 0)
RANK = 128
mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]


@torch.no_grad()
def capture():
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None; qkin = None; att = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn; h = rms(xin)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if li == L:
            qkin = h; att = pat.abs().sum(1)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        x = x + blk.mlp(rms(x))
    return qkin, att


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double()); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T


QKIN, ATT = capture()
H = QKIN.reshape(-1, D)
C = (H.double().T @ H.double()) / H.shape[0]
Cs = sym_pow(C, 0.5); Cis = sym_pow(C, -0.5)
ev, U = torch.linalg.eigh(Cs @ (R_qk.double().T @ R_qk.double()) @ Cs)
ENC = (Cis.float() @ U[:, -RANK:].float())
Z = H @ ENC
cur = IDX.reshape(-1)
att2 = ATT.clone(); dg = torch.arange(T, device=DEV); att2[:, dg, dg] = -1
argk = att2.argmax(-1).view(Bt, T)
keytok = IDX.gather(1, argk).reshape(-1)
offset = (torch.arange(T, device=DEV)[None] - argk).reshape(-1)
pair = cur.long() * Vsz + keytok.long()
up, inv, cnt = torch.unique(pair, return_inverse=True, return_counts=True)
keep = cnt >= 6
pm = torch.zeros(len(up), RANK, device=DEV); pm.index_add_(0, inv, Z); pm /= cnt[:, None].float()
pmk = pm[keep]; upk = up[keep]; cntk = cnt[keep]
print(f'{len(upk)} frequent pairs (>=6 occ)', flush=True)


def kmeans(X, K, iters=25):
    g = torch.Generator(device='cpu').manual_seed(0)
    Cc = X[torch.randperm(len(X), generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        asg = ((X * X).sum(1, True) - 2 * X @ Cc.T + (Cc * Cc).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(Cc); ct = torch.zeros(K, device=X.device)
        Cn.index_add_(0, asg, X); ct.index_add_(0, asg, torch.ones(len(X), device=X.device))
        mm = ct > 0; Cc[mm] = Cn[mm] / ct[mm][:, None]
    return asg, Cc


K = 48
asg, _ = kmeans(F.normalize(pmk, dim=1), K)
dec = lambda t: repr(tk.decode([int(t)]))
lines = ['# Composed pair-features for layer-1 QK (current → attended equivalence classes)\n']
lines.append(f'{len(upk)} frequent (current,attended) pairs clustered into {K} classes by their JOINT '
             f'layer-1 QK code (F33/F34: composed beats individual). Each class = current→attended '
             f'compositions that make layer-1 attention behave the same.\n')
sizes = torch.bincount(asg, minlength=K)
for c in torch.argsort(sizes, descending=True)[:24].tolist():
    members = torch.nonzero(asg == c).squeeze(1)
    members = members[torch.argsort(cntk[members], descending=True)][:10]
    exs = [f'{dec(upk[i]//Vsz)}→{dec(upk[i]%Vsz)}' for i in members.tolist()]
    lines.append(f'- **class {c}** ({int(sizes[c])} pairs): ' + '  '.join(exs))
open(f'{OUT}/composed_pair_features.md', 'w').write('\n'.join(lines) + '\n')
print('wrote composed_pair_features.md', flush=True)
print('\n--- sample composed pair-classes (current -> attended) ---', flush=True)
for c in torch.argsort(sizes, descending=True)[:8].tolist():
    members = torch.nonzero(asg == c).squeeze(1)
    members = members[torch.argsort(cntk[members], descending=True)][:7]
    exs = [f'{dec(upk[i]//Vsz)}->{dec(upk[i]%Vsz)}' for i in members.tolist()]
    print(f'class {c}: ' + '  '.join(exs), flush=True)
print('bilin18 composed qualitative done', flush=True)
