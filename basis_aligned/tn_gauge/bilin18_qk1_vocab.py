"""Compression RELATIVE TO THE INPUT VOCAB for layer-1 QK (Logan 2026-07-21, correcting F26).
F26 clustered the continuous ACTIVATION state (positions) -> continuous, clustering loses. But
the compression Logan wants is relative to the INPUT: do the current-token identities reduce to
few equivalence classes for QK-1, with the residual variation being context (the cross terms)?
Measure, in the used-subspace (r=128) that QK-1 reads:
  (1) between-token variance fraction of the QK-1 code z (how token-determined vs context);
  (2) cluster the VOCAB (current token) into K classes by mean-z; replace each position's z by
      its TOKEN's class-mean z (NOT the state's own cluster -- this is a VOCAB reduction, not a
      state reduction like F26); ΔCE vs K.
Low ΔCE at small K => current-token identity reduces to few QK-1 equivalence classes = input-
relative compression F26 missed. Held-out-ish: this is a first cut (in-sample), reports the
token-determined ceiling. No V×V (cluster the vocab into K).
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
Vsz = cfg['vocab_size']
TOK = build_eval_tokens(n_chunks=16, seq_len=513)[:12]
IDX = TOK[:, :-1].to(DEV); TGT = TOK[:, 1:].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
L = 1
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
R = torch.cat([getattr(m.transformer.h[L].attn, n).weight.data.float() for n in NAMES], 0)
RANK = 128


@torch.no_grad()
def qk_input():
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x
        if li == L:
            return rms(xin)
        a = blk.attn; h = rms(xin)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        x = x + blk.mlp(rms(x))


@torch.no_grad()
def patch_forward(h_qk):
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn
        hq = h_qk if li == L else rms(xin); hv = rms(xin)
        qkf = lambda lin, hh: apply_rot(F.rms_norm(lin(hh).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        q, k = qkf(a.c_q, hq), qkf(a.c_k, hq); q2, k2 = qkf(a.c_q2, hq), qkf(a.c_k2, hq)
        v = a.c_v(hv).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', q, k) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        x = x + blk.mlp(rms(x))
    return 30 * torch.tanh(m.lm_head(rms(x)) / 30)


def cce(h_qk):
    return F.cross_entropy(patch_forward(h_qk).float().reshape(-1, Vsz), TGT.reshape(-1)).item()


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double()); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T


H = qk_input().reshape(-1, D)                              # (Ntok, D) QK-1 input
tok = IDX.reshape(-1)                                      # current token id per position
C = (H.double().T @ H.double()) / H.shape[0]
Cs = sym_pow(C, 0.5); Cis = sym_pow(C, -0.5)
ev, U = torch.linalg.eigh(Cs @ (R.double().T @ R.double()) @ Cs)
Wz = U[:, -RANK:].float()
ENC = (Cis.float() @ Wz)                                   # h -> z (used-subspace code)
DEC = (Cs.float() @ Wz).T
Z = H @ ENC                                                # (Ntok, r) codes

# (1) between-token variance fraction of z
uniq = torch.unique(tok)
tot_var = ((Z - Z.mean(0)) ** 2).sum().item()
tok_mean = torch.zeros(Vsz, RANK, device=DEV); cnt = torch.zeros(Vsz, device=DEV)
tok_mean.index_add_(0, tok, Z); cnt.index_add_(0, tok, torch.ones(len(tok), device=DEV))
nz = cnt > 0; tok_mean[nz] /= cnt[nz][:, None]
within = ((Z - tok_mean[tok]) ** 2).sum().item()
between_frac = 1 - within / tot_var
CE0 = cce(rms(H.reshape(Bt, T, D)))
CEref = F.cross_entropy(reference_forward(m, IDX, 'bf16').float().reshape(-1, Vsz), TGT.reshape(-1)).item()
print(f'gate {CE0:.4f} vs ref {CEref:.4f}; between-token variance fraction of QK-1 code: {between_frac:.3f}', flush=True)
res = {'baseline_ce': round(CE0, 4), 'gate': round(abs(CE0 - CEref), 5),
       'between_token_var_frac': round(between_frac, 3), 'n_unique_tokens': int(len(uniq)), 'vocab_cluster': {},
       'token_mean_only_dce': None}

# token-mean only (K = n_unique, replace z by its token mean) -- the token-determined ceiling
h_tm = (tok_mean[tok] @ DEC).reshape(Bt, T, D)
res['token_mean_only_dce'] = round(cce(rms(h_tm)) - CE0, 4)
print(f'replace z by current-token MEAN (={len(uniq)} classes): ΔCE {res["token_mean_only_dce"]:+.4f}', flush=True)


def kmeans(X, K, iters=15):
    g = torch.Generator(device='cpu').manual_seed(0)
    Cc = X[torch.randperm(len(X), generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        a = ((X * X).sum(1, True) - 2 * X @ Cc.T + (Cc * Cc).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(Cc); ct = torch.zeros(K, device=X.device)
        Cn.index_add_(0, a, X); ct.index_add_(0, a, torch.ones(len(X), device=X.device))
        m2 = ct > 0; Cc[m2] = Cn[m2] / ct[m2][:, None]
    return a, Cc


# (2) cluster the VOCAB (unique tokens) into K by mean-z; each position gets its token's class mean
print('\ncluster VOCAB into K classes by mean QK-1 code; replace z by token-class-mean, ΔCE:', flush=True)
for K in [32, 128, 512, 2048]:
    if K >= len(uniq):
        continue
    a_u, cent = kmeans(tok_mean[uniq], K)                 # cluster unique tokens
    tok2class = torch.full((Vsz,), -1, device=DEV, dtype=torch.long)
    tok2class[uniq] = a_u
    class_mean_z = cent[tok2class[tok]]                   # each position -> its token's class centroid
    h_cm = (class_mean_z @ DEC).reshape(Bt, T, D)
    d = cce(rms(h_cm)) - CE0
    res['vocab_cluster'][K] = round(d, 4)
    print(f'  K={K:5d} vocab classes: ΔCE {d:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_qk1_vocab.json', 'w'), indent=2)
print('\ninterpretation: token-mean ΔCE = context (cross-term) part; vocab-cluster K = input-relative compression', flush=True)
print('bilin18 qk1 vocab done', flush=True)
