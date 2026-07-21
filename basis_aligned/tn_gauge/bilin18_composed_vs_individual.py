"""Composed vs individual features for layer-1 QK (Logan 2026-07-21): does clustering the
(current-token, attended-token) PAIRS by their joint QK-1 code beat clustering each side
individually? Empirical + data-validated: at each position collect the layer-1 QK code z_t
(used-subspace), current token c_t, and the token it attends to most a_t (argmax layer-1
attention). Group by real (c,a) pairs. Then:
  COMPOSED: cluster the P frequent pairs by mean-z into K classes; FVU of reconstructing pair
    mean-z from centroids. bits = K * r.
  INDIVIDUAL: cluster current tokens into K1 and attended tokens into K2 by their marginal
    mean-z; represent each pair by the mean-z of its (c-class, a-class) cell; FVU. bits =
    (K1+K2)*r [token->class] + K1*K2*r [cell table]  (dominant K1*K2*r).
Compare FVU at matched pair-classes (K vs K1*K2). If COMPOSED reaches lower FVU with fewer
classes -> pairs perform the same function beyond what individual tokens predict = Logan's claim.
Structural (FVU of the QK-1 code), not ΔCE; data-validated (real co-occurring pairs).
"""
import sys, json
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
Vsz = cfg['vocab_size']
TOK = build_eval_tokens(n_chunks=48, seq_len=513)[:48]
IDX = TOK[:, :-1].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
L = 1
R_qk = torch.cat([getattr(m.transformer.h[L].attn, n).weight.data.float() for n in ['c_q', 'c_k', 'c_q2', 'c_k2']], 0)
RANK = 128


@torch.no_grad()
def capture():
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    qkin = None; att = None
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
            qkin = h.reshape(-1, D); att = pat.abs().sum(1)          # (B,Tq,Tk) over heads
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        x = x + blk.mlp(rms(x))
    return qkin, att


QKIN, ATT = capture()
# used-subspace ENC (fit on QK-1 input)
def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double()); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T
C = (QKIN.double().T @ QKIN.double()) / QKIN.shape[0]
Cs = sym_pow(C, 0.5); Cis = sym_pow(C, -0.5)
ev, U = torch.linalg.eigh(Cs @ (R_qk.double().T @ R_qk.double()) @ Cs)
ENC = (Cis.float() @ U[:, -RANK:].float())                    # (D, r)
Z = (QKIN @ ENC)                                              # (Ntok, r) QK-1 codes

# per-position current token and argmax-attended token (offset>=1 to skip self)
cur = IDX.reshape(-1)
ATTflat = ATT.clone()
diag = torch.arange(T, device=DEV)
ATTflat[:, diag, diag] = -1                                   # exclude self
att_arg = ATTflat.argmax(-1).reshape(-1)                      # (Ntok,) key position
key_tok = IDX.gather(1, att_arg.view(Bt, T)).reshape(-1)      # attended token id
posrow = torch.arange(T, device=DEV).repeat(Bt)
valid = posrow >= 5                                           # skip sequence start

c = cur[valid]; a = key_tok[valid]; z = Z[valid]
pair = c.long() * Vsz + a.long()
upair, inv, cnt = torch.unique(pair, return_inverse=True, return_counts=True)
# pair mean-z
pmean = torch.zeros(len(upair), RANK, device=DEV)
pmean.index_add_(0, inv, z); pmean /= cnt[:, None].float()
# keep frequent pairs
keep = cnt >= 8
P = int(keep.sum())
pm = pmean[keep]; pc = (upair[keep] // Vsz); pa = (upair[keep] % Vsz); pw = cnt[keep].float()
print(f'{P} frequent (current,attended) pairs (>=8 occ); QK-1 code dim {RANK}', flush=True)
totvar = ((pm - (pm * pw[:, None]).sum(0) / pw.sum()) ** 2 * pw[:, None]).sum().item()


def kmeans(X, K, w, iters=20):
    g = torch.Generator(device='cpu').manual_seed(0)
    Cc = X[torch.randperm(len(X), generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        asg = ((X * X).sum(1, True) - 2 * X @ Cc.T + (Cc * Cc).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(Cc); ct = torch.zeros(K, device=X.device)
        Cn.index_add_(0, asg, X * w[:, None]); ct.index_add_(0, asg, w)
        mm = ct > 0; Cc[mm] = Cn[mm] / ct[mm][:, None]
    return asg, Cc


def wfvu(recon):
    return (((pm - recon) ** 2) * pw[:, None]).sum().item() / totvar


res = {'n_pairs': P, 'r': RANK, 'composed': {}, 'individual': {}}
print('COMPOSED (cluster pairs) vs INDIVIDUAL (cluster each side), FVU of pair QK-1 code:', flush=True)
print('  #pair-classes | composed FVU | individual FVU (K1=K2)', flush=True)
# marginal token means for individual clustering
def token_means(tokids):
    uu = torch.unique(tokids)
    mp = {int(t): pm[tokids == t].mean(0) for t in uu} if False else None
    return uu
for K1 in [4, 8, 16, 32]:
    Kpairs = K1 * K1
    # composed
    asg_c, cent_c = kmeans(pm, min(Kpairs, P), pw)
    fvu_c = wfvu(cent_c[asg_c])
    # individual: cluster current-token reps and attended-token reps by their mean pair-code
    # current-token feature = weighted mean pm over pairs sharing that current token
    uc = torch.unique(pc); ua = torch.unique(pa)
    cfeat = torch.stack([(pm[pc == t] * pw[pc == t, None]).sum(0) / pw[pc == t].sum() for t in uc])
    afeat = torch.stack([(pm[pa == t] * pw[pa == t, None]).sum(0) / pw[pa == t].sum() for t in ua])
    ac, _ = kmeans(cfeat, min(K1, len(uc)), torch.ones(len(uc), device=DEV))
    aa, _ = kmeans(afeat, min(K1, len(ua)), torch.ones(len(ua), device=DEV))
    c2cl = {int(t): int(ac[i]) for i, t in enumerate(uc)}
    a2cl = {int(t): int(aa[i]) for i, t in enumerate(ua)}
    cell = torch.tensor([c2cl[int(pc[i])] * K1 + a2cl[int(pa[i])] for i in range(P)], device=DEV)
    # cell centroids
    cent_cell = torch.zeros(K1 * K1, RANK, device=DEV); ct = torch.zeros(K1 * K1, device=DEV)
    cent_cell.index_add_(0, cell, pm * pw[:, None]); ct.index_add_(0, cell, pw)
    mm = ct > 0; cent_cell[mm] /= ct[mm][:, None]
    fvu_i = wfvu(cent_cell[cell])
    res['composed'][Kpairs] = round(fvu_c, 4); res['individual'][Kpairs] = round(fvu_i, 4)
    print(f'  {Kpairs:5d} (K1={K1}) |   {fvu_c:.4f}     |   {fvu_i:.4f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_composed_vs_individual.json', 'w'), indent=2)
comp_wins = all(res['composed'][k] <= res['individual'][k] for k in res['composed'])
res['composed_beats_individual'] = bool(comp_wins)
json.dump(res, open(f'{OUT}/bilin18_composed_vs_individual.json', 'w'), indent=2)
print(f'\ncomposed beats individual at matched pair-classes: {comp_wins}', flush=True)
print('bilin18 composed vs individual done', flush=True)
