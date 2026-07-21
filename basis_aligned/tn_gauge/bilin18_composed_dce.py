"""Strengthen F33 (Logan 2026-07-21): composed vs individual at the BINDING metric (ΔCE) with more
data. At matched number of classes K, quantize each position's layer-1 QK code z three ways and
patch layer-1 QK, ΔCE:
  FREE      : kmeans on z into K (best possible K-quantization; upper bound)
  COMPOSED  : cluster the real (current,attended) PAIRS into K; each position -> its pair's cluster
  INDIVIDUAL: cluster current tokens into K1 and attended into K2 (K1*K2=K); position -> cell centroid
If INDIVIDUAL >> FREE while COMPOSED ~ FREE, the QK-1 code carries joint (current x attended)
structure beyond individual token classes = F33 at the binding metric. Data-validated (real
attention argmax for the attended token); more sequences than F33. Gate: no-quantization = reference.
"""
import sys, json
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
TOK = build_eval_tokens(n_chunks=24, seq_len=513)[:24]
IDX = TOK[:, :-1].to(DEV); TGT = TOK[:, 1:].to(DEV)
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


@torch.no_grad()
def ce_patch(h_qk):
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
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
    xf = rms(x); tot = 0.0; n = 0
    for i in range(0, Bt, 4):
        lg = 30 * torch.tanh(m.lm_head(xf[i:i+4]) / 30)
        t = TGT[i:i+4]
        tot += F.cross_entropy(lg.float().reshape(-1, Vsz), t.reshape(-1)).item() * t.numel(); n += t.numel()
    return tot / n


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double()); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T


QKIN, ATT = capture()
H = QKIN.reshape(-1, D)
C = (H.double().T @ H.double()) / H.shape[0]
Cs = sym_pow(C, 0.5); Cis = sym_pow(C, -0.5)
ev, U = torch.linalg.eigh(Cs @ (R_qk.double().T @ R_qk.double()) @ Cs)
Wz = U[:, -RANK:].float(); ENC = (Cis.float() @ Wz); DEC = (Cs.float() @ Wz).T
Z = H @ ENC
CE0 = ce_patch(rms(H.reshape(Bt, T, D)))
CEref = 0.0; _n = 0
for _i in range(0, Bt, 4):
    _lg = reference_forward(m, IDX[_i:_i+4], 'bf16').float()
    CEref += F.cross_entropy(_lg.reshape(-1, Vsz), TGT[_i:_i+4].reshape(-1)).item() * TGT[_i:_i+4].numel(); _n += TGT[_i:_i+4].numel()
CEref /= _n
print(f'gate {CE0:.4f} vs ref {CEref:.4f} (Δ {abs(CE0-CEref):.1e})', flush=True)

cur = IDX.reshape(-1)
att2 = ATT.clone(); dg = torch.arange(T, device=DEV); att2[:, dg, dg] = -1
keytok = IDX.gather(1, att2.argmax(-1).view(Bt, T)).reshape(-1)


def kmeans(X, K, iters=15):
    g = torch.Generator(device='cpu').manual_seed(0)
    Cc = X[torch.randperm(len(X), generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        asg = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 16384):
            xx = X[i:i+16384]
            asg[i:i+16384] = ((xx*xx).sum(1,True) - 2*xx@Cc.T + (Cc*Cc).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(Cc); ct = torch.zeros(K, device=X.device)
        Cn.index_add_(0, asg, X); ct.index_add_(0, asg, torch.ones(len(X), device=X.device))
        mm = ct > 0; Cc[mm] = Cn[mm]/ct[mm][:,None]
    return asg, Cc


def dce_from_assign(asg, K):
    cent = torch.zeros(K, RANK, device=DEV); ct = torch.zeros(K, device=DEV)
    cent.index_add_(0, asg, Z); ct.index_add_(0, asg, torch.ones(len(Z), device=DEV))
    mm = ct > 0; cent[mm] /= ct[mm][:, None]
    h = (cent[asg] @ DEC).reshape(Bt, T, D)
    return ce_patch(rms(h)) - CE0


# token-class maps (cluster tokens by their mean z)
def token_class(tokids, K1):
    uu = torch.unique(tokids)
    tm = torch.zeros(len(uu), RANK, device=DEV)
    for i, t in enumerate(uu):
        tm[i] = Z[tokids == t].mean(0)
    a_u, _ = kmeans(tm, min(K1, len(uu)))
    mp = torch.full((Vsz,), 0, device=DEV, dtype=torch.long); mp[uu] = a_u
    return mp


res = {'baseline_ce': round(CE0, 4), 'gate': round(abs(CE0-CEref), 5), 'free': {}, 'composed': {}, 'individual': {}}
print('ΔCE at matched K: FREE (best) vs COMPOSED (pairs) vs INDIVIDUAL (c-class x a-class):', flush=True)
print('  K | free | composed | individual', flush=True)
pair = cur.long() * Vsz + keytok.long()
for K1 in [8, 16, 32]:
    K = K1 * K1
    # FREE
    a_free, _ = kmeans(Z, K); d_free = dce_from_assign(a_free, K)
    # INDIVIDUAL
    cmap = token_class(cur, K1); amap = token_class(keytok, K1)
    a_ind = cmap[cur] * K1 + amap[keytok]; d_ind = dce_from_assign(a_ind, K)
    # COMPOSED: cluster pairs by mean-z, assign positions by pair
    up, inv, cnt = torch.unique(pair, return_inverse=True, return_counts=True)
    pm = torch.zeros(len(up), RANK, device=DEV); pm.index_add_(0, inv, Z); pm /= cnt[:, None].float()
    a_pair, _ = kmeans(pm, min(K, len(up)))
    a_comp = a_pair[inv]; d_comp = dce_from_assign(a_comp, min(K, len(up)))
    res['free'][K] = round(d_free, 4); res['composed'][K] = round(d_comp, 4); res['individual'][K] = round(d_ind, 4)
    print(f'  {K:4d} | {d_free:+.4f} | {d_comp:+.4f} | {d_ind:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_composed_dce.json', 'w'), indent=2)
print(f'\n(n_pairs={len(up)}) composed<=individual at all K: '
      f'{all(res["composed"][k] <= res["individual"][k] for k in res["composed"])}', flush=True)
print('bilin18 composed dce done', flush=True)
