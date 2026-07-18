"""Bottom-up contextual tables (Logan 2026-07-20): the earliest streams
(attn0, mlp0, attn1, mlp1) are sequence-determined, not token-determined —
attn0-out(i) = sum_j P(t_i,t_j,D)v(t_j). Test the next context order:
BIGRAM-conditional mean tables (frequent bigrams, unigram backoff) vs the
0th-order unigram tables, audited in the windowed-D qk-read harness at W=1
(where unigram costs +0.861) and W=2 (+0.429). If bigram tables crack the
gap, second-order context is n-gram-shaped and the contextual atoms are
nameable token-pairs (the co-occurrence structure Logan flagged).
Also reports per-stream R^2 of unigram vs bigram prediction + coverage."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/ngram_tables.json'
NG_STREAMS = ['attn0', 'mlp0', 'attn1', 'mlp1']
M_BIGRAMS = 600000
MIN_CNT = 4
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL, V = cfg['n_layer'], cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 6144, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]
print(f'train tokens {TRAIN.numel()/1e6:.1f}M', flush=True)


def created_layer(nm):
    return int(nm[4:] if nm.startswith('attn') else nm[3:])


# ---- pass 0: pick frequent bigrams ----
codes_all = []
for i in range(0, len(TRAIN), 256):
    b = TRAIN[i:i + 256, :-1]
    codes_all.append((b[:, :-1].long() * V + b[:, 1:].long()).reshape(-1))
codes_all = torch.cat(codes_all)
uc, cnt = torch.unique(codes_all, return_counts=True)
keep = cnt >= MIN_CNT
uc, cnt = uc[keep], cnt[keep]
if len(uc) > M_BIGRAMS:
    top = cnt.topk(M_BIGRAMS).indices
    uc = uc[top]
uc, _ = uc.sort()
print(f'{len(uc)} frequent bigrams (cnt>={MIN_CNT})', flush=True)
del codes_all


def bigram_index(prev, cur):
    """(...,)->index into uc or -1"""
    code = prev.long() * V + cur.long()
    pos = torch.searchsorted(uc.to(code.device), code)
    pos = pos.clamp_max(len(uc) - 1)
    hit = uc.to(code.device)[pos] == code
    return torch.where(hit, pos, torch.full_like(pos, -1))


# ---- pass 1: accumulate unigram + bigram means for NG_STREAMS ----
acc_u = {nm: torch.zeros(V, D) for nm in NG_STREAMS}
cnt_u = torch.zeros(V)
acc_b = {nm: torch.zeros(len(uc), D) for nm in NG_STREAMS}
cnt_b = torch.zeros(len(uc))
UC_DEV = uc.to(DEV)


@torch.no_grad()
def collect(idx):
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    got = {}
    for li in (0, 1):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + attn_out
        got[f'attn{li}'] = attn_out
        rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        mlp_out = blk.mlp(x * rms2)
        x = x + mlp_out
        got[f'mlp{li}'] = mlp_out
    flat = idx.reshape(-1).cpu()
    bi = torch.full((B, T), -1, dtype=torch.long, device=DEV)
    bi[:, 1:] = bigram_index(idx[:, :-1], idx[:, 1:])
    bif = bi.reshape(-1).cpu()
    hit = bif >= 0
    cnt_u.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
    cnt_b.index_add_(0, bif[hit], torch.ones(int(hit.sum())))
    for nm in NG_STREAMS:
        z = got[nm].reshape(-1, D).float().cpu()
        acc_u[nm].index_add_(0, flat, z)
        acc_b[nm].index_add_(0, bif[hit], z[hit])


for i in range(0, len(TRAIN), 8):
    collect(TRAIN[i:i + 8, :-1].to(DEV))
    if i % 512 == 0:
        print(f'  collect {i}/{len(TRAIN)}', flush=True)

TAB_U, TAB_B = {}, {}
for nm in NG_STREAMS:
    t = acc_u[nm] / cnt_u.clamp_min(1)[:, None]
    t[cnt_u == 0] = acc_u[nm].sum(0) / cnt_u.sum()
    TAB_U[nm] = t
    TAB_B[nm] = acc_b[nm] / cnt_b.clamp_min(1)[:, None]   # rows with cnt 0 unused
del acc_u, acc_b
print('tables built', flush=True)

# quick R^2 on a held-back slice
r2 = {}
with torch.no_grad():
    idx = ALL[12:16, :-1].to(DEV)      # audit-adjacent slice
    B, T = idx.shape
    x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    # recompute streams via collect-body (dup for simplicity)
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    got = {}
    for li in (0, 1):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        x = x + attn_out
        got[f'attn{li}'] = attn_out
        rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        mlp_out = blk.mlp(x * rms2)
        x = x + mlp_out
        got[f'mlp{li}'] = mlp_out
    flat = idx.reshape(-1).cpu()
    bi = torch.full((B, T), -1, dtype=torch.long, device=DEV)
    bi[:, 1:] = bigram_index(idx[:, :-1], idx[:, 1:])
    bif = bi.reshape(-1).cpu()
    cov = float((bif >= 0).float().mean())
    for nm in NG_STREAMS:
        z = got[nm].reshape(-1, D).float().cpu()
        pu = TAB_U[nm][flat]
        pb = pu.clone()
        hit = bif >= 0
        pb[hit] = TAB_B[nm][bif[hit]]
        tot = (z - z.mean(0)).pow(2).sum()
        r2[nm] = {'unigram': round(1 - float((z - pu).pow(2).sum() / tot), 3),
                  'bigram+backoff': round(1 - float((z - pb).pow(2).sum() / tot), 3)}
        print(f'{nm}: R2 unigram {r2[nm]["unigram"]:.3f} -> bigram {r2[nm]["bigram+backoff"]:.3f}', flush=True)
print(f'bigram coverage {cov:.3f}', flush=True)

RAW = torch.load(f'{QK}/stream_tables.pt')
RAW = {nm: torch.nan_to_num(t.float(), posinf=65504, neginf=-65504)
       for nm, t in RAW.items()}


@torch.no_grad()
def audit_ce(W, use_bigram):
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        idx_cpu = idx.cpu()
        bi = torch.full((B, T), -1, dtype=torch.long, device=DEV)
        bi[:, 1:] = bigram_index(idx[:, :-1], idx[:, 1:])
        bif = bi.cpu()
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        streams, tabs = {}, {}
        for li, blk in enumerate(m.transformer.h):
            lam0, lam1 = blk.lambdas[0], blk.lambdas[1]
            x = lam0 * x + lam1 * x0
            for nm in streams:
                streams[nm] = lam0 * streams[nm]
                tabs[nm] = lam0 * tabs[nm]
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            h_qk = h
            if li >= 1:
                old = [nm for nm in streams if created_layer(nm) < li - W]
                if old:
                    xp = x
                    for nm in old:
                        xp = xp - streams[nm] + tabs[nm]
                    h_qk = F.rms_norm(xp, (xp.size(-1),))
            qn = lambda lin: apply_rot(F.rms_norm(lin(h_qk).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + attn_out
            streams[f'attn{li}'] = attn_out
            nm = f'attn{li}'
            if use_bigram and nm in NG_STREAMS:
                g = TAB_U[nm][idx_cpu]
                hit = bif >= 0
                g[hit] = TAB_B[nm][bif[hit]]
                tabs[nm] = g.to(DEV, x.dtype)
            else:
                tabs[nm] = (TAB_U[nm][idx_cpu].to(DEV, x.dtype) if nm in NG_STREAMS
                            else RAW[nm][idx_cpu].to(DEV, x.dtype))
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x * rms2)
            x = x + mlp_out
            streams[f'mlp{li}'] = mlp_out
            nm = f'mlp{li}'
            if use_bigram and nm in NG_STREAMS:
                g = TAB_U[nm][idx_cpu]
                hit = bif >= 0
                g[hit] = TAB_B[nm][bif[hit]]
                tabs[nm] = g.to(DEV, x.dtype)
            else:
                tabs[nm] = (TAB_U[nm][idx_cpu].to(DEV, x.dtype) if nm in NG_STREAMS
                            else RAW[nm][idx_cpu].to(DEV, x.dtype))
        xf = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


tot, n = 0.0, 0
with torch.no_grad():
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        lg = None
        idx = b[:, :-1]
        B, T = idx.shape
        x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        xf = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel(); n += b[:, 1:].numel()
base = tot / n
res = {'baseline': base, 'r2': r2, 'bigram_coverage': cov,
       'n_bigrams': len(uc), 'arms': {}}
print(f'baseline {base:.4f}', flush=True)
for W in (1, 2):
    for ub in (False, True):
        name = f'W={W} {"bigram-backoff bottom4" if ub else "unigram (ref)"}'
        d = audit_ce(W, ub) - base
        res['arms'][name] = round(d, 4)
        print(f'{name}: dCE {d:+.4f}', flush=True)
        with open(OUT, 'w') as fh:
            json.dump(res, fh, indent=2)
torch.save({'uc': uc, 'tab_b': {nm: t.half() for nm, t in TAB_B.items()},
            'tab_u': {nm: t.half() for nm, t in TAB_U.items()}},
           f'{QK}/ngram_tables.pt')
print('ngram tables done', flush=True)
