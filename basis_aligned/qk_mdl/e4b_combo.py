"""Method trio for table MDL (Logan 2026-07-20):
 (a) SHARED codebook across all 37 streams (per-stream RMS normalized; one
     global k-means; bits = k*d atoms + 37*V*log2(k) indices + 37 scales)
 (b) LOW-RANK factorization per table (V*r + r*d floats, rank r)
 (c) EDGE-GUIDED per-stream k (bits allocated by each stream's causal weight
     from the edge map; total budget ~= uniform vq1024)
All audited in the windowed-D W=4 harness; baseline = uniform vq1024 (+0.094).
Bits convention: structural bits + estimation tokens side by side."""
"""Region controls for D-composed (after 6x data WORSENED it, D-3):
 A. early-estimated tables (stream_tables.pt, 524k tokens, chunks 20-1044)
    audited on LATE chunks — does the +0.099 flagship number generalize
    across document regions?
 B. late-estimated tables (chunks 1044-2068, same 524k size) audited on the
    EARLY audit — isolates region-match from data amount.
Both at W=4, full tables."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT2 = f'{QK}/d_composed4.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 6144, seq_len=513)
AUDIT_EARLY = ALL[4:20]

def created_layer(nm):
    return int(nm[4:]) if nm.startswith('attn') else int(nm[3:])



@torch.no_grad()
def audit_ce(audit, tables, W=None):
    tot, n = 0.0, 0
    for i in range(0, len(audit), 4):
        b = audit[i:i + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        idx_cpu = idx.cpu()
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
            if W is not None and li >= 1:
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
            tabs[f'attn{li}'] = tables[f'attn{li}'][idx_cpu].to(DEV, x.dtype)
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x * rms2)
            x = x + mlp_out
            streams[f'mlp{li}'] = mlp_out
            tabs[f'mlp{li}'] = tables[f'mlp{li}'][idx_cpu].to(DEV, x.dtype)
        xf = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                             b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n



import math
RAW = torch.load(f'{QK}/stream_tables.pt')
# mlp17 has 621 fp16 overflows -> sanitize (clamp to fp16 max)
RAW = {nm: torch.nan_to_num(t.float(), nan=0.0, posinf=65504.0, neginf=-65504.0).half()
       for nm, t in RAW.items()}
NAMES = list(RAW.keys())
em = json.load(open(f'{QK}/edge_heatmap.json'))
# per-source causal weight from the edge map (sum |zero dCE| over dests)
imp = {nm: 0.0 for nm in NAMES}
for key, v in em['edges'].items():
    if key.endswith('|zero'):
        sn = key.split('->')[0]
        if sn in imp:
            imp[sn] += abs(v)

def kmeans(X, k, seed=0, fit_sub=200000):
    g = torch.Generator(); g.manual_seed(seed)
    if len(X) > fit_sub:
        Xf = X[torch.randperm(len(X), generator=g)[:fit_sub].to(X.device)]
    else:
        Xf = X
    C = Xf[torch.randperm(len(Xf), generator=g)[:k].to(X.device)].clone()
    for _ in range(8):
        a_ = torch.empty(len(Xf), dtype=torch.long, device=X.device)
        for i in range(0, len(Xf), 2048):
            xx = Xf[i:i + 2048]
            a_[i:i + 2048] = ((xx*xx).sum(1,True) - 2*xx@C.T + (C*C).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C); c2 = torch.zeros(k, device=X.device)
        Cn.index_add_(0, a_, Xf); c2.index_add_(0, a_, torch.ones(len(Xf), device=X.device))
        nz = c2 > 0; C[nz] = Cn[nz]/c2[nz][:,None]
    # assign ALL rows
    aa = torch.empty(len(X), dtype=torch.long, device=X.device)
    for i in range(0, len(X), 2048):
        xx = X[i:i + 2048]
        aa[i:i + 2048] = ((xx*xx).sum(1,True) - 2*xx@C.T + (C*C).sum(1)[None]).argmin(1)
    return aa, C

def vq_per_stream(kmap):
    out = {}
    for nm in NAMES:
        X = RAW[nm].float().to(DEV)
        aa, C = kmeans(X, kmap[nm], seed=hash(nm) % 2**31)
        out[nm] = C[aa].half().cpu()
        del X; torch.cuda.empty_cache()
    return out

base_ce = audit_ce(AUDIT_EARLY, {nm: t for nm, t in RAW.items()}, W=None)
res = {'baseline_ce': base_ce, 'arms': {}}
print(f'baseline {base_ce:.4f}', flush=True)

def bits_vq(kmap):
    atoms = sum(k*D for k in kmap.values())
    idx = sum(V*math.log2(k) for k in kmap.values())
    return atoms, idx

def run(name, tables, floats, idxbits):
    d = audit_ce(AUDIT_EARLY, tables, W=4) - base_ce
    res['arms'][name] = {'dce': round(d, 4), 'Mfloats': round(floats/1e6, 2),
                         'Mindexbits': round(idxbits/1e6, 1)}
    print(f'{name}: dCE {d:+.4f} | {floats/1e6:.1f}M floats + {idxbits/1e6:.0f}M idx bits', flush=True)
    with open(f'{QK}/e4_table_mdl.json', 'w') as fh:
        json.dump(res, fh, indent=2)

# baseline arm: uniform vq1024
kmap_u = {nm: 1024 for nm in NAMES}
run('uniform vq1024 (baseline)', vq_per_stream(kmap_u), *bits_vq(kmap_u))

# (b) low-rank
for r in (32, 128):
    tabs = {}
    for nm in NAMES:
        X = RAW[nm].double()          # CPU fp64: cusolver diverges on these
        U_, S_, Vh = torch.svd_lowrank(X, q=r, niter=4)
        tabs[nm] = ((U_ * S_) @ Vh.T).half()
    run(f'low-rank r={r}', tabs, len(NAMES)*(V*r + r*D), 0)

# (a) shared codebook
for ks in (4096, 8192):
    scales = {nm: RAW[nm].float().pow(2).mean().sqrt().clamp_min(1e-8) for nm in NAMES}
    XL = torch.cat([ (RAW[nm].float()/scales[nm]) for nm in NAMES]).to(DEV)
    aa, C = kmeans(XL, ks, seed=7)
    tabs = {}
    off = 0
    for nm in NAMES:
        tabs[nm] = (C[aa[off:off+V]].cpu() * scales[nm]).half()
        off += V
    del XL; torch.cuda.empty_cache()
    run(f'SHARED codebook k={ks}', tabs, ks*D + len(NAMES), len(NAMES)*V*math.log2(ks))

# (c) edge-guided k: tiers by causal weight, budget ~ uniform 1024
order = sorted(NAMES, key=lambda nm: -imp[nm])
kmap_e = {}
for i, nm in enumerate(order):
    kmap_e[nm] = 4096 if i < 8 else (1024 if i < 22 else 64)
run('edge-guided k (top8=4096, mid=1024, tail=64)', vq_per_stream(kmap_e), *bits_vq(kmap_e))

# combo: low-rank r=32 basis + vq1024 on the 32-dim coefficients
tabs = {}
for nm in NAMES:
    X = RAW[nm].double()
    U_, S_, Vh = torch.svd_lowrank(X, q=32, niter=4)
    coef = (U_ * S_).float().to(DEV)              # (V, 32)
    aa, C = kmeans(coef, 1024, seed=hash(nm) % 2**31)
    tabs[nm] = (C[aa].cpu().double() @ Vh.T).half()
    torch.cuda.empty_cache()
run('COMBO r=32 basis + vq1024 coefs', tabs,
    len(NAMES)*(32*D + 1024*32), len(NAMES)*V*10)
print('e4 combo done', flush=True)

