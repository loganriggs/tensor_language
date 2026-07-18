"""TOTAL-SYSTEM AUDIT: windowed-D (W=6; qk+v reads everywhere, mlp reads
L1-12; COMBO tables = r32 basis + vq1024 coefs) COMBINED with 3%-density
block-sparse rulebooks (top-2048 class-pair blocks/head) at ALL 18 layers.
The one number the consolidated MDL accounting should headline — and a
superadditivity test between the two reduction families.
Original: Windowed-D extended to ALL reads: q,k (selection), v (content), and the
MLP input each read the residual; replace streams older than W layers with
their cond-mean tables in each read independently. If this composes, the
model's ENTIRE long-range information flow is token-static — only local
(W-layer) computation runs live. Arms build up read-by-read.
Note v is carriage: the old-stream replacement preserves token identity (cond-mean
BY TOKEN), so this tests whether long-range carriage is also 0th-order."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/combined_final.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 6144, seq_len=513)
AUDIT_EARLY = ALL[4:20]
AUDIT_LATE = ALL[-16:]
EST_LATE = ALL[1044:2068]



CLS = torch.load(f'{QK}/ngram2_pairclass.pt')['cls']
RAW = torch.load(f'{QK}/stream_tables.pt')
RAW = {nm: torch.nan_to_num(t.float(), posinf=65504.0, neginf=-65504.0) for nm, t in RAW.items()}

# combo tables: r=32 basis + vq1024 coefficients (as e4b champion)
def kmeans(X, k, seed):
    g = torch.Generator(); g.manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k].to(X.device)].clone()
    for _ in range(8):
        a_ = torch.empty(len(X), dtype=torch.long, device=X.device)
        for i in range(0, len(X), 4096):
            xx = X[i:i + 4096]
            a_[i:i + 4096] = ((xx*xx).sum(1,True) - 2*xx@C.T + (C*C).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C); c2 = torch.zeros(k, device=X.device)
        Cn.index_add_(0, a_, X); c2.index_add_(0, a_, torch.ones(len(X), device=X.device))
        nz = c2 > 0; C[nz] = Cn[nz]/c2[nz][:,None]
    return a_, C

TABLES = {}
for nm, t in RAW.items():
    X = t.double()
    U_, S_, Vh = torch.svd_lowrank(X, q=32, niter=4)
    coef = (U_ * S_).float().to(DEV)
    aa, C = kmeans(coef, 1024, seed=hash(nm) % 2**31)
    TABLES[nm] = (C[aa].cpu().double() @ Vh.T).half()
    torch.cuda.empty_cache()
print('combo tables built', flush=True)

# per-layer block energies in ONE pass, then keep-masks
NL = 18
energy = torch.zeros(NL, NH, 256, 256, device=DEV)
CLS_DEV = CLS.to(DEV)
with torch.no_grad():
    for i in range(0, len(AUDIT_EARLY), 4):
        b = AUDIT_EARLY[i:i + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        cls_pos = CLS_DEV[idx]
        code = cls_pos[:, :, None] * 256 + cls_pos[:, None, :]
        x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        tri = mask
        codef = code[:, tri].reshape(-1)
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
            for hh in range(NH):
                pf = pat[:, hh][:, tri].reshape(-1).float()
                energy[li, hh].view(-1).index_add_(0, codef, pf * pf)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
            x = x + a.c_proj(y)
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
KEEP = torch.zeros(NL, NH, 256, 256, dtype=torch.bool, device=DEV)
for li in range(NL):
    for hh in range(NH):
        KEEP[li, hh].view(-1)[energy[li, hh].view(-1).topk(2048).indices] = True
print('block masks built', flush=True)


@torch.no_grad()
def audit_total(W=6, use_blocks=True, use_tables=True):
    MLPL = set(range(1, 13))
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT_EARLY), 4):
        b = AUDIT_EARLY[i:i + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        idx_cpu = idx.cpu()
        cls_pos = CLS_DEV[idx]
        x = m.transformer.wte(idx); x = F.rms_norm(x, (x.size(-1),))
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
            h_qk = h_v = h
            if use_tables and li >= 1:
                old = [nm for nm in streams if created_layer(nm) < li - W]
                if old:
                    xp = x
                    for nm in old:
                        xp = xp - streams[nm] + tabs[nm]
                    hp = F.rms_norm(xp, (xp.size(-1),))
                    h_qk = hp; h_v = hp
            qn = lambda lin: apply_rot(F.rms_norm(lin(h_qk).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            v = a.c_v(h_v).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            if use_blocks:
                kq = cls_pos[:, :, None].expand(B, T, T)
                kk = cls_pos[:, None, :].expand(B, T, T)
                for hh in range(NH):
                    kmh = KEEP[li, hh][kq.reshape(-1), kk.reshape(-1)].view(B, T, T)
                    pat[:, hh] = pat[:, hh] * kmh
            attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + attn_out
            streams[f'attn{li}'] = attn_out
            tabs[f'attn{li}'] = TABLES[f'attn{li}'][idx_cpu].to(DEV, x.dtype)
            x_mlp = x
            if use_tables and li in MLPL:
                for nm in [nm for nm in streams if created_layer(nm) < li - W]:
                    x_mlp = x_mlp - streams[nm] + tabs[nm]
            rms2 = x_mlp.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x_mlp * rms2)
            x = x + mlp_out
            streams[f'mlp{li}'] = mlp_out
            tabs[f'mlp{li}'] = TABLES[f'mlp{li}'][idx_cpu].to(DEV, x.dtype)
        xf = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


base = audit_total(use_blocks=False, use_tables=False)
res = {'baseline': base, 'arms': {}}
print(f'baseline {base:.4f}', flush=True)
for name, kw in [('rulebooks only (3% blocks, all layers)', dict(use_blocks=True, use_tables=False)),
                 ('windowed combo tables only (W=6)', dict(use_blocks=False, use_tables=True)),
                 ('TOTAL SYSTEM (both)', dict(use_blocks=True, use_tables=True))]:
    d = audit_total(**kw) - base
    res['arms'][name] = round(d, 4)
    print(f'{name}: dCE {d:+.4f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('combined final done', flush=True)
