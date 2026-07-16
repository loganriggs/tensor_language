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
OUT = f'{QK}/d_composed4.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 6144, seq_len=513)
AUDIT_EARLY = ALL[4:20]
AUDIT_LATE = ALL[-16:]
EST_LATE = ALL[1044:2068]


@torch.no_grad()
def estimate(train):
    acc, cnt = {}, torch.zeros(V)
    for i in range(0, len(train), 8):
        idx = train[i:i + 8, :-1].to(DEV)
        B, T = idx.shape
        flat = idx.reshape(-1).cpu()
        cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
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
            attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + attn_out
            acc.setdefault(f'attn{li}', torch.zeros(V, D)).index_add_(
                0, flat, attn_out.reshape(-1, D).float().cpu())
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x * rms2)
            x = x + mlp_out
            acc.setdefault(f'mlp{li}', torch.zeros(V, D)).index_add_(
                0, flat, mlp_out.reshape(-1, D).float().cpu())
    seen = cnt > 0
    tabs = {}
    for nm, a in acc.items():
        t = a / cnt.clamp_min(1)[:, None]
        t[~seen] = a.sum(0) / cnt.sum()
        tabs[nm] = t.half()
    return tabs


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


res = {}
early_tabs = torch.load(f'{QK}/stream_tables.pt')
base_late = audit_ce(AUDIT_LATE, early_tabs, W=None)
d_a = audit_ce(AUDIT_LATE, early_tabs, W=4) - base_late
res['late_baseline_ce'] = base_late
res['A: early tables, LATE audit, W=4'] = d_a
print(f'late baseline {base_late:.4f}; A (cross-region W=4): dCE {d_a:+.4f}', flush=True)
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2)

late_tabs = estimate(EST_LATE)
base_early = audit_ce(AUDIT_EARLY, late_tabs, W=None)
d_b = audit_ce(AUDIT_EARLY, late_tabs, W=4) - base_early
res['early_baseline_ce'] = base_early
res['B: late tables (same size), EARLY audit, W=4'] = d_b
print(f'early baseline {base_early:.4f}; B (late-est W=4): dCE {d_b:+.4f}', flush=True)
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2)
print('d composed 4 done', flush=True)
