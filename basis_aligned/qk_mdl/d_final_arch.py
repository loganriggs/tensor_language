"""MLP-read failure localizer: composed MLP-read windowing costs +0.86 (W=4)
where qk+v cost +0.11. Is that (a) local MLP input fidelity, or (b) knock-on —
tabled MLP reads corrupt mlp_out, which is the next layer's dominant QK input
(SI-1)? Arms: single-layer MLP-read windowing (marginals) at L in {2,5,9,13,16};
bottom-only composed (L1-6); top-only composed (L7-17). Marginals tiny + composed
big => knock-on; marginals big at specific layers => local."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/d_final_arch.json'
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
            v = a.c_v(h_v).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + attn_out
            acc.setdefault(f'attn{li}', torch.zeros(V, D)).index_add_(
                0, flat, attn_out.reshape(-1, D).float().cpu())
            x_mlp = x
            if W is not None and 'mlp' in reads and li in MLP_LAYERS:
                old2 = [nm for nm in streams if created_layer(nm) < li - W]
                for nm in old2:
                    x_mlp = x_mlp - streams[nm] + tabs[nm]
            rms2 = x_mlp.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x_mlp * rms2)
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
def audit_ce(audit, tables, W=None, reads=('qk',)):
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
            h_qk = h_v = h
            if W is not None and li >= 1:
                old = [nm for nm in streams if created_layer(nm) < li - W]
                if old:
                    xp = x
                    for nm in old:
                        xp = xp - streams[nm] + tabs[nm]
                    hp = F.rms_norm(xp, (xp.size(-1),))
                    if 'qk' in reads:
                        h_qk = hp
                    if 'v' in reads:
                        h_v = hp
            qn = lambda lin: apply_rot(F.rms_norm(lin(h_qk).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            v = a.c_v(h_v).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + attn_out
            streams[f'attn{li}'] = attn_out
            tabs[f'attn{li}'] = tables[f'attn{li}'][idx_cpu].to(DEV, x.dtype)
            x_mlp = x
            if W is not None and 'mlp' in reads and li in MLP_LAYERS:
                old2 = [nm for nm in streams if created_layer(nm) < li - W]
                for nm in old2:
                    x_mlp = x_mlp - streams[nm] + tabs[nm]
            rms2 = x_mlp.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x_mlp * rms2)
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
tabs = torch.load(f'{QK}/stream_tables.pt')
MLP_LAYERS = set()
base = audit_ce(AUDIT_EARLY, tabs, W=None)
res['baseline_ce'] = base
print(f'baseline {base:.4f}', flush=True)
for arm, layers, Wa in (('qk+v all + mlp L1-12, W=6', set(range(1, 13)), 6),
                        ('qk+v all + mlp L1-12, W=4', set(range(1, 13)), 4),
                        ('qk+v all + mlp L1-15, W=6', set(range(1, 16)), 6)):
    globals()['MLP_LAYERS'] = layers
    d = audit_ce(AUDIT_EARLY, tabs, W=Wa, reads=('qk', 'v', 'mlp')) - base
    res[arm] = d
    print(f'{arm}: dCE {d:+.4f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('d final arch done', flush=True)
