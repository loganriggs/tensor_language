"""Windowed-D transfer test on sqrd12 (162M, single QK branch, row-normalized
squared attention — the model that resisted score-space compression ~15x).
QK-reads windowed only, W in {2,4,6}. Original: Logan's method D composed, window form (guided by SI-1 + C-1): at EVERY
layer L>=1, the QK read (q,k inputs only; v/OV and the residual stay live)
replaces streams CREATED more than W layers back with their cond-mean-by-token
tables (estimated once at creation, rescaled analytically by the lambda
products); the embedding stream is exactly token-determined so it is free.
Streams inside the window are the PATCHED model's own live streams, so errors
can only chain W layers deep instead of 17. Arms: W=2, W=3, W=0 (all tabled —
composed control, expect wall-scale), plus per-arm sanity vs c_window.
No training anywhere."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/d_sqrd12.json'
m, cfg = load_elriggs('sqrd12')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL = cfg['n_layer']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]

# ---- one estimation pass: cond-mean of each stream AT CREATION ----
acc = {}
cnt = torch.zeros(V)
with torch.no_grad():
    for i in range(0, len(TRAIN), 8):
        idx = TRAIN[i:i + 8, :-1].to(DEV)
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
            q, k = qn(a.c_q), qn(a.c_k)
            v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            pat = s1.square().masked_fill(~mask, 0.0)
            pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
            attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + attn_out
            acc.setdefault(f'attn{li}', torch.zeros(V, D)).index_add_(
                0, flat, attn_out.reshape(-1, D).float().cpu())
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x * rms2)
            x = x + mlp_out
            acc.setdefault(f'mlp{li}', torch.zeros(V, D)).index_add_(
                0, flat, mlp_out.reshape(-1, D).float().cpu())
        if i % 256 == 0:
            print(f'  estimate {i}/{len(TRAIN)}', flush=True)
seen = cnt > 0
TABLES = {}
for nm, a in acc.items():
    t = a / cnt.clamp_min(1)[:, None]
    t[~seen] = a.sum(0) / cnt.sum()
    TABLES[nm] = t.half()          # fp16 storage; cast at gather
del acc
print('creation-time stream tables built', flush=True)


def created_layer(nm):
    return int(nm[4:]) if nm.startswith('attn') else int(nm[3:])


@torch.no_grad()
def audit_ce(W=None):
    """W: window size (streams created >= L-W stay live in layer-L's QK read);
    W=None -> fully live baseline."""
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 4):
        b = AUDIT[i:i + 4].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
        idx_cpu = idx.cpu()
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0, v1 = x, None
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        streams = {}     # live streams of the PATCHED model
        tabs = {}        # gathered tables, rescaled in lockstep
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
            q, k = qn(a.c_q), qn(a.c_k)
            v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            pat = s1.square().masked_fill(~mask, 0.0)
            pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
            attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + attn_out
            streams[f'attn{li}'] = attn_out
            tabs[f'attn{li}'] = TABLES[f'attn{li}'][idx_cpu].to(DEV, x.dtype)
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x * rms2)
            x = x + mlp_out
            streams[f'mlp{li}'] = mlp_out
            tabs[f'mlp{li}'] = TABLES[f'mlp{li}'][idx_cpu].to(DEV, x.dtype)
        xf = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                             b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


base = audit_ce()
print(f'baseline CE {base:.4f}', flush=True)
res = {'baseline_ce': base, 'arms': {}}
for W, name in ((6, 'W=6'), (4, 'W=4'), (2, 'W=2'), (0, 'W=0 control')):
    d = audit_ce(W=W) - base
    res['arms'][name] = d
    print(f'D-composed {name}: dCE {d:+.4f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('d sqrd12 done', flush=True)
