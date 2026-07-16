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
OUT = f'{QK}/d_sqrd12c.json'
m, cfg = load_elriggs('sqrd12')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL = cfg['n_layer']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]

TABLES = torch.load(f'{QK}/stream_tables_sqrd12.pt')


def created_layer(nm):
    return int(nm[4:]) if nm.startswith('attn') else int(nm[3:])


@torch.no_grad()
def audit_ce(W=None, reads=("qk",), mlp_layers=()):
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
            q, k = qn(a.c_q), qn(a.c_k)
            v = a.c_v(h_v).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            pat = s1.square().masked_fill(~mask, 0.0)
            pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)
            attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + attn_out
            streams[f'attn{li}'] = attn_out
            tabs[f'attn{li}'] = TABLES[f'attn{li}'][idx_cpu].to(DEV, x.dtype)
            x_mlp = x
            if W is not None and li in mlp_layers:
                for nm in [nm for nm in streams if created_layer(nm) < li - W]:
                    x_mlp = x_mlp - streams[nm] + tabs[nm]
            rms2 = x_mlp.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x_mlp * rms2)
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
NLYR = cfg['n_layer']
for name, reads, mlpl, W in (
        ('ALL reads W=3', ('qk', 'v'), tuple(range(1, NLYR)), 3),
        ('ALL reads W=2', ('qk', 'v'), tuple(range(1, NLYR)), 2),
        ('ALL reads W=4', ('qk', 'v'), tuple(range(1, NLYR)), 4)):
    d = audit_ce(W=W, reads=reads, mlp_layers=mlpl) - base
    res['arms'][name] = d
    print(f'D-sqrd12b {name}: dCE {d:+.4f}', flush=True)
    with open(OUT, 'w') as fh:
        json.dump(res, fh, indent=2)
print('d sqrd12c done', flush=True)
