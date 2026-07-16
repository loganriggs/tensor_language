"""Logan's method C, interventional: patch layer L's QK READ (q,k inputs only;
v and the residual stay fully live) with per-stream substitutions:
  i)  dominant channel: mlp(L-1) stream -> cond-mean-by-token table
  ii) window: every stream OLDER than L-2 -> cond-mean (recent + emb live)
  iii) all streams -> cond-mean (full 0th-order QK read; ~ depth-sweep check)
Layers tested: 2 (short-window regime), 5 (contextual layer), 9 (diffuse regime).
Stream cond-means estimated in one tracked pass over 524k tokens (streams l<=8)."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
TEST_L = [2, 5, 9]
MAX_STREAM_L = 8            # capture streams from layers 0..8
OUT = f'{QK}/c_window.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT, TRAIN = ALL[4:20], ALL[20:]

SNAMES = ['emb'] + [f'{t}{l}' for l in range(MAX_STREAM_L + 1) for t in ('attn', 'mlp')]


def track_streams(idx, upto):
    """live forward, returning (streams dict name->tensor, x at layer `upto`
    BEFORE the block's lambda mix, plus the lambda-mixed x and h)."""
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    streams = {'emb': x.clone()}
    v1 = None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        lam0, lam1 = blk.lambdas[0], blk.lambdas[1]
        x = lam0 * x + lam1 * x0
        for n in streams:
            streams[n] = lam0 * streams[n]
        streams['emb'] = streams['emb'] + lam1 * x0
        if li == upto:
            return streams, x
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
        if li <= MAX_STREAM_L:
            streams[f'attn{li}'] = attn_out
        rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
        mlp_out = blk.mlp(x * rms2)
        x = x + mlp_out
        if li <= MAX_STREAM_L:
            streams[f'mlp{li}'] = mlp_out
    raise RuntimeError('upto beyond depth')


# ---- estimate cond-means of every stream AS READ AT each test layer ----
# (streams get rescaled by lambda products per layer, so estimate at read time)
acc = {(L, n): torch.zeros(V, D) for L in TEST_L for n in SNAMES}
cnt = torch.zeros(V)
with torch.no_grad():
    for i in range(0, len(TRAIN), 8):
        idx = TRAIN[i:i + 8, :-1].to(DEV)
        flat = idx.reshape(-1).cpu()
        cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        for L in TEST_L:
            streams, _ = track_streams(idx, L)
            for n, s in streams.items():
                acc[(L, n)].index_add_(0, flat, s.reshape(-1, D).float().cpu())
        if i % 256 == 0:
            print(f'  estimate {i}/{len(TRAIN)}', flush=True)
seen = cnt > 0
tables = {}
for (L, n), a in acc.items():
    t = a / cnt.clamp_min(1)[:, None]
    t[~seen] = a.sum(0) / cnt.sum()
    tables[(L, n)] = t
del acc
print('stream cond-means built', flush=True)


@torch.no_grad()
def audit_ce(L=None, subs=None):
    """subs: list of stream names to replace by cond-mean at layer L's QK read."""
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
        streams = {'emb': x.clone()}
        for li, blk in enumerate(m.transformer.h):
            lam0, lam1 = blk.lambdas[0], blk.lambdas[1]
            x = lam0 * x + lam1 * x0
            for nm in streams:
                streams[nm] = lam0 * streams[nm]
            streams['emb'] = streams['emb'] + lam1 * x0
            a = blk.attn
            h = F.rms_norm(x, (x.size(-1),))
            h_qk = h
            if li == L and subs:
                xp = x.clone()
                for nm in subs:
                    xp = xp - streams[nm] + tables[(L, nm)][idx_cpu].to(DEV, x.dtype)
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
            if li <= MAX_STREAM_L:
                streams[f'attn{li}'] = attn_out
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x * rms2)
            x = x + mlp_out
            if li <= MAX_STREAM_L:
                streams[f'mlp{li}'] = mlp_out
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
for L in TEST_L:
    avail = [n for n in SNAMES if n == 'emb'
             or int(n[4:] if n.startswith('attn') else n[3:]) < L]
    arms = {
        f'L{L} i) mlp{L-1} stream tabled': [f'mlp{L-1}'],
        f'L{L} ii) window: all older than L-2 tabled':
            [n for n in avail if n != 'emb'
             and int(n[4:] if n.startswith('attn') else n[3:]) < L - 2],
        f'L{L} iii) ALL streams tabled': avail,
    }
    for name, subs in arms.items():
        d = audit_ce(L=L, subs=subs) - base
        res['arms'][name] = d
        print(f'{name}: dCE {d:+.4f}', flush=True)
        with open(OUT, 'w') as fh:
            json.dump(res, fh, indent=2)
print('c window done', flush=True)
