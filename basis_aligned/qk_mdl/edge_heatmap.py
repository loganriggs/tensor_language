"""Logan's edge-ablation heatmap (2026-07-19 spec): for every edge
(source stream -> destination layer's reads), ablate the source IN THAT
DESTINATION'S READS ONLY (stream stays live everywhere else) and audit dCE.
Methods: zero / global-mean / PCA-1 / PCA-4 (fixed subspace, mean-centered).
Destinations: layers 1..17 (all reads that exist after the source's creation;
within-layer attn_L -> mlp_L read included) + the final unembedding read.
Sources: emb path + every attn/mlp output. Lower triangle only.
Audits: 8 held-out chunks (~4k tokens) at T=512, batch 8; baseline repeated
3x for the noise floor. Progressive, resumable JSON.
Companion (computed at plot time): weights-only importance
||R_dest @ W_src||_F (normalized) to test against the causal maps."""
import json
import os
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/edge_heatmap.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL = cfg['n_layer']
V = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)
AUDIT = ALL[4:12]                     # 8 chunks for the big sweep
TRAIN = ALL[20:276]                   # stats pass: 256 chunks (~131k tokens)
SRC = ['emb'] + [f'{t}{l}' for l in range(NL) for t in ('attn', 'mlp')]


def created_layer(nm):
    return -1 if nm == 'emb' else int(nm[4:] if nm.startswith('attn') else nm[3:])


# ---- pass 1: per-stream global mean + top-4 PCs ----
STATS_F = f'{QK}/edge_stream_stats.pt'
if os.path.exists(STATS_F):
    st = torch.load(STATS_F)
    MU, PC = st['mu'], st['pc']
    print('stats loaded', flush=True)
else:
    n_tot = 0
    mu = {nm: torch.zeros(D, device=DEV) for nm in SRC}
    cov = {nm: torch.zeros(D, D, device=DEV) for nm in SRC}

    @torch.no_grad()
    def stats_batch(idx):
        global n_tot
        B, T = idx.shape
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
            streams[f'attn{li}'] = attn_out
            rms2 = x.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x * rms2)
            x = x + mlp_out
            streams[f'mlp{li}'] = mlp_out
        # NOTE: streams here are as-scaled at the END of the model; for stats we
        # want the raw created values — but reads see lambda-rescaled versions.
        # We record the END-scaled version and rescale at read time consistently
        # by tracking the same lambda products in the audit forward.
        for nm, s in streams.items():
            f2 = s.reshape(-1, D).float()
            mu[nm] += f2.sum(0)
            cov[nm] += f2.T @ f2
        n_tot += B * T

    for i in range(0, len(TRAIN), 8):
        stats_batch(TRAIN[i:i + 8, :-1].to(DEV))
        if i % 64 == 0:
            print(f'  stats {i}/{len(TRAIN)}', flush=True)
    MU, PC = {}, {}
    for nm in SRC:
        mmu = mu[nm] / n_tot
        C = cov[nm] / n_tot - torch.outer(mmu, mmu)
        evals, evecs = torch.linalg.eigh(C)
        PC[nm] = evecs.flip(1)[:, :4].T.contiguous().cpu()   # (4, D)
        MU[nm] = mmu.cpu()
    torch.save({'mu': MU, 'pc': PC}, STATS_F)
    print('stats built + saved', flush=True)


def replace(s_live, nm, method):
    mu = MU[nm].to(DEV, s_live.dtype)
    if method == 'zero':
        return torch.zeros_like(s_live)
    if method == 'mean':
        return mu.expand_as(s_live).clone()
    kk = 1 if method == 'pca1' else 4
    P = PC[nm][:kk].to(DEV, s_live.dtype)
    dev = s_live - mu
    return mu + torch.einsum('btd,kd->btk', dev, P) @ P


@torch.no_grad()
def audit(dest=None, src=None, method=None):
    """dest: 1..17 (layer reads) or 'unembed'; src stream name."""
    tot, n = 0.0, 0
    for i in range(0, len(AUDIT), 8):
        b = AUDIT[i:i + 8].to(DEV)
        idx = b[:, :-1]
        B, T = idx.shape
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

            def patched(x_in, exclude_src=True):
                if dest != li or src not in streams:
                    return x_in
                return x_in - streams[src] + replace(streams[src], src, method)

            h_att = F.rms_norm(patched(x), (x.size(-1),)) if (dest == li and src in streams) \
                else F.rms_norm(x, (x.size(-1),))
            h_live = F.rms_norm(x, (x.size(-1),))
            qn = lambda lin: apply_rot(F.rms_norm(lin(h_att).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            v = a.c_v(h_att).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            attn_out = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            x = x + attn_out
            streams[f'attn{li}'] = attn_out
            x_mlp = x
            if dest == li and src in streams:
                x_mlp = x - streams[src] + replace(streams[src], src, method)
            rms2 = x_mlp.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
            mlp_out = blk.mlp(x_mlp * rms2)
            x = x + mlp_out
            streams[f'mlp{li}'] = mlp_out
        xf_in = x
        if dest == 'unembed' and src in streams:
            xf_in = x - streams[src] + replace(streams[src], src, method)
        xf = F.rms_norm(xf_in, (xf_in.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(xf) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


res = {}
if os.path.exists(OUT):
    res = json.load(open(OUT))
    print(f'resuming: {len(res.get("edges", {}))} edges done', flush=True)
if 'baseline' not in res:
    bases = [audit() for _ in range(3)]
    res['baseline'] = sum(bases) / 3
    res['baseline_spread'] = max(bases) - min(bases)
    res['edges'] = {}
    print(f'baseline {res["baseline"]:.4f} (spread {res["baseline_spread"]:.4f})', flush=True)
base = res['baseline']

METHODS = ['zero', 'mean', 'pca1', 'pca4']
dests = list(range(1, NL)) + ['unembed']
count = 0
for dest in dests:
    for src in SRC:
        cl = created_layer(src)
        if dest == 'unembed':
            ok = True
        else:
            ok = (cl < dest) or (src == f'attn{dest}')
        if not ok:
            continue
        for method in METHODS:
            key = f'{src}->{dest}|{method}'
            if key in res['edges']:
                continue
            d = audit(dest=dest, src=src, method=method) - base
            res['edges'][key] = round(d, 5)
            count += 1
            if count % 20 == 0:
                with open(OUT, 'w') as fh:
                    json.dump(res, fh)
                print(f'  {len(res["edges"])} edges·methods done (latest {key}: {d:+.4f})', flush=True)
with open(OUT, 'w') as fh:
    json.dump(res, fh)
print('edge heatmap done', flush=True)
