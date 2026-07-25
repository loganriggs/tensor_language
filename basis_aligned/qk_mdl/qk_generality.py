"""TICK 220 (Logan): the GENERALITY SPECTRUM.

(A) Scalar-replaceability: replace each head's attention pattern with its data-mean
OFFSET KERNEL (attention as a fixed decay curve, zero content dependence; estimated
on 32 disjoint documents). r_h = dCE(kernel)/dCE(zeroed) on 64 held-out documents:
0 = the head is a positional scalar; 1 = fully content-dependent. All 18 heads,
both layers.
(B) Class-damage matrix: per-head zero-ablation per-position damage split across six
context classes (repeat/induction, after-determiner, subword target, punctuation
target, near-boundary, capitalized target), enrichment vs class base rate;
hierarchically clustered -> is the ensemble organized universal -> broad -> specific?

Derived from tick 217; original header:
"""
"""TICK 217 (Logan unblocked L3): PRUNED-SLIVER COMPOSITION — how little of block 0
must actually RUN to produce layer-1's pattern context?

The lookup nulls (ticks 214-215) proved the missing signal is composed across
positions; the missed links sit at offsets 0-2. So: compute layer-1's factors from a
REDUCED block-0 state — attention restricted to a causal window of W positions (the
sliver), full or skipped block-0 MLP — while the real forward pass elsewhere uses the
true state (route-only patching as always). Arms: W in {1, 2, 4, 8, 16} with MLP, and
W=4 without the block-0 MLP. Anchors: static tables +0.0515, best code generator
+0.032, exact state 0. The resulting curve measures the MINIMAL COMPOSITION -
fidelity trade: if W=4 with MLP lands near oracle, the "algorithm" needed for the
pattern route is a four-token window of block 0 — small even if unnamed.

Derived from tick 214; original header:
"""
"""TICK 214 (Logan): (b-fixed) per-head JOINT oracle repairs (tick-213 mode bug
fixed — asserts on the patch this time) + RESIDUAL-STAGE zoo: train second-stage
models ON THE RESIDUAL (true adapter coords minus frozen stage-1 prediction) with NEW
inputs — a local WINDOW of token-identity codes (embedding-PCA-32 at offsets 0..-3),
testing the tick-212/213 hypothesis that the missing computation is fine lexical
context of recent tokens. Arms: linear window-only; swiglu window-only; linear
window+stage-1 code (control: old inputs alone should add ~nothing).

Original tick-213 header:
"""
"""TICK 213 (Logan): CLASSIFY the missing signal and LOCALIZE the damaged
interactions.

(a) Cluster the worst-512 positions by residual DIRECTION (cosine k-means, k=6) —
Logan's attribution-similarity classification. Per cluster: size, dominant channels,
subword statistics, text snippets.
(b) PER-HEAD JOINT oracle repair: restore oracle corrections for ALL FOUR maps of one
layer-1 head at a time (generated elsewhere). Tick 212 showed per-map repairs backfire
(errors couple through the pattern product); per-head-joint repair is the coupling-
respecting granularity — its ranking names which head's missing context causes the
damage, testably.
(c) MISSED-LINK analysis: at the worst-200 positions, compare exact vs generated
layer-1 attention patterns; per head, the distribution of |dPattern| over key-offsets
and whether the damaged keys are subword fragments — which INTERACTIONS are missed.

Derived from tick 212; original header:
"""
"""TICK 212 (Logan): what is the generator MISSING? Error analysis of the best
generated interface (mixed swiglu, +0.0319), in the tick-164 tradition that found the
anchor structure.

On 64 held-out documents (route-only patching: generated corrections enter ONLY the
layer-1 QK factors; the residual carries the true MLP output for all other readers):

(a) Per-position dCE of the generated model vs exact. Worst-200 positions dumped with
    context snippets.
(b) Commonality statistics, worst-200 vs all: fraction where the TARGET is a subword
    continuation (no leading space, alphabetic); fraction within 3 tokens after a
    newline; fraction where the current token is a subword fragment; mean distance to
    previous newline.
(c) The MISSING RESIDUAL in adapter coordinates: R = (true deviation - generated
    correction) per channel (576 dims total). Worst-200 vs median positions: norm
    ratio, channel breakdown (which map x head carries the miss), and PCA of the
    worst-position residuals (is what's missing COMMON structure — low rank — or
    idiosyncratic?).
(d) REPAIR ATTRIBUTION: on the same 64 documents, replace generated with ORACLE
    corrections for one factor map at a time (q1/k1/q2/k2) — which route's missing
    context carries the CE cost.
(e) NAMING the interface: R^2 of each of the 16 oracle code dims (per the three most
    important channels) against cheap position features: log distance to previous
    newline, subword-continuation flags (current and previous token), position index,
    inside-quote parity. Is the ten-dimensional signal humanly simple?
"""
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
N_EST, TAU, N_CAP, R_AD, N_EVAL = 1024, 8.0, 256, 16, 64
tok = AutoTokenizer.from_pretrained('gpt2')

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
blk0 = m.transformer.h[0]
a1 = m.transformer.h[1].attn
MAPS = (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2))
SEQS = FINEWEB[:N_EVAL]


@torch.no_grad()
def block01(idx):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    x = blk0.lambdas[0] * x + blk0.lambdas[1] * x0
    a = blk0.attn
    hcur = F.rms_norm(x, (x.size(-1),))
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]

    def qk(lin):
        z = lin(hcur).view(B, T, NH, HD)
        return apply_rot(F.rms_norm(z, (HD,)), cosb, sinb)

    v = a.c_v(hcur).view(B, T, NH, HD)
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    q, k = qk(a.c_q), qk(a.c_k)
    q2, k2 = qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
    x = x + a.c_proj(yh.reshape(B, T, -1))
    mo = blk0.mlp(F.rms_norm(x, (x.size(-1),)))
    x = x + mo
    blk1 = m.transformer.h[1]
    return blk1.lambdas[0] * x + blk1.lambdas[1] * x0, mo, yh


# ---- tables (shrunk) ----
print('tables...', flush=True)
sum_x = torch.zeros(V, D, device=DEV)
cnt = torch.zeros(V, device=DEV)
with torch.no_grad():
    for i in range(0, N_EST, 4):
        idx = COOC[i:i + 4].to(DEV)[:, :-1]
        x, _, _ = block01(idx)
        sum_x.index_add_(0, idx.reshape(-1), x.float().reshape(-1, D))
        cnt.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=DEV))
wte = m.transformer.wte.weight.detach().float().to(DEV)
mean_x = torch.where((cnt > 0)[:, None], sum_x / cnt[:, None].clamp_min(1), wte)
shr = (cnt / (cnt + TAU))[:, None] * mean_x + (TAU / (cnt + TAU))[:, None] * wte
TABLES = {}
with torch.no_grad():
    xn = F.rms_norm(shr, (D,))
    for name, lin in MAPS:
        TABLES[name] = F.rms_norm(lin(xn).view(V, NH, HD).float(), (HD,)).contiguous()
del sum_x, mean_x, shr, xn
torch.cuda.empty_cache()

# ---- refit generator pieces exactly as tick 211 (train split of cooc) ----


from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
N_EVAL2 = 64
SEQS2 = FINEWEB[:N_EVAL2]
T1 = SEQS2.shape[1] - 1

# ---- offset kernels per (layer, head), estimated on 32 cooc docs ----
print('estimating offset kernels...', flush=True)
KER = {0: torch.zeros(NH, 512, device=DEV), 1: torch.zeros(NH, 512, device=DEV)}
KCNT = torch.zeros(512, device=DEV)


@torch.no_grad()
def run_patterns(idx):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
    pats = {}
    for li, blk in enumerate(m.transformer.h[:2]):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        hcur = F.rms_norm(x, (x.size(-1),))

        def qk(lin):
            z = lin(hcur).view(B, T, NH, HD)
            return apply_rot(F.rms_norm(z, (HD,)), cosb, sinb)

        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        q2, k2 = qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        pats[li] = pat
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    return pats


with torch.no_grad():
    for i in range(0, 32, 4):
        idx = COOC[i:i + 4].to(DEV)[:, :-1]
        pats = run_patterns(idx)
        B, T = idx.shape
        ar = torch.arange(T, device=DEV)
        off = (ar[:, None] - ar[None, :]).clamp(0, 511)
        valid = ar[:, None] >= ar[None, :]
        for li in (0, 1):
            p = pats[li]
            for h in range(NH):
                KER[li][h].scatter_add_(0, off[valid].reshape(-1),
                                        p[:, h][:, valid].mean(0).reshape(-1))
        KCNT.scatter_add_(0, off[valid].reshape(-1), torch.ones(int(valid.sum()), device=DEV))
for li in (0, 1):
    KER[li] = KER[li] / KCNT[None].clamp_min(1) * 8   # 8 batches of B=4 averaged via mean(0)
print('kernels ready', flush=True)


@torch.no_grad()
def per_pos_mod(target, mode, batch=4):
    """target=(layer, head); mode 'zero' or 'kernel'; None = exact."""
    outs = []
    for i in range(0, N_EVAL2, batch):
        b = SEQS2[i:i + batch].to(DEV)
        idx = b[:, :-1]
        dt = m.transformer.wte.weight.dtype
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        v1 = None
        B, T = idx.shape
        cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        ar = torch.arange(T, device=DEV)
        off = (ar[:, None] - ar[None, :]).clamp(0, 511)
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            hcur = F.rms_norm(x, (x.size(-1),))

            def qk(lin):
                z = lin(hcur).view(B, T, NH, HD)
                return apply_rot(F.rms_norm(z, (HD,)), cosb, sinb)

            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None:
                v1 = v
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            q, k = qk(a.c_q), qk(a.c_k)
            q2, k2 = qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            if target is not None and li == target[0]:
                hh = target[1]
                if mode == 'zero':
                    pat = pat.clone()
                    pat[:, hh] = 0
                else:
                    kern = KER[li][hh][off].masked_fill(~mask, 0.0)
                    pat = pat.clone()
                    pat[:, hh] = kern[None]
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
            x = x + a.c_proj(y)
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        x = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(x) / 30)
        ls = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1),
                             reduction='none')
        outs.append(ls.view(b.shape[0], -1).cpu())
    return torch.cat(outs, 0)


base = per_pos_mod(None, None)

# ---- context classes ----
flat_cur = SEQS2[:, :-1].reshape(-1)
flat_tgt = SEQS2[:, 1:].reshape(-1)
sub_flag = torch.zeros(V, dtype=torch.bool)
punc_flag = torch.zeros(V, dtype=torch.bool)
cap_flag = torch.zeros(V, dtype=torch.bool)
det_flag = torch.zeros(V, dtype=torch.bool)
for t in range(V):
    s = tok.decode([t])
    sub_flag[t] = (not s.startswith(' ')) and s[:1].isalpha()
    punc_flag[t] = len(s.strip()) > 0 and not any(c.isalnum() for c in s)
    cap_flag[t] = s.startswith(' ') and len(s) > 1 and s[1].isupper()
    det_flag[t] = s.strip().lower() in ('the', 'a', 'an', 'this', 'these', 'that')
rep = torch.zeros(len(flat_cur), dtype=torch.bool)
nearb = torch.zeros(len(flat_cur), dtype=torch.bool)
for si in range(N_EVAL2):
    row = SEQS2[si, :-1]
    seen = set()
    lastnl = -100
    for p in range(T1):
        t = int(row[p])
        gi = si * T1 + p
        rep[gi] = t in seen
        seen.add(t)
        if tok.decode([t]).startswith('\n'):
            lastnl = p
        nearb[gi] = (p - lastnl) <= 3
CLS = {'induction(repeat)': rep, 'after-det': det_flag[flat_cur],
       'subword-tgt': sub_flag[flat_tgt], 'punct-tgt': punc_flag[flat_tgt],
       'near-boundary': nearb, 'cap-tgt': cap_flag[flat_tgt]}
rates = {k: float(v.float().mean()) for k, v in CLS.items()}
print('class rates:', {k: round(v, 3) for k, v in rates.items()}, flush=True)

out = {'class_rates': rates}
MAT = {}
for li in (0, 1):
    for h in range(NH):
        dz = (per_pos_mod((li, h), 'zero') - base).flatten()
        dk = (per_pos_mod((li, h), 'kernel') - base).flatten()
        z, kk_ = float(dz.mean()), float(dk.mean())
        r = kk_ / z if z > 1e-5 else float('nan')
        pos = dz.clamp_min(0)
        tot = float(pos.sum()) or 1e-9
        enr = {c: round((float(pos[msk].sum()) / tot) / max(rates[c], 1e-6), 2)
               for c, msk in CLS.items()}
        MAT[f'l{li}h{h}'] = enr
        out[f'l{li}h{h}'] = {'zero_dce': round(z, 5), 'kernel_dce': round(kk_, 5),
                             'content_ratio': round(r, 3) if r == r else None,
                             'class_enrichment': enr}
        print(f'l{li}h{h}: zero {z:+.5f} kernel {kk_:+.5f} content-ratio '
              f'{r:.2f} | top class ' +
              max(enr, key=enr.get) + f' {max(enr.values()):.1f}x', flush=True)
        json.dump(out, open(f'{QK}/qk_generality.json', 'w'), indent=2)
print('GENERALITY DONE', flush=True)
