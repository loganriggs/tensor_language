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

# ---- reduced block-0 state (windowed attention), side computation ----


@torch.no_grad()
def block1_input_windowed(idx, W, use_mlp=True):
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    B, T = idx.shape
    blk = m.transformer.h[0]
    x = blk.lambdas[0] * x + blk.lambdas[1] * x0
    a = blk.attn
    hcur = F.rms_norm(x, (x.size(-1),))
    cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]

    def qk(lin):
        z = lin(hcur).view(B, T, NH, HD)
        return apply_rot(F.rms_norm(z, (HD,)), cosb, sinb)

    v = a.c_v(hcur).view(B, T, NH, HD)
    ar = torch.arange(T, device=idx.device)
    mask = (ar[None, :] <= ar[:, None]) & (ar[None, :] > ar[:, None] - W)
    q, k = qk(a.c_q), qk(a.c_k)
    q2, k2 = qk(a.c_q2), qk(a.c_k2)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
    pat = (s1 * s2).masked_fill(~mask[None, None], 0.0)
    y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
    x = x + a.c_proj(y)
    if use_mlp:
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    blk1 = m.transformer.h[1]
    return blk1.lambdas[0] * x + blk1.lambdas[1] * x0


@torch.no_grad()
def audit_sliver(W, use_mlp=True, batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]
        xred = block1_input_windowed(idx, W, use_mlp)
        hred = F.rms_norm(xred, (D,))
        B, T = idx.shape
        red = {}
        for name, lin in MAPS:
            red[name] = F.rms_norm(lin(hred).view(B, T, NH, HD).float(), (HD,))
        dt = m.transformer.wte.weight.dtype
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        v1 = None
        cos, sin = rope_tables(T, HD, idx.device, dt, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            hcur = F.rms_norm(x, (x.size(-1),))

            def factors(lin, name=None):
                if li == 1 and name is not None:
                    z = red[name].to(hcur.dtype)
                else:
                    z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,))
                return apply_rot(z, cosb, sinb)

            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None:
                v1 = v
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            q, k = factors(a.c_q, 'q1'), factors(a.c_k, 'k1')
            q2f, k2f = factors(a.c_q2, 'q2'), factors(a.c_k2, 'k2')
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2v = torch.einsum('bqhd,bkhd->bhqk', q2f, k2f) / HD
            pat = (s1 * s2v).masked_fill(~mask, 0.0)
            yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            x = x + a.c_proj(yh4.reshape(B, T, -1))
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        x = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(x) / 30)
        ce = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


BASE = 3.07630
out = {}
for W in (1, 2, 4, 8, 16):
    ce = audit_sliver(W, True)
    out[f'W{W}_mlp_dce'] = round(ce - BASE, 5)
    print(f'sliver W={W} (with MLP): dCE {ce - BASE:+.5f} '
          f'(static +0.0515, best generator +0.032)', flush=True)
    json.dump(out, open(f'{QK}/qk_sliver.json', 'w'), indent=2)
ce = audit_sliver(4, False)
out['W4_nomlp_dce'] = round(ce - BASE, 5)
print(f'sliver W=4 (NO MLP): dCE {ce - BASE:+.5f}', flush=True)
json.dump(out, open(f'{QK}/qk_sliver.json', 'w'), indent=2)
print('SLIVER DONE', flush=True)
