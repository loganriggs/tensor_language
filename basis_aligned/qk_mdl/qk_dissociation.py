"""TICK 219 (Logan): DOUBLE DISSOCIATION for a named subsection of sub-circuit A.

Target: layer-1 head 7's determiner channel (its validated archetypes {The...} and
{a/an...}). Conditions, all with layer-1 pattern = static tables (route-only):
  T0: full tables                       (reference: sub-circuit A intact)
  T1: h7 keys MINUS det-channel         (necessity: only this subsection removed)
  T2: h7 keys det-channel ONLY          (sufficiency: only this subsection kept)
  T3: h7 zeroed                         (calibration: the whole head gone)
Metric: per-position dCE on 128 held-out documents, split by position class —
DET positions (current token in the archetypes' own top-token set: the/The/a/an/this)
versus all others. Predictions: T1 damages DET positions selectively; T2 preserves
DET positions near T0 while matching T3 on the head's other business.

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


# ---- archetype detector directions for l1-h7 (dets: archetypes 1={The}, 2={an/a}, 4={this}) ----
s1b = torch.load(f'{QK}/qk_l1_stage1.pt', map_location=DEV)
s23 = torch.load(f'{QK}/qk_l1_stage23.pt', map_location=DEV)
Dn7 = s1b['h7_Dn'].to(DEV)
Dn7 = Dn7 / Dn7.norm(dim=1, keepdim=True).clamp_min(1e-8)
U7 = s23['h7_U'].to(DEV)
DET_RS = [1, 2, 4]
G1 = torch.linalg.qr(torch.stack([Dn7[:, :HD].T @ U7[:, r] for r in DET_RS], 1)).Q
G2 = torch.linalg.qr(torch.stack([Dn7[:, HD:2 * HD].T @ U7[:, r] for r in DET_RS], 1)).Q

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('gpt2')
DET_TOKENS = set()
S7 = torch.zeros(V, Dn7.shape[0], device=DEV)
S7.scatter_(1, s1b['h7_idx'].long().to(DEV), s1b['h7_coeff'].to(DEV))
for r in DET_RS:
    load = S7 @ U7[:, r]
    for t in load.argsort(descending=True)[:8].tolist():
        DET_TOKENS.add(t)
print('DET tokens:', [tok.decode([t]) for t in sorted(DET_TOKENS)][:16], flush=True)
DET_MASK_V = torch.zeros(V, dtype=torch.bool)
for t in DET_TOKENS:
    DET_MASK_V[t] = True


def tables_variant(kind):
    T = {k: v.clone() for k, v in TABLES.items()}
    if kind == 'T0':
        return T
    for name, Gd in (('k1', G1), ('k2', G2)):
        col = T[name][:, 7]
        proj = (col @ Gd) @ Gd.T
        if kind == 'T1':
            T[name][:, 7] = col - proj
        elif kind == 'T2':
            T[name][:, 7] = proj
        elif kind == 'T3':
            T[name][:, 7] = 0
    if kind == 'T3':
        T['q1'] = T['q1'].clone(); T['q1'][:, 7] = 0
    return T


N_EVAL2 = 128
SEQS2 = FINEWEB[:N_EVAL2]


@torch.no_grad()
def per_pos_tables(T, batch=4):
    outs = []
    for i in range(0, N_EVAL2, batch):
        b = SEQS2[i:i + batch].to(DEV)
        idx = b[:, :-1]
        red = {}
        B, Tn = idx.shape
        for name in ('q1', 'k1', 'q2', 'k2'):
            red[name] = T[name][idx] if T is not None else None
        dt = m.transformer.wte.weight.dtype
        x = m.transformer.wte(idx)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        v1 = None
        cos, sin = rope_tables(Tn, HD, idx.device, dt, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(Tn, Tn, device=idx.device, dtype=torch.bool))
        for li, blk in enumerate(m.transformer.h):
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            a = blk.attn
            hcur = F.rms_norm(x, (x.size(-1),))

            def factors(lin, name=None):
                if li == 1 and T is not None and name is not None:
                    z = red[name].to(hcur.dtype)
                else:
                    z = F.rms_norm(lin(hcur).view(B, Tn, NH, HD), (HD,))
                return apply_rot(z, cosb, sinb)

            v = a.c_v(hcur).view(B, Tn, NH, HD)
            if v1 is None:
                v1 = v
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            q, k = factors(a.c_q, 'q1'), factors(a.c_k, 'k1')
            q2f, k2f = factors(a.c_q2, 'q2'), factors(a.c_k2, 'k2')
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2v = torch.einsum('bqhd,bkhd->bhqk', q2f, k2f) / HD
            pat = (s1 * s2v).masked_fill(~mask, 0.0)
            yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            x = x + a.c_proj(yh4.reshape(B, Tn, -1))
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
        x = F.rms_norm(x, (x.size(-1),))
        logits = 30 * torch.tanh(m.lm_head(x) / 30)
        ls = F.cross_entropy(logits.float().reshape(-1, V), b[:, 1:].reshape(-1),
                             reduction='none')
        outs.append(ls.view(b.shape[0], -1).cpu())
    return torch.cat(outs, 0)


base = per_pos_tables(None)
CUR = SEQS2[:, :-1].reshape(-1)
DETPOS = DET_MASK_V[CUR]
print(f'DET positions: {int(DETPOS.sum())} of {len(CUR)} '
      f'({float(DETPOS.float().mean())*100:.1f}%)', flush=True)
out = {'n_det_positions': int(DETPOS.sum())}
for kind in ('T0', 'T1', 'T2', 'T3'):
    lm_ = per_pos_tables(tables_variant(kind))
    d = (lm_ - base).flatten()
    row = {'overall': round(float(d.mean()), 5),
           'det_pos': round(float(d[DETPOS].mean()), 5),
           'other_pos': round(float(d[~DETPOS].mean()), 5)}
    out[kind] = row
    print(f'{kind}: overall {row["overall"]:+.5f} | det-positions {row["det_pos"]:+.5f} '
          f'| other {row["other_pos"]:+.5f}', flush=True)
    json.dump(out, open(f'{QK}/qk_dissociation.json', 'w'), indent=2)
print('DISSOCIATION DONE', flush=True)
