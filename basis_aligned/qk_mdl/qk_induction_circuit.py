"""End-to-end MINIMAL CIRCUIT for a specific task: INDUCTION (predict the token that followed
the last time this token appeared). Eval = real FineWeb 64-token prefixes repeated once (128-token
seq with an exact repeat). Induction advantage = CE(first copy) - CE(second copy) on the same
tokens: the second copy is predictable ONLY by copying from the first via induction.
Causal test (Logan): mean-ablate EVERY component (each head, each MLP) except a kept set; find the
minimal kept set that still computes induction. Phase 1 knockout-importance ranks components;
Phase 2 greedily builds the minimal sufficient circuit and verifies retention.
"""
import json, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))

# induction eval: 64-token real prefix, repeated -> 128 tokens
P = 64; NSEQ = 48
pref = FINEWEB[:NSEQ, 1:1+P]                          # real tokens
EV = torch.cat([pref, pref], 1).to(DEV)               # (NSEQ, 128) exact repeat
SEC = torch.arange(P, 2*P-1, device=DEV)              # second-copy predict positions (pred token at t+1)
FIR = torch.arange(1, P-1, device=DEV)                # first-copy positions (same tokens, no induction)


@torch.no_grad()
def forward(keep=None, collect_mean=False):
    """keep: set of ('h',li,h) / ('m',li) components to keep live; others mean-ablated.
    If keep is None: clean forward. collect_mean: also return per-component mean outputs."""
    idx = EV[:, :-1]; B, T = idx.shape
    dt = m.transformer.wte.weight.dtype; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, dt, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); means = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)  # (B,T,NH,HD)
        if collect_mean: means[('h', li)] = yh4.mean((0, 1))                                   # (NH,HD)
        if keep is not None:
            for h in range(NH):
                if ('h', li, h) not in keep: yh4[:, :, h, :] = MEAN[('h', li)][h]
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect_mean: means[('m', li)] = mo.mean((0, 1))                                    # (D,)
        if keep is not None and ('m', li) not in keep:
            mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()
    tgt = EV[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(B, T)
    adv = ce[:, FIR].mean().item() - ce[:, SEC].mean().item()   # induction advantage
    return adv, means

# clean pass: means + full advantage
_, MEANS0 = forward(keep=None, collect_mean=True)
MEAN = MEANS0
ADV_FULL, _ = forward(keep=None)
# also the fully-ablated floor (keep nothing)
ADV_NONE, _ = forward(keep=set())
print(f"induction advantage: full {ADV_FULL:+.4f} | fully-mean-ablated {ADV_NONE:+.4f}", flush=True)

ALL = [('h', li, h) for li in range(NL) for h in range(NH)] + [('m', li) for li in range(NL)]
# Phase 1: knockout importance — ablate ONE component from full, adv drop
KO = {}
for c in ALL:
    keep = set(ALL) - {c}
    adv, _ = forward(keep=keep); KO[c] = ADV_FULL - adv
rank = sorted(ALL, key=lambda c: -KO[c])
print("top-12 knockout importance (adv drop when this component removed from full):", flush=True)
for c in rank[:12]:
    print(f"  {c}: {KO[c]:+.4f}", flush=True)

# Phase 2: greedy build minimal sufficient circuit (add by knockout rank, verify)
keep = set(); curve = []
for c in rank:
    keep.add(c); adv, _ = forward(keep=keep); curve.append((str(c), round(adv, 4)))
    frac = (adv - ADV_NONE) / (ADV_FULL - ADV_NONE)
    if frac >= 0.90:
        print(f"minimal circuit reaches 90% at {len(keep)} components: adv {adv:+.4f} ({frac:.1%})", flush=True)
        break
minimal = sorted(keep, key=lambda c: (c[1], c[0]))
adv_min, _ = forward(keep=keep)
res = {'adv_full': round(ADV_FULL, 4), 'adv_none': round(ADV_NONE, 4),
       'minimal_size': len(keep), 'adv_minimal': round(adv_min, 4),
       'retention': round((adv_min - ADV_NONE) / (ADV_FULL - ADV_NONE), 4),
       'minimal_components': [str(c) for c in minimal],
       'top_knockout': [(str(c), round(KO[c], 4)) for c in rank[:20]],
       'greedy_curve': curve}
json.dump(res, open(f'{QK}/qk_induction_circuit.json', 'w'), indent=2)
print("MINIMAL INDUCTION CIRCUIT:", [str(c) for c in minimal], flush=True)
print("QK INDUCTION CIRCUIT DONE", flush=True)
