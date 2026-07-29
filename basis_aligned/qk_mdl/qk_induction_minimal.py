"""Induction minimal circuit v2 — proper sufficiency search + necessity dissection.
(1) Backward elimination: remove components least-important-first while induction advantage stays
    >= 90% of full. Remaining live set = minimal SUFFICIENT circuit under mean-ablation.
(2) Necessity: ablate ONLY the core candidates (MLP1; +h3.8,h2.5; induction head h5.5) to show
    which single removals break induction.
(3) Readout vs compute: keep the whole output stack (layers 13-17) as readout infrastructure and
    search the minimal COMPUTE set in layers 0-12 that drives induction through it.
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
P = 64; NSEQ = 48
pref = FINEWEB[:NSEQ, 1:1+P]
EV = torch.cat([pref, pref], 1).to(DEV)
SEC = torch.arange(P, 2*P-1, device=DEV); FIR = torch.arange(1, P-1, device=DEV)


@torch.no_grad()
def forward(keep=None, collect_mean=False):
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
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if collect_mean: means[('h', li)] = yh4.mean((0, 1))
        if keep is not None:
            for h in range(NH):
                if ('h', li, h) not in keep: yh4[:, :, h, :] = MEAN[('h', li)][h]
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if collect_mean: means[('m', li)] = mo.mean((0, 1))
        if keep is not None and ('m', li) not in keep: mo = MEAN[('m', li)].expand_as(mo)
        x = x + mo
    lg = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float(); tgt = EV[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction='none').view(B, T)
    return ce[:, FIR].mean().item() - ce[:, SEC].mean().item(), means

_, MEAN0 = forward(None, True); MEAN = MEAN0
ADV_FULL, _ = forward(None); ADV_NONE, _ = forward(set())
ALL = [('h', li, h) for li in range(NL) for h in range(NH)] + [('m', li) for li in range(NL)]
def ret(a): return (a - ADV_NONE) / (ADV_FULL - ADV_NONE)
THRESH = ADV_NONE + 0.90 * (ADV_FULL - ADV_NONE)
print(f"full {ADV_FULL:+.4f} none {ADV_NONE:+.4f} thresh(90%) {THRESH:+.4f}", flush=True)

# knockout importance at full for ordering
KO = {c: ADV_FULL - forward(set(ALL) - {c})[0] for c in ALL}
order = sorted(ALL, key=lambda c: KO[c])   # least important first

# (1) backward elimination
live = set(ALL)
for c in order:
    trial = live - {c}
    if forward(trial)[0] >= THRESH: live = trial
adv_live, _ = forward(live)
minimal = sorted(live, key=lambda c: (c[1], c[0]))
print(f"BACKWARD-ELIM minimal sufficient: {len(live)} comps, adv {adv_live:+.4f} ({ret(adv_live):.1%})", flush=True)
print("  kept:", [str(c) for c in minimal], flush=True)

# (2) necessity of the core
core_tests = {
    'ablate_only_MLP1': {('m', 1)},
    'ablate_only_h3.8': {('h', 3, 8)},
    'ablate_only_h2.5': {('h', 2, 5)},
    'ablate_only_indhead_h5.5': {('h', 5, 5)},
    'ablate_core_MLP1+h3.8+h2.5': {('m', 1), ('h', 3, 8), ('h', 2, 5)},
    'ablate_core+indhead': {('m', 1), ('h', 3, 8), ('h', 2, 5), ('h', 5, 5)},
}
nec = {}
for name, abl in core_tests.items():
    keep = set(ALL) - abl; a, _ = forward(keep); nec[name] = round(a, 4)
    print(f"  {name}: adv {a:+.4f} ({ret(a):.1%})", flush=True)

# (3) readout fixed (layers 13-17 all live), minimal compute set in layers 0-12
READOUT = {c for c in ALL if c[1] >= 13}
compute_pool = [c for c in ALL if c[1] <= 12]
live2 = READOUT | set(compute_pool)
adv_all2, _ = forward(live2)
order2 = sorted(compute_pool, key=lambda c: KO[c])
for c in order2:
    trial = live2 - {c}
    if forward(trial)[0] >= THRESH: live2 = trial
compute_min = sorted([c for c in live2 if c[1] <= 12], key=lambda c: (c[1], c[0]))
adv_live2, _ = forward(live2)
print(f"READOUT-FIXED: compute set {len(compute_min)} comps (layers<=12) + full readout; adv {adv_live2:+.4f} ({ret(adv_live2):.1%})", flush=True)
print("  compute kept:", [str(c) for c in compute_min], flush=True)

res = {'adv_full': round(ADV_FULL, 4), 'adv_none': round(ADV_NONE, 4),
       'minimal_sufficient_size': len(live), 'minimal_sufficient_adv': round(adv_live, 4),
       'minimal_components': [str(c) for c in minimal],
       'necessity': nec,
       'readout_fixed_compute_size': len(compute_min), 'readout_fixed_adv': round(adv_live2, 4),
       'readout_fixed_compute': [str(c) for c in compute_min],
       'top_knockout': [(str(c), round(KO[c], 4)) for c in sorted(ALL, key=lambda c: -KO[c])[:15]]}
json.dump(res, open(f'{QK}/qk_induction_minimal.json', 'w'), indent=2)
print("QK INDUCTION MINIMAL DONE", flush=True)
