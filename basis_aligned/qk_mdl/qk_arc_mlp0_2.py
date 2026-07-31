"""ALGORITHM ARC #2, part 2 (H3): WHERE DOES mlp.L0's CONTRIBUTION FLOW -- direct readout or
via later layers (the category engine, blocks 1-3)?

DOWNSTREAM-FLOW DECOMPOSITION. Mean-ablate mlp.L0 (mo0 -> per-position decomposition mean MEANF
from part 1, the census floor intervention) but keep the true deviation dev0 = mo0 - MEANF, and
RESTORE it into the residual stream at the entry of block k (with its exact linear propagation
coefficient prod_{j=1..k-1} lambda0_j -- the deviation's coefficient in a clean run, since the
residual recurrence x <- lambda0*x + lambda1*x0 is linear and the x0 skip carries no mlp0 content).
  restore_1        == no ablation (exactness gate: delta cross-entropy must be ~0)
  restore_k        == only blocks 1..k-1 saw the ablated stream => dCE(restore_k) = damage
                      attributable to mlp0's consumers among blocks 1..k-1
  readout_only     == restore only at the final readout => all mediated damage, direct path kept
  remove_direct    == clean run, subtract the direct linear path at readout => direct-only damage
  ablate           == never restore (must reproduce the census floor 1.2341)
Forward skeleton verbatim from qk_allterm_census.py / part 1. Held FW[448:600,:128], paired
standard errors, batch 6.

CATEGORY-PROBE ANGLE (adapted from qk_category_engine.py, probe + labels verbatim): ridge probe
for the NEXT token's 6-way category (subword/punct/capital/digit/funcword/other) on the residual
after block 3 and block 4, intact vs mlp.L0 mean-ablated. If the category code collapses toward
the embedding-level baseline, mlp.L0 feeds the category engine.
Appends 'H3_downstream' to qk_arc_mlp0.json."""
import json, os, sys, time, subprocess, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_arc_mlp0.json'

def gpu_guard(min_free=4500, tries=45, sleep=20):
    for _ in range(tries):
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        ).decode().split('\n')[0].strip())
        if free >= min_free:
            print(f"GPU guard: {free} MiB free -- proceeding.", flush=True); return
        print(f"GPU guard: only {free} MiB free (<{min_free}); sleeping {sleep}s ...", flush=True)
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out waiting for free memory")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128].to(DEV); B0 = 6
S_, T_ = HELD.shape
MEANF = torch.load(f'{QK}/qk_arc_mlp0_means.pt', map_location=DEV)['MEANF']   # (T,D) per-position mean of mo0
LAM0 = [float(b.lambdas[0]) for b in m.transformer.h]
# deviation coefficient at entry of block k (before block k's lambda mix) = prod_{j=1..k-1} lam0_j
CREST = {k: float(np.prod(LAM0[1:k])) for k in range(1, NL+1)}   # CREST[NL] = readout coefficient
print(f"bilin18 held {S_}x{T_}; direct-path coefficient to readout = {CREST[NL]:.3e}", flush=True)

@torch.no_grad()
def fwd(idx, mode='full', k=None):
    """mode: 'full' | 'ablate' | 'restore' (at entry of block k, 1<=k<=17) | 'readout_only'
    | 'remove_direct'. Skeleton verbatim from qk_allterm_census.py."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    dev0 = None
    for li in range(NL):
        blk = m.transformer.h[li]
        if mode == 'restore' and li == k and dev0 is not None:
            x = x + CREST[k]*dev0
        x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, kk_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, kk_)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == 0 and mode != 'full':
            dev0 = mo - MEANF.unsqueeze(0)
            if mode in ('ablate', 'restore', 'readout_only'):
                mo = MEANF.unsqueeze(0).expand(B, -1, -1).to(x.dtype)
        x = x + mo
    if mode == 'readout_only': x = x + CREST[NL]*dev0
    if mode == 'remove_direct': x = x - CREST[NL]*dev0
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                           idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)

print("BASE ...", flush=True)
base = torch.cat([fwd(HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
print(f"base CE {float(base.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

def run(mode, k=None):
    return torch.cat([fwd(HELD[i:i+B0], mode=mode, k=k).cpu() for i in range(0, S_, B0)], 0)

flow = {}
cfgs = [('ablate', 'ablate', None), ('remove_direct', 'remove_direct', None),
        ('readout_only', 'readout_only', None)]
cfgs += [(f'restore_at_block_{k}', 'restore', k) for k in (1, 2, 3, 4, 5, 6, 9, 12, 15)]
for name, mode, k in cfgs:
    mn, se = dstat(run(mode, k))
    flow[name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:20s} dCE {mn:+.4f} +- {se:.5f}", flush=True)

CENSUS_FLOOR = 1.2341
assert abs(flow['restore_at_block_1']['dCE']) < 0.003, "exactness gate FAILED: restore_1 not ~0"
assert abs(flow['ablate']['dCE'] - CENSUS_FLOOR) < 0.03, "floor gate FAILED vs census 1.2341"
print("gates pass: restore_1 ~ 0, ablate ~ census floor", flush=True)

# =====================================================================================
# CATEGORY PROBE (adapted verbatim from qk_category_engine.py; ablate = {0} with MEANF)
# =====================================================================================
tok = AutoTokenizer.from_pretrained('gpt2')
import string as _string
_P = set(_string.punctuation)
FUNC = {'the','of','and','to','a','in','is','that','it','for','was','as','with','on','be','at','by','this','are','from','or','an','but','not','which'}
CAT = torch.full((V,), 5, dtype=torch.long)
for i in range(50257):
    s = tok.convert_ids_to_tokens(i)
    if s is None: continue
    core = s.replace('Ġ', ''); lead = s.startswith('Ġ')
    if len(core) and all(c in _P for c in core): CAT[i] = 1
    elif len(core) and all(c.isdigit() for c in core): CAT[i] = 3
    elif core.lower() in FUNC: CAT[i] = 4
    elif not lead and len(core) and core[0].isalpha() and core[0].islower(): CAT[i] = 0
    elif lead and len(core) and core[0].isupper(): CAT[i] = 2
CAT = CAT.to(DEV)
DEPTHS = ['embed', 'blk3', 'blk4']

@torch.no_grad()
def collect(idx, ablate0=False):
    B, T = idx.shape; x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool)); out = {'embed': x.reshape(-1, D)}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if ablate0 and li == 0: mo = MEANF[:T].unsqueeze(0).expand(B, -1, -1).to(x.dtype)
        x = x + mo
        if f'blk{li}' in DEPTHS: out[f'blk{li}'] = x.reshape(-1, D)
        if li >= 4: break
    return out

def gather(rng, ablate0=False):
    R = {d: [] for d in DEPTHS}; Y = []
    for i in rng:
        idx = FW[i:i+4, :128].to(DEV)
        res = collect(idx[:, :-1], ablate0)
        lab = CAT[idx[:, 1:].reshape(-1)]
        for d in DEPTHS: R[d].append(res[d])
        Y.append(lab)
    return {d: torch.cat(R[d]) for d in DEPTHS}, torch.cat(Y)

print("CATEGORY PROBE: gathering residuals (intact + mlp.L0-ablated) ...", flush=True)
Rtr, Ytr = gather(range(0, 240, 4))
Rte, Yte = gather(range(300, 400, 4))
Rtr_a, _ = gather(range(0, 240, 4), ablate0=True)
Rte_a, _ = gather(range(300, 400, 4), ablate0=True)
Yoh = F.one_hot(Ytr, 6).float()
maj = torch.bincount(Yte, minlength=6).max().item() / Yte.numel()

def probe_acc(Xtr, Xte):
    Xtr = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1, device=DEV)], 1).double()
    Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1, device=DEV)], 1).double()
    Wp = torch.linalg.solve(Xtr.T @ Xtr + 50*torch.eye(Xtr.shape[1], device=DEV, dtype=torch.double), Xtr.T @ Yoh.double())
    pred = (Xte @ Wp).argmax(1)
    return float((pred == Yte).float().mean())

probe = {'majority': round(maj, 4)}
for d in DEPTHS:
    probe[f'{d}_intact'] = round(probe_acc(Rtr[d], Rte[d]), 4)
    if d != 'embed':
        probe[f'{d}_mlp0_ablated'] = round(probe_acc(Rtr_a[d], Rte_a[d]), 4)
    print(f"  probe {d}: intact {probe[f'{d}_intact']}"
          + (f" | mlp0-ablated {probe[f'{d}_mlp0_ablated']}" if d != 'embed' else ''), flush=True)

res = json.load(open(OUT))
res['H3_downstream'] = {
 'method': 'mean-ablate mlp.L0 to the per-position decomposition mean; restore the true deviation '
           'at the entry of block k with its exact linear propagation coefficient '
           'prod lambda0_(1..k-1); restore_1 == no ablation (exactness gate), readout_only == all '
           'mediated damage with direct path kept, remove_direct == direct-only damage, '
           'ablate == census floor',
 'direct_path_coefficient_to_readout': CREST[NL],
 'lambda0_per_block': [round(l, 4) for l in LAM0],
 'flow': flow,
 'census_floor_ref': CENSUS_FLOOR,
 'category_probe': dict(probe,
    ref_category_engine='qk_category_engine.json (probe machinery source; there blk4 intact 0.6107 '
                        'vs mlp0-3 all ablated 0.5097, global-mean ablation)',
    labels='6-way next-token category: subword/punct/capital/digit/funcword/other',
    train='FW[0:240:4] x4 seqs', test='FW[300:400:4] x4 seqs',
    note='ablation here = mlp.L0 only, per-position mean MEANF (part-1 decomposition mean)')}
json.dump(res, open(OUT, 'w'), indent=1)
print("Saved H3 to qk_arc_mlp0.json", flush=True)
print("QK ARC MLP0 PART2 DONE", flush=True)
