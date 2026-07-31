"""DEPTH-FIRST ALGORITHM ARC #4: WHERE DOES THE ITERATED-SQUARING CASCADE END?

Established anchors: block 2's MrxMr damage is 84% removed by freezing block 3's mlp clean
(section 96, qk_arc_square.json downstream_routing: total 0.0288 -> freeze_only_mlp3 0.0047);
block 3's full-mlp damage is 83% removed by freezing block 4's mlp clean (section 98,
qk_arc_mlp3.json H1_freeze_patch_routing: total 0.6163 -> freeze_only_mlp4 0.1050).

THIS SCRIPT, for k in {4,5,6,7,8,10,12,14}: mean-ablate block k's mlp output to its
per-position held-set mean (the census floor intervention), then measure
  (a) total_damage        -- FLOOR GATE vs qk_allterm_census.json floor_dCE to three decimals
  (b) freeze_next (k+1)   -- ablate + freeze ONLY block k+1's mlp to its clean cached value;
                             NEXT-SQUARE FRACTION = (total - this)/total
  (c) direct_only         -- ablate + freeze ALL later attn+mlp clean;
                             DIRECT-READOUT FRACTION = this/total
  (d) restore_at_(k+1)    -- exactness gate (~0: empty consumer set, coefficient 1)
      restore_at_(k+3)    -- damage attributable to consumers {k+1, k+2} (spread profile)

Machinery VERBATIM from qk_arc_mlp3_2.py: fwd_route (freeze-patch: replace component output
with clean cached value), restore-at-block-k with exact linear propagation coefficient
prod lambda0_(k+1..j-1) (residual recurrence x <- lambda0*x + lambda1*x0 is linear in the
deviation; x0 skip carries no block-k mlp content), paired standard errors, GPU guard.
Ablation = FULL mean-ablation of the block's mlp output (per-position held-set mean MO_MEAN,
identical to the census MEANF up to the decomposition gate < 1e-4).

Held FW[448:600,:128], batch 6. Writes qk_cascade_end.json."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_cascade_end.json'

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
KS = [4, 5, 6, 7, 8, 10, 12, 14]
LAM0 = [float(b.lambdas[0]) for b in m.transformer.h]
# deviation coefficient at entry of block j (before block j's lambda mix) = prod lambda0_(k+1..j-1)
CREST = {k: {j: float(np.prod(LAM0[k+1:j])) for j in range(k+1, NL)} for k in KS}
CENSUS = {int(li): rec['floor_dCE'] for li, rec in
          json.load(open(f'{QK}/qk_allterm_census.json'))['layers'].items()}
print(f"bilin18 NL={NL} D={D} held {S_}x{T_}; blocks {KS}", flush=True)

# =====================================================================================
# PASS 0: per-position held-set means MO_MEAN[k] of each target block's mlp output
# (the census floor intervention; MEANF == MO_MEAN up to the decomposition gate < 1e-4)
# =====================================================================================
@torch.no_grad()
def fwd_collect_means(idx, sums):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li in sums: sums[li] += mo.float().sum(0)
        x = x + mo

print("PASS 0: per-position mlp-output means for target blocks ...", flush=True)
sums = {k: torch.zeros(T_, D, device=DEV) for k in KS}
for i in range(0, S_, B0):
    fwd_collect_means(HELD[i:i+B0], sums)
MO_MEAN = {k: (sums[k]/S_) for k in KS}
del sums

# =====================================================================================
# forward with (i) mean-ablation of block k's mlp, (ii) later-component freeze-patching
# (VERBATIM qk_arc_mlp3_2.py fwd_route), (iii) restore-at-block-j with exact linear
# propagation coefficient (VERBATIM qk_arc_mlp3_2.py fwd mode='restore').
# config: dict(k=int, ablate=bool, restore_at=None|j, freeze_attn=set, freeze_mlp=set)
# =====================================================================================
@torch.no_grad()
def fwd_route(idx, config=None, cache=None, fill_cache=False, want_logits=False, dev_norm_out=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cfg_ = config or {}
    LI = cfg_.get('k'); fa = cfg_.get('freeze_attn', ()); fm = cfg_.get('freeze_mlp', ())
    rj = cfg_.get('restore_at'); dev = None
    for li in range(NL):
        blk = m.transformer.h[li]
        if rj is not None and li == rj and dev is not None:
            x = x + CREST[LI][rj]*dev
        x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1))
        if li in fa: aout = cache[li][0]
        x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li in fm: mo = cache[li][1]
        if LI is not None and li == LI:
            if dev_norm_out is not None:
                dev_norm_out.append((mo - MO_MEAN[LI][None]).norm(dim=-1).cpu())
            if cfg_.get('ablate'):
                dev = mo - MO_MEAN[LI].unsqueeze(0).to(x.dtype)
                mo = MO_MEAN[LI].unsqueeze(0).expand(B, -1, -1).to(x.dtype)
        if fill_cache: cache[li] = (aout, mo)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return (ce, logits) if want_logits else (ce, None)

print("BASE ...", flush=True)
base = torch.cat([fwd_route(HELD[i:i+B0])[0].cpu() for i in range(0, S_, B0)], 0)
print(f"base CE {float(base.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

# =====================================================================================
# CONFIG TABLE: 5 per block + one global sanity, all sharing one clean cache per batch
# =====================================================================================
cfgs = [('sanity_freeze_all_no_ablate',
         {'k': None, 'freeze_attn': set(range(NL)), 'freeze_mlp': set(range(NL))})]
for k in KS:
    LATER = list(range(k+1, NL)); AF, AM = set(LATER), set(LATER)
    cfgs += [
        (f'k{k}_total_damage',      {'k': k, 'ablate': True}),
        (f'k{k}_restore_at_{k+1}',  {'k': k, 'ablate': True, 'restore_at': k+1}),
        (f'k{k}_freeze_next_mlp{k+1}', {'k': k, 'ablate': True, 'freeze_mlp': {k+1}}),
        (f'k{k}_direct_only',       {'k': k, 'ablate': True, 'freeze_attn': AF, 'freeze_mlp': AM}),
        (f'k{k}_restore_at_{k+3}',  {'k': k, 'ablate': True, 'restore_at': k+3}),
    ]
print(f"{len(cfgs)} configs x {S_//B0 + (S_ % B0 > 0)} batches ...", flush=True)

acc = {name: [] for name, _ in cfgs}
t0 = time.time()
for i in range(0, S_, B0):
    idx = HELD[i:i+B0]
    cache = {}
    fwd_route(idx, fill_cache=True, cache=cache)
    for name, c in cfgs:
        ce, _ = fwd_route(idx, config=c, cache=cache)
        acc[name].append(ce.cpu())
    del cache
    if (i//B0) % 5 == 0:
        print(f"  batch {i//B0+1}/{(S_+B0-1)//B0} ({time.time()-t0:.0f}s)", flush=True)
print(f"forwards done ({time.time()-t0:.0f}s)", flush=True)

raw = {}
for name, _ in cfgs:
    mn, se = dstat(torch.cat(acc[name], 0))
    raw[name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:28s} dCE {mn:+.4f} +- {se:.5f}", flush=True)

# =====================================================================================
# GATES + per-block synthesis
# =====================================================================================
assert abs(raw['sanity_freeze_all_no_ablate']['dCE']) < 0.003, "freeze-all sanity FAILED"
blocks = {}
gates_ok = True
for k in KS:
    tot = raw[f'k{k}_total_damage']['dCE']
    floor_diff = abs(tot - CENSUS[k])
    ok = floor_diff < 0.002
    gates_ok &= ok
    ex = raw[f'k{k}_restore_at_{k+1}']['dCE']
    frz = raw[f'k{k}_freeze_next_mlp{k+1}']['dCE']
    dro = raw[f'k{k}_direct_only']['dCE']
    r3 = raw[f'k{k}_restore_at_{k+3}']['dCE']
    blocks[k] = {
        'census_floor': CENSUS[k],
        'total_damage_dCE': tot, 'total_damage_SE': raw[f'k{k}_total_damage']['SE'],
        'floor_gate_abs_diff': round(floor_diff, 4), 'floor_gate_pass': ok,
        'exactness_restore_next_dCE': ex,
        'freeze_next_mlp_dCE': frz, 'freeze_next_mlp_SE': raw[f'k{k}_freeze_next_mlp{k+1}']['SE'],
        'next_square_fraction_removed': round((tot - frz)/tot, 4) if tot else None,
        'direct_only_dCE': dro, 'direct_only_SE': raw[f'k{k}_direct_only']['SE'],
        'direct_readout_fraction': round(dro/tot, 4) if tot else None,
        'restore_at_kplus3_dCE': r3,
        'first_two_consumers_fraction': round(r3/tot, 4) if tot else None,
    }
    print(f"k={k:2d} floor {tot:+.4f} (census {CENSUS[k]:+.4f}, diff {floor_diff:.4f}, "
          f"{'PASS' if ok else 'FAIL'}) | next-square frac {(tot-frz)/tot:.3f} | "
          f"direct frac {dro/tot:.3f} | first-two-consumers frac {r3/tot:.3f} | "
          f"exact gate {ex:+.4f}", flush=True)
assert gates_ok, "FLOOR GATE FAILED for at least one block (see floor_gate_pass)"
for k in KS:
    assert abs(blocks[k]['exactness_restore_next_dCE']) < 0.003, f"exactness gate FAILED k={k}"
print("ALL GATES PASS: floors reproduce census to three decimals; restore_(k+1) ~ 0", flush=True)

res = {
 'meta': {
  'model': 'bilin18', 'held': 'FW[448:600,:128]', 'batch': B0,
  'question': 'where does the iterated-squaring cascade end -- next-square mediation fraction, '
              'direct-readout fraction, and consumption spread vs block depth',
  'method': 'mean-ablate block k mlp output to its per-position held-set mean (census floor '
            'intervention); freeze-patch: replace later component outputs with clean cached '
            'values (VERBATIM qk_arc_mlp3_2.py fwd_route); restore-at-block-j: re-inject the '
            'deviation at entry of block j with exact linear coefficient prod lambda0_(k+1..j-1) '
            '(VERBATIM qk_arc_mlp3_2.py mode=restore); paired standard errors over held positions',
  'currency': 'delta cross-entropy per valid held position (nats)',
  'anchors': {'k2_MrxMr_next_square_fraction': 0.84, 'k3_full_mlp_next_square_fraction': 0.83,
              'source': 'qk_arc_square.json downstream_routing (section 96); '
                        'qk_arc_mlp3.json H1_freeze_patch_routing (section 98)'},
  'base_ce': round(float(base.mean()), 4),
  'lambda0_per_block': [round(l, 4) for l in LAM0]},
 'raw_configs': raw,
 'blocks': {str(k): blocks[k] for k in KS},
}
json.dump(res, open(OUT, 'w'), indent=1)
print(f"Saved {OUT}", flush=True)
print("QK CASCADE END (script 1) DONE", flush=True)
