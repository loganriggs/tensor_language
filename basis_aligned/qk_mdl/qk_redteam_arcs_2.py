"""ADVERSARIAL RED-TEAM of the depth-first arc series (sections 96-100), script 2 of 2.

ATTACK 3 -- SMOOTHNESS of the flow map (section 99). The "no sharp end" claim rests on 8
measured blocks with k = 9, 11, 13 interpolated. Measure the missing blocks with the identical
qk_cascade_end.py battery (floor gate vs census, next-square fraction, direct-readout fraction,
first-two-consumers fraction) and compare each against the linear interpolation of its measured
neighbors (k = 8/10, 10/12, 12/14 from qk_cascade_end.json). Large deviation = structure the
coarse grid missed.

ATTACK 4 -- AGGREGATION on h.L7.0 (section 100). "Real in aggregate (global z 11.4), individually
insignificant" could hide a narrow circuit: check whether the global delta cross-entropy is
broadly spread or concentrated. Splits: (a) deciles of the head's own activation norm;
(b) per-sequence contributions -- share of the total carried by the top sequences, and the
global z recomputed with the top-5 / top-15 sequences removed; (c) sign spread -- fraction of
sequences with positive mean damage (binomial test); (d) position concentration -- share of the
net damage carried by the top 1 percent of positions.

Machinery VERBATIM from qk_cascade_end.py (attack 3, whole battery) and qk_arc_h70.py (attack 4,
forward + single-head mean-ablation + paired statistics). Held FW[448:600,:128], batch 6,
GPU guard. Appends attack3/attack4 into qk_redteam_arcs.json."""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import json, sys, time, math, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_redteam_arcs.json'

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
KS = [9, 11, 13]
LAM0 = [float(b.lambdas[0]) for b in m.transformer.h]
CREST = {k: {j: float(np.prod(LAM0[k+1:j])) for j in range(k+1, NL)} for k in KS}
CENSUS = {int(li): rec['floor_dCE'] for li, rec in
          json.load(open(f'{QK}/qk_allterm_census.json'))['layers'].items()}
print(f"bilin18 NL={NL} D={D} held {S_}x{T_}; ATTACK 3 blocks {KS}", flush=True)

# =====================================================================================
# ATTACK 3 -- VERBATIM qk_cascade_end.py battery at KS = [9, 11, 13]
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

@torch.no_grad()
def fwd_route(idx, config=None, cache=None, fill_cache=False):
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
        if LI is not None and li == LI and cfg_.get('ablate'):
            dev = mo - MO_MEAN[LI].unsqueeze(0).to(x.dtype)
            mo = MO_MEAN[LI].unsqueeze(0).expand(B, -1, -1).to(x.dtype)
        if fill_cache: cache[li] = (aout, mo)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce

print("BASE ...", flush=True)
base = torch.cat([fwd_route(HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
print(f"base CE {float(base.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

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
print(f"ATTACK 3: {len(cfgs)} configs x {(S_+B0-1)//B0} batches ...", flush=True)
acc = {name: [] for name, _ in cfgs}
t0 = time.time()
for i in range(0, S_, B0):
    idx = HELD[i:i+B0]
    cache = {}
    fwd_route(idx, fill_cache=True, cache=cache)
    for name, c in cfgs:
        acc[name].append(fwd_route(idx, config=c, cache=cache).cpu())
    del cache
    if (i//B0) % 5 == 0:
        print(f"  batch {i//B0+1}/{(S_+B0-1)//B0} ({time.time()-t0:.0f}s)", flush=True)
print(f"ATTACK 3 forwards done ({time.time()-t0:.0f}s)", flush=True)

raw3 = {}
for name, _ in cfgs:
    mn, se = dstat(torch.cat(acc[name], 0))
    raw3[name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:28s} dCE {mn:+.4f} +- {se:.5f}", flush=True)
del acc
assert abs(raw3['sanity_freeze_all_no_ablate']['dCE']) < 0.003, "freeze-all sanity FAILED"

# neighbors from the original cascade run for the interpolation comparison
CASC = json.load(open(f'{QK}/qk_cascade_end.json'))['blocks']
def interp(k, key):
    lo, hi = k-1, k+1
    return 0.5*(CASC[str(lo)][key] + CASC[str(hi)][key])

blocks3 = {}
gates_ok = True
for k in KS:
    tot = raw3[f'k{k}_total_damage']['dCE']
    floor_diff = abs(tot - CENSUS[k]); ok = floor_diff < 0.002; gates_ok &= ok
    ex = raw3[f'k{k}_restore_at_{k+1}']['dCE']
    frz = raw3[f'k{k}_freeze_next_mlp{k+1}']['dCE']
    dro = raw3[f'k{k}_direct_only']['dCE']
    r3 = raw3[f'k{k}_restore_at_{k+3}']['dCE']
    nsf = (tot-frz)/tot if tot else None
    drf = dro/tot if tot else None
    blocks3[k] = {
        'census_floor': CENSUS[k], 'total_damage_dCE': tot,
        'total_damage_SE': raw3[f'k{k}_total_damage']['SE'],
        'floor_gate_abs_diff': round(floor_diff, 4), 'floor_gate_pass': ok,
        'exactness_restore_next_dCE': ex,
        'next_square_fraction_removed': round(nsf, 4),
        'direct_readout_fraction': round(drf, 4),
        'first_two_consumers_fraction': round(r3/tot, 4) if tot else None,
        'interpolated_next_square_fraction': round(interp(k, 'next_square_fraction_removed'), 4),
        'interpolated_direct_fraction': round(interp(k, 'direct_readout_fraction'), 4),
        'next_square_interp_deviation': round(nsf - interp(k, 'next_square_fraction_removed'), 4),
        'direct_interp_deviation': round(drf - interp(k, 'direct_readout_fraction'), 4),
    }
    print(f"k={k:2d} floor {tot:+.4f} (census {CENSUS[k]:+.4f} {'PASS' if ok else 'FAIL'}) | "
          f"next-square {nsf:.3f} (interp {interp(k,'next_square_fraction_removed'):.3f}) | "
          f"direct {drf:.3f} (interp {interp(k,'direct_readout_fraction'):.3f}) | exact {ex:+.4f}",
          flush=True)
assert gates_ok, "FLOOR GATE FAILED (attack 3)"
for k in KS:
    assert abs(blocks3[k]['exactness_restore_next_dCE']) < 0.003, f"exactness gate FAILED k={k}"

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res['attack3_flow_map_smoothness'] = {
    'method': 'VERBATIM qk_cascade_end.py battery at the interpolated blocks k=9,11,13; '
              'deviation vs linear interpolation of measured neighbors',
    'raw': raw3, 'blocks': {str(k): blocks3[k] for k in KS}}
json.dump(res, open(OUT, 'w'), indent=1)
del MO_MEAN
torch.cuda.empty_cache()

# =====================================================================================
# ATTACK 4 -- h.L7.0 aggregation: forward + single-head mean-ablation VERBATIM qk_arc_h70.py
# =====================================================================================
LI_T, H_T = 7, 0
print(f"ATTACK 4: h.L{LI_T}.{H_T} aggregation analysis ...", flush=True)

@torch.no_grad()
def forward_h(idx, ablate_target=False, yhmean_t=None, collect=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    out = {}
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0)
        yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        if collect and li == LI_T:
            Wr = a.c_proj.weight.view(D, NH, HD)
            comp = torch.einsum('bthc,ohc->btho', yh4[:, :, H_T:H_T+1], Wr[:, H_T:H_T+1, :])
            out['hnorm'] = comp[:, :, 0, :].norm(dim=-1).cpu().numpy()
            YH_SUM[:] += yh4[:, :, H_T].sum(0)
        if ablate_target and li == LI_T:
            yh4 = yh4.clone(); yh4[:, :, H_T] = yhmean_t.unsqueeze(0)
        x = x + a.c_proj(yh4.reshape(B, T, -1))
        mo = blk.mlp(F.rms_norm(x, (D,)))
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return logits

YH_SUM = torch.zeros(T_, HD, device=DEV)
head_act = np.zeros((S_, T_), np.float32)
print("PASS A: target-head norm + per-position yh mean (truncated collect) ...", flush=True)
@torch.no_grad()
def collect_pass():
    for i in range(0, S_, B0):
        idx = HELD[i:i+B0]
        B, T = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        for li in range(LI_T+1):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
            def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0)
            yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            if li == LI_T:
                Wr = a.c_proj.weight.view(D, NH, HD)
                comp = torch.einsum('bthc,ohc->btho', yh4[:, :, H_T:H_T+1], Wr[:, H_T:H_T+1, :])
                head_act[i:i+B] = comp[:, :, 0, :].norm(dim=-1).cpu().numpy()
                YH_SUM[:] += yh4[:, :, H_T].sum(0)
            x = x + a.c_proj(yh4.reshape(B, T, -1))
            mo = blk.mlp(F.rms_norm(x, (D,)))
            x = x + mo
collect_pass()
YHMEAN_T = YH_SUM / S_
print("PASS A done.", flush=True)

print("PASS B: base + ablated forwards, per-position delta cross-entropy ...", flush=True)
DCE = np.zeros((S_, T_-1), np.float32)
t0 = time.time()
for i in range(0, S_, B0):
    idx = HELD[i:i+B0]
    b = idx.shape[0]
    bl = forward_h(idx).float()
    al = forward_h(idx, ablate_target=True, yhmean_t=YHMEAN_T).float()
    blp = F.log_softmax(bl[:, :T_-1], -1)
    alp = F.log_softmax(al[:, :T_-1], -1)
    tgt = idx[:, 1:].unsqueeze(-1)
    bce = -blp.gather(-1, tgt).squeeze(-1)
    ace = -alp.gather(-1, tgt).squeeze(-1)
    DCE[i:i+b] = (ace - bce).cpu().numpy()
    del bl, al, blp, alp
print(f"PASS B done ({time.time()-t0:.0f}s)", flush=True)

# global gate (census: 0.017004, z 11.4; all positions with a next token are valid)
d = DCE.reshape(-1).astype(np.float64)
n = d.size
gm = d.mean(); gse = d.std(ddof=1)/math.sqrt(n); gz = gm/gse
print(f"GATE: global dCE {gm:.6f} +- {gse:.6f} z {gz:.2f} (census 0.017004, z 11.4) n={n}", flush=True)
gate4 = abs(gm - 0.017004) < 0.003

# (a) deciles of the head's own activation norm at the predicting position
act = head_act[:, :T_-1].reshape(-1)
edges = np.quantile(act, np.linspace(0, 1, 11))
deciles = []
for q in range(10):
    lo, hi = edges[q], edges[q+1]
    mk = (act >= lo) & (act <= hi if q == 9 else act < hi)
    dd = d[mk]
    mnq = dd.mean(); seq_ = dd.std(ddof=1)/math.sqrt(dd.size)
    deciles.append({'decile': q+1, 'act_lo': round(float(lo), 3), 'act_hi': round(float(hi), 3),
                    'n': int(dd.size), 'dCE': round(float(mnq), 6), 'SE': round(float(seq_), 6),
                    'z': round(float(mnq/seq_), 2),
                    'share_of_total': round(float(dd.sum()/d.sum()), 4)})
    print(f"  activation decile {q+1}: dCE {mnq:+.6f} +- {seq_:.6f} z {mnq/seq_:+.2f} "
          f"share {dd.sum()/d.sum():+.3f}", flush=True)

# (b) per-sequence contributions
seq_mean = DCE.mean(1); seq_sum = DCE.sum(1)
order = np.argsort(-np.abs(seq_sum))
tot_sum = float(seq_sum.sum())
top5_share = float(np.abs(seq_sum[order[:5]]).sum()/np.abs(seq_sum).sum())
top5_net_share = float(seq_sum[order[:5]].sum()/tot_sum)
top15_net_share = float(seq_sum[order[:15]].sum()/tot_sum)
def drop_top(kk):
    keep = np.ones(S_, bool); keep[order[:kk]] = False
    dk = DCE[keep].reshape(-1).astype(np.float64)
    mk_ = dk.mean(); sk = dk.std(ddof=1)/math.sqrt(dk.size)
    return {'k_dropped': kk, 'dCE': round(float(mk_), 6), 'SE': round(float(sk), 6),
            'z': round(float(mk_/sk), 2)}
drop5 = drop_top(5); drop15 = drop_top(15)
pos_seq = int((seq_mean > 0).sum())
sign_z = (pos_seq - S_/2)/math.sqrt(S_/4)
per_seq_z = seq_mean / (DCE.std(1, ddof=1)/math.sqrt(T_-1))
top_seqs = [{'seq': int(s), 'mean_dCE': round(float(seq_mean[s]), 5),
             'net_share': round(float(seq_sum[s]/tot_sum), 4)} for s in order[:8]]
print(f"sequences: {pos_seq}/{S_} positive-mean (sign z {sign_z:.2f}); "
      f"top-5 |contribution| share {top5_share:.3f}, net share {top5_net_share:.3f}; "
      f"z after dropping top-5 {drop5['z']}, top-15 {drop15['z']}", flush=True)

# (c) position concentration of the NET damage
ds = np.sort(d)[::-1]
k1 = max(1, int(0.01*n)); k5 = max(1, int(0.05*n))
conc = {'top1pct_positive_positions_net_share': round(float(ds[:k1].sum()/tot_sum), 3),
        'top5pct_positive_positions_net_share': round(float(ds[:k5].sum()/tot_sum), 3),
        'bottom1pct_negative_positions_net_share': round(float(ds[-k1:].sum()/tot_sum), 3),
        'positive_position_fraction': round(float((d > 0).mean()), 4)}
print(f"position concentration: {conc}", flush=True)

res = json.load(open(OUT))
res['attack4_h70_aggregation'] = {
 'method': 'VERBATIM qk_arc_h70.py forward + per-position-mean single-head ablation; '
           'per-position delta cross-entropy split by head-activation decile, by sequence, '
           'and by position concentration',
 'gate': {'global_dCE': round(float(gm), 6), 'SE': round(float(gse), 6),
          'z': round(float(gz), 2), 'census_expected': 0.017004, 'pass': bool(gate4), 'n': n},
 'activation_deciles': deciles,
 'per_sequence': {
    'n_sequences': S_, 'positive_mean_sequences': pos_seq,
    'sign_test_z': round(float(sign_z), 2),
    'sequences_with_abs_z_above_2': int((np.abs(per_seq_z) > 2).sum()),
    'top5_abs_contribution_share': round(top5_share, 4),
    'top5_net_share': round(top5_net_share, 4), 'top15_net_share': round(top15_net_share, 4),
    'global_z_after_drop_top5': drop5, 'global_z_after_drop_top15': drop15,
    'top_sequences': top_seqs},
 'position_concentration': conc,
}
json.dump(res, open(OUT, 'w'), indent=1)
assert gate4, f"ATTACK 4 gate FAILED: {gm} vs 0.017004"
print(f"Saved {OUT}", flush=True)
print("QK REDTEAM ARCS (script 2) DONE", flush=True)
