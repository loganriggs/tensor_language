"""CROSS-LAYER COMPOSITION experiment 3 -- GATING-PARTNER IDENTIFICATION for h.L7.0.

h.L7.0 (section 100: 0.017 nats global, 88% mediated, context-dependent sign) writes a residual
stream h that downstream bilinear feed-forwards consume. Logan's hypothesis: the head looks
"conditional" from outside because a downstream bilinear form MULTIPLIES h by another stream
quantity; folding h into the consumer makes the gating partner explicit in the algebra.

STEP 1 (consumer scan): for each downstream feed-forward block L in 7..17, replace h's
lambda-decayed direct contribution to that block's mlp input with its per-position held mean
(everything else intact, including all attention paths), and measure delta cross-entropy.
The largest consumer L* is where the fold happens.
STEP 2 (fold): at L*, split the mlp input into SEVEN exact groups
  {E, Ae' (attention-earlier minus h), Ar, Me, Mr, h}  (6 groups; h removed from its container)
-> 21 exact pair terms. GATES: group sum == x_pre (1e-4), terms+bias == mo (1e-4), mean-only
floor reproduces the census floor at L*, and drop-all-h-involving-terms is consistent with the
step-1 ablation at L*. Then drop each h-involving cross term alone (necessity) and add each
h-involving term alone to the all-non-h configuration (sufficiency). The GATING PARTNER is the
group whose single cross term with h carries the h-effect.
STEP 3 (sign-flip examples): per-position paired delta cross-entropy for dropping the h x partner
term; among positions with top-quartile h-stream norm, show concrete held examples where the term
HELPS (context A) vs HURTS (context B), with the partner-stream state and decoded context.

Machinery: forward + head write VERBATIM from qk_arc_h70.py (c_proj weight slice for head 0 of
layer 7); groups/pair-terms VERBATIM from qk_allterm_census.py; held FW[448:600,:128], paired
standard errors; per-position means computed on held (census convention). Batch 4, <4GB."""
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
from transformers import AutoTokenizer
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_xfold_gate.json'
LI_T, H_T = 7, 0

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
HELD = FW[448:600, :128].to(DEV); B0 = 4
S_, T_ = HELD.shape
tokzr = AutoTokenizer.from_pretrained('gpt2')
print(f"bilin18 target h.L{LI_T}.{H_T}; held {S_}x{T_}", flush=True)

# census floors per layer for the step-2 gate
CENSUS = json.load(open(f'{QK}/qk_allterm_census.json'))['layers']

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())

# =====================================================================================
# Core forward. Computes the target-head residual write rh at layer 7 (VERBATIM slice
# construction from qk_arc_h70.py), tracks its lambda-decayed stream h, plus the census
# group accumulators (E coefficient, Ae, Me, Mr).
# modes:
#   'collect_rh'    : accumulate per-position rh sums (for RH_MEAN)
#   'ablate_at'     : LI = consumer layer; mlp_LI recomputed on x_pre - h + h_mean
#   'collect_terms' : LI consumer; 21-term means + gates (needs W, RH_MEAN)
#   'energy'        : centered term energies
#   'subset'        : mo_LI -> MEANF + sum of kept term deviations; optional per-position CE out
#   'perpos'        : like subset but also returns per-position term/stream stats
# =====================================================================================
GN = ['E', 'Ae', 'Ar', 'Me', 'Mr', 'h']
NG = 6
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GN[i]}x{GN[j]}' for (i, j) in PAIRS]
NT = len(PAIRS)
IDX = {n: k for k, n in enumerate(PNAMES)}
H_TERMS = [k for k, (i, j) in enumerate(PAIRS) if (i == 5 or j == 5)]

def pair_terms(groups, xpre, Lw, Rw, Dw):
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    PL = [g @ Lw.T for g in groups]; PR = [g @ Rw.T for g in groups]
    terms = []
    for (i, j) in PAIRS:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ Dw.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms

@torch.no_grad()
def fwd(idx, LI=None, mode=None, subset=None, TMEAN=None, MEANF=None, stats=None, W=None,
        RH_MEAN=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    track = mode is not None
    if track:
        cE = torch.ones((), device=DEV)
        SA = torch.zeros_like(x); SM = torch.zeros_like(x); MR = torch.zeros_like(x)
        h = torch.zeros_like(x); hm = torch.zeros(T, D, device=DEV) if RH_MEAN is not None else None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        if track and (LI is None or li <= LI):
            cE = blk.lambdas[0]*cE + blk.lambdas[1]
            SA = blk.lambdas[0]*SA; SM = blk.lambdas[0]*SM; MR = blk.lambdas[0]*MR
            h = blk.lambdas[0]*h
            if hm is not None: hm = blk.lambdas[0]*hm
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh4.reshape(B, T, -1)); x = x + aout
        if track and li == LI_T:
            Wr = a.c_proj.weight.view(D, NH, HD)
            rh = torch.einsum('btc,dc->btd', yh4[:, :, H_T, :], Wr[:, H_T, :])
            h = h + rh                                      # decays from next layer on
            if hm is not None: hm = hm + RH_MEAN
            if mode == 'collect_rh': stats['rh_sum'] += rh.sum(0)
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if track and LI is not None and li == LI and mode != 'collect_rh':
            if mode == 'ablate_at':
                x_mod = x - h + hm.unsqueeze(0)
                mo = blk.mlp(F.rms_norm(x_mod, (D,)))
            else:
                if li == LI_T:                               # h's container is Ar at layer 7 itself
                    groups = [cE*x0, SA, aout - h, SM, MR, h]
                else:                                        # h's container is Ae downstream
                    groups = [cE*x0, SA - h, aout, SM, MR, h]
                terms = pair_terms(groups, x, W[0], W[1], W[2])
                if mode == 'collect_terms':
                    gs = sum(groups)
                    stats['grp_err'] = max(stats['grp_err'],
                                           float(((gs - x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).max()))
                    for kk in range(NT): stats['tsum'][kk] += terms[kk].sum(0)
                    recon = sum(terms) + W[3]
                    num = (recon - mo).norm(dim=-1); den = mo.norm(dim=-1).clamp_min(1e-8)
                    stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                    stats['fro_num'] += float((recon - mo).pow(2).sum())
                    stats['fro_den'] += float(mo.pow(2).sum())
                elif mode == 'energy':
                    for kk in range(NT): stats['te'][kk] += float((terms[kk] - TMEAN[kk]).pow(2).sum())
                    stats['tot'] += float((mo - MEANF).pow(2).sum())
                elif mode in ('subset', 'perpos'):
                    new = MEANF.unsqueeze(0).expand(B, -1, -1)
                    for kk in subset: new = new + (terms[kk] - TMEAN[kk])
                    if mode == 'perpos':
                        kk = stats['watch']
                        stats['xnorm'].append((terms[kk] - TMEAN[kk]).norm(dim=-1).cpu())
                        stats['hnorm'].append(h.norm(dim=-1).cpu())
                        stats['pnorm'].append(groups[stats['partner_g']].norm(dim=-1).cpu())
                    mo = new.to(x.dtype)
                del terms, groups
        x = x + mo
        if track and (LI is None or li < LI):
            SA = SA + aout; SM = SM + MR; MR = mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

# =====================================================================================
# PASS 0: base CE + per-position rh mean
# =====================================================================================
print("PASS 0: base + rh mean ...", flush=True)
st0 = {'rh_sum': torch.zeros(T_, D, device=DEV)}
ces = []
for i in range(0, S_, B0):
    ces.append(fwd(HELD[i:i+B0], LI=NL-1, mode='collect_rh', stats=st0).cpu())
base = torch.cat(ces, 0)
RH_MEAN = st0['rh_sum']/S_
print(f"base CE {float(base.mean()):.4f}  rh mean norm {float(RH_MEAN.norm(dim=-1).mean()):.3f}", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

# =====================================================================================
# STEP 1: consumer scan L = 7..17
# =====================================================================================
print("STEP 1: consumer scan ...", flush=True)
res = {'meta': {'model': 'bilin18', 'target': f'h.L{LI_T}.{H_T}', 'held': 'FW[448:600,:128]',
                'batch': B0, 'base_ce': round(float(base.mean()), 4),
                'census_full_head_ablation_dCE': 0.017004,
                'note': 'h stream = lambda-decayed DIRECT residual write of head 0 layer 7; '
                        'mediated-through-attention components not included in h'},
       'consumer_scan': {}}
scan = []
for L in range(LI_T, NL):
    out = []
    for i in range(0, S_, B0):
        out.append(fwd(HELD[i:i+B0], LI=L, mode='ablate_at', RH_MEAN=RH_MEAN).cpu())
    mn, se = dstat(torch.cat(out, 0))
    scan.append((L, mn, se))
    res['consumer_scan'][str(L)] = {'dCE': round(mn, 5), 'SE': round(se, 5)}
    print(f"  mlp L{L:2d}: replace h-in-input with mean  dCE {mn:+.5f} +- {se:.5f}", flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
LSTAR = max(scan, key=lambda z: z[1])[0]
res['largest_consumer'] = LSTAR
print(f"LARGEST CONSUMER: mlp block {LSTAR}", flush=True)

# =====================================================================================
# STEP 2: fold h into consumer LSTAR -- 21-term split, partner identification
# =====================================================================================
print(f"STEP 2: term fold at L{LSTAR} ...", flush=True)
W = mlp_wts(LSTAR)
st = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
      'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0, 'grp_err': 0.0}
for i in range(0, S_, B0):
    fwd(HELD[i:i+B0], LI=LSTAR, mode='collect_terms', stats=st, W=W, RH_MEAN=RH_MEAN)
TMEAN = torch.stack([t/S_ for t in st['tsum']])
MEANF = TMEAN.sum(0) + W[3]
gate_fro = (st['fro_num']/st['fro_den'])**0.5
print(f"GATE recon global {gate_fro:.2e} maxpos {st['maxrel']:.2e} groupsum {st['grp_err']:.2e}", flush=True)
assert gate_fro < 1e-4 and st['grp_err'] < 1e-4, "fold gate FAILED"

st2 = {'te': [0.0]*NT, 'tot': 0.0}
for i in range(0, S_, B0):
    fwd(HELD[i:i+B0], LI=LSTAR, mode='energy', TMEAN=TMEAN, MEANF=MEANF, stats=st2, W=W, RH_MEAN=RH_MEAN)
shares = {PNAMES[k]: round(st2['te'][k]/st2['tot'], 6) for k in range(NT)}
res['fold'] = {'layer': LSTAR,
               'gate': {'recon_rel_err_global': gate_fro, 'recon_rel_err_max_pos': st['maxrel'],
                        'group_sum_rel_err_max': st['grp_err']},
               'energy_shares': shares,
               'h_term_energy': {PNAMES[k]: shares[PNAMES[k]] for k in H_TERMS}}
print("h-term energies:", res['fold']['h_term_energy'], flush=True)

def run(subset, mode='subset', stats=None):
    out = []
    for i in range(0, S_, B0):
        out.append(fwd(HELD[i:i+B0], LI=LSTAR, mode=mode, subset=subset, TMEAN=TMEAN, MEANF=MEANF,
                       stats=stats, W=W, RH_MEAN=RH_MEAN).cpu())
    return torch.cat(out, 0)

NONH = [k for k in range(NT) if k not in H_TERMS]
cfgs = [('mean_only', []), ('all_terms', list(range(NT))),
        ('all_non_h', NONH), ('drop_all_h_terms', NONH)]
cfgs += [(f'drop_{PNAMES[k]}', [j for j in range(NT) if j != k]) for k in H_TERMS]
cfgs += [(f'nonh_plus_{PNAMES[k]}', NONH + [k]) for k in H_TERMS]
res['fold']['configs'] = {}
for name, sub in cfgs:
    if name == 'drop_all_h_terms': continue   # duplicate of all_non_h, skip
    mn, se = dstat(run(sub))
    res['fold']['configs'][name] = {'dCE': round(mn, 5), 'SE': round(se, 5)}
    print(f"  {name:22s} [{len(sub):2d}] dCE {mn:+.5f} +- {se:.5f}", flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)

cf = res['fold']['configs']
floorL = cf['mean_only']['dCE']
census_floor = CENSUS[str(LSTAR)]['floor_dCE']
drop_all_h = cf['all_non_h']['dCE']
res['fold']['gates2'] = {'floor_vs_census': [floorL, census_floor],
                         'drop_all_h_vs_step1_ablate': [drop_all_h, res['consumer_scan'][str(LSTAR)]['dCE']]}
print("floor vs census:", floorL, census_floor, flush=True)
# partner = h-term whose DROP costs most
drop_costs = {PNAMES[k]: cf[f'drop_{PNAMES[k]}']['dCE'] for k in H_TERMS}
partner_term = max(drop_costs, key=drop_costs.get)
res['fold']['h_term_drop_costs'] = drop_costs
res['fold']['partner_term'] = partner_term
res['fold']['partner_fraction_of_drop_all_h'] = round(drop_costs[partner_term]/max(drop_all_h, 1e-9), 4)
res['fold']['partner_fraction_of_full_head_effect'] = round(drop_costs[partner_term]/0.017004, 4)
print(f"GATING PARTNER TERM: {partner_term}  (drop cost {drop_costs[partner_term]:+.5f}; "
      f"drop-all-h {drop_all_h:+.5f})", flush=True)
json.dump(res, open(OUT, 'w'), indent=1)

# =====================================================================================
# STEP 3: per-position sign-flip examples for the partner term
# =====================================================================================
print("STEP 3: per-position examples ...", flush=True)
kpart = IDX[partner_term]
i_, j_ = PAIRS[kpart]
partner_g = i_ if j_ == 5 else j_                    # the non-h group index
st3 = {'watch': kpart, 'partner_g': partner_g, 'xnorm': [], 'hnorm': [], 'pnorm': []}
ce_drop = run([j for j in range(NT) if j != kpart], mode='perpos', stats=st3)
xnorm = torch.cat(st3['xnorm'], 0).numpy()          # (S,T) centered cross-term norm
hnorm = torch.cat(st3['hnorm'], 0).numpy()          # (S,T) h-stream norm at LSTAR
pnorm = torch.cat(st3['pnorm'], 0).numpy()          # (S,T) partner-group norm
dce_pos = (ce_drop - base).numpy()                  # (S,T-1) positive => term was helping there

held_np = HELD.cpu().numpy()
hq = np.quantile(hnorm[:, :T_-1], 0.75)
mask_hi = hnorm[:, :T_-1] >= hq
flat = dce_pos.copy(); flat[~mask_hi] = 0.0
order_pos = np.dstack(np.unravel_index(np.argsort(-flat, axis=None), flat.shape))[0]
order_neg = np.dstack(np.unravel_index(np.argsort(flat, axis=None), flat.shape))[0]

def describe(s, p):
    lo = max(0, p-14)
    ctx = tokzr.decode([int(t) for t in held_np[s, lo:p+1]])
    nxt = tokzr.decode([int(held_np[s, p+1])])
    return {'seq': int(s), 'pos': int(p), 'context': ctx, 'actual_next': nxt,
            'dCE_drop_term_here': round(float(dce_pos[s, p]), 4),
            'h_stream_norm': round(float(hnorm[s, p]), 3),
            'partner_group_norm': round(float(pnorm[s, p]), 3),
            'cross_term_dev_norm': round(float(xnorm[s, p]), 3)}

helps, hurts = [], []
for s, p in order_pos:
    if len(helps) >= 5: break
    helps.append(describe(int(s), int(p)))
for s, p in order_neg:
    if len(hurts) >= 5: break
    if flat[int(s), int(p)] >= 0: break
    hurts.append(describe(int(s), int(p)))
# aggregate gate statistic: partner norm in helping vs hurting vs inert positions (top-quartile h)
sel = mask_hi
d = dce_pos[sel]; pn = pnorm[:, :T_-1][sel]; hn = hnorm[:, :T_-1][sel]
top = np.argsort(-d)[:200]; bot = np.argsort(d)[:200]
mid = np.argsort(np.abs(d))[:200]
res['examples'] = {
    'partner_term': partner_term,
    'note': 'dCE_drop_term_here > 0 => the h x partner term was HELPING at that position',
    'help_examples': helps, 'hurt_examples': hurts,
    'aggregate_gate_stats_top_quartile_h': {
        'partner_norm_mean_helping_top200': round(float(pn[top].mean()), 3),
        'partner_norm_mean_hurting_top200': round(float(pn[bot].mean()), 3),
        'partner_norm_mean_inert_200': round(float(pn[mid].mean()), 3),
        'h_norm_mean_helping_top200': round(float(hn[top].mean()), 3),
        'h_norm_mean_hurting_top200': round(float(hn[bot].mean()), 3),
        'h_norm_mean_inert_200': round(float(hn[mid].mean()), 3)}}
for e in helps[:3]: print("HELP", json.dumps(e), flush=True)
for e in hurts[:3]: print("HURT", json.dumps(e), flush=True)
print(json.dumps(res['examples']['aggregate_gate_stats_top_quartile_h'], indent=1), flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
print("QK XFOLD GATE DONE", flush=True)
