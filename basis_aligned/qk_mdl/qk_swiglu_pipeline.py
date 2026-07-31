"""CROSS-MODEL GENERALITY of the section-89 provenance pipeline map: does softmax SwiGLU
swiglu18 show the same RECENCY-TO-HISTORY structure as bilin18 (early layers driven by recent
components, each layer's own attention going causally dead late, history groups taking over)?

swiglu18's MLPs are NOT bilinear, so the exact 15-term decomposition of qk_allterm_census.py
does NOT port. Instead: the GROUP-level version. At each layer L, the pre-norm MLP input
x_pre is an EXACT lambda-weighted sum of five coarse groups (accumulators VERBATIM from
qk_allterm_census.py):
  E  = embedding stream (coefficient-tracked x0),
  Ae = attention-EARLIER (lambda-decayed sum of all prior attention outputs),
  Ar = attention-RECENT (current layer's own attention output, coefficient 1),
  Me = MLP-EARLIER (lambda-decayed sum of mlp outputs before the previous one),
  Mr = MLP-RECENT (lambda-decayed immediately preceding mlp output).
CAUSAL measurement by INPUT-GROUP ablation: replace the group's contribution to x_pre with
its per-position held mean (all other groups intact), recompute THAT layer's MLP output from
the modified input (mo' = mlp(rms_norm(x_pre - group + group_mean)); the residual stream and
all downstream layers continue normally with mo'), measure global delta cross-entropy vs the
base model, paired standard error. This is exact as an intervention (no bilinearity needed).
Denominator: the layer's full MLP mean-ablation floor (mo -> per-position held mean), verified
against qk_general_completeness_swiglu18.json hub/per_layer_tail full_mlp_dCE.

FORWARD VERBATIM from qk_general_classpush_swiglu.py (softmax attention, value-lerp, SwiGLU
MLP as a black box; sanity base CE 3.41). Group accumulators / gate / per-position means /
dstat harness pattern VERBATIM from qk_allterm_census.py, adapted: groups are ablated at the
INPUT of each MLP, one layer at a time. Held-back FW[448:600,:128], FULL 152 sequences (one
held pass = 1.1 s, so no position subsample needed), batch 6, peak <4GB.
Output: qk_swiglu_pipeline.json (incremental per-layer dump)."""
import json, os, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_swiglu_pipeline.json'

# ---------------- GPU GUARD (verbatim from census) ----------------
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

m, cfg = load_elriggs('swiglu18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128].to(DEV); B0 = 6
S_, T_ = HELD.shape
print(f"swiglu18 NL={NL} D={D} NH={NH} held {S_}x{T_}", flush=True)

GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']
NG = 5

# prior per-layer full-MLP mean-ablation floors for verification (same held slice/currency)
try:
    PRIOR = {r['layer']: r['full_mlp_dCE'] for r in
             json.load(open(f'{QK}/qk_general_completeness_swiglu18.json'))['hub']['per_layer_tail']}
except Exception:
    PRIOR = {}

# =====================================================================================
# Forward: swiglu18 VERBATIM (qk_general_classpush_swiglu.py) + census group accumulators.
# mode: None (full model)
#       'collect'  (per-position group means + MLP-output means + group-sum gate, ALL layers
#                   in one pass; LI ignored)
#       'floor'    (at layer LI: mo -> per-position held mean MOMEAN)
#       'groupabl' (at layer LI: x_pre -> x_pre - group[gi] + GMEAN[gi]; mo recomputed from
#                   the modified input; residual stream continues with the ORIGINAL x + mo')
# Returns per-position CE (B, T-1).
# =====================================================================================
@torch.no_grad()
def fwd(idx, LI=None, mode=None, gi=None, GMEAN=None, MOMEAN=None, stats=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    track = mode is not None
    collect_all = (mode == 'collect')
    LIeff = (NL - 1) if collect_all else LI
    if track:
        cE = torch.ones((), device=DEV)
        SA = torch.zeros_like(x); SM = torch.zeros_like(x); MR = torch.zeros_like(x)
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        if track and li <= LIeff:
            cE = blk.lambdas[0]*cE + blk.lambdas[1]
            SA = blk.lambdas[0]*SA; SM = blk.lambdas[0]*SM; MR = blk.lambdas[0]*MR
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        sc = (torch.einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)).masked_fill(~mask, float('-inf'))
        pat = F.softmax(sc, -1)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout          # x is now x_pre of the MLP
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if track and (li == LIeff or collect_all):
            groups = [cE*x0, SA, aout, SM, MR]                       # E, Ae, Ar, Me, Mr
            if mode == 'collect':
                st = stats[li]
                gs = sum(groups)
                st['grp_err'] = max(st['grp_err'],
                                    float(((gs - x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).max()))
                for g in range(NG):
                    st['gsq'][g] += float(groups[g].pow(2).sum())
                    st['gsum'][g] += groups[g].sum(0)
                st['xsq'] += float(x.pow(2).sum())
                st['mosum'] += mo.sum(0)
            elif mode == 'floor':
                mo = MOMEAN.unsqueeze(0).expand(B, -1, -1).to(x.dtype)
            elif mode == 'groupabl':
                xabl = x - groups[gi] + GMEAN[gi].unsqueeze(0)
                mo = blk.mlp(F.rms_norm(xabl, (D,)))
            del groups
        x = x + mo
        if track and li < LIeff:
            SA = SA + aout; SM = SM + MR; MR = mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

# ---------------- base CE (full model), once ----------------
print("BASE: full-model cross-entropy on held slice ...", flush=True)
base = torch.cat([fwd(HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
base_ce = float(base.mean())
print(f"base CE mean {base_ce:.4f} (sanity: 3.41)", flush=True)
assert abs(base_ce - 3.41) < 0.02, f"base CE {base_ce} != 3.41 sanity -- forward is wrong"

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

# ---------------- PASS 1: per-position group means + MLP-output means, all layers ----------------
print("PASS 1: per-position group means + MLP-output means + group-sum gate (all layers) ...", flush=True)
ST = [{'gsum': [torch.zeros(T_, D, device=DEV) for _ in range(NG)],
       'mosum': torch.zeros(T_, D, device=DEV),
       'gsq': [0.0]*NG, 'xsq': 0.0, 'grp_err': 0.0} for _ in range(NL)]
for i in range(0, S_, B0):
    fwd(HELD[i:i+B0], mode='collect', stats=ST)
GMEANS = [torch.stack([ST[li]['gsum'][g]/S_ for g in range(NG)]) for li in range(NL)]   # (NG,T,D) each
MOMEANS = [ST[li]['mosum']/S_ for li in range(NL)]
for li in range(NL):
    gshare = [round(ST[li]['gsq'][g]/ST[li]['xsq'], 4) for g in range(NG)]
    ST[li]['gshare'] = gshare
    print(f"L{li:2d} GATE group-sum rel err max {ST[li]['grp_err']:.2e} | "
          f"group msq shares {dict(zip(GNAMES, gshare))}", flush=True)

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if start > 0 and os.path.exists(OUT):
    res = json.load(open(OUT))
else:
    res = {'meta': {
        'model': 'swiglu18', 'arch': 'softmax attention + SwiGLU MLP (black box)',
        'held': 'FW[448:600,:128]', 'batch': B0, 'n_sequences': S_,
        'position_subsample': 'NONE -- full 152 held sequences (one held pass = 1.1 s)',
        'groups': {'E': 'embedding stream (coefficient-tracked x0)',
                   'Ae': 'attention-EARLIER (lambda-decayed sum of all prior attention outputs)',
                   'Ar': 'attention-RECENT (current layer attention output)',
                   'Me': 'MLP-EARLIER (lambda-decayed sum of mlp outputs before the previous one)',
                   'Mr': 'MLP-RECENT (lambda-decayed immediately preceding mlp output)'},
        'intervention': 'INPUT-group ablation at layer L: x_pre -> x_pre - group + per-position held '
                        'mean of the group; mo recomputed = mlp(rms_norm(modified x_pre)); residual '
                        'stream and downstream continue normally with the recomputed mo. EXACT as an '
                        'intervention -- no bilinearity needed. Floor = mo -> per-position held mean.',
        'machinery': 'forward VERBATIM qk_general_classpush_swiglu.py (softmax attention, value-lerp, '
                     'SwiGLU black box); group accumulators / gate / means / dstat VERBATIM '
                     'qk_allterm_census.py, adapted to input-side ablation',
        'currency': 'GLOBAL delta cross-entropy per valid held position (nats), paired standard error',
        'floor_verification_source': 'qk_general_completeness_swiglu18.json hub/per_layer_tail full_mlp_dCE',
        'base_ce': round(base_ce, 4)}, 'layers': {}}

# ---------------- PASS 2: per-layer floor + 5 input-group ablations ----------------
for LI in range(start, NL):
    tL = time.time()
    st = ST[LI]
    rec = {'gate': {'group_sum_rel_err_max': st['grp_err'], 'pass': bool(st['grp_err'] < 1e-4)},
           'group_msq_share_of_xpre': dict(zip(GNAMES, st['gshare']))}
    if st['grp_err'] >= 1e-4:
        print(f"L{LI}: GROUP-SUM GATE FAILED -- skipping causal measurements", flush=True)
        res['layers'][str(LI)] = rec; json.dump(res, open(OUT, 'w'), indent=1); continue

    def run(mode, gi=None):
        out = []
        for i in range(0, S_, B0):
            out.append(fwd(HELD[i:i+B0], LI=LI, mode=mode, gi=gi,
                           GMEAN=GMEANS[LI], MOMEAN=MOMEANS[LI]).cpu())
        return torch.cat(out, 0)

    floor, floor_se = dstat(run('floor'))
    prior = PRIOR.get(LI)
    rec['floor_dCE'] = round(floor, 5); rec['floor_SE'] = round(floor_se, 5)
    rec['floor_prior_ref'] = prior
    rec['floor_matches_prior'] = (bool(abs(floor - prior) < max(0.02, 0.05*abs(prior)))
                                  if prior is not None else None)
    print(f"L{LI:2d} floor {floor:+.4f} +- {floor_se:.5f} (prior {prior}) "
          f"match={rec['floor_matches_prior']}", flush=True)

    rec['groups'] = {}
    dead = []
    for g in range(NG):
        if st['gshare'][g] < 1e-8:          # group identically absent (zero stream)
            rec['groups'][GNAMES[g]] = {'dCE': 0.0, 'SE': 0.0, 'frac_of_floor': 0.0,
                                        'absent': True, 'z': 0.0}
            print(f"  L{LI} ablate {GNAMES[g]:2s}  ABSENT (zero stream)", flush=True)
            continue
        mn, se = dstat(run('groupabl', gi=g))
        frac = mn/floor if abs(floor) > 1e-9 else None
        z = mn/se if se > 0 else 0.0
        isdead = bool(abs(mn) < max(0.01, 0.01*abs(floor)))
        if isdead: dead.append(GNAMES[g])
        rec['groups'][GNAMES[g]] = {'dCE': round(mn, 5), 'SE': round(se, 5),
                                    'frac_of_floor': round(frac, 4) if frac is not None else None,
                                    'absent': False, 'z': round(z, 1), 'dead_causal': isdead}
        print(f"  L{LI} ablate {GNAMES[g]:2s}  dCE {mn:+.4f} +- {se:.5f}  "
              f"frac_of_floor {frac:+.3f}  z {z:.1f}{'  DEAD' if isdead else ''}", flush=True)
    rec['groups_absent'] = [GNAMES[g] for g in range(NG) if st['gshare'][g] < 1e-8]
    rec['groups_dead_causal'] = dead
    print(f"L{LI} DONE ({time.time()-tL:.0f}s) dead={dead} absent={rec['groups_absent']}", flush=True)
    res['layers'][str(LI)] = rec
    json.dump(res, open(OUT, 'w'), indent=1)
    torch.cuda.empty_cache()

# ---------------- cross-model reference: bilin18 census group involvement ----------------
try:
    census = json.load(open(f'{QK}/qk_allterm_census.json'))
    PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
    PN = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
    cmp_tab = {}
    for li in range(NL):
        r = census['layers'].get(str(li))
        if not r or 'energy_shares' not in r: continue
        inv = {}
        for g in range(NG):
            s = sum(v for k, v in r['energy_shares'].items()
                    if GNAMES[g] in k.split('x'))
            inv[GNAMES[g]] = round(s, 4)
        cmp_tab[str(li)] = {'floor_dCE': r.get('floor_dCE'),
                            'group_energy_involvement': inv,
                            'groups_dead_causal': r.get('groups_dead_causal'),
                            'groups_absent': r.get('groups_absent')}
    res['bilin18_census_reference'] = {
        'source': 'qk_allterm_census.json (term-level; involvement = summed centered energy share '
                  'of the 15 pair terms touching each group)',
        'per_layer': cmp_tab}
    json.dump(res, open(OUT, 'w'), indent=1)
except Exception as e:
    print(f"bilin18 reference skipped: {e}", flush=True)

print("QK SWIGLU PIPELINE DONE", flush=True)
