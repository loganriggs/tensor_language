"""ADVERSARIAL RED-TEAM of RESULTS section 103 (cross-layer folding) -- five attacks.

A1  BLOCK-0-DEAD AT BLOCK 3: the enshrined number (drop all six block-0 terms -0.0000) held the
    shared gauge rho2 fixed (computed from the FULL x_pre3) and replaced dropped terms by their
    per-position means. Here: direct INPUT substitution -- remove block 0's lambda-decayed mlp
    write from block 3's mlp input (mean-replace and hard-zero variants) with the gauge FREE
    (rms_norm recomputed on the modified input), everything else in the network untouched.
    Also the same mean-replacement at block 2's and block 1's mlp inputs (the claim "block 0
    reaches block 3 only through blocks 1-2" needs nonzero consumption there).
A2  CO-DOMINANCE AT BLOCK 3: paired per-position standard error on the DIFFERENCE of the two
    single-term deletions (ArxM2 +0.0077 vs M2xM2 +0.0062) and of the two keep-alones
    (0.2483 vs 0.2564); permutation control on the top-3 sufficiency (+0.033): exhaustive
    triples from the top-10 energy terms plus 40 random triples from all 21.
A3  TOKEN-COVERAGE SENSITIVITY of the block-1 fold: how the 19.5% train-unseen positions are
    handled (fallback = global train mean of mo0 inside s(t)); headline +0.3587 split into
    covered vs uncovered CE positions; substitution restricted to covered-only / uncovered-only
    positions; explicitness R2 split the same way; check the 41.7%-vs-0.1% comparison set.
A4  ' D' SIGN-FLIP CAUSAL GATE: substitute block 1's output with the folded prediction ONLY at
    positions whose current token is t (t in {' the', ',', ' D'}), versus the token-blind linear
    prediction at the same positions; paired delta cross-entropy. Plus capital-class next-token
    probability at ' D' positions under base / folded-at-D / blind-at-D.
A5  h.L7.0 PARTNER: paired per-position differences among drop_Mrxh / drop_hxh / drop_Mexh
    (is the leading cross term significantly ahead at all?); and the 0.9975 helping-vs-hurting
    head-write centroid cosine recomputed after removing the head's GLOBAL mean write
    (subtracted and projected-out variants).

Machinery VERBATIM from qk_xfold_terms.py / qk_xfold_table.py / qk_xfold_gate(_2).py
(forward skeleton originally from qk_hub_streampairs.py). bilin18; TRAIN FW[0:256,:128] for
fitting; held FW[448:600,:128] causal numbers with paired standard errors; batch 4, <4GB,
GPU guard (shared GPU). Output: qk_redteam_xfold.json. Not committed.
"""
import json, os, sys, time, subprocess
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
from transformers import AutoTokenizer
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_redteam_xfold.json'

def gpu_guard(min_free=4500, tries=90, sleep=20):
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
TRAIN = FW[0:256, :128].to(DEV)
HELD = FW[448:600, :128].to(DEV); B0 = 4
STR, _ = TRAIN.shape; S_, T_ = HELD.shape
tokzr = AutoTokenizer.from_pretrained('gpt2')
print(f"bilin18 NL={NL} D={D} NH={NH} held {S_}x{T_}", flush=True)
RES = {'meta': {'model': 'bilin18', 'train': 'FW[0:256,:128]', 'held': 'FW[448:600,:128]',
                'batch': B0, 'note': 'red-team of section 103; machinery verbatim from '
                'qk_xfold_terms/table/gate; paired SEs'}}

def save():
    json.dump(RES, open(OUT, 'w'), indent=1)

def dstat_vs(ce, ref):
    d = (ce - ref).flatten().double()
    return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

# =====================================================================================
# SHARED: layer-3 term machinery (VERBATIM qk_xfold_terms.py) + M0-stream tracking
# =====================================================================================
LI3 = 3
GNAMES = ['E', 'Ae', 'Ar', 'M0', 'M1', 'M2']
NG = 6
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
NT = len(PAIRS)
IDX = {n: k for k, n in enumerate(PNAMES)}
b3 = m.transformer.h[LI3].mlp
Lw3 = b3.Left.weight.detach().float(); Rw3 = b3.Right.weight.detach().float()
Dw3 = b3.Down.weight.detach().float(); bias3 = b3.Down_bias.detach().float()

def pair_terms3(groups, xpre):
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    PL = [g @ Lw3.T for g in groups]; PR = [g @ Rw3.T for g in groups]
    terms = []
    for (i, j) in PAIRS:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ Dw3.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms

@torch.no_grad()
def fwd3(idx, mode=None, subset=None, TMEAN=None, MEANF=None, stats=None,
         m0_layer=None, m0_style=None, M0MEANS=None):
    """VERBATIM skeleton from qk_xfold_terms.fwd with an extra mode:
    'm0abl': at layer m0_layer, the mlp input has block-0's decayed mlp write removed
             ('zero') or mean-replaced ('mean'); rms_norm gauge recomputed (FREE).
    'collect': term sums at layer 3 + M0-stream sums at mlp-input of layers 1,2,3 + gates."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    track = mode is not None
    if track:
        cE = torch.ones((), device=DEV)
        SA = torch.zeros_like(x)
        Ml = []
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        if track and li <= LI3:
            cE = blk.lambdas[0]*cE + blk.lambdas[1]
            SA = blk.lambdas[0]*SA
            Ml = [blk.lambdas[0]*mm for mm in Ml]
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if track and mode == 'm0abl' and li == m0_layer:
            xin = x - Ml[0]
            if m0_style == 'mean': xin = xin + M0MEANS[li].unsqueeze(0)
            mo = blk.mlp(F.rms_norm(xin, (D,)))
        if track and mode == 'collect' and li in (1, 2, 3):
            stats['m0sum'][li] += Ml[0].sum(0)
        if track and li == LI3 and mode in ('collect', 'subset'):
            groups = [cE*x0, SA, aout, Ml[0], Ml[1], Ml[2]]
            terms = pair_terms3(groups, x)
            if mode == 'collect':
                gs = sum(groups)
                stats['grp_err'] = max(stats['grp_err'],
                                       float(((gs - x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).max()))
                for kk in range(NT): stats['tsum'][kk] += terms[kk].sum(0)
                recon = sum(terms) + bias3
                num = (recon - mo).norm(dim=-1); den = mo.norm(dim=-1).clamp_min(1e-8)
                stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                stats['fro_num'] += float((recon - mo).pow(2).sum()); stats['fro_den'] += float(mo.pow(2).sum())
            elif mode == 'subset':
                new = MEANF.unsqueeze(0).expand(B, -1, -1)
                for kk in subset: new = new + (terms[kk] - TMEAN[kk])
                mo = new.to(x.dtype)
            del terms, groups
        x = x + mo
        if track and li < LI3:
            SA = SA + aout; Ml.append(mo)
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

print("SHARED PASS: base CE + layer-3 term means + M0-stream means ...", flush=True)
st = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
      'm0sum': {1: torch.zeros(T_, D, device=DEV), 2: torch.zeros(T_, D, device=DEV),
                3: torch.zeros(T_, D, device=DEV)},
      'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0, 'grp_err': 0.0}
ces = []
for i in range(0, S_, B0):
    ces.append(fwd3(HELD[i:i+B0], mode='collect', stats=st).cpu())
base = torch.cat(ces, 0)
TMEAN = torch.stack([t/S_ for t in st['tsum']])
MEANF = TMEAN.sum(0) + bias3
M0MEANS = {li: st['m0sum'][li]/S_ for li in (1, 2, 3)}
gate_fro = (st['fro_num']/st['fro_den'])**0.5
print(f"base CE {float(base.mean()):.4f} (expect 3.4946) | recon gate {gate_fro:.2e} "
      f"| grp {st['grp_err']:.2e}", flush=True)
assert gate_fro < 1e-4 and st['grp_err'] < 1e-4
RES['shared'] = {'base_ce': round(float(base.mean()), 4),
                 'recon_gate': gate_fro, 'group_gate': st['grp_err'],
                 'M0_stream_mean_norm_at_L': {str(li): round(float(M0MEANS[li].norm(dim=-1).mean()), 3)
                                              for li in (1, 2, 3)}}
save()

def run3(subset):
    out = []
    for i in range(0, S_, B0):
        out.append(fwd3(HELD[i:i+B0], mode='subset', subset=subset, TMEAN=TMEAN, MEANF=MEANF).cpu())
    return torch.cat(out, 0)

def run3_m0(layer, style):
    out = []
    for i in range(0, S_, B0):
        out.append(fwd3(HELD[i:i+B0], mode='m0abl', m0_layer=layer, m0_style=style,
                        M0MEANS=M0MEANS).cpu())
    return torch.cat(out, 0)

# =====================================================================================
# ATTACK 1: block-0-dead at block 3, gauge freed / direct input substitution
# =====================================================================================
print("\nATTACK 1: block-0 input substitution ...", flush=True)
M0INV = [k for k, (i, j) in enumerate(PAIRS) if (i == 3 or j == 3)]
a1 = {}
mn, se = dstat_vs(run3([k for k in range(NT) if k not in M0INV]), base)
a1['reproduce_drop_M0_involving_gauge_fixed'] = {'dCE': round(mn, 5), 'SE': round(se, 5)}
print(f"  reproduce drop_M0_involving (gauge fixed) {mn:+.5f} +- {se:.5f}", flush=True)
for name, (L, style) in [('L3_input_meanreplace_gauge_free', (3, 'mean')),
                         ('L3_input_zero_gauge_free', (3, 'zero')),
                         ('L2_input_meanreplace_gauge_free', (2, 'mean')),
                         ('L2_input_zero_gauge_free', (2, 'zero')),
                         ('L1_input_meanreplace_gauge_free', (1, 'mean'))]:
    mn, se = dstat_vs(run3_m0(L, style), base)
    a1[name] = {'dCE': round(mn, 5), 'SE': round(se, 5)}
    print(f"  {name:36s} dCE {mn:+.5f} +- {se:.5f}", flush=True)
    RES['attack1_block0_dead'] = a1; save()
RES['attack1_block0_dead'] = a1; save()

# =====================================================================================
# ATTACK 2: co-dominance paired differences + triple permutation control
# =====================================================================================
print("\nATTACK 2: co-dominance ...", flush=True)
a2 = {}
ENERGY_RANK = ['M2xM2', 'ArxM2', 'M1xM2', 'ArxAr', 'AexM2', 'ArxM1', 'AexAr', 'M1xM1',
               'AexM1', 'AexAe']            # from qk_xfold_terms.json energy_rank[:10]
order10 = [IDX[n] for n in ENERGY_RANK]

ce_dropAr = run3([j for j in range(NT) if j != IDX['ArxM2']])
ce_dropM2 = run3([j for j in range(NT) if j != IDX['M2xM2']])
mnA, seA = dstat_vs(ce_dropAr, base); mnM, seM = dstat_vs(ce_dropM2, base)
mnD, seD = dstat_vs(ce_dropAr, ce_dropM2)
a2['drop_ArxM2'] = {'dCE': round(mnA, 5), 'SE': round(seA, 5)}
a2['drop_M2xM2'] = {'dCE': round(mnM, 5), 'SE': round(seM, 5)}
a2['paired_difference_drop'] = {'mean': round(mnD, 5), 'SE': round(seD, 5),
                                'z': round(mnD/max(seD, 1e-12), 2)}
print(f"  necessity diff (ArxM2 - M2xM2) {mnD:+.5f} +- {seD:.5f}  z={mnD/max(seD,1e-12):.2f}", flush=True)

ce_onlyM2 = run3([IDX['M2xM2']]); ce_onlyAr = run3([IDX['ArxM2']])
mnD2, seD2 = dstat_vs(ce_onlyM2, ce_onlyAr)
a2['only_M2xM2'] = {'dCE': round(dstat_vs(ce_onlyM2, base)[0], 5)}
a2['only_ArxM2'] = {'dCE': round(dstat_vs(ce_onlyAr, base)[0], 5)}
a2['paired_difference_keep_alone'] = {'mean': round(mnD2, 5), 'SE': round(seD2, 5),
                                      'z': round(mnD2/max(seD2, 1e-12), 2)}
print(f"  sufficiency diff (only_M2xM2 - only_ArxM2) {mnD2:+.5f} +- {seD2:.5f}  "
      f"z={mnD2/max(seD2,1e-12):.2f}", flush=True)
RES['attack2_codominance'] = a2; save()

# triple permutation control
from itertools import combinations
named = tuple(sorted([IDX['M2xM2'], IDX['ArxM2'], IDX['M1xM2']]))
triples = [tuple(sorted(c)) for c in combinations(order10, 3)]
rng = np.random.default_rng(0)
extra = set()
while len(extra) < 40:
    c = tuple(sorted(rng.choice(NT, 3, replace=False).tolist()))
    if c not in triples and c not in extra: extra.add(c)
alltrip = triples + sorted(extra)
print(f"  triple sweep: {len(alltrip)} configurations ...", flush=True)
trip_res = {}
t0 = time.time()
for ii, c in enumerate(alltrip):
    mn, se = dstat_vs(run3(list(c)), base)
    trip_res['+'.join(PNAMES[k] for k in c)] = (round(mn, 5), round(se, 5))
    if ii % 20 == 0:
        print(f"    {ii}/{len(alltrip)} ({time.time()-t0:.0f}s)", flush=True)
srt = sorted(trip_res.items(), key=lambda z: z[1][0])
named_name = '+'.join(PNAMES[k] for k in named)
named_rank = [n for n, _ in srt].index(named_name) + 1
a2['triple_sweep'] = {
    'n_triples': len(alltrip),
    'named_triple': named_name, 'named_dCE': trip_res[named_name][0],
    'named_rank': named_rank,
    'best10': [{'terms': n, 'dCE': v[0], 'SE': v[1]} for n, v in srt[:10]]}
print(f"  named triple rank {named_rank}/{len(alltrip)}; best: {srt[0]}", flush=True)
RES['attack2_codominance'] = a2; save()
del ce_dropAr, ce_dropM2, ce_onlyM2, ce_onlyAr

# =====================================================================================
# SHARED: block-1 fold machinery (VERBATIM qk_xfold_table.py)
# =====================================================================================
print("\nSHARED: block-1 table fit (train) ...", flush=True)
LI1 = 1
b0, b1 = m.transformer.h[0], m.transformer.h[1]
L1 = b1.mlp.Left.weight.detach().float(); R1 = b1.mlp.Right.weight.detach().float()
D1w = b1.mlp.Down.weight.detach().float(); bias1 = b1.mlp.Down_bias.detach().float()
lamE = (b1.lambdas[0] * (b0.lambdas[0] + b0.lambdas[1]) + b1.lambdas[1]).item()
lam0 = b1.lambdas[0].item()

def Bnum(a, b):
    return ((a @ L1.T) * (b @ R1.T)) @ D1w.T

@torch.no_grad()
def fwd_to1(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    out = {}
    for li in range(2):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == 0: out['a0c'] = aout; out['mo0'] = mo
        if li == 1: out['aout1'] = aout; out['x_pre1'] = x; out['mo1'] = mo
        x = x + mo
    return out

tok_sum = torch.zeros(V, D, device=DEV); tok_cnt = torch.zeros(V, device=DEV)
mu_sum = torch.zeros(D, device=DEV); npos = 0
for i in range(0, STR, B0):
    o = fwd_to1(TRAIN[i:i+B0])
    fl = TRAIN[i:i+B0].reshape(-1)
    tok_sum.index_add_(0, fl, o['mo0'].reshape(-1, D).float())
    tok_cnt.index_add_(0, fl, torch.ones_like(fl, dtype=torch.float))
    mu_sum += o['x_pre1'].reshape(-1, D).float().sum(0)
    npos += fl.numel()
GLOB = tok_sum.sum(0)/tok_cnt.sum()
seen = tok_cnt > 0
TT = torch.where(seen[:, None], tok_sum/tok_cnt.clamp_min(1)[:, None], GLOB[None])
MU = mu_sum/npos
WTE_N = F.rms_norm(m.transformer.wte.weight.detach().float(), (D,))
S_ALL = lamE*WTE_N + lam0*TT
del tok_sum
cov_lut = seen.clone()                                   # (V,) bool: token seen in train
cov_frac = float(cov_lut[HELD.reshape(-1)].float().mean())
print(f"held coverage {cov_frac:.4f} (expect 0.805); lamE {lamE:.4f} lam0 {lam0:.6f}", flush=True)

@torch.no_grad()
def fwd_full1(idx, sub=None, only_mask=None, want_probs_mask=None, capvec=None):
    """VERBATIM qk_xfold_table.fwd_full + only_mask (substitute only where current token in mask
    LUT) + optional capital-class next-token probability capture at positions in want_probs_mask."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    a0c = None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == 0: a0c = aout
        if li == LI1 and sub is not None:
            if sub == 'floor':
                pred = MO1_PMEAN.unsqueeze(0).expand(B, -1, -1).to(x.dtype)
            else:
                xf = x.float(); rho2 = xf.pow(2).sum(-1, keepdim=True)/D
                s = S_ALL[idx]; r = (lam0*a0c + aout).float()
                if sub == 'folded': num = Bnum(s, s) + Bnum(s, r) + Bnum(r, s)
                elif sub == 'blind':
                    num = (Bnum(MU[None, None].expand_as(xf), xf) + Bnum(xf, MU[None, None].expand_as(xf))
                           - Bnum(MU, MU)[None, None])
                pred = (num/rho2 + bias1).to(x.dtype)
            if only_mask is not None:
                mo = torch.where(only_mask[idx][..., None], pred, mo)
            else:
                mo = pred
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    if want_probs_mask is not None:
        pm = want_probs_mask[idx]                        # (B,T) positions of interest
        cap_p = []
        if pm.any():
            lg = logits[pm].float()                      # (n,V)
            pr = F.softmax(lg, dim=-1)
            cap_p = (pr * capvec[None]).sum(-1).cpu()
        return ce, cap_p, pm.cpu()
    return ce

# per-position held mean of mo1 for the floor
mo1_sum = torch.zeros(T_, D, device=DEV)
for i in range(0, S_, B0):
    o = fwd_to1(HELD[i:i+B0]); mo1_sum += o['mo1'].float().sum(0)
MO1_PMEAN = mo1_sum/S_

# =====================================================================================
# ATTACK 3: token-coverage sensitivity of the block-1 fold
# =====================================================================================
print("\nATTACK 3: coverage sensitivity ...", flush=True)
a3 = {'held_coverage': round(cov_frac, 4),
      'fallback': 'unseen tokens fall back to the GLOBAL train mean of mo0 inside s(t); '
                  'the embedding part lamE*rms(wte) is exact for every token',
      's_component_norms_on_held': {}}
# component norms of s(t) on held positions
hf = HELD.reshape(-1)
a3['s_component_norms_on_held'] = {
    'lamE_wte_mean_norm': round(float((lamE*WTE_N[hf]).norm(dim=-1).mean()), 2),
    'lam0_table_mean_norm': round(float((lam0*TT[hf]).norm(dim=-1).mean()), 2),
    'lam0_table_mean_norm_covered': round(float((lam0*TT[hf[cov_lut[hf]]]).norm(dim=-1).mean()), 2)}

# explicitness split (truncated passes)
MO1_GMEAN = MO1_PMEAN.mean(0)
accE = {g: {'ef': 0.0, 'eb': 0.0, 'dg': 0.0} for g in ('all', 'cov', 'unc')}
for i in range(0, S_, B0):
    idx = HELD[i:i+B0]
    o = fwd_to1(idx)
    x = o['x_pre1'].float(); mo1 = o['mo1'].float()
    rho2 = x.pow(2).sum(-1, keepdim=True)/D
    s = S_ALL[idx]; r = (lam0*o['a0c'] + o['aout1']).float()
    pf = (Bnum(s, s) + Bnum(s, r) + Bnum(r, s))/rho2 + bias1
    pb = (Bnum(MU[None, None].expand_as(x), x) + Bnum(x, MU[None, None].expand_as(x))
          - Bnum(MU, MU)[None, None])/rho2 + bias1
    cm = cov_lut[idx]
    for g, msk in [('all', torch.ones_like(cm)), ('cov', cm), ('unc', ~cm)]:
        if msk.any():
            accE[g]['ef'] += float((mo1 - pf)[msk].pow(2).sum())
            accE[g]['eb'] += float((mo1 - pb)[msk].pow(2).sum())
            accE[g]['dg'] += float((mo1 - MO1_GMEAN[None, None])[msk].pow(2).sum())
a3['explicitness_R2_global_centering'] = {
    g: {'folded': round(1 - accE[g]['ef']/accE[g]['dg'], 4),
        'blind': round(1 - accE[g]['eb']/accE[g]['dg'], 4)} for g in ('all', 'cov', 'unc')}
a3['same_set_note'] = ('the section-103 41.7%-vs-0.1% comparison: both R2 accumulated over the '
                       'IDENTICAL full held position set in one pass (code-verified); same-set '
                       'comparison confirmed')
print("  explicitness split:", json.dumps(a3['explicitness_R2_global_centering']), flush=True)
RES['attack3_coverage'] = a3; save()

# causal runs with per-position CE
base1 = torch.cat([fwd_full1(HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
print(f"  base (attack-3 skeleton) CE {float(base1.mean()):.4f}", flush=True)
runs = {}
for subname in ['floor', 'folded', 'blind']:
    runs[subname] = torch.cat([fwd_full1(HELD[i:i+B0], sub=subname).cpu()
                               for i in range(0, S_, B0)], 0)
    mn, se = dstat_vs(runs[subname], base1)
    a3[f'full_sub_{subname}'] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  full substitution {subname:6s} dCE {mn:+.4f} +- {se:.5f}", flush=True)
# masked substitutions
unc_lut = ~cov_lut
for subname in ['folded', 'blind']:
    for gname, lut in [('covered_only', cov_lut), ('uncovered_only', unc_lut)]:
        ce = torch.cat([fwd_full1(HELD[i:i+B0], sub=subname, only_mask=lut).cpu()
                        for i in range(0, S_, B0)], 0)
        mn, se = dstat_vs(ce, base1)
        a3[f'sub_{subname}_{gname}'] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
        print(f"  substitute {subname} at {gname:14s} dCE {mn:+.4f} +- {se:.5f}", flush=True)
        if subname == 'floor': continue
# floor restricted the same way (for recovered-fraction denominators)
for gname, lut in [('covered_only', cov_lut), ('uncovered_only', unc_lut)]:
    ce = torch.cat([fwd_full1(HELD[i:i+B0], sub='floor', only_mask=lut).cpu()
                    for i in range(0, S_, B0)], 0)
    mn, se = dstat_vs(ce, base1)
    a3[f'sub_floor_{gname}'] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  substitute floor at {gname:14s} dCE {mn:+.4f} +- {se:.5f}", flush=True)
# per-CE-position split of the full-substitution runs (diagnostic; leakage caveat)
cov_ce = cov_lut[HELD[:, :T_-1]].cpu().numpy()
d_fold = (runs['folded'] - base1).numpy(); d_blind = (runs['blind'] - base1).numpy()
d_floor = (runs['floor'] - base1).numpy()
a3['per_position_split_of_full_sub'] = {
    'note': 'CE positions grouped by the CURRENT token coverage at that position; substitution '
            'was global, so upstream-position leakage applies (diagnostic only)',
    'folded_covered_mean': round(float(d_fold[cov_ce].mean()), 4),
    'folded_uncovered_mean': round(float(d_fold[~cov_ce].mean()), 4),
    'blind_covered_mean': round(float(d_blind[cov_ce].mean()), 4),
    'blind_uncovered_mean': round(float(d_blind[~cov_ce].mean()), 4),
    'floor_covered_mean': round(float(d_floor[cov_ce].mean()), 4),
    'floor_uncovered_mean': round(float(d_floor[~cov_ce].mean()), 4),
    'n_covered': int(cov_ce.sum()), 'n_uncovered': int((~cov_ce).sum())}
RES['attack3_coverage'] = a3; save()
del runs

# =====================================================================================
# ATTACK 4: per-token substitution gate for ' the' / ',' / ' D' + capital-class probe
# =====================================================================================
print("\nATTACK 4: per-token gates ...", flush=True)
def tid(s):
    ids = tokzr.encode(s); assert len(ids) == 1, (s, ids); return ids[0]

# lex1 capital class over the vocabulary (VERBATIM qk_xfold_table_2 lex1, capital branch)
BRACKETS_OPEN = set("([{<"); BRACKETS_CLOSE = set(")]}>")
QUOTE_OPEN = set("“‘`"); QUOTE_CLOSE = set("”’"); QUOTE_STRAIGHT = set("\"'")
PUNCT = set(".,;:!?—–-…*|/\\~@#%^&+=_")
COORDINATORS = {"and","or","but","nor","yet","so"}
DETERMINERS = {"the","a","an","this","that","these","those","some","any","each",
               "every","no","another","such"}
PRONOUNS = {"i","we","you","he","she","it","they","them","us","me","him","her","which","who"}
def lex1(s):
    if s == "": return 'other'
    if ('�' in s) or (s == tokzr.eos_token or '<|endoftext|>' in s): return 'special'
    if '\n' in s: return 'newline'
    body = s.strip(); low = body.lower()
    if body == "": return 'other'
    if all(ch in QUOTE_OPEN for ch in body): return 'quote_open'
    if all(ch in QUOTE_CLOSE for ch in body): return 'quote_close'
    if all(ch in QUOTE_STRAIGHT for ch in body): return 'quote'
    if all(ch in BRACKETS_OPEN for ch in body): return 'bracket_open'
    if all(ch in BRACKETS_CLOSE for ch in body): return 'bracket_close'
    if any(ch.isdigit() for ch in body): return 'digit'
    if all((ch in PUNCT or ch in QUOTE_STRAIGHT or ch in QUOTE_OPEN or ch in QUOTE_CLOSE
            or ch in BRACKETS_OPEN or ch in BRACKETS_CLOSE) for ch in body): return 'punct'
    if low in DETERMINERS: return 'determiner'
    if low in COORDINATORS: return 'coordinator'
    if low in PRONOUNS: return 'pronoun'
    if body[0].isupper(): return 'capital'
    lead_space = s.startswith(' ')
    if lead_space and body.isalpha() and len(body) > 1: return 'word'
    if (not lead_space) and body.isalpha() and body[0].islower(): return 'subword'
    return 'other'
print("  building lex1 vocabulary classes ...", flush=True)
capvec = torch.zeros(V, device=DEV)
for t in range(V):
    if lex1(tokzr.decode([t])) == 'capital': capvec[t] = 1.0
print(f"  capital-class vocabulary size {int(capvec.sum())}", flush=True)

a4 = {}
held_np = HELD.cpu().numpy()
for sname in [' the', ',', ' D']:
    t = tid(sname)
    lut = torch.zeros(V, dtype=torch.bool, device=DEV); lut[t] = True
    pos_mask_np = (held_np == t)
    n_tok = int(pos_mask_np.sum())
    ce_f = torch.cat([fwd_full1(HELD[i:i+B0], sub='folded', only_mask=lut).cpu()
                      for i in range(0, S_, B0)], 0)
    ce_b = torch.cat([fwd_full1(HELD[i:i+B0], sub='blind', only_mask=lut).cpu()
                      for i in range(0, S_, B0)], 0)
    mnf, sef = dstat_vs(ce_f, base1); mnb, seb = dstat_vs(ce_b, base1)
    mnd, sed = dstat_vs(ce_b, ce_f)             # blind minus folded: >0 => folded better
    # local: CE at the substituted positions themselves (predicting the next token there)
    loc = pos_mask_np[:, :T_-1]
    locf = float((ce_f - base1).numpy()[loc].mean()) if loc.any() else None
    locb = float((ce_b - base1).numpy()[loc].mean()) if loc.any() else None
    locd = (ce_b - ce_f).numpy()[loc]
    a4[repr(sname)] = {
        'n_held_positions': n_tok,
        'folded_at_token_dCE': {'dCE': round(mnf, 5), 'SE': round(sef, 5)},
        'blind_at_token_dCE': {'dCE': round(mnb, 5), 'SE': round(seb, 5)},
        'blind_minus_folded_paired': {'mean': round(mnd, 5), 'SE': round(sed, 5),
                                      'z': round(mnd/max(sed, 1e-12), 2)},
        'local_dCE_at_token_positions': {
            'folded': round(locf, 4), 'blind': round(locb, 4),
            'blind_minus_folded_mean': round(float(locd.mean()), 4),
            'blind_minus_folded_SE': round(float(locd.std(ddof=1)/np.sqrt(locd.size)), 4)
            if locd.size > 1 else None}}
    print(f"  {sname!r:8s} n={n_tok}: folded {mnf:+.5f}+-{sef:.5f} blind {mnb:+.5f}+-{seb:.5f} "
          f"diff {mnd:+.5f}+-{sed:.5f} | local folded {locf:+.4f} blind {locb:+.4f}", flush=True)
    RES['attack4_token_gate'] = a4; save()

# capital-class probability at ' D' positions
tD = tid(' D')
lutD = torch.zeros(V, dtype=torch.bool, device=DEV); lutD[tD] = True
cap_runs = {}
for subname in [None, 'folded', 'blind']:
    caps = []
    for i in range(0, S_, B0):
        ce, cp, pm = fwd_full1(HELD[i:i+B0], sub=subname,
                               only_mask=(lutD if subname is not None else None),
                               want_probs_mask=lutD, capvec=capvec)
        if len(cp): caps.append(cp)
    cap_runs['base' if subname is None else subname] = torch.cat(caps, 0).numpy()
# ground truth: fraction of actual next tokens after ' D' that are capital-class
nextcap = []
for s in range(S_):
    for p in range(T_-1):
        if held_np[s, p] == tD:
            nextcap.append(float(capvec[held_np[s, p+1]].item()))
a4['capital_probe_at_D'] = {
    'n_positions': int(cap_runs['base'].size),
    'mean_capital_prob': {k: round(float(v.mean()), 4) for k, v in cap_runs.items()},
    'per_position_capital_prob': {k: [round(float(x), 4) for x in v]
                                  for k, v in cap_runs.items()},
    'actual_next_token_capital_fraction': round(float(np.mean(nextcap)), 4) if nextcap else None,
    'note': 'probability mass on the lex1 capital class for the NEXT token, measured AT the '
            "' D' positions; substitutions applied only at ' D' positions"}
print("  capital prob at ' D':", a4['capital_probe_at_D']['mean_capital_prob'],
      "actual-next capital fraction", a4['capital_probe_at_D']['actual_next_token_capital_fraction'],
      flush=True)
RES['attack4_token_gate'] = a4; save()
del base1

# =====================================================================================
# ATTACK 5: h.L7.0 -- paired term-deletion differences + centroid-cosine artifact check
# (machinery VERBATIM qk_xfold_gate_2.py)
# =====================================================================================
print("\nATTACK 5: h.L7.0 ...", flush=True)
LI_T, H_T = 7, 0; LSTAR = 7
GN7 = ['E', 'Ae', 'Ar', 'Me', 'Mr', 'h']
PAIRS7 = [(i, j) for i in range(6) for j in range(i, 6)]
PNAMES7 = [f'{GN7[i]}x{GN7[j]}' for (i, j) in PAIRS7]
NT7 = len(PAIRS7)
K_MRH = PNAMES7.index('Mrxh'); K_HH = PNAMES7.index('hxh'); K_MEH = PNAMES7.index('Mexh')
b7 = m.transformer.h[LSTAR].mlp
Lw7 = b7.Left.weight.detach().float(); Rw7 = b7.Right.weight.detach().float()
Dw7 = b7.Down.weight.detach().float(); bias7 = b7.Down_bias.detach().float()

def pair_terms7(groups, xpre):
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    PL = [g @ Lw7.T for g in groups]; PR = [g @ Rw7.T for g in groups]
    terms = []
    for (i, j) in PAIRS7:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ Dw7.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms

@torch.no_grad()
def fwd7(idx, mode, TMEAN=None, MEANF=None, cap=None, dropk=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cE = torch.ones((), device=DEV)
    SA = torch.zeros_like(x); SM = torch.zeros_like(x); MR = torch.zeros_like(x)
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        if li <= LSTAR:
            cE = blk.lambdas[0]*cE + blk.lambdas[1]
            SA = blk.lambdas[0]*SA; SM = blk.lambdas[0]*SM; MR = blk.lambdas[0]*MR
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh4.reshape(B, T, -1)); x = x + aout
        if li == LI_T:
            Wr = a.c_proj.weight.view(D, NH, HD)
            rh = torch.einsum('btc,dc->btd', yh4[:, :, H_T, :], Wr[:, H_T, :])
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == LSTAR and mode != 'base':
            groups = [cE*x0, SA, aout - rh, SM, MR, rh]
            terms = pair_terms7(groups, x)
            if mode == 'collect':
                for kk in range(NT7): cap['tsum'][kk] += terms[kk].sum(0)
                cap['rhsum'] += rh.sum(0)
                recon = sum(terms) + bias7
                cap['fro_num'] += float((recon - mo).pow(2).sum())
                cap['fro_den'] += float(mo.pow(2).sum())
            elif mode == 'drop':
                new = MEANF.unsqueeze(0).expand(B, -1, -1)
                for kk in range(NT7):
                    if kk != dropk: new = new + (terms[kk] - TMEAN[kk])
                mo = new.to(x.dtype)
            elif mode == 'capture':
                cap['h'].append(rh.cpu())
            del terms, groups
        x = x + mo
        if li < LSTAR:
            SA = SA + aout; SM = SM + MR; MR = mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

cap7 = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT7)],
        'rhsum': torch.zeros(T_, D, device=DEV), 'fro_num': 0.0, 'fro_den': 0.0}
base7_l = []
for i in range(0, S_, B0):
    base7_l.append(fwd7(HELD[i:i+B0], 'collect', cap=cap7).cpu())
base7 = torch.cat(base7_l, 0)
TMEAN7 = torch.stack([t/S_ for t in cap7['tsum']])
MEANF7 = TMEAN7.sum(0) + bias7
RH_MEAN = cap7['rhsum']/S_
gate7 = (cap7['fro_num']/cap7['fro_den'])**0.5
print(f"  layer-7 fold gate {gate7:.2e}; base CE {float(base7.mean()):.4f}", flush=True)
assert gate7 < 1e-4

a5 = {'fold_gate': gate7}
ced = {}
for nm, kk in [('Mrxh', K_MRH), ('hxh', K_HH), ('Mexh', K_MEH)]:
    out = []
    for i in range(0, S_, B0):
        out.append(fwd7(HELD[i:i+B0], 'drop', TMEAN=TMEAN7, MEANF=MEANF7, dropk=kk).cpu())
    ced[nm] = torch.cat(out, 0)
    mn, se = dstat_vs(ced[nm], base7)
    a5[f'drop_{nm}'] = {'dCE': round(mn, 5), 'SE': round(se, 5)}
    print(f"  drop_{nm:5s} dCE {mn:+.5f} +- {se:.5f}", flush=True)
for na, nb in [('Mrxh', 'hxh'), ('Mrxh', 'Mexh'), ('hxh', 'Mexh')]:
    mn, se = dstat_vs(ced[na], ced[nb])
    a5[f'paired_diff_{na}_minus_{nb}'] = {'mean': round(mn, 5), 'SE': round(se, 5),
                                          'z': round(mn/max(se, 1e-12), 2)}
    print(f"  paired diff drop_{na} - drop_{nb}: {mn:+.5f} +- {se:.5f} z={mn/max(se,1e-12):.2f}",
          flush=True)
RES['attack5_h70'] = a5; save()

# centroid-cosine artifact check
cap7b = {'h': []}
for i in range(0, S_, B0):
    fwd7(HELD[i:i+B0], 'capture', TMEAN=TMEAN7, cap=cap7b)
Hs = torch.cat(cap7b['h'], 0).numpy()                    # (S,T,D) raw head write
dce7 = (ced['Mrxh'] - base7).numpy()
hnorm = np.linalg.norm(Hs, axis=-1)
hq = np.quantile(hnorm[:, :T_-1], 0.75)
sel = hnorm[:, :T_-1] >= hq
d = dce7.copy(); d[~sel] = 0.0
si, pi = np.unravel_index(np.argsort(-d, axis=None), d.shape)
help_pos = list(zip(si[:200], pi[:200]))
si2, pi2 = np.unravel_index(np.argsort(d, axis=None), d.shape)
hurt_pos = list(zip(si2[:200], pi2[:200]))

def centroid(X, poss):
    return np.stack([X[s, p] for s, p in poss], 0).mean(0)

def coss(a, b): return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

gmean = Hs.reshape(-1, D).mean(0)
u = gmean/np.linalg.norm(gmean)
Hsub = Hs - gmean[None, None]
Hproj = Hs - np.einsum('stp,p->st', Hs, u)[..., None]*u[None, None]
cos_raw = coss(centroid(Hs, help_pos), centroid(Hs, hurt_pos))
cos_sub = coss(centroid(Hsub, help_pos), centroid(Hsub, hurt_pos))
cos_prj = coss(centroid(Hproj, help_pos), centroid(Hproj, hurt_pos))
a5['centroid_cosines'] = {
    'raw_help_vs_hurt': round(cos_raw, 4),
    'after_subtracting_global_mean_write': round(cos_sub, 4),
    'after_projecting_out_global_mean_direction': round(cos_prj, 4),
    'help_centroid_cos_to_global_mean': round(coss(centroid(Hs, help_pos), gmean), 4),
    'hurt_centroid_cos_to_global_mean': round(coss(centroid(Hs, hurt_pos), gmean), 4),
    'global_mean_norm_over_mean_write_norm': round(float(np.linalg.norm(gmean)/hnorm.mean()), 4)}
print("  centroid cosines:", json.dumps(a5['centroid_cosines']), flush=True)
RES['attack5_h70'] = a5; save()
print("\nQK REDTEAM XFOLD DONE", flush=True)
