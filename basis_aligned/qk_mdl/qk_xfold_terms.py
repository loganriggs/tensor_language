"""CROSS-LAYER COMPOSITION experiment 1 -- SECOND-ORDER PROVENANCE (terms of terms) at block 3.

Logan's directive: the tensor network lets us fold any component into its consumers. The section-89
census wrote block 3's input as 5 groups (E, Ae, Ar, Me, Mr); here we EXPAND the mlp-earlier group
Me into its per-block upstream constituents (block 0's mlp write, block 1's mlp write; Mr = block 2's
mlp write is already separate), so block 3's bilinear form becomes an exact sum over CROSS-LAYER
term pairs among SIX groups {E, Ae, Ar, M0, M1, M2} -> 21 terms.
  GATE 1: group sum == x_pre3 (rel err < 1e-4);  GATE 2: 21 terms + bias == mo3 (rel err < 1e-4);
  GATE 3: mean-only floor reproduces census 0.6163; merged configs reproduce census keep-alone
          numbers (M2xM2 == only_MrxMr 0.2483; {M0xM2,M1xM2} == only_MexMr 0.2661;
          ArxM2 == only_ArxMr 0.2564).
Then: causal ranking of cross-layer terms (keep-alone, greedy top-k, drop-one), diagonal/cross,
M2/M1/M0-involving groups, and token-identity variance fraction for the top terms (who supplies
CONTEXT -- terms with low token-explained variance are contextual). Specifically: is the dominant
term M2xM2 (the iterated square = fourth-order in the stream), and which pairing gates the mixer?

Machinery VERBATIM from qk_allterm_census.py (pair_terms polarization, gates, keep-subset harness,
forward skeleton); per-block mlp stream tracking replaces the census SM/MR accumulators (verified
equivalent by the merged-config gates). Held FW[448:600,:128], paired SE, batch 4, <4GB."""
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_xfold_terms.json'

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
HELD = FW[448:600, :128].to(DEV); B0 = 4; LI = 3
S_, T_ = HELD.shape
print(f"bilin18 NL={NL} D={D} NH={NH} held {S_}x{T_}  target layer {LI}", flush=True)

# ---- SIX groups at layer 3: E, Ae, Ar, M0, M1, M2 (M2 == census Mr; M0+M1 == census Me) ----
GNAMES = ['E', 'Ae', 'Ar', 'M0', 'M1', 'M2']
NG = 6
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
NT = len(PAIRS)   # 21
DIAG = [k for k, (i, j) in enumerate(PAIRS) if i == j]
CROSS = [k for k, (i, j) in enumerate(PAIRS) if i != j]
IDX = {n: k for k, n in enumerate(PNAMES)}

b3 = m.transformer.h[LI].mlp
Lw = b3.Left.weight.detach().float(); Rw = b3.Right.weight.detach().float()
Dw = b3.Down.weight.detach().float(); bias = b3.Down_bias.detach().float()

def pair_terms(groups, xpre):
    """21 interaction terms (list of (B,T,D)), sharing the common 1/rho^2 gauge; sum+bias == mo3.
    VERBATIM construction from qk_allterm_census.pair_terms, generalized to 6 groups."""
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    PL = [g @ Lw.T for g in groups]; PR = [g @ Rw.T for g in groups]
    terms = []
    for (i, j) in PAIRS:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ Dw.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms

@torch.no_grad()
def fwd(idx, mode=None, subset=None, TMEAN=None, MEANF=None, stats=None):
    """Forward verbatim from qk_allterm_census.py, with PER-BLOCK mlp stream accumulators.
    mode: None | 'collect' | 'energy' | 'subset' | 'tokvar' (per-token accumulation of top terms)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    track = mode is not None
    if track:
        cE = torch.ones((), device=DEV)
        SA = torch.zeros_like(x)
        Ml = []                                   # per-block mlp writes, lambda-decayed
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        if track and li <= LI:
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
        if track and li == LI:
            groups = [cE*x0, SA, aout, Ml[0], Ml[1], Ml[2]]   # E, Ae, Ar, M0, M1, M2; x is x_pre here
            terms = pair_terms(groups, x)
            if mode == 'collect':
                gs = sum(groups)
                stats['grp_err'] = max(stats['grp_err'],
                                       float(((gs - x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).max()))
                for kk in range(NT): stats['tsum'][kk] += terms[kk].sum(0)
                stats['mosum'] += mo.sum(0)
                recon = sum(terms) + bias
                num = (recon - mo).norm(dim=-1); den = mo.norm(dim=-1).clamp_min(1e-8)
                stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                stats['fro_num'] += float((recon - mo).pow(2).sum()); stats['fro_den'] += float(mo.pow(2).sum())
            elif mode == 'energy':
                for kk in range(NT): stats['te'][kk] += float((terms[kk] - TMEAN[kk]).pow(2).sum())
                stats['tot'] += float((mo - MEANF).pow(2).sum())
            elif mode == 'tokvar':
                fl = LUT[idx.reshape(-1)]          # compact held-vocab ids
                for kk in stats['which']:
                    stats['tsum_tok'][kk].index_add_(0, fl, terms[kk].reshape(-1, D))
                stats['tcnt'].index_add_(0, fl, torch.ones_like(fl, dtype=torch.float))
            elif mode == 'subset':
                new = MEANF.unsqueeze(0).expand(B, -1, -1)
                for kk in subset: new = new + (terms[kk] - TMEAN[kk])
                mo = new.to(x.dtype)
            del terms, groups
        x = x + mo
        if track and li < LI:
            SA = SA + aout; Ml.append(mo)
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

# ---------------- PASS 1: term means + decomposition gates ----------------
print("PASS 1: term means + reconstruction gates ...", flush=True)
st = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
      'mosum': torch.zeros(T_, D, device=DEV), 'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0,
      'grp_err': 0.0}
for i in range(0, S_, B0): fwd(HELD[i:i+B0], mode='collect', stats=st)
TMEAN = torch.stack([t/S_ for t in st['tsum']])
MO_MEAN = st['mosum']/S_
MEANF = TMEAN.sum(0) + bias
gate_fro = (st['fro_num']/st['fro_den'])**0.5
mean_consist = float((MEANF - MO_MEAN).norm()/MO_MEAN.norm())
print(f"GATE recon: global {gate_fro:.2e} | maxpos {st['maxrel']:.2e} | groupsum {st['grp_err']:.2e} "
      f"| meancons {mean_consist:.2e}", flush=True)
assert gate_fro < 1e-4 and st['grp_err'] < 1e-4, "decomposition gate FAILED"

# ---------------- PASS 2: base CE + centered term energies ----------------
print("PASS 2: base + term energies ...", flush=True)
st2 = {'te': [0.0]*NT, 'tot': 0.0}
ces = []
for i in range(0, S_, B0):
    ce = fwd(HELD[i:i+B0], mode='energy', TMEAN=TMEAN, MEANF=MEANF, stats=st2); ces.append(ce.cpu())
base = torch.cat(ces, 0)
shares = {PNAMES[k]: round(st2['te'][k]/st2['tot'], 4) for k in range(NT)}
order = sorted(range(NT), key=lambda k: -st2['te'][k])
print(f"base CE {float(base.mean()):.4f}", flush=True)
print("energy rank:", [(PNAMES[k], shares[PNAMES[k]]) for k in order[:10]], flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

def run(subset):
    out = []
    for i in range(0, S_, B0):
        out.append(fwd(HELD[i:i+B0], mode='subset', subset=subset, TMEAN=TMEAN, MEANF=MEANF).cpu())
    return torch.cat(out, 0)

# ---------------- PASS 3: subset configurations ----------------
CENSUS3 = {'floor': 0.6163, 'only_MrxMr': 0.2483, 'only_ArxMr': 0.2564, 'only_MexMr': 0.2661}
cfgs = [('mean_only', []), ('all_terms', list(range(NT)))]
cfgs += [(f'only_{PNAMES[k]}', [k]) for k in order[:10]]
cfgs += [(f'top{kk}_energy', order[:kk]) for kk in range(2, 9)]
cfgs += [(f'allbut_{PNAMES[k]}', [j for j in range(NT) if j != k]) for k in order[:6]]
cfgs += [('diagonal', DIAG), ('cross', CROSS)]
# census-consistency merged configs
cfgs += [('census_only_MexMr_merged', [IDX['M0xM2'], IDX['M1xM2']])]
# stream-involving groups
for g, gn in enumerate(GNAMES):
    inv = [k for k, (i, j) in enumerate(PAIRS) if (i == g or j == g)]
    cfgs.append((f'{gn}_involving', inv))
    cfgs.append((f'drop_{gn}_involving', [k for k in range(NT) if k not in inv]))
res = {'meta': {'model': 'bilin18', 'layer': LI, 'held': 'FW[448:600,:128]', 'batch': B0,
                'groups': GNAMES, 'n_terms': NT,
                'note': 'census Me expanded into per-block M0, M1; M2 == census Mr',
                'base_ce': round(float(base.mean()), 4)},
       'gate': {'recon_rel_err_global': gate_fro, 'recon_rel_err_max_pos': st['maxrel'],
                'group_sum_rel_err_max': st['grp_err'], 'mean_consistency': mean_consist},
       'census_ref': CENSUS3,
       'energy_shares': shares, 'energy_rank': [PNAMES[k] for k in order], 'configs': {}}
print(f"PASS 3: {len(cfgs)} subset configurations ...", flush=True)
t0 = time.time()
for name, sub in cfgs:
    mn, se = dstat(run(sub))
    res['configs'][name] = {'terms': [PNAMES[k] for k in sub], 'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:26s} [{len(sub):2d}] dCE {mn:+.4f} +- {se:.5f}   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)

# census gates on merged configs
c = res['configs']
res['census_gates'] = {
    'floor_matches': [c['mean_only']['dCE'], CENSUS3['floor']],
    'M2xM2_vs_only_MrxMr': [c['only_M2xM2']['dCE'] if 'only_M2xM2' in c else None, CENSUS3['only_MrxMr']],
    'ArxM2_vs_only_ArxMr': [c['only_ArxM2']['dCE'] if 'only_ArxM2' in c else None, CENSUS3['only_ArxMr']],
    'merged_MexMr': [c['census_only_MexMr_merged']['dCE'], CENSUS3['only_MexMr']]}
print("census gates:", json.dumps(res['census_gates']), flush=True)

# ---------------- PASS 4: token-identity variance fraction for the top terms (context test) ----
print("PASS 4: token-identity variance for top terms ...", flush=True)
TOPK = order[:6]
HV = torch.unique(HELD.reshape(-1))                # compact held vocabulary
NHV = HV.numel()
LUT = torch.zeros(V, dtype=torch.long, device=DEV); LUT[HV] = torch.arange(NHV, device=DEV)
print(f"held vocabulary size {NHV}", flush=True)
st3 = {'which': TOPK, 'tsum_tok': {kk: torch.zeros(NHV, D, device=DEV) for kk in TOPK},
       'tcnt': torch.zeros(NHV, device=DEV)}
for i in range(0, S_, B0):
    fwd(HELD[i:i+B0], mode='tokvar', TMEAN=TMEAN, MEANF=MEANF, stats=st3)
# second pass: residual variance around token-conditional means (and shuffled control)
held_flat = HELD.reshape(-1)
perm = torch.randperm(held_flat.numel(), device=DEV)
shuf_flat = held_flat[perm]
st4 = {'which': TOPK, 'ss_tok': {kk: 0.0 for kk in TOPK}, 'ss_shuf': {kk: 0.0 for kk in TOPK},
       'ss_tot': {kk: 0.0 for kk in TOPK}}
cnt = st3['tcnt'].clamp_min(1.0)
TOKMEAN = {kk: st3['tsum_tok'][kk]/cnt[:, None] for kk in TOPK}
GLOBM = {kk: TMEAN[kk].mean(0) for kk in TOPK}

@torch.no_grad()
def fwd_tokres(idx, shuf_idx, stats):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cE = torch.ones((), device=DEV); SA = torch.zeros_like(x); Ml = []
    for li in range(LI+1):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        cE = blk.lambdas[0]*cE + blk.lambdas[1]
        SA = blk.lambdas[0]*SA; Ml = [blk.lambdas[0]*mm for mm in Ml]
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == LI:
            groups = [cE*x0, SA, aout, Ml[0], Ml[1], Ml[2]]
            terms = pair_terms(groups, x)
            flr = LUT[idx.reshape(-1)]; fls = LUT[shuf_idx.reshape(-1)]
            for kk in stats['which']:
                tv = terms[kk].reshape(-1, D)
                stats['ss_tok'][kk] += float((tv - TOKMEAN[kk][flr]).pow(2).sum())
                stats['ss_shuf'][kk] += float((tv - TOKMEAN[kk][fls]).pow(2).sum())
                stats['ss_tot'][kk] += float((tv - GLOBM[kk][None]).pow(2).sum())
        x = x + mo
        if li < LI: SA = SA + aout; Ml.append(mo)

SHUF = shuf_flat.reshape(S_, T_)
for i in range(0, S_, B0):
    fwd_tokres(HELD[i:i+B0], SHUF[i:i+B0], st4)
res['token_variance_fraction'] = {}
for kk in TOPK:
    r2 = 1.0 - st4['ss_tok'][kk]/st4['ss_tot'][kk]
    r2s = 1.0 - st4['ss_shuf'][kk]/st4['ss_tot'][kk]
    res['token_variance_fraction'][PNAMES[kk]] = {'token_R2': round(r2, 4), 'shuffled_control_R2': round(r2s, 4)}
    print(f"  {PNAMES[kk]:8s} token-identity R2 {r2:.4f} (shuffled control {r2s:.4f})", flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
print("QK XFOLD TERMS DONE", flush=True)
