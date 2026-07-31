"""ADVERSARIAL RED-TEAM of the fold-first attribution arc (sections 84-86), part 1 of 2.

ATTACK 1 (gauge smuggling, section 86): the stream-pair terms share the gauge scalar 1/rho^2 with
rho^2 = ||x_pre||^2/D computed from the FULL input (all four streams). So "keep 5 terms" still sees
full-input information through the gauge. Recompute keep-top-5 sufficiency with the gauge computed
(a) from the KEPT groups' sum only (E+M0+A1 -- the information the kept terms are entitled to), and
(b) from the per-position MEAN input (removes ALL per-sample dependence). Also quantify how much of
the hub's centered output energy the per-sample gauge variation itself carries (freeze the gauge at
the per-position mean, keep ALL terms; plus per-position correlation of 1/rho^2 with the deviation
norm of mo1).

ATTACK 2 (per-position-mean confound, section 84): all sufficiency configs are "per-position mean +
kept part"; the per-position mean carries positional information. Rerun keep-top-144 SVD sufficiency
with a GLOBAL (position-independent) mean, and compare the two mean-only floors.

ATTACK 3 (dead-row triviality, section 86): attention-0 and MLP-0 both enter the layer-1 input with
the same lambda coefficient (~0.0127) while the embedding enters at ~8.15; MLP-0's terms are top
contributors yet attention-0's are dead. Measure the RAW stream content norms (before coefficient)
and the coefficiented shares of x_pre to decide: dead content, or dead only by coefficient?

Machinery VERBATIM from qk_hub_streampairs.py (streams, pair terms, keep-subset harness) and
qk_hub_hierarchy.py (train-gram SVD basis, keep-projection harness). Held FW[448:600,:128], paired
standard errors, batch 6, <4GB. Output: qk_redteam_fold.json (script _2 extends it)."""
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_redteam_fold.json'

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
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV); B0 = 6; LI = 1
S_, T_ = HELD.shape
b0, b1 = m.transformer.h[0], m.transformer.h[1]
L1 = b1.mlp.Left.weight.detach().float(); R1 = b1.mlp.Right.weight.detach().float()
D1w = b1.mlp.Down.weight.detach().float(); bias1 = b1.mlp.Down_bias.detach().float()
lamE = (b1.lambdas[0] * (b0.lambdas[0] + b0.lambdas[1]) + b1.lambdas[1]).item()
lam0 = b1.lambdas[0].item()
print(f"bilin18 NL={NL} D={D} hidden={L1.shape[0]}  lamE={lamE:.4f} lam0={lam0:.4f}", flush=True)

PAIRS = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
PNAMES = ['ExE', 'ExA0', 'ExM0', 'ExA1', 'A0xA0', 'A0xM0', 'A0xA1', 'M0xM0', 'M0xA1', 'A1xA1']
NT = len(PAIRS)
IDX = {n: k for k, n in enumerate(PNAMES)}
# section-86 keep-top-5 (energy order M0xA1, A1xA1, M0xM0, ExM0, ExA1); kept groups = {E, M0, A1}
KEEP5 = [IDX[n] for n in ['M0xA1', 'A1xA1', 'M0xM0', 'ExM0', 'ExA1']]

def pair_numers(E_, A0_, M0_, A1_):
    """The 10 pre-gauge numerators N_k (list of (B,T,D)); term_k = N_k / rho2 (shared gauge).
    Polarization VERBATIM from qk_hub_streampairs.pair_terms, gauge division factored out."""
    Ss = [E_, A0_, M0_, A1_]
    PL = [s @ L1.T for s in Ss]; PR = [s @ R1.T for s in Ss]
    numers = []
    for (i, j) in PAIRS:
        n_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ D1w.T)
        if i != j: n_ = 2.0 * n_
        numers.append(n_)
    return numers

gram = torch.zeros(D, D, device=DEV)   # train gram of mo1 (attack 2, verbatim qk_hub_hierarchy)

@torch.no_grad()
def fwd(idx, mode=None, stats=None, MX=None, MO=None,
        subset=None, gauge='full', TM=None, MEANF=None,
        keepP=None, keepmean=None, want_gram=False):
    """Forward verbatim from qk_hub_streampairs.py / qk_hub_hierarchy.py.
    mode: None full | 'g1' pass A1 collect | 'g2' pass A2 collect (needs MX, MO)
          | 'subset' (mo1 -> MEANF + sum_kept (N_k/rho2_gauge - TM[k])), gauge in {full,kept,meanx}
          | 'svdkeep' (mo1 -> keepmean + dev@P@P.T; keepP None = mean only)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    a0c = m0c = None
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
        if li == 0: a0c, m0c = aout, mo
        if li == LI:
            if want_gram: gram.add_(torch.einsum('btd,bte->de', mo, mo))
            if mode in ('g1', 'g2', 'subset'):
                E_ = lamE * x0; A0_ = lam0 * a0c; M0_ = lam0 * m0c; A1_ = aout
                rho2 = x.pow(2).sum(-1, keepdim=True) / D                       # full gauge
                if mode == 'g1':
                    stats['xsum'] += x.sum(0); stats['mosum'] += mo.sum(0)
                    gs = E_ + A0_ + M0_ + A1_
                    stats['grp_err'] = max(stats['grp_err'],
                        float(((gs - x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).max()))
                    # raw and coefficiented stream norms (attack 3)
                    for nm, raw, co in (('E', x0, E_), ('A0', a0c, A0_), ('M0', m0c, M0_), ('A1', aout, A1_)):
                        stats['rawsq'][nm] += float(raw.pow(2).sum())
                        stats['cosq'][nm] += float(co.pow(2).sum())
                    stats['xsq'] += float(x.pow(2).sum())
                    numers = pair_numers(E_, A0_, M0_, A1_)
                    for kk in range(NT): stats['tsum'][kk] += (numers[kk]/rho2).sum(0)
                    recon = sum(numers)/rho2 + bias1
                    num = (recon - mo).norm(dim=-1); den = mo.norm(dim=-1).clamp_min(1e-8)
                    stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                    stats['fro_num'] += float((recon - mo).pow(2).sum()); stats['fro_den'] += float(mo.pow(2).sum())
                    del numers
                elif mode == 'g2':
                    kept = E_ + M0_ + A1_                                       # kept groups (no A0)
                    rho2_k = kept.pow(2).sum(-1, keepdim=True) / D
                    rho2_mx = (MX.pow(2).sum(-1, keepdim=True) / D).unsqueeze(0)  # (1,T,1) per-position
                    numers = pair_numers(E_, A0_, M0_, A1_)
                    for kk in range(NT):
                        stats['tsum_k'][kk] += (numers[kk]/rho2_k).sum(0)
                        stats['tsum_mx'][kk] += (numers[kk]/rho2_mx).sum(0)
                    # gauge-frozen full output vs true: energy the per-sample gauge carries
                    mo_mx = sum(numers)/rho2_mx + bias1
                    dev = mo - MO.unsqueeze(0)
                    stats['gfreeze_num'] += float((mo - mo_mx).pow(2).sum())
                    stats['dev_den'] += float(dev.pow(2).sum())
                    # per-position correlation stats of g=1/rho2 with ||dev(mo1)||
                    gsc = (1.0/rho2).squeeze(-1); nsc = dev.norm(dim=-1)        # (B,T)
                    stats['Sg'] += gsc.sum(0); stats['Sgg'] += (gsc*gsc).sum(0)
                    stats['Sn'] += nsc.sum(0); stats['Snn'] += (nsc*nsc).sum(0)
                    stats['Sgn'] += (gsc*nsc).sum(0)
                    stats['Sr'] += rho2.squeeze(-1).sum(0); stats['Srr'] += rho2.squeeze(-1).pow(2).sum(0)
                    del numers
                elif mode == 'subset':
                    if gauge == 'full':
                        rr = rho2
                    elif gauge == 'kept':
                        kept = E_ + M0_ + A1_
                        rr = kept.pow(2).sum(-1, keepdim=True) / D
                    elif gauge == 'meanx':
                        rr = (MX.pow(2).sum(-1, keepdim=True) / D).unsqueeze(0)
                    numers = pair_numers(E_, A0_, M0_, A1_)
                    new = MEANF.unsqueeze(0).expand(B, -1, -1)
                    for kk in subset: new = new + (numers[kk]/rr - TM[kk])
                    mo = new.to(x.dtype)
                    del numers
            elif mode == 'svdkeep':
                dev = mo - keepmean.unsqueeze(0)
                mo = keepmean.unsqueeze(0) + ((dev @ keepP) @ keepP.T if keepP is not None
                                              else torch.zeros_like(dev))
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

res = {'meta': {'model': 'bilin18', 'held': 'FW[448:600,:128]', 'batch': B0,
                'currency': 'delta cross-entropy per valid held position (nats), paired standard error',
                'lamE': lamE, 'lam0': lam0}}

# ---------------- PASS A1: means, gates, stream norms, full-gauge term means ----------------
print("PASS A1: means + gates + stream norms + full-gauge term means ...", flush=True)
st = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
      'xsum': torch.zeros(T_, D, device=DEV), 'mosum': torch.zeros(T_, D, device=DEV),
      'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0, 'grp_err': 0.0,
      'rawsq': {k: 0.0 for k in 'E A0 M0 A1'.split()},
      'cosq': {k: 0.0 for k in 'E A0 M0 A1'.split()}, 'xsq': 0.0}
for i in range(0, S_, B0): fwd(HELD[i:i+B0], mode='g1', stats=st)
MX = st['xsum']/S_; MO = st['mosum']/S_
TM_full = torch.stack([t/S_ for t in st['tsum']])
MEANF = TM_full.sum(0) + bias1
gate_fro = (st['fro_num']/st['fro_den'])**0.5
print(f"GATE recon global {gate_fro:.2e} maxpos {st['maxrel']:.2e} groupsum {st['grp_err']:.2e}", flush=True)
assert gate_fro < 1e-4, "decomposition gate FAILED"
NP = S_ * T_
res['gate'] = {'recon_rel_err_global': gate_fro, 'recon_rel_err_max_pos': st['maxrel'],
               'group_sum_rel_err_max': st['grp_err']}
# attack 3 numbers
rms_raw = {k: (st['rawsq'][k]/NP)**0.5 for k in st['rawsq']}
rms_co = {k: (st['cosq'][k]/NP)**0.5 for k in st['cosq']}
share_x = {k: st['cosq'][k]/st['xsq'] for k in st['cosq']}
res['attack3_dead_row'] = {
    'rms_norm_raw_content': {k: round(v, 4) for k, v in rms_raw.items()},
    'rms_norm_coefficiented': {k: round(v, 4) for k, v in rms_co.items()},
    'share_of_xpre_energy_coefficiented': {k: round(v, 5) for k, v in share_x.items()},
    'raw_ratio_A0_over_M0': round(rms_raw['A0']/rms_raw['M0'], 4),
    'note': 'A0 and M0 share the same coefficient lam0; E raw content is the unit-rms embedding'}
print("ATTACK 3 raw rms norms:", {k: round(v, 3) for k, v in rms_raw.items()},
      "| coefficiented:", {k: round(v, 3) for k, v in rms_co.items()},
      "| A0/M0 raw ratio:", round(rms_raw['A0']/rms_raw['M0'], 4), flush=True)

# ---------------- PASS A2: alt-gauge term means + gauge-variance quantification ----------------
print("PASS A2: kept-sum and mean-x gauge term means + gauge stats ...", flush=True)
st2 = {'tsum_k': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
       'tsum_mx': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
       'gfreeze_num': 0.0, 'dev_den': 0.0,
       'Sg': torch.zeros(T_, device=DEV), 'Sgg': torch.zeros(T_, device=DEV),
       'Sn': torch.zeros(T_, device=DEV), 'Snn': torch.zeros(T_, device=DEV),
       'Sgn': torch.zeros(T_, device=DEV),
       'Sr': torch.zeros(T_, device=DEV), 'Srr': torch.zeros(T_, device=DEV)}
for i in range(0, S_, B0): fwd(HELD[i:i+B0], mode='g2', stats=st2, MX=MX, MO=MO)
TM_kept = torch.stack([t/S_ for t in st2['tsum_k']])
TM_mx = torch.stack([t/S_ for t in st2['tsum_mx']])
n = float(S_)
mg = st2['Sg']/n; mn_ = st2['Sn']/n
vg = (st2['Sgg']/n - mg*mg).clamp_min(0); vn = (st2['Snn']/n - mn_*mn_).clamp_min(0)
cov = st2['Sgn']/n - mg*mn_
corr = (cov/(vg.sqrt()*vn.sqrt()).clamp_min(1e-12)).cpu()
mr = st2['Sr']/n; vr = (st2['Srr']/n - mr*mr).clamp_min(0)
cv_rho2 = (vr.sqrt()/mr).cpu()
gfrac = st2['gfreeze_num']/st2['dev_den']
res['attack1_gauge_variance'] = {
    'frozen_gauge_residual_energy_frac_of_centered_mo1':  round(gfrac, 5),
    'note_frozen': 'keep ALL terms, gauge frozen at per-position mean input; '
                   '||mo1_true - mo1_frozen||^2 / ||mo1 - per-position mean||^2',
    'corr_invrho2_vs_devnorm_mean_over_positions': round(float(corr.mean()), 4),
    'corr_invrho2_vs_devnorm_median': round(float(corr.median()), 4),
    'cv_rho2_mean_over_positions': round(float(cv_rho2.mean()), 4)}
print(f"gauge-frozen residual energy fraction {gfrac:.4f} | corr(1/rho2, devnorm) mean "
      f"{float(corr.mean()):+.3f} median {float(corr.median()):+.3f} | CV(rho2) {float(cv_rho2.mean()):.3f}", flush=True)

# ---------------- base CE ----------------
print("BASE pass ...", flush=True)
base = torch.cat([fwd(HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
print(f"base CE {float(base.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

def run_subset(subset, gauge, TM):
    out = []
    for i in range(0, S_, B0):
        out.append(fwd(HELD[i:i+B0], mode='subset', subset=subset, gauge=gauge,
                       TM=TM, MEANF=MEANF, MX=MX).cpu())
    return torch.cat(out, 0)

# ---------------- ATTACK 1: keep-top-5 under the three gauges ----------------
res['attack1_configs'] = {}
print("ATTACK 1: keep-top-5 under full / kept-sum / mean-x gauges ...", flush=True)
for name, sub, gg, TM in [
        ('mean_only_repro', [], 'full', TM_full),
        ('keep5_full_gauge_repro', KEEP5, 'full', TM_full),
        ('keep5_gauge_keptsum', KEEP5, 'kept', TM_kept),
        ('keep5_gauge_meanx', KEEP5, 'meanx', TM_mx),
        ('allterms_gauge_meanx', list(range(NT)), 'meanx', TM_mx),
        ('top3_full_gauge_repro', KEEP5[:3], 'full', TM_full),
        ('top3_gauge_meanx', KEEP5[:3], 'meanx', TM_mx)]:
    mn, se = dstat(run_subset(sub, gg, TM))
    res['attack1_configs'][name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:26s} dCE {mn:+.4f} +- {se:.5f}", flush=True)

# ---------------- ATTACK 2: SVD keep-144 under per-position vs global mean ----------------
print("ATTACK 2: train gram pass ...", flush=True)
for i in range(0, TRAIN.shape[0], B0): fwd(TRAIN[i:i+B0], want_gram=True)
SVD = torch.linalg.eigh(gram)[1].flip(1)
GMEAN = MO.mean(0)                                  # global (position-independent) mean of mo1
GM_T = GMEAN.unsqueeze(0).expand(T_, D).contiguous()
def run_keep(P, mean):
    out = []
    for i in range(0, S_, B0):
        out.append(fwd(HELD[i:i+B0], mode='svdkeep', keepP=P, keepmean=mean).cpu())
    return torch.cat(out, 0)
res['attack2_configs'] = {}
P144 = SVD[:, :144].contiguous()
for name, P, mean in [
        ('mean_only_perpos', None, MO), ('mean_only_global', None, GM_T),
        ('keep144_perpos_mean_repro', P144, MO), ('keep144_global_mean', P144, GM_T),
        ('keep288_global_mean', SVD[:, :288].contiguous(), GM_T),
        ('keep145_global_plus_meandir', None, None)]:
    if name == 'keep145_global_plus_meandir':
        # global mean + 144 SVD directions + the (single) direction of positional mean variation? ->
        # fairer: augment the basis with the top principal direction of the per-position means
        pm = MO - GMEAN.unsqueeze(0)
        u = torch.linalg.svd(pm, full_matrices=False)[2][0]     # top right-singular vector (D,)
        Paug = torch.cat([P144, u.unsqueeze(1)], 1)
        Paug = torch.linalg.qr(Paug)[0].contiguous()
        P, mean = Paug, GM_T
    mn, se = dstat(run_keep(P, mean))
    res['attack2_configs'][name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:28s} dCE {mn:+.4f} +- {se:.5f}", flush=True)

json.dump(res, open(OUT, 'w'), indent=1)
print("QK REDTEAM FOLD PART 1 DONE", flush=True)
