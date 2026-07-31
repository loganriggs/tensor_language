"""DEPTH-FIRST ALGORITHM ARC #1, script 1: WHAT ALGORITHM does LAYER 2's "square the previous
block's output" implement? Section 89: layer 2's MrxMr term (bilinear self-interaction of the
lambda-decayed layer-1 MLP output stream) alone leaves 0.093 of the 0.739-nat mean-ablation
floor; {MrxMr, ArxMr} leaves 0.011. A bilinear square T2(Mr,Mr) computes PAIRWISE PRODUCTS of
layer-1 output features. Candidate readings:
  H1 FEATURE COINCIDENCE / AND-GATES  -- cross-products of distinct layer-1 features load-bearing;
  H2 MAGNITUDE/CONFIDENCE NONLINEARITY -- diagonal self-products carry it, cross negligible;
  H3 QUADRATIC BASIS FOR DOWNSTREAM    -- (script 2) consumed by later layers, not local readout.

DISCRIMINATING TEST (this script): decompose Mr into its top-K SVD directions (TRAIN gram of
layer-1 MLP output, construction VERBATIM qk_mlp1_tail.py), then
  T2(Mr,Mr)/rho2 = sum_{i,j} P_ij B_ij + remainder,   P_ij = a_i a_j / rho2,
  B_ij = 0.5*(L2 u_i * R2 u_j + L2 u_j * R2 u_i) @ Down2.T   (FIXED D-vectors, cached once),
so per position everything is coefficient products -- exact, cheap. Split into DIAGONAL (i=j),
OFF-DIAGONAL cross (i!=j), and the beyond-K remainder. Measure: (a) centered ENERGY split;
(b) CAUSAL split with the census keep/drop harness (mo2 -> per-position decomposition mean +
kept deviations), delta cross-entropy +- paired standard error on held FW[448:600,:128];
(c) per-PAIR energies (rank-1 formula) + causal necessity of top pairs + pair sufficiency curve.

Machinery VERBATIM from qk_allterm_census.py (5-group streams, pair_terms polarization through
the actual layer-2 mlp weights, recon gate, per-position means, keep-subset harness, dstat) /
qk_hub_streampairs.py lineage; SVD dirs VERBATIM qk_mlp1_tail.py construction (LI=1 mo gram on
TRAIN). GPU guard verbatim. Batch 6, footprint <4GB. Output: qk_arc_square.json (+ state .pt
for script 2)."""
import json, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'

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
TRAIN = FW[0:256, :128].to(DEV)          # discovery slice -- ONLY for the Mr SVD basis
S_, T_ = HELD.shape
LI = 2                                    # the layer whose MLP we decompose
KMAX = 64                                 # SVD dirs computed; primary causal split at K=32
print(f"bilin18 NL={NL} D={D} NH={NH} held {S_}x{T_}  LI={LI} KMAX={KMAX}", flush=True)

# ---- the five coarse groups and the 15 group-pair terms (VERBATIM census) ----
GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']
NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
NT = len(PAIRS)
MRMR = PNAMES.index('MrxMr'); ARMR = PNAMES.index('ArxMr')

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
W = mlp_wts(LI)

def pair_terms(groups, xpre, Lw, Rw, Dw):
    """VERBATIM construction from qk_allterm_census.pair_terms."""
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    PL = [g @ Lw.T for g in groups]; PR = [g @ Rw.T for g in groups]
    terms = []
    for (i, j) in PAIRS:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ Dw.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms, rho2

# =====================================================================================
# PASS 0: top-KMAX SVD directions of layer-1's MLP output from the TRAIN gram
# (construction VERBATIM qk_mlp1_tail.py, LI_SRC=1). Directions of Mr = directions of
# mo1 (Mr = lambda * mo1, scaling preserves directions).
# =====================================================================================
LSRC = 1
gram = torch.zeros(D, D, device=DEV)
@torch.no_grad()
def fwd_gram(idx):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(LSRC + 1):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh4 = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh4.reshape(B, T, -1)); mo = blk.mlp(F.rms_norm(x, (D,)))
        if li == LSRC: gram.add_(torch.einsum('btd,bte->de', mo, mo))
        x = x + mo
print("PASS 0: TRAIN gram of layer-1 MLP output -> SVD directions ...", flush=True)
for i in range(0, TRAIN.shape[0], B0): fwd_gram(TRAIN[i:i+B0])
_evals, _evecs = torch.linalg.eigh(gram)
U = _evecs[:, -KMAX:].T.flip(0).contiguous()          # (KMAX, D) descending
evals_desc = _evals.flip(0).cpu().numpy()
spec_capture = {K: float(evals_desc[:K].sum()/_evals.sum().item()) for K in (8, 16, 32, 64)}
print(f"gram spectrum capture (TRAIN): {spec_capture}", flush=True)
del gram, _evecs

# ---- cache the fixed bilinear pair vectors B_ij = T2(u_i,u_j) core (no gauge) ----
PLu = U @ W[0].T; PRu = U @ W[1].T                    # (K, hidden)
S_hid = 0.5*(PLu[:, None, :]*PRu[None, :, :] + PLu[None, :, :]*PRu[:, None, :])   # (K,K,hidden) sym
Bp = torch.einsum('ijh,dh->ijd', S_hid, W[2])         # (K,K,D) symmetric
del S_hid, PLu, PRu
Bnorm2 = Bp.pow(2).sum(-1)                            # (K,K)
print(f"cached B_ij pair vectors: {tuple(Bp.shape)}", flush=True)

# =====================================================================================
# forward VERBATIM from qk_allterm_census.fwd (coarse-group accumulators), extended:
#   'collect' -> census stats + P-mean accumulation + coefficient dump
#   'energy'  -> census term energies + diag/cross/remainder part energies + per-pair
#   'subset'  -> mo2 = MEANF + sum kept term devs + [pair-mask proj dev] + [remainder dev]
# =====================================================================================
@torch.no_grad()
def fwd(idx, mode=None, keep=None, M=None, rem=False, TMEAN=None, MEANF=None,
        PMEAN=None, stats=None, MMEANCACHE=None):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    track = mode is not None
    if track:
        cE = torch.ones((), device=DEV)
        SA = torch.zeros_like(x); SM = torch.zeros_like(x); MR = torch.zeros_like(x)
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0; a = blk.attn; hcur = F.rms_norm(x, (D,))
        if track and li <= LI:
            cE = blk.lambdas[0]*cE + blk.lambdas[1]
            SA = blk.lambdas[0]*SA; SM = blk.lambdas[0]*SM; MR = blk.lambdas[0]*MR
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
            groups = [cE*x0, SA, aout, SM, MR]         # E, Ae, Ar, Me, Mr; x is x_pre here
            terms, rho2 = pair_terms(groups, x, W[0], W[1], W[2])
            av = MR @ U.T                              # (B,T,K) coefficients of Mr on the SVD basis
            P = av[:, :, :, None] * av[:, :, None, :] / rho2[..., None]   # (B,T,K,K)
            projfull = torch.einsum('btij,ijd->btd', P, Bp)
            if mode == 'collect':
                stats['tsum_all'] += torch.stack(terms).sum(1)      # (NT,T,D)
                stats['mosum'] += mo.sum(0)
                recon = sum(terms) + W[3]
                num = (recon - mo).norm(dim=-1); den = mo.norm(dim=-1).clamp_min(1e-8)
                stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                stats['fro_num'] += float((recon - mo).pow(2).sum()); stats['fro_den'] += float(mo.pow(2).sum())
                stats['Psum'] += P.sum(0)                           # (T,K,K)
                stats['projfullsum'] += projfull.sum(0)             # (T,D)
                stats['cap_num'] += float((av.pow(2)).sum()); stats['cap_den'] += float(MR.pow(2).sum())
                # machinery gate: projected square via Bp/P must equal direct polarization of PMr
                Mrp = av @ U
                tt = ((Mrp @ W[0].T) * (Mrp @ W[1].T)) @ W[2].T / rho2
                stats['pgate'] = max(stats['pgate'],
                                     float(((projfull - tt).norm(dim=-1)/tt.norm(dim=-1).clamp_min(1e-8)).max()))
                stats['a_dump'].append(av.cpu()); stats['rho2_dump'].append(rho2[..., 0].cpu())
            elif mode == 'energy':
                for kk in range(NT): stats['te'][kk] += float((terms[kk] - TMEAN[kk]).pow(2).sum())
                stats['tot'] += float((mo - MEANF).pow(2).sum())
                Pc = P - PMEAN[None]
                stats['pairsq'] += Pc.pow(2).sum((0, 1))            # (K,K)  sum over data of (P-Pmean)^2
                for K in (8, 16, 32, 64):
                    ar = torch.arange(K, device=DEV)
                    dg = torch.einsum('bti,id->btd', Pc[:, :, ar, ar], Bp[ar, ar])
                    MK = torch.zeros(KMAX, KMAX, device=DEV); MK[:K, :K] = 1.0
                    pj = torch.einsum('btij,ijd->btd', Pc*MK, Bp)
                    cr = pj - dg
                    rm = (terms[MRMR] - TMEAN[MRMR]) - pj
                    stats[f'ediag{K}'] += float(dg.pow(2).sum())
                    stats[f'ecross{K}'] += float(cr.pow(2).sum())
                    stats[f'erem{K}'] += float(rm.pow(2).sum())
                    del dg, pj, cr, rm
            elif mode == 'subset':
                new = MEANF.unsqueeze(0).expand(B, -1, -1)
                for kk in keep: new = new + (terms[kk] - TMEAN[kk])
                if M is not None:
                    key = id(M)
                    if key not in MMEANCACHE:
                        MMEANCACHE[key] = torch.einsum('tij,ijd->td', PMEAN*M, Bp)
                    new = new + torch.einsum('btij,ijd->btd', P*M, Bp) - MMEANCACHE[key][None]
                if rem:
                    if 'full' not in MMEANCACHE:
                        MMEANCACHE['full'] = torch.einsum('tij,ijd->td', PMEAN, Bp)
                    new = new + (terms[MRMR] - projfull) - (TMEAN[MRMR] - MMEANCACHE['full'])[None]
                mo = new.to(x.dtype)
            del terms, groups, P, projfull, av
        x = x + mo
        if track and li < LI:
            SA = SA + aout; SM = SM + MR; MR = mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

# =====================================================================================
# PASS 1: per-position means + decomposition gates + coefficient dump
# =====================================================================================
print("PASS 1: term/part means + gates ...", flush=True)
st = {'tsum_all': torch.zeros(NT, T_, D, device=DEV), 'mosum': torch.zeros(T_, D, device=DEV),
      'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0, 'Psum': torch.zeros(T_, KMAX, KMAX, device=DEV),
      'projfullsum': torch.zeros(T_, D, device=DEV), 'cap_num': 0.0, 'cap_den': 0.0, 'pgate': 0.0,
      'a_dump': [], 'rho2_dump': []}
for i in range(0, S_, B0): fwd(HELD[i:i+B0], mode='collect', stats=st)
TMEAN = st['tsum_all'] / S_
MO_MEAN = st['mosum'] / S_
MEANF = TMEAN.sum(0) + W[3]
PMEAN = st['Psum'] / S_
gate_fro = (st['fro_num']/st['fro_den'])**0.5
mean_consist = float((MEANF - MO_MEAN).norm()/MO_MEAN.norm())
span_capture_held = st['cap_num']/st['cap_den']
A_HELD = torch.cat(st['a_dump'], 0)                    # (S,T,K)
RHO2_HELD = torch.cat(st['rho2_dump'], 0)              # (S,T)
# projection-machinery mean consistency: mean of projfull vs einsum(PMEAN,Bp)
projmean_consist = float((st['projfullsum']/S_ - torch.einsum('tij,ijd->td', PMEAN, Bp)).norm()
                         / (st['projfullsum']/S_).norm())
print(f"GATE recon: global {gate_fro:.2e} maxpos {st['maxrel']:.2e} meancons {mean_consist:.2e} | "
      f"proj machinery gate {st['pgate']:.2e} projmean consist {projmean_consist:.2e} | "
      f"Mr span capture (held, K=64) {span_capture_held:.4f}", flush=True)
assert gate_fro < 1e-4, "decomposition gate FAILED"
assert st['pgate'] < 1e-3, "projected-square machinery gate FAILED"
del st

# =====================================================================================
# PASS 2: base CE + centered energies (terms, parts at K in {8,16,32,64}, per-pair)
# =====================================================================================
print("PASS 2: base + energies ...", flush=True)
st2 = {'te': [0.0]*NT, 'tot': 0.0, 'pairsq': torch.zeros(KMAX, KMAX, device=DEV)}
for K in (8, 16, 32, 64):
    st2[f'ediag{K}'] = 0.0; st2[f'ecross{K}'] = 0.0; st2[f'erem{K}'] = 0.0
ces = []
for i in range(0, S_, B0):
    ces.append(fwd(HELD[i:i+B0], mode='energy', TMEAN=TMEAN, MEANF=MEANF, PMEAN=PMEAN, stats=st2).cpu())
base = torch.cat(ces, 0)
print(f"base CE {float(base.mean()):.4f}", flush=True)
tot = st2['tot']
term_shares = {PNAMES[k]: round(st2['te'][k]/tot, 4) for k in range(NT)}
part_shares = {}
for K in (8, 16, 32, 64):
    part_shares[K] = {'diag': round(st2[f'ediag{K}']/tot, 4), 'cross': round(st2[f'ecross{K}']/tot, 4),
                      'remainder': round(st2[f'erem{K}']/tot, 4),
                      'diag_frac_of_diag_plus_cross': round(st2[f'ediag{K}'] /
                                                            max(st2[f'ediag{K}']+st2[f'ecross{K}'], 1e-12), 4)}
    print(f"K={K:2d} energy shares of mo2 variance: diag {part_shares[K]['diag']} "
          f"cross {part_shares[K]['cross']} remainder {part_shares[K]['remainder']}", flush=True)
print(f"MrxMr total share (census cross-check): {term_shares['MrxMr']}", flush=True)

# per-pair centered energies: contribution of unordered pair (i<j) = 2*P_ij*B_ij (rank-1)
pair_energy = (st2['pairsq'] * Bnorm2).cpu().numpy()   # (K,K); off-diag pair energy = 4*.. handled below
PE = []
for i in range(KMAX):
    for j in range(i+1, KMAX):
        PE.append((i, j, 4.0*float(pair_energy[i, j])/tot))   # (2 P_ij B_ij) -> factor 4 on square
PE.sort(key=lambda r: -r[2])
DE = [(i, float(pair_energy[i, i])/tot) for i in range(KMAX)]
DE.sort(key=lambda r: -r[1])
print("top-10 cross pairs by centered energy share:", [(f"({i},{j})", round(e, 4)) for i, j, e in PE[:10]], flush=True)
print("top-10 diagonal self-products by energy share:", [(i, round(e, 4)) for i, e in DE[:10]], flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

MMEANCACHE = {}
def run(keep, M=None, rem=False):
    out = []
    for i in range(0, S_, B0):
        out.append(fwd(HELD[i:i+B0], mode='subset', keep=keep, M=M, rem=rem,
                       TMEAN=TMEAN, MEANF=MEANF, PMEAN=PMEAN, MMEANCACHE=MMEANCACHE).cpu())
    return torch.cat(out, 0)

def mask_of(pairs_diag=(), pairs_cross=(), full_upto=None, minus=None):
    Mk = torch.zeros(KMAX, KMAX, device=DEV)
    if full_upto is not None: Mk[:full_upto, :full_upto] = 1.0
    for i in pairs_diag: Mk[i, i] = 1.0
    for (i, j) in pairs_cross: Mk[i, j] = 1.0; Mk[j, i] = 1.0
    if minus is not None: Mk = Mk - minus; Mk = Mk.clamp(0, 1)
    return Mk

OTHERS = [k for k in range(NT) if k != MRMR]
diag_only = {K: mask_of(pairs_diag=range(K)) for K in (16, 32)}
full_K = {K: mask_of(full_upto=K) for K in (16, 32, 64)}
cross_only = {K: (full_K[K] - diag_only.get(K, mask_of(pairs_diag=range(K)))).clamp(0, 1) for K in (16, 32)}

configs = [
    # gates (replicate census numbers)
    ('all15_via_parts', OTHERS, full_K[64], True),
    ('drop_MrxMr', OTHERS, None, False),
    ('only_MrxMr_full', [], full_K[64], True),
    # main split, full context (other 14 terms kept), K=32
    ('keep_diag32_dropcross_droprem', OTHERS, diag_only[32], False),
    ('keep_cross32_dropdiag_droprem', OTHERS, cross_only[32], False),
    ('keep_proj32_droprem', OTHERS, full_K[32], False),
    ('keep_rem32_only', OTHERS, (full_K[64]-full_K[32]).clamp(0, 1), True),
    ('drop_diag32_keep_rest', OTHERS, (full_K[64]-diag_only[32]).clamp(0, 1), True),
    ('drop_cross32_keep_rest', OTHERS, (full_K[64]-cross_only[32]).clamp(0, 1), True),
    # K=16 variants
    ('keep_diag16_dropcross_droprem', OTHERS, diag_only[16], False),
    ('keep_cross16_dropdiag_droprem', OTHERS, cross_only[16], False),
    # minimal context (mean + MrxMr parts only; mirrors census only_MrxMr = 0.0925)
    ('minimal_diag32', [], diag_only[32], False),
    ('minimal_cross32', [], cross_only[32], False),
    ('minimal_proj32', [], full_K[32], False),
]
# pair sufficiency curve: diag32 + top-p cross pairs (by energy), full context
for p in (1, 2, 4, 8, 16, 32, 64, 128):
    Mk = mask_of(pairs_diag=range(32), pairs_cross=[(i, j) for i, j, _ in PE[:p]])
    configs.append((f'keep_diag32_top{p}pairs', OTHERS, Mk, False))
# pair necessity: full minus one top pair
for (i, j, e) in PE[:5]:
    configs.append((f'drop_pair_{i}_{j}', OTHERS,
                    mask_of(full_upto=64, minus=mask_of(pairs_cross=[(i, j)])), True))

res = {'meta': {
    'model': 'bilin18', 'layer': LI, 'held': 'FW[448:600,:128]', 'batch': B0, 'KMAX': KMAX,
    'question': 'what algorithm does layer 2\'s MrxMr (square of the lambda-decayed layer-1 MLP '
                'output) implement: H1 cross-product AND-gates vs H2 diagonal magnitude '
                'nonlinearity vs H3 quadratic basis for downstream (script 2)',
    'basis': 'top-64 SVD directions of layer-1 MLP output, TRAIN gram FW[0:256,:128] '
             '(construction verbatim qk_mlp1_tail.py); Mr = lambda-decayed mo1, same directions',
    'machinery': 'VERBATIM qk_allterm_census.py (groups, pair_terms, gate, keep-subset harness); '
                 'projected square via cached fixed pair vectors B_ij = T2-core(u_i,u_j), '
                 'per-position coefficient products P_ij = a_i a_j / rho2',
    'currency': 'delta cross-entropy per held position (nats), paired standard error',
    'base_ce': round(float(base.mean()), 4),
    'gates': {'recon_rel_err_global': gate_fro, 'mean_consistency': mean_consist,
              'proj_machinery_max_rel_err': None, 'projmean_consistency': projmean_consist,
              'Mr_span_capture_held_K64': round(span_capture_held, 4),
              'gram_spectrum_capture_train': spec_capture}},
    'energy': {'term_shares': term_shares, 'part_shares_by_K': part_shares,
               'top_cross_pairs': [(i, j, round(e, 5)) for i, j, e in PE[:20]],
               'top_diag': [(i, round(e, 5)) for i, e in DE[:15]]},
    'configs': {}}

print(f"PASS 3: {len(configs)} subset configurations ...", flush=True)
t0 = time.time()
for name, keep, Mk, rem in configs:
    mn, se = dstat(run(keep, Mk, rem))
    res['configs'][name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:32s} dCE {mn:+.4f} +- {se:.5f}   ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(f'{QK}/qk_arc_square.json', 'w'), indent=1)

# reference numbers from the census for the verdict block
cen = json.load(open(f'{QK}/qk_allterm_census.json'))['layers']['2']
res['census_ref'] = {'floor': cen['floor_dCE'], 'allbut_MrxMr': cen['configs']['allbut_MrxMr']['dCE'],
                     'only_MrxMr': cen['configs']['only_MrxMr']['dCE'],
                     'top2': cen['configs']['top2_energy']['dCE']}
json.dump(res, open(f'{QK}/qk_arc_square.json', 'w'), indent=1)

torch.save({'U': U.cpu(), 'PMEAN': PMEAN.cpu(), 'TMEAN': TMEAN.cpu(), 'MEANF': MEANF.cpu(),
            'base_ce': base, 'A_HELD': A_HELD, 'RHO2_HELD': RHO2_HELD,
            'pair_energy_share_topcross': PE[:200], 'diag_energy_share': DE,
            'evals_desc': evals_desc}, f'{QK}/qk_arc_square_state.pt')
print("QK ARC SQUARE (script 1) DONE", flush=True)
