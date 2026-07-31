"""WHY IS LAYER 17 A CANCELLING MIXER? Part 1 of 2: structural tests (H2, H3).
Section 89 + red-team attack 4 established that bilin18's final feed-forward computes through
CANCELLATION: its two dominant group-pair terms (AexMr share 0.349, MrxMr share 0.224) are
anti-aligned at cosine -0.842 (other heavy pairs -0.934, -0.579), cancellation index 1.54 vs
0.65 at healthy layer 1, and the greedy keep-curve is non-monotone. Three hypotheses:
  H1 push-pull sharpening (conditional difference; tested in part 2 via class signatures),
  H2 null-space waste (the cancelling mass lives in directions the readout barely sees:
     project each term's centered output onto the unembedding row-space (top-K right singular
     vectors of the vocab-centered W_U, K = 144/288/576 of D=1152) vs its complement; H2
     predicts anti-alignment concentrated in the LOW-logit-relevance complement),
  H3 gain control (the opposing components regulate the layer's write NORM, not direction;
     note the readout rms_norm is scale-invariant so the OVERALL residual norm is exactly
     null -- the meaningful H3 reading is that cancellation keeps ||mo17|| small so the write
     does not overweight the layer's direction against the accumulated residual. Test: for
     keep-subset interventions, build hybrids (direction from subset, per-position norm from
     full; and vice versa) and see which factor carries the causal damage).
Machinery VERBATIM from qk_allterm_census.py (5-group pair_terms polarization, forward with
coarse-group stream accumulators, per-position term means, keep-subset harness; terms verified
to reconstruct mo17 at ~7e-7). Extra logit-metric Gram uses M = Wc^T Wc (Wc = vocab-centered
unembedding) -- the first-order metric the softmax actually sees.
GATES: recon < 1e-4; mean-only floor reproduces 0.4206; full-space cos(AexMr,MrxMr) reproduces
-0.842; keep-top2-energy reproduces census 0.2023. Held FW[448:600,:128], paired SE, batch 6.
Output: qk_L17_mixer.json (partial; part 2 adds class signatures + verdict) and
qk_L17_mixer_means.pt (TMEAN/MEANF/DEVN for part 2)."""
import json, os, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_L17_mixer.json'

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
print(f"bilin18 NL={NL} D={D} NH={NH} held {S_}x{T_}", flush=True)

# ---- the five coarse groups and the 15 group-pair terms (VERBATIM qk_allterm_census.py) ----
GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']
NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
NT = len(PAIRS)   # 15
LI = 17
IA = PNAMES.index('AexMr')   # dominant term 1, share 0.349
IB = PNAMES.index('MrxMr')   # dominant term 2, share 0.224 (the -0.842 partner)
IC = PNAMES.index('AexMe')   # 2nd by energy (for the census top-2 config)
SUBSETS = {'pair_AexMr_MrxMr': [IA, IB], 'top2_energy_AexMr_AexMe': [IA, IC]}

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
W = mlp_wts(LI)

def pair_terms(groups, xpre, Lw, Rw, Dw):
    """15 interaction terms (list of (B,T,D)), sharing the common 1/rho^2 gauge; sum+bias == mo_L.
    VERBATIM construction from qk_hub_streampairs.pair_terms, generalized to 5 groups."""
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    PL = [g @ Lw.T for g in groups]; PR = [g @ Rw.T for g in groups]
    terms = []
    for (i, j) in PAIRS:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ Dw.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms

@torch.no_grad()
def fwd(idx, mode=None, subset=None, hyb=None, TMEAN=None, MEANF=None, stats=None):
    """Forward verbatim from qk_allterm_census.py (coarse-group stream accumulators to LI=17).
    mode: None (full) | 'collect' (term means + recon gate) | 'cov2' (Grams in full space,
    unembed row-spaces, logit metric; + deviation norms; + norm/direction stats for SUBSETS)
    | 'subset' (mo -> MEANF + sum_{k in subset}(term_k - TMEAN[k]); hyb None|'normfull'|'dirfull')."""
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
            groups = [cE*x0, SA, aout, SM, MR]     # E, Ae, Ar, Me, Mr; x is x_pre here
            terms = pair_terms(groups, x, W[0], W[1], W[2])
            if mode == 'collect':
                for kk in range(NT): stats['tsum'][kk] += terms[kk].sum(0)
                stats['mosum'] += mo.sum(0)
                recon = sum(terms) + W[3]
                num = (recon - mo).norm(dim=-1); den = mo.norm(dim=-1).clamp_min(1e-8)
                stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                stats['fro_num'] += float((recon - mo).pow(2).sum()); stats['fro_den'] += float(mo.pow(2).sum())
            elif mode == 'cov2':
                devs = torch.stack([(terms[kk] - TMEAN[kk]) for kk in range(NT)])   # (15,B,T,D)
                stats['devn'].append(devs.norm(dim=-1).cpu())                        # (15,B,T)
                dflat = devs.reshape(NT, -1, D)
                Ff = dflat.reshape(NT, -1)
                stats['Cfull'] += (Ff @ Ff.T).double()
                A = torch.einsum('kpd,de->kpe', dflat, stats['U576'])                # (15,P,576)
                for Kc in (144, 288, 576):
                    Fa = A[:, :, :Kc].reshape(NT, -1)
                    stats[f'G{Kc}'] += (Fa @ Fa.T).double()
                Fm = (dflat @ stats['Mlog']).reshape(NT, -1)
                stats['GM'] += (Fm @ Ff.T).double()
                stats['tot'] += float((mo - MEANF).pow(2).sum())
                # norm/direction stats for the two keep-subsets (H3 groundwork)
                for sname, sub in SUBSETS.items():
                    mo_s = MEANF.unsqueeze(0).expand(B, -1, -1)
                    for kk in sub: mo_s = mo_s + (terms[kk] - TMEAN[kk])
                    nf = mo.norm(dim=-1); ns = mo_s.norm(dim=-1)
                    cm = (mo*mo_s).sum(-1)/(nf*ns).clamp_min(1e-8)
                    xf = x + mo; xfs = x + mo_s
                    cx = (xf*xfs).sum(-1)/(xf.norm(dim=-1)*xfs.norm(dim=-1)).clamp_min(1e-8)
                    st = stats['sub'][sname]
                    st['nfull'].append(nf.cpu()); st['nsub'].append(ns.cpu())
                    st['cos_mo'].append(cm.cpu()); st['cos_xf'].append(cx.cpu())
                    st['nxf_ratio'].append((xfs.norm(dim=-1)/xf.norm(dim=-1).clamp_min(1e-8)).cpu())
                # x_pre norm for reference (write strength relative to carrier)
                stats['xpre_n'].append(x.norm(dim=-1).cpu())
                del devs, dflat, Ff, A, Fm
            elif mode == 'subset':
                new = MEANF.unsqueeze(0).expand(B, -1, -1)
                for kk in subset: new = new + (terms[kk] - TMEAN[kk])
                if hyb == 'normfull':      # direction from subset, per-position norm from full
                    new = new * (mo.norm(dim=-1, keepdim=True)
                                 / new.norm(dim=-1, keepdim=True).clamp_min(1e-8))
                elif hyb == 'dirfull':     # direction from full, per-position norm from subset
                    new = mo * (new.norm(dim=-1, keepdim=True)
                                / mo.norm(dim=-1, keepdim=True).clamp_min(1e-8))
                mo = new.to(x.dtype)
            del terms, groups
        x = x + mo
        if track and li < LI:
            SA = SA + aout; SM = SM + MR; MR = mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

# ---------------- base CE (full model), once ----------------
print("BASE: full-model cross-entropy on held slice ...", flush=True)
base = torch.cat([fwd(HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
print(f"base CE mean {float(base.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

# ---------------- PASS 1: term means + decomposition gate ----------------
st = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
      'mosum': torch.zeros(T_, D, device=DEV), 'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0}
for i in range(0, S_, B0): fwd(HELD[i:i+B0], mode='collect', stats=st)
TMEAN = torch.stack([t/S_ for t in st['tsum']])
MEANF = TMEAN.sum(0) + W[3]
gate_fro = (st['fro_num']/st['fro_den'])**0.5
print(f"L17 GATE recon global {gate_fro:.2e} maxpos {st['maxrel']:.2e}", flush=True)
assert gate_fro < 1e-4, "decomposition gate FAILED"

# ---------------- unembedding row-space basis (H2) ----------------
WU = m.lm_head.weight.detach().float()                     # (V, D)
Wc = WU - WU.mean(0, keepdim=True)                         # center over vocab: softmax-invariant shift removed
Mlog64 = (Wc.T @ Wc).double().cpu()                        # (D, D) logit metric
ev, evec = torch.linalg.eigh(Mlog64)                       # ascending
ev = ev.flip(0); evec = evec.flip(-1)                      # descending
tr = float(ev.sum())
spec = {f'top{K}_trace_share': round(float(ev[:K].sum())/tr, 4) for K in (144, 288, 576)}
print(f"unembed spectrum trace shares: {spec}", flush=True)
U576 = evec[:, :576].float().to(DEV)
Mlog = Mlog64.float().to(DEV)

# ---------------- PASS 2: Grams (full / row-space / complement / logit metric) + H3 stats ----------------
st2 = {'Cfull': torch.zeros(NT, NT, dtype=torch.float64, device=DEV),
       'G144': torch.zeros(NT, NT, dtype=torch.float64, device=DEV),
       'G288': torch.zeros(NT, NT, dtype=torch.float64, device=DEV),
       'G576': torch.zeros(NT, NT, dtype=torch.float64, device=DEV),
       'GM': torch.zeros(NT, NT, dtype=torch.float64, device=DEV),
       'tot': 0.0, 'devn': [], 'xpre_n': [], 'U576': U576, 'Mlog': Mlog,
       'sub': {s: {'nfull': [], 'nsub': [], 'cos_mo': [], 'cos_xf': [], 'nxf_ratio': []}
               for s in SUBSETS}}
for i in range(0, S_, B0):
    fwd(HELD[i:i+B0], mode='cov2', TMEAN=TMEAN, MEANF=MEANF, stats=st2)
DEVN = torch.cat(st2['devn'], dim=1)     # (15, S, T)
torch.save({'TMEAN': TMEAN.cpu(), 'MEANF': MEANF.cpu(), 'PNAMES': PNAMES, 'PAIRS': PAIRS,
            'DEVN': DEVN, 'IA': IA, 'IB': IB, 'IC': IC},
           f'{QK}/qk_L17_mixer_means.pt')

Cfull = st2['Cfull'].cpu().numpy(); tot = st2['tot']
Grow = {K: st2[f'G{K}'].cpu().numpy() for K in (144, 288, 576)}
GM = st2['GM'].cpu().numpy()

def cosof(G, a, b):
    d = (G[a, a]*G[b, b])**0.5
    return float(G[a, b]/max(d, 1e-30))

def cancel_idx(G, idxs=None):
    idxs = list(range(NT)) if idxs is None else idxs
    e = sum(G[k, k] for k in idxs)
    s = sum(G[a, b] for a in idxs for b in idxs)
    return float(e/max(s, 1e-30))

HEAVY_PAIRS = [('AexMr', 'MrxMr'), ('AexAr', 'ArxMr'), ('AexAr', 'MrxMr'), ('AexMe', 'MexMr')]
cos_full_main = cosof(Cfull, IA, IB)
print(f"GATE full-space cos(AexMr,MrxMr) = {cos_full_main:.3f} (expect -0.842)", flush=True)
assert abs(cos_full_main - (-0.842)) < 0.02, "anti-alignment reproduction gate FAILED"

test2 = {'unembed_spectrum': spec,
         'centering': 'W_U centered over vocab (uniform-logit-shift direction removed); '
                      'row-space basis = top-K eigenvectors of Wc^T Wc; the final rms_norm has no '
                      'learned gain, so no further centering applies',
         'energy_share_full': {PNAMES[k]: round(Cfull[k, k]/tot, 4) for k in range(NT)},
         'cancellation_index': {'full': round(cancel_idx(Cfull), 4)},
         'pairs': {}}
for K in (144, 288, 576):
    GK = Grow[K]; GC = Cfull - GK
    test2['cancellation_index'][f'rowspace_K{K}'] = round(cancel_idx(GK), 4)
    test2['cancellation_index'][f'complement_K{K}'] = round(cancel_idx(GC), 4)
for (na, nb) in HEAVY_PAIRS:
    a, b = PNAMES.index(na), PNAMES.index(nb)
    rec = {'cos_full': round(cosof(Cfull, a, b), 4),
           'cos_logit_metric': round(cosof(GM, a, b), 4)}
    for K in (144, 288, 576):
        GK = Grow[K]; GC = Cfull - GK
        rec[f'K{K}'] = {
            'cos_rowspace': round(cosof(GK, a, b), 4),
            'cos_complement': round(cosof(GC, a, b), 4),
            'energy_rowfrac_A': round(float(GK[a, a]/Cfull[a, a]), 4),
            'energy_rowfrac_B': round(float(GK[b, b]/Cfull[b, b]), 4),
            'cross_overlap_rowspace_share': round(float(GK[a, b]/Cfull[a, b]), 4),
            'pair_sum_survival_rowspace': round(float((GK[a, a]+GK[b, b]+2*GK[a, b])
                                                      / (GK[a, a]+GK[b, b])), 4),
            'pair_sum_survival_complement': round(float((GC[a, a]+GC[b, b]+2*GC[a, b])
                                                        / (GC[a, a]+GC[b, b])), 4)}
    test2['pairs'][f'{na}__{nb}'] = rec
    print(f"T2 {na}x{nb}: full {rec['cos_full']:+.3f} logit-metric {rec['cos_logit_metric']:+.3f} | "
          + ' '.join(f"K{K} row {rec[f'K{K}']['cos_rowspace']:+.3f} comp {rec[f'K{K}']['cos_complement']:+.3f}"
                     f" (rowfrac {rec[f'K{K}']['energy_rowfrac_A']:.2f}/{rec[f'K{K}']['energy_rowfrac_B']:.2f})"
                     for K in (144, 288, 576)), flush=True)
print(f"T2 cancellation indices: {test2['cancellation_index']}", flush=True)

def qtile(lst):
    v = torch.cat([t.reshape(-1) for t in lst]).float()
    qs = torch.quantile(v, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {'mean': round(float(v.mean()), 4),
            'q05': round(float(qs[0]), 4), 'q25': round(float(qs[1]), 4),
            'q50': round(float(qs[2]), 4), 'q75': round(float(qs[3]), 4),
            'q95': round(float(qs[4]), 4)}

test3_stats = {}
xpre_med = qtile(st2['xpre_n'])['q50']
for sname in SUBSETS:
    s = st2['sub'][sname]
    nr = [a/b.clamp_min(1e-8) for a, b in zip(s['nsub'], s['nfull'])]
    test3_stats[sname] = {'norm_mo_full': qtile(s['nfull']), 'norm_mo_subset': qtile(s['nsub']),
                          'norm_ratio_subset_over_full': qtile(nr),
                          'cos_mo_subset_vs_full': qtile(s['cos_mo']),
                          'cos_xfinal_subset_vs_full': qtile(s['cos_xf']),
                          'xfinal_norm_ratio': qtile(s['nxf_ratio'])}
    print(f"T3 stats {sname}: ||mo|| full med {test3_stats[sname]['norm_mo_full']['q50']:.2f} "
          f"subset med {test3_stats[sname]['norm_mo_subset']['q50']:.2f} "
          f"(x_pre med {xpre_med:.2f}); cos_mo med {test3_stats[sname]['cos_mo_subset_vs_full']['q50']:.3f}; "
          f"cos_xfinal med {test3_stats[sname]['cos_xfinal_subset_vs_full']['q50']:.4f}", flush=True)
del st2
torch.cuda.empty_cache()

# ---------------- PASS 3: H3 hybrid interventions ----------------
def run(subset, hyb=None):
    out = []
    for i in range(0, S_, B0):
        out.append(fwd(HELD[i:i+B0], mode='subset', subset=subset, hyb=hyb,
                       TMEAN=TMEAN, MEANF=MEANF).cpu())
    return torch.cat(out, 0)

test3 = {'configs': {}, 'stats': test3_stats, 'xpre_norm_median': xpre_med,
         'note': 'the readout rms_norm is scale-invariant, so overall x_final scaling is exactly '
                 'null; norm here = per-position ||mo17|| (the write strength against the carrier '
                 'x_pre). normfull = direction from kept subset, per-position norm copied from the '
                 'full mo17; dirfull = direction from full mo17, norm from the kept subset.'}
cfgs = [('mean_only', [], None), ('all_terms', list(range(NT)), None)]
for sname, sub in SUBSETS.items():
    cfgs += [(f'{sname}', sub, None), (f'{sname}_normfull', sub, 'normfull'),
             (f'{sname}_dirfull', sub, 'dirfull')]
for name, sub, hyb in cfgs:
    mn, se = dstat(run(sub, hyb))
    test3['configs'][name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"T3 {name:34s} dCE {mn:+.4f} +- {se:.5f}", flush=True)
floor = test3['configs']['mean_only']['dCE']
assert abs(floor - 0.4206) < 0.02, "floor reproduction gate FAILED"
assert abs(test3['configs']['top2_energy_AexMr_AexMe']['dCE'] - 0.2023) < 0.02, \
    "census top-2 reproduction gate FAILED"
for sname in SUBSETS:
    d0 = test3['configs'][sname]['dCE']
    dn = test3['configs'][f'{sname}_normfull']['dCE']
    dd = test3['configs'][f'{sname}_dirfull']['dCE']
    test3[f'{sname}_rescue'] = {
        'damage_plain': d0,
        'rescue_frac_by_norm_matching': round((d0-dn)/d0, 4) if d0 else None,
        'damage_frac_carried_by_norm_alone': round(dd/d0, 4) if d0 else None}
    print(f"T3 {sname}: plain {d0:+.4f}; norm-matched (dir from subset) {dn:+.4f} -> "
          f"norm-matching rescues {100*(d0-dn)/d0:.1f}%; full-dir at subset-norm {dd:+.4f} -> "
          f"norm alone carries {100*dd/d0:.1f}% of damage", flush=True)

res = {'meta': {'model': 'bilin18', 'layer': 17, 'held': 'FW[448:600,:128]', 'batch': B0,
                'base_ce': round(float(base.mean()), 4),
                'machinery': 'VERBATIM qk_allterm_census.py (pair_terms, forward, keep-subset '
                             'harness); Grams as qk_redteam_fold_2.py attack 4',
                'gates': {'recon_rel_err_global': gate_fro, 'recon_max_pos': st['maxrel'],
                          'cos_full_AexMr_MrxMr': round(cos_full_main, 4),
                          'floor_dCE': floor,
                          'top2_energy_dCE': test3['configs']['top2_energy_AexMr_AexMe']['dCE']},
                'currency': 'delta cross-entropy per held position (nats), paired standard error'},
       'test2_subspace': test2, 'test3_norm_vs_direction': test3}
json.dump(res, open(OUT, 'w'), indent=1)
print("QK L17 MIXER PART 1 DONE", flush=True)
