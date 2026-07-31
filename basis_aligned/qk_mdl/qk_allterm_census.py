"""MODEL-WIDE STREAM-PAIR PROVENANCE CENSUS: extend the section-86 MLP1 anatomy to EVERY feed-forward
block of bilin18. Each MLP is exactly bilinear (mo_L = T_L(x,x)/rho^2 + bias, rms-as-gauge), and the
pre-norm input at layer L is an EXACT lambda-weighted sum of the initial embedding stream and every
prior component output. STREAM-SET BLOWUP HANDLED BY COARSE GROUPS (Logan's both-sides reduction):
at each layer the up-to-(2L+2) streams are grouped into at most FIVE groups
  E  = embedding (coefficient-tracked x0),
  Ae = attention-EARLIER (all prior attention outputs, lambda-decayed sum),
  Ar = attention-RECENT (the current layer's own attention output, coefficient 1),
  Me = MLP-EARLIER (all mlp outputs before the immediately preceding one, lambda-decayed sum),
  Mr = MLP-RECENT (the immediately preceding mlp output, lambda-decayed),
so at most 15 group-pair terms per layer, uniform across depth. The coarsening is PRINCIPLED and the
decomposition stays EXACT: groups are sums of streams and T is bilinear over the group sum -- the
15 terms + bias reconstruct mo_L to machine precision (GATED <1e-4 per layer before any causal claim).
Layer 1 reproduces section-86 exactly under the renaming A0->Ae(-ish: here Ae=lam*attn0), M0->Mr, A1->Ar.

Machinery VERBATIM from qk_hub_streampairs.py: pair_terms polarization through the actual mlp weights
(0.5*(PL_i*PR_j + PL_j*PR_i) @ Down.T, x2 off-diagonal, shared 1/rho^2 gauge), the fold gate, the
per-position term means, the keep-subset intervention harness (mo -> per-position decomposition-mean
+ sum of kept terms' deviations). Forward skeleton lineage qk_hub_maprestrict.py / qk_hub_threshold.py.

PER LAYER (economical, 18 layers): (a) mean-only floor (= full mean-ablation; auto-verified against
qk_mlp_superposition.json per_layer_tail full_mlp_dCE); (b) each of the TOP-6 terms by centered energy
kept alone; (c) greedy sufficiency curve top-1..top-6 (extended 8/10/12/15 only if 95% not reached);
(d) deletion (all-but-term) for the top-3 terms; diagonal-only vs cross-only; dead-group causal check
(drop all terms involving a candidate-dead group). Held FW[448:600,:128], paired SE, batch 6, <4GB.
Output: qk_allterm_census.json (incremental per-layer dump; resumable via argv[1]=start_layer)."""
import json, os, sys, time, subprocess
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_allterm_census.json'

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

# ---- the five coarse groups and the 15 group-pair terms (uniform across layers) ----
GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']
NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
NT = len(PAIRS)   # 15
DIAG = [k for k, (i, j) in enumerate(PAIRS) if i == j]
CROSS = [k for k, (i, j) in enumerate(PAIRS) if i != j]

# prior per-layer full-MLP mean-ablation floors for verification (section-73 machinery)
PRIOR = {r['layer']: r['full_mlp_dCE'] for r in
         json.load(open(f'{QK}/qk_mlp_superposition.json'))['per_layer_tail']}

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())

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
def fwd(idx, LI=None, mode=None, subset=None, TMEAN=None, MEANF=None, stats=None, W=None):
    """Forward verbatim from qk_hub_streampairs.py / qk_hub_threshold.py, with the coarse-group
    stream accumulators (E coefficient, Ae, Me, Mr running lambda-decayed sums) up to layer LI.
    mode: None (full model) | 'collect' (term sums + recon gate + group norms)
          | 'energy' (centered energies) | 'subset' (mo -> MEANF + sum_{k in subset} (term_k - TMEAN[k]))."""
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
                gs = sum(groups)
                stats['grp_err'] = max(stats['grp_err'],
                                       float(((gs - x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).max()))
                for g in range(NG): stats['gsq'][g] += float(groups[g].pow(2).sum())
                stats['xsq'] += float(x.pow(2).sum())
                for kk in range(NT): stats['tsum'][kk] += terms[kk].sum(0)
                stats['mosum'] += mo.sum(0)
                recon = sum(terms) + W[3]
                num = (recon - mo).norm(dim=-1); den = mo.norm(dim=-1).clamp_min(1e-8)
                stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                stats['fro_num'] += float((recon - mo).pow(2).sum()); stats['fro_den'] += float(mo.pow(2).sum())
            elif mode == 'energy':
                for kk in range(NT): stats['te'][kk] += float((terms[kk] - TMEAN[kk]).pow(2).sum())
                stats['tot'] += float((mo - MEANF).pow(2).sum())
            elif mode == 'subset':
                new = MEANF.unsqueeze(0).expand(B, -1, -1)
                for kk in subset: new = new + (terms[kk] - TMEAN[kk])
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

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if start > 0 and os.path.exists(OUT):
    res = json.load(open(OUT))
else:
    res = {'meta': {
        'model': 'bilin18', 'held': 'FW[448:600,:128]', 'batch': B0,
        'groups': {'E': 'embedding stream (coefficient-tracked x0)',
                   'Ae': 'attention-EARLIER (lambda-decayed sum of all prior attention outputs)',
                   'Ar': 'attention-RECENT (current layer attention output)',
                   'Me': 'MLP-EARLIER (lambda-decayed sum of mlp outputs before the previous one)',
                   'Mr': 'MLP-RECENT (lambda-decayed immediately preceding mlp output)'},
        'coarsening_note': 'streams grouped into <=5 groups per layer (bounded, principled); decomposition '
                           'remains EXACT because groups are sums of streams and T is bilinear; '
                           'gated per layer at recon rel err < 1e-4 before any causal number',
        'machinery': 'VERBATIM qk_hub_streampairs.py (pair_terms polarization, gate, keep-subset harness); '
                     'forward skeleton qk_hub_threshold.py/qk_hub_maprestrict.py lineage',
        'currency': 'delta cross-entropy per valid held position (nats), paired standard error',
        'floor_verification_source': 'qk_mlp_superposition.json per_layer_tail full_mlp_dCE',
        'base_ce': round(float(base.mean()), 4)}, 'layers': {}}

for LI in range(start, NL):
    tL = time.time()
    W = mlp_wts(LI)
    # ---- PASS 1: term means + decomposition gate ----
    st = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
          'mosum': torch.zeros(T_, D, device=DEV), 'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0,
          'grp_err': 0.0, 'gsq': [0.0]*NG, 'xsq': 0.0}
    for i in range(0, S_, B0): fwd(HELD[i:i+B0], LI=LI, mode='collect', stats=st, W=W)
    TMEAN = torch.stack([t/S_ for t in st['tsum']])
    MO_MEAN = st['mosum']/S_
    MEANF = TMEAN.sum(0) + W[3]
    gate_fro = (st['fro_num']/st['fro_den'])**0.5
    mean_consist = float((MEANF - MO_MEAN).norm()/MO_MEAN.norm())
    gshare = [round(st['gsq'][g]/st['xsq'], 4) for g in range(NG)]
    print(f"L{LI:2d} GATE recon global {gate_fro:.2e} maxpos {st['maxrel']:.2e} groupsum {st['grp_err']:.2e} "
          f"meancons {mean_consist:.2e} | group msq shares {dict(zip(GNAMES, gshare))}", flush=True)
    rec = {'gate': {'recon_rel_err_global': gate_fro, 'recon_rel_err_max_pos': st['maxrel'],
                    'group_sum_rel_err_max': st['grp_err'], 'mean_consistency': mean_consist,
                    'pass': bool(gate_fro < 1e-4)},
           'group_msq_share_of_xpre': dict(zip(GNAMES, gshare))}
    if gate_fro >= 1e-4:
        print(f"L{LI}: DECOMPOSITION GATE FAILED -- skipping causal measurements", flush=True)
        res['layers'][str(LI)] = rec; json.dump(res, open(OUT, 'w'), indent=1); continue

    # ---- PASS 2: centered term energies ----
    st2 = {'te': [0.0]*NT, 'tot': 0.0}
    for i in range(0, S_, B0):
        fwd(HELD[i:i+B0], LI=LI, mode='energy', TMEAN=TMEAN, MEANF=MEANF, stats=st2, W=W)
    shares = {PNAMES[k]: round(st2['te'][k]/st2['tot'], 4) for k in range(NT)}
    active = [k for k in range(NT) if st2['te'][k]/st2['tot'] > 1e-9]
    order = sorted(active, key=lambda k: -st2['te'][k])
    rec['energy_shares'] = shares
    rec['energy_rank'] = [PNAMES[k] for k in order]
    print(f"L{LI} energy rank: {[(PNAMES[k], shares[PNAMES[k]]) for k in order[:6]]}", flush=True)

    def run(subset):
        out = []
        for i in range(0, S_, B0):
            out.append(fwd(HELD[i:i+B0], LI=LI, mode='subset', subset=subset,
                           TMEAN=TMEAN, MEANF=MEANF, W=W).cpu())
        return torch.cat(out, 0)

    # ---- PASS 3: subset configurations (economical) ----
    cfgs = [('mean_only', []), ('all_terms', list(range(NT)))]
    cfgs += [(f'only_{PNAMES[k]}', [k]) for k in order[:6]]
    cfgs += [(f'top{kk}_energy', order[:kk]) for kk in range(2, min(6, len(order)) + 1)]
    cfgs += [(f'allbut_{PNAMES[k]}', [j for j in range(NT) if j != k]) for k in order[:3]]
    cfgs += [('diagonal', [k for k in DIAG if k in active]), ('cross', [k for k in CROSS if k in active])]
    rec['configs'] = {}
    for name, sub in cfgs:
        mn, se = dstat(run(sub))
        rec['configs'][name] = {'terms': [PNAMES[k] for k in sub], 'dCE': round(mn, 4), 'SE': round(se, 5)}
        print(f"  L{LI} {name:16s} [{len(sub):2d}] dCE {mn:+.4f} +- {se:.5f}", flush=True)
    floor = rec['configs']['mean_only']['dCE']; floor_se = rec['configs']['mean_only']['SE']
    rec['floor_dCE'] = floor; rec['floor_SE'] = floor_se
    rec['floor_prior_ref'] = PRIOR.get(LI)
    rec['floor_matches_prior'] = bool(abs(floor - PRIOR.get(LI, floor)) < max(0.02, 0.05*abs(PRIOR.get(LI, floor))))

    # terms-to-95%: smallest k with kept-top-k dCE < 5% of floor (extend if needed)
    thresh = 0.05 * floor
    n95 = None
    for kk in range(1, min(6, len(order)) + 1):
        nm = 'top1_energy' if kk == 1 else f'top{kk}_energy'
        if kk == 1: rec['configs']['top1_energy'] = rec['configs'][f'only_{PNAMES[order[0]]}']
        if rec['configs'][nm]['dCE'] < thresh: n95 = kk; break
    if n95 is None:
        for kk in [8, 10, 12, 15]:
            kk2 = min(kk, len(order))
            nm = f'top{kk2}_energy'
            if nm not in rec['configs']:
                mn, se = dstat(run(order[:kk2]))
                rec['configs'][nm] = {'terms': [PNAMES[k] for k in order[:kk2]], 'dCE': round(mn, 4), 'SE': round(se, 5)}
                print(f"  L{LI} {nm:16s} [{kk2:2d}] dCE {mn:+.4f} +- {se:.5f}", flush=True)
            if rec['configs'][nm]['dCE'] < thresh: n95 = kk2; break
            if kk2 == len(order): break
    rec['terms_to_95pct'] = n95
    rec['sufficient_terms'] = [PNAMES[k] for k in order[:n95]] if n95 else None

    # dead groups: absent (zero stream) vs causally dead (present but all involving terms inert)
    absent = [GNAMES[g] for g in range(NG) if gshare[g] < 1e-8]
    cand_dead = []
    for g in range(NG):
        if gshare[g] < 1e-8: continue
        inv = [k for k, (i, j) in enumerate(PAIRS) if (i == g or j == g)]
        if max(st2['te'][k]/st2['tot'] for k in inv) < 0.005: cand_dead.append(g)
    rec['groups_absent'] = absent
    rec['groups_candidate_dead_by_energy'] = [GNAMES[g] for g in cand_dead]
    for g in cand_dead:
        inv = set(k for k, (i, j) in enumerate(PAIRS) if (i == g or j == g))
        sub = [k for k in range(NT) if k not in inv]
        mn, se = dstat(run(sub))
        rec['configs'][f'drop_{GNAMES[g]}_involving'] = {'terms': [PNAMES[k] for k in sub],
                                                        'dCE': round(mn, 4), 'SE': round(se, 5)}
        print(f"  L{LI} drop_{GNAMES[g]}_involving dCE {mn:+.4f} +- {se:.5f}", flush=True)
    rec['groups_dead_causal'] = [GNAMES[g] for g in cand_dead
                                 if abs(rec['configs'][f'drop_{GNAMES[g]}_involving']['dCE'])
                                 < max(0.01, 0.01*abs(floor))]
    print(f"L{LI} DONE floor {floor:+.4f} (prior {PRIOR.get(LI)}) terms_to_95 {n95} "
          f"({time.time()-tL:.0f}s)", flush=True)
    res['layers'][str(LI)] = rec
    json.dump(res, open(OUT, 'w'), indent=1)
    del TMEAN, MEANF, MO_MEAN, st, st2
    torch.cuda.empty_cache()

print("QK ALLTERM CENSUS DONE", flush=True)
