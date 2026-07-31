"""LOGAN'S SUGGESTION: decompose the MLP1 hub by FOLDING ITS INPUT with what comes before -- the
exact stream-pair interaction decomposition the bilinear architecture licenses. MLP1 is exactly
bilinear (mo1 = T1(x,x)/(||x||^2/D) + bias, rms-as-gauge verified 1.2e-6 in qk_mlp1_composed_fold),
and its pre-norm input is an EXACT sum of four analytic streams: x_pre1 = E + A0 + M0 + A1 with
E = (lam0*(b0l0+b0l1)+lam1)*x0 (embedding), A0 = lam0*attn0-out, M0 = lam0*mlp0-out, A1 = attn1-out.
Bilinearity => mo1 decomposes EXACTLY into 10 pairwise interaction terms T1(S_i,S_j) (coef 2 off-
diagonal) sharing the common 1/rho^2 gauge scalar, plus the bias. PROVENANCE-based, architecture-
given split of the hub. Question: is the redundant hub code STRUCTURED BY PROVENANCE?

RUN: (1) verify sum-of-terms+bias == mo1 (~1e-5); (2) per-term centered energy shares on held data;
(3) SUFFICIENCY by provenance: mo1 -> per-position mean + subsets of the terms' deviations (singles,
all-but-one = necessity, stream-involving groups, diagonal vs cross, top-k by energy); global delta
cross-entropy +- paired SE vs full model on held FW[448:600,:128].
Harness verbatim from qk_hub_threshold.py / qk_hub_hierarchy.py; streams verbatim from
qk_mlp1_composed_fold.py; GPU guard verbatim from qk_unsup_classpush.py. Batch 6, <4GB."""
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
HELD = FW[448:600, :128].to(DEV); B0 = 6; LI = 1
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

def pair_terms(E_, A0_, M0_, A1_, xpre):
    """10 interaction terms (list of (B,T,D)), sharing the common 1/rho^2 gauge; sum+bias1 == mo1."""
    rho2 = xpre.pow(2).sum(-1, keepdim=True) / D
    Ss = [E_, A0_, M0_, A1_]
    PL = [s @ L1.T for s in Ss]; PR = [s @ R1.T for s in Ss]
    terms = []
    for (i, j) in PAIRS:
        t_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ D1w.T)
        if i != j: t_ = 2.0 * t_
        terms.append(t_ / rho2)
    return terms

@torch.no_grad()
def fwd(idx, mode=None, subset=None, TMEAN=None, MEANF=None, stats=None):
    """Forward verbatim from qk_hub_threshold.py, with stream-pair machinery at layer LI.
    mode: None (full model) | 'collect' (term sums + recon check) | 'energy' (centered energies)
          | 'subset' (mo1 -> MEANF + sum_{k in subset} (term_k - TMEAN[k]))."""
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
        if li == LI and mode is not None:
            E_ = lamE * x0; A0_ = lam0 * a0c; M0_ = lam0 * m0c; A1_ = aout
            terms = pair_terms(E_, A0_, M0_, A1_, x)   # x is x_pre1 here
            if mode == 'collect':
                for kk in range(NT): stats['tsum'][kk] += terms[kk].sum(0)
                stats['mosum'] += mo.sum(0)
                recon = sum(terms) + bias1
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
            del terms
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

# ---------------- PASS 1: per-term per-position means + DECOMPOSITION GATE ----------------
print("PASS 1: term means + reconstruction gate ...", flush=True)
st = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
      'mosum': torch.zeros(T_, D, device=DEV), 'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0}
for i in range(0, S_, B0): fwd(HELD[i:i+B0], mode='collect', stats=st)
TMEAN = torch.stack([t/S_ for t in st['tsum']])          # (10,T,D)
MO_MEAN = st['mosum']/S_
MEANF = TMEAN.sum(0) + bias1                              # mean consistent with the decomposition
gate_fro = (st['fro_num']/st['fro_den'])**0.5
gate_max = st['maxrel']
mean_consist = float((MEANF - MO_MEAN).norm()/MO_MEAN.norm())
print(f"GATE recon: global rel err {gate_fro:.2e} | max per-position rel err {gate_max:.2e} "
      f"| mean-consistency {mean_consist:.2e}", flush=True)
assert gate_fro < 1e-4, "decomposition gate FAILED"
torch.save({'TMEAN': TMEAN.cpu(), 'MEANF': MEANF.cpu(), 'PNAMES': PNAMES, 'PAIRS': PAIRS,
            'lamE': lamE, 'lam0': lam0}, f'{QK}/qk_hub_streampairs_means.pt')

# ---------------- PASS 2: base CE (full model) + centered term energies ----------------
print("PASS 2: base + term energies ...", flush=True)
st2 = {'te': [0.0]*NT, 'tot': 0.0}
ces = []
for i in range(0, S_, B0):
    ce = fwd(HELD[i:i+B0], mode='energy', TMEAN=TMEAN, MEANF=MEANF, stats=st2); ces.append(ce.cpu())
base = torch.cat(ces, 0)
shares = {PNAMES[k]: round(st2['te'][k]/st2['tot'], 4) for k in range(NT)}
print("term energy shares (||term-mean||^2 / ||mo1-mean||^2):", shares, "sum:",
      round(sum(shares.values()), 3), flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

def run(subset):
    out = []
    for i in range(0, S_, B0):
        out.append(fwd(HELD[i:i+B0], mode='subset', subset=subset, TMEAN=TMEAN, MEANF=MEANF).cpu())
    return torch.cat(out, 0)

# ---------------- PASS 3: sufficiency / necessity by provenance ----------------
IDX = {n: k for k, n in enumerate(PNAMES)}
order = sorted(range(NT), key=lambda k: -st2['te'][k])
configs = [('mean_only', []), ('all_terms', list(range(NT)))]
configs += [(f'only_{PNAMES[k]}', [k]) for k in range(NT)]
configs += [(f'allbut_{PNAMES[k]}', [j for j in range(NT) if j != k]) for k in range(NT)]
configs += [('E_involving', [IDX[n] for n in ['ExE', 'ExA0', 'ExM0', 'ExA1']]),
            ('A0_involving', [IDX[n] for n in ['ExA0', 'A0xA0', 'A0xM0', 'A0xA1']]),
            ('M0_involving', [IDX[n] for n in ['ExM0', 'A0xM0', 'M0xM0', 'M0xA1']]),
            ('A1_involving', [IDX[n] for n in ['ExA1', 'A0xA1', 'M0xA1', 'A1xA1']]),
            ('diagonal', [IDX[n] for n in ['ExE', 'A0xA0', 'M0xM0', 'A1xA1']]),
            ('cross', [IDX[n] for n in ['ExA0', 'ExM0', 'ExA1', 'A0xM0', 'A0xA1', 'M0xA1']])]
configs += [(f'top{kk}_energy', order[:kk]) for kk in (2, 3, 4, 5)]
res = {'gate': {'recon_rel_err_global': gate_fro, 'recon_rel_err_max_pos': gate_max,
                'mean_consistency': mean_consist},
       'energy_shares': shares, 'energy_rank': [PNAMES[k] for k in order], 'configs': {}}
print(f"PASS 3: {len(configs)} subset configurations ...", flush=True)
t0 = time.time()
for name, sub in configs:
    mn, se = dstat(run(sub))
    res['configs'][name] = {'terms': [PNAMES[k] for k in sub], 'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"  {name:16s} [{len(sub):2d} terms] dCE {mn:+.4f} ± {se:.5f}   ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(f'{QK}/qk_hub_streampairs.json', 'w'), indent=1)
print("QK HUB STREAMPAIRS DONE", flush=True)
