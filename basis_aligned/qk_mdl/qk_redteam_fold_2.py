"""ADVERSARIAL RED-TEAM of the fold-first attribution arc, part 2 of 2: ATTACK 4 -- the layer-17
"mutually-cancelling mixer" pathology of section 89. Could the pathology (diagonal-only worse than
the mean-only floor; energy shares summing to ~1.54; non-monotone greedy curve) be an artifact of
the keep-subset construction interacting with the readout, or of the shared post-gauge bookkeeping?

TESTS: (a) measure the actual covariance/cosine structure between layer 17's centered term outputs
(15x15 Gram of per-position-centered deviations): are the big terms genuinely anti-aligned? Report
the cancellation index sum_i ||t_i||^2 / ||sum_i t_i||^2 and the most negative pairwise cosines,
with layer 1 (from the same construction) as the healthy contrast. (b) alternative bookkeeping:
keep-subsets defined on the PRE-gauge polynomial numerators (per-position numerator means, the
per-sample gauge 1/rho^2 applied to the reassembled numerator) -- if the pathology signature
(diagonal worse than mean-only, greedy non-monotone at top-4) persists under this second
bookkeeping, it is genuine and not a gauge artifact.

Machinery VERBATIM from qk_allterm_census.py (five coarse groups, pair-term polarization, gates,
keep-subset harness). Held FW[448:600,:128], paired standard errors, batch 6, <4GB.
Extends qk_redteam_fold.json (run part 1 first)."""
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
HELD = FW[448:600, :128].to(DEV); B0 = 6
S_, T_ = HELD.shape

GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']
NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]
NT = len(PAIRS)
DIAG = [k for k, (i, j) in enumerate(PAIRS) if i == j]
CROSS = [k for k, (i, j) in enumerate(PAIRS) if i != j]

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())

def pair_numers(groups, Lw, Rw, Dw):
    """Pre-gauge numerators N_k; term_k = N_k/rho2. VERBATIM polarization from qk_allterm_census."""
    PL = [g @ Lw.T for g in groups]; PR = [g @ Rw.T for g in groups]
    numers = []
    for (i, j) in PAIRS:
        n_ = 0.5 * ((PL[i] * PR[j] + PL[j] * PR[i]) @ Dw.T)
        if i != j: n_ = 2.0 * n_
        numers.append(n_)
    return numers

@torch.no_grad()
def fwd(idx, LI=None, mode=None, subset=None, TMEAN=None, MEANF=None, PMEAN=None, PMEANALL=None,
        stats=None, W=None):
    """Forward + coarse-group accumulators verbatim from qk_allterm_census.py.
    mode: None full | 'collect' (post-gauge term means + pre-gauge numerator means + gate)
          | 'cov' (15x15 covariance of centered post-gauge terms)
          | 'subset' (post-gauge bookkeeping, verbatim census)
          | 'subset_pre' (PRE-gauge bookkeeping: mo -> (PMEANALL + sum_kept (N_k - PMEAN_k))/rho2 + bias)."""
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
            groups = [cE*x0, SA, aout, SM, MR]
            rho2 = x.pow(2).sum(-1, keepdim=True) / D
            numers = pair_numers(groups, W[0], W[1], W[2])
            if mode == 'collect':
                for kk in range(NT):
                    stats['tsum'][kk] += (numers[kk]/rho2).sum(0)
                    stats['psum'][kk] += numers[kk].sum(0)
                stats['mosum'] += mo.sum(0)
                recon = sum(numers)/rho2 + W[3]
                num = (recon - mo).norm(dim=-1); den = mo.norm(dim=-1).clamp_min(1e-8)
                stats['maxrel'] = max(stats['maxrel'], float((num/den).max()))
                stats['fro_num'] += float((recon - mo).pow(2).sum()); stats['fro_den'] += float(mo.pow(2).sum())
            elif mode == 'cov':
                devs = torch.stack([(numers[kk]/rho2 - TMEAN[kk]).reshape(-1) for kk in range(NT)])
                stats['C'] += devs @ devs.T
                stats['tot'] += float((mo - MEANF).pow(2).sum())
            elif mode == 'subset':
                new = MEANF.unsqueeze(0).expand(B, -1, -1)
                for kk in subset: new = new + (numers[kk]/rho2 - TMEAN[kk])
                mo = new.to(x.dtype)
            elif mode == 'subset_pre':
                newp = PMEANALL.unsqueeze(0).expand(B, -1, -1)
                for kk in subset: newp = newp + (numers[kk] - PMEAN[kk])
                mo = (newp/rho2 + W[3]).to(x.dtype)
            del numers, groups
        x = x + mo
        if track and li < LI:
            SA = SA + aout; SM = SM + MR; MR = mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1), reduction='none').view(B, T-1)
    return ce

res = json.load(open(OUT)) if __import__('os').path.exists(OUT) else {}
print("BASE pass ...", flush=True)
base = torch.cat([fwd(HELD[i:i+B0]).cpu() for i in range(0, S_, B0)], 0)
print(f"base CE {float(base.mean()):.4f}", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

res['attack4'] = {}
for LI, tag in [(17, 'L17'), (1, 'L1_contrast')]:
    W = mlp_wts(LI)
    print(f"=== {tag}: PASS 1 collect ===", flush=True)
    st = {'tsum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
          'psum': [torch.zeros(T_, D, device=DEV) for _ in range(NT)],
          'mosum': torch.zeros(T_, D, device=DEV), 'maxrel': 0.0, 'fro_num': 0.0, 'fro_den': 0.0}
    for i in range(0, S_, B0): fwd(HELD[i:i+B0], LI=LI, mode='collect', stats=st, W=W)
    TMEAN = torch.stack([t/S_ for t in st['tsum']])
    PMEAN = torch.stack([t/S_ for t in st['psum']])
    PMEANALL = PMEAN.sum(0)
    MEANF = TMEAN.sum(0) + W[3]
    gate_fro = (st['fro_num']/st['fro_den'])**0.5
    print(f"{tag} GATE recon global {gate_fro:.2e} maxpos {st['maxrel']:.2e}", flush=True)
    assert gate_fro < 1e-4

    print(f"=== {tag}: PASS 2 covariance ===", flush=True)
    st2 = {'C': torch.zeros(NT, NT, device=DEV, dtype=torch.float64), 'tot': 0.0}
    for i in range(0, S_, B0):
        fwd(HELD[i:i+B0], LI=LI, mode='cov', TMEAN=TMEAN, MEANF=MEANF, stats=st2, W=W)
    C = st2['C'].cpu().numpy(); tot = st2['tot']
    en = np.diag(C)
    shares = {PNAMES[k]: round(en[k]/tot, 4) for k in range(NT)}
    active = [k for k in range(NT) if en[k]/tot > 1e-9]
    order = sorted(active, key=lambda k: -en[k])
    sumC = float(np.ones(NT) @ C @ np.ones(NT))
    cancel_idx = float(en.sum()/sumC)
    dd = np.sqrt(np.clip(en, 1e-30, None))
    cosM = C/np.outer(dd, dd)
    pairs = []
    for a_ in range(NT):
        for b_ in range(a_+1, NT):
            if a_ in active and b_ in active and min(en[a_], en[b_])/tot > 0.005:
                pairs.append((PNAMES[a_], PNAMES[b_], round(float(cosM[a_, b_]), 3),
                              round(en[a_]/tot, 3), round(en[b_]/tot, 3)))
    pairs.sort(key=lambda p: p[2])
    rec = {'gate_recon_rel_err_global': gate_fro,
           'energy_shares': shares, 'shares_sum': round(float(en.sum()/tot), 4),
           'cancellation_index_sumE_over_normsq_of_sum': round(cancel_idx, 4),
           'energy_rank': [PNAMES[k] for k in order[:8]],
           'most_negative_cosine_pairs': pairs[:8],
           'most_positive_cosine_pairs': pairs[-3:]}
    print(f"{tag}: shares sum {en.sum()/tot:.4f} cancellation index {cancel_idx:.4f}", flush=True)
    print(f"{tag}: most negative pairs {pairs[:5]}", flush=True)

    def run(subset, pre=False):
        out = []
        for i in range(0, S_, B0):
            out.append(fwd(HELD[i:i+B0], LI=LI, mode='subset_pre' if pre else 'subset',
                           subset=subset, TMEAN=TMEAN, MEANF=MEANF, PMEAN=PMEAN,
                           PMEANALL=PMEANALL, W=W).cpu())
        return torch.cat(out, 0)

    if LI == 17:
        cfgs = [('mean_only', []), ('all_terms', list(range(NT))),
                ('diagonal', [k for k in DIAG if k in active]),
                ('cross', [k for k in CROSS if k in active])]
        cfgs += [(f'top{kk}_energy', order[:kk]) for kk in (1, 2, 3, 4, 5, 6)]
        rec['configs_postgauge'] = {}; rec['configs_pregauge'] = {}
        for name, sub in cfgs:
            mn, se = dstat(run(sub, pre=False))
            rec['configs_postgauge'][name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
            mnp, sep = dstat(run(sub, pre=True))
            rec['configs_pregauge'][name] = {'dCE': round(mnp, 4), 'SE': round(sep, 5)}
            print(f"  L17 {name:14s} post {mn:+.4f} +- {se:.5f} | pre {mnp:+.4f} +- {sep:.5f}", flush=True)
    res['attack4'][tag] = rec
    json.dump(res, open(OUT, 'w'), indent=1)
    del TMEAN, PMEAN, MEANF, st, st2
    torch.cuda.empty_cache()

print("QK REDTEAM FOLD PART 2 DONE", flush=True)
