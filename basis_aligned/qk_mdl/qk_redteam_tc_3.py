"""RED-TEAM of RESULTS §104 (term-sparse whole-model compression LOSES to rank allocation).

§104 is a NEGATIVE.  The job here is to check the negative is FAIR -- that the term scheme was
not handicapped by arbitrary choices.  Four attacks:

  (1) RANK-ALLOCATION FAIRNESS at the 16x budget (861,456,384 folded-tensor coefficients):
      the §104 scheme-3 variants used a uniform input rank R per used group with output
      Ro = ceil(R/2).  Attacked with (a) per-term ranks proportional to census keep-alone causal
      importance (within-layer and global normalisations, plus a damped exponent), (b) no output
      restriction at all -- every coefficient spent on the input side, (c) early-block boosted
      ranks (mu x R for layers 0-4), and additionally (d) a sweep of the arbitrary Ro/R ratio
      (0.25 / 1 / 2 versus the 0.5 that §104 fixed by fiat).  Any variant that closes most of
      the +1.9009 -> +0.8032 gap softens the negative.
  (2) BASIS-CAP CHECK: stored bases were capped at KMAX=768 columns.  Verify the cap never binds
      at any fitted rank (as-specified scheme 3 and every uniform variant), and that the rank
      search's own upper limit (hi=KMAX) was never reached.
  (3) PROFILE-MULTIPLIER ARTIFACT: §104's early-stack failure concentration was measured with
      the 125% term profile.  Re-measure the per-region breakdown at the 100% profile, and with
      a MIXED profile (early blocks 150%, the rest 100%) at the same 16x budget.
  (4) EXAMPLE GATE: verify the three quoted examples by direct substitution at their exact
      positions (probabilities AND ranks of the true token under base / term scheme / rank
      allocation), plus three counter-examples where rank allocation beats the term scheme by
      >= 0.1 nats, selected by the mirrored rule.  Also re-select term-favourable examples under
      an ABSOLUTE preservation criterion |CE_term - CE_base| < 0.5 (§104's selection used the
      signed (CE_term - CE_base) < 0.5, which also admits positions where the term scheme BEATS
      the base model -- not preservation).

  Machinery VERBATIM from qk_termcompress.py / _2.py (same forwards, same caches, same budget
  formula); the compress forward is extended with PER-TERM input ranks, gated against the shared
  path.  TRAIN FW[0:256] for grams, held FW[448:600,:128] for every causal number, paired
  standard errors throughout.  Batch<=6, GPU guard, <4GB.  Output: qk_redteam_tc.json."""
import json, math, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_redteam_tc_3.json'

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
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV)
S_, T_ = HELD.shape; STR = TRAIN.shape[0]
B0T = 4; B0R = 6; KMAX = 768
GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']; NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]; NT = len(PAIRS)
IDX = {n: k for k, n in enumerate(PNAMES)}
FULLBLK = D * D * (D + 1) // 2; FULL = NL * FULLBLK
B16 = NL * 288 * 576 * 577 // 2          # the §92 16x core budget
print(f"bilin18 NL={NL} D={D} held {S_}x{T_} 16x budget {B16}", flush=True)

# ---------------- caches ----------------
C = torch.load(f'{QK}/qk_termcompress_cache.pt', map_location='cpu', weights_only=False)
base_cached = C['base']
TMEAN = C['TMEAN'].to(DEV); MG = C['MG'].to(DEV); MEANF = C['MEANF'].to(DEV)
GEV, GVEC_C, TEV, TVEC = C['GEV'], C['GVEC'], dict(C['TEV']), dict(C['TVEC'])
RANKED, ACTIVE, PROFILES = C['RANKED'], C['ACTIVE'], C['PROFILES']
CE2 = torch.load(f'{QK}/qk_termcompress_ce2.pt', map_location='cpu', weights_only=False)
CE_STORE = CE2['CE']
cache92 = torch.load(f'{QK}/qk_rank_alloc_cache.pt', map_location='cpu', weights_only=True)
INb = [b.to(DEV) for b in cache92['INb']]; OUTb = [b.to(DEV) for b in cache92['OUTb']]
MX = [t.to(DEV) for t in cache92['MX']];   MO = [t.to(DEV) for t in cache92['MO']]
CEN = json.load(open(f'{QK}/qk_allterm_census.json'))['layers']
TC = json.load(open(f'{QK}/qk_termcompress.json'))

GV = {k: (v.to(DEV) if v is not None else None) for k, v in GVEC_C.items()}   # ~300MB, 768 cols

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
WTS = [mlp_wts(li) for li in range(NL)]

def numrank(ev, tol=1e-6):
    if ev.sum() <= 0: return 0
    return int((ev > tol*ev[0]).sum())
GNUM = {p: min(numrank(v), KMAX) for p, v in GEV.items()}
GNUM_UNCAP = {p: numrank(v) for p, v in GEV.items()}

# ---------------- the forward (verbatim + per-term input ranks) ----------------
@torch.no_grad()
def fwd_terms(idx, mode=None, spec=None, stats=None, ret_logits=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    track = mode is not None
    if track:
        cE = torch.ones((), device=DEV)
        SA = torch.zeros_like(x); SM = torch.zeros_like(x); MR = torch.zeros_like(x)
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        a = blk.attn; hcur = F.rms_norm(x, (D,))
        if track:
            cE = blk.lambdas[0]*cE + blk.lambdas[1]
            SA = blk.lambdas[0]*SA; SM = blk.lambdas[0]*SM; MR = blk.lambdas[0]*MR
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k_)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if track:
            Lw, Rw, Dw, bias = WTS[li]
            groups = [cE*x0, SA, aout, SM, MR]
            rho2 = x.pow(2).sum(-1, keepdim=True) / D
            if mode == 'gram':
                mine = [kk for (L, kk) in stats['pairs'] if L == li]
                if mine:
                    used = set()
                    for kk in mine:
                        i, j = PAIRS[kk]; used.add(i); used.add(j)
                    PLd = {g: groups[g] @ Lw.T for g in used}
                    PRd = {g: groups[g] @ Rw.T for g in used}
                    for kk in mine:
                        i, j = PAIRS[kk]
                        t = 0.5*((PLd[i]*PRd[j] + PLd[j]*PRd[i]) @ Dw.T)
                        if i != j: t = 2.0*t
                        t = t / rho2
                        stats['Gt'][(li, kk)] += torch.einsum('btd,bte->de', t, t)
                    del PLd, PRd
            elif mode == 'compress':
                sp = spec[li]
                if sp is not None:
                    kept = sp['kept']; gr = sp.get('gr'); gtr = sp.get('gtr')
                    Po = sp.get('Po'); PoS = sp.get('PoShared')
                    devsum = torch.zeros(B, T, D, device=DEV)
                    if gtr is None:
                        used = set()
                        for kk in kept:
                            i, j = PAIRS[kk]; used.add(i); used.add(j)
                        gl = {}
                        for g in used:
                            gg = groups[g]
                            if gr is not None and g in gr and GV[(li, g)] is not None:
                                P = GV[(li, g)][:, :gr[g]]; mg = MG[li, g]
                                gg = mg + ((gg - mg) @ P) @ P.T
                            gl[g] = gg
                        PLd = {g: gl[g] @ Lw.T for g in used}
                        PRd = {g: gl[g] @ Rw.T for g in used}
                        for kk in kept:
                            i, j = PAIRS[kk]
                            t = 0.5*((PLd[i]*PRd[j] + PLd[j]*PRd[i]) @ Dw.T)
                            if i != j: t = 2.0*t
                            t = t / rho2
                            dev = t - TMEAN[li, kk]
                            if Po is not None and kk in Po:
                                P = Po[kk]; dev = (dev @ P) @ P.T
                            devsum = devsum + dev
                        del PLd, PRd, gl
                    else:
                        for kk in kept:
                            i, j = PAIRS[kk]; ri, rj = gtr[kk]
                            def pr(g, r):
                                if GV[(li, g)] is None or r >= D: return groups[g]
                                P = GV[(li, g)][:, :r]; mg = MG[li, g]
                                return mg + ((groups[g] - mg) @ P) @ P.T
                            gi = pr(i, ri); gj = gi if i == j else pr(j, rj)
                            t = 0.5*((gi @ Lw.T)*(gj @ Rw.T) + (gj @ Lw.T)*(gi @ Rw.T))
                            t = (t @ Dw.T)
                            if i != j: t = 2.0*t
                            t = t / rho2
                            dev = t - TMEAN[li, kk]
                            if Po is not None and kk in Po:
                                P = Po[kk]; dev = (dev @ P) @ P.T
                            devsum = devsum + dev
                            del gi, gj, t, dev
                    if PoS is not None:
                        devsum = (devsum @ PoS) @ PoS.T
                    mo = MEANF[li].unsqueeze(0) + devsum
            del groups
        x = x + mo
        if track:
            SA = SA + aout; SM = SM + MR; MR = mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    if ret_logits: return logits
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce

@torch.no_grad()
def fwd_group_grams(idx, stats):
    """train grams of the five provenance groups (verbatim recurrences, gram mode of part 1)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cE = torch.ones((), device=DEV)
    SA = torch.zeros_like(x); SM = torch.zeros_like(x); MR = torch.zeros_like(x)
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        a = blk.attn; hcur = F.rms_norm(x, (D,))
        cE = blk.lambdas[0]*cE + blk.lambdas[1]
        SA = blk.lambdas[0]*SA; SM = blk.lambdas[0]*SM; MR = blk.lambdas[0]*MR
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k_)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        aout = a.c_proj(yh.reshape(B, T, -1)); x = x + aout
        mo = blk.mlp(F.rms_norm(x, (D,)))
        if li in stats['chunk']:
            groups = [cE*x0, SA, aout, SM, MR]
            for g in range(NG):
                stats['Gg'][(li, g)] += torch.einsum('btd,bte->de', groups[g], groups[g])
            del groups
        x = x + mo
        SA = SA + aout; SM = SM + MR; MR = mo

@torch.no_grad()
def fwd_rank(idx, PIN=None, POUT=None, MX=None, MO=None, ret_logits=False):
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
        a = blk.attn; hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        if PIN is not None and PIN[li] is not None:
            xr = MX[li].unsqueeze(0) + ((x - MX[li].unsqueeze(0)) @ PIN[li]) @ PIN[li].T
            mo = blk.mlp(F.rms_norm(xr, (D,)))
            if POUT is not None and POUT[li] is not None:
                mo = MO[li].unsqueeze(0) + ((mo - MO[li].unsqueeze(0)) @ POUT[li]) @ POUT[li].T
        else:
            mo = blk.mlp(F.rms_norm(x, (D,)))
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    if ret_logits: return logits
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce

# ---------------- base CE gate ----------------
print("BASE re-run + gate vs the §104 cache ...", flush=True)
base = torch.cat([fwd_terms(HELD[i:i+B0R]).cpu() for i in range(0, S_, B0R)], 0)
base_gate = float((base - base_cached).abs().max())
print(f"base CE {float(base.mean()):.4f} (cache {float(base_cached.mean()):.4f}, maxabs {base_gate:.2e})",
      flush=True)
assert base_gate < 1e-5, "base gate FAILED"

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))
def pairdiff(ce_a, ce_b):
    """mean and paired SE of (a - b) per position."""
    d = (ce_a - ce_b).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

res = {'meta': {'model': 'bilin18', 'held': 'FW[448:600,:128]', 'train': 'FW[0:256,:128]',
                'base_ce': round(float(base.mean()), 4), 'base_gate_maxabs': base_gate,
                'budget_16x': int(B16), 'full_params': int(FULL),
                'reference_scheme1_16x': TC['scheme1']['16x']['dCE'],
                'reference_best_term_16x': TC['scheme3_variants']['3b_perterm_125pct_16x']['dCE']}}

# ---------------- profiles ----------------
N95 = TC['meta']['terms_to_95_census']
def kprof(mult, layers=None, other=1.0):
    out = []
    for L in range(NL):
        mm = mult if (layers is None or L in layers) else other
        out.append(min(len(ACTIVE[L]), max(1, math.ceil(mm * N95[L]))))
    return out
PROF = {'100pct': kprof(1.00), '125pct': kprof(1.25),
        'early150_rest100': kprof(1.50, layers=set(range(5)), other=1.00)}
KEPT = {t: [RANKED[L][:p[L]] for L in range(NL)] for t, p in PROF.items()}
for t, p in PROF.items(): print(f"profile {t}: {p} (sum {sum(p)})", flush=True)

# ---------------- extra term-output grams ----------------
need = sorted({(L, kk) for t in KEPT for L in range(NL) for kk in KEPT[t][L]} - set(TVEC.keys()))
print(f"extra term output grams needed: {len(need)}", flush=True)
if need:
    t0 = time.time()
    stg = {'pairs': set(need), 'Gt': {p: torch.zeros(D, D, device=DEV) for p in need}}
    for i in range(0, STR, B0T): fwd_terms(TRAIN[i:i+B0T], mode='gram', stats=stg)
    for p, G in stg['Gt'].items():
        w, vec = torch.linalg.eigh(G)
        TEV[p] = w.flip(0).clamp_min(0).double().cpu().numpy()
        TVEC[p] = vec.flip(1)[:, :KMAX].contiguous().cpu()
    del stg; torch.cuda.empty_cache()
    print(f"extra grams done ({time.time()-t0:.0f}s)", flush=True)
TNUM = {p: min(numrank(v), KMAX) for p, v in TEV.items()}

# ===== allocation machinery (verbatim from qk_redteam_tc.py) =====
def cost_of(alloc):
    """alloc: {L: {kk: (ri, rj, ro)}} -> folded-tensor coefficient count."""
    b = 0
    for L, d in alloc.items():
        for kk, (ri, rj, ro) in d.items():
            i, j = PAIRS[kk]
            b += ri*(ri+1)//2 * ro if i == j else ri*rj*ro
    return b

def alloc_uniform(kept, R, rho=0.5, mu_early=1.0, early=frozenset(range(5)), no_out=False):
    al = {}
    for L in range(NL):
        RL = R * (mu_early if L in early else 1.0)
        d = {}
        for kk in kept[L]:
            i, j = PAIRS[kk]
            ri = max(1, min(int(round(RL)), GNUM[(L, i)]))
            rj = max(1, min(int(round(RL)), GNUM[(L, j)]))
            if no_out: ro = D
            else: ro = max(1, min(int(math.ceil(rho*RL)), TNUM[(L, kk)], KMAX))
            d[kk] = (ri, rj, ro)
        al[L] = d
    return al

def term_importance():
    """Per-term causal importance from the §89 census: how much of the layer floor the term
    recovers when kept ALONE (floor - only_dCE), available for the top-6 energy terms; terms
    beyond the top-6 are extrapolated by their energy share relative to the 6th term, scaled by
    the 6th term's recovered value (keeps them ranked below, monotone in energy)."""
    W = {}
    for L in range(NL):
        r = CEN[str(L)]; floor = r['floor_dCE']; er = r['energy_rank']
        sh = r['energy_shares']
        vals = {}
        for n in er[:6]:
            key = f'only_{n}'
            if key in r['configs']:
                vals[IDX[n]] = max(floor - r['configs'][key]['dCE'], 1e-4*max(floor, 1e-3))
        if vals:
            vmin = min(vals.values()); s6 = max(sh.get(er[min(5, len(er)-1)], 1e-9), 1e-12)
        else:
            vmin, s6 = 1.0, 1.0
        for n in er[6:]:
            vals[IDX[n]] = max(vmin * (sh.get(n, 0.0)/s6), 1e-6)
        W[L] = vals
    return W
WIMP = term_importance()

def alloc_importance(kept, c, gamma=1.0, scope='layer', rho=0.5, rmin=16):
    """rank ~ c * (normalised importance)^gamma; 'layer' normalises within each layer (layers
    equal on average), 'global' normalises across all kept terms (high-floor layers get more)."""
    if scope == 'global':
        allw = [WIMP[L][kk] for L in range(NL) for kk in kept[L] if kk in WIMP[L]]
        gm = float(np.mean([w**gamma for w in allw]))
    al = {}
    for L in range(NL):
        ws = {kk: WIMP[L].get(kk, 1e-6)**gamma for kk in kept[L]}
        if scope == 'layer':
            nrm = float(np.mean(list(ws.values()))) if ws else 1.0
        else:
            nrm = gm
        d = {}
        for kk in kept[L]:
            i, j = PAIRS[kk]
            RL = c * ws[kk]/max(nrm, 1e-12)
            ri = max(rmin, min(int(round(RL)), GNUM[(L, i)]))
            rj = max(rmin, min(int(round(RL)), GNUM[(L, j)]))
            ro = max(1, min(int(math.ceil(rho*RL)), TNUM[(L, kk)], KMAX))
            d[kk] = (ri, rj, ro)
        al[L] = d
    return al

def fit_scalar(fn, B, lo=0.05, hi=2000.0, iters=70):
    best = None
    for _ in range(iters):
        mid = 0.5*(lo+hi)
        al = fn(mid); b = cost_of(al)
        if b <= B: best = (mid, b, al); lo = mid
        else: hi = mid
    if best is None: best = (lo, cost_of(fn(lo)), fn(lo))
    return best

def build_spec(kept, alloc, per_term_inputs=True, layers=None, no_out=False):
    spec = []
    for L in range(NL):
        if layers is not None and L not in layers: spec.append(None); continue
        d = alloc[L]
        sp = {'kept': kept[L]}
        if per_term_inputs:
            sp['gtr'] = {kk: (d[kk][0], d[kk][1]) for kk in kept[L]}
        else:
            gr = {}
            for kk in kept[L]:
                i, j = PAIRS[kk]
                gr[i] = max(gr.get(i, 0), d[kk][0]); gr[j] = max(gr.get(j, 0), d[kk][1])
            sp['gr'] = gr
        if not no_out:
            sp['Po'] = {kk: TVEC[(L, kk)][:, :d[kk][2]].to(DEV) for kk in kept[L]}
        spec.append(sp)
    return spec

def run(spec):
    ce = torch.cat([fwd_terms(HELD[i:i+B0T], mode='compress', spec=spec).cpu()
                    for i in range(0, S_, B0T)], 0)
    return ce

CE1_16 = CE_STORE['s1_16x']; CE3_16 = CE_STORE['3b_perterm_125pct_16x']
CELLS = {}
def cell(name, kept, alloc, note='', per_term_inputs=True, no_out=False, layers=None, store=True):
    b = cost_of(alloc)
    spec = build_spec(kept, alloc, per_term_inputs, layers, no_out)
    t0 = time.time(); ce = run(spec); del spec; torch.cuda.empty_cache()
    mn, se = dstat(ce)
    dv1, sv1 = pairdiff(ce, CE1_16)          # vs rank allocation at 16x
    dv3, sv3 = pairdiff(ce, CE3_16)          # vs §104's best term variant
    rec = {'note': note, 'budget': int(b), 'budget_frac_of_16x': round(b/B16, 4),
           'n_terms': int(sum(len(k) for k in kept)),
           'dCE': round(mn, 4), 'SE': round(se, 5),
           'vs_rank_alloc_16x': [round(dv1, 4), round(sv1, 5)],
           'vs_best_term_104': [round(dv3, 4), round(sv3, 5)],
           'secs': round(time.time()-t0, 1)}
    if store: CELLS[name] = ce
    print(f"[{name}] budget {b/1e6:.0f}M ({b/B16:.3f}x16x) dCE {mn:+.4f} +- {se:.5f} | "
          f"vs rank-alloc {dv1:+.4f} +- {sv1:.5f}", flush=True)
    return rec



# =====================================================================================
# ATTACK 1c -- push the winning direction (richer term sets) to its boundary
# =====================================================================================
print("\n=== ATTACK 1c: all-terms family ===", flush=True)
A1 = {'cells': {}}
PROF['allterms'] = [len(ACTIVE[L]) for L in range(NL)]
KEPT['allterms'] = [list(ACTIVE[L]) for L in range(NL)]
need = sorted({(L, kk) for L in range(NL) for kk in KEPT['allterms'][L]} - set(TVEC.keys()))
print(f"extra term output grams needed: {len(need)}", flush=True)
for chunk in [need[i:i+60] for i in range(0, len(need), 60)]:
    stg = {'pairs': set(chunk), 'Gt': {p: torch.zeros(D, D, device=DEV) for p in chunk}}
    for i in range(0, STR, B0T): fwd_terms(TRAIN[i:i+B0T], mode='gram', stats=stg)
    for p, G in stg['Gt'].items():
        w, vec = torch.linalg.eigh(G)
        TEV[p] = w.flip(0).clamp_min(0).double().cpu().numpy()
        TVEC[p] = vec.flip(1)[:, :KMAX].contiguous().cpu()
    del stg; torch.cuda.empty_cache()
TNUM = {p: min(numrank(v), KMAX) for p, v in TEV.items()}

BUD = {'16x': B16, '4x': NL*1152*576*577//2, '128x': NL*144*288*289//2}
REF = {'16x': (TC['scheme1']['16x']['dCE'], 's1_16x'), '4x': (TC['scheme1']['4x']['dCE'], 's1_4x'),
       '128x': (TC['scheme1']['128x']['dCE'], 's1_128x')}

def fitted(kept, B, **kw):
    R, b, al = fit_scalar(lambda c: alloc_uniform(kept, c, **kw), B)
    Ri = int(round(R)); al = alloc_uniform(kept, Ri, **kw)
    while cost_of(al) > B: Ri -= 1; al = alloc_uniform(kept, Ri, **kw)
    return Ri, al

def runcell(name, ptag, btag, **kw):
    Ri, al = fitted(KEPT[ptag], BUD[btag], **kw)
    b = cost_of(al)
    spec = build_spec(KEPT[ptag], al, False, None, False)
    ce = run(spec); del spec; torch.cuda.empty_cache()
    mn, se = dstat(ce); dv, sv = pairdiff(ce, CE_STORE[REF[btag][1]])
    rec = {'profile': ptag, 'budget_tag': btag, 'R_in': Ri, 'kw': {k: str(v) for k, v in kw.items()},
           'n_terms': int(sum(len(k) for k in KEPT[ptag])), 'budget': int(b),
           'budget_frac': round(b/BUD[btag], 4), 'dCE': round(mn, 4), 'SE': round(se, 5),
           'rank_alloc_same_budget': REF[btag][0],
           'paired_vs_rank_alloc': [round(dv, 4), round(sv, 5)],
           'paired_z': round(dv/max(sv, 1e-12), 1)}
    A1['cells'][name] = rec
    print(f"[{name}] R={Ri} budget {b/1e6:.0f}M dCE {mn:+.4f} +- {se:.5f} | vs rank-alloc "
          f"{dv:+.4f} +- {sv:.5f} (z {rec['paired_z']})", flush=True)
    json.dump(A1, open(OUT, 'w'), indent=1)
    return rec

runcell('allterms_rho1.5_16x', 'allterms', '16x', rho=1.5)
runcell('allterms_rho2.0_16x', 'allterms', '16x', rho=2.0)
runcell('allterms_rho1_mu2_16x', 'allterms', '16x', rho=1.0, mu_early=2.0)
runcell('allterms_rho1_mu3_16x', 'allterms', '16x', rho=1.0, mu_early=3.0)
best = min(A1['cells'].items(), key=lambda kv: kv[1]['dCE'])
print(f"best 16x in the all-terms family: {best[0]} {best[1]['dCE']}", flush=True)
kwbest = {k: float(v) for k, v in best[1]['kw'].items()}
runcell('allterms_best_4x', 'allterms', '4x', **kwbest)
runcell('allterms_best_128x', 'allterms', '128x', **kwbest)
A1['peak_gpu_MiB'] = round(torch.cuda.max_memory_allocated()/2**20, 1)
json.dump(A1, open(OUT, 'w'), indent=1)
print("QK REDTEAM TERMCOMPRESS PART 3 DONE", flush=True)
