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
OUT = f'{QK}/qk_redteam_tc.json'

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

# =====================================================================================
# ATTACK 2 -- basis cap
# =====================================================================================
print("\n=== ATTACK 2: 768-column basis cap ===", flush=True)
cap = {'KMAX': KMAX, 'fitted_ranks': {}, 'binds': []}
for tag in ['128x', '16x', '4x']:
    s = TC['scheme3'][tag]; mx = 0
    for rec in s['ranks_per_layer']:
        for k, v in rec['groups'].items():
            mx = max(mx, v)
            if v >= KMAX: cap['binds'].append(['scheme3_asspec', tag, 'group', k, v])
        for k, v in rec['term_out'].items():
            mx = max(mx, v)
            if v >= KMAX: cap['binds'].append(['scheme3_asspec', tag, 'term_out', k, v])
    cap['fitted_ranks'][f'scheme3_asspec_{tag}'] = {'max_rank': mx, 'f': s['f']}
for k, v in TC['scheme3_variants'].items():
    if not isinstance(v, dict) or 'R_in' not in v: continue
    cap['fitted_ranks'][k] = {'R_in': v['R_in'], 'R_out': v['R_out']}
    if v['R_in'] >= KMAX or v['R_out'] >= KMAX: cap['binds'].append(['variant', k, v['R_in'], v['R_out']])
# does the per-group numerical-rank CAP alter any effective rank?  (min(R, GNUM) vs min(R, uncapped))
alter = [[f'L{L}', GNAMES[g], GNUM_UNCAP[(L, g)]] for L in range(NL) for g in range(NG)
         if GNUM[(L, g)] != GNUM_UNCAP[(L, g)]]
cap['groups_whose_numerical_rank_exceeds_KMAX'] = alter
cap['max_R_ever_requested'] = max([v['R_in'] for v in TC['scheme3_variants'].values()
                                   if isinstance(v, dict) and 'R_in' in v] + [576])
cap['verdict'] = ('cap NEVER binds: max fitted rank across every §104 configuration is '
                  f"{max(list(v.get('max_rank', 0) for v in cap['fitted_ranks'].values()) + [v.get('R_in', 0) for v in cap['fitted_ranks'].values()])}"
                  f" < {KMAX}") if not cap['binds'] else 'cap BINDS -- see binds'
res['attack2_basis_cap'] = cap
print(json.dumps(cap, indent=1)[:1500], flush=True)
json.dump(res, open(OUT, 'w'), indent=1)

# =====================================================================================
# allocation machinery
# =====================================================================================
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
# ATTACK 1 -- rank-allocation fairness at the 16x budget
# =====================================================================================
print("\n=== ATTACK 1: allocation fairness at the 16x budget ===", flush=True)
A1 = {'reference': {'rank_alloc_16x_dCE': TC['scheme1']['16x']['dCE'],
                    'best_term_104_dCE': TC['scheme3_variants']['3b_perterm_125pct_16x']['dCE'],
                    'gap_nats': round(1.9009 - 0.8032, 4)}, 'cells': {}}

# (0) reproduction gate: §104's best 16x variant, rebuilt here, shared-input path AND per-term path
R0, b0, al0 = fit_scalar(lambda c: alloc_uniform(KEPT['125pct'], c, rho=0.5), B16)
R0i = int(round(R0))
al0 = alloc_uniform(KEPT['125pct'], R0i, rho=0.5)
print(f"repro: fitted R={R0i} budget {cost_of(al0)/1e6:.0f}M (§104 said R=229, 857M)", flush=True)
A1['cells']['repro_104_best_sharedpath'] = cell(
    'repro_104_best_sharedpath', KEPT['125pct'], al0,
    'reproduction of §104 3b_perterm_125pct_16x through the shared-input path',
    per_term_inputs=False)
A1['cells']['repro_104_best_pertermpath'] = cell(
    'repro_104_best_pertermpath', KEPT['125pct'], al0,
    'same allocation through the NEW per-term input path (machinery equivalence gate)',
    per_term_inputs=True)
A1['machinery_gate'] = {
    'shared_vs_perterm_dCE_diff': round(A1['cells']['repro_104_best_pertermpath']['dCE']
                                        - A1['cells']['repro_104_best_sharedpath']['dCE'], 6),
    'shared_vs_104_json_diff': round(A1['cells']['repro_104_best_sharedpath']['dCE'] - 1.9009, 6)}
print("machinery gate:", A1['machinery_gate'], flush=True)
json.dump(res | {'attack1_allocation_fairness': A1}, open(OUT, 'w'), indent=1)

# (d) the arbitrary Ro/R ratio
for rho in [0.25, 1.0, 2.0]:
    R, b, al = fit_scalar(lambda c: alloc_uniform(KEPT['125pct'], c, rho=rho), B16)
    Ri = int(round(R)); al = alloc_uniform(KEPT['125pct'], Ri, rho=rho)
    while cost_of(al) > B16: Ri -= 1; al = alloc_uniform(KEPT['125pct'], Ri, rho=rho)
    A1['cells'][f'ratio_rho{rho}'] = cell(f'ratio_rho{rho}', KEPT['125pct'], al,
        f'uniform input rank R={Ri}, output rank = ceil({rho}*R) (§104 fixed rho=0.5)')
    A1['cells'][f'ratio_rho{rho}']['R_in'] = Ri
    json.dump(res | {'attack1_allocation_fairness': A1}, open(OUT, 'w'), indent=1)

# (a) per-term ranks proportional to census keep-alone causal importance
for scope, gamma in [('layer', 1.0), ('layer', 0.5), ('global', 1.0)]:
    fn = lambda c, s=scope, g=gamma: alloc_importance(KEPT['125pct'], c, gamma=g, scope=s)
    c_, b, al = fit_scalar(fn, B16)
    rs = [al[L][kk][0] for L in range(NL) for kk in KEPT['125pct'][L]]
    A1['cells'][f'importance_{scope}_g{gamma}'] = cell(
        f'importance_{scope}_g{gamma}', KEPT['125pct'], al,
        f'per-term input rank proportional to (census keep-alone importance)^{gamma}, '
        f'{scope}-normalised; rank range {min(rs)}-{max(rs)}')
    A1['cells'][f'importance_{scope}_g{gamma}']['rank_range'] = [int(min(rs)), int(max(rs))]
    json.dump(res | {'attack1_allocation_fairness': A1}, open(OUT, 'w'), indent=1)

# (b) no output restriction at all -- every coefficient on the input side
for ptag in ['125pct', '100pct']:
    fn = lambda c, p=ptag: alloc_uniform(KEPT[p], c, no_out=True)
    R, b, al = fit_scalar(fn, B16)
    Ri = int(round(R)); al = alloc_uniform(KEPT[ptag], Ri, no_out=True)
    while cost_of(al) > B16: Ri -= 1; al = alloc_uniform(KEPT[ptag], Ri, no_out=True)
    A1['cells'][f'no_output_{ptag}'] = cell(f'no_output_{ptag}', KEPT[ptag], al,
        f'outputs EXACT (ro=D=1152), all budget on inputs: R={Ri}', no_out=True)
    A1['cells'][f'no_output_{ptag}']['R_in'] = Ri
    json.dump(res | {'attack1_allocation_fairness': A1}, open(OUT, 'w'), indent=1)

# (c) early-block boosted ranks
for mu in [1.5, 2.0, 3.0]:
    fn = lambda c, u=mu: alloc_uniform(KEPT['125pct'], c, rho=0.5, mu_early=u)
    R, b, al = fit_scalar(fn, B16)
    Ri = int(round(R)); al = alloc_uniform(KEPT['125pct'], Ri, rho=0.5, mu_early=mu)
    while cost_of(al) > B16: Ri -= 1; al = alloc_uniform(KEPT['125pct'], Ri, rho=0.5, mu_early=mu)
    Re = min(int(round(mu*Ri)), 768)
    A1['cells'][f'early_boost_mu{mu}'] = cell(f'early_boost_mu{mu}', KEPT['125pct'], al,
        f'layers 0-4 get rank {Re}, layers 5-17 rank {Ri} (§104 used {R0i} everywhere)')
    A1['cells'][f'early_boost_mu{mu}'].update({'R_early': Re, 'R_rest': Ri})
    json.dump(res | {'attack1_allocation_fairness': A1}, open(OUT, 'w'), indent=1)

best = min([(v['dCE'], k) for k, v in A1['cells'].items() if not k.startswith('repro')])
A1['best_variant'] = {'name': best[1], 'dCE': best[0],
                      'gap_closed_frac': round((1.9009 - best[0])/(1.9009 - 0.8032), 4)}
print(f"BEST attack-1 variant: {best[1]} dCE {best[0]:+.4f}  "
      f"(gap closed {A1['best_variant']['gap_closed_frac']:.1%})", flush=True)

# (bonus) the §104 'decisive control': group-factorised input restriction at rank 576 vs §92's
# joint restriction at rank 576 -- with a PAIRED standard error on the difference.
print("bonus: paired test of the rank-576 'ties' control ...", flush=True)
al576 = alloc_uniform(KEPT['125pct'], 576, no_out=True)
spec = build_spec(KEPT['125pct'], al576, per_term_inputs=False, no_out=True)
ce576 = run(spec); del spec; torch.cuda.empty_cache()
mn, se = dstat(ce576); dv, sv = pairdiff(ce576, CE_STORE['s1_4x'])
A1['rank576_control'] = {
    'group_factorised_input_only_R576_dCE': [round(mn, 4), round(se, 5)],
    'joint_input_only_rank576_92anchor_dCE': [TC['scheme1']['4x']['dCE'], TC['scheme1']['4x']['SE']],
    'paired_difference': [round(dv, 4), round(sv, 5)],
    'paired_z': round(dv/max(sv, 1e-12), 2),
    'note': ('§104 called these a statistical TIE from the two unpaired standard errors; this is '
             'the paired test on the same positions.')}
print("rank576 control:", A1['rank576_control'], flush=True)
res['attack1_allocation_fairness'] = A1
json.dump(res, open(OUT, 'w'), indent=1)

# =====================================================================================
# ATTACK 3 -- profile-multiplier artifact in the early-stack failure concentration
# =====================================================================================
print("\n=== ATTACK 3: is the early-stack concentration a profile artifact? ===", flush=True)
REGIONS = {'early_0_4': list(range(0, 5)), 'distributed_5_11': list(range(5, 12)),
           'readout_12_17': list(range(12, 18))}
A3 = {'reference_125pct': {'whole': 1.9009,
      'regions': {r: TC['regional']['regions'][r]['scheme3']['dCE'] for r in REGIONS},
      'scheme1_regions': {r: TC['regional']['regions'][r]['scheme1']['dCE'] for r in REGIONS}},
      'profiles': {}}
for ptag in ['100pct', 'early150_rest100']:
    fn = lambda c, p=ptag: alloc_uniform(KEPT[p], c, rho=0.5)
    R, b, al = fit_scalar(fn, B16)
    Ri = int(round(R)); al = alloc_uniform(KEPT[ptag], Ri, rho=0.5)
    while cost_of(al) > B16: Ri -= 1; al = alloc_uniform(KEPT[ptag], Ri, rho=0.5)
    rec = {'R_in': Ri, 'R_out': int(math.ceil(0.5*Ri)), 'k_profile': PROF[ptag],
           'budget': int(cost_of(al))}
    w = cell(f'A3_{ptag}_whole', KEPT[ptag], al, f'whole model, profile {ptag}, R={Ri}',
             per_term_inputs=False, store=False)
    rec['whole'] = {'dCE': w['dCE'], 'SE': w['SE']}
    rec['regions'] = {}
    for rtag, lay in REGIONS.items():
        rr = cell(f'A3_{ptag}_{rtag}', KEPT[ptag], al, f'region {rtag} only',
                  per_term_inputs=False, layers=set(lay), store=False)
        rec['regions'][rtag] = {'dCE': rr['dCE'], 'SE': rr['SE']}
    e = rec['regions']['early_0_4']['dCE']
    rec['early_share_of_region_sum'] = round(
        e/sum(rec['regions'][r]['dCE'] for r in REGIONS), 4)
    rec['ratio_to_scheme1_per_region'] = {
        r: round(rec['regions'][r]['dCE']/A3['reference_125pct']['scheme1_regions'][r], 2)
        for r in REGIONS}
    A3['profiles'][ptag] = rec
    print(f"  [{ptag}] early share {rec['early_share_of_region_sum']:.3f} ratios "
          f"{rec['ratio_to_scheme1_per_region']}", flush=True)
    json.dump(res | {'attack3_profile_artifact': A3}, open(OUT, 'w'), indent=1)
ref = A3['reference_125pct']
ref['early_share_of_region_sum'] = round(
    ref['regions']['early_0_4']/sum(ref['regions'].values()), 4)
ref['ratio_to_scheme1_per_region'] = {r: round(ref['regions'][r]/ref['scheme1_regions'][r], 2)
                                      for r in REGIONS}
res['attack3_profile_artifact'] = A3
json.dump(res, open(OUT, 'w'), indent=1)

# =====================================================================================
# ATTACK 4 -- example gate by direct substitution
# =====================================================================================
print("\n=== ATTACK 4: example gate ===", flush=True)
from transformers import AutoTokenizer
tokzr = AutoTokenizer.from_pretrained('gpt2')
held_np = HELD.cpu().numpy()
d = (CE1_16 - CE3_16)                       # positive: the term scheme is better here
absnear3 = ((CE3_16 - base).abs() < 0.5)    # ABSOLUTE preservation by the term scheme
absnear1 = ((CE1_16 - base).abs() < 0.5)    # ABSOLUTE preservation by rank allocation
signed3 = ((CE3_16 - base) < 0.5)           # §104's own (signed) criterion

QUOTED = [(147, 124), (83, 82), (151, 95)]
# counter-examples: mirrored rule -- rank allocation better by >=0.1, and IT preserves the base
cand1 = ((CE3_16 - CE1_16) * absnear1.float()).flatten()
tv, ti = torch.topk(cand1, 60)
seen, counters = set(), []
for v, fl in zip(tv.tolist(), ti.tolist()):
    s, p = fl // (T_-1), fl % (T_-1)
    if s in seen or v < 0.1: continue
    seen.add(s); counters.append((int(s), int(p)))
    if len(counters) >= 3: break
# term-favourable examples under the CORRECTED absolute criterion
cand3 = (d * absnear3.float()).flatten()
tv3, ti3 = torch.topk(cand3, 60)
seen3, corrected = set(), []
for v, fl in zip(tv3.tolist(), ti3.tolist()):
    s, p = fl // (T_-1), fl % (T_-1)
    if s in seen3: continue
    seen3.add(s); corrected.append((int(s), int(p)))
    if len(corrected) >= 4: break

allpos = QUOTED + counters + corrected
seqs = sorted({s for s, _ in allpos}); smap = {s: i for i, s in enumerate(seqs)}
batch = HELD[torch.tensor(seqs, device=DEV)]
LG = {}
lg_all = []
for i in range(0, len(seqs), B0R):
    lg_all.append(fwd_terms(batch[i:i+B0R], ret_logits=True).float().cpu())
LG['base'] = torch.cat(lg_all, 0)
PIN = [INb[l][:, :576].contiguous() for l in range(NL)]
POUT = [OUTb[l][:, :288].contiguous() for l in range(NL)]
lg_all = []
for i in range(0, len(seqs), B0R):
    lg_all.append(fwd_rank(batch[i:i+B0R], PIN=PIN, POUT=POUT, MX=MX, MO=MO, ret_logits=True).float().cpu())
LG['rank_alloc_16x'] = torch.cat(lg_all, 0)
spec = build_spec(KEPT['125pct'], al0, per_term_inputs=False)
lg_all = []
for i in range(0, len(seqs), B0R):
    lg_all.append(fwd_terms(batch[i:i+B0R], mode='compress', spec=spec, ret_logits=True).float().cpu())
LG['term_scheme_16x'] = torch.cat(lg_all, 0)
del spec; torch.cuda.empty_cache()

def case_of(s, p, kind):
    bi = smap[s]; tru = int(held_np[s, p+1])
    c = {'kind': kind, 'seq': int(s), 'pos': int(p),
         'context_tail': tokzr.decode([int(t) for t in held_np[s, max(0, p-14):p+1]]),
         'true_next': tokzr.decode([tru]),
         'CE_base': round(float(base[s, p]), 3),
         'CE_rank_alloc': round(float(CE1_16[s, p]), 3),
         'CE_term_scheme': round(float(CE3_16[s, p]), 3)}
    for nm in ['base', 'term_scheme_16x', 'rank_alloc_16x']:
        pr = F.softmax(LG[nm][bi, p], -1)
        tp, ti_ = torch.topk(pr, 3)
        c[f'top3_{nm}'] = [[tokzr.decode([int(t)]), round(float(q), 4)] for q, t in zip(tp, ti_)]
        c[f'p_true_{nm}'] = round(float(pr[tru]), 5)
        c[f'rank_true_{nm}'] = int((pr > pr[tru]).sum()) + 1
    c['term_preserves_base_abs'] = bool(abs(float(CE3_16[s, p]) - float(base[s, p])) < 0.5)
    c['term_beats_base'] = bool(float(CE3_16[s, p]) < float(base[s, p]) - 0.5)
    return c

A4 = {'quoted_in_104': [case_of(s, p, 'quoted') for s, p in QUOTED],
      'counterexamples_rank_alloc_wins': [case_of(s, p, 'counterexample') for s, p in counters],
      'corrected_term_favourable': [case_of(s, p, 'corrected-selection') for s, p in corrected],
      'aggregate': {
        'frac_term_better_by_0.1': round(float((d > 0.1).float().mean()), 4),
        'frac_rank_better_by_0.1': round(float((d < -0.1).float().mean()), 4),
        'mean_delta_term_minus_rank': round(float((CE3_16 - CE1_16).mean()), 4),
        'n_positions_signed_criterion': int(signed3.sum()),
        'n_positions_absolute_criterion': int(absnear3.sum()),
        'frac_term_better_by_0.1_AND_absolute_preserving':
            round(float(((d > 0.1) & absnear3).float().mean()), 4),
        'frac_rank_better_by_0.1_AND_absolute_preserving':
            round(float(((d < -0.1) & absnear1).float().mean()), 4)},
      'selection_note': ('§104 selected term-favourable examples with the SIGNED rule '
                         '(CE_term - CE_base) < 0.5, which also admits positions where the term '
                         'scheme BEATS the base model by any margin -- those are not preservation '
                         'of a base behaviour.  The absolute rule |CE_term - CE_base| < 0.5 is '
                         'applied here as the corrected selection.')}
for c in A4['quoted_in_104'] + A4['counterexamples_rank_alloc_wins'] + A4['corrected_term_favourable']:
    print(json.dumps(c), flush=True)
print(json.dumps(A4['aggregate'], indent=1), flush=True)
res['attack4_examples'] = A4
json.dump(res, open(OUT, 'w'), indent=1)
res['meta']['peak_gpu_MiB'] = round(torch.cuda.max_memory_allocated()/2**20, 1)
json.dump(res, open(OUT, 'w'), indent=1)
print(f"peak GPU {res['meta']['peak_gpu_MiB']} MiB", flush=True)
print("QK REDTEAM TERMCOMPRESS DONE", flush=True)
