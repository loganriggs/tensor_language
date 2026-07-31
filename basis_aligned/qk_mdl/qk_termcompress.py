"""CROSS-LAYER TERM STRUCTURE vs PER-LAYER RANK ALLOCATION -- whole-model compression at matched budgets.

Logan's directive (next step after RESULTS §87/§92/§89/§103): does spending parameters on the
architecture's own cross-layer PRODUCT terms beat per-layer rank allocation for whole-model
compression?  Three schemes, all evaluated as WHOLE-MODEL substitutions (all 18 feed-forward
blocks simultaneously replaced by their compressed forms; attention untouched):

  SCHEME 1 (baseline, reproduced not redone): §92's uniform per-layer restricted cores
     (Kin,Kout) = (288,144) / (576,288) / (576,full) at 128x / 16x / 4x compression.
     Machinery + bases + means VERBATIM from qk_rank_alloc_cache.pt (the §92 artifacts).
     GATE: must reproduce §92's +1.4561 / +0.8032 / +0.3516.
  SCHEME 2 (term-sparse, exact terms): each block keeps only its top-k provenance terms
     (five input groups E/Ae/Ar/Me/Mr -> <=15 group-pair terms, §89 census), each kept term
     EXACT; dropped terms replaced by their per-position held means (the census keep-subset
     harness, now applied to ALL blocks at once).  k profile = §89 terms-to-95% scaled by a
     global multiplier (50/75/100/125%).  Ranking within a block: census keep-alone dCE for
     the top-6 energy terms, census energy order beyond.
  SCHEME 3 (term-sparse + restricted cores, the cross-layer bet): same kept terms as the
     100% profile, but each term's input group deviations are projected onto the group's
     top-r train-gram directions and each term's output deviation onto the term's own top-r_o
     train-gram directions (§85 map-restriction method, applied per term); the retained trace
     fraction f is binary-searched so the total budget matches scheme 1's budget points.

  PARAMETER ACCOUNTING (folded-tensor coefficients, the §87/§92 currency):
     full model     = NL * D * D*(D+1)/2                      = 13,771,358,208
     scheme 1       = sum_L Kout_L * Kin_L*(Kin_L+1)/2        (verbatim §92)
     scheme 2 term  = |A| * |B| * D   (cross)   or  |A|*(|A|+1)/2 * D   (diagonal),
                      |A| = numerical rank of group A's train gram (eig > 1e-6 * max;
                      secondary accounting at 99.99% trace, and a per-block cap at the
                      full block cost since all terms restrict ONE shared tensor)
     scheme 3 term  = r_A * r_B * r_o (cross)   or  r_A*(r_A+1)/2 * r_o (diagonal)
     Per-position mean tables (T x D per layer) are not counted, matching §87/§92 (MX/MO
     were not counted there either).  The shared 1/rho^2 gauge uses the true residual norm
     (a per-position scalar; §86 red-team: the gauge carries 0.13% of function).

  GATES before any claim: (a) per-layer term reconstruction ~1e-6 and group-sum ~1e-7 on the
  whole-model tracking forward; (b) whole-model ALL-TERMS substitution dCE ~ 0 (exact
  reassembly); (c) scheme-1 anchors reproduce §92; (d) base CE matches the §92 cache tensor.

  TRAIN FW[0:256] for all fitting (grams/bases), held FW[448:600] for all causal numbers,
  paired standard errors.  Batch<=6, <4GB, GPU guard.  Output: qk_termcompress.json +
  qk_termcompress_cache.pt (bases/means/per-config CE for qk_termcompress_2.py)."""
import json, math, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_termcompress.json'
CPT = f'{QK}/qk_termcompress_cache.pt'

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
B0T = 4   # term-mode batch
B0R = 6   # rank-mode / plain batch
S_, T_ = HELD.shape
STR, _ = TRAIN.shape
KMAX = 768   # stored basis width cap
print(f"bilin18 NL={NL} D={D} held {S_}x{T_} train {STR}x{T_}", flush=True)

GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']; NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]; NT = len(PAIRS)
IDX = {n: k for k, n in enumerate(PNAMES)}
FULLBLK = D * D * (D + 1) // 2
FULL = NL * FULLBLK

# ---- census-derived per-layer term rankings and the terms-to-95% profile ----
CEN = json.load(open(f'{QK}/qk_allterm_census.json'))['layers']
ACTIVE, RANKED, N95 = [], [], []
for L in range(NL):
    r = CEN[str(L)]
    er = r['energy_rank']                       # active terms, energy-sorted
    top6 = er[:6]
    ka = {n: r['configs'][f'only_{n}']['dCE'] for n in top6 if f'only_{n}' in r['configs']}
    top6s = sorted(top6, key=lambda n: ka.get(n, 1e9))
    RANKED.append([IDX[n] for n in (top6s + er[6:])])
    ACTIVE.append([IDX[n] for n in er])
    N95.append(r['terms_to_95pct'])
print("terms-to-95 profile:", N95, flush=True)

def kprof(mult):
    return [min(len(ACTIVE[L]), max(1, math.ceil(mult * N95[L]))) for L in range(NL)]
PROFILES = {'50pct': kprof(0.50), '75pct': kprof(0.75), '100pct': kprof(1.00), '125pct': kprof(1.25)}
KEPT100 = [RANKED[L][:PROFILES['100pct'][L]] for L in range(NL)]
for tag, p in PROFILES.items(): print(f"profile {tag}: {p} (sum {sum(p)})", flush=True)

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
WTS = [mlp_wts(li) for li in range(NL)]

# global per-position stores (filled by the collect pass)
TMEAN = torch.zeros(NL, NT, T_, D, device=DEV)
MG    = torch.zeros(NL, NG, T_, D, device=DEV)
MEANF = torch.zeros(NL, T_, D, device=DEV)

@torch.no_grad()
def fwd_terms(idx, mode=None, spec=None, stats=None):
    """Forward with CONTINUOUS five-group provenance tracking at every layer (recurrences
    verbatim from qk_allterm_census.py, applied at each layer in turn).
    mode: None (full model) | 'collect' (per-position term/group means + gates, base dynamics)
          | 'gram' (train grams of groups + kept-term outputs for layers in stats['chunk'])
          | 'compress' (whole-model substitution per spec: per-layer None=exact, or dict with
             kept=[term idx], Pg={group: (D,r) basis} or None, Po={term: (D,ro) basis} or None)."""
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
            if mode == 'collect':
                PL = [g @ Lw.T for g in groups]; PR = [g @ Rw.T for g in groups]
                recon = torch.zeros_like(mo)
                for kk in range(NT):
                    i, j = PAIRS[kk]
                    t = 0.5*((PL[i]*PR[j] + PL[j]*PR[i]) @ Dw.T)
                    if i != j: t = 2.0*t
                    t = t / rho2
                    stats['tsum'][li][kk] += t.sum(0)
                    recon += t
                for g in range(NG): stats['gsum'][li][g] += groups[g].sum(0)
                stats['mosum'][li] += mo.sum(0)
                gs = sum(groups)
                stats['grp_err'][li] = max(stats['grp_err'][li],
                    float(((gs - x).norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-8)).max()))
                recon = recon + bias
                stats['fro_num'][li] += float((recon - mo).pow(2).sum())
                stats['fro_den'][li] += float(mo.pow(2).sum())
                stats['maxrel'][li] = max(stats['maxrel'][li],
                    float(((recon - mo).norm(dim=-1)/mo.norm(dim=-1).clamp_min(1e-8)).max()))
                del PL, PR, recon
            elif mode == 'gram' and li in stats['chunk']:
                for g in range(NG):
                    stats['Gg'][(li, g)] += torch.einsum('btd,bte->de', groups[g], groups[g])
                used = set()
                for kk in KEPT100[li]:
                    i, j = PAIRS[kk]; used.add(i); used.add(j)
                PLd = {g: groups[g] @ Lw.T for g in used}
                PRd = {g: groups[g] @ Rw.T for g in used}
                for kk in KEPT100[li]:
                    i, j = PAIRS[kk]
                    t = 0.5*((PLd[i]*PRd[j] + PLd[j]*PRd[i]) @ Dw.T)
                    if i != j: t = 2.0*t
                    t = t / rho2
                    stats['Gt'][(li, kk)] += torch.einsum('btd,bte->de', t, t)
                del PLd, PRd
            elif mode == 'compress':
                sp = spec[li]
                if sp is not None:
                    kept = sp['kept']; Pg = sp.get('Pg'); Po = sp.get('Po')
                    used = set()
                    for kk in kept:
                        i, j = PAIRS[kk]; used.add(i); used.add(j)
                    gl = {}
                    for g in used:
                        gg = groups[g]
                        if Pg is not None and g in Pg:
                            P = Pg[g]; mg = MG[li, g]
                            gg = mg + ((gg - mg) @ P) @ P.T
                        gl[g] = gg
                    PLd = {g: gl[g] @ Lw.T for g in used}
                    PRd = {g: gl[g] @ Rw.T for g in used}
                    new = MEANF[li].unsqueeze(0).expand(B, -1, -1).clone()
                    for kk in kept:
                        i, j = PAIRS[kk]
                        t = 0.5*((PLd[i]*PRd[j] + PLd[j]*PRd[i]) @ Dw.T)
                        if i != j: t = 2.0*t
                        t = t / rho2
                        dev = t - TMEAN[li, kk]
                        if Po is not None and kk in Po:
                            P = Po[kk]; dev = (dev @ P) @ P.T
                        new = new + dev
                    mo = new
                    del PLd, PRd, gl
            del groups
        x = x + mo
        if track:
            SA = SA + aout; SM = SM + MR; MR = mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce

# ---------------- scheme-1 forward (VERBATIM qk_rank_alloc.py, per-layer None allowed) ----
@torch.no_grad()
def fwd_rank(idx, PIN=None, POUT=None, MX=None, MO=None):
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
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce

# ---------------- base CE + gate vs the §92 cache ----------------
print("BASE: full-model cross-entropy ...", flush=True)
base = torch.cat([fwd_terms(HELD[i:i+B0R]).cpu() for i in range(0, S_, B0R)], 0)
cache92 = torch.load(f'{QK}/qk_rank_alloc_cache.pt', map_location='cpu', weights_only=True)
base_gate = float((base - cache92['base']).abs().max())
print(f"base CE {float(base.mean()):.4f}  (cache92 {float(cache92['base'].mean()):.4f}; "
      f"max abs diff {base_gate:.2e})", flush=True)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

res = {'meta': {
    'model': 'bilin18', 'held': 'FW[448:600,:128]', 'train': 'FW[0:256,:128]',
    'batch_terms': B0T, 'batch_rank': B0R, 'base_ce': round(float(base.mean()), 4),
    'base_vs_cache92_maxabs': base_gate,
    'profiles': {k: v for k, v in PROFILES.items()}, 'terms_to_95_census': N95,
    'full_params': int(FULL), 'full_block': int(FULLBLK),
    'accounting': 'folded-tensor coefficients; scheme1 Kout*Kin*(Kin+1)/2 per block; '
                  'scheme2 |A|x|B|xD per cross term (|A|(|A|+1)/2 x D diagonal), |A| = numerical '
                  'rank of group train gram (eig > 1e-6*max; secondary 99.99% trace; per-block cap '
                  'at full block cost); scheme3 rA x rB x ro per term at shared trace fraction f. '
                  'Per-position mean tables and the scalar gauge not counted (matches §87/§92).'},
    'gates': {}, 'scheme1': {}, 'scheme2': {}, 'scheme3': {}}

# ---------------- collect pass: per-position means + reconstruction gates ----------------
print("COLLECT: per-position term/group means + gates (all 18 layers, one pass) ...", flush=True)
st = {'tsum': [[torch.zeros(T_, D, device=DEV) for _ in range(NT)] for _ in range(NL)],
      'gsum': [[torch.zeros(T_, D, device=DEV) for _ in range(NG)] for _ in range(NL)],
      'mosum': [torch.zeros(T_, D, device=DEV) for _ in range(NL)],
      'grp_err': [0.0]*NL, 'maxrel': [0.0]*NL, 'fro_num': [0.0]*NL, 'fro_den': [0.0]*NL}
t0 = time.time()
for i in range(0, S_, B0T): fwd_terms(HELD[i:i+B0T], mode='collect', stats=st)
gate_rec = []
for L in range(NL):
    for kk in range(NT): TMEAN[L, kk] = st['tsum'][L][kk] / S_
    for g in range(NG):  MG[L, g] = st['gsum'][L][g] / S_
    MEANF[L] = TMEAN[L].sum(0) + WTS[L][3]
    fro = (st['fro_num'][L]/max(st['fro_den'][L], 1e-30))**0.5
    mc = float((MEANF[L] - st['mosum'][L]/S_).norm()/(st['mosum'][L]/S_).norm())
    gate_rec.append({'layer': L, 'recon_rel_err_global': fro, 'recon_rel_err_max_pos': st['maxrel'][L],
                     'group_sum_rel_err_max': st['grp_err'][L], 'mean_consistency': mc})
    print(f"  L{L:2d} recon {fro:.2e} maxpos {st['maxrel'][L]:.2e} grpsum {st['grp_err'][L]:.2e} "
          f"meancons {mc:.2e}", flush=True)
assert all(g['recon_rel_err_global'] < 1e-4 and g['group_sum_rel_err_max'] < 1e-4 for g in gate_rec), \
    "reconstruction gate FAILED"
res['gates']['per_layer_reconstruction'] = gate_rec
del st; torch.cuda.empty_cache()
print(f"collect done ({time.time()-t0:.0f}s)", flush=True)

# ---------------- GATE A: whole-model exact reassembly (all active terms kept) ----------------
print("GATE A: whole-model ALL-TERMS substitution (must be ~0) ...", flush=True)
spec_all = [{'kept': ACTIVE[L]} for L in range(NL)]
ce = torch.cat([fwd_terms(HELD[i:i+B0T], mode='compress', spec=spec_all).cpu()
                for i in range(0, S_, B0T)], 0)
mn, se = dstat(ce)
res['gates']['all_terms_reassembly'] = {'dCE': mn, 'SE': se}
print(f"  ALL-TERMS whole-model dCE {mn:+.6f} +- {se:.6f}", flush=True)
assert abs(mn) < 1e-3, "exact reassembly gate FAILED"

# ---------------- GATE B / SCHEME 1: reproduce the §92 uniform anchors ----------------
print("SCHEME 1: §92 uniform anchors from the cache ...", flush=True)
INb = [b.to(DEV) for b in cache92['INb']]; OUTb = [b.to(DEV) for b in cache92['OUTb']]
MX = [t.to(DEV) for t in cache92['MX']];   MO = [t.to(DEV) for t in cache92['MO']]
REF92 = {'128x': 1.4561, '16x': 0.8032, '4x': 0.3516}
UNIFORM = {'128x': (288, 144), '16x': (576, 288), '4x': (576, None)}
CE_STORE = {}
for tag, (Kin, Kout) in UNIFORM.items():
    PIN = [INb[l][:, :Kin].contiguous() for l in range(NL)]
    POUT = [OUTb[l][:, :Kout].contiguous() for l in range(NL)] if Kout else None
    ce = torch.cat([fwd_rank(HELD[i:i+B0R], PIN=PIN, POUT=POUT, MX=MX, MO=MO).cpu()
                    for i in range(0, S_, B0R)], 0)
    mn, se = dstat(ce); CE_STORE[f's1_{tag}'] = ce
    cp = NL * (Kout if Kout else D) * Kin*(Kin+1)//2
    res['scheme1'][tag] = {'Kin': Kin, 'Kout': Kout, 'dCE': round(mn, 4), 'SE': round(se, 5),
                           'core_params': int(cp), 'compression_x': round(FULL/cp, 1),
                           'ref_92': REF92[tag], 'anchor_ok': bool(abs(mn - REF92[tag]) < 0.01)}
    print(f"  [{tag}] uniform ({Kin},{Kout}): dCE {mn:+.4f} +- {se:.5f}  "
          f"(ref §92 {REF92[tag]:+.4f}, params {cp/1e6:.0f}M)", flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
assert all(res['scheme1'][t]['anchor_ok'] for t in REF92), "scheme-1 anchor gate FAILED"
del INb, OUTb, PIN, POUT; torch.cuda.empty_cache()

# ---------------- TRAIN gram passes (chunked over layers) ----------------
print("TRAIN grams: groups + kept-term outputs, 5 layer-chunks ...", flush=True)
GEV, GVEC, TEV, TVEC = {}, {}, {}, {}
CHUNKS = [list(range(0, 4)), list(range(4, 8)), list(range(8, 12)),
          list(range(12, 16)), list(range(16, 18))]
for ch in CHUNKS:
    tc = time.time()
    stg = {'chunk': set(ch),
           'Gg': {(L, g): torch.zeros(D, D, device=DEV) for L in ch for g in range(NG)},
           'Gt': {(L, kk): torch.zeros(D, D, device=DEV) for L in ch for kk in KEPT100[L]}}
    for i in range(0, STR, B0T): fwd_terms(TRAIN[i:i+B0T], mode='gram', stats=stg)
    for (L, g), G in stg['Gg'].items():
        if float(G.abs().max()) < 1e-20:
            GEV[(L, g)] = np.zeros(D, dtype=np.float64); GVEC[(L, g)] = None; continue
        w, vec = torch.linalg.eigh(G)
        GEV[(L, g)] = w.flip(0).clamp_min(0).double().cpu().numpy()
        GVEC[(L, g)] = vec.flip(1)[:, :KMAX].contiguous().cpu()
    for (L, kk), G in stg['Gt'].items():
        w, vec = torch.linalg.eigh(G)
        TEV[(L, kk)] = w.flip(0).clamp_min(0).double().cpu().numpy()
        TVEC[(L, kk)] = vec.flip(1)[:, :KMAX].contiguous().cpu()
    del stg; torch.cuda.empty_cache()
    print(f"  chunk {ch} done ({time.time()-tc:.0f}s)", flush=True)

# ---------------- budget machinery ----------------
def numrank(ev, tol=1e-6):
    if ev.sum() <= 0: return 0
    return int((ev > tol*ev[0]).sum())
def tracerank(ev, f):
    s = ev.sum()
    if s <= 0: return 0
    cum = np.cumsum(ev)/s
    return int(min(np.searchsorted(cum, f) + 1, D))
GDIM_NUM = {(L, g): numrank(GEV[(L, g)]) for L in range(NL) for g in range(NG)}
GDIM_9999 = {(L, g): tracerank(GEV[(L, g)], 0.9999) for L in range(NL) for g in range(NG)}

def s2_budget(prof, gdim):
    per = []
    for L in range(NL):
        c = 0
        for kk in RANKED[L][:prof[L]]:
            i, j = PAIRS[kk]
            ri, rj = gdim[(L, i)], gdim[(L, j)]
            c += ri*(ri+1)//2 * D if i == j else ri*rj*D
        per.append(c)
    return per

def s3_config(f):
    """Per-layer spec ranks + budget at shared trace fraction f."""
    budget = 0; ranks = []
    for L in range(NL):
        gr = {}; term_ro = {}
        for kk in KEPT100[L]:
            i, j = PAIRS[kk]
            for g in (i, j):
                if g not in gr: gr[g] = min(tracerank(GEV[(L, g)], f), KMAX)
            ro = min(tracerank(TEV[(L, kk)], f), KMAX)
            term_ro[kk] = ro
            ri, rj = gr[i], gr[j]
            budget += ri*(ri+1)//2 * ro if i == j else ri*rj*ro
        ranks.append((gr, term_ro))
    return budget, ranks

def fit_f(B):
    lo, hi = 1e-6, 1.0 - 1e-9; best = None
    for _ in range(60):
        mid = 0.5*(lo + hi)
        b, rk = s3_config(mid)
        if b <= B: best = (mid, b, rk); lo = mid
        else: hi = mid
    if best is None: best = (lo,) + s3_config(lo)
    return best

def build_s3_spec(ranks, layers=None):
    spec = []
    for L in range(NL):
        if layers is not None and L not in layers:
            spec.append(None); continue
        gr, term_ro = ranks[L]
        Pg = {g: GVEC[(L, g)][:, :max(r, 1)].to(DEV) for g, r in gr.items() if GVEC[(L, g)] is not None}
        Po = {kk: TVEC[(L, kk)][:, :max(ro, 1)].to(DEV) for kk, ro in term_ro.items()}
        spec.append({'kept': KEPT100[L], 'Pg': Pg, 'Po': Po})
    return spec

res['group_dims'] = {'numrank_1e-6': {f'L{L}': {GNAMES[g]: GDIM_NUM[(L, g)] for g in range(NG)}
                                      for L in range(NL)},
                     'rank_9999trace': {f'L{L}': {GNAMES[g]: GDIM_9999[(L, g)] for g in range(NG)}
                                        for L in range(NL)}}

# ---------------- SCHEME 2: term-sparse, exact terms ----------------
print("SCHEME 2: term-sparse profiles ...", flush=True)
for tag, prof in PROFILES.items():
    spec = [{'kept': RANKED[L][:prof[L]]} for L in range(NL)]
    ce = torch.cat([fwd_terms(HELD[i:i+B0T], mode='compress', spec=spec).cpu()
                    for i in range(0, S_, B0T)], 0)
    mn, se = dstat(ce); CE_STORE[f's2_{tag}'] = ce
    per_num = s2_budget(prof, GDIM_NUM); per_99 = s2_budget(prof, GDIM_9999)
    bn, b9 = sum(per_num), sum(per_99)
    bcap = sum(min(c, FULLBLK) for c in per_num)
    res['scheme2'][tag] = {'k_profile': prof, 'n_terms': sum(prof),
        'dCE': round(mn, 4), 'SE': round(se, 5),
        'budget_numrank': int(bn), 'budget_9999trace': int(b9), 'budget_capped': int(bcap),
        'compression_x_numrank': round(FULL/bn, 2), 'compression_x_capped': round(FULL/bcap, 2)}
    print(f"  [{tag}] {sum(prof)} terms: dCE {mn:+.4f} +- {se:.5f}  budget numrank {bn/1e9:.2f}B "
          f"(capped {bcap/1e9:.2f}B, 99.99% {b9/1e9:.2f}B)", flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)

# ---------------- SCHEME 3: term-sparse + restricted cores at matched budgets ----------------
print("SCHEME 3: restricted terms at the §92 budgets ...", flush=True)
BUDGETS = {'128x': NL*144*288*289//2, '16x': NL*288*576*577//2, '4x': NL*1152*576*577//2}
for tag, B in BUDGETS.items():
    f, b, ranks = fit_f(B)
    spec = build_s3_spec(ranks)
    ce = torch.cat([fwd_terms(HELD[i:i+B0T], mode='compress', spec=spec).cpu()
                    for i in range(0, S_, B0T)], 0)
    mn, se = dstat(ce); CE_STORE[f's3_{tag}'] = ce
    exr = ranks[2][0]; exo = ranks[2][1]   # layer-2 worked example
    ex_k = KEPT100[2][0]; exi, exj = PAIRS[ex_k]
    ex_cost = (exr[exi]*(exr[exi]+1)//2*exo[ex_k] if exi == exj
               else exr[exi]*exr[exj]*exo[ex_k])
    res['scheme3'][tag] = {'f': f, 'budget_target': int(B), 'budget_used': int(b),
        'dCE': round(mn, 4), 'SE': round(se, 5), 'compression_x': round(FULL/b, 1),
        'ranks_per_layer': [{'groups': {GNAMES[g]: r for g, r in gr.items()},
                             'term_out': {PNAMES[kk]: ro for kk, ro in tro.items()}}
                            for gr, tro in ranks],
        'worked_example': {'layer': 2, 'term': PNAMES[ex_k],
            'input_ranks': [exr[exi], exr[exj]], 'out_rank': exo[ex_k], 'cost': int(ex_cost),
            'formula': 'rA*(rA+1)/2*ro diagonal else rA*rB*ro'}}
    print(f"  [{tag}] f={f:.5f} budget {b/1e6:.0f}M/{B/1e6:.0f}M: dCE {mn:+.4f} +- {se:.5f}", flush=True)
    del spec; torch.cuda.empty_cache()
    json.dump(res, open(OUT, 'w'), indent=1)

# ---------------- persist cache for script 2 (regions + examples) ----------------
print("saving cache for qk_termcompress_2.py ...", flush=True)
torch.save({'base': base, 'CE': CE_STORE,
            'TMEAN': TMEAN.cpu(), 'MG': MG.cpu(), 'MEANF': MEANF.cpu(),
            'GEV': GEV, 'GVEC': GVEC, 'TEV': TEV, 'TVEC': TVEC,
            'RANKED': RANKED, 'ACTIVE': ACTIVE, 'PROFILES': PROFILES,
            'KEPT100': KEPT100, 'GDIM_NUM': GDIM_NUM, 'GDIM_9999': GDIM_9999},
           CPT)
json.dump(res, open(OUT, 'w'), indent=1)
print("QK TERMCOMPRESS PART 1 DONE", flush=True)
