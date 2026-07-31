"""QK TERMCOMPRESS part 2 -- scheme-3 allocation variants, the per-region diagnostic slice, and
the concrete divergence examples.

Part 1 found the AS-SPECIFIED scheme 3 (shared trace-fraction rule) loses catastrophically and
non-monotonically (+3.46/+3.51/+3.03 at 128x/16x/4x): the trace rule assigns rank 1 to
functionally live groups (Ae everywhere; the readout's term outputs get ranks 1-3) -- exactly
the §92 pathology (gram-trace concentration anti-correlates with functional need).  Part 1 also
showed the 100% profile's own term-dropping floor (+0.878) exceeds scheme 1 at 16x/4x, so the
cross-layer bet needs the richer 125% profile (floor +0.113) there.  This script gives the bet
its fair, §92-consistent form:

  (0) extra train gram pass for the 125%-profile terms missing from part 1's cache;
  (V) SCHEME-3 VARIANTS at matched budgets 128x/16x/4x:
      3b PER-TERM UNIFORM: every used group projected to the same input rank R (capped at its
         numerical rank), every kept term's output deviation to Ro = ceil(R/2) of its own
         train-gram directions; R binary-searched to the budget.  Profiles 100% and 125%.
      3c SHARED-OUTPUT UNIFORM: same input side, but the block's kept-term-deviation SUM is
         projected once onto the §92 output basis (OUTb, rank Ko = ceil(R/2)) -- preserving
         cross-term cancellation inside a shared subspace.  Direct test of the hypothesis that
         PER-TERM output truncation is what breaks the §89/§91 cancelling term geometry.
         Budget: term core = rA*rB*Ko (diagonal rA*(rA+1)/2*Ko), §92-style (bases not counted).
  (A) REGIONAL BREAKDOWN at the 16x matched budget (schemes 1 and 3-best exact; scheme 2 at its
      100% profile, own larger budget): each scheme applied to ONLY early 0-4 / distributed
      5-11 / readout 12-17, others exact.  §102's prediction: consumption-shaped tail -> scheme
      3 loses least in the distributed region; unstructured tail -> most.
  (B) EXAMPLES: held positions with the largest per-position cross-entropy divergence between
      scheme 1 at 16x and the best scheme-3 variant at 16x (in the better scheme's favor),
      decoded, with top-3 next-token predictions under base / scheme 1 / scheme 3.

Machinery VERBATIM from qk_termcompress.py (forwards copied unchanged; compress mode extended
with an optional shared output projector).  Held FW[448:600,:128], TRAIN FW[0:256] for grams,
paired standard errors, batch<=6, GPU guard.  Updates qk_termcompress.json in place."""
import json, math, subprocess, sys, time
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
HELD = FW[448:600, :128].to(DEV); TRAIN = FW[0:256, :128].to(DEV)
STR = TRAIN.shape[0]
B0T = 4; B0R = 6
S_, T_ = HELD.shape
KMAX = 768
GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']; NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]
PNAMES = [f'{GNAMES[i]}x{GNAMES[j]}' for (i, j) in PAIRS]; NT = len(PAIRS)
FULLBLK = D * D * (D + 1) // 2
FULL = NL * FULLBLK

print("loading part-1 cache ...", flush=True)
C = torch.load(CPT, map_location='cpu', weights_only=False)
base = C['base']; CE_STORE = C['CE']
TMEAN = C['TMEAN'].to(DEV); MG = C['MG'].to(DEV); MEANF = C['MEANF'].to(DEV)
GEV, GVEC, TEV, TVEC = C['GEV'], C['GVEC'], C['TEV'], C['TVEC']
RANKED, KEPT100, PROFILES = C['RANKED'], C['KEPT100'], C['PROFILES']
res = json.load(open(OUT))

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
WTS = [mlp_wts(li) for li in range(NL)]

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
                    kept = sp['kept']; Pg = sp.get('Pg'); Po = sp.get('Po')
                    PoS = sp.get('PoShared')
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
                    devsum = torch.zeros(B, T, D, device=DEV)
                    for kk in kept:
                        i, j = PAIRS[kk]
                        t = 0.5*((PLd[i]*PRd[j] + PLd[j]*PRd[i]) @ Dw.T)
                        if i != j: t = 2.0*t
                        t = t / rho2
                        dev = t - TMEAN[li, kk]
                        if Po is not None and kk in Po:
                            P = Po[kk]; dev = (dev @ P) @ P.T
                        devsum = devsum + dev
                    if PoS is not None:
                        devsum = (devsum @ PoS) @ PoS.T
                    mo = MEANF[li].unsqueeze(0) + devsum
                    del PLd, PRd, gl
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

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

# ---------------- (0) extra term output grams for the 125% profile ----------------
KEPT125 = [RANKED[L][:PROFILES['125pct'][L]] for L in range(NL)]
missing = [(L, kk) for L in range(NL) for kk in KEPT125[L] if (L, kk) not in TEV]
print(f"extra term grams needed for the 125% profile: {len(missing)}", flush=True)
if missing:
    t0 = time.time()
    stg = {'pairs': set(missing), 'Gt': {p: torch.zeros(D, D, device=DEV) for p in missing}}
    for i in range(0, STR, B0T): fwd_terms(TRAIN[i:i+B0T], mode='gram', stats=stg)
    for p, G in stg['Gt'].items():
        w, vec = torch.linalg.eigh(G)
        TEV[p] = w.flip(0).clamp_min(0).double().cpu().numpy()
        TVEC[p] = vec.flip(1)[:, :KMAX].contiguous().cpu()
    del stg; torch.cuda.empty_cache()
    print(f"extra grams done ({time.time()-t0:.0f}s)", flush=True)

cache92 = torch.load(f'{QK}/qk_rank_alloc_cache.pt', map_location='cpu', weights_only=True)
INb = [b.to(DEV) for b in cache92['INb']]; OUTb = [b.to(DEV) for b in cache92['OUTb']]
MX = [t.to(DEV) for t in cache92['MX']];   MO = [t.to(DEV) for t in cache92['MO']]

# ---------------- (V) scheme-3 variants: uniform ranks, per-term vs shared output ----------
def numrank(ev, tol=1e-6):
    if ev.sum() <= 0: return 0
    return int((ev > tol*ev[0]).sum())
GNUM = {p: min(numrank(v), KMAX) for p, v in GEV.items()}
TNUM = {p: min(numrank(v), KMAX) for p, v in TEV.items()}

def uni_ranks(kept, R, shared):
    Ro = max(1, (R + 1)//2)
    budget = 0; ranks = []
    for L in range(NL):
        gr = {}; tro = {}
        for kk in kept[L]:
            i, j = PAIRS[kk]
            for g in (i, j):
                if g not in gr: gr[g] = max(1, min(R, GNUM[(L, g)]))
            ro = Ro if shared else max(1, min(Ro, TNUM[(L, kk)]))
            tro[kk] = ro
            ri, rj = gr[i], gr[j]
            budget += ri*(ri+1)//2 * ro if i == j else ri*rj*ro
        ranks.append((gr, tro))
    return budget, ranks

def fit_R(kept, B, shared):
    lo, hi = 1, KMAX; best = None
    while lo <= hi:
        mid = (lo + hi)//2
        b, rk = uni_ranks(kept, mid, shared)
        if b <= B: best = (mid, b, rk); lo = mid + 1
        else: hi = mid - 1
    return best

def build_spec_uni(kept, ranks, shared, Ko, layers=None):
    spec = []
    for L in range(NL):
        if layers is not None and L not in layers:
            spec.append(None); continue
        gr, tro = ranks[L]
        Pg = {g: GVEC[(L, g)][:, :r].to(DEV) for g, r in gr.items() if GVEC[(L, g)] is not None}
        sp = {'kept': kept[L], 'Pg': Pg}
        if shared:
            sp['PoShared'] = OUTb[L][:, :Ko].contiguous()
        else:
            sp['Po'] = {kk: TVEC[(L, kk)][:, :ro].to(DEV) for kk, ro in tro.items()}
        spec.append(sp)
    return spec

BUDGETS = {'128x': NL*144*288*289//2, '16x': NL*288*576*577//2, '4x': NL*1152*576*577//2}
res.setdefault('scheme3_variants', {})
res['scheme3_variants']['note'] = (
    'Part-1 scheme 3 (shared trace-fraction rule) is kept above as the as-specified result; '
    'its rule assigns rank 1 to functionally live groups (the §92 trace-concentration '
    'pathology). Variants here use UNIFORM ranks: input rank R per used group (capped at the '
    'group numerical rank), output Ro=ceil(R/2) -- per-term own basis (3b) or the §92 shared '
    'block output basis applied to the kept-term-deviation sum (3c); R fit to the budget.')
best16 = {'dCE': float('inf'), 'key': None}
for shared, vtag in [(False, '3b_perterm'), (True, '3c_sharedout')]:
    for ptag in ['100pct', '125pct']:
        kept = [RANKED[L][:PROFILES[ptag][L]] for L in range(NL)]
        for btag, B in BUDGETS.items():
            fit = fit_R(kept, B, shared)
            if fit is None:
                print(f"[{vtag} {ptag} {btag}] unfittable", flush=True); continue
            R, b, ranks = fit
            Ko = max(1, (R + 1)//2)
            spec = build_spec_uni(kept, ranks, shared, Ko)
            ce = torch.cat([fwd_terms(HELD[i:i+B0T], mode='compress', spec=spec).cpu()
                            for i in range(0, S_, B0T)], 0)
            mn, se = dstat(ce)
            key = f'{vtag}_{ptag}_{btag}'
            CE_STORE[key] = ce
            res['scheme3_variants'][key] = {
                'R_in': R, 'R_out': Ko, 'n_terms': sum(len(k) for k in kept),
                'budget_target': int(B), 'budget_used': int(b),
                'compression_x': round(FULL/b, 1), 'dCE': round(mn, 4), 'SE': round(se, 5)}
            print(f"[{key}] R={R} Ro={Ko} budget {b/1e6:.0f}M/{B/1e6:.0f}M: "
                  f"dCE {mn:+.4f} +- {se:.5f}", flush=True)
            if btag == '16x' and mn < best16['dCE']:
                best16 = {'dCE': mn, 'key': key, 'shared': shared, 'ptag': ptag,
                          'kept': kept, 'ranks': ranks, 'Ko': Ko, 'R': R}
            del spec; torch.cuda.empty_cache()
            json.dump(res, open(OUT, 'w'), indent=1)
res['scheme3_variants']['best_16x'] = best16['key']
print(f"best 16x scheme-3 variant: {best16['key']} dCE {best16['dCE']:+.4f}", flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
torch.save({'base': base, 'CE': CE_STORE}, f'{QK}/qk_termcompress_ce2.pt')

# ---------------- (A) regional breakdown at the 16x matched budget ----------------
REGIONS = {'early_0_4': list(range(0, 5)), 'distributed_5_11': list(range(5, 12)),
           'readout_12_17': list(range(12, 18))}
res['regional'] = {'matched_budget': '16x (861,456,384 core parameters) for schemes 1 and 3-best; '
                   'scheme 2 at its 100% terms-to-95 profile (own, larger budget)',
                   'scheme3_variant_used': best16['key'],
                   'whole_model': {'scheme1': res['scheme1']['16x']['dCE'],
                                   'scheme2_100pct': res['scheme2']['100pct']['dCE'],
                                   'scheme3': round(best16['dCE'], 4)},
                   'regions': {}}
for rtag, lay in REGIONS.items():
    rrec = {}
    # scheme 1: uniform (576,288) on region layers only
    PIN = [INb[l][:, :576].contiguous() if l in lay else None for l in range(NL)]
    POUT = [OUTb[l][:, :288].contiguous() if l in lay else None for l in range(NL)]
    ce = torch.cat([fwd_rank(HELD[i:i+B0R], PIN=PIN, POUT=POUT, MX=MX, MO=MO).cpu()
                    for i in range(0, S_, B0R)], 0)
    mn, se = dstat(ce); rrec['scheme1'] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"[{rtag}] scheme1 region-only: {mn:+.4f} +- {se:.5f}", flush=True)
    # scheme 2: 100% profile on region layers only
    prof = PROFILES['100pct']
    spec = [{'kept': RANKED[L][:prof[L]]} if L in lay else None for L in range(NL)]
    ce = torch.cat([fwd_terms(HELD[i:i+B0T], mode='compress', spec=spec).cpu()
                    for i in range(0, S_, B0T)], 0)
    mn, se = dstat(ce); rrec['scheme2_100pct'] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"[{rtag}] scheme2 region-only: {mn:+.4f} +- {se:.5f}", flush=True)
    # scheme 3 (best 16x variant) on region layers only
    spec = build_spec_uni(best16['kept'], best16['ranks'], best16['shared'], best16['Ko'],
                          layers=set(lay))
    ce = torch.cat([fwd_terms(HELD[i:i+B0T], mode='compress', spec=spec).cpu()
                    for i in range(0, S_, B0T)], 0)
    mn, se = dstat(ce); rrec['scheme3'] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"[{rtag}] scheme3 region-only: {mn:+.4f} +- {se:.5f}", flush=True)
    del spec; torch.cuda.empty_cache()
    res['regional']['regions'][rtag] = rrec
    json.dump(res, open(OUT, 'w'), indent=1)

for sch in ['scheme1', 'scheme2_100pct', 'scheme3']:
    ssum = sum(res['regional']['regions'][r][sch]['dCE'] for r in REGIONS)
    res['regional'][f'{sch}_region_sum_vs_joint'] = [round(ssum, 4), res['regional']['whole_model'][sch]]
json.dump(res, open(OUT, 'w'), indent=1)

# ---------------- (B) concrete divergence examples at 16x ----------------
print("EXAMPLES: scheme 3 preserves, scheme 1 loses ...", flush=True)
from transformers import AutoTokenizer
tokzr = AutoTokenizer.from_pretrained('gpt2')
ce1 = CE_STORE['s1_16x']; ce3 = CE_STORE[best16['key']]
delta = (ce1 - ce3)                       # positive: scheme 3 better at this position
near3 = (ce3 - base) < 0.5                # scheme 3 preserves the prediction
cand = (delta * near3.float()).flatten()
topv, topi = torch.topk(cand, 40)
held_np = HELD.cpu().numpy()
seen, examples = set(), []
for v, fl in zip(topv.tolist(), topi.tolist()):
    s, p = fl // (T_-1), fl % (T_-1)      # predicting token p+1
    if s in seen: continue
    seen.add(s); examples.append((s, p, v))
    if len(examples) >= 5: break
seqs = sorted(set(s for s, _, _ in examples))
batch = HELD[torch.tensor(seqs, device=DEV)]
smap = {s: i for i, s in enumerate(seqs)}
lg_base = fwd_terms(batch, ret_logits=True).float()
PIN = [INb[l][:, :576].contiguous() for l in range(NL)]
POUT = [OUTb[l][:, :288].contiguous() for l in range(NL)]
lg_s1 = fwd_rank(batch, PIN=PIN, POUT=POUT, MX=MX, MO=MO, ret_logits=True).float()
spec = build_spec_uni(best16['kept'], best16['ranks'], best16['shared'], best16['Ko'])
lg_s3 = fwd_terms(batch, mode='compress', spec=spec, ret_logits=True).float()
res['examples'] = {'selection': f'largest per-position (scheme1_CE - scheme3_CE) at 16x, '
                                f'scheme3 = {best16["key"]}, with scheme3_CE within 0.5 nats '
                                f'of base; one per sequence', 'cases': []}
for s, p, v in examples:
    bi = smap[s]
    ctx = tokzr.decode([int(t) for t in held_np[s, max(0, p-14):p+1]])
    tru = tokzr.decode([int(held_np[s, p+1])])
    case = {'seq': int(s), 'pos': int(p), 'context_tail': ctx, 'true_next': tru,
            'CE_base': round(float(base[s, p]), 3), 'CE_scheme1': round(float(ce1[s, p]), 3),
            'CE_scheme3': round(float(ce3[s, p]), 3), 'delta': round(v, 3)}
    for nm, lg in [('base', lg_base), ('scheme1', lg_s1), ('scheme3', lg_s3)]:
        pr = F.softmax(lg[bi, p], -1)
        tp, ti = torch.topk(pr, 3)
        case[f'top3_{nm}'] = [[tokzr.decode([int(t)]), round(float(q), 3)]
                              for q, t in zip(tp, ti)]
        case[f'p_true_{nm}'] = round(float(pr[int(held_np[s, p+1])]), 4)
    res['examples']['cases'].append(case)
    print(json.dumps(case), flush=True)

# aggregate: where does scheme 3 beat scheme 1 position-wise?
d = (ce1 - ce3).flatten()
res['examples']['aggregate'] = {
    'frac_positions_scheme3_better': round(float((d > 0).float().mean()), 4),
    'frac_scheme3_better_by_0.1': round(float((d > 0.1).float().mean()), 4),
    'frac_scheme1_better_by_0.1': round(float((d < -0.1).float().mean()), 4),
    'mean_delta': round(float(d.mean()), 4)}
json.dump(res, open(OUT, 'w'), indent=1)
print("QK TERMCOMPRESS PART 2 DONE", flush=True)
