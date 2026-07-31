"""QK TERMCOMPRESS part 2b -- loss attribution for the scheme-3 defeat.

Part 2's verdict: every uniform-rank cross-layer-term scheme loses to §92 rank allocation at
every matched budget (best 16x variant +1.90 vs +0.80), and the shared-output variant ties the
per-term variant, so output-side cancellation-breaking is NOT the dominant cost.  This addendum
decomposes the best 16x variant's loss (125% profile, R=229, Ro=115, +1.9009) into components:

  (a) term dropping alone            = scheme-2 125% profile, already measured (+0.1127);
  (b) INPUT side alone: per-group projection at R=229, outputs exact (budget above 16x --
      diagnostic, not budget-matched);
  (c) input side alone at R=576: the DIRECT comparison with §92's joint input-only restriction
      at rank 576 (+0.3516, its 4x anchor).  If the group-factorized restriction at the SAME
      nominal rank is much worse than the joint one, the deficit is structural: bilinear
      products couple kept-subspace-of-A with dropped-subspace-of-B mass that a joint
      projection of the summed stream retains;
  (d) OUTPUT side alone at Ro=115 per term, inputs exact;
  (e) output side alone shared per block (Ko=115, §92 output basis), inputs exact.

Machinery identical to qk_termcompress_2.py (same cache, same compress forward).  Held
FW[448:600,:128], paired standard errors.  Updates qk_termcompress.json under 'loss_attribution'."""
import json, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_termcompress.json'

def gpu_guard(min_free=4500, tries=45, sleep=20):
    for _ in range(tries):
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']
        ).decode().split('\n')[0].strip())
        if free >= min_free:
            print(f"GPU guard: {free} MiB free -- proceeding.", flush=True); return
        print(f"GPU guard: only {free} MiB free; sleeping {sleep}s ...", flush=True)
        time.sleep(sleep)
    raise RuntimeError("GPU guard timed out")
gpu_guard()

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
HELD = FW[448:600, :128].to(DEV); B0T = 4
S_, T_ = HELD.shape; KMAX = 768
GNAMES = ['E', 'Ae', 'Ar', 'Me', 'Mr']; NG = 5
PAIRS = [(i, j) for i in range(NG) for j in range(i, NG)]

C = torch.load(f'{QK}/qk_termcompress_cache.pt', map_location='cpu', weights_only=False)
base = C['base']
TMEAN = C['TMEAN'].to(DEV); MG = C['MG'].to(DEV); MEANF = C['MEANF'].to(DEV)
GEV, GVEC = C['GEV'], C['GVEC']
RANKED, PROFILES = C['RANKED'], C['PROFILES']
CE2 = torch.load(f'{QK}/qk_termcompress_ce2.pt', map_location='cpu', weights_only=False)
TVEC = {}
res = json.load(open(OUT))
cache92 = torch.load(f'{QK}/qk_rank_alloc_cache.pt', map_location='cpu', weights_only=True)
OUTb = [b.to(DEV) for b in cache92['OUTb']]
KEPT125 = [RANKED[L][:PROFILES['125pct'][L]] for L in range(NL)]

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
WTS = [mlp_wts(li) for li in range(NL)]

@torch.no_grad()
def fwd_terms(idx, mode=None, spec=None, stats=None):
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
            if mode == 'compress':
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
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

def numrank(ev, tol=1e-6):
    if ev.sum() <= 0: return 0
    return int((ev > tol*ev[0]).sum())
GNUM = {p: min(numrank(v), KMAX) for p, v in GEV.items()}

# lazily load per-term output bases only if needed (variant d)
TEVV = C['TEV']; TVC = C['TVEC']
try:
    CE2b = torch.load(f'{QK}/qk_termcompress_ce2.pt', map_location='cpu', weights_only=False)
except Exception:
    CE2b = None

def spec_for(R_in=None, Ro_perterm=None, Ko_shared=None):
    spec = []
    for L in range(NL):
        sp = {'kept': KEPT125[L]}
        if R_in is not None:
            used = set()
            for kk in KEPT125[L]:
                i, j = PAIRS[kk]; used.add(i); used.add(j)
            sp['Pg'] = {g: GVEC[(L, g)][:, :max(1, min(R_in, GNUM[(L, g)]))].to(DEV)
                        for g in used if GVEC[(L, g)] is not None}
        if Ro_perterm is not None:
            sp['Po'] = {kk: TVC[(L, kk)][:, :max(1, Ro_perterm)].to(DEV)
                        for kk in KEPT125[L] if (L, kk) in TVC}
        if Ko_shared is not None:
            sp['PoShared'] = OUTb[L][:, :Ko_shared].contiguous()
        spec.append(sp)
    return spec

CASES = [
    ('input_only_R229', dict(R_in=229)),
    ('input_only_R576', dict(R_in=576)),
    ('output_only_perterm_Ro115', dict(Ro_perterm=115)),
    ('output_only_shared_Ko115', dict(Ko_shared=115)),
]
res['loss_attribution'] = {
    'reference': {'best_16x_variant_both_sides': res['scheme3_variants']['3b_perterm_125pct_16x']['dCE'],
                  'term_drop_only_125pct': res['scheme2']['125pct']['dCE'],
                  'scheme1_input_only_rank576_92anchor': res['scheme1']['4x']['dCE']},
    'cases': {}}
for name, kw in CASES:
    spec = spec_for(**kw)
    ce = torch.cat([fwd_terms(HELD[i:i+B0T], mode='compress', spec=spec).cpu()
                    for i in range(0, S_, B0T)], 0)
    mn, se = dstat(ce)
    res['loss_attribution']['cases'][name] = {'dCE': round(mn, 4), 'SE': round(se, 5)}
    print(f"[{name}] dCE {mn:+.4f} +- {se:.5f}", flush=True)
    del spec; torch.cuda.empty_cache()
    json.dump(res, open(OUT, 'w'), indent=1)
print("QK TERMCOMPRESS PART 2B DONE", flush=True)
