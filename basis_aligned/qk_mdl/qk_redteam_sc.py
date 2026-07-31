"""RED TEAM of RESULTS section 105 (shared cores across layers REFUTED).

Four attacks on the fairness of the negative -- was shared structure missed BY CONSTRUCTION?

  1. GAUGE   -- the raw Frobenius cosine between two blocks' folded tensors is not invariant
                to a per-block change of residual coordinates.  For a pair (l, m) fit the best
                ORTHOGONAL input rotation R and output rotation S maximizing
                    <T_m, (S (x) R (x) R) T_l>  /  (||T_l|| ||T_m||)
                (both input slots share R -- both read the same stream; S acts on the output
                slot).  Method: alternating maximization -- exact orthogonal Procrustes for S
                given R (closed form: S = U V^T of the D x D output-slice correlation, value =
                nuclear norm), projected gradient ascent with polar retraction and backtracking
                line search for R given S.  Analytic gradients, chunked over the hidden feature
                index; nothing of size D^3 is ever materialized.  Controls: two far pairs and a
                random-factor tensor (the chance level of the fit).  Plus a hard UPPER BOUND
                over the FULL orthogonal group on every mode (von Neumann trace inequality on
                the output-mode unfolding), which no gauge fit can exceed.
                Compression relevance: a gauge match would be a real scheme -- one core plus
                (R, S) costs 2 D^2 = 2.65M coefficients vs 765M for a block.

  2. REFIT   -- causally refit the mixing coefficients at the mild budgets k = 12 and k = 16
                (where section 105's z is smallest, 10 and 5.1).  Atoms fixed (the pooled-metric
                principal atoms C_i = sum_m U[m,i] T_m); free A in R^{18 x k}, W = A U_k^T, same
                budget (18k mixing scalars).  Minimize TRAIN cross-entropy by Adam with a
                validation split of TRAIN for early stopping; evaluate the best iterate on held.

  3. CENTER  -- mean-centered atoms: share the deviation from the mean tensor.  Centered Gram
                = J G J, J = I - 11^T/18.  Reconstruction T^_l = Tbar + sum_i b_li C^c_i is
                still a linear mixture of the 18 blocks, so the same substitution forward
                applies; budget counts the mean as one free atom (k centered atoms + mean =
                k+1 blocks, compared against the UNCENTERED k+1 cell).

  4. OUTW    -- output-slot weighting.  Section 105's pooled metric weighted input slots only.
                Two shared output metrics, both pooled over blocks so that the whole thing is a
                single change of coordinates (hence the atoms stay legitimate):
                  (a) activation:  Sout = (mean_l E_train[b_l b_l^T])^{1/2}, b_l the block's
                      bilinear output;
                  (b) consumption: Sout = (mean_l E_train[g_l g_l^T])^{1/2}, g_l = d(loss)/d(b_l)
                      -- a Fisher-style metric on exactly the directions the rest of the model
                      reads.
                Census recomputed with input AND output weighting, INCLUDING a norm-free census
                (spectrum of the unit-diagonal cosine matrix) that cannot be faked by norm
                concentration, plus causal whole-model substitution with the new atoms.

  TRAIN FW[0:256] for all fitting, held FW[448:600] for all causal numbers, paired standard
  errors.  Forward verbatim from qk_sharedcore.py.  Reuses qk_sharedcore_cache.pt.
  Output: qk_redteam_sc.json.  Usage: python qk_redteam_sc.py <stage> [<stage> ...]
"""
import json, math, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = os.environ.get('RT_OUT', f'{QK}/qk_redteam_sc.json')
CPT = f'{QK}/qk_sharedcore_cache.pt'
STAGES = sys.argv[1:]

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
FDIM = cfg['expansion_factor'] * D
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV)
B0M = 4; B0 = 6
S_, T_ = HELD.shape; STR = TRAIN.shape[0]
FULLBLK = D * D * (D + 1) // 2; FULL = NL * FULLBLK

for _p in m.parameters(): _p.requires_grad_(False)

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
WTS = [mlp_wts(li) for li in range(NL)]
BIAS = [WTS[li][3] for li in range(NL)]

RCPT = f'{QK}/qk_redteam_sc_cache.pt'
cache = torch.load(CPT, map_location='cpu', weights_only=True)
rcache = torch.load(RCPT, map_location='cpu', weights_only=True) if os.path.exists(RCPT) else {}
G = cache['G'].numpy(); Gp = cache['G_pool'].numpy()
UG = cache['UG'].numpy(); UP = cache['UP'].numpy()
Sig = cache['Sig']; base = cache['base']

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res.setdefault('meta', {
    'model': 'bilin18', 'held': 'FW[448:600,:128]', 'train': 'FW[0:256,:128]',
    'target': 'RESULTS section 105', 'source_json': 'qk_sharedcore.json'})
res.setdefault('gates', {})
NOWRITE = 'smoke' in STAGES
def dump():
    if NOWRITE: return
    json.dump(res, open(OUT, 'w'), indent=1)

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

# =====================================================================================
#  shared forward skeletons (VERBATIM from qk_sharedcore.py)
# =====================================================================================
def bil(mm, h):
    Lw, Rw, Dw, _ = WTS[mm]
    return ((h @ Lw.T) * (h @ Rw.T)) @ Dw.T

@torch.no_grad()
def fwd_core(idx, hook):
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
        q, k_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k_)/HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
        pat = (s1*s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        h = F.rms_norm(x, (D,))
        mo = hook(li, h, B)
        if mo is None: mo = blk.mlp(h)
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                           reduction='none').view(B, T-1)

def eval_hook(hook, tag, batch=B0M, idxs=None):
    idxs = HELD if idxs is None else idxs
    n = idxs.shape[0]; t0 = time.time()
    ce = torch.cat([fwd_core(idxs[i:i+batch], hook).cpu() for i in range(0, n, batch)], 0)
    mn, se = dstat(ce)
    print(f"  [{tag}] dCE {mn:+.4f} +- {se:.5f} ({time.time()-t0:.0f}s)", flush=True)
    return ce, mn, se

def mix_hook(W, BM):
    Wt = torch.as_tensor(W, dtype=torch.float32)
    def hook(li, h, B):
        mo = (BIAS[li] + BM[li, li]).unsqueeze(0).expand(B, -1, -1).clone()
        for mm in range(NL):
            wlm = float(Wt[li, mm])
            if abs(wlm) < 1e-12: continue
            mo += wlm * (bil(mm, h) - BM[li, mm])
        return mo
    return hook

# =====================================================================================
#  ATTACK 1: GAUGE FREEDOM
# =====================================================================================
# ---- gauge machinery (module level so the long-run stage can reuse it) ----
res.setdefault('gauge', {})
CH = 1024

def prep(li):
    Lw, Rw, Dw, _ = WTS[li]
    return Lw.to(DEV), Rw.to(DEV), Dw.to(DEV)

def rand_block(seed, ref):
    g = torch.Generator(device='cpu').manual_seed(seed)
    Lw, Rw, Dw = ref
    out = []
    for W in (Lw, Rw, Dw):
        out.append((torch.randn(W.shape, generator=g) * W.std().cpu()).to(DEV))
    return tuple(out)

def obj_grad(Pl, Pm, R, S, need_grad=True):
    """<T_m, (S x R x R) T_l> and d/dR, exact, chunked over l's feature index."""
    Ll, Rl, Dl = Pl; Lm, Rm, Dm = Pm
    DS = S.T @ Dm                              # (D, F_m)
    X = Ll @ R.T; Y = Rl @ R.T                 # (F_l, D)
    tot = torch.zeros((), dtype=torch.float64, device=DEV)
    gX = torch.zeros_like(X) if need_grad else None
    gY = torch.zeros_like(Y) if need_grad else None
    for f0 in range(0, Ll.shape[0], CH):
        f1 = min(f0 + CH, Ll.shape[0])
        Mc = Dl[:, f0:f1].T @ DS               # (c, F_m)
        A = X[f0:f1] @ Lm.T; Bm = Y[f0:f1] @ Rm.T
        C = X[f0:f1] @ Rm.T; E = Y[f0:f1] @ Lm.T
        tot += (Mc.double() * (0.5*(A.double()*Bm.double() + C.double()*E.double()))).sum()
        if need_grad:
            gX[f0:f1] = 0.5*((Mc*Bm) @ Lm + (Mc*E) @ Rm)
            gY[f0:f1] = 0.5*((Mc*A) @ Rm + (Mc*C) @ Lm)
        del Mc, A, Bm, C, E
    gR = (gX.T @ Ll + gY.T @ Rl) if need_grad else None
    return float(tot), gR

def out_corr(Pl, Pm, R):
    """Mout[o,o'] = <T_m[o], (R x R)T_l[o']> ; max over orthogonal S is ||Mout||_nuclear."""
    Ll, Rl, Dl = Pl; Lm, Rm, Dm = Pm
    X = Ll @ R.T; Y = Rl @ R.T
    acc = torch.zeros(D, D, dtype=torch.float64, device=DEV)
    for f0 in range(0, Ll.shape[0], CH):
        f1 = min(f0 + CH, Ll.shape[0])
        A = X[f0:f1] @ Lm.T; Bm = Y[f0:f1] @ Rm.T
        C = X[f0:f1] @ Rm.T; E = Y[f0:f1] @ Lm.T
        K = 0.5*(A*Bm + C*E)                   # (c, F_m)
        acc += (Dl[:, f0:f1] @ (K @ Dm.T)).double()   # [o', o]
        del A, Bm, C, E, K
    return acc.T                                # [o, o']

def sqnorm(P):
    v, _ = obj_grad(P, P, torch.eye(D, device=DEV), torch.eye(D, device=DEV), need_grad=False)
    return v

def polar(Mt):
    U, _, Vh = torch.linalg.svd(Mt.double(), full_matrices=False)
    return (U @ Vh).float()

def fit_pair(Pl, Pm, tag, iters=40, verbose=True, R0=None):
    nl, nm = math.sqrt(sqnorm(Pl)), math.sqrt(sqnorm(Pm))
    R = torch.eye(D, device=DEV) if R0 is None else R0.clone()
    S = torch.eye(D, device=DEV)
    raw, _ = obj_grad(Pl, Pm, R, S, need_grad=False)
    hist = []
    # initial exact S-step
    Mo = out_corr(Pl, Pm, R)
    U_, s_, Vh_ = torch.linalg.svd(Mo)
    S = (U_ @ Vh_).float(); cur = float(s_.sum())
    lr = None
    for it in range(iters):
        val, gR = obj_grad(Pl, Pm, R, S, need_grad=True)
        gn = gR.norm().item()
        if lr is None: lr = math.sqrt(D) / max(gn, 1e-12) * 0.5
        best = (val, R, lr)
        for trial in [lr*4, lr*2, lr, lr*0.5, lr*0.25, lr*0.0625]:
            Rt = polar(R + trial * gR)
            vt, _ = obj_grad(Pl, Pm, Rt, S, need_grad=False)
            if vt > best[0]: best = (vt, Rt, trial)
        val, R, lr = best
        Mo = out_corr(Pl, Pm, R)
        U_, s_, Vh_ = torch.linalg.svd(Mo)
        S = (U_ @ Vh_).float(); new = float(s_.sum())
        hist.append(new / (nl*nm))
        if verbose and (it % 5 == 0 or it == iters-1):
            print(f"    [{tag}] iter {it:3d} aligned cos {new/(nl*nm):.4f} "
                  f"(raw {raw/(nl*nm):+.4f}) lr {lr:.2e}", flush=True)
        if it > 6 and abs(hist[-1] - hist[-4]) < 3e-5: break
        cur = new
    # von Neumann upper bound over the FULL orthogonal group on every mode
    Ml = out_corr(Pl, Pl, torch.eye(D, device=DEV))
    Mm = out_corr(Pm, Pm, torch.eye(D, device=DEV))
    sl = torch.linalg.eigvalsh(0.5*(Ml+Ml.T)).clamp_min(0).sqrt().flip(0)
    sm = torch.linalg.eigvalsh(0.5*(Mm+Mm.T)).clamp_min(0).sqrt().flip(0)
    ub = float((sl*sm).sum()) / (nl*nm)
    out = {'raw_cosine': raw/(nl*nm), 'aligned_cosine': hist[-1],
           'upper_bound_orthogonal': ub, 'iters_run': len(hist),
           'trajectory': [round(v, 5) for v in hist]}
    print(f"  [{tag}] RAW {out['raw_cosine']:+.4f} -> ALIGNED {out['aligned_cosine']:.4f} "
          f"(hard upper bound {ub:.4f})", flush=True)
    return out


def gauge_stage():
    res.setdefault('gauge', {})
    # GATE: objective at identity reproduces the cached Gram
    if 'gauge_identity' not in res['gates']:
        P15, P16 = prep(15), prep(16)
        v, _ = obj_grad(P15, P16, torch.eye(D, device=DEV), torch.eye(D, device=DEV), need_grad=False)
        rel = abs(v - G[15, 16]) / abs(G[15, 16])
        print(f"GATE gauge objective at identity: {v:.8e} vs cached Gram {G[15,16]:.8e} rel {rel:.2e}",
              flush=True)
        assert rel < 1e-5, "gauge identity gate FAILED"
        # small-case double-precision gate: the gauged objective against an EXPLICIT fold, and
        # the analytic gradient against autograd (the full-size finite difference is below the
        # float32 noise floor of a 2.3e6 objective, so it is done small and exact instead)
        g0 = torch.Generator(device='cpu').manual_seed(7)
        d_, f_ = 9, 6
        rnd = lambda *s: torch.randn(*s, generator=g0, dtype=torch.float64, device='cpu').to(DEV)
        Pa = (rnd(f_, d_), rnd(f_, d_), rnd(d_, f_)); Pb = (rnd(f_, d_), rnd(f_, d_), rnd(d_, f_))
        Rq, _ = torch.linalg.qr(rnd(d_, d_)); Sq, _ = torch.linalg.qr(rnd(d_, d_))
        def fold(P):
            Lw, Rw, Dw = P
            Tt = torch.einsum('of,fi,fj->oij', Dw, Lw, Rw); return 0.5*(Tt + Tt.transpose(1, 2))
        Ta = fold((Pa[0] @ Rq.T, Pa[1] @ Rq.T, Sq @ Pa[2])); Tb = fold(Pb)
        exp = float((Ta*Tb).sum())
        got, _ = obj_grad(Pa, Pb, Rq, Sq, need_grad=False)
        rel2 = abs(got-exp)/abs(exp)
        Rv = Rq.clone().requires_grad_(True)
        val = (fold((Pa[0] @ Rv.T, Pa[1] @ Rv.T, Sq @ Pa[2])) * Tb).sum(); val.backward()
        _, gA = obj_grad(Pa, Pb, Rq, Sq, need_grad=True)
        relg = float((gA - Rv.grad).norm()/Rv.grad.norm())
        print(f"GATE gauged objective vs explicit fold: {got:.10e} vs {exp:.10e} rel {rel2:.2e}; "
              f"analytic gradient vs autograd rel {relg:.2e}", flush=True)
        assert rel2 < 1e-10 and relg < 1e-10, "gauge small-case gate FAILED"
        res['gates']['gauge_identity'] = {
            'obj_identity': v, 'cached_gram': float(G[15, 16]), 'rel_err': rel,
            'smallcase_obj_rel_err': rel2, 'smallcase_grad_rel_err': relg}
        dump()

    PAIRS = [(15, 16, 'aligned_15_16'), (14, 15, 'aligned_14_15'), (5, 6, 'aligned_5_6'),
             (2, 17, 'control_far_2_17'), (4, 12, 'control_far_4_12')]
    for (l, mm, tag) in PAIRS:
        if tag in res['gauge']: continue
        t0 = time.time()
        o = fit_pair(prep(l), prep(mm), tag)
        o.update({'block_l': l, 'block_m': mm,
                  'raw_cosine_cached': float(G[l, mm]/math.sqrt(G[l, l]*G[mm, mm])),
                  'weighted_cosine_perblock': float(
                      np.array(json.load(open(f'{QK}/qk_sharedcore.json'))
                               ['census']['per_block_input_metric']['cosine_matrix'])[l, mm]),
                  'secs': round(time.time()-t0, 1)}
                 )
        res['gauge'][tag] = o; dump()
    if 'control_random_15' not in res['gauge']:
        Pr = rand_block(11, prep(15))
        o = fit_pair(prep(15), Pr, 'control_random_15')
        o['note'] = ('block 15 against a random-factor tensor of matched shape and factor scale '
                     '-- the chance level of the gauge fit')
        res['gauge']['control_random_15'] = o; dump()
    if 'control_random_pair' not in res['gauge']:
        Pa = rand_block(21, prep(15)); Pb = rand_block(22, prep(15))
        o = fit_pair(Pa, Pb, 'control_random_pair')
        o['note'] = 'two independent random-factor tensors -- pure chance level'
        res['gauge']['control_random_pair'] = o; dump()
    print("GAUGE STAGE DONE", flush=True)


def gauge2_stage():
    """Convergence and local-optimum robustness for the deepest attack: the real pair 15-16 and
    the random-tensor control at 4x the iteration budget, plus random-orthogonal restarts from
    random orthogonal initializations (guards against a poor identity-initialized optimum)."""
    res.setdefault('gauge', {})
    JOBS = [('long_15_16', lambda: (prep(15), prep(16)),
             int(os.environ.get('G2_ITERS', 160)), None)]
    for (tag, mk, iters, R0) in JOBS:
        if tag in res['gauge']: continue
        Pl, Pm = mk()
        o = fit_pair(Pl, Pm, tag, iters=iters, R0=R0)
        o['note'] = f'{iters}-iteration run (convergence check)'
        res['gauge'][tag] = o; dump()
    for s in [int(x) for x in os.environ.get('G2_SEEDS', '101,202').split(',')]:
        tag = f'restart{s}_15_16'
        if tag in res['gauge']: continue
        g = torch.Generator(device='cpu').manual_seed(s)
        Rr, _ = torch.linalg.qr(torch.randn(D, D, generator=g).to(DEV).double())
        o = fit_pair(prep(15), prep(16), tag,
                     iters=int(os.environ.get('G2_ITERS', 60)), R0=Rr.float())
        o['note'] = 'random-orthogonal initialization (local-optimum check)'
        res['gauge'][tag] = o; dump()
    print("GAUGE-2 STAGE DONE", flush=True)

def fwd_train(idx, W, BM, ckpt=True):
    """Differentiable-in-W mixture forward (checkpointed per block)."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    v1 = None
    for li in range(NL):
        def blockfn(x, x0, v1_, Wrow, li=li):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
            a = blk.attn; hcur = F.rms_norm(x, (D,))
            def qk(l):
                z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            vv = v if v1_ is None else ((1-a.lamb)*v + a.lamb*v1_.view_as(v))
            if v1_ is None: vv = v  # first layer: v1 = v
            q, k_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k_)/HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0)
            yh = torch.einsum('bhqk,bkhd->bqhd', pat, vv)
            x = x + a.c_proj(yh.reshape(B, T, -1))
            h = F.rms_norm(x, (D,))
            mo = (BIAS[li] + BM[li, li]).unsqueeze(0)
            acc = 0
            for mm in range(NL):
                acc = acc + Wrow[mm] * (bil(mm, h) - BM[li, mm])
            x = x + mo + acc
            return x, v
        if ckpt:
            x, v = checkpoint(blockfn, x, x0, v1, W[li], use_reentrant=False)
        else:
            x, v = blockfn(x, x0, v1, W[li])
        if v1 is None: v1 = v
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1))


# =====================================================================================
#  ATTACK 2: CAUSALLY REFIT MIXING
# =====================================================================================
def refit_stage():
    res.setdefault('refit', {})
    BM = cache['BM'].to(DEV)
    cache92 = torch.load(f'{QK}/qk_rank_alloc_cache.pt', map_location='cpu', weights_only=True)
    INb = [b.to(DEV) for b in cache92['INb']]; OUTb = [b.to(DEV) for b in cache92['OUTb']]
    MX = [t.to(DEV) for t in cache92['MX']]; MO = [t.to(DEV) for t in cache92['MO']]

    @torch.no_grad()
    def fwd_rank(idx, Kin, Kout):
        PIN = [INb[l][:, :Kin].contiguous() for l in range(NL)]
        POUT = None if Kout >= D else [OUTb[l][:, :Kout].contiguous() for l in range(NL)]
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
            xr = MX[li].unsqueeze(0) + ((x - MX[li].unsqueeze(0)) @ PIN[li]) @ PIN[li].T
            mo = blk.mlp(F.rms_norm(xr, (D,)))
            if POUT is not None:
                mo = MO[li].unsqueeze(0) + ((mo - MO[li].unsqueeze(0)) @ POUT[li]) @ POUT[li].T
            x = x + mo
        logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
        return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                               reduction='none').view(B, T-1)

    NVAL = 64
    TR_FIT = TRAIN[:STR-NVAL]; TR_VAL = TRAIN[STR-NVAL:]
    LRS = [float(x) for x in os.environ.get('REFIT_LRS', '1e-3,3e-4,1e-4').split(',')]
    STEPS = int(os.environ.get('REFIT_STEPS', 300))
    BF = 4
    for k in [16, 12]:
        key = f'k{k}'
        if key in res['refit']: continue
        Uk = torch.from_numpy(UP[:, :k].copy()).float().to(DEV)
        best = (1e9, Uk.clone(), -1, None)     # (val CE, A, step, lr)
        runs = {}
        for lr in LRS:
            A = torch.nn.Parameter(Uk.clone())
            opt = torch.optim.Adam([A], lr=lr)
            vhist = []; thist = []
            t0 = time.time(); perm = torch.randperm(TR_FIT.shape[0]); ptr = 0; runavg = None
            for step in range(STEPS+1):
                if step % 25 == 0:
                    with torch.no_grad():
                        vl = float(np.mean([
                            float(fwd_train(TR_VAL[i:i+BF], A @ Uk.T, BM, ckpt=False))
                            for i in range(0, NVAL, BF)]))
                        tp = float(np.mean([
                            float(fwd_train(TR_FIT[i:i+BF], A @ Uk.T, BM, ckpt=False))
                            for i in range(0, 32, BF)]))
                    vhist.append((step, round(vl, 5)))
                    thist.append((step, round(tp, 5)))
                    if vl < best[0]: best = (vl, A.detach().clone(), step, lr)
                    print(f"  [refit k={k} lr={lr:g}] step {step:4d} val CE {vl:.5f} "
                          f"fit-probe CE {tp:.5f} "
                          f"(global best {best[0]:.5f} @ lr {best[3]} step {best[2]}) "
                          f"{time.time()-t0:.0f}s", flush=True)
                if step == STEPS: break
                if ptr + BF > TR_FIT.shape[0]:
                    perm = torch.randperm(TR_FIT.shape[0]); ptr = 0
                idx = TR_FIT[perm[ptr:ptr+BF]]; ptr += BF
                loss = fwd_train(idx, A @ Uk.T, BM)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                fl = float(loss)
                runavg = fl if runavg is None else 0.9*runavg + 0.1*fl
            runs[f'lr{lr:g}'] = {'val_history': vhist, 'fit_probe_history': thist}
            del A, opt; torch.cuda.empty_cache()
        Abest = best[1]
        Wref = (Abest @ Uk.T).detach().cpu().numpy()
        Wpca = (Uk @ Uk.T).detach().cpu().numpy()
        ce_ref, mn_ref, se_ref = eval_hook(mix_hook(Wref, BM), f'refit k={k}')
        ce_pca, mn_pca, se_pca = eval_hook(mix_hook(Wpca, BM), f'pca   k={k}')
        Kin, Kout = {16: (1152, 1024), 12: (1006, 1006)}[k]
        ce_rk = torch.cat([fwd_rank(HELD[i:i+B0], Kin, Kout).cpu() for i in range(0, S_, B0)], 0)
        mn_rk, se_rk = dstat(ce_rk)
        def paired(a, b):
            d = (a - b).flatten().double()
            mu = float(d.mean()); se = float(d.std()/np.sqrt(d.numel()))
            return {'diff': round(mu, 4), 'SE': round(se, 5),
                    'z': (round(mu/se, 1) if se > 0 else None)}
        rec = {'k': k, 'atoms': 'pooled_metric_U[:, :k] (fixed)', 'free_params': 18*k,
               'train_fit': f'Adam, batch {BF}, {STEPS} steps, learning-rate sweep {LRS}, '
                            'validation split TRAIN[192:256], early stop on best validation CE '
                            'across all runs and steps',
               'runs': runs, 'best_step': best[2], 'best_lr': best[3],
               'best_val_ce': round(best[0], 5),
               'refit_dCE': round(mn_ref, 4), 'refit_SE': round(se_ref, 5),
               'pca_dCE': round(mn_pca, 4), 'pca_SE': round(se_pca, 5),
               'rank_alloc_dCE': round(mn_rk, 4), 'rank_alloc_SE': round(se_rk, 5),
               'rank_config': [Kin, Kout],
               'paired_refit_minus_pca': paired(ce_ref, ce_pca),
               'paired_refit_minus_rank': paired(ce_ref, ce_rk),
               'W_refit_diag': [round(float(v), 4) for v in np.diag(Wref)],
               'A_shift_from_U': float(np.linalg.norm(Abest.cpu().numpy() - UP[:, :k])
                                       / np.linalg.norm(UP[:, :k]))}
        res['refit'][key] = rec; dump()
        print(f"  [k={k}] refit {mn_ref:+.4f} vs pca {mn_pca:+.4f} vs rank {mn_rk:+.4f}; "
              f"refit-rank paired {rec['paired_refit_minus_rank']}", flush=True)
        torch.cuda.empty_cache()
    print("REFIT STAGE DONE", flush=True)

# =====================================================================================
#  ATTACK 3: MEAN-CENTERED ATOMS
# =====================================================================================
def center_stage():
    res.setdefault('center', {})
    BM = cache['BM'].to(DEV)
    J = np.eye(NL) - np.ones((NL, NL))/NL

    def census(Gm):
        Gm = 0.5*(Gm + Gm.T)
        w = np.clip(np.linalg.eigvalsh(Gm)[::-1], 0, None)
        frac = np.cumsum(w)/w.sum()
        dg = np.sqrt(np.clip(np.diag(Gm), 1e-30, None)); C = Gm/np.outer(dg, dg)
        iu = np.triu_indices(NL, 1)
        wc = np.clip(np.linalg.eigvalsh(0.5*(C+C.T))[::-1], 0, None)
        fc = np.cumsum(wc)/wc.sum()
        order = np.argsort(-np.abs(C[iu]))[:5]
        return {'rank90': int(np.searchsorted(frac, 0.90)+1),
                'rank95': int(np.searchsorted(frac, 0.95)+1),
                'rank99': int(np.searchsorted(frac, 0.99)+1),
                'participation_ratio': round(float(w.sum()**2/(w**2).sum()), 3),
                'top_eig_frac': round(float(w[0]/w.sum()), 4),
                'mean_abs_offdiag_cosine': round(float(np.abs(C[iu]).mean()), 4),
                'max_abs_offdiag_cosine': round(float(np.abs(C[iu]).max()), 4),
                'top_pairs': [[int(iu[0][o]), int(iu[1][o]), round(float(C[iu][o]), 3)]
                              for o in order],
                'normfree_rank90': int(np.searchsorted(fc, 0.90)+1),
                'normfree_rank95': int(np.searchsorted(fc, 0.95)+1),
                'normfree_top_eig_frac': round(float(wc[0]/wc.sum()), 4)}

    for tag, Gm in [('raw_frobenius', G), ('pooled_input_metric', Gp)]:
        if f'census_centered_{tag}' in res['center']: continue
        Gc = J @ Gm @ J
        res['center'][f'census_centered_{tag}'] = census(Gc)
        res['center'][f'census_uncentered_{tag}'] = census(Gm)
        print(f"  [centered {tag}] {res['center'][f'census_centered_{tag}']}", flush=True)
        dump()

    def Weff(Gm, k):
        Gc = 0.5*(J @ Gm @ J + (J @ Gm @ J).T)
        w, U = np.linalg.eigh(Gc); U = U[:, ::-1][:, :k]
        P = U @ U.T
        s = P.sum(1)
        return P + np.outer(1 - s, np.ones(NL))/NL

    # budget-matched cells: k centered atoms + the mean = k+1 blocks
    CELLS = [(1, 2), (3, 4), (7, 8), (11, 12), (15, 16)]
    REF = {2: 3.2010, 4: 2.8694, 8: 1.3310, 12: 0.0775, 16: 0.0171}
    for (k, tot) in CELLS:
        for vtag, Gm in [('pooled_metric', Gp), ('frobenius', G)]:
            key = f'k{k}plusmean_{vtag}'
            if key in res['center']: continue
            W = Weff(Gm, k)
            ce, mn, se = eval_hook(mix_hook(W, BM), key)
            res['center'][key] = {
                'centered_atoms': k, 'total_atoms_counted': tot, 'variant': vtag,
                'dCE': round(mn, 4), 'SE': round(se, 5),
                'budget': int(tot*FULLBLK + NL*tot), 'compression_x': round(NL/tot, 3),
                'uncentered_ref_same_budget': REF[tot],
                'W_diag': [round(float(v), 4) for v in np.diag(W)]}
            dump()
    print("CENTER STAGE DONE", flush=True)

# =====================================================================================
#  ATTACK 4: OUTPUT-SLOT WEIGHTING
# =====================================================================================
def outw_stage():
    res.setdefault('outw', {})
    CH = 1024

    def fwd_probe(idx, probes=None):
        """Base forward written with the block bilinear split out explicitly.  probes[li], if
        given, is a zero tensor requiring grad added to block li's bilinear output, so that
        probes[li].grad is the per-position gradient of the loss with respect to that output
        (the model's own weights stay frozen -- only activation gradients flow)."""
        B, T = idx.shape
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
        cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        outs = []
        for li in range(NL):
            blk = m.transformer.h[li]; x = blk.lambdas[0]*x + blk.lambdas[1]*x0
            a = blk.attn; hcur = F.rms_norm(x, (D,))
            def qk(l):
                z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k_, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k_)/HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD
            pat = (s1*s2).masked_fill(~mask, 0.0)
            yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
            x = x + a.c_proj(yh.reshape(B, T, -1))
            h = F.rms_norm(x, (D,))
            b_out = bil(li, h)                      # bilinear part only (the folded tensor)
            if probes is not None: b_out = b_out + probes[li]
            outs.append(b_out)
            x = x + BIAS[li] + b_out
        logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                             reduction='none').view(B, T-1)
        return ce, outs

    if 'gate_fwd_probe' not in res['gates']:
        with torch.no_grad():
            ce = torch.cat([fwd_probe(HELD[i:i+B0])[0].cpu() for i in range(0, S_, B0)], 0)
        mn, se = dstat(ce)
        print(f"GATE fwd_probe reproduces base: dCE {mn:+.2e}", flush=True)
        assert abs(mn) < 1e-5, "fwd_probe gate FAILED"
        res['gates']['gate_fwd_probe'] = {'dCE_vs_base': mn}; dump()

    if 'Sout_act' not in rcache or 'Sout_grad' not in rcache:
        print("TRAIN pass: block output second moments (activation + consumption gradient) ...",
              flush=True)
        Gact = torch.zeros(NL, D, D, device=DEV)
        Ggrd = torch.zeros(NL, D, D, device=DEV)
        ntok = 0
        BF = 2
        for i in range(0, STR, BF):
            idx = TRAIN[i:i+BF]
            B, T = idx.shape
            probes = [torch.zeros(B, T, D, device=DEV, requires_grad=True) for _ in range(NL)]
            ce, outs = fwd_probe(idx, probes)
            ce.sum().backward()
            with torch.no_grad():
                for li in range(NL):
                    o = outs[li].detach()
                    Gact[li] += torch.einsum('btd,bte->de', o, o)
                    g = probes[li].grad
                    Ggrd[li] += torch.einsum('btd,bte->de', g, g)
            ntok += B*T
            del outs, probes, ce
            if (i//BF) % 32 == 0: print(f"    train {i}/{STR}", flush=True)
        rcache['Sout_act'] = (Gact/ntok).cpu(); rcache['Sout_grad'] = (Ggrd/ntok).cpu()
        torch.save(rcache, RCPT)
        del Gact, Ggrd; torch.cuda.empty_cache()
    Oact = rcache['Sout_act']; Ogrd = rcache['Sout_grad']

    def sqrt_psd(Mres):
        w, vecs = torch.linalg.eigh(Mres.double())
        return (vecs * w.clamp_min(0).sqrt().unsqueeze(0)) @ vecs.T

    Sin = sqrt_psd(Sig.mean(0)).float().to(DEV)
    SO = {'act': sqrt_psd(Oact.mean(0)).float().to(DEV),
          'grad': sqrt_psd(Ogrd.mean(0)).float().to(DEV)}
    # normalize the output metrics to unit mean eigenvalue (cosines are scale-free anyway)
    for kk in SO: SO[kk] = SO[kk] / SO[kk].diagonal().mean()

    def closed_gram_pair(Ll, Rl, Dl, Lm, Rm, Dm):
        tot = torch.zeros((), dtype=torch.float64, device=DEV)
        for f0 in range(0, Ll.shape[0], CH):
            f1 = min(f0+CH, Ll.shape[0])
            A = Ll[f0:f1] @ Lm.T; Bm = Rl[f0:f1] @ Rm.T
            C = Ll[f0:f1] @ Rm.T; E = Rl[f0:f1] @ Lm.T
            M = Dl.T[f0:f1] @ Dm
            tot += (M.double() * (0.5*(A.double()*Bm.double() + C.double()*E.double()))).sum()
            del A, Bm, C, E, M
        return float(tot)

    def build(Ls, Rs, Ds, tag):
        Gm = np.zeros((NL, NL))
        for l in range(NL):
            for mm in range(l, NL):
                Gm[l, mm] = Gm[mm, l] = closed_gram_pair(Ls[l], Rs[l], Ds[l], Ls[mm], Rs[mm], Ds[mm])
        print(f"  [{tag}] gram built", flush=True)
        return Gm

    def census(Gm):
        Gm = 0.5*(Gm + Gm.T)
        w, U = np.linalg.eigh(Gm); w = w[::-1].copy(); U = U[:, ::-1].copy()
        wc = np.clip(w, 0, None); frac = np.cumsum(wc)/wc.sum()
        dg = np.sqrt(np.clip(np.diag(Gm), 1e-30, None)); C = Gm/np.outer(dg, dg)
        iu = np.triu_indices(NL, 1)
        wcc = np.clip(np.linalg.eigvalsh(0.5*(C+C.T))[::-1], 0, None); fc = np.cumsum(wcc)/wcc.sum()
        order = np.argsort(-np.abs(C[iu]))[:6]
        adj = float(np.mean([abs(C[i, i+1]) for i in range(NL-1)]))
        nonadj = float(np.mean([abs(C[i, j]) for i in range(NL) for j in range(i+2, NL)]))
        return {'rank90': int(np.searchsorted(frac, 0.90)+1),
                'rank95': int(np.searchsorted(frac, 0.95)+1),
                'rank99': int(np.searchsorted(frac, 0.99)+1),
                'participation_ratio': round(float(wc.sum()**2/(wc**2).sum()), 3),
                'top_eig_frac': round(float(wc[0]/wc.sum()), 4),
                'block_sqnorm_ratio_max_over_median': round(
                    float(np.diag(Gm).max()/np.median(np.diag(Gm))), 1),
                'mean_abs_offdiag_cosine': round(float(np.abs(C[iu]).mean()), 4),
                'max_abs_offdiag_cosine': round(float(np.abs(C[iu]).max()), 4),
                'mean_abs_cos_adjacent': round(adj, 4),
                'mean_abs_cos_nonadjacent': round(nonadj, 4),
                'top_pairs': [[int(iu[0][o]), int(iu[1][o]), round(float(C[iu][o]), 3)]
                              for o in order],
                'normfree_rank90': int(np.searchsorted(fc, 0.90)+1),
                'normfree_rank95': int(np.searchsorted(fc, 0.95)+1),
                'normfree_top_eig_frac': round(float(wcc[0]/wcc.sum()), 4)}, U

    Ls = [w[0].to(DEV) for w in WTS]; Rs = [w[1].to(DEV) for w in WTS]; Ds = [w[2].to(DEV) for w in WTS]
    Us = {}
    VARIANTS = [('out_act_only', None, 'act'), ('out_grad_only', None, 'grad'),
                ('in_pooled_out_act', Sin, 'act'), ('in_pooled_out_grad', Sin, 'grad')]
    for (tag, Si, so) in VARIANTS:
        if f'census_{tag}' in res['outw'] and f'U_{tag}' in rcache:
            Us[tag] = rcache[f'U_{tag}'].numpy(); continue
        Lw_ = [(Ls[l] @ Si) if Si is not None else Ls[l] for l in range(NL)]
        Rw_ = [(Rs[l] @ Si) if Si is not None else Rs[l] for l in range(NL)]
        Dw_ = [SO[so] @ Ds[l] for l in range(NL)]
        Gm = build(Lw_, Rw_, Dw_, tag)
        cen, U = census(Gm)
        res['outw'][f'census_{tag}'] = cen
        rcache[f'G_{tag}'] = torch.from_numpy(Gm); rcache[f'U_{tag}'] = torch.from_numpy(U)
        Us[tag] = U
        print(f"  [{tag}] rank90 {cen['rank90']} normfree90 {cen['normfree_rank90']} "
              f"mean|cos| {cen['mean_abs_offdiag_cosine']} max {cen['max_abs_offdiag_cosine']} "
              f"top {cen['top_pairs'][:3]}", flush=True)
        torch.save(rcache, RCPT); dump()
    del Ls, Rs, Ds; torch.cuda.empty_cache()

    # causal check with the best output-weighted atoms
    BM = cache['BM'].to(DEV)
    REF = {2: 3.2010, 4: 2.8694, 8: 1.3310, 12: 0.0775, 16: 0.0171}
    for tag in ['in_pooled_out_grad', 'in_pooled_out_act']:
        U = Us[tag]
        for k in [16, 12, 8, 4]:
            key = f'k{k}_{tag}'
            if key in res['outw']: continue
            W = U[:, :k] @ U[:, :k].T
            ce, mn, se = eval_hook(mix_hook(W, BM), key)
            res['outw'][key] = {'k': k, 'variant': tag, 'dCE': round(mn, 4), 'SE': round(se, 5),
                                'sec105_best_ref': REF[k],
                                'W_diag': [round(float(v), 4) for v in np.diag(W)]}
            dump()
    print("OUTW STAGE DONE", flush=True)

def smoke_stage():
    """Cheap correctness smoke test of the machinery used by attacks 2 and 3 (no writes)."""
    BM = cache['BM'].to(DEV)
    # (a) mean-centred W at k=17 must reproduce the identity mixture (exact model)
    J = np.eye(NL) - np.ones((NL, NL))/NL
    Gc = 0.5*(J @ Gp @ J + (J @ Gp @ J).T)
    w, U = np.linalg.eigh(Gc); U = U[:, ::-1][:, :17]
    P = U @ U.T; s_ = P.sum(1); W = P + np.outer(1-s_, np.ones(NL))/NL
    print(f"smoke: centered k=17 W deviation from identity "
          f"{np.abs(W-np.eye(NL)).max():.2e}", flush=True)
    hook = mix_hook(W, BM)
    ce = torch.cat([fwd_core(HELD[i:i+4], hook).cpu() for i in range(0, 24, 4)], 0)
    mn = float((ce - base[:24]).mean())
    print(f"smoke: centered k=17 mixture dCE {mn:+.2e}", flush=True)
    assert abs(mn) < 1e-3, "centered identity smoke FAILED"
    # (b) refit forward: differentiable mixture at W = I must reproduce the base cross-entropy,
    #     and a gradient step must move the loss
    # (b) differentiable mixture forward at W = I must reproduce the base cross-entropy, and a
    #     gradient with respect to the mixing coefficients must be nonzero
    Uk = torch.from_numpy(UP[:, :16].copy()).float().to(DEV)
    A = torch.nn.Parameter(Uk.clone())
    with torch.no_grad():
        I18 = torch.eye(NL, device=DEV)
        l_id = float(fwd_train(HELD[:4], I18, BM, ckpt=False))
        ce_ref = float(base[:4].mean())
    print(f"smoke: differentiable forward at W=I gives CE {l_id:.6f} vs base {ce_ref:.6f} "
          f"(diff {l_id-ce_ref:+.2e})", flush=True)
    assert abs(l_id - ce_ref) < 1e-3, "differentiable forward smoke FAILED"
    t0 = time.time(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    loss = fwd_train(TRAIN[:2], A @ Uk.T, BM); loss.backward()
    print(f"smoke: train step loss {float(loss):.5f} grad norm {float(A.grad.norm()):.4e} "
          f"({time.time()-t0:.1f}s) mem {torch.cuda.max_memory_allocated()/2**30:.2f} GB",
          flush=True)
    assert float(A.grad.norm()) > 0
    print("smoke OK", flush=True)


# =====================================================================================
#  ATTACK 4b: NORM-EQUALIZED (RELATIVE-ERROR) ATOMS
#  Section 105's atoms minimize the ABSOLUTE weighted Frobenius error summed over blocks, an
#  objective two blocks dominate (block 17 at 2.3e10 and block 0 at 5.2e9 vs ~2e7 mid-stack in
#  the pooled input metric) -- which is exactly why small k degenerates to selecting the big
#  blocks and mean-ablating the middle.  Since section 92 measured per-layer restriction costs
#  to be nearly FLAT across the stack, the loss-relevant objective is RELATIVE error per block.
#  Rank-k principal subspace of the UNIT-NORMALIZED stack = eigenvectors of the cosine matrix;
#  the reconstruction is still a linear mixture, W[l,m] = P[l,m] * ||T_l|| / ||T_m||.
# =====================================================================================
def normw_stage():
    res.setdefault('normw', {})
    BM = cache['BM'].to(DEV)
    J = np.eye(NL) - np.ones((NL, NL))/NL
    REF = {2: 3.2010, 4: 2.8694, 8: 1.3310, 12: 0.0775, 16: 0.0171}
    GRAMS = {'raw_frobenius': G, 'pooled_input_metric': Gp}
    for nm in ['in_pooled_out_grad', 'in_pooled_out_act']:
        if f'G_{nm}' in rcache: GRAMS[nm] = rcache[f'G_{nm}'].numpy()

    def Wnorm(Gm, k, centered=False):
        Gu = 0.5*(J @ Gm @ J + (J @ Gm @ J).T) if centered else 0.5*(Gm + Gm.T)
        dg = np.sqrt(np.clip(np.diag(Gu), 1e-30, None))
        C = Gu / np.outer(dg, dg)
        w, U = np.linalg.eigh(0.5*(C + C.T)); U = U[:, ::-1][:, :k]
        P = U @ U.T
        Q = P * np.outer(dg, 1.0/dg)
        if not centered: return Q, float(np.clip(w[::-1], 0, None)[:k].sum()/NL)
        Weff = Q + np.outer(1 - Q.sum(1), np.ones(NL))/NL
        return Weff, float(np.clip(w[::-1], 0, None)[:k].sum()/NL)

    for gtag, Gm in GRAMS.items():
        for k in [16, 12, 8, 4, 2]:
            key = f'k{k}_norm_{gtag}'
            if key in res['normw']: continue
            W, cap = Wnorm(Gm, k)
            _, mn, se = eval_hook(mix_hook(W, BM), key)
            res['normw'][key] = {'k': k, 'gram': gtag, 'centered': False,
                                 'dCE': round(mn, 4), 'SE': round(se, 5),
                                 'relative_energy_captured': round(cap, 4),
                                 'sec105_best_ref': REF[k],
                                 'W_diag': [round(float(v), 4) for v in np.diag(W)]}
            dump()
    for gtag in ['pooled_input_metric']:
        Gm = GRAMS[gtag]
        for (k, tot) in [(15, 16), (11, 12), (7, 8), (3, 4), (1, 2)]:
            key = f'k{k}plusmean_norm_{gtag}'
            if key in res['normw']: continue
            W, cap = Wnorm(Gm, k, centered=True)
            _, mn, se = eval_hook(mix_hook(W, BM), key)
            res['normw'][key] = {'centered_atoms': k, 'total_atoms_counted': tot, 'gram': gtag,
                                 'centered': True, 'dCE': round(mn, 4), 'SE': round(se, 5),
                                 'relative_energy_captured': round(cap, 4),
                                 'sec105_best_ref_same_budget': REF[tot],
                                 'W_diag': [round(float(v), 4) for v in np.diag(W)]}
            dump()
    print("NORMW STAGE DONE", flush=True)


# =====================================================================================
#  PAIRED HEADLINE COMPARISONS for the winning red-team variants (centred / norm-equalized)
#  against section 92 rank allocation at the same budget, per-position paired standard errors.
# =====================================================================================
def centpair_stage():
    res.setdefault('centpair', {})
    BM = cache['BM'].to(DEV)
    cache92 = torch.load(f'{QK}/qk_rank_alloc_cache.pt', map_location='cpu', weights_only=True)
    INb = [b.to(DEV) for b in cache92['INb']]; OUTb = [b.to(DEV) for b in cache92['OUTb']]
    MX = [t.to(DEV) for t in cache92['MX']]; MO = [t.to(DEV) for t in cache92['MO']]

    @torch.no_grad()
    def fwd_rank(idx, Kin, Kout):
        PIN = [INb[l][:, :Kin].contiguous() for l in range(NL)]
        POUT = None if Kout >= D else [OUTb[l][:, :Kout].contiguous() for l in range(NL)]
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
            xr = MX[li].unsqueeze(0) + ((x - MX[li].unsqueeze(0)) @ PIN[li]) @ PIN[li].T
            mo = blk.mlp(F.rms_norm(xr, (D,)))
            if POUT is not None:
                mo = MO[li].unsqueeze(0) + ((mo - MO[li].unsqueeze(0)) @ POUT[li]) @ POUT[li].T
            x = x + mo
        logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
        return F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                               reduction='none').view(B, T-1)

    J = np.eye(NL) - np.ones((NL, NL))/NL
    def Wcent(Gm, k):
        Gc = 0.5*(J @ Gm @ J + (J @ Gm @ J).T)
        w, U = np.linalg.eigh(Gc); U = U[:, ::-1][:, :k]
        P = U @ U.T; s_ = P.sum(1)
        return P + np.outer(1 - s_, np.ones(NL))/NL

    CELLS = [(15, 16, 1152, 1024), (11, 12, 1006, 1006), (7, 8, 879, 879), (3, 4, 696, 696)]
    for (k, tot, Kin, Kout) in CELLS:
        key = f'centered_k{k}plusmean_vs_rank_{tot}'
        if key in res['centpair']: continue
        W = Wcent(Gp, k)
        ce_a, mn_a, se_a = eval_hook(mix_hook(W, BM), f'centered k{k}+mean')
        ce_b = torch.cat([fwd_rank(HELD[i:i+B0], Kin, Kout).cpu() for i in range(0, S_, B0)], 0)
        mn_b, se_b = dstat(ce_b)
        d = (ce_a - ce_b).flatten().double()
        mu = float(d.mean()); se = float(d.std()/np.sqrt(d.numel()))
        res['centpair'][key] = {
            'centered_atoms': k, 'total_atoms_counted': tot, 'compression_x': round(NL/tot, 3),
            'centered_dCE': round(mn_a, 4), 'centered_SE': round(se_a, 5),
            'rank_alloc_config': [Kin, Kout], 'rank_alloc_dCE': round(mn_b, 4),
            'rank_alloc_SE': round(se_b, 5),
            'paired_centered_minus_rank': round(mu, 4), 'paired_SE': round(se, 5),
            'z': round(mu/se, 1) if se > 0 else None}
        print(f"  [{key}] centered {mn_a:+.4f} vs rank {mn_b:+.4f} paired {mu:+.4f} "
              f"+- {se:.5f} z={mu/se:.1f}", flush=True)
        dump()
    # the output-weighted atom family (attack 4) at its own best budgets, paired the same way
    OWCELLS = [(16, 1152, 1024), (12, 1006, 1006), (8, 879, 879)]
    for nm in ['in_pooled_out_grad', 'in_pooled_out_act']:
        if f'U_{nm}' not in rcache: continue
        U = rcache[f'U_{nm}'].numpy()
        for (k, Kin, Kout) in OWCELLS:
            key = f'{nm}_k{k}_vs_rank'
            if key in res['centpair']: continue
            W = U[:, :k] @ U[:, :k].T
            ce_a, mn_a, se_a = eval_hook(mix_hook(W, BM), f'{nm} k{k}')
            ce_b = torch.cat([fwd_rank(HELD[i:i+B0], Kin, Kout).cpu() for i in range(0, S_, B0)], 0)
            mn_b, se_b = dstat(ce_b)
            d = (ce_a - ce_b).flatten().double()
            mu = float(d.mean()); se = float(d.std()/np.sqrt(d.numel()))
            res['centpair'][key] = {
                'atoms': nm, 'k': k, 'compression_x': round(NL/k, 3),
                'shared_dCE': round(mn_a, 4), 'shared_SE': round(se_a, 5),
                'rank_alloc_config': [Kin, Kout], 'rank_alloc_dCE': round(mn_b, 4),
                'rank_alloc_SE': round(se_b, 5),
                'paired_shared_minus_rank': round(mu, 4), 'paired_SE': round(se, 5),
                'z': round(mu/se, 1) if se > 0 else None}
            print(f"  [{key}] shared {mn_a:+.4f} vs rank {mn_b:+.4f} paired {mu:+.4f} "
                  f"+- {se:.5f} z={mu/se:.1f}", flush=True)
            dump()
    # BEST-OF-BOTH: mean-centred atoms IN the consumption (gradient) metric, the two red-team
    # improvements combined, at the two mild budgets where each alone reached a tie.
    if 'G_in_pooled_out_grad' in rcache:
        Gg = rcache['G_in_pooled_out_grad'].numpy()
        for (k, tot, Kin, Kout) in [(15, 16, 1152, 1024), (11, 12, 1006, 1006), (7, 8, 879, 879)]:
            key = f'centered_gradmetric_k{k}plusmean_vs_rank_{tot}'
            if key in res['centpair']: continue
            W = Wcent(Gg, k)
            ce_a, mn_a, se_a = eval_hook(mix_hook(W, BM), f'centered-grad k{k}+mean')
            ce_b = torch.cat([fwd_rank(HELD[i:i+B0], Kin, Kout).cpu() for i in range(0, S_, B0)], 0)
            mn_b, se_b = dstat(ce_b)
            d = (ce_a - ce_b).flatten().double()
            mu = float(d.mean()); se = float(d.std()/np.sqrt(d.numel()))
            res['centpair'][key] = {
                'centered_atoms': k, 'total_atoms_counted': tot, 'metric': 'in_pooled_out_grad',
                'compression_x': round(NL/tot, 3),
                'shared_dCE': round(mn_a, 4), 'shared_SE': round(se_a, 5),
                'rank_alloc_config': [Kin, Kout], 'rank_alloc_dCE': round(mn_b, 4),
                'rank_alloc_SE': round(se_b, 5),
                'paired_shared_minus_rank': round(mu, 4), 'paired_SE': round(se, 5),
                'z': round(mu/se, 1) if se > 0 else None}
            print(f"  [{key}] shared {mn_a:+.4f} vs rank {mn_b:+.4f} paired {mu:+.4f} "
                  f"+- {se:.5f} z={mu/se:.1f}", flush=True)
            dump()
    # norm-equalized (relative-error) atoms in the RAW Frobenius metric -- the one cell of
    # attack 4b that improved on section 105 -- paired against rank allocation.
    def Wnorm_raw(Gm, k):
        Gu = 0.5*(Gm + Gm.T); dg = np.sqrt(np.clip(np.diag(Gu), 1e-30, None))
        C = Gu/np.outer(dg, dg)
        w, U = np.linalg.eigh(0.5*(C+C.T)); U = U[:, ::-1][:, :k]
        return (U @ U.T) * np.outer(dg, 1.0/dg)
    for (k, Kin, Kout) in [(8, 879, 879), (12, 1006, 1006), (16, 1152, 1024)]:
        key = f'normeq_raw_k{k}_vs_rank'
        if key in res['centpair']: continue
        W = Wnorm_raw(G, k)
        ce_a, mn_a, se_a = eval_hook(mix_hook(W, BM), f'norm-equalized raw k{k}')
        ce_b = torch.cat([fwd_rank(HELD[i:i+B0], Kin, Kout).cpu() for i in range(0, S_, B0)], 0)
        mn_b, se_b = dstat(ce_b)
        d = (ce_a - ce_b).flatten().double()
        mu = float(d.mean()); se = float(d.std()/np.sqrt(d.numel()))
        res['centpair'][key] = {
            'atoms': 'norm_equalized_raw_frobenius', 'k': k, 'compression_x': round(NL/k, 3),
            'shared_dCE': round(mn_a, 4), 'shared_SE': round(se_a, 5),
            'rank_alloc_config': [Kin, Kout], 'rank_alloc_dCE': round(mn_b, 4),
            'rank_alloc_SE': round(se_b, 5),
            'paired_shared_minus_rank': round(mu, 4), 'paired_SE': round(se, 5),
            'z': round(mu/se, 1) if se > 0 else None}
        print(f"  [{key}] shared {mn_a:+.4f} vs rank {mn_b:+.4f} paired {mu:+.4f} "
              f"+- {se:.5f} z={mu/se:.1f}", flush=True)
        dump()
    print("CENTPAIR STAGE DONE", flush=True)

if 'smoke' in STAGES: smoke_stage()
if 'gauge' in STAGES: gauge_stage()
if 'gauge2' in STAGES: gauge2_stage()
if 'refit' in STAGES: refit_stage()
if 'center' in STAGES: center_stage()
if 'outw' in STAGES: outw_stage()
if 'normw' in STAGES: normw_stage()
if 'centpair' in STAGES: centpair_stage()
dump()
print("QK REDTEAM SC DONE", flush=True)
