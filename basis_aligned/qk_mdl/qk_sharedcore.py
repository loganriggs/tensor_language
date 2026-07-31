"""LAYER-MODE SHARING OF THE FOLDED CORES -- do cores SHARED across layers buy the compression
that per-layer rank allocation (RESULTS section 92) and term sparsity (section 104) could not?

Each feed-forward block of bilin18 is exactly a symmetric folded tensor on the normalized input:
    mo_L(h) = T_L(h, h) + bias_L,   T_L[o,i,j] = sum_f Down_L[o,f] * sym(Left_L[f,i] Right_L[f,j]),
with h = rms_norm(x) (the 1/rho^2 gauge folded into the normalization).  T_L has
D * D*(D+1)/2 = 765,075,456 independent coefficients (the section-87/92/104 budget currency);
all 18 blocks live in the SAME residual coordinate system, so they are directly comparable
vectors of ~765M coefficients each.

PART 1 (this script):
  A. SHARING CENSUS (exact, closed-form): the 18x18 Gram matrix G[l,m] = <T_l, T_m> under the
     full-square Frobenius inner product (off-diagonal (i,j)/(j,i) both counted; rotation-
     invariant).  Never materializes the tensors for G: with A=L_l L_m^T, Bm=R_l R_m^T,
     C=L_l R_m^T, E=R_l L_m^T, M=D_l^T D_m (all F x F, F=4608),
         G[l,m] = sum_{f,g} M[f,g] * 0.5*(A*Bm + C*E)[f,g]     (verified on a small case,
     and the trace verified a second way by explicit chunked materialization of each T_l
     along the output dimension -- the chunking checksum).
     Census outputs: eigenvalue spectrum of G (layer-mode rank at 90/95/99% Frobenius energy,
     participation ratio), the cosine matrix (nearest neighbors, early/distributed/readout
     region structure per section 99), and TWO function-weighted variants:
       - per-block input-metric census: T~_l = T_l with both input slots hit by S_l = Sigma_l^{1/2},
         Sigma_l = train second moment of the block's OWN normalized input h_l (closed form via
         Left_l -> Left_l S_l, Right_l -> Right_l S_l).  Diagnostic only (metrics differ per block).
       - pooled input-metric census: same S for every block (S = (mean_l Sigma_l)^{1/2}); a single
         change of metric, so its eigenvectors give a legitimate alternative atom set.
  B. SHARED-ATOM COMPRESSION: layer-mode rank-k approximation.  From G's top-k eigenvectors U_k
     (Gram trick = exact PCA of the 18 x 765M matrix): shared cores C_i = sum_m U[m,i] T_m,
     mixing a_{l,i} = U[l,i], i.e. T^_l = sum_m W[l,m] T_m with W = U_k U_k^T.  Because T^_l is a
     MIXTURE of the original blocks' tensors, the substituted model needs no materialization:
     block l's bilinear part becomes sum_m W[l,m] * bil_m(h_l) where bil_m is block m's bilinear
     evaluated on block l's input.  Per-position held mean tables (uncounted, exactly the
     section-92 MX/MO precedent, held-collected there too) recenter each contribution:
         mo_l(t) = bias_l + BM[l,l](t) + sum_m W[l,m] * (bil_m(h_l) - BM[l,m](t)),
     BM[l,m](t) = held per-position mean of bil_m at layer l's input under the BASE forward.
     W = I reproduces the exact model (GATE, delta cross-entropy ~ 0).
     Budget: k x 765,075,456 + 18k mixing scalars -> compression 18/k.
     Evaluated as whole-model substitution for k in {2,4,8,12,16}, Frobenius atoms AND
     pooled-metric atoms.
  C. RANK-ALLOCATION BASELINE AT MATCHED BUDGETS: scheme-1 (section 92 uniform per-layer
     restricted cores, machinery + bases VERBATIM from qk_rank_alloc_cache.pt) at the same five
     budgets (k/18 of full), three (Kin,Kout) ratio candidates per budget, best taken.
     GATE: the (576,288) anchor must reproduce section 92's +0.8032.

  TRAIN FW[0:256] for all fitting (input second moments), held FW[448:600] for all causal
  numbers, paired standard errors.  Batch<=4 for mixture forwards, 6 elsewhere, <4GB, GPU guard.
  Output: qk_sharedcore.json + qk_sharedcore_cache.pt (G matrices, U, BM, means for part 2)."""
import json, math, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)
DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
OUT = f'{QK}/qk_sharedcore.json'
CPT = f'{QK}/qk_sharedcore_cache.pt'

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
B0M = 4   # mixture-forward batch
B0 = 6    # plain / rank-forward batch
S_, T_ = HELD.shape
STR = TRAIN.shape[0]
FULLBLK = D * D * (D + 1) // 2          # 765,075,456
FULL = NL * FULLBLK                     # 13,771,358,208
REGIONS = {'early': list(range(0, 5)), 'distributed': list(range(5, 12)),
           'readout': list(range(12, 18))}
print(f"bilin18 NL={NL} D={D} F={FDIM} held {S_}x{T_} train {STR}x{T_} "
      f"fullblk {FULLBLK} full {FULL}", flush=True)

def mlp_wts(li):
    b = m.transformer.h[li].mlp
    return (b.Left.weight.detach().float(), b.Right.weight.detach().float(),
            b.Down.weight.detach().float(), b.Down_bias.detach().float())
WTS = [mlp_wts(li) for li in range(NL)]
BIAS = [WTS[li][3] for li in range(NL)]

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res.setdefault('meta', {
    'model': 'bilin18', 'held': 'FW[448:600,:128]', 'train': 'FW[0:256,:128]',
    'batch_mix': B0M, 'batch_rank': B0, 'full_block': int(FULLBLK), 'full_params': int(FULL),
    'inner_product': 'full-square Frobenius over T_l[o,i,j] (symmetric tensor, off-diagonal '
                     'counted twice; rotation-invariant)',
    'accounting': 'shared-atom budget = k * D*D*(D+1)/2 + 18k mixing scalars; scheme-1 budget = '
                  'NL * Kout * Kin*(Kin+1)/2; per-position held mean tables uncounted on BOTH '
                  'sides (section-92 MX/MO precedent, held-collected there as here)'})
res.setdefault('gates', {}); res.setdefault('census', {})
res.setdefault('shared_atom', {}); res.setdefault('scheme1_matched', {})
def dump(): json.dump(res, open(OUT, 'w'), indent=1)

# =====================================================================================
# A0. closed-form Gram formula -- small-case self-test against explicit folding
# =====================================================================================
def closed_gram_pair(Ll, Rl, Dl, Lm, Rm, Dm, chunk=512):
    """<T_l, T_m> full-square Frobenius, exact closed form, fp64 accumulation."""
    tot = torch.zeros((), dtype=torch.float64, device=Ll.device)
    Ff = Ll.shape[0]
    for f0 in range(0, Ff, chunk):
        f1 = min(f0 + chunk, Ff)
        A = Ll[f0:f1] @ Lm.T; Bm = Rl[f0:f1] @ Rm.T
        C = Ll[f0:f1] @ Rm.T; E = Rl[f0:f1] @ Lm.T
        M = Dl.T[f0:f1] @ Dm
        tot += (M.double() * (0.5 * (A.double() * Bm.double() + C.double() * E.double()))).sum()
    return float(tot)

def selftest():
    g = torch.Generator().manual_seed(1)
    d, f = 7, 5
    Ll, Rl = torch.randn(f, d, generator=g).double(), torch.randn(f, d, generator=g).double()
    Lm, Rm = torch.randn(f, d, generator=g).double(), torch.randn(f, d, generator=g).double()
    Dl, Dm = torch.randn(d, f, generator=g).double(), torch.randn(d, f, generator=g).double()
    def fold(Lw, Rw, Dw):
        T = torch.einsum('of,fi,fj->oij', Dw, Lw, Rw)
        return 0.5 * (T + T.transpose(1, 2))
    Tl, Tm = fold(Ll, Rl, Dl), fold(Lm, Rm, Dm)
    exp = float((Tl * Tm).sum())
    got = closed_gram_pair(Ll, Rl, Dl, Lm, Rm, Dm, chunk=3)
    rel = abs(got - exp) / abs(exp)
    print(f"self-test closed-form Gram: explicit {exp:.10e} closed {got:.10e} rel {rel:.2e}", flush=True)
    assert rel < 1e-12, "closed-form Gram self-test FAILED"
selftest()

# =====================================================================================
# A1. the raw Frobenius Gram G, with the two-way chunking checksum
# =====================================================================================
def build_gram(Ls, Rs, Ds, tag):
    G = np.zeros((NL, NL), dtype=np.float64)
    t0 = time.time()
    for l in range(NL):
        for mm in range(l, NL):
            v = closed_gram_pair(Ls[l], Rs[l], Ds[l], Ls[mm], Rs[mm], Ds[mm])
            G[l, mm] = G[mm, l] = v
        print(f"  [{tag}] Gram row {l} done ({time.time()-t0:.0f}s)", flush=True)
    return G

def explicit_sqnorm(li, ochunk=8):
    """||T_l||_F^2 (full-square) by explicit chunked materialization along outputs."""
    Lw, Rw, Dw, _ = WTS[li]
    tot = torch.zeros((), dtype=torch.float64, device=DEV)
    for o0 in range(0, D, ochunk):
        o1 = min(o0 + ochunk, D)
        scaled = Lw.unsqueeze(0) * Dw[o0:o1].unsqueeze(-1)       # (oc,F,D)
        Tc = torch.einsum('ofi,fj->oij', scaled, Rw)
        Tc = 0.5 * (Tc + Tc.transpose(1, 2))
        tot += Tc.double().pow(2).sum()
        del scaled, Tc
    return float(tot)

def explicit_pair01(ochunk=8):
    """<T_0, T_1> by explicit chunked materialization (extra checksum)."""
    L0, R0, D0, _ = WTS[0]; L1, R1, D1, _ = WTS[1]
    tot = torch.zeros((), dtype=torch.float64, device=DEV)
    for o0 in range(0, D, ochunk):
        o1 = min(o0 + ochunk, D)
        def chunkT(Lw, Rw, Dw):
            scaled = Lw.unsqueeze(0) * Dw[o0:o1].unsqueeze(-1)
            Tc = torch.einsum('ofi,fj->oij', scaled, Rw)
            return 0.5 * (Tc + Tc.transpose(1, 2))
        tot += (chunkT(L0, R0, D0).double() * chunkT(L1, R1, D1).double()).sum()
    return float(tot)

cache = {}
if os.path.exists(CPT):
    cache = torch.load(CPT, map_location='cpu', weights_only=True)
    print("cache loaded:", sorted(cache.keys()), flush=True)

if 'G' not in cache:
    print("GRAM: raw Frobenius, closed form ...", flush=True)
    Ls = [w[0] for w in WTS]; Rs = [w[1] for w in WTS]; Ds = [w[2] for w in WTS]
    G = build_gram(Ls, Rs, Ds, 'raw')
    print("GRAM checksum: explicit chunked squared norms ...", flush=True)
    sq = np.array([explicit_sqnorm(li) for li in range(NL)])
    tr_closed, tr_explicit = float(np.trace(G)), float(sq.sum())
    pair01_closed, pair01_explicit = float(G[0, 1]), explicit_pair01()
    print(f"  trace closed {tr_closed:.8e} explicit {tr_explicit:.8e} "
          f"rel {abs(tr_closed-tr_explicit)/tr_explicit:.2e}", flush=True)
    print(f"  pair(0,1) closed {pair01_closed:.8e} explicit {pair01_explicit:.8e}", flush=True)
    assert abs(tr_closed - tr_explicit) / abs(tr_explicit) < 1e-5, "trace checksum FAILED"
    assert abs(pair01_closed - pair01_explicit) / (abs(pair01_explicit) + 1e-30) < 1e-5, \
        "pair(0,1) checksum FAILED"
    cache['G'] = torch.from_numpy(G)
    cache['diag_explicit'] = torch.from_numpy(sq)
    res['gates']['gram_checksum'] = {
        'trace_closed': tr_closed, 'trace_explicit_chunked': tr_explicit,
        'trace_rel_err': abs(tr_closed - tr_explicit) / tr_explicit,
        'pair01_closed': pair01_closed, 'pair01_explicit_chunked': pair01_explicit,
        'per_block_sqnorm_max_rel_err': float(np.max(np.abs(np.diag(G) - sq) / sq))}
    torch.save(cache, CPT); dump()
G = cache['G'].numpy()

# =====================================================================================
# A2. train input second moments Sigma_l (of the normalized MLP input h_l)
# =====================================================================================
@torch.no_grad()
def fwd_track(idx, mode, store):
    """Base forward; mode 'sigma' accumulates train h_l grams; mode 'collect' accumulates
    held per-position sums of bil_m(h_l) for all (l,m) plus h_l sums.  Skeleton VERBATIM
    from qk_termcompress.fwd_terms / section-92 fwd."""
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
        if mode == 'sigma':
            store['Sig'][li] += torch.einsum('btd,bte->de', h, h)
            store['n'][li] += B * T
        elif mode == 'collect':
            for mm in range(NL):
                Lw, Rw, Dw, _ = WTS[mm]
                bl = ((h @ Lw.T) * (h @ Rw.T)) @ Dw.T
                store['BM'][li, mm] += bl.sum(0)
                del bl
            store['H'][li] += h.sum(0)
        x = x + blk.mlp(h)
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce

if 'Sig' not in cache:
    print("TRAIN pass: input second moments Sigma_l ...", flush=True)
    st = {'Sig': torch.zeros(NL, D, D, device=DEV), 'n': [0]*NL}
    for i in range(0, STR, B0): fwd_track(TRAIN[i:i+B0], 'sigma', st)
    Sig = (st['Sig'] / st['n'][0]).cpu()
    cache['Sig'] = Sig
    torch.save(cache, CPT)
    del st; torch.cuda.empty_cache()
Sig = cache['Sig']

def sqrt_psd(Mres):
    w, vecs = torch.linalg.eigh(Mres.double())
    return (vecs * w.clamp_min(0).sqrt().unsqueeze(0)) @ vecs.T

if 'G_w' not in cache or 'G_pool' not in cache:
    print("GRAM: function-weighted variants ...", flush=True)
    Sroot = [sqrt_psd(Sig[li]).float().to(DEV) for li in range(NL)]
    Spool = sqrt_psd(Sig.mean(0)).float().to(DEV)
    Ls = [w[0] for w in WTS]; Rs = [w[1] for w in WTS]; Ds = [w[2] for w in WTS]
    Lw_ = [Ls[li] @ Sroot[li] for li in range(NL)]; Rw_ = [Rs[li] @ Sroot[li] for li in range(NL)]
    cache['G_w'] = torch.from_numpy(build_gram(Lw_, Rw_, Ds, 'per-block-metric'))
    del Lw_, Rw_
    Lp = [Ls[li] @ Spool for li in range(NL)]; Rp = [Rs[li] @ Spool for li in range(NL)]
    cache['G_pool'] = torch.from_numpy(build_gram(Lp, Rp, Ds, 'pooled-metric'))
    del Lp, Rp, Sroot, Spool; torch.cuda.empty_cache()
    torch.save(cache, CPT)

# =====================================================================================
# A3. census numbers from the three Grams
# =====================================================================================
def census(Gm, tag):
    Gm = 0.5 * (Gm + Gm.T)
    w, U = np.linalg.eigh(Gm)
    w = w[::-1].copy(); U = U[:, ::-1].copy()
    wc = np.clip(w, 0, None)
    frac = np.cumsum(wc) / wc.sum()
    ranks = {p: int(np.searchsorted(frac, p) + 1) for p in (0.90, 0.95, 0.99)}
    pr = float(wc.sum()**2 / (wc**2).sum())
    dg = np.sqrt(np.diag(Gm))
    Cos = Gm / np.outer(dg, dg)
    off = Cos - np.eye(NL) * Cos
    nn = [{'block': l, 'nearest': int(np.argmax(np.abs(off[l]))),
           'cosine': float(off[l, np.argmax(np.abs(off[l]))])} for l in range(NL)]
    reg = {}
    for rn, blks in REGIONS.items():
        sub = Cos[np.ix_(blks, blks)]
        within = sub[np.triu_indices(len(blks), 1)]
        other = [b for b in range(NL) if b not in blks]
        across = Cos[np.ix_(blks, other)].flatten()
        ws, Us = np.linalg.eigh(Gm[np.ix_(blks, blks)])
        ws = np.clip(ws[::-1], 0, None)
        fs = np.cumsum(ws) / ws.sum()
        reg[rn] = {'mean_abs_cos_within': float(np.abs(within).mean()),
                   'mean_abs_cos_across': float(np.abs(across).mean()),
                   'rank90': int(np.searchsorted(fs, 0.90) + 1),
                   'rank95': int(np.searchsorted(fs, 0.95) + 1),
                   'n_blocks': len(blks),
                   'eig_frac': [round(float(v), 5) for v in ws / ws.sum()]}
    out = {'eigenvalues': [float(v) for v in w],
           'eig_fraction': [round(float(v), 6) for v in wc / wc.sum()],
           'cum_fraction': [round(float(v), 6) for v in frac],
           'rank_at_90_95_99': ranks, 'participation_ratio': round(pr, 3),
           'block_sqnorms': [float(v) for v in np.diag(Gm)],
           'cosine_matrix': [[round(float(v), 4) for v in row] for row in Cos],
           'nearest_neighbors': nn, 'regions': reg,
           'mean_abs_offdiag_cosine': float(np.abs(off[np.triu_indices(NL, 1)]).mean()),
           'max_abs_offdiag_cosine': float(np.abs(off[np.triu_indices(NL, 1)]).max())}
    print(f"[{tag}] rank90/95/99 {ranks} PR {pr:.2f} "
          f"mean|cos| {out['mean_abs_offdiag_cosine']:.3f} max|cos| {out['max_abs_offdiag_cosine']:.3f}",
          flush=True)
    return out, w, U

res['census']['raw_frobenius'], wG, UG = census(G, 'raw')
res['census']['per_block_input_metric'], _, _ = census(cache['G_w'].numpy(), 'per-block metric')
res['census']['pooled_input_metric'], wP, UP = census(cache['G_pool'].numpy(), 'pooled metric')
dump()

# =====================================================================================
# B0. base cross-entropy + held collect pass (BM mean tables)
# =====================================================================================
cache92 = torch.load(f'{QK}/qk_rank_alloc_cache.pt', map_location='cpu', weights_only=True)
print("BASE: full-model cross-entropy ...", flush=True)
base = torch.cat([fwd_track(HELD[i:i+B0], None, None).cpu() for i in range(0, S_, B0)], 0)
bg = float((base - cache92['base']).abs().max())
print(f"base CE {float(base.mean()):.4f} (cache92 {float(cache92['base'].mean()):.4f}; "
      f"max abs diff {bg:.2e})", flush=True)
res['gates']['base_vs_cache92_maxabs'] = bg
res['meta']['base_ce'] = round(float(base.mean()), 4)
assert bg < 1e-3, "base CE gate FAILED"

def dstat(ce):
    d = (ce - base).flatten().double(); return float(d.mean()), float(d.std()/np.sqrt(d.numel()))

if 'BM' not in cache:
    print("COLLECT: held per-position means BM[l,m] + input means ...", flush=True)
    st = {'BM': torch.zeros(NL, NL, T_, D, device=DEV), 'H': torch.zeros(NL, T_, D, device=DEV)}
    t0 = time.time()
    for i in range(0, S_, B0M): fwd_track(HELD[i:i+B0M], 'collect', st)
    cache['BM'] = (st['BM'] / S_).cpu(); cache['Hmean'] = (st['H'] / S_).cpu()
    torch.save(cache, CPT)
    del st; torch.cuda.empty_cache()
    print(f"collect done ({time.time()-t0:.0f}s)", flush=True)
BM = cache['BM'].to(DEV)   # (NL, NL, T, D) ~191MB

# =====================================================================================
# B1. mixture substitution forward
# =====================================================================================
@torch.no_grad()
def fwd_mix(idx, W):
    """Whole-model substitution: every block's bilinear replaced by the W-mixture of all
    blocks' bilinears, per-position-mean-recentered.  W = I is the exact model."""
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
        mo = (BIAS[li] + BM[li, li]).unsqueeze(0).expand(B, -1, -1).clone()
        for mm in range(NL):
            wlm = float(W[li, mm])
            if abs(wlm) < 1e-12: continue
            Lw, Rw, Dw, _ = WTS[mm]
            bl = ((h @ Lw.T) * (h @ Rw.T)) @ Dw.T
            mo += wlm * (bl - BM[li, mm])
            del bl
        x = x + mo
    logits = 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T-1)
    return ce

def eval_mix(W, tag):
    t0 = time.time()
    ce = torch.cat([fwd_mix(HELD[i:i+B0M], W).cpu() for i in range(0, S_, B0M)], 0)
    mn, se = dstat(ce)
    print(f"  [{tag}] dCE {mn:+.4f} +- {se:.5f} ({time.time()-t0:.0f}s)", flush=True)
    return mn, se

# GATE: k = 18 (identity mixture) must reproduce the exact model
if 'identity_k18' not in res['gates']:
    print("GATE: k=18 identity mixture ...", flush=True)
    mn, se = eval_mix(np.eye(NL), 'k=18 identity')
    res['gates']['identity_k18'] = {'dCE': mn, 'SE': se}
    dump()
    assert abs(mn) < 1e-3, "identity mixture gate FAILED"

def resid_frac(Gm, U, k):
    """per-block and total relative squared Frobenius residual of rank-k layer-mode approx."""
    P = U[:, :k] @ U[:, :k].T
    Rm = (np.eye(NL) - P) @ Gm @ (np.eye(NL) - P)
    per = np.diag(Rm) / np.diag(Gm)
    return float(np.trace(Rm) / np.trace(Gm)), [round(float(v), 5) for v in per]

KS = [16, 12, 8, 4, 2]
for k in KS:
    for vtag, (Gm, U) in {'frobenius': (G, UG), 'pooled_metric': (cache['G_pool'].numpy(), UP)}.items():
        key = f'k{k}_{vtag}'
        if key in res['shared_atom']: continue
        W = U[:, :k] @ U[:, :k].T
        mn, se = eval_mix(W, key)
        tot, per = resid_frac(Gm, U, k)
        bud = k * FULLBLK + NL * k
        res['shared_atom'][key] = {
            'k': k, 'variant': vtag, 'dCE': round(mn, 4), 'SE': round(se, 5),
            'budget': int(bud), 'compression_x': round(FULL / bud, 3),
            'frob_resid_total': round(tot, 5), 'frob_resid_per_block': per,
            'W_diag': [round(float(v), 4) for v in np.diag(W)]}
        dump()

# =====================================================================================
# C. scheme-1 rank allocation at the SAME budgets (section-92 machinery verbatim)
# =====================================================================================
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

INb = [b.to(DEV) for b in cache92['INb']]; OUTb = [b.to(DEV) for b in cache92['OUTb']]
MX = [t.to(DEV) for t in cache92['MX']];   MO = [t.to(DEV) for t in cache92['MO']]

def eval_rank(Kin, Kout, tag):
    PIN = [INb[l][:, :Kin].contiguous() for l in range(NL)]
    POUT = None if Kout >= D else [OUTb[l][:, :Kout].contiguous() for l in range(NL)]
    t0 = time.time()
    ce = torch.cat([fwd_rank(HELD[i:i+B0], PIN=PIN, POUT=POUT, MX=MX, MO=MO).cpu()
                    for i in range(0, S_, B0)], 0)
    mn, se = dstat(ce)
    print(f"  [{tag}] ({Kin},{Kout}): dCE {mn:+.4f} +- {se:.5f} ({time.time()-t0:.0f}s)", flush=True)
    return mn, se

# GATE: section-92 16x anchor
if 'scheme1_anchor_16x' not in res['gates']:
    print("GATE: scheme-1 anchor (576,288) vs section-92 +0.8032 ...", flush=True)
    mn, se = eval_rank(576, 288, 'anchor 16x')
    res['gates']['scheme1_anchor_16x'] = {'dCE': mn, 'SE': se, 'ref_92': 0.8032,
                                          'anchor_ok': bool(abs(mn - 0.8032) < 0.01)}
    dump()
    assert res['gates']['scheme1_anchor_16x']['anchor_ok'], "scheme-1 anchor gate FAILED"

MATCH = {2: [(696, 348), (552, 552), (383, 1152)],
         4: [(876, 438), (696, 696), (542, 1152)],
         8: [(1108, 554), (879, 879), (767, 1152)],
         12: [(1152, 768), (1006, 1006), (940, 1152)],
         16: [(1152, 1024), (1107, 1107), (1086, 1152)]}
for k in KS:
    tgt = k * FULLBLK
    for (Kin, Kout) in MATCH[k]:
        key = f'k{k}_Kin{Kin}_Kout{Kout}'
        if key in res['scheme1_matched']: continue
        bud = NL * min(Kout, D) * Kin * (Kin + 1) // 2
        assert bud <= tgt * 1.001, (key, bud, tgt)
        mn, se = eval_rank(Kin, Kout, key)
        res['scheme1_matched'][key] = {
            'k_matched': k, 'Kin': Kin, 'Kout': int(min(Kout, D)),
            'dCE': round(mn, 4), 'SE': round(se, 5), 'budget': int(bud),
            'budget_target': int(tgt), 'budget_frac_of_target': round(bud / tgt, 4),
            'compression_x': round(FULL / bud, 3)}
        dump()

# save eigen data for part 2
cache['UG'] = torch.from_numpy(UG.copy()); cache['wG'] = torch.from_numpy(wG.copy())
cache['UP'] = torch.from_numpy(UP.copy()); cache['wP'] = torch.from_numpy(wP.copy())
cache['base'] = base
torch.save(cache, CPT)
dump()
print("QK SHAREDCORE PART 1 DONE", flush=True)
