"""FOLD NECESSITY (bilin18 part). How much of bilin18's whole-model substitutability REQUIRES the
exact bilinear FOLD (rms-gauge + exactly-multilinear-attention + MLP composed-fold-to-1e-6), versus
how much a GENERIC fold-free surrogate (a rank-matched empirical low-rank affine map, applicable to
ANY architecture) already recovers.

Conventions copied VERBATIM from qk_wholemodel_substitutable.py (the reviewed whole-model harness):
forward with modes None (real) / 'chain' (composed fold on PCA-64/head bottleneck streams) / 'floor'
(mean-input MLPs = joint MLP floor); per-token dCE with paired SE on HELD-BACK FW[448:600].

ADDED: mode 'surr<r>' -- every MLP replaced by a rank-r empirical affine map y ~= h@W + b (reduced-rank
ridge regression) fit on TRAIN FW[0:256] from the module's REAL input h=rms_norm(x) to its REAL output.
This is architecture-general: no gauge identity, no bilinear tensor, no stream reconstruction; the SAME
code runs on swiglu18 (qk_fold_necessity_2.py). Rank matched to the fold's attention bottleneck 64*NH.

Outputs the bilin18 block of qk_fold_necessity.json.
"""
import json, sys, subprocess, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
torch.manual_seed(0)

def gpu_guard(need=4500):
    while True:
        free = int(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits']).decode().split('\n')[0])
        if free >= need: return
        print(f"  gpu guard: {free} MiB free < {need}, sleeping 20s", flush=True); time.sleep(20)
gpu_guard()

DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
print(f"bilin18: NH{NH} HD{HD} D{D} NL{NL} V{V}", flush=True)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256]; HELD = FW[448:600]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
UNIFORM = float(np.log(V))
RANK_MATCH = 64 * NH                 # = fold's attention PCA-64/head bottleneck rank
RANKS = [RANK_MATCH, D]              # matched, and full-rank best-linear upper bound

BLKS = [m.transformer.h[i] for i in range(NL)]
WT = [(b.mlp.Left.weight.detach().float(), b.mlp.Right.weight.detach().float(),
       b.mlp.Down.weight.detach().float(), b.mlp.Down_bias.detach().float()) for b in BLKS]
def T_ev(li, u, v):
    Lw, Rw, Dw, _ = WT[li]
    return 0.5*(((u @ Lw.T) * (v @ Rw.T)) @ Dw.T + ((v @ Lw.T) * (u @ Rw.T)) @ Dw.T)
lam = [(b.lambdas[0].item(), b.lambdas[1].item()) for b in BLKS]
# coefficient recurrence for x_pre_l over streams {e, a_0..a_l, m_0..m_{l-1}} (VERBATIM from harness)
CO = []; cur = {'e': lam[0][0]+lam[0][1]}
for l in range(NL):
    xp = dict(cur); xp[('a', l)] = 1.0; CO.append(xp)
    nx = dict(xp); nx[('m', l)] = 1.0
    if l < NL-1:
        cur = {k: lam[l+1][0]*v for k, v in nx.items()}; cur['e'] = cur.get('e', 0.0)+lam[l+1][1]

# PCA-64/head bases from cooc statistics (disjoint from held slice) -- VERBATIM
acc = [torch.zeros(NH, HD, HD, device=DEV, dtype=torch.float64) for _ in range(NL)]
hinsum = [torch.zeros(D, device=DEV, dtype=torch.float64) for _ in range(NL)]; hn=[0]
@torch.no_grad()
def collect(idx):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        b = BLKS[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        acc[li] += torch.einsum('nhd,nhe->hde', yh.reshape(-1, NH, HD).double(), yh.reshape(-1, NH, HD).double())
        x = x + a.c_proj(yh.reshape(B, T, -1)); hh_=F.rms_norm(x,(D,)); hinsum[li]+=hh_.reshape(-1,D).double().sum(0); x = x + b.mlp(hh_)
    hn[0]+=x.shape[0]*x.shape[1]
for i in range(0, 64, 8):
    collect(COOC[i:i+8].to(DEV)[:, :128])
def make_bases():
    Q = []
    for li in range(NL):
        cw = BLKS[li].attn.c_proj.weight.detach().float(); cs = []
        for hh in range(NH):
            ev, evec = torch.linalg.eigh(acc[li][hh])
            cs.append(cw[:, hh*HD:(hh+1)*HD] @ evec[:, ev.argsort(descending=True)[:64]].float())
        Qx, _ = torch.linalg.qr(torch.cat(cs, 1)); Q.append(Qx)
    return Q
QB = make_bases()
MU = [F.rms_norm((hinsum[li]/hn[0]).float(),(D,)) for li in range(NL)]
print("pca bases + means ready", flush=True)

# ---- fit rank-r empirical affine surrogate per layer on TRAIN FW[0:256] (reduced-rank ridge) ----
# accumulate centered covariances of (h=rms_norm(x) -> y=mlp(h)) for every layer
Sxx = [torch.zeros(D, D, device=DEV, dtype=torch.float64) for _ in range(NL)]
Sxy = [torch.zeros(D, D, device=DEV, dtype=torch.float64) for _ in range(NL)]
sx  = [torch.zeros(D, device=DEV, dtype=torch.float64) for _ in range(NL)]
sy  = [torch.zeros(D, device=DEV, dtype=torch.float64) for _ in range(NL)]
ntr = [0]
@torch.no_grad()
def collect_train(idx):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        b = BLKS[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
        h = F.rms_norm(x, (D,)); y = b.mlp(h)
        hf = h.reshape(-1, D).double(); yf = y.reshape(-1, D).double()
        Sxx[li] += hf.T @ hf; Sxy[li] += hf.T @ yf; sx[li] += hf.sum(0); sy[li] += yf.sum(0)
        x = x + y
    ntr[0] += B*T
for i in range(0, len(TRAIN), 4):
    collect_train(TRAIN[i:i+4].to(DEV))
    if i % 64 == 0: gpu_guard()
print(f"train covariances collected on {ntr[0]} tokens", flush=True)

# reduced-rank ridge -> nested rank-r affine maps
Wr = {r: [None]*NL for r in RANKS}; br = {r: [None]*NL for r in RANKS}
for li in range(NL):
    n = ntr[0]; mx = sx[li]/n; my = sy[li]/n
    Cxx = Sxx[li] - n*torch.outer(mx, mx); Cxy = Sxy[li] - n*torch.outer(mx, my)
    ridge = 1e-4 * (torch.diagonal(Cxx).mean())
    Wols = torch.linalg.solve(Cxx + ridge*torch.eye(D, device=DEV, dtype=torch.float64), Cxy)  # (D,D)
    G = Wols.T @ Cxx @ Wols                      # centered Yhat covariance
    ev, evec = torch.linalg.eigh((G + G.T)/2)
    order = ev.argsort(descending=True)
    for r in RANKS:
        Vr = evec[:, order[:r]]
        Br = (Wols @ Vr) @ Vr.T                  # rank-r reduced-rank map
        Wr[r][li] = Br.float(); br[r][li] = (my - mx @ Br).float()
del Sxx, Sxy; torch.cuda.empty_cache()
print("rank-matched affine surrogates fit", flush=True)


@torch.no_grad()
def forward(idx, mode, WW=None, bb2=None):
    """mode None real | 'chain' composed fold | 'floor' mean-input MLPs | 'surr' rank-r affine MLPs."""
    B, T2 = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    a_list = []; mh = []
    for li in range(NL):
        b = BLKS[li]; a = b.attn
        x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(l): z = F.rms_norm(l(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T2, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
        aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T2, -1))
        x = x + aout
        if mode == 'floor':
            x = x + b.mlp(MU[li].expand(B, T2, D).to(x.dtype)); continue
        if mode == 'surr':
            h = F.rms_norm(x, (D,)).reshape(-1, D)
            x = x + (h @ WW[li] + bb2[li]).view(B, T2, D).to(x.dtype); continue
        if mode == 'chain':
            af = aout.reshape(-1, D); a_list.append(af)
            co = CO[li]; xp = co['e']*x0.reshape(-1, D)
            for j in range(li+1):
                if ('a', j) in co: xp = xp + co[('a', j)]*((a_list[j] @ QB[j]) @ QB[j].T)
                if ('m', j) in co: xp = xp + co[('m', j)]*mh[j]
            r = xp.pow(2).sum(1)/D
            mo = (T_ev(li, xp, xp)/r.unsqueeze(1) + WT[li][3])
            mh.append(mo); x = x + mo.view(B, T2, D).to(x.dtype)
        else:
            x = x + b.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

@torch.no_grad()
def per_tok(mode, WW=None, bb2=None):
    ces = []
    for i in range(0, len(HELD), 4):
        b = HELD[i:i+4].to(DEV)
        lg = forward(b[:, :-1], mode, WW, bb2)
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1), reduction='none')
        ces.append(ce.cpu())
    return torch.cat(ces)

ce_real = per_tok(None); base = float(ce_real.mean())
def rep(ce, ref):
    d = ce - ref; return {'dCE': round(float(d.mean()), 5), 'SE': round(float(d.std()/np.sqrt(d.numel())), 6)}
res = {'model': 'bilin18', 'base_CE': round(base, 5), 'uniform_ceiling_lnV': round(UNIFORM, 4),
       'n_tokens': int(ce_real.numel()), 'rank_match': RANK_MATCH, 'ranks': RANKS}

ce_chain = per_tok('chain'); res['exact_fold'] = rep(ce_chain, ce_real)
ce_floor = per_tok('floor'); res['joint_mlp_floor'] = rep(ce_floor, ce_real)
res['surrogate'] = {}
for r in RANKS:
    ce_s = per_tok('surr', Wr[r], br[r]); res['surrogate'][f'rank{r}'] = rep(ce_s, ce_real)
print(f"bilin18 base CE {base:.5f}", flush=True)
print(f"  exact fold dCE +{res['exact_fold']['dCE']:.5f} (SE {res['exact_fold']['SE']})", flush=True)
for r in RANKS:
    print(f"  surrogate rank{r} dCE +{res['surrogate'][f'rank{r}']['dCE']:.5f} (SE {res['surrogate'][f'rank{r}']['SE']})", flush=True)
print(f"  joint MLP floor dCE +{res['joint_mlp_floor']['dCE']:.5f}", flush=True)

# ---- decomposition (matched rank) ----
fl = res['joint_mlp_floor']['dCE']; ex = res['exact_fold']['dCE']; su = res['surrogate'][f'rank{RANK_MATCH}']['dCE']
res['decomposition'] = {
    'floor_dCE': fl, 'exact_fold_dCE': ex, 'surrogate_matched_dCE': su,
    'exact_fold_total_gain': round(fl - ex, 5),           # headroom recovered by exact fold
    'generic_gain': round(fl - su, 5),                    # headroom recovered by generic surrogate
    'fold_specific_gain': round(su - ex, 5),              # extra the fold buys beyond generic
    'fold_pct_of_total_gain': round(100*(su - ex)/(fl - ex), 3),
    'generic_pct_of_total_gain': round(100*(fl - su)/(fl - ex), 3),
    'frac_floor_captured_exact': round(1 - ex/fl, 5),
    'frac_floor_captured_surrogate': round(1 - su/fl, 5),
}
print(f"  DECOMP: exact-fold gain {fl-ex:.4f} | generic gain {fl-su:.4f} | fold-specific {su-ex:.4f}", flush=True)
print(f"  fold buys {res['decomposition']['fold_pct_of_total_gain']:.2f}% of the substitutability; "
      f"{res['decomposition']['generic_pct_of_total_gain']:.2f}% is architecture-general", flush=True)

# ---- per-layer REPRESENTATION exactness: exact fold (bilinear, ~1e-6) vs surrogate reconstruction ----
@torch.no_grad()
def rep_exactness():
    en = [0.0]*NL; sn = {r: [0.0]*NL for r in RANKS}; yn = [0.0]*NL
    nb = 0
    for i in range(0, len(HELD), 4):
        idx = HELD[i:i+4].to(DEV)[:, :-1]; B, T = idx.shape
        x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
        cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        for li in range(NL):
            b = BLKS[li]; a = b.attn
            x = (b.lambdas[0]+b.lambdas[1])*x0 if li == 0 else b.lambdas[0]*x + b.lambdas[1]*x0
            hcur = F.rms_norm(x, (D,))
            def qk(l): z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            pat = ((torch.einsum('bqhd,bkhd->bhqk', q, k)/HD)*(torch.einsum('bqhd,bkhd->bhqk', q2, k2)/HD)).masked_fill(~mask, 0.0)
            x = x + a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1))
            h = F.rms_norm(x, (D,)).reshape(-1, D); y = b.mlp(h)
            # exact bilinear tensor applied to the RAW pre-rms input (this is the fold identity)
            xr = x.reshape(-1, D); rr = xr.pow(2).sum(1)/D
            yhat_fold = T_ev(li, xr, xr)/rr.unsqueeze(1) + WT[li][3]
            en[li] += float((yhat_fold - y).pow(2).sum()); yn[li] += float(y.pow(2).sum())
            for r in RANKS:
                yhat_s = h @ Wr[r][li] + br[r][li]
                sn[r][li] += float((yhat_s - y).pow(2).sum())
            x = x + y.view(B, T, D)
        nb += 1
    fold_rel = [ (en[li]/max(yn[li],1e-12))**0.5 for li in range(NL) ]
    surr_rel = {r: [ (sn[r][li]/max(yn[li],1e-12))**0.5 for li in range(NL) ] for r in RANKS}
    return fold_rel, surr_rel
fold_rel, surr_rel = rep_exactness()
res['representation_exactness'] = {
    'exact_fold_rel_err_per_layer': [round(v, 8) for v in fold_rel],
    'exact_fold_rel_err_mean': round(float(np.mean(fold_rel)), 8),
    'surrogate_rel_err_per_layer': {f'rank{r}': [round(v, 5) for v in surr_rel[r]] for r in RANKS},
    'surrogate_rel_err_mean': {f'rank{r}': round(float(np.mean(surr_rel[r])), 5) for r in RANKS},
}
print(f"  REPRESENTATION: exact-fold rel err mean {res['representation_exactness']['exact_fold_rel_err_mean']:.2e} "
      f"(strictly-bilinear-only) | surrogate rank{RANK_MATCH} rel err mean "
      f"{res['representation_exactness']['surrogate_rel_err_mean'][f'rank{RANK_MATCH}']:.3f}", flush=True)

out = {}
try: out = json.load(open(f'{QK}/qk_fold_necessity.json'))
except FileNotFoundError: pass
out['bilin18'] = res
json.dump(out, open(f'{QK}/qk_fold_necessity.json', 'w'), indent=2)
print("QK FOLD NECESSITY (bilin18) DONE", flush=True)
