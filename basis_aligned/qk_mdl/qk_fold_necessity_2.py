"""FOLD NECESSITY (swiglu18 part). Apply the SAME architecture-general fold-free surrogate (rank-matched
empirical affine MLP replacement, reduced-rank ridge fit on TRAIN FW[0:256]) to swiglu18 -- the softmax +
gated-SwiGLU twin of bilin18 that CANNOT be folded (no exact bilinear tensor exists for a gated MLP).

swiglu18 forward copied VERBATIM from qk_content_gate_swiglu18.py / qk_atlas_swiglu18.py: softmax
attention (single QK branch), per-head rms+rotary, v1-lerp value path, MLP as black box. Measured on the
same HELD-BACK FW[448:600], batch<=4. There is NO exact-fold arm (unfoldable); we report the generic
surrogate's whole-model dCE + joint MLP floor, and compare its decomposition to bilin18's.
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
m, cfg = load_elriggs('swiglu18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
print(f"swiglu18: NH{NH} HD{HD} D{D} NL{NL} V{V}", flush=True)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256]; HELD = FW[448:600]
COOC = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_cooc_tokens.npy').astype(np.int64))
UNIFORM = float(np.log(V))
RANK_MATCH = 64 * NH; RANKS = [RANK_MATCH, D]
BLKS = [m.transformer.h[i] for i in range(NL)]

# ---- swiglu18 softmax forward (VERBATIM convention) with modes None/floor/surr ----
@torch.no_grad()
def forward(idx, mode, MU=None, WW=None, bb2=None):
    B, T2 = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T2, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T2, T2, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = BLKS[li]; a = blk.attn
        x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T2, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T2, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        sc = (torch.einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)).masked_fill(~mask, float('-inf'))
        pat = F.softmax(sc, -1)
        aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T2, -1))
        x = x + aout
        if mode == 'floor':
            x = x + blk.mlp(MU[li].expand(B, T2, D).to(x.dtype)); continue
        if mode == 'surr':
            h = F.rms_norm(x, (D,)).reshape(-1, D)
            x = x + (h @ WW[li] + bb2[li]).view(B, T2, D).to(x.dtype); continue
        x = x + blk.mlp(F.rms_norm(x, (D,)))
    return 30*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30).float()

# mean MLP-input per layer (for the joint floor), from cooc
hinsum = [torch.zeros(D, device=DEV, dtype=torch.float64) for _ in range(NL)]; hn=[0]
@torch.no_grad()
def collect_mu(idx):
    B, T = idx.shape; x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = BLKS[li]; a = blk.attn
        x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        sc = (torch.einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)).masked_fill(~mask, float('-inf'))
        aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', F.softmax(sc, -1), v).reshape(B, T, -1))
        x = x + aout; hh_ = F.rms_norm(x, (D,)); hinsum[li] += hh_.reshape(-1, D).double().sum(0); x = x + blk.mlp(hh_)
    hn[0] += B*T
for i in range(0, 64, 8):
    collect_mu(COOC[i:i+8].to(DEV)[:, :128])
MU = [F.rms_norm((hinsum[li]/hn[0]).float(), (D,)) for li in range(NL)]

# ---- fit rank-r affine surrogate per layer on TRAIN FW[0:256] (identical recipe to bilin18) ----
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
        blk = BLKS[li]; a = blk.attn
        x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
        hcur = F.rms_norm(x, (D,))
        def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
        q, k = qk(a.c_q), qk(a.c_k)
        sc = (torch.einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)).masked_fill(~mask, float('-inf'))
        aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', F.softmax(sc, -1), v).reshape(B, T, -1))
        x = x + aout; h = F.rms_norm(x, (D,)); y = blk.mlp(h)
        hf = h.reshape(-1, D).double(); yf = y.reshape(-1, D).double()
        Sxx[li] += hf.T @ hf; Sxy[li] += hf.T @ yf; sx[li] += hf.sum(0); sy[li] += yf.sum(0)
        x = x + y
    ntr[0] += B*T
for i in range(0, len(TRAIN), 4):
    collect_train(TRAIN[i:i+4].to(DEV))
    if i % 64 == 0: gpu_guard()
print(f"train covariances collected on {ntr[0]} tokens", flush=True)

Wr = {r: [None]*NL for r in RANKS}; br = {r: [None]*NL for r in RANKS}
for li in range(NL):
    n = ntr[0]; mx = sx[li]/n; my = sy[li]/n
    Cxx = Sxx[li] - n*torch.outer(mx, mx); Cxy = Sxy[li] - n*torch.outer(mx, my)
    ridge = 1e-4 * (torch.diagonal(Cxx).mean())
    Wols = torch.linalg.solve(Cxx + ridge*torch.eye(D, device=DEV, dtype=torch.float64), Cxy)
    G = Wols.T @ Cxx @ Wols
    ev, evec = torch.linalg.eigh((G + G.T)/2); order = ev.argsort(descending=True)
    for r in RANKS:
        Vr = evec[:, order[:r]]; Br = (Wols @ Vr) @ Vr.T
        Wr[r][li] = Br.float(); br[r][li] = (my - mx @ Br).float()
del Sxx, Sxy; torch.cuda.empty_cache()
print("rank-matched affine surrogates fit", flush=True)

@torch.no_grad()
def per_tok(mode, MUx=None, WW=None, bb2=None):
    ces = []
    for i in range(0, len(HELD), 4):
        b = HELD[i:i+4].to(DEV)
        lg = forward(b[:, :-1], mode, MUx, WW, bb2)
        ce = F.cross_entropy(lg.reshape(-1, V), b[:, 1:].reshape(-1), reduction='none')
        ces.append(ce.cpu())
    return torch.cat(ces)

ce_real = per_tok(None); base = float(ce_real.mean())
def rep(ce, ref):
    d = ce - ref; return {'dCE': round(float(d.mean()), 5), 'SE': round(float(d.std()/np.sqrt(d.numel())), 6)}
res = {'model': 'swiglu18', 'base_CE': round(base, 5), 'uniform_ceiling_lnV': round(UNIFORM, 4),
       'n_tokens': int(ce_real.numel()), 'rank_match': RANK_MATCH, 'ranks': RANKS,
       'foldable': False, 'exact_fold': None}
ce_floor = per_tok('floor', MU); res['joint_mlp_floor'] = rep(ce_floor, ce_real)
res['surrogate'] = {}
for r in RANKS:
    res['surrogate'][f'rank{r}'] = rep(per_tok('surr', None, Wr[r], br[r]), ce_real)
print(f"swiglu18 base CE {base:.5f}", flush=True)
for r in RANKS:
    print(f"  surrogate rank{r} dCE +{res['surrogate'][f'rank{r}']['dCE']:.5f} (SE {res['surrogate'][f'rank{r}']['SE']})", flush=True)
print(f"  joint MLP floor dCE +{res['joint_mlp_floor']['dCE']:.5f}", flush=True)

fl = res['joint_mlp_floor']['dCE']; su = res['surrogate'][f'rank{RANK_MATCH}']['dCE']
res['decomposition'] = {
    'floor_dCE': fl, 'surrogate_matched_dCE': su, 'generic_gain': round(fl - su, 5),
    'frac_floor_captured_surrogate': round(1 - su/fl, 5),
}
print(f"  generic surrogate captures {res['decomposition']['frac_floor_captured_surrogate']:.1%} of the MLP floor", flush=True)

# per-layer surrogate representation error (no exact-fold arm; unfoldable)
@torch.no_grad()
def surr_rep():
    sn = {r: [0.0]*NL for r in RANKS}; yn = [0.0]*NL
    for i in range(0, len(HELD), 4):
        idx = HELD[i:i+4].to(DEV)[:, :-1]; B, T = idx.shape
        x0 = F.rms_norm(m.transformer.wte(idx), (D,)); x = None; v1 = None
        cos, sin = rope_tables(T, HD, DEV, x0.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        for li in range(NL):
            blk = BLKS[li]; a = blk.attn
            x = (blk.lambdas[0]+blk.lambdas[1])*x0 if li == 0 else blk.lambdas[0]*x + blk.lambdas[1]*x0
            hcur = F.rms_norm(x, (D,))
            def qk(lin): z = F.rms_norm(lin(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
            v = a.c_v(hcur).view(B, T, NH, HD)
            if v1 is None: v1 = v
            v = (1-a.lamb)*v + a.lamb*v1.view_as(v)
            q, k = qk(a.c_q), qk(a.c_k)
            sc = (torch.einsum('bqhd,bkhd->bhqk', q, k)/(HD**0.5)).masked_fill(~mask, float('-inf'))
            aout = a.c_proj(torch.einsum('bhqk,bkhd->bqhd', F.softmax(sc, -1), v).reshape(B, T, -1))
            x = x + aout; h = F.rms_norm(x, (D,)).reshape(-1, D); y = blk.mlp(h)
            yn[li] += float(y.pow(2).sum())
            for r in RANKS:
                sn[r][li] += float((h @ Wr[r][li] + br[r][li] - y).pow(2).sum())
            x = x + y.view(B, T, D)
    return {r: [ (sn[r][li]/max(yn[li],1e-12))**0.5 for li in range(NL) ] for r in RANKS}
sr = surr_rep()
res['representation_exactness'] = {
    'exact_fold_rel_err_mean': None,
    'surrogate_rel_err_per_layer': {f'rank{r}': [round(v, 5) for v in sr[r]] for r in RANKS},
    'surrogate_rel_err_mean': {f'rank{r}': round(float(np.mean(sr[r])), 5) for r in RANKS},
}
print(f"  surrogate rank{RANK_MATCH} rel err mean {res['representation_exactness']['surrogate_rel_err_mean'][f'rank{RANK_MATCH}']:.3f}", flush=True)

out = {}
try: out = json.load(open(f'{QK}/qk_fold_necessity.json'))
except FileNotFoundError: pass
out['swiglu18'] = res
json.dump(out, open(f'{QK}/qk_fold_necessity.json', 'w'), indent=2)
print("QK FOLD NECESSITY (swiglu18) DONE", flush=True)
