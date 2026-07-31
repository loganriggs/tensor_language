"""HOW LINEAR CAN bilin18 BE? (Logan) — linearize MLPs and attention as far as possible, then keep only
the small nonlinear remainder.

Exact per-block decomposition:   mlp(x) = [W x + b]  +  r(x)         (r = the nonlinear remainder)
Knob: keep r only in its top-k principal output directions (k=0 -> fully linear, k=1152 -> exact).

MLP stages (all 18 blocks, SEQUENTIAL refit: each block's linear map is fit under the already-linearized
upstream, so every map sees the distribution it will actually face):
    all-linear (k=0), then k in {8, 32, 128}

Attention notions of "linear" (attention cannot be linearized positionwise -- that would destroy mixing --
so the nonlinearity is removed from the PATTERN):
    static   : pattern frozen to its per-(head, i, j) TRAIN mean -> output is linear in the values
    1-branch : pattern (s1*s2) -> s1 * mean(s2), i.e. half the multiplicative degree

Combined: maximally-linear model = all MLPs linear + static attention.
Held FW[448:600,:128], paired standard errors. Gate: exact mode reproduces base CE; per-block linear caps
reproduce qk_degree_ablation.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

torch.manual_seed(0); DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18'); NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV); B0 = 6
T_LEN = TRAIN.shape[1]

LIN = {}          # li -> (W, b)
RES_B = {}        # li -> basis of the nonlinear remainder (D x kmax)
STATIC = {}       # li -> (H, T, T) mean pattern
S2MEAN = {}       # li -> (H, T, T) mean of branch-2 scores


@torch.no_grad()
def fwd(idx, mlp_mode=None, res_k=0, attn_mode=None, upto=None,
        fit_layer=None, acc=None, collect_pat=None):
    """mlp_mode: None | 'linear' (applies to all layers < upto if upto else all)
       res_k: keep top-k directions of the nonlinear remainder (0 = pure linear)
       attn_mode: None | 'static' | '1branch'
       fit_layer: collect (xhat, mo) at this layer into acc; collect_pat: dict to accumulate patterns"""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        hcur = F.rms_norm(x, (D,))
        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        if attn_mode == 'static' and li in STATIC:
            pat = STATIC[li].unsqueeze(0).expand(B, -1, -1, -1)
        else:
            q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            if attn_mode == '1branch' and li in S2MEAN:
                s2 = S2MEAN[li].unsqueeze(0)
            else:
                s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            if collect_pat is not None:
                collect_pat.setdefault(li, [torch.zeros(NH, T, T, device=DEV), 0])
                collect_pat[li][0] += pat.sum(0); collect_pat[li][1] += B
                collect_pat.setdefault(('s2', li), [torch.zeros(NH, T, T, device=DEV), 0])
                collect_pat[('s2', li)][0] += s2.expand(B, -1, -1, -1).sum(0) if s2.shape[0] == 1 else s2.sum(0)
                collect_pat[('s2', li)][1] += B
        yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        xhat = F.rms_norm(x, (D,))
        mo_true = blk.mlp(xhat)
        if fit_layer is not None and li == fit_layer:
            acc.append((xhat.detach(), mo_true.detach()))
        use_lin = mlp_mode == 'linear' and li in LIN and (upto is None or li < upto)
        if use_lin:
            W, b = LIN[li]; mo = xhat @ W + b
            if res_k > 0 and li in RES_B:
                P = RES_B[li][:, :res_k]
                mo = mo + ((mo_true - (xhat @ W + b)) @ P) @ P.T
        else:
            mo = mo_true
        x = x + mo
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T - 1)
    return ce


def held(**kw):
    return torch.cat([fwd(HELD[i:i + B0], **kw) for i in range(0, HELD.shape[0], B0)])


base = held()
BASE = float(base.mean()); print(f'GATE base CE {BASE:.4f}', flush=True)
res = {'meta': {'base_ce': round(BASE, 4), 'held': 'FW[448:600,:128]', 'train': 'FW[0:256,:128]'}}


def rep(name, **kw):
    ce = held(**kw); d = ce - base
    v = (round(float(d.mean()), 4), round(float(d.mean(1).std() / np.sqrt(d.shape[0])), 4))
    print(f'  {name:44s} dCE {v[0]:+.4f} +- {v[1]:.4f}', flush=True)
    return v


# ---------- stage 1: sequential linear fits for all 18 MLPs ----------
print('fitting 18 sequential linear maps (each under the already-linearized upstream) ...', flush=True)
KMAX = 128
for li in range(NL):
    A = torch.zeros(D + 1, D + 1, device=DEV, dtype=torch.float64)
    Bm = torch.zeros(D + 1, D, device=DEV, dtype=torch.float64)
    G = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    for i in range(0, TRAIN.shape[0], B0):
        acc = []
        fwd(TRAIN[i:i + B0], mlp_mode='linear', upto=li, fit_layer=li, acc=acc)
        xh, mo = acc[0]
        xa = torch.cat([xh, torch.ones_like(xh[..., :1])], -1).reshape(-1, D + 1).double()
        A += xa.T @ xa; Bm += xa.T @ mo.reshape(-1, D).double()
    sol = torch.linalg.solve(A + 1.0 * torch.eye(D + 1, device=DEV, dtype=torch.float64), Bm).float()
    LIN[li] = (sol[:D], sol[D])
    # remainder basis
    for i in range(0, TRAIN.shape[0], B0):
        acc = []
        fwd(TRAIN[i:i + B0], mlp_mode='linear', upto=li, fit_layer=li, acc=acc)
        xh, mo = acc[0]
        r = (mo - (xh @ LIN[li][0] + LIN[li][1])).reshape(-1, D).double()
        G += r.T @ r
    RES_B[li] = torch.linalg.eigh(G.float())[1].flip(1)[:, :KMAX].contiguous()
    print(f'  layer {li} fitted', flush=True)

print('=== MLP linearization (all 18 blocks, sequential fits) ===', flush=True)
res['all_mlp_linear'] = rep('all 18 MLPs LINEAR (k=0)', mlp_mode='linear')
for k in (8, 32, 128):
    res[f'all_mlp_linear_k{k}'] = rep(f'all 18 MLPs linear + top-{k} nonlinear dirs', mlp_mode='linear', res_k=k)
json.dump(res, open(f'{QK}/qk_linearize.json', 'w'), indent=1)

# ---------- stage 2: attention ----------
print('collecting static patterns (intact model) ...', flush=True)
cp = {}
for i in range(0, TRAIN.shape[0], B0):
    fwd(TRAIN[i:i + B0], collect_pat=cp)
for li in range(NL):
    STATIC[li] = cp[li][0] / cp[li][1]
    S2MEAN[li] = cp[('s2', li)][0] / cp[('s2', li)][1]
del cp; torch.cuda.empty_cache()

print('=== attention linearization ===', flush=True)
res['attn_static_all'] = rep('all attention STATIC (frozen mean pattern)', attn_mode='static')
res['attn_1branch_all'] = rep('all attention SINGLE-BRANCH (half degree)', attn_mode='1branch')

print('=== combined ===', flush=True)
res['both_linear'] = rep('all MLPs linear + all attention static', mlp_mode='linear', attn_mode='static')
res['both_1branch'] = rep('all MLPs linear + single-branch attention', mlp_mode='linear', attn_mode='1branch')
res['both_k32'] = rep('MLPs linear+top-32 nonlin + 1-branch attn', mlp_mode='linear', res_k=32, attn_mode='1branch')
json.dump(res, open(f'{QK}/qk_linearize.json', 'w'), indent=1)
print('SAVED qk_linearize.json', flush=True)
