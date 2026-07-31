"""DEGREE ABLATION (Logan): does the model need the polynomial DEGREE its bilinear blocks provide,
or only the token-specific information they carry?

Each bilinear block computes a pure quadratic of its input. Cap the degree by replacing a block's
output with the best LINEAR function of the same input (ridge least squares, fit on TRAIN), and
compare against two other substitutes that hold different things fixed:

    substitute            degree in stream   token information
    ------------------------------------------------------------------
    linear cap            1                  linear only
    token table (blk 0)   -                  ARBITRARY function of token, no context
    per-position mean     0                  none                      <- the floor

If "degree" is what matters, the linear cap should be much worse than the token table.
If "token-specific information" is what matters, the reverse.

Also: does the early stack still BUILD the next-token category code when linearized? Fresh 6-way
linear probe on the residual after block 3 under each condition (fit TRAIN, eval HELD).

Held FW[448:600,:128], paired standard errors. Gates: full-model mode reproduces base CE exactly;
block-0 mean-ablation reproduces the known floor 1.2341; the token table reproduces ~85.5% recovery.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot
from transformers import AutoTokenizer

torch.manual_seed(0); DEV = 'cuda'; QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
m, cfg = load_elriggs('bilin18'); NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']; NL = len(m.transformer.h)
FW = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN = FW[0:256, :128].to(DEV); HELD = FW[448:600, :128].to(DEV); B0 = 6
TARGETS = [0, 1, 3, 5, 7]
tok = AutoTokenizer.from_pretrained('gpt2')


@torch.no_grad()
def fwd(idx, sub=None, collect_in=None, collect_res=None):
    """sub: {li: ('linear',W,b) | ('table',Tbl) | ('mean',M)}. collect_in: list of li to return (xhat, mo).
       collect_res: li -> also return the residual stream AFTER that block."""
    B, T = idx.shape
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16'); cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    got = {}; res_out = None
    for li in range(NL):
        blk = m.transformer.h[li]; x = blk.lambdas[0] * x + blk.lambdas[1] * x0; a = blk.attn
        hcur = F.rms_norm(x, (D,))
        def qk(l):
            z = F.rms_norm(l(hcur).view(B, T, NH, HD), (HD,)); return apply_rot(z, cosb, sinb)
        v = a.c_v(hcur).view(B, T, NH, HD)
        if v1 is None: v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        q, k, q2, k2 = qk(a.c_q), qk(a.c_k), qk(a.c_q2), qk(a.c_k2)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD; s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0); yh = torch.einsum('bhqk,bkhd->bqhd', pat, v)
        x = x + a.c_proj(yh.reshape(B, T, -1))
        xhat = F.rms_norm(x, (D,))
        mo = blk.mlp(xhat)
        if collect_in is not None and li in collect_in:
            got[li] = (xhat.clone(), mo.clone())
        if sub is not None and li in sub:
            kind = sub[li][0]
            if kind == 'linear':
                W, b = sub[li][1], sub[li][2]; mo = xhat @ W + b
            elif kind == 'table':
                mo = sub[li][1][idx]
            elif kind == 'mean':
                mo = sub[li][1].unsqueeze(0).expand(B, -1, -1)
        x = x + mo
        if collect_res is not None and li == collect_res:
            res_out = x.clone()
    logits = 30 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(), idx[:, 1:].reshape(-1),
                         reduction='none').view(B, T - 1)
    return ce, got, res_out


def held_ce(sub=None):
    out = []
    for i in range(0, HELD.shape[0], B0):
        ce, _, _ = fwd(HELD[i:i + B0], sub=sub); out.append(ce)
    return torch.cat(out)


print('collecting TRAIN activations + fitting linear caps ...', flush=True)
A = {l: torch.zeros(D + 1, D + 1, device=DEV, dtype=torch.float64) for l in TARGETS}
Bm = {l: torch.zeros(D + 1, D, device=DEV, dtype=torch.float64) for l in TARGETS}
tbl_sum = torch.zeros(V, D, device=DEV); tbl_cnt = torch.zeros(V, device=DEV)
for i in range(0, TRAIN.shape[0], B0):
    idx = TRAIN[i:i + B0]
    _, got, _ = fwd(idx, collect_in=TARGETS)
    for l in TARGETS:
        xh, mo = got[l]
        xa = torch.cat([xh, torch.ones_like(xh[..., :1])], -1).reshape(-1, D + 1).double()
        A[l] += xa.T @ xa; Bm[l] += xa.T @ mo.reshape(-1, D).double()
        if l == 0:
            flat = idx.reshape(-1)
            tbl_sum.index_add_(0, flat, mo.reshape(-1, D)); tbl_cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
LIN = {}
for l in TARGETS:
    sol = torch.linalg.solve(A[l] + 1.0 * torch.eye(D + 1, device=DEV, dtype=torch.float64), Bm[l]).float()
    LIN[l] = (sol[:D], sol[D])
TBL = tbl_sum / tbl_cnt.clamp(min=1)[:, None]
gmean = TBL[tbl_cnt > 0].mean(0)
TBL[tbl_cnt == 0] = gmean
print(f'  token table coverage on TRAIN: {(tbl_cnt>0).float().mean():.3f} of vocab', flush=True)

print('per-position means (held) ...', flush=True)
S, T = HELD.shape
msum = {l: torch.zeros(T, D, device=DEV) for l in TARGETS}
for i in range(0, S, B0):
    _, got, _ = fwd(HELD[i:i + B0], collect_in=TARGETS)
    for l in TARGETS: msum[l] += got[l][1].sum(0)
MEAN = {l: msum[l] / S for l in TARGETS}

base = held_ce(None)
print(f'GATE base CE {base.mean():.4f}', flush=True)
res = {'meta': {'base_ce': round(float(base.mean()), 4), 'held': 'FW[448:600,:128]', 'train': 'FW[0:256,:128]',
                'linear_fit': 'ridge least squares (lambda=1) on TRAIN, predicting the block output from its rms-normed input'}}


def report(name, sub):
    ce = held_ce(sub); d = (ce - base)
    v = (round(float(d.mean()), 4), round(float(d.mean(1).std() / np.sqrt(d.shape[0])), 4))
    print(f'  {name:34s} dCE {v[0]:+.4f} +- {v[1]:.4f}', flush=True)
    return v


print('=== block-level degree ablation ===', flush=True)
cells = {}
for l in TARGETS:
    cells[f'L{l}_mean_floor'] = report(f'block {l}: per-position mean (floor)', {l: ('mean', MEAN[l])})
    cells[f'L{l}_linear'] = report(f'block {l}: LINEAR cap', {l: ('linear', *LIN[l])})
    if l == 0:
        cells['L0_table'] = report('block 0: token table', {0: ('table', TBL)})
res['block_cells'] = cells

print('=== early-stack joint (blocks 0-3) ===', flush=True)
joint_lin = {l: ('linear', *LIN[l]) for l in [0, 1, 3] if l in LIN}
res['joint_linear_0_1_3'] = report('blocks 0,1,3 all LINEAR', joint_lin)
res['joint_mean_0_1_3'] = report('blocks 0,1,3 all mean-ablated', {l: ('mean', MEAN[l]) for l in [0, 1, 3]})

# ---- capability: next-token category probe on the residual after block 3 ----
print('=== category-code capability probe (residual after block 3) ===', flush=True)
strs = [tok.decode([i]) for i in range(V)]


def cat_of(s):
    t = s.strip()
    if t == '': return 5
    if t.isdigit(): return 3
    if not any(c.isalnum() for c in t): return 4
    if t[:1].isupper(): return 2
    return 0 if s.startswith(' ') else 1


CAT = torch.tensor([cat_of(s) for s in strs], device=DEV)


def probe_acc(sub):
    XT, YT = [], []
    for i in range(0, TRAIN.shape[0], B0):
        idx = TRAIN[i:i + B0]; _, _, r = fwd(idx, sub=sub, collect_res=3)
        XT.append(F.rms_norm(r, (D,))[:, :-1].reshape(-1, D)); YT.append(CAT[idx[:, 1:].reshape(-1)])
    X = torch.cat(XT).double(); Y = torch.cat(YT)
    Xa = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    Yh = F.one_hot(Y, 6).double()
    Wp = torch.linalg.solve(Xa.T @ Xa + 1.0 * torch.eye(D + 1, device=DEV, dtype=torch.float64), Xa.T @ Yh)
    corr = tot = 0
    for i in range(0, HELD.shape[0], B0):
        idx = HELD[i:i + B0]; _, _, r = fwd(idx, sub=sub, collect_res=3)
        Xh = F.rms_norm(r, (D,))[:, :-1].reshape(-1, D).double()
        Xh = torch.cat([Xh, torch.ones_like(Xh[:, :1])], 1)
        pred = (Xh @ Wp).argmax(1); y = CAT[idx[:, 1:].reshape(-1)]
        corr += int((pred == y).sum()); tot += len(y)
    return round(corr / tot, 4)


res['probe'] = {'intact': probe_acc(None),
                'blocks_0_1_3_linear': probe_acc(joint_lin),
                'block0_table': probe_acc({0: ('table', TBL)}),
                'blocks_0_1_3_mean': probe_acc({l: ('mean', MEAN[l]) for l in [0, 1, 3]})}
print('  probe accuracies:', res['probe'], flush=True)

json.dump(res, open(f'{QK}/qk_degree_ablation.json', 'w'), indent=1)
print('SAVED qk_degree_ablation.json', flush=True)
