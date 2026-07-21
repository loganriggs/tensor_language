"""DMRG iteration to fix the compounding (Logan 2026-07-21, realizes the original DMRG vision).
F38: compressing all 18 layers' QK at once with used-subspaces fit on ORIGINAL activations costs
+0.24 held-out at r=128 (vs ~+0.005 per layer) because early-layer compression shifts later
activations. Fix: re-fit each layer's used-subspace on the COMPRESSED model's activations and
iterate to a fixed point (the DMRG sweep). Does ΔCE converge back down toward per-layer? r=128,
held-out. Gate at iter 0 matches F38.
"""
import sys, json
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens, reference_forward
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
NL = len(m.transformer.h)
Vsz = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=24, seq_len=513)
TRAIN, TEST = ALL[:12], ALL[12:24]
rms = lambda x: F.rms_norm(x, (D,))
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
R_ = 128
W0 = {li: {n: getattr(m.transformer.h[li].attn, n).weight.data.clone() for n in NAMES} for li in range(NL)}


@torch.no_grad()
def qk_inputs(tok):
    idx = tok[:, :-1].to(DEV); B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(idx)); x0 = x; v1 = None; out = {}
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn; h = rms(xin); out[li] = h.reshape(-1, D)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(B, T, D))
        x = x + blk.mlp(rms(x))
    return out


@torch.no_grad()
def ce_on(tok):
    idx = tok[:, :-1].to(DEV); tgt = tok[:, 1:].to(DEV); B = idx.shape[0]; tot = 0.0; n = 0
    for i in range(0, B, 4):
        lg = reference_forward(m, idx[i:i+4], 'bf16').float()
        tot += F.cross_entropy(lg.reshape(-1, Vsz), tgt[i:i+4].reshape(-1)).item() * tgt[i:i+4].numel()
        n += tgt[i:i+4].numel()
    return tot / n


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double()); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T


def used_P(X, Rm, r):
    C = (X.double().T @ X.double()) / X.shape[0]
    Cs = sym_pow(C, 0.5); Cis = sym_pow(C, -0.5)
    ev, U = torch.linalg.eigh(Cs @ (Rm.double().T @ Rm.double()) @ Cs)
    return (Cs @ U[:, -r:] @ U[:, -r:].T @ Cis).float()


def reset():
    for li in range(NL):
        for n in NAMES:
            getattr(m.transformer.h[li].attn, n).weight.data.copy_(W0[li][n])


def apply_P(P):
    reset()
    for li in range(NL):
        for n in NAMES:
            getattr(m.transformer.h[li].attn, n).weight.data.copy_((W0[li][n].float() @ P[li]).to(W0[li][n].dtype))


CE0 = ce_on(TEST)
res = {'baseline_ce_test': round(CE0, 4), 'r': R_, 'iters': {}}
print(f'baseline CE(TEST) {CE0:.4f}; DMRG iteration (re-fit used-subspace on compressed activations), r={R_}:', flush=True)
# iter 0: fit on original
reset(); X = qk_inputs(TRAIN)
P = {li: used_P(X[li], torch.cat([W0[li][n].float() for n in NAMES], 0), R_) for li in range(NL)}
apply_P(P)
d = ce_on(TEST) - CE0; res['iters'][0] = round(d, 4)
print(f'  iter 0 (fit on original): held-out ΔCE {d:+.4f}', flush=True)
for it in range(1, 5):
    X = qk_inputs(TRAIN)                                  # activations on the CURRENT compressed model
    P = {li: used_P(X[li], torch.cat([W0[li][n].float() for n in NAMES], 0), R_) for li in range(NL)}
    apply_P(P)
    d = ce_on(TEST) - CE0; res['iters'][it] = round(d, 4)
    print(f'  iter {it} (re-fit on compressed): held-out ΔCE {d:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_dmrg_iter.json', 'w'), indent=2)
reset()
print(f'\nDMRG iteration r={R_}: ΔCE {res["iters"][0]:+.4f} -> {res["iters"][max(res["iters"])]:+.4f} '
      f'(per-layer isolated was ~+0.005)', flush=True)
print('bilin18 dmrg iter done', flush=True)
