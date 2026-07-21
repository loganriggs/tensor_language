"""End-to-end QK compression of the whole model (Logan 2026-07-21): the synthesis of F21-F25.
Apply the activation-aware used-subspace to ALL 18 layers' query/key simultaneously and measure
held-out ΔCE + total bits. Tests whether the per-layer wins compound (compressing every layer's QK
shifts activations; does the used-subspace fit on the original still hold when all are compressed?).
Fit per-layer used-subspace on TRAIN QK inputs, apply W'=W P to c_q/c_k/c_q2/c_k2 of every layer,
held-out ΔCE on TEST at rank r. Gate: r=D reproduces the model. Matched-bits vs raw.
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
RAW = NL * 4 * D * D * 32


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
    idx = tok[:, :-1].to(DEV); tgt = tok[:, 1:].to(DEV); B = idx.shape[0]
    tot = 0.0; n = 0
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


XTR = qk_inputs(TRAIN)
W0 = {li: {n: getattr(m.transformer.h[li].attn, n).weight.data.clone() for n in NAMES} for li in range(NL)}
CE0 = ce_on(TEST)
res = {'baseline_ce_test': round(CE0, 4), 'raw_Mbit': round(RAW / 1e6, 1), 'sweep': {}}
print(f'baseline CE(TEST) {CE0:.4f}; raw all-QK {RAW/1e6:.0f} Mbit. compress ALL {NL} layers QK, held-out:', flush=True)
for r in [64, 128, 256, D]:
    for li in range(NL):
        Rm = torch.cat([W0[li][n].float() for n in NAMES], 0)
        P = used_P(XTR[li], Rm, r)
        for n in NAMES:
            getattr(m.transformer.h[li].attn, n).weight.data.copy_((W0[li][n].float() @ P).to(W0[li][n].dtype))
    dce = ce_on(TEST) - CE0
    bits = NL * (D * r + 4 * r * D) * 32
    res['sweep'][r] = {'dce': round(dce, 4), 'Mbit': round(bits / 1e6, 1), 'pct_raw': round(100 * bits / RAW, 1)}
    print(f'  r={r:4d}: held-out ΔCE {dce:+.4f}   {bits/1e6:6.0f} Mbit ({100*bits/RAW:4.1f}% of raw)', flush=True)
    for li in range(NL):
        for n in NAMES:
            getattr(m.transformer.h[li].attn, n).weight.data.copy_(W0[li][n])
    json.dump(res, open(f'{OUT}/bilin18_allqk_usedsub.json', 'w'), indent=2)
print('bilin18 all-qk usedsub done', flush=True)
