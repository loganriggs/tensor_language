"""Used-subspace QK compression: HELD-OUT frontier (Logan 2026-07-21). F24 showed the
activation-aware used-subspace beats generic low-rank at r=128, but the subspace was FIT on the
same tokens ΔCE was measured on -> in-sample (the positive-controls trap). Here: fit the used-
subspace on TRAIN tokens, measure ΔCE on HELD-OUT TEST tokens, full r-frontier vs generic low-
rank (weight-only, no fit). If the near-free compression HOLDS out-of-sample -> real general
method; if it degrades -> F24 was partly overfitting. Layers 1 (single-source) and 9 (distributed).
"""
import json, sys
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
Vsz = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=24, seq_len=513)
TRAIN, TEST = ALL[:12], ALL[12:24]                       # disjoint token windows
rms = lambda x: F.rms_norm(x, (D,))
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
RAW_BITS = 4 * D * D * 32
LAYERS = [1, 9]
RANKS = [16, 32, 64, 128, 256]


@torch.no_grad()
def qk_inputs(tok):
    """QK input activations rms(xin_L) per layer, on the given tokens."""
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
    idx = tok[:, :-1].to(DEV); tgt = tok[:, 1:].to(DEV)
    return F.cross_entropy(reference_forward(m, idx, 'bf16').float().reshape(-1, Vsz), tgt.reshape(-1)).item()


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double()); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T


def used_P(X, R, r):
    C = (X.double().T @ X.double()) / X.shape[0]
    Cis = sym_pow(C, -0.5); Cs = sym_pow(C, 0.5)
    ev, U = torch.linalg.eigh(Cs @ (R.double().T @ R.double()) @ Cs)
    Wz = U[:, -r:]
    return (Cs @ Wz @ Wz.T @ Cis).float()


def lowrank(W, r):
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vh[:r]).float()


XTR = qk_inputs(TRAIN)                                    # fit on TRAIN
CE0 = ce_on(TEST)                                         # baseline on TEST
res = {'baseline_ce_test': round(CE0, 4), 'raw_Mbit': round(RAW_BITS / 1e6, 2), 'layers': {}}
print(f'baseline CE (TEST) {CE0:.4f}; used-subspace fit on TRAIN, ΔCE on HELD-OUT TEST:', flush=True)
for L in LAYERS:
    A = m.transformer.h[L].attn
    W0 = {n: getattr(A, n).weight.data.clone() for n in NAMES}
    R = torch.cat([W0[n].float() for n in NAMES], 0)
    res['layers'][L] = {'used': {}, 'lowrank': {}}
    print(f'\nlayer {L}:  r | USED (held-out) | generic low-rank | used Mbit | lowrank Mbit', flush=True)
    for r in RANKS:
        P = used_P(XTR[L], R, r)
        for n in NAMES:
            getattr(A, n).weight.data.copy_((W0[n].float() @ P).to(W0[n].dtype))
        d_used = ce_on(TEST) - CE0
        for n in NAMES:
            getattr(A, n).weight.data.copy_((lowrank(W0[n], r)).to(W0[n].dtype))
        d_lr = ce_on(TEST) - CE0
        for n in NAMES:
            getattr(A, n).weight.data.copy_(W0[n])
        ub = (D * r + 4 * r * D) * 32; lb = 4 * r * (2 * D) * 32
        res['layers'][L]['used'][r] = {'dce': round(d_used, 4), 'Mbit': round(ub / 1e6, 2)}
        res['layers'][L]['lowrank'][r] = {'dce': round(d_lr, 4), 'Mbit': round(lb / 1e6, 2)}
        print(f'          {r:4d} |   {d_used:+.4f}      |   {d_lr:+.4f}      | {ub/1e6:6.2f}   | {lb/1e6:6.2f}', flush=True)
        json.dump(res, open(f'{OUT}/bilin18_used_frontier.json', 'w'), indent=2)
# held-out verdict
ok = all(res['layers'][L]['used'][128]['dce'] < res['layers'][L]['lowrank'][128]['dce'] for L in LAYERS)
res['held_out_used_beats_lowrank_r128'] = bool(ok)
json.dump(res, open(f'{OUT}/bilin18_used_frontier.json', 'w'), indent=2)
print(f'\nHELD-OUT: used-subspace beats generic low-rank at r=128 both layers: {ok}', flush=True)
print('bilin18 used frontier done', flush=True)
