"""Task 1 (Logan 2026-07-21 overnight): the composed compression of layer-1 QK's dependence,
folding the bilinear layer. Two reductions: (F28) keep only the ~1024 of 4608 bilinear hidden
units QK reads (input side, weight-only ranking); (F24) project QK weights onto the ~128-dim
used-subspace (weight side). Do they COMPOSE? Measure ΔCE for none / F24-only / F28-only /
composed. If composed ≈ F24+F28, the layer-1 selection folds to {~1024 units -> 128-dim subspace}
with additive cost. Gate: patch-forward = reference. (used-subspace fit in-sample here; F25
already showed it holds held-out.)
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
TOK = build_eval_tokens(n_chunks=10, seq_len=513)[:8]
IDX = TOK[:, :-1].to(DEV); TGT = TOK[:, 1:].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
mlp0 = m.transformer.h[0].mlp
DHID = mlp0.Left.weight.shape[0]
Down = mlp0.Down.weight.data.float()
A1 = m.transformer.h[1].attn
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
W0 = {n: getattr(A1, n).weight.data.clone() for n in NAMES}
R = torch.cat([W0[n].float() for n in NAMES], 0)
KEEPK, RANK = 1024, 128


@torch.no_grad()
def capture():
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(IDX)); x0 = x
    blk = m.transformer.h[0]
    x = blk.lambdas[0] * x + blk.lambdas[1] * x0
    xin = x; a = blk.attn; h = rms(xin)
    qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
    v = a.c_v(h).view(Bt, T, NH, HD)
    s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
    s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
    pat = (s1 * s2).masked_fill(~mask, 0.0)
    x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
    hm = rms(x); hidden = (mlp0.Left(hm) * mlp0.Right(hm))
    x = x + mlp0.Down(hidden)
    blk1 = m.transformer.h[1]
    xin1 = blk1.lambdas[0] * x + blk1.lambdas[1] * x0
    return hidden.reshape(-1, DHID), xin1.reshape(-1, D), float(blk1.lambdas[0].item())


HID, XIN1, lam = capture()
w_only = (R @ Down).norm(dim=0)
keep = torch.topk(w_only, KEEPK).indices
dropped = torch.ones(DHID, device=DEV, dtype=torch.bool); dropped[keep] = False
Mdrop = (HID[:, dropped] @ Down[:, dropped].T)
XIN1_drop = XIN1 - lam * Mdrop


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double()); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T


def used_P(X, r):
    C = (X.double().T @ X.double()) / X.shape[0]
    Cs = sym_pow(C, 0.5); Cis = sym_pow(C, -0.5)
    ev, U = torch.linalg.eigh(Cs @ (R.double().T @ R.double()) @ Cs)
    return (Cs @ U[:, -r:] @ U[:, -r:].T @ Cis).float()


P = used_P(XIN1, RANK)


@torch.no_grad()
def patch_forward(h_qk_L1):
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn
        h_qk = h_qk_L1 if (li == 1 and h_qk_L1 is not None) else rms(xin)
        h_v = rms(xin)
        qkf = lambda lin, hh: apply_rot(F.rms_norm(lin(hh).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        q, k = qkf(a.c_q, h_qk), qkf(a.c_k, h_qk); q2, k2 = qkf(a.c_q2, h_qk), qkf(a.c_k2, h_qk)
        v = a.c_v(h_v).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', q, k) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        x = x + blk.mlp(rms(x))
    return 30 * torch.tanh(m.lm_head(rms(x)) / 30)


def ce(h_qk_L1):
    return F.cross_entropy(patch_forward(h_qk_L1).float().reshape(-1, Vsz), TGT.reshape(-1)).item()


def set_qk(project):
    for n in NAMES:
        getattr(A1, n).weight.data.copy_((W0[n].float() @ P if project else W0[n].float()).to(W0[n].dtype))


full = XIN1.reshape(Bt, T, D); drop = XIN1_drop.reshape(Bt, T, D)
CE0 = ce(rms(full))
CEref = F.cross_entropy(reference_forward(m, IDX, 'bf16').float().reshape(-1, Vsz), TGT.reshape(-1)).item()
print(f'gate {CE0:.4f} vs ref {CEref:.4f} (Δ {abs(CE0-CEref):.1e})', flush=True)
res = {'baseline_ce': round(CE0, 4), 'keep_units': KEEPK, 'used_rank': RANK, 'arms': {}}
set_qk(False); res['arms']['F28_keep1024_units'] = round(ce(rms(drop)) - CE0, 4)
set_qk(True); res['arms']['F24_usedsubspace_r128'] = round(ce(rms(full)) - CE0, 4)
set_qk(True); res['arms']['composed_both'] = round(ce(rms(drop)) - CE0, 4)
set_qk(False)
res['arms']['sum_of_individual'] = round(res['arms']['F28_keep1024_units'] + res['arms']['F24_usedsubspace_r128'], 4)
json.dump(res, open(f'{OUT}/bilin18_composed.json', 'w'), indent=2)
for kk, vv in res['arms'].items():
    print(f'  {kk}: ΔCE {vv:+.4f}', flush=True)
print(f"\ncomposed {res['arms']['composed_both']:+.4f} vs sum {res['arms']['sum_of_individual']:+.4f} "
      f"-> {'COMPOSE additively' if abs(res['arms']['composed_both']-res['arms']['sum_of_individual'])<0.02 else 'not additive'}", flush=True)
print('bilin18 composed done', flush=True)
