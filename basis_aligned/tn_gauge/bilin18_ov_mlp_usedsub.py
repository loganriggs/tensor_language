"""Task 3 (Logan 2026-07-21 overnight): repeat the used-subspace reduction for the OV and
BILINEAR circuits, not just query/key. Does the activation-aware used-subspace (F24/F25) find
input-interaction reductions there too? Per circuit, fit on TRAIN, held-out ΔCE on TEST vs
generic low-rank:
  OV   : read = c_v (value), input = rms(xin)          -> compress c_v
  MLP  : reads = [L; R] (bilinear gates), input = rms(x_mid=xin+attn_out) -> compress L,R
whitened used-subspace: W = top-r eigvecs of C^{1/2} R^T R C^{1/2}, P = C^{1/2} W W^T C^{-1/2};
apply W'=W P. Layers 1 and 9. ΔCE via reference_forward.
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
DHID = m.transformer.h[0].mlp.c_fc.weight.shape[0] if hasattr(m.transformer.h[0].mlp, 'c_fc') else None
Vsz = cfg['vocab_size']
ALL = build_eval_tokens(n_chunks=24, seq_len=513)
TRAIN, TEST = ALL[:12], ALL[12:24]
rms = lambda x: F.rms_norm(x, (D,))
LAYERS = [1, 9]
RANKS = [64, 128, 256]
# discover mlp weight attr names
mlp0 = m.transformer.h[0].mlp
print('mlp attrs:', [n for n, _ in mlp0.named_parameters()], flush=True)


@torch.no_grad()
def capture_inputs(tok):
    """per layer: OV/QK input rms(xin), and MLP input rms(x_mid=xin+attn_out)."""
    idx = tok[:, :-1].to(DEV); B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(idx)); x0 = x; v1 = None
    ov_in, mlp_in = {}, {}
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn; h = rms(xin)
        ov_in[li] = h.reshape(-1, D)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(B, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(B, T, D))
        mlp_in[li] = rms(x).reshape(-1, D)
        x = x + blk.mlp(rms(x))
    return ov_in, mlp_in


@torch.no_grad()
def ce(tok):
    idx = tok[:, :-1].to(DEV); tgt = tok[:, 1:].to(DEV)
    return F.cross_entropy(reference_forward(m, idx, 'bf16').float().reshape(-1, Vsz), tgt.reshape(-1)).item()


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm.double()); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T


def used_P(X, R, r):
    C = (X.double().T @ X.double()) / X.shape[0]
    Cs = sym_pow(C, 0.5); Cis = sym_pow(C, -0.5)
    ev, U = torch.linalg.eigh(Cs @ (R.double().T @ R.double()) @ Cs)
    Wz = U[:, -r:]
    return (Cs @ Wz @ Wz.T @ Cis).float()


def lowrank(W, r):
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vh[:r]).float()


OVtr, MLPtr = capture_inputs(TRAIN)
CE0 = ce(TEST)
res = {'baseline_ce_test': round(CE0, 4), 'ov': {}, 'mlp': {}}
print(f'baseline CE(TEST) {CE0:.4f}. used-subspace for OV and MLP (held-out) vs generic low-rank:', flush=True)

# figure out MLP weight names (L,R gates)
mnames = [n for n, _ in mlp0.named_parameters()]
LR = [n.split('.')[0] for n in mnames if 'weight' in n]  # e.g. ['c_fc','c_fc2','c_proj'] or ['w1','w2','w3']

for L in LAYERS:
    # ---- OV: compress c_v ----
    A = m.transformer.h[L].attn
    W0 = A.c_v.weight.data.clone()
    Xov = OVtr[L]
    res['ov'][L] = {}
    print(f'\nlayer {L} OV (c_v): r | used ΔCE | lowrank ΔCE', flush=True)
    for r in RANKS:
        P = used_P(Xov, W0.float(), r)
        A.c_v.weight.data.copy_((W0.float() @ P).to(W0.dtype))
        d_u = ce(TEST) - CE0
        A.c_v.weight.data.copy_(lowrank(W0, r).to(W0.dtype))
        d_l = ce(TEST) - CE0
        A.c_v.weight.data.copy_(W0)
        res['ov'][L][r] = {'used': round(d_u, 4), 'lowrank': round(d_l, 4)}
        print(f'          {r} | {d_u:+.4f} | {d_l:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_ov_mlp_usedsub.json', 'w'), indent=2)

    # ---- MLP: compress the input reads (gate weights that read the residual) ----
    mlp = m.transformer.h[L].mlp
    gate_names = [n for n in ['Left', 'Right', 'c_fc', 'c_fc2', 'w1', 'w2', 'gate_proj', 'up_proj', 'L', 'R']
                  if hasattr(mlp, n)]
    Wg = {n: getattr(mlp, n).weight.data.clone() for n in gate_names}
    R_mlp = torch.cat([Wg[n].float() for n in gate_names], 0)
    Xmlp = MLPtr[L]
    res['mlp'][L] = {'gate_names': gate_names}
    print(f'layer {L} MLP (gates {gate_names}): r | used ΔCE | lowrank ΔCE', flush=True)
    for r in RANKS:
        P = used_P(Xmlp, R_mlp, r)
        for n in gate_names:
            getattr(mlp, n).weight.data.copy_((Wg[n].float() @ P).to(Wg[n].dtype))
        d_u = ce(TEST) - CE0
        for n in gate_names:
            getattr(mlp, n).weight.data.copy_(lowrank(Wg[n], r).to(Wg[n].dtype))
        d_l = ce(TEST) - CE0
        for n in gate_names:
            getattr(mlp, n).weight.data.copy_(Wg[n])
        res['mlp'][L][r] = {'used': round(d_u, 4), 'lowrank': round(d_l, 4)}
        print(f'          {r} | {d_u:+.4f} | {d_l:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_ov_mlp_usedsub.json', 'w'), indent=2)
print('\nbilin18 ov mlp usedsub done', flush=True)
