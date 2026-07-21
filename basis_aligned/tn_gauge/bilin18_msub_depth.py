"""Does F22's interpretive-subspace compression GENERALIZE across depth, or is it layer-1
-specific? (Logan 2026-07-21). F22: projecting h[1]'s query/key onto block-0's mlp-output
subspace compresses it ~6x better than generic low-rank (gated M-specific). F19: deep-layer
selection is distributed. Test: for each attention layer L, project L's query/key read maps
onto the top-r principal directions of block(L-1)'s mlp output, ΔCE, vs generic low-rank r on
the same maps. Fixed r=128. If the interpretive-subspace advantage decays with depth -> layer-1
-specific (consistent with F19); if it holds -> a general method. ΔCE via reference_forward.
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
NL = len(m.transformer.h)
Vsz = cfg['vocab_size']
AUD = build_eval_tokens(n_chunks=10, seq_len=513)[:8]
IDX = AUD[:, :-1].to(DEV); TGT = AUD[:, 1:].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
R = 128
TARGETS = [1, 2, 3, 6, 9, 12]


@torch.no_grad()
def capture_mlp_outputs():
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
    mlps = {}
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn; h = rms(xin)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        mo = blk.mlp(rms(x)); x = x + mo
        mlps[li] = mo.reshape(-1, D)
    return mlps


@torch.no_grad()
def ce():
    return F.cross_entropy(reference_forward(m, IDX, 'bf16').float().reshape(-1, Vsz), TGT.reshape(-1)).item()


def pca(X, r):
    _, _, Vh = torch.linalg.svd(X.double() - X.double().mean(0), full_matrices=False)
    return Vh[:r].float()


def lowrank(W, r):
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vh[:r]).float()


MLPS = capture_mlp_outputs()
CE0 = ce()
res = {'baseline_ce': round(CE0, 4), 'r': R, 'by_layer': {}}
print(f'baseline CE {CE0:.4f}; r={R}. project L QK onto block(L-1) mlp-subspace vs generic low-rank:', flush=True)
print('  L | msubspace ΔCE | generic-lowrank ΔCE', flush=True)
for L in TARGETS:
    A = m.transformer.h[L].attn
    W0 = {n: getattr(A, n).weight.data.clone() for n in NAMES}
    U = pca(MLPS[L - 1], R); P = U.T @ U
    for n in NAMES:
        getattr(A, n).weight.data.copy_((W0[n].float() @ P).to(W0[n].dtype))
    d_ms = ce() - CE0
    for n in NAMES:
        getattr(A, n).weight.data.copy_((lowrank(W0[n], R)).to(W0[n].dtype))
    d_lr = ce() - CE0
    for n in NAMES:
        getattr(A, n).weight.data.copy_(W0[n])
    res['by_layer'][L] = {'msub_dce': round(d_ms, 4), 'lowrank_dce': round(d_lr, 4)}
    print(f'  {L:2d} |   {d_ms:+.4f}     |   {d_lr:+.4f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_msub_depth.json', 'w'), indent=2)
wins = sum(1 for L in TARGETS if res['by_layer'][L]['msub_dce'] < res['by_layer'][L]['lowrank_dce'])
res['msub_beats_lowrank_at'] = f'{wins}/{len(TARGETS)}'
json.dump(res, open(f'{OUT}/bilin18_msub_depth.json', 'w'), indent=2)
print(f'\nmsubspace beats generic low-rank at {wins}/{len(TARGETS)} layers (r={R})', flush=True)
print('bilin18 msub depth done', flush=True)
