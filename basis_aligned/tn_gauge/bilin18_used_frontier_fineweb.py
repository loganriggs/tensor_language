"""Large-data held-out used-subspace frontier on FineWeb (Logan 2026-07-21). F25 fit the covariance
on ~6k Pile tokens. Here: fit on ~200k FineWeb tokens (the model's TRAINING distribution), held-out
ΔCE on disjoint FineWeb, r-frontier vs generic low-rank. Does a much larger covariance estimate +
the training distribution change the frontier? Layers 1 and 9. Also report Pile-fit for comparison.
"""
import sys, json
import numpy as np, torch, torch.nn.functional as F
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
rms = lambda x: F.rms_norm(x, (D,))
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
LAYERS = [1, 9]
RANKS = [64, 128, 256]
fw = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
FW_TRAIN, FW_TEST = fw[:500], fw[500:600]              # 500 fit (~256k tok), 100 held-out


@torch.no_grad()
def qk_input_cov(seqs, layers):
    """accumulate per-layer QK-input covariance over seqs (chunked)."""
    Csum = {li: torch.zeros(D, D, device=DEV, dtype=torch.float64) for li in layers}
    ntok = 0
    for i in range(0, seqs.shape[0], 20):
        idx = seqs[i:i + 20, :-1].to(DEV); B, T = idx.shape
        mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
        cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
        cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
        x = rms(m.transformer.wte(idx)); x0 = x; v1 = None
        for li in range(max(layers) + 1):
            blk = m.transformer.h[li]
            x = blk.lambdas[0] * x + blk.lambdas[1] * x0
            xin = x; a = blk.attn; h = rms(xin)
            if li in layers:
                hf = h.reshape(-1, D); Csum[li] += hf.double().T @ hf.double()
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
        ntok += idx.numel()
    return {li: Csum[li] / ntok for li in layers}, ntok


@torch.no_grad()
def ce_on(seqs):
    tot = 0.0; n = 0
    for i in range(0, seqs.shape[0], 4):
        idx = seqs[i:i + 4, :-1].to(DEV); tgt = seqs[i:i + 4, 1:].to(DEV)
        lg = reference_forward(m, idx, 'bf16').float()
        tot += F.cross_entropy(lg.reshape(-1, Vsz), tgt.reshape(-1)).item() * tgt.numel(); n += tgt.numel()
    return tot / n


def sym_pow(Cm, p, ridge=1e-3):
    ev, U = torch.linalg.eigh(Cm); ev = ev.clamp_min(ridge * ev.max())
    return (U * ev.pow(p)) @ U.T


def used_P(Cn, Rm, r):
    Cs = sym_pow(Cn, 0.5); Cis = sym_pow(Cn, -0.5)
    ev, U = torch.linalg.eigh(Cs @ (Rm.double().T @ Rm.double()) @ Cs)
    return (Cs @ U[:, -r:] @ U[:, -r:].T @ Cis).float()


def lowrank(W, r):
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vh[:r]).float()


print('fitting FineWeb covariance (500 seqs)...', flush=True)
Cfw, ntok = qk_input_cov(FW_TRAIN, LAYERS)
print(f'  fit on {ntok} FineWeb tokens', flush=True)
W0 = {li: {n: getattr(m.transformer.h[li].attn, n).weight.data.clone() for n in NAMES} for li in LAYERS}
CE0 = ce_on(FW_TEST)
res = {'baseline_ce_fw_test': round(CE0, 4), 'fit_tokens': ntok, 'layers': {}}
print(f'baseline CE (FineWeb held-out) {CE0:.4f}; used-subspace (fit {ntok} FineWeb tok) vs generic low-rank:', flush=True)
for L in LAYERS:
    A = m.transformer.h[L].attn
    Rm = torch.cat([W0[L][n].float() for n in NAMES], 0)
    res['layers'][L] = {'used': {}, 'lowrank': {}}
    print(f'\nlayer {L}:  r | USED (held-out) | generic low-rank', flush=True)
    for r in RANKS:
        P = used_P(Cfw[L], Rm, r)
        for n in NAMES:
            getattr(A, n).weight.data.copy_((W0[L][n].float() @ P).to(W0[L][n].dtype))
        du = ce_on(FW_TEST) - CE0
        for n in NAMES:
            getattr(A, n).weight.data.copy_(lowrank(W0[L][n], r).to(W0[L][n].dtype))
        dl = ce_on(FW_TEST) - CE0
        for n in NAMES:
            getattr(A, n).weight.data.copy_(W0[L][n])
        res['layers'][L]['used'][r] = round(du, 4); res['layers'][L]['lowrank'][r] = round(dl, 4)
        print(f'          {r} |   {du:+.4f}      |   {dl:+.4f}', flush=True)
        json.dump(res, open(f'{OUT}/bilin18_used_frontier_fineweb.json', 'w'), indent=2)
print('\nbilin18 used frontier fineweb done', flush=True)
