"""Layer-1 QK MDL frontier on the flagship (Logan 2026-07-21): the banked baseline he asked
for ('have the MDL values to compare future things against'). Two weight-compression methods on
bilin18 h[1]'s query/key maps (c_q,c_k,c_q2,c_k2, 1152x1152 each), matched-bits, ΔCE binding:
  LOW-RANK r   : rank-r factors, bits = 4 * r*(2D) * 32
  MAGNITUDE-PRUNE keep-f : bits = (f*4D^2)*32 [values] + (f*4D^2)*log2(D^2) [indices]  (side by side)
Gate: full (r=D / keep=1) reproduces the model. Compares against the regime-1 baseline (rotation
= 0 compression, raw 4*D^2*32 bits). ΔCE via reference_forward.
"""
import json, sys, math
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, build_eval_tokens, reference_forward
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
D = cfg['n_embd']; Vsz = cfg['vocab_size']
AUD = build_eval_tokens(n_chunks=10, seq_len=513)[:8]
IDX = AUD[:, :-1].to(DEV); TGT = AUD[:, 1:].to(DEV)
A1 = m.transformer.h[1].attn
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
W0 = {n: getattr(A1, n).weight.data.clone() for n in NAMES}
RAW_BITS = 4 * D * D * 32


@torch.no_grad()
def ce():
    return F.cross_entropy(reference_forward(m, IDX, 'bf16').float().reshape(-1, Vsz), TGT.reshape(-1)).item()


def lowrank(W, r):
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vh[:r]).float()


def prune(W, f):
    k = max(1, int(f * W.numel()))
    thr = W.abs().reshape(-1).topk(k).values.min()
    return torch.where(W.abs() >= thr, W, torch.zeros_like(W))


CE0 = ce()
res = {'baseline_ce': round(CE0, 4), 'raw_Mbit': round(RAW_BITS / 1e6, 2), 'lowrank': {}, 'prune': {}}
print(f'baseline CE {CE0:.4f}; raw QK {RAW_BITS/1e6:.2f} Mbit. layer-1 QK MDL frontier:', flush=True)
print('\nLOW-RANK:', flush=True)
for r in [16, 32, 64, 128, 256, 512, 1152]:
    for n in NAMES:
        getattr(A1, n).weight.data.copy_(lowrank(W0[n], r).to(W0[n].dtype))
    d = ce() - CE0
    bits = 4 * r * (2 * D) * 32
    res['lowrank'][r] = {'dce': round(d, 4), 'Mbit': round(bits / 1e6, 3), 'pct_raw': round(100 * bits / RAW_BITS, 1)}
    print(f'  r={r:4d}: ΔCE {d:+.4f}  {bits/1e6:6.2f} Mbit ({100*bits/RAW_BITS:5.1f}% raw)', flush=True)
    for n in NAMES:
        getattr(A1, n).weight.data.copy_(W0[n])
    json.dump(res, open(f'{OUT}/bilin18_qk1_mdl_frontier.json', 'w'), indent=2)
print('\nMAGNITUDE-PRUNE (structural values + index bits side by side):', flush=True)
idx_bits_per = math.log2(D * D)
for f in [0.5, 0.25, 0.125, 0.0625, 0.03125]:
    for n in NAMES:
        getattr(A1, n).weight.data.copy_(prune(W0[n].float(), f).to(W0[n].dtype))
    d = ce() - CE0
    nnz = f * 4 * D * D
    vbits = nnz * 32; ibits = nnz * idx_bits_per
    res['prune'][f] = {'dce': round(d, 4), 'val_Mbit': round(vbits / 1e6, 3),
                       'idx_Mbit': round(ibits / 1e6, 3), 'total_Mbit': round((vbits + ibits) / 1e6, 3),
                       'pct_raw': round(100 * (vbits + ibits) / RAW_BITS, 1)}
    print(f'  keep {f:.4f}: ΔCE {d:+.4f}  {vbits/1e6:.2f}+{ibits/1e6:.2f}={((vbits+ibits)/1e6):.2f} Mbit '
          f'({100*(vbits+ibits)/RAW_BITS:.1f}% raw)', flush=True)
    for n in NAMES:
        getattr(A1, n).weight.data.copy_(W0[n])
    json.dump(res, open(f'{OUT}/bilin18_qk1_mdl_frontier.json', 'w'), indent=2)
print('\nregime-1 rotation baseline: 0% compression (raw bits); this frontier is what to beat.', flush=True)
print('bilin18 qk1 mdl frontier done', flush=True)
