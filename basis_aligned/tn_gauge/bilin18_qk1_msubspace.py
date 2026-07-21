"""Does the interpretive finding (layer-1 reads the bilinear output M) BUY compression?
(Logan 2026-07-21). F18: bilin18 h[1] query/key selects on block-0's mlp output M. Hypothesis:
a SHARED M-subspace read basis for all 4 query/key matrices should beat 4 INDEPENDENT low-rank
factorizations on bits: M-structured rank-r = U_M (D x r, shared) + 4 read factors (r x D) =
5rD floats; generic low-rank = 4 * r*(2D) = 8rD floats. If ΔCE is comparable, it beats the F21
frontier. Test: project the 4 QK read maps onto the top-r principal directions of M's activations,
ΔCE vs r, bits = 5rD*32; overlay on F21's generic low-rank. Gate: r=D -> ΔCE 0. ΔCE via
reference_forward; matched-bits.
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
AUD = build_eval_tokens(n_chunks=10, seq_len=513)[:8]
IDX = AUD[:, :-1].to(DEV); TGT = AUD[:, 1:].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
A1 = m.transformer.h[1].attn
NAMES = ['c_q', 'c_k', 'c_q2', 'c_k2']
W0 = {n: getattr(A1, n).weight.data.clone() for n in NAMES}
RAW_BITS = 4 * D * D * 32


@torch.no_grad()
def capture():
    """block-0 mlp output M, and h[1]'s actual QK input rms(xin1) -- for the control."""
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
    Mout = None
    for li in range(2):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x
        if li == 1:
            qk_input = rms(xin).reshape(-1, D)          # what h[1] QK actually reads
            return Mout, qk_input
        a = blk.attn; h = rms(xin)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', pat, v).reshape(Bt, T, D))
        Mout = blk.mlp(rms(x)).reshape(-1, D)           # block-0 mlp output = M


@torch.no_grad()
def ce():
    return F.cross_entropy(reference_forward(m, IDX, 'bf16').float().reshape(-1, Vsz), TGT.reshape(-1)).item()


M, QKIN = capture()
def pca(X):
    _, _, Vh = torch.linalg.svd(X.double() - X.double().mean(0), full_matrices=False)
    return Vh.float()
UM = pca(M)                                            # M principal directions
UR = pca(QKIN)                                         # QK-input (residual) principal directions -- CONTROL
CE0 = ce()
res = {'baseline_ce': round(CE0, 4), 'raw_Mbit': round(RAW_BITS / 1e6, 2), 'msub': {}, 'resid_ctrl': {}}
print(f'baseline CE {CE0:.4f}; QK-input projection: M-subspace vs residual-PCA (control):', flush=True)
print('  r | M-subspace ΔCE | residual-PCA ΔCE | Mbit(5rD)', flush=True)
for r in [32, 64, 128, 256, 512, 1152]:
    bits = (D * r + 4 * r * D) * 32                    # shared basis + 4 read factors = 5rD
    row = {}
    for tag, U in [('msub', UM), ('resid_ctrl', UR)]:
        P = U[:r].T @ U[:r]
        for n in NAMES:
            getattr(A1, n).weight.data.copy_((W0[n].float() @ P).to(W0[n].dtype))
        d = ce() - CE0
        res[tag][r] = {'dce': round(d, 4), 'Mbit': round(bits / 1e6, 3), 'pct_raw': round(100 * bits / RAW_BITS, 1)}
        row[tag] = d
        for n in NAMES:
            getattr(A1, n).weight.data.copy_(W0[n])
    print(f'  {r:4d} |   {row["msub"]:+.4f}      |   {row["resid_ctrl"]:+.4f}       | {bits/1e6:.2f}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_qk1_msubspace.json', 'w'), indent=2)

# compare to F21 generic low-rank at matched bits
fr = json.load(open(f'{OUT}/bilin18_qk1_mdl_frontier.json'))
print('\nvs generic low-rank (F21) at comparable bits:', flush=True)
for r in [64, 128, 256]:
    msub = res['msub'][r]
    gen = fr['lowrank'].get(str(r), {})
    print(f'  r={r}: M-subspace ΔCE {msub["dce"]:+.4f} @ {msub["Mbit"]:.1f}Mbit | '
          f'generic ΔCE {gen.get("dce","?")} @ {gen.get("Mbit","?")}Mbit', flush=True)
# verdict: does M-structure beat generic at matched ΔCE?
res['verdict'] = 'see comparison'
json.dump(res, open(f'{OUT}/bilin18_qk1_msubspace.json', 'w'), indent=2)
print('bilin18 qk1 msubspace done', flush=True)
