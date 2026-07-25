"""TICK 202: basis-free structure of the block-0 Bilinear MLP — rank spectra of the
reader-composed channels and eigenfeatures of the composed quadratic forms.

The recon (tick 201) showed the MLP is dense in the neuron basis. The bilinear-layer-
specific tool (Sharkey; Pearce & Dooms) is exact eigendecomposition: for any read
direction u in the residual stream, the MLP's contribution along u is the quadratic
form h^T M_u h with M_u = sum_j (u . d_j) sym(l_j r_j^T) — symmetric, exactly
eigendecomposable from weights alone.

(1) CHANNEL RANK: for each of the 36 layer-1 reader maps (q1/k1/q2/k2 x 9 heads),
    A_H = W_head @ Down (128 x 4608): singular spectrum -> effective rank and r90.
    Low rank here = the MLP->l1-QK channel compresses even though neurons don't.
(2) EIGENFEATURES: for representative readers (l1-h1 k1 = subword giant; l1-h7 q1 =
    determiner reader; l1-h3 k1 = lexical head) and the top-2 singular read directions:
    eigendecomposition of M_u (1152^2), top +/-6 eigenvalues; eigenvector token dumps
    via alignment with (a) rms-normed embeddings and (b) the shrunk mean-residual rows
    (which basis the eigenfeature speaks). Spectrum concentration = how many quadratic
    features carry the channel.
(3) CALIBRATION: whole-block-0-MLP zeroed, full audit (how much the object matters).
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
tok = AutoTokenizer.from_pretrained('gpt2')

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))

blk0 = m.transformer.h[0]
mlp = blk0.mlp
Lw = mlp.Left.weight.detach().float()
Rw = mlp.Right.weight.detach().float()
Dw = mlp.Down.weight.detach().float()
NJ = Lw.shape[0]
a1 = m.transformer.h[1].attn
wte = m.transformer.wte.weight.detach().float().to(DEV)
EMB = F.rms_norm(wte, (D,))
L1TAB = torch.load(f'{QK}/qk_l1_tables.pt', map_location='cpu')

out = {}
# ---- (1) channel rank spectra ----
ranks = {}
for mapname, lin in (('q1', a1.c_q), ('k1', a1.c_k), ('q2', a1.c_q2), ('k2', a1.c_k2)):
    W = lin.weight.detach().float()
    for h in range(NH):
        A = W[h * HD:(h + 1) * HD] @ Dw
        s = torch.linalg.svdvals(A)
        e2 = s ** 2
        eff = float(e2.sum() ** 2 / (e2 ** 2).sum())
        cs = e2.cumsum(0) / e2.sum()
        r90 = int((cs < 0.90).sum()) + 1
        ranks[f'{mapname}_h{h}'] = {'eff_rank': round(eff, 1), 'r90': r90}
effs = [v['eff_rank'] for v in ranks.values()]
r90s = [v['r90'] for v in ranks.values()]
out['channel_rank'] = {'min_eff': min(effs), 'median_eff': float(np.median(effs)),
                       'max_eff': max(effs), 'median_r90': float(np.median(r90s)),
                       'detail': ranks}
print(f'channel effective rank (of 128): min {min(effs):.0f} median '
      f'{np.median(effs):.0f} max {max(effs):.0f}; median r90 {np.median(r90s):.0f}',
      flush=True)
json.dump(out, open(f'{QK}/qk_mlp_eigen.json', 'w'), indent=2)

# ---- (2) eigenfeatures for representative readers ----


def eigenfeatures(mapname, h, n_dirs=2, n_eig=6):
    lin = {'q1': a1.c_q, 'k1': a1.c_k, 'q2': a1.c_q2, 'k2': a1.c_k2}[mapname]
    W = lin.weight.detach().float()[h * HD:(h + 1) * HD]
    A = W @ Dw                                              # (128, 4608)
    U_, S_, Vh = torch.linalg.svd(A, full_matrices=False)
    res = []
    for di in range(n_dirs):
        wj = Vh[di] * S_[di]                                # neuron weights (4608,)
        Lg = Lw.to(DEV)
        Rg = Rw.to(DEV)
        M = Lg.T @ (wj.to(DEV)[:, None] * Rg)
        M = 0.5 * (M + M.T)
        evals, evecs = torch.linalg.eigh(M)
        e2 = evals ** 2
        topfrac = float(e2.sort(descending=True).values[:2 * n_eig].sum() / e2.sum())
        feats = []
        idxs = evals.abs().argsort(descending=True)[:n_eig]
        for ei in idxs.tolist():
            v = evecs[:, ei]
            al_e = EMB @ v
            top = al_e.abs().argsort(descending=True)[:6]
            feats.append({'eval': round(float(evals[ei]), 4),
                          'tokens': [tok.decode([t]).replace('\n', '\\n')
                                     for t in top.tolist()]})
        res.append({'sv': round(float(S_[di]), 3), 'top12_eig_energy': round(topfrac, 3),
                    'eigfeats': feats})
        del Lg, Rg, M
        torch.cuda.empty_cache()
    return res


for mapname, h, label in (('k1', 1, 'l1h1-subword'), ('q1', 7, 'l1h7-determiner'),
                          ('k1', 3, 'l1h3-lexical')):
    ef = eigenfeatures(mapname, h)
    out[f'eig_{label}'] = ef
    print(f'{label} ({mapname}): dir0 top-12-eig energy {ef[0]["top12_eig_energy"]}; '
          f'lead eigvec tokens: {ef[0]["eigfeats"][0]["tokens"][:5]}', flush=True)
    json.dump(out, open(f'{QK}/qk_mlp_eigen.json', 'w'), indent=2)

# ---- (3) whole-MLP calibration ----
BASE = 3.07630


@torch.no_grad()
def audit(batch=4):
    tot, n = 0.0, 0
    for i in range(0, len(FINEWEB), batch):
        b = FINEWEB[i:i + batch].to(DEV)
        idx = b[:, :-1]
        logits = reference_forward(m, idx, 'bf16').float()
        ce = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


backup = mlp.Down.weight.data.clone()
mlp.Down.weight.data = torch.zeros_like(backup)
ce = audit()
mlp.Down.weight.data = backup
out['whole_mlp_zero_dce'] = round(ce - BASE, 5)
print(f'whole block-0 MLP zeroed: dCE {ce - BASE:+.5f}', flush=True)
json.dump(out, open(f'{QK}/qk_mlp_eigen.json', 'w'), indent=2)
print('MLP EIGEN DONE', flush=True)
