"""TICK 190 (Logan): causal per-archetype ablation on real text — "which documents is
each archetype most useful for?"

Ablation: an archetype's key channel is a direction per branch (g1 = D_k1^T a_r,
g2 = D_k2^T b_r, unit-normalized, from the DISPLAYED inventories: minimal dictionaries
for the seven scaffold heads, polished mode dictionaries for heads 0/4). Ablating
archetype r of head h projects those directions out of head h's exact key tables:
  k1[t] -= (k1[t].g1) g1,   k2[t] -= (k2[t].g2) g2   (all tokens t)
— a structured zero: the head loses the ability to score that class channel, everything
else intact (chosen over mean-ablation: the pattern is a product, and the class channel
being *absent* is the counterfactual our decomposition defines; scores from patched
tables recompute rotary exactly via scores_from_factors).

Scoring: 64 held-out FineWeb sequences (32,704 predictions), per-position cross-entropy
delta vs the unpatched model. Per archetype (top 10 by extraction order per head, 90
total): mean dCE (usefulness on ordinary text) and the 6 hardest-hit positions with
surrounding text, for the artifact's "where it matters" panels.
"""
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, reference_forward
from tier2_folding import branch_factors, scores_from_factors
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
N_SEQ, TOP_R, TOP_POS = 64, 10, 6
tok = AutoTokenizer.from_pretrained('gpt2')

m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
TAB = {}
for br, (qn, kn) in ((1, ('q1', 'k1')), (2, ('q2', 'k2'))):
    qh, kh = branch_factors(m, br)
    TAB[qn], TAB[kn] = qh.float().to(DEV), kh.float().to(DEV)
FINEWEB = torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
SEQS = FINEWEB[:N_SEQ]


@torch.no_grad()
def per_pos_loss(tabs, batch=4):
    outs = []
    for i in range(0, N_SEQ, batch):
        b = SEQS[i:i + batch].to(DEV)
        idx = b[:, :-1]

        def patch(li, s1, s2):
            if li != 0:
                return s1, s2
            n1 = scores_from_factors(tabs['q1'], tabs['k1'], idx, HD)
            n2 = scores_from_factors(tabs['q2'], tabs['k2'], idx, HD)
            return n1.to(s1.dtype), n2.to(s2.dtype)

        logits = reference_forward(m, idx, 'bf16',
                                   score_patch=None if tabs is None else patch).float()
        ls = F.cross_entropy(logits.reshape(-1, V), b[:, 1:].reshape(-1), reduction='none')
        outs.append(ls.view(b.shape[0], -1).cpu())
    return torch.cat(outs, 0)                                 # (N_SEQ, T-1)


base = per_pos_loss(None)
print(f'baseline mean CE {float(base.mean()):.4f}', flush=True)

mh_pt = torch.load(f'{QK}/qk_minimal_heads.pt', map_location=DEV)
mh_js = json.load(open(f'{QK}/qk_minimal_heads.json'))
polish = {0: torch.load(f'{QK}/qk_h0_polish_g025.pt', map_location=DEV),
          4: torch.load(f'{QK}/qk_h04_polish.pt', map_location=DEV)}


def detectors(h, r):
    """Unit key-channel directions (g1, g2) in head space for archetype r of head h."""
    if h in (0, 4):
        bb = polish[h]
        D1 = bb[f'h{h}_k1_Dm'].to(DEV)
        D2 = bb[f'h{h}_k2_Dm'].to(DEV)
        D1 = D1 / D1.norm(dim=1, keepdim=True).clamp_min(1e-8)
        D2 = D2 / D2.norm(dim=1, keepdim=True).clamp_min(1e-8)
        g1 = D1.T @ bb[f'h{h}_AJ'][:, r].to(DEV)
        g2 = D2.T @ bb[f'h{h}_BJ'][:, r].to(DEV)
    else:
        P = mh_pt[f'h{h}']
        Dn = P['Dm'].to(DEV)
        Dn = Dn / Dn.norm(dim=1, keepdim=True).clamp_min(1e-8)
        U = P['U'].to(DEV)
        g1 = Dn[:, :HD].T @ U[:, r]
        g2 = Dn[:, HD:2 * HD].T @ U[:, r]
    return (g1 / g1.norm().clamp_min(1e-12), g2 / g2.norm().clamp_min(1e-12))


def snippet(i, p):
    ctx = tok.decode(SEQS[i, max(0, p - 14):p + 1].tolist())
    tgt = tok.decode([int(SEQS[i, p + 1])])
    return ctx.replace('\n', '⏎'), tgt.replace('\n', '⏎')


out = {}
HEADS = [1, 2, 3, 5, 6, 7, 8, 0, 4]
for h in HEADS:
    n_arch = len(polish[h][f'h{h}_lamJ']) if h in (0, 4) else mh_pt[f'h{h}']['U'].shape[1]
    rows = []
    for r in range(min(TOP_R, n_arch)):
        g1, g2 = detectors(h, r)
        tabs = {k: v.clone() for k, v in TAB.items()}
        tabs['k1'][:, h] -= (tabs['k1'][:, h] @ g1)[:, None] * g1[None, :]
        tabs['k2'][:, h] -= (tabs['k2'][:, h] @ g2)[:, None] * g2[None, :]
        la = per_pos_loss(tabs)
        delta = (la - base)
        mean_dce = float(delta.mean())
        flat = delta.flatten()
        top = flat.topk(TOP_POS).indices
        exs = []
        for t in top.tolist():
            i, p = t // delta.shape[1], t % delta.shape[1]
            ctx, tgt = snippet(i, p)
            exs.append({'ctx': ctx, 'tgt': tgt, 'dce': round(float(flat[t]), 3)})
        rows.append({'r': r, 'mean_dce': round(mean_dce, 5), 'top': exs})
        print(f'h{h} r{r}: mean dCE {mean_dce:+.5f} | worst {exs[0]["dce"]:+.2f} '
              f'"...{exs[0]["ctx"][-30:]}" -> "{exs[0]["tgt"]}"', flush=True)
        del tabs, la, delta
        torch.cuda.empty_cache()
    out[f'h{h}'] = rows
    json.dump(out, open(f'{QK}/qk_arch_ablation.json', 'w'), indent=1)
print('ARCH ABLATION DONE', flush=True)
