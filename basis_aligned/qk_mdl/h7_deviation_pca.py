"""WW probe 4: H7's payload is invisible to token-conditional means (WW-4) —
so decompose its per-position OUTPUT DEVIATIONS from the token mean:
capture o_i = (pattern_H7 @ v_H7)_i @ Wo_H7 per position, subtract the
current-token conditional mean, PCA the deviations, logit-lens the top
directions, and report variance explained. Same for H5 as contrast."""
import json
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
from transformers import AutoTokenizer

torch.manual_seed(0)
DEV = 'cuda'
QK = '/workspace/tensor_language/basis_aligned/qk_mdl'
L = 5
HEADS = (7, 5)
OUT = f'{QK}/h7_deviation_pca.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
TOK = build_eval_tokens(n_chunks=128, seq_len=513)[:, :-1]
tok = AutoTokenizer.from_pretrained('gpt2')
Wo = m.transformer.h[L].attn.c_proj.weight.detach().float()
U = m.lm_head.weight.detach().float().to(DEV)

# pass 1: per-token mean of head outputs; pass 2: deviation covariance
acc = {h: torch.zeros(V, D) for h in HEADS}
cnt = torch.zeros(V)
cov = {h: torch.zeros(D, D, device=DEV) for h in HEADS}
ncov = 0


@torch.no_grad()
def head_outputs(idx):
    B, T = idx.shape
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0, v1 = x, None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, x.dtype, 'bf16')
    cosb, sinb = cos[None, :, None, :], sin[None, :, None, :]
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
        q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
        v = a.c_v(h).view(B, T, NH, HD)
        v1 = v if v1 is None else v1
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
        s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
        pat = (s1 * s2).masked_fill(~mask, 0.0)
        if li == L:
            outs = {}
            for hh in HEADS:
                y = torch.einsum('bqk,bkd->bqd', pat[:, hh], v[:, :, hh])
                outs[hh] = y @ Wo[:, hh * HD:(hh + 1) * HD].T.to(DEV)
            return outs
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))


with torch.no_grad():
    for i in range(0, len(TOK), 4):
        idx = TOK[i:i + 4].to(DEV)
        outs = head_outputs(idx)
        flat = idx.reshape(-1).cpu()
        cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
        for h in HEADS:
            acc[h].index_add_(0, flat, outs[h].reshape(-1, D).float().cpu())
    mean = {h: (acc[h] / cnt.clamp_min(1)[:, None]) for h in HEADS}
    for i in range(0, len(TOK), 4):
        idx = TOK[i:i + 4].to(DEV)
        outs = head_outputs(idx)
        for h in HEADS:
            dev = outs[h].reshape(-1, D).float() - mean[h][idx.reshape(-1).cpu()].to(DEV)
            cov[h] += dev.T @ dev
        ncov += idx.numel()

res = {}
for h in HEADS:
    C = cov[h] / ncov
    evals, evecs = torch.linalg.eigh(C)
    evals, evecs = evals.flip(0), evecs.flip(1)
    total = evals.sum().item()
    top = evals[:10] / total
    dirs = evecs[:, :5].T                                   # (5, D)
    lens = F.rms_norm(dirs, (D,)) @ U.T
    ex = {}
    for j in range(5):
        top5p = lens[j].topk(5).indices.cpu().tolist()
        top5n = (-lens[j]).topk(5).indices.cpu().tolist()
        ex[f'dir{j} (+)'] = [repr(tok.decode([w])) for w in top5p]
        ex[f'dir{j} (-)'] = [repr(tok.decode([w])) for w in top5n]
    res[f'H{h}'] = {'var_explained_top10': [round(v.item(), 4) for v in top],
                    'deviation_share_of_total': round(
                        total / (total + mean[h].pow(2).mean().item() * D), 4),
                    'lens_examples': ex}
    print(f'H{h}: top-10 dev-PC var shares {[round(v.item(), 3) for v in top[:5]]}', flush=True)
    print(json.dumps(ex, indent=1), flush=True)
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2)
print('h7 deviation pca done', flush=True)
