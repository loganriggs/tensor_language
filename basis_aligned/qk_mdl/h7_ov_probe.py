"""WW arc probe 3: WHAT does H7 transport into the hub? Build cond-mean v
content per source token at L5 for heads {7, 5, 0}, map through that head's
c_proj slice, and logit-lens the output: does the forwarded content decode to
the source token itself (copying), and how strongly does it align with the
token's embedding? Crude-lens caveat: 12 more layers process the hub before
the unembedding — treat as directional, with examples."""
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
HEADS = (7, 5, 0)
OUT = f'{QK}/h7_ov_probe.json'
m, cfg = load_elriggs('bilin18')
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
V = cfg['vocab_size']
TRAIN = build_eval_tokens(n_chunks=20 + 1024, seq_len=513)[20:]
tok = AutoTokenizer.from_pretrained('gpt2')

acc = torch.zeros(V, NH, HD)
cnt = torch.zeros(V)
with torch.no_grad():
    for i in range(0, len(TRAIN), 8):
        idx = TRAIN[i:i + 8, :-1].to(DEV)
        B, T = idx.shape
        flat = idx.reshape(-1).cpu()
        cnt.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
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
            v = a.c_v(h).view(B, T, NH, HD)
            v1 = v if v1 is None else v1
            v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
            if li == L:
                acc.index_add_(0, flat, v.reshape(-1, NH, HD).float().cpu())
                break
            qn = lambda lin: apply_rot(F.rms_norm(lin(h).view(B, T, NH, HD), (HD,)), cosb, sinb)
            q, k, q2, k2 = qn(a.c_q), qn(a.c_k), qn(a.c_q2), qn(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / HD
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / HD
            pat = (s1 * s2).masked_fill(~mask, 0.0)
            y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
            x = x + a.c_proj(y)
            x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
seen = cnt > 20                       # frequent tokens only
vbar = acc / cnt.clamp_min(1)[:, None, None]
print(f'{int(seen.sum())} tokens with count>20', flush=True)

Wo = m.transformer.h[L].attn.c_proj.weight.detach().float()   # (D, NH*HD)
E = m.transformer.wte.weight.detach().float()
U = m.lm_head.weight.detach().float()
res = {'n_tokens': int(seen.sum())}
ids = torch.nonzero(seen).squeeze(1)
sample_ids = ids[torch.randperm(len(ids), generator=torch.Generator().manual_seed(3))[:15]]
examples = {}
for h in HEADS:
    Wo_h = Wo[:, h * HD:(h + 1) * HD]                      # (D, HD)
    o = vbar[ids, h].to(DEV) @ Wo_h.T.to(DEV)              # (n, D)
    on = F.rms_norm(o, (D,))
    logits = on @ U.T.to(DEV)                              # crude lens
    top1 = logits.argmax(1).cpu()
    copy_rate = (top1 == ids).float().mean().item()
    ranks = (logits > logits[torch.arange(len(ids)), ids.to(DEV)][:, None]).sum(1).float()
    emb_cos = F.cosine_similarity(o, E[ids].to(DEV), dim=1)
    res[f'H{h}'] = {'lens_top1_is_source': round(copy_rate, 4),
                    'median_rank_of_source': int(ranks.median().item()),
                    'mean_cos_with_embedding': round(emb_cos.mean().item(), 4)}
    print(f'H{h}: lens-top1==source {copy_rate:.3f} · median rank {int(ranks.median())} '
          f'· cos(emb) {emb_cos.mean():.3f}', flush=True)
    ex = {}
    for t in sample_ids.tolist():
        j = (ids == t).nonzero().item()
        top5 = logits[j].topk(5).indices.cpu().tolist()
        ex[repr(tok.decode([t]))] = [repr(tok.decode([w])) for w in top5]
    examples[f'H{h}'] = ex
res['examples'] = examples
with open(OUT, 'w') as fh:
    json.dump(res, fh, indent=2)
print('h7 ov probe done', flush=True)
