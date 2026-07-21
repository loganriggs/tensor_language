"""Rank vs CONTEXT LENGTH (Logan 2026-07-21, the other axis). tick-147 varied #tokens at fixed
512-context; here vary the context length (256/512/1024/2048) at ~matched token count, to check
whether longer contexts reveal more QK-input covariance rank (the 512-context estimate could miss
rank that later positions in longer contexts span). bilin18 uses RoPE (no hard cap). Layer-1 QK
input; ~100k tokens each. Pile data (rank was distribution-robust in tick 147)."""
import sys, json
import torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
rms = lambda x: F.rms_norm(x, (D,))
L = 1
TARGET_TOK = 100000


@torch.no_grad()
def qk_input_L1(idx):
    B, T = idx.shape
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    x = rms(m.transformer.wte(idx)); x0 = x; v1 = None
    for li in range(L + 1):
        blk = m.transformer.h[li]
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn; h = rms(xin)
        if li == L:
            return h.reshape(-1, D)
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


def effrank(Cn):
    s = torch.linalg.svdvals(Cn); p = s / s.sum(); return float(torch.exp(-(p * (p + 1e-12).log()).sum()))
def rank_at(Cn, f):
    s = torch.linalg.svdvals(Cn); c = torch.cumsum(s, 0) / s.sum(); return int((c < f).sum()) + 1


res = {'target_tokens': TARGET_TOK, 'by_ctxlen': {}}
print('rank vs context length (layer-1 QK input, ~100k tokens each, Pile):', flush=True)
print('  ctx_len | tokens | eff-rank | rank@90% | rank@99%', flush=True)
for Lctx in [256, 512, 1024, 2048]:
    nseq = TARGET_TOK // Lctx + 2
    bs = max(1, 8192 // Lctx)                          # scale batch to bound attention memory
    seqs = build_eval_tokens(n_chunks=nseq, seq_len=Lctx + 1)[:nseq]
    Csum = torch.zeros(D, D, device=DEV, dtype=torch.float64); ntok = 0
    for i in range(0, nseq, bs):
        idx = seqs[i:i + bs, :-1].to(DEV)
        hf = qk_input_L1(idx)
        Csum += hf.double().T @ hf.double(); ntok += idx.numel()
        if ntok >= TARGET_TOK:
            break
    Cn = Csum / ntok
    er, r90, r99 = effrank(Cn), rank_at(Cn, 0.90), rank_at(Cn, 0.99)
    res['by_ctxlen'][Lctx] = {'tokens': ntok, 'eff_rank': round(er, 1), 'rank90': r90, 'rank99': r99}
    print(f'  {Lctx:5d}   | {ntok:6d} |  {er:6.1f}  |  {r90:4d}   |  {r99:4d}', flush=True)
    json.dump(res, open(f'{OUT}/bilin18_rank_vs_seqlen.json', 'w'), indent=2)
print('\nbilin18 rank vs seqlen done', flush=True)
