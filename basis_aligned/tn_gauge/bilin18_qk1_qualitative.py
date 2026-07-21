"""Qualitative examples + data validation for the layer-1 QK equivalence classes (Logan 2026-07-21).
F30: layer-1 QK is ~82% current-token-determined -> tokens have QK-1 signatures. Show WHAT those
classes are (decoded tokens) and validate on REAL co-occurrence (attention-weighted pairs that
actually happen). Produces qualitative_examples_qk1.md for Logan.
  (1) cluster the vocab by mean layer-1 QK input signature -> QUERY-side classes (tokens that
      select the same way); decode top tokens per class.
  (2) real attention co-occurrence: sample high layer-1 attention (query pos, key pos) pairs from
      data, decode (query token -> attended token) -- data-validated, only pairs that occur.
GPT-2 tokenizer; Pile data.
"""
import sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot, build_eval_tokens
from transformers import AutoTokenizer
torch.manual_seed(0)
DEV = 'cuda'
OUT = '/workspace/tensor_language/basis_aligned/tn_gauge'
m, cfg = load_elriggs('bilin18')
for p in m.parameters():
    p.requires_grad_(False)
NH, HD, D = cfg['n_head'], cfg['n_embd'] // cfg['n_head'], cfg['n_embd']
Vsz = cfg['vocab_size']
tok = AutoTokenizer.from_pretrained('gpt2')
TOK = build_eval_tokens(n_chunks=48, seq_len=513)[:48]
IDX = TOK[:, :-1].to(DEV)
Bt, T = IDX.shape
rms = lambda x: F.rms_norm(x, (D,))
L = 1


@torch.no_grad()
def capture():
    x = rms(m.transformer.wte(IDX)); x0 = x; v1 = None
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    cos, sin = rope_tables(T, HD, DEV, torch.float32, 'bf16')
    cosr, sinr = cos[None, :, None, :], sin[None, :, None, :]
    qkin = None; pat_L1 = None
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = x; a = blk.attn; h = rms(xin)
        if li == L:
            qkin = h.reshape(-1, D)
        qk = lambda lin: apply_rot(F.rms_norm(lin(h).view(Bt, T, NH, HD), (HD,)), cosr, sinr)
        v = a.c_v(h).view(Bt, T, NH, HD)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        s1 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q), qk(a.c_k)) / HD
        s2 = torch.einsum('bqnh,bknh->bnqk', qk(a.c_q2), qk(a.c_k2)) / HD
        p = (s1 * s2).masked_fill(~mask, 0.0)
        if li == L:
            pat_L1 = p                                    # (B, NH, Tq, Tk)
        x = xin + a.c_proj(torch.einsum('bnqk,bknh->bqnh', p, v).reshape(Bt, T, D))
        x = x + blk.mlp(rms(x))
    return qkin, pat_L1


QKIN, PAT = capture()
tokid = IDX.reshape(-1)
# per-token mean signature (tokens with >=20 occurrences)
sig = torch.zeros(Vsz, D, device=DEV); cnt = torch.zeros(Vsz, device=DEV)
sig.index_add_(0, tokid, QKIN); cnt.index_add_(0, tokid, torch.ones(len(tokid), device=DEV))
freq = cnt >= 20
sig[cnt > 0] /= cnt[cnt > 0][:, None]
toks = torch.nonzero(freq).squeeze(1)
Sig = sig[toks]


def kmeans(X, K, iters=20):
    g = torch.Generator(device='cpu').manual_seed(0)
    C = X[torch.randperm(len(X), generator=g)[:K].to(X.device)].clone()
    for _ in range(iters):
        a = ((X * X).sum(1, True) - 2 * X @ C.T + (C * C).sum(1)[None]).argmin(1)
        Cn = torch.zeros_like(C); ct = torch.zeros(K, device=X.device)
        Cn.index_add_(0, a, X); ct.index_add_(0, a, torch.ones(len(X), device=X.device))
        mm = ct > 0; C[mm] = Cn[mm] / ct[mm][:, None]
    return a, C


K = 40
assign, cent = kmeans(F.normalize(Sig, dim=1), K)
lines = ['# Layer-1 QK equivalence classes — qualitative examples (bilin18)\n']
lines.append(f'{len(toks)} tokens with >=20 occurrences, clustered into {K} classes by their '
             f'layer-1 QK input signature (F30: ~82% of QK-1 is current-token-determined).\n')
lines.append('Each class = tokens that make layer-1 attention select the same way.\n')
# order classes by size, show up to ~20 with example tokens
sizes = torch.bincount(assign, minlength=K)
order = torch.argsort(sizes, descending=True)
for c in order[:24].tolist():
    members = toks[assign == c]
    # sort members by frequency, take top 14
    mf = cnt[members]
    top = members[torch.argsort(mf, descending=True)][:14]
    decoded = [repr(tok.decode([t.item()])) for t in top]
    lines.append(f'- **class {c}** ({int(sizes[c])} tokens): ' + ', '.join(decoded))

# ---- data-validated attention co-occurrence: real (query token -> attended token) pairs ----
lines.append('\n## Real attention co-occurrence (data-validated) — layer-1, top-attended pairs\n')
lines.append('For sampled positions, the token that layer-1 attends to most (summed over heads), '
             'showing only pairs that actually occur in the data.\n')
patsum = PAT.abs().sum(1)                                 # (B, Tq, Tk) over heads
# for a sample of query positions (skip first few), find argmax key
ex = []
gsel = torch.Generator(device='cpu').manual_seed(1)
bsel = torch.randint(0, Bt, (40,), generator=gsel)
qsel = torch.randint(20, T, (40,), generator=gsel)
for b, q in zip(bsel.tolist(), qsel.tolist()):
    row = patsum[b, q, :q + 1]
    kk = int(row.argmax().item())
    qt = tok.decode([IDX[b, q].item()]); kt = tok.decode([IDX[b, kk].item()])
    ex.append(f'- q={repr(qt)}  →  attends to  {repr(kt)}  (offset {q-kk})')
lines += ex[:30]

open(f'{OUT}/qualitative_examples_qk1.md', 'w').write('\n'.join(lines) + '\n')
print(f'wrote qualitative_examples_qk1.md ({len(toks)} tokens, {K} classes, {len(ex)} attention examples)', flush=True)
print('\n--- sample classes ---', flush=True)
for c in order[:8].tolist():
    members = toks[assign == c]
    top = members[torch.argsort(cnt[members], descending=True)][:10]
    print(f'class {c}: ' + ', '.join(repr(tok.decode([t.item()])) for t in top), flush=True)
print('bilin18 qk1 qualitative done', flush=True)
