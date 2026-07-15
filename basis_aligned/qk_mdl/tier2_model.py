"""Tier-2 target models (Logan's Elriggs checkpoints) — loader, fp-controlled
reference forward, and the CE sanity gate.

Verified from jacclust/tt_model.py + twoqk.py (prior program, re-verified here):
  - gpt2-bilinear-sqrd-attn-18l-9h-1152embd (546M): TWO QK branches; pattern =
    (q1.k1)(q2.k2)/D^2, causal-masked, UNNORMALIZED (row-normalizing was only
    the single-branch variant's trained behavior). Bilinear MLPs.
  - gpt2-sqrd-attn-12l-6h-768embd (162M): ONE branch; pattern = (q.k/D)^2,
    causal-masked, ROW-NORMALIZED (sum-to-1 per query => per-query positive
    scale gauge EXISTS for this model, unlike the tiny models). ReLU^2 MLP.
  Common: token embed -> global rms_norm -> x0 skip; per block
  x = l0*x + l1*x0; attn input rms_norm; per-head QK rms_norm THEN RoPE
  (convention y1 = x1 c + x2 s, y2 = -x1 s + x2 c — opposite rotation sign to
  the tiny models); v mixed with block-0 v via lamb; final rms_norm; logits
  soft-capped 30*tanh(./30). Rotary tables in the source are computed in bf16;
  the reference forward exposes table_dtype = 'bf16' (deployed) or 'exact'.
"""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language')
from huggingface_hub import hf_hub_download
import jacclust.tt_model as TT

REPOS = {'bilin18': 'Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd',
         'sqrd12': 'Elriggs/gpt2-sqrd-attn-12l-6h-768embd'}


def load_elriggs(short, device='cuda', dtype=torch.float32):
    repo = REPOS[short]
    cfg = json.load(open(hf_hub_download(repo, 'config.json')))
    cfg.pop('step', None)
    m = TT.GPT(TT.GPTConfig(**cfg)).to(device=device, dtype=dtype).eval()
    m.load_state_dict(torch.load(hf_hub_download(repo, 'pytorch_model.bin'),
                                 map_location=device, weights_only=True))
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg


def rope_tables(T, head_dim, device, dtype, table_dtype='bf16'):
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(T, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    cos, sin = freqs.cos(), freqs.sin()
    if table_dtype == 'bf16':
        cos, sin = cos.bfloat16(), sin.bfloat16()
    return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)


def apply_rot(x, c, s):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * c + x2 * s, -x1 * s + x2 * c], -1)


@torch.no_grad()
def reference_forward(m, idx, table_dtype='bf16', score_patch=None):
    """Replicates tt_model semantics with controlled fp. score_patch:
    optional fn(layer_idx, head_scores_dict) -> modified per-branch scores for
    layer 0; signature f(li, s1, s2) -> (s1, s2) [bilinear] or f(li, s, None).
    Returns logits."""
    cfg = m.config
    dt = m.transformer.wte.weight.dtype
    x = m.transformer.wte(idx)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    v1 = None
    B, T = idx.shape
    for li, blk in enumerate(m.transformer.h):
        x = blk.lambdas[0] * x + blk.lambdas[1] * x0
        a = blk.attn
        h = F.rms_norm(x, (x.size(-1),))
        nh, hd = a.n_head, a.head_dim
        cos, sin = rope_tables(T, hd, idx.device, dt, table_dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]

        def qk(lin):
            z = lin(h).view(B, T, nh, hd)
            return apply_rot(F.rms_norm(z, (hd,)), cos, sin)

        v = a.c_v(h).view(B, T, nh, hd)
        if v1 is None:
            v1 = v
        v = (1 - a.lamb) * v + a.lamb * v1.view_as(v)
        mask = torch.tril(torch.ones(T, T, device=idx.device, dtype=torch.bool))
        if cfg.bilinear_attn:
            q, k = qk(a.c_q), qk(a.c_k)
            q2, k2 = qk(a.c_q2), qk(a.c_k2)
            s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / hd
            s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / hd
            if score_patch is not None:
                s1, s2 = score_patch(li, s1, s2)
            pat = (s1 * s2).masked_fill(~mask, 0.0)          # UNNORMALIZED
        else:
            q, k = qk(a.c_q), qk(a.c_k)
            s = torch.einsum('bqhd,bkhd->bhqk', q, k) / hd
            if score_patch is not None:
                s, _ = score_patch(li, s, None)
            pat = s.square().masked_fill(~mask, 0.0)
            pat = pat / pat.sum(-1, keepdim=True).clamp_min(1e-9)  # NORMALIZED
        y = torch.einsum('bhqk,bkhd->bqhd', pat, v).reshape(B, T, -1)
        x = x + a.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (x.size(-1),)))
    x = F.rms_norm(x, (x.size(-1),))
    logits = m.lm_head(x)
    return 30 * torch.tanh(logits / 30)


def build_eval_tokens(n_chunks=32, seq_len=1025, seed=0):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    ds = load_dataset('NeelNanda/pile-10k', split='train')
    ids, chunks = [], []
    for doc in ds:
        ids.extend(tok(doc['text'])['input_ids'])
        while len(ids) >= seq_len:
            chunks.append(torch.tensor(ids[:seq_len]))
            ids = ids[seq_len:]
            if len(chunks) >= n_chunks:
                return torch.stack(chunks)
    return torch.stack(chunks)


@torch.no_grad()
def eval_ce(m, tokens, batch=4, table_dtype='bf16', score_patch=None):
    tot, n = 0.0, 0
    for i in range(0, len(tokens), batch):
        b = tokens[i:i + batch].to(next(m.parameters()).device)
        logits = reference_forward(m, b[:, :-1], table_dtype, score_patch).float()
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             b[:, 1:].reshape(-1))
        tot += ce.item() * b[:, 1:].numel()
        n += b[:, 1:].numel()
    return tot / n


if __name__ == '__main__':
    report = {}
    TOKENS = build_eval_tokens()
    print(f'eval tokens: {tuple(TOKENS.shape)}')
    for short in ['bilin18', 'sqrd12']:
        m, cfg = load_elriggs(short)
        ce_bf = eval_ce(m, TOKENS, table_dtype='bf16')
        ce_ex = eval_ce(m, TOKENS, table_dtype='exact')
        report[short] = {'ce_bf16_tables': ce_bf, 'ce_exact_tables': ce_ex,
                         'params_M': sum(p.numel() for p in m.parameters()) / 1e6}
        ok = 3.0 <= ce_bf <= 4.0
        print(f"{short}: CE {ce_bf:.4f} (bf16 tables) / {ce_ex:.4f} (exact tables)  "
              f"{report[short]['params_M']:.0f}M params  "
              f"-> {'REASONABLE (3-4)' if ok else 'OUT OF RANGE — STOP'}")
        del m
        torch.cuda.empty_cache()
    with open('/workspace/tensor_language/basis_aligned/qk_mdl/tier2_ce_gate.json', 'w') as fh:
        json.dump(report, fh, indent=2)
    print('saved tier2_ce_gate.json')
