"""Part B's census, redone on bilin18's real two-QK attention.

Part B was built on toys, and two of its three placements assume a softmax that
bilin18 does not have. bilin18's attention is

    pattern = (q1.k1 / D) * (q2.k2 / D),  causally masked to zero, NEVER normalised

(`jacclust/tt_model.py:134-144`), with per-head RMS-norm and RoPE applied to both QK
sets first. Entries are signed and rows do not sum to one, so every entropy in the
plan's B1 taxonomy is undefined here. The toys said to use scale-free statistics
instead, and `THEORY.md` T8 proves why: (W1,W2) -> (cW1, W2/c) is exactly function
preserving, so any per-factor statistic must be invariant to it. The participation
ratio is; softmax entropy is not.

THE CLAIM UNDER TEST. The earlier jacclust program recorded
`PR(product) < min(PR(s1), PR(s2))` at **100% of bilin18's 162 heads**, read as
evidence that the two branches are genuinely conjunctive. B0 on the toys found this
signature fires on control tasks with nothing to conjoin AND, in this exact
unnormalised placement, on randomly initialised weights. If that carries over, the
recorded 100% is not evidence of anything.

So the decisive comparison here is bilin18 against the SAME ARCHITECTURE WITH RANDOM
WEIGHTS, head for head, on the same data.
"""

import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/tensor_language')
import jacclust.tt_model as TT
from tier2_model import load_elriggs

DEV = 'cuda'
MIN_KEYS = 64          # only score query rows with at least this many causal keys


def apply_rope(x, cos, sin):
    return TT.apply_rotary_emb(x, cos, sin)


@torch.no_grad()
def head_scores(attn, x):
    """Reproduce the module's two score fields exactly, per head."""
    B, T, C = x.shape
    H, D = attn.n_head, attn.head_dim
    q = attn.c_q(x).view(B, T, H, D)
    k = attn.c_k(x).view(B, T, H, D)
    q2 = attn.c_q2(x).view(B, T, H, D)
    k2 = attn.c_k2(x).view(B, T, H, D)
    cos, sin = attn.rotary(q)
    q, k = F.rms_norm(q, (D,)), F.rms_norm(k, (D,))
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    q2, k2 = F.rms_norm(q2, (D,)), F.rms_norm(k2, (D,))
    q2, k2 = apply_rope(q2, cos, sin), apply_rope(k2, cos, sin)
    s1 = torch.einsum('bqhd,bkhd->bhqk', q, k) / D
    s2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2) / D
    return s1, s2


def pr_rows(s, mask):
    """Participation ratio per query row over its causal keys, normalised by the
    number of keys, averaged over rows with enough keys. Scale-free and
    sign-tolerant, unlike entropy. s: (B,T,T), mask: (T,T) lower triangular."""
    s = s * mask
    num = (s ** 2).sum(-1) ** 2
    den = (s ** 4).sum(-1).clamp_min(1e-300)
    n = mask.sum(-1).clamp_min(1)                 # (T,) keys available per query
    pr = (num / den) / n                          # (B,T)
    keep = n >= MIN_KEYS                          # (T,)
    return float(pr[:, keep].mean())


def negfrac(s, mask):
    s = s * mask
    keep = mask.sum(-1) >= MIN_KEYS
    f = s.clamp(max=0).abs().sum(-1) / s.abs().sum(-1).clamp_min(1e-30)
    return float(f[:, keep].mean())


@torch.no_grad()
def census(model, tokens, tag, n_batch=4):
    rows = []
    xs = {}
    hooks = []

    def mk(li):
        def hook(mod, inp, outp):
            xs[li] = inp[0].detach()
        return hook

    for li, blk in enumerate(model.transformer.h):
        hooks.append(blk.attn.register_forward_hook(mk(li)))
    idx = tokens[:n_batch].to(DEV)
    model(idx[:, :-1].contiguous(), idx[:, 1:].contiguous())
    for h in hooks:
        h.remove()

    for li, blk in enumerate(model.transformer.h):
        s1, s2 = head_scores(blk.attn, xs[li])
        B, H, T, _ = s1.shape
        mask = torch.tril(torch.ones(T, T, device=s1.device, dtype=s1.dtype))
        prod = s1 * s2
        W1 = (blk.attn.c_q.weight.T @ blk.attn.c_k.weight).double()
        W2 = (blk.attn.c_q2.weight.T @ blk.attn.c_k2.weight).double()
        for h in range(H):
            p1, p2 = pr_rows(s1[:, h], mask), pr_rows(s2[:, h], mask)
            pp = pr_rows(prod[:, h], mask)
            D = blk.attn.head_dim
            a, b = W1[h * D:(h + 1) * D], W2[h * D:(h + 1) * D]
            cos = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
            rows.append({'model': tag, 'layer': li, 'head': h,
                         'PR_factor1': p1, 'PR_factor2': p2, 'PR_product': pp,
                         'fires': bool(pp < min(p1, p2)),
                         'drop': min(p1, p2) - pp,
                         'negmass_product': negfrac(prod[:, h], mask),
                         'qk_cosine': cos})
    return rows


def summarise(rows, tag):
    n = len(rows)
    fires = sum(r['fires'] for r in rows)
    drop = sorted(r['drop'] for r in rows)
    neg = sorted(r['negmass_product'] for r in rows)
    cos = sorted(abs(r['qk_cosine']) for r in rows)
    s = {'model': tag, 'n_heads': n, 'n_fires': fires, 'frac_fires': fires / n,
         'median_drop': drop[n // 2], 'median_negmass': neg[n // 2],
         'median_abs_qk_cosine': cos[n // 2]}
    print(f"  {tag:26s} sharpening fires on {fires:3d}/{n} heads ({100*fires/n:5.1f}%) | "
          f"median PR drop {s['median_drop']:.3f} | median negative mass "
          f"{s['median_negmass']:.3f} | median |cos(W1,W2)| {s['median_abs_qk_cosine']:.3f}")
    return s


def main():
    t0 = time.time()
    tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_eval_tokens.pt')
    out = {'claim_under_test': 'PR(product) < min(PR(factors)) at 100% of heads was read '
                               'as evidence of conjunction; the toys say it is not specific'}

    print('== the trained model ==')
    model, cfg = load_elriggs('bilin18', device=DEV)
    trained = census(model, tokens, 'bilin18 (trained)')
    out['trained'] = trained
    out['trained_summary'] = summarise(trained, 'bilin18 (trained)')

    print('\n== the SAME architecture, random weights (the decisive control) ==')
    torch.manual_seed(0)
    rnd = TT.GPT(TT.GPTConfig(**{k: v for k, v in cfg.items()})).to(DEV).eval()
    for p in rnd.parameters():
        p.requires_grad_(False)
    rows_r = census(rnd, tokens, 'random init')
    out['random'] = rows_r
    out['random_summary'] = summarise(rows_r, 'random init')

    print('\n== per-layer breakdown (trained) ==')
    out['by_layer'] = {}
    for li in range(cfg['n_layer']):
        rs = [r for r in trained if r['layer'] == li]
        f = sum(r['fires'] for r in rs)
        md = sorted(r['drop'] for r in rs)[len(rs) // 2]
        mn = sorted(r['negmass_product'] for r in rs)[len(rs) // 2]
        out['by_layer'][li] = {'fires': f, 'n': len(rs), 'median_drop': md,
                               'median_negmass': mn}
        print(f"  layer {li:2d}: fires {f}/{len(rs)} | median PR drop {md:.3f} | "
              f"median negative mass {mn:.3f}")

    ft, fr = out['trained_summary']['frac_fires'], out['random_summary']['frac_fires']
    out['verdict'] = ('the signature does not discriminate: it fires at essentially the '
                      'same rate on random weights' if fr > 0.8 * ft else
                      'the signature does discriminate trained from random')
    print(f"\nVERDICT: {out['verdict']}")

    out['runtime_s'] = time.time() - t0
    p = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
         'bilin18_attention_results.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'wrote {p} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
