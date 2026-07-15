# Layer 1: conditional-mean codebooks (selection is nearly token-deterministic)

Everything before this file compressed **layer 0**, where folding the embedding into the
QK maps is *exact*: a token's query/key factor is a fixed function of its identity. At
layer 1 that breaks — the input is the layer-0 output, which depends on context, so
there is no weights-only vocab table to fold.

## Method: estimate the table from data instead of folding it from weights

For each head and branch, run the model over 524k pile tokens, capture the layer-1
query/key factors **post-QK-norm, pre-RoPE** (the same gauge point the layer-0 folding
uses), and average them by current-token identity:

```python
# capture pass (layers 0-1 only, no logits needed)
z = F.rms_norm(lin(h1).view(B, T, NH, HD), (HD,))     # h1 = layer-1 normed input
acc[name].index_add_(0, tokens.reshape(-1), z.reshape(-1, NH*HD))
cnt.index_add_(0, tokens.reshape(-1), ones)
qbar = acc / cnt   # (V, NH, HD): the 0th-order-in-context factor table
```

Unseen vocab rows (9% of audit tokens) fall back to the global mean. The tables are then
renormalized to unit RMS — the QK-norm shell is where the live factors actually live —
and patched into the model with the *same* `scores_from_factors` RoPE expansion used for
layer 0. Audit = full-18-layer ΔCE at T=512 (binding metric).

## Results (`../l1_condmean_qk.json`)

| arm | ΔCE |
|---|---|
| zero layer-1 scores (control) | +2.820 |
| cond-mean tables, raw | +0.040 |
| **cond-mean tables, unit-RMS** | **+0.014** |
| + vq256 (shared [q\|k] partition per head-branch) | +0.092 (L2-fit) |
| + vq1024 | +0.064 (L2-fit) |

## What this means

1. **Layer-1 attention selection is ~0th-order in context.** The layer is heavily
   load-bearing (+2.82 when silenced), yet replacing every layer-1 query/key with a
   pure function of the current token costs +0.014. Whatever context the layer-0 output
   mixes into the layer-1 QK factors, the *selection decision* barely uses it. This is
   the same asymmetry seen everywhere in the program — selection tolerates coarseness,
   carriage doesn't — now in the **context** dimension: the tier-3 0th-order lookup that
   collapsed for OV *content* (file 06) works nearly for free for *selection*.
2. **Gauge matters**: raw conditional means (whose norm shrinks for high-variance
   tokens) cost 3× more than unit-RMS renormalized ones. Averaging must respect the
   QK-norm shell.
3. **Classing costs more here than at layer 0** (vq256: +0.092 vs +0.008): the
   data-estimated tables are noisier objects than folded weights, so class centroids
   blur genuinely distinct rows. CE-trained class tables (assignments frozen, ~1M floats
   through the frozen model) are the standard repair — `../l1_ce_codebook.py`.

Caveats: single eval distribution (pile-10k, T=512); the +0.014 includes the 9%
global-mean fallback, so it is an upper bound on the intrinsic 0th-order error;
estimated on the same distribution as the audit (disjoint chunks).
