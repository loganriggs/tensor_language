# TN-gauge / layer-1 QK investigation — synthesis (F1–F39)

A step-back over the whole arc. Full per-finding detail is in `GOALS.md`; this is the
navigable summary. Everything is gate-verified (each patched/compressed forward reproduces
the reference model at full rank) and held-out where a fitted object is involved.

## Origin
Logan's program: reduce a bilinear transformer (bilin18, 546M) to a legible tensor
network via a shared overcomplete code basis, DMRG-style. Two "regimes" were tried, then
the work pivoted to a concrete target Logan steered to: the query/key of the second
attention layer and what feeds it.

## Regime 1 — exact rotation gauges are EMPTY (F1–F8)
Rotating any private bond to sparsify its cores is an exact gauge (ΔCE=0) but buys almost
nothing: OV L1 drops 7% on the toy, ~0% on the flagship; QK RoPE-torus 1.4%/0.2%. The
residual bond is pinned by the embedding/unembedding boundaries; the value bus is shared
across depth (value-residual). **Rotation gives ~0 compression** — the banked baseline.

## Regime 2 — overcomplete code-propagation is a Pareto trade, not a free win (F1–F12)
A single shared dictionary coding every bond is too lossy (ΔCE +0.59 even at k=64); capacity
and per-bond dicts help but the additive-propagation regime (one shared Φ) is exactly the
lossy config. Write-seeded "births" capture the right subspace for a *fixed* dictionary but
are useless as a *training* init (clustered/rank-limited; random init trains better). Net:
no clean free reduction here.

## The productive line — layer-1 query/key (F13–F37)
**What it reads.** Layer-1 selection runs almost entirely on the block-0 **bilinear (mlp)
output** — removing it costs +0.68 nats, the attention output and embedding are droppable
(F13 toy, F18 flagship). Weight-only, query/key reads only ~1024 of 4608 bilinear hidden
units; the null is a linear property of Down∘R (F28).

**The used-subspace (the key positive, F24/F25).** The query/key reads a ~128-dim
*activation-weighted* input subspace. Identifying it (whitened-optimal, data-driven) beats
generic weight-only low-rank at every layer, held-out — the interpretation gives the
compression basis that structure-blind methods can't reach. Query/key compresses to ~14% of
raw for near-free per layer.

**Input-relative structure (F30–F37).** The query/key code is 82% current-token-determined
(a vocab table, +0.0008) + 18% context. The token classes decode to **grammatical categories**
(determiners, prepositions, auxiliaries, wh-words); the composed (current×attended) pair
classes decode to **syntactic dependencies** (aux-verb→subject, clause→sentence-boundary,
determiner→preposition, of→noun). **Composed features beat individual** at the binding metric
(individual floors at 0.66 FVU, composed reaches 0.05; F33/F34), and this holds at layer 2
though selection gets more distributed/less token-determined with depth (F37).

## End-to-end + the DMRG bridge (F38–F39)
All 18 layers' query/key compress together to **~28% of raw bits for +0.06 nats held-out**
(~3.6× the whole model's query/key). Below that the per-layer wins compound (+0.24 at 14%),
and the **DMRG iteration does not fix it** (F39): re-fitting each layer's subspace on the
compressed activations is a no-op because the per-layer compression is near-lossless (bases
already self-consistent). The compounding is inherent error accumulation through depth — the
sweep helps only when per-layer bases are mutually *inconsistent*, which they aren't here.

## Headline takeaways
1. **Rotation buys nothing; overcompleteness is a Pareto trade; the win is activation-aware.**
   The used-subspace (data-driven, whitened-optimal) is the one method that beats structure-
   blind compression, held-out, at every layer.
2. **Interpretation *is* compression, input-relatively.** Layer-1 query/key is a vocab table
   (grammatical categories) plus a syntactic-dependency cross-term; the composed pair is the
   right feature unit and provably beats individual token features.
3. **The bilinear layer is the workhorse.** Selection reads it richly (high-dim), OV and the
   query/key are low-dim; the early bilinear layer computes the features later selection reads.
4. **Depth erodes legibility.** Single-source structure and token-determination decrease with
   depth (F19, F37); deep selection is distributed. Cross-layer compression accumulates error
   irreducibly (F39).

## Gate discipline (load-bearing throughout)
Controls/gates caught, before any number was reported: a dead L1 optimizer (planted control),
a wrong per-layer gauge (ΔCE check), a broken inline forward twice (reference-CE), a whitening
inversion (layer-1 sanity check), an in-sample overfit (held-out split), a normalization bug
(fraction>1). None of the headline numbers rest on an un-gated claim.
