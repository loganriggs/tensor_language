# Inside the window: naming the irreducibly contextual computations

File 11 reduced bilin18's live core to: a short local window everywhere, two contextual
heads at L5, and the top MLPs. This file names what the two heads do.

## The two heads split selection-vs-carriage (WW-1, WW-2, WW-3)

Signatures on natural text and on repeated sequences (A+A, random A of length 256),
plus causal ablations (ΔCE; repeat = second-half only):

| | L5.H5 | L5.H7 |
|---|---|---|
| induction signature (natural / repeat) | 16.8× / **53×** random | ~1× / none |
| positional profile | flat to Δ64 (2.4× decay) | local, high through Δ≈4 |
| zeroed: natural / repeat-2nd-half | +0.03 / +0.13 | **+1.04 / +6.68** |
| cond-mean tabled: natural / repeat | +0.08 / +0.28 | +0.10 / +1.95 |

- **H5 is the match head**: classic induction pattern ("attend where my previous
  occurrence pointed"), intensifying exactly when the context contains repeats. But its
  causal footprint is small — the signal it selects is barely cashed in by this model
  (bilin18 is weak at literal copying overall: repeat-2nd-half baseline CE 5.48 vs 3.23
  natural).
- **H7 is the transport head**: no match structure in any context, strictly local
  selection — yet causally dominant everywhere and catastrophically so on repeats.
  Coherent with the interaction map (file 11): attn5's output is the persistent hub
  stream the entire upper model reads, and H7 is its heavy contributor. Its
  contextuality is *which nearby token's content to forward*, not *where to match*.
- The correlational/causal dissociation (H5's big signature vs H7's big ablation) is
  the same lesson as the conjunction test's generic-vs-conditioned gap: pattern
  statistics identify selection structure; only ablations identify load-bearing
  carriage.

So the familiar selection/carriage split reappears INSIDE the contextual core: one head
selects by content-matching (un-tableable because matching compares context to context),
one head carries content chosen by local context (un-tableable because the choice is
contextual even though the range is local).

## What the two heads carry (WW-4)

Logit-lens on the per-source-token conditional-mean OV content (2961 frequent tokens;
crude-lens caveat applies — 12 layers process this before the unembedding):

| | lens top-1 = source | median rank of source | cos with embedding |
|---|---|---|---|
| H5 | 0.147 | **25** / 50k | 0.020 |
| H7 | 0.000 | 4072 | 0.065 |
| H0 (free head) | 0.006 | 8035 | 0.027 |

**H5 carries token IDENTITY** — the complete textbook induction head (match selection
+ identity carriage), just weakly cashed in by this model. **H7's token-conditional
content is a near-constant generic direction** (every source token decodes to '-',
' and', '(' …): its causal payload lives in the context DEVIATIONS that conditional
means average away — which is precisely why H7 resists every 0th-order treatment.
Deviation-PCA probe (context-conditioned) queued to name the payload.

## The contextual core, causally closed (WW-5, WW-6)

Deviation-PCA around token-conditional means: H7's deviations are ~5% of its output
energy, 63% of it in ONE direction that lenses to the same connective/structure feature
as its mean. Causal confirmation — replace each head's output by its projection onto
(mean + top-k deviation PCs) with LIVE coefficients:

| rank k (+mean) | H7 natural / repeat | H5 natural / repeat |
|---|---|---|
| 1 | **+0.0001** / +0.049 | +0.073 / −0.238 |
| 2 | +0.003 / +0.016 | +0.035 / −0.327 |
| 8 | −0.000 / +0.087 | +0.013 / −0.108 |

**H7 is causally a rank-one gain head**: one fixed hub direction, one context-dependent
scalar — its natural-text function is complete at rank 1, and even its +6.68 repeat role
survives the rank-1 bottleneck at 99%. H5 is the opposite (high-rank identity content;
rank-1 hurts) — with a curious bonus: low-rank filtering of H5's content IMPROVES repeat
prediction (−0.33 at rank 2): the model under-cashes its own induction signal, and
denoising the carried identity strengthens it. (H7's rank-4 repeat number is
non-monotonic — small-sample PCs; treat the k=1 result as the finding.)

**Arc conclusion:** everything irreducibly contextual in this 546M model's attention is
(a) one content-match head whose payload is token identity, and (b) one scalar gain on
one structure feature. All other selection and transport: token-static tables + a local
window (file 11). The remaining live mystery is the top MLPs.

## Postscript: why the model under-cashes H5 (WW-7)

Two hypotheses tested (`../h5_undercash.py`): content noise (replace H5's carried v by
clean cond-mean identity, pattern live) vs amplitude starvation (scale H5's pattern).
Cleaned content: repeat **−0.170** (natural +0.023). Scaled pattern: repeats WORSEN
monotonically (α=1.5: +0.34; α=4: +3.37); cleaning+scaling adds nothing over cleaning.
**Boundary found by card 2 (cards/card2_denoising.md):** the cleaning gain is
DISTRIBUTION-DEPENDENT — on natural-word repeats cleaning HURTS (−0.16): H5 carries
context-MIXED identity; the context part is noise on degenerate data, signal on real
text. The corrected statement: **The induction head carries a noisy copy of token identity, and the model's small
gain on it is rational** — boosting amplifies noise; denoising (low-rank filter,
cond-mean table) is what helps. Carriage fidelity, not matching or amplitude, is the
induction bottleneck in this model.

Caveats: over-random ratios for the seven free heads are numerically meaningless
(signed means near zero); use the conditional means in the json files. Repeat data is
synthetic (uniform random tokens); H7's mechanism deserves an OV-side decomposition
(which content features does it forward into the hub?) — queued.

Files: `../l5_heads_function.py/.json`, `../l5_heads_function_rep.py/.json`,
`../l5_h5_causal.py/.json`.
