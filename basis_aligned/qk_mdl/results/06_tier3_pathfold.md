# Tier 3 opener: path-folded lookup codebooks — an informative NEGATIVE

**Question:** tick 8 showed layer-1 key vectors are, on induction data, summarized by
their conditional mean given the previous token (identity hit rate 0.44). Can the whole
L0→L1 path be REPLACED by that summary — k(position j) := k̄(token_{j−1}), a V×128
lookup table folding OV transport + norms + mixing into vocab space?

**Answer: no.** Held-out audit (tables fit on seeds 0–19, audited on 30–34; base
P(copy) 0.744):

| replacement | ΔP(copy) |
|---|---|
| identity branches, k-side lookup | −0.642 |
| identity branches, q+k lookup | −0.615 |
| all 8 (head, branch), q+k lookup | −0.703 |
| identity branches, SHARED k table (sign-aligned) | −0.731 |
| identity branches, SHARED q+k tables | −0.744 (to chance) |

**Interpretation — structure-visible ≠ computation-sufficient.** The conditional mean
exposes the identity conjunct's *direction* (that's why the structure metric works and
matches the causal ablations), but the running circuit consumes context-dependent
components the 0th-order-in-context lookup discards: per-context norm scales, the actual
(not average) prev-token pattern weights, and within-condition variance that the product
attention multiplies against the other branch. This sharpens what a Tier-3 codebook must
be: at least first-order in context (e.g., live L0 pattern × quantized OV content),
rather than a pure vocab-space table. Logan's deeper-layer note applies from here on:
each MLP-bearing layer adds a second input path family (attention-out + residual token),
and path selection at depth will need heuristics or CE/KL-trained pruning.

(The shared-table arms were meant to test cross-head dictionary sharing; since even the
per-head tables fail, the sharing question is unresolved rather than answered.)

## Resolution (tick 19): first-order path codebooks fix the failure

The diagnosis above was tested directly on the rp induction model
(`../first_order_path.py`): keep the layer-0 pattern LIVE (context-dependent weights
intact) and quantize only the OV *content* it transports, with a vq-k table on the
value vectors. The 0th-order collapse disappears and degradation becomes graceful:

| arm | ΔP(copy) @k=64 | @k=256 | @k=1024 |
|---|---|---|---|
| 0th-order lookup (above) | −0.62…−0.74 | — | — |
| live pattern × classed content (all keys) | −0.18 | −0.09 | −0.04 |
| classed content, identity-branch keys only | −0.37 | −0.25 | −0.14 |

Two findings. **FO-1:** the missing ingredient was exactly the context-dependent
pattern weights — first-order-in-context codebooks (live pattern × vocab-classed
content) are the right Tier-3 object. **FO-2 (consistency effect):** classing content
in only ONE coupled path is ~3× worse than classing it uniformly everywhere — partial
replacement breaks internal consistency between paths that the bilinear product
compares. Same composition theme as the cross-block sides and grand-combined L2 stage:
uniform coarseness composes; mixed precision does not.
