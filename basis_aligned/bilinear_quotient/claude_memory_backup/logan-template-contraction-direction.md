---
name: logan-template-contraction-direction
description: Logan's 19 Sep direction — score folded tensors against template tensors (identity, shift, contrast, duplication), split equivariant vs word-specific circuits; more to come
metadata:
  type: project
---

Logan (19 Sep 2026, ~05:58 UTC): single-token folding targets a fixed *vector*; copying behaviours are a fixed *map*. Target for copying/induction =
identity between token embedding and unembedding. Copying-from-position-j = contract the logit polynomial's output mode with W_U and input mode j with
E, trace against identity (the identity-like score of tensor-sim). Induction = degree-3 term with two identity couplings (QK score identity-like AND
the gated OV path identity-like) — search for it directly, then multiply. Equivariance (Schur): the only token-basis-equivariant bilinear form is the
inner product → decompose folded tensors into equivariant (copy/match/induction) and non-equivariant (word-specific, e.g. UK/US) parts before choosing
targets. Other templates: shift operators on positional embeddings (previous-token, n-back), identity minus something (copy suppression, IOI negatives),
rank-1 open/close contrast for bracket counting (degree distribution across positions), same-token duplication (identity coupling between two input
positions, no output copy). Practical: a library of template tensors, fold the model to slices of the right shape, Frobenius inner products; a head
scoring high on no template = unnamed candidate. He said "Keep this in mind, will send more."

**Why:** gives a principled, target-free split of the whole model and a direct search for induction/copying.
**How to apply:** first runner v389 (19 Sep) — identity-likeness of every head's OV path (W_U O_h V_h E^T vs I) and QK forms (E Q^T K E^T, both
bilinear factors) for all 162 heads, weights only; later shift templates need the rotary positional structure. See [[claude-circuit-lane-2026-09-17]].

**Second message (19 Sep ~06:01 UTC), families beyond single-token and pairwise-difference folding (the model is exactly polynomial in the tokens):**
- Richer output targets: fold back from a *subspace* (top-k SVD of W_U restricted to a token class — colour words, UK spellings; class mean vs class residual is the clean shared-vs-idiosyncratic split); from a probe / steering vector (circuit for a concept); from an attention score (QK ⊙ Q2K2 is bilinear, no softmax — "why does head 9.8 attend here" is a folding target); from a DCT feature / transcoder latent at layer L.
- Input-side partial evaluation: fix all positions but one and fold forward (the slot's role); fix one token at one position and fold forward (what "the" does to everything after it); contrast on both ends at once, (e_a − e_b) × (logit c − logit d) isolates the interaction slice.
- Exact structural decompositions: degree-per-position expansion (exact path expansion by attention hops); mixed partials ∂²logit/∂x_i∂x_j across positions → pairwise interaction matrices, eigendecompose (eigenfilters); exact Shapley via the closed-form multilinear extension.
- Distributional folding: plug empirical / Gaussian token moments (Isserlis / Gram) → expected circuit and its variance (load-bearing on-distribution vs OOD-only slices); fold the difference of two checkpoints / models on the same target (tensor-sim).
- His first pick in general: both-ends contrast + degree expansion (is UK/US a two-hop path or a sum of one-hop shortcuts). **His instruction to me: do the subspace idea.** Started as v390 (19 Sep 06:02).
