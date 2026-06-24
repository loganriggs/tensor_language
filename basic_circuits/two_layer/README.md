# Two stacked bilinear layers (next project)

Existing analyses moved here from the original flat directory; this is the home
for the two-layer bilinear work going forward. All pure-numpy, self-contained
(no external checkpoints).

- `decomp_exact.py` — NO training. Builds the exact symmetric degree-4 tensor for a
  4-way-AND monomial, matricizes to (ij)(kl), and shows the orthogonal/eigh
  canonical form is forced into ±-mixtures of complementary pairings (so SVD can't
  rank/select among the 3 pairings). The "orthogonal isn't the right structure"
  demonstration, plus the mixed-degree (const-padded) case.
- `train2layer.py` — trains a 2-layer bilinear net on 4-way AND (m=8 + const,
  5-hot, T=70). Builds the factored representation {Acheck_p, Bcheck_p, Wo},
  verifies the factored quartic reproduces the forward pass to ~6e-14, then shows
  the same pairing-mix pathology in a *learned* model.
- `mixed.py` — 2-layer bilinear with the folded residual (const coordinate). Mixed
  targets (28 degree-2 + 28 degree-4 ANDs). Trains to 100% TPR/TNR, folds to
  x-space, decomposes one target of each degree; finds const-routing entangles the
  detectors.

- `toy_2layer.py` — toy (mirrors `../toy_and/`): 2 stacked bilinear layers, no
  residual, computing the `C(7,4)=35` four-wise ANDs on 5-hot-of-7 inputs (21
  inputs; each input has 5 co-active ANDs). 21 inputs chosen so the minimal net has
  fewer params than decisions -> genuine superposition, not memorization. Sweeps
  `(h1,h2)` -> minimal frontier **(3,5) or (6,4)** for all 35 (an L-shaped layer-
  width trade-off). Logit ladder + degree-4 ladder decomposition: the genuine 4-AND
  signal term barely matters (no-interference 61%, no-signal 92%) — the computation
  is in the distributed interference. See results/toy_2layer.md.

See `../CONTEXT.md` open threads #2/#4 for the next steps (degree-stratified
Tucker; bond canonicalization for depth 2).
