# Architecture channels — value-residual, x0, massive dims, embedding dominance, clamps

**One line:** bilin18's four standing channels and what each grounds.

## Value residual (v = ½v + ½·v1, block-0 values routed to every layer)
- Block-0's value is EXACTLY static per-token content (R²=1.0 from embedding; static-table
  replacement costs 0.0; shuffled null +0.52; §1076). The channel broadcasts each word's static
  c_v value; content = pooled bag of c_v VALUES (why raw-embedding bags fail, §1072).
- Ablation (lamb=0): +3.3 nats, content-tilted (rare/freq 2.69); per-layer cost concentrates at
  L2-4 (content onset) (§985-987, §1075 [partial dup, corrected]).
- OOD: the ONE channel relied on MORE on code (cost-frac 1.01→1.10; §1081).

## x0 re-injection (x = λ₀x + λ₁x₀ every block; λ₁≈8 saturated)
- Keeps the embedding ever-present (~8/9 of block input); why class is re-derived every block
  (§962); current token stays linearly recoverable at the FINAL residual (R² 0.73; §690).
- Ablation (λ₁=0): +2.3 nats. BOTH channels are content-heavy; x0 only RELATIVELY more
  grammar-weighted (2.96 vs 3.84) — NOT a clean dissociation (§987 correction).
- Front λ₀ near-zero RESETS the stream (L1 0.013, L5 0.064); a writer 12 layers back arrives
  ×∏λ₀ ≈ 2e-4 (§689).

## Massive activations / gain control (dims 645, 990, 981, 880, 750, 111, 373, 43…)
- = the rms-norm GAIN CONTROLLER, not attention sinks (no softmax; §676-680). Removing the DC
  offset costs +1.58. Host w_freq (88%; §676). ~85% of residual sum-of-squares.
- Written by the multiplicative gates collectively (§688-691); delivered as a constant from
  position 0 by mlp4 → head 5.7 (see `attn-sink-5-7.md`); after L5 the residual is mostly one
  fixed vector with text riding on top (prior sink arc: constant = 62-72% of residual magnitude
  L6-8).
- Independent of embedding-dominance (overlap 2/10; §691).

## Output/readout clamps
- Logits = 30·tanh(lm_head(rmsnorm(x))/30). MLP = Down[(Lx)·(Rx)] + b — every output dim an
  exact quadratic form. The 30·tanh clamp is needed for whole-model simultaneous keep metrics
  (§799).

## Gotchas
- Full channel ablations are off-distribution (graded, near-zero nonlinear tail; §987).
- Any experiment touching the residual scale must respect the gain dims (removing/perturbing the
  baseline is catastrophic and NON-LOCAL; partial removals worse than full — §1087/§1091).
