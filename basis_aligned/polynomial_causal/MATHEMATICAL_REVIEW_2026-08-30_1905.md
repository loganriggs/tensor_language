# Mathematical review — 2026-08-30 19:05 UTC (self-reviewed)

Grounded in today's artifacts: `STREAM_ERROR_PRICE_V1_RESULT.md`, `PRICE_CLIFF_SUBLAYER_V1_RESULT.md`,
`OBSERVABILITY_QUOTIENT_V1_RESULT.md`, lane 1 §2101–§2121. The day's positive result (an eight-direction
selector for mlp4/mlp5 units, certified on eight fresh windows) and its two negatives (metric-constructed
bases/spans hurt; the tail-span "gain" is coverage) all rest on one object that has not yet been named
mathematically: the matrix G_k = E[∂CE/∂x_k ∂CE/∂x_kᵀ] at the stream entering block k.

## 1. What G_k is: the empirical Fisher pulled back to the stream

With true next-token labels, G_k is the **empirical Fisher** of the loss with respect to the stream at site k.
The **true Fisher** replaces the label by a sample from the model's own predictive distribution p:
F_k = E_x E_{y∼p}[g gᵀ] = E_x[J_kᵀ (diag p − p pᵀ) J_k], J_k = ∂logits/∂x_k — the pullback of the KL metric
on output distributions through the downstream map. Two facts about this object matter for the program:

- **Second-order price.** For a stream perturbation δ at site k, the KL between perturbed and unperturbed
  predictions is ½ δᵀ F_k δ + O(‖δ‖³); the CE increase under true labels is the empirical version. So the
  price of error should be *quadratic in the norm at small norm*, with the Fisher as the metric. The measured
  curve (`stream_error_price_v1`) gives log₂-exponents between r = 0.25 and 0.5 of **2.2–2.6 at every block
  1–17** (block 0: 6.9, at CE increases of 10⁻⁴), and between 0.5 and 1.0 of 3–4 at blocks 1–5 (higher-order
  growth through the bilinear band), 1.2–1.4 at blocks 6–8 (saturation: CE increases of 3–4 nat cannot keep
  doubling), ~2 late. **The quadratic regime is real and universal below r ≈ 0.5.** That is the regime in
  which the Fisher is *the* price, and in which a program's error budget is a quadratic form.
- **Label-freedom.** If F_k's top eigenvectors coincide with G_k's, the eight directions are a property of
  the model and its inputs alone — computable with no labels, i.e. a *weights-and-unlabeled-data* object,
  the same status as lane 1's fold tables (§283). Kunstner, Balles & Hennig (NeurIPS 2019, "Limitations of the
  empirical Fisher approximation for natural gradient descent") show the two can differ badly when the model
  is far from the data; at CE 3.4–3.9 on fresh text bilin18 is not near, so this is a real question, not a
  formality. Martens (JMLR 2020, "New insights and perspectives on the natural gradient method") is the
  standard reference for the pullback and Gauss–Newton identities used here.

Assumptions that may fail: third-order terms are not small at r ≥ 0.5 (measured); the per-position Fisher
ignores cross-position coupling (the cliff result says the cost is local, so this is mild after block 5 and
untested before it).

## 2. Composition through a block: the chain rule as a composability test

The whole point of the alternate entry point is a quotient that composes across RMSNorm/residual interfaces.
The Fisher composes exactly by the chain rule: F_k = E[J_{k→k+1}ᵀ F_{k+1} J_{k→k+1}] with J_{k→k+1} the
Jacobian of one block (attention + MLP + norms + residual). The eight at block 5 and the eight at block 6
overlap only 0.47 as raw subspaces (§2111) — but they *should not* be compared raw; the block-6 eight pulled
back through block 5's Jacobian should match the block-5 eight. If it does (overlap ≥ 0.7), the metric is a
single composable object propagated by the network's own Jacobians, and "the eight directions at site k" can
be computed at one site and transported. If it does not, per-position nonlinearity inside a block breaks
first-order transport and the metric must be measured per site — still usable, not composable.

Cheapest experiment: VJPs of the eight top eigenvectors of G_6 through block 5 (eight backward passes per
batch), span them, compare with G_5's eight.

## 3. Pricing projection stand-ins by coverage (MDL, prequential)

§2121: a tail program that replaces the projection of a real MLP's output on an 8-dim span "improves" by
0.2 nat when the span is chosen to cover less variance. The benchmark credits "the tail MLP replaced by a
rank-8 program"; the honest credit is the program's **prequential** contribution: the bits of the module's
output it accounts for (explained energy under a Gaussian residual model) net of the CE it costs. Operational
definition for the registry: credit = fidelity × covered-energy share, where covered-energy share is the
fraction of the module's output variance (or of its observable energy, u^T F u) inside the replaced subspace.
Rung 28 (running) measures whether covered variance alone predicts the tail-span gain (ρ ≤ −0.7); if so, the
registry amendment is licensed and the tail entries are re-credited.

## 4. Pruned

- Tensor/CP rank of the response tensor: closed by v1's rejection and §2098–§2100 (private document code).
- Simultaneous factorization / shared dictionaries: no survivor to factor.
- Bisimulation / finite-state quotient: the linear quotient is two-thirds of the stream; a finite quotient
  needs a state definition the day's results do not yet supply.
- Hankel / minimal realization: no sequence-to-sequence linear structure has been identified to realize.
- Metric-constructed bases or spans: negative twice (§2105, §2120); do not retry.

## 5. Ranked moves

1. **Fisher identity check** (label-free eight; quadratic price at r ≤ ¼ with the Fisher as metric) —
   `fisher_metric_v1.py`, registered, queued.
2. **Chain-rule composition of the eight through block 5** — same script.
3. **Coverage-weighted credit for projection stand-ins** — registry amendment, pending rung 28.

## 6. Executed

`fisher_metric_v1.py` written and queued (see header for the four registered predictions). Not an outcome
until its artifact lands.

## RESULT — 19:12 UTC (`fisher_metric_v1_results.json`; ledger §2123)

pred_c HELD (Fisher trace prices a small random error at 0.58× prediction, both sites); pred_a FAILED
(true-vs-empirical top-8 overlap 0.55/0.51 — the eight are half label-dependent); pred_b FAILED at block 6
(exponent 2.80 even at r ≤ ¼ — super-quadratic at the cliff); pred_d FAILED (Jacobian pullback overlap 0.40,
below the raw 0.475 — no first-order transport). Move 1's conclusion: the certified selector is an empirical,
site-local object with the Fisher's scale but not its label-freedom or composability. Rung 29 (a true-Fisher
selector, the operational label-freedom test) is queued; move 3 (coverage credit) was licensed by §2122.
