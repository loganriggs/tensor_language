# Do PR-regularized DCT factors find real sparse circuits? — results

Claude (Fable), 22 September 2026. Raw JSON and logs are in `results/`; tables are rendered by `scripts/summarize.py`.

## 0. What was run, and how it differs from the handoff

- **Model code.** Running AJ's cloned repository against the real checkpoints is blocked in this environment (the permission classifier refuses to execute the external code), so the checks run on this repository's own loader (`circuit_checks/inrepo_model.py`, built on `jacclust/tt_model.py`). That file is the same modded-nanogpt code as AJ's `tensor_model.py` (same `Block(x, v1, x0)`, λ-mixing, attention `(y, v1)`, bilinear `Left/Right/Down`), and the handoff adapter runs on it unchanged. `check_adapter_parity.py` gives forward parity **0.00e+00** against AJ's `TensorMiddleSpan` on a random model (the one call to his code that was allowed), and `run_checks.py` asserts parity of the adapter against the repository's block code at startup on the real checkpoint: **0.00e+00**.
- **Fitting loop.** `circuit_checks/dct_fit.py` is a port of `AsymmetricQuadraticDCT.fit` and `ParticipationRegularizedDCT.fit`, written after reading both in full: same random-normal init under the seed, QR of L and R every iteration, forward-over-forward mixed Hessians, gradient of ½(u·H[l,r])² − w·S·PR/D per factor per context, averaged gradients replacing the factors (β = 1), penalty scale S = baseline mean total energy / factors. `tests/test_fit.py`: a planted single-unit circuit is recovered (cos u = 1.000, span of (l, r) inside the plant's span with canonical correlations 1.000), the penalty concentrates PR on the toy, and weight 0 reproduces the plain class bit for bit.
- **Data.** AdvBench `target` strings exactly as AJ reads them (unique, `random.Random(1729)` shuffle, right-padded with EOS to 32 tokens; median 16 tokens, so the last three scored positions are usually EOS padding attending to the text). Off-distribution set: the first 32 GPT-2 tokens of FineWeb documents (`fineweb` runs), which have no padding.
- **Everything downstream of the fit** (E1–E4) is the handoff's `circuit_checks` code, plus two additions described next.

## 1. A property of the objective that changes how E2 and E4 must be read

u·H[l, r] is a mixed second derivative, so it is **symmetric in l ↔ r**: the DCT score is a symmetric bilinear form B_u(l, r), and |B_u| over unit vectors is maximised at **l = r = top eigenvector of B_u**, with a continuum of equally good pairs when the spectrum is degenerate. For a single bilinear unit (a·n)(b·n) the pairs (l, r) = (a, b) and l = r = (a+b)/√2 score identically; the fixed-point iteration lands on the latter (straight and swapped cosines both 0.77 on the toy). On the real model **every penalised factor has cos(l, r) = 1.00** (unpenalised: 0.4–0.8). Consequences:

- The "asymmetric" parametrisation is redundant in practice; the fitted object is (u, v) with l = r = v.
- The handoff's E2 alignment |cos(l,a)||cos(r,b)| is capped near 0.5 for a factor that *is* a neuron, and the E4 similarity |cos u|·|cos l||cos r| understates agreement between fits that sit at different points of the degenerate set. Both are still reported; alongside them I report **span versions** (`span_alignment`, `span_similarity` in `checks.py`): the mean squared canonical correlation between span{l, r} and span{a_h, b_h}, and |cos u| × the same between two factors' spans. A random pair scores ~0.01 on the span alignment (null in the tables).

## 2. E0 — reproduction at AJ's scale (bilinear, AdvBench, 4 train / 8 held-out prompts, 4 factors, 5 iterations, seed 0)

| w | median held-out PR | held-out energy / baseline | E1 top-1 | E1 top-5 | E1 all-AJ-MLPs | E1 source MLP | E1 attention | E2 span align (best) | E2 read-off energy ratio | E3 top-1 @0.05 | E3 top-5 @0.05 | E3 random-5 @0.05 | E3 top-1 @0.2 | cos(l,r) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 304.6 | 1.00 | 0.06 | 0.07 | 0.23 | 0.85 | 0.58 | 0.37 | 1.45 | 0.11 | 0.12 | −0.00 | 0.15 | 0.67 |
| 1 | 1.8 | 1.33 | 0.46 | 0.59 | 0.76 | 0.42 | 0.61 | 0.40 | 0.57 | 0.54 | 0.65 | −0.00 | 0.57 | 1.00 |
| 10 | 1.7 | 1.37 | 0.52 | 0.60 | 0.78 | 0.41 | 0.61 | 0.44 | 0.50 | 0.60 | 0.69 | −0.00 | 0.75 | 1.00 |
| 100 | 1.5 | 1.23 | 0.64 | 0.71 | 0.85 | 0.34 | 0.61 | 0.51 | 0.72 | 0.68 | 0.74 | −0.00 | 0.75 | 1.00 |

Medians over the 4 factors; E2 null (random pair / span) ≈ 0.007 / 0.010 per layer. Per-factor detail is in `results/e0_ajscale_bilinear.json`.

**The headline replicates.** Median held-out PR drops from 305 to 1.5–1.8 and held-out energy is *higher* than the unpenalised fit (1.2–1.4×), so "keeps 70–100% of causal strength" is, if anything, conservative at this scale. Three things the checks add:

1. **Unpenalised, the top-energy factor is already a single neuron.** At w = 0 the highest-energy factor has PR 1.5, top-1 completeness 1.03, and a raw neuron scores 1.16× its energy as a factor (E2 read-off ratio). The other three unpenalised factors are diffuse (PR 300–700) and their interaction lives mostly in the **source-block MLP** (completeness 0.83–0.87), outside AJ's PR range — the loophole the handoff anticipated is what an unpenalised diffuse factor looks like.
2. **The penalty moves the interaction into the measured range, but not into one unit, and not out of attention.** At w ≥ 1: top-1 completeness 0.22–1.03 (median 0.46–0.64), all-AJ-range MLPs 0.54–1.05, source MLP 0.30–0.59, **attention 0.57–0.64 for every factor**. Freezing all attention removes about 60% of every factor's interaction at every penalty weight; the groups' fractions sum to well over 1, so pathways overlap and cancel (freezing all MLPs and attention together gives exactly 1.00). PR ≈ 1.5 therefore means "of the part carried by intermediate MLP units, one unit dominates", which is what PR measures and no more.
3. **Finite ablations agree at 5% of the residual norm and not at 20%.** E3 top-1 patching removes 0.4–1.1 of the finite interaction at scale 0.05 (random-5: 0.00); at 0.2 the fractions are −15 to +1.9, i.e. the finite response is far outside the quadratic regime there.
4. **The penalised factors are partly, not fully, neuron-aligned.** Span alignment to the best unit 0.27–0.66 (null 0.01); a raw neuron as a factor recovers 0.3–1.1 of the factor's energy.

## 3. Full-scale runs on the bilinear checkpoint (32 train / 64 held-out prompts, 8 factors, 10 iterations, seeds 0 1 2, weights 0 / 0.1 / 1)

Medians over 24 factors (8 × 3 seeds); E1–E3 on 16 held-out contexts; E4 across the three seed pairs and across two disjoint halves of the training prompts (both halves fitted from seed 0). Full tables with all columns: `python scripts/summarize.py results/full_bilinear_*.json`.

### 3a. AdvBench prompts (AJ's distribution)

| w | median held-out PR | held-out energy / baseline | E1 top-1 | E1 top-5 | E1 all-AJ-MLPs | E1 source MLP | E1 attention | E2 span align | E2 read-off ratio | E3 top-1 @0.05 | E3 random-5 | E4 seeds (pair / span) | E4 split (pair / span) | top-5 Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 386 | 1.00 | 0.01 | 0.05 | 0.24 | 0.85 | 0.52 | 0.47 | 0.09 | 0.02 | 0.00 | 0.20 / 0.17 | 0.82 / 0.64 | 0.79 |
| 0.1 | 3.1 | 0.70 | 0.12 | 0.26 | 0.75 | 0.60 | 0.55 | 0.44 | 0.34 | 0.22 | 0.00 | 0.08 / 0.06 | 0.53 / 0.42 | 0.51 |
| 1 | 2.7 | 0.66 | 0.18 | 0.32 | 0.85 | 0.58 | 0.59 | 0.44 | 0.43 | 0.22 | 0.00 | 0.16 / 0.10 | 0.83 / 0.60 | 0.79 |

Per-factor counts (24 per weight): PR ≤ 5 — 0 / 10 / 11; top-1 completeness in [0.5, 1.5] — 0 / 9 / 10; raw neuron beats the factor (read-off ≥ 1) — 0 / 1 / 0; cross-seed span match > 0.8 — **0 / 0 / 0 of 24** (best pair 0.58); split-stable — 2 / 1 / 3 of 8.

### 3b. FineWeb prompts (off-distribution: real text, 32 tokens, no padding)

| w | median held-out PR | held-out energy / baseline | E1 top-1 | E1 top-5 | E1 all-AJ-MLPs | E1 source MLP | E1 attention | E2 span align | E2 read-off ratio | E3 top-1 @0.05 | E3 random-5 | E4 seeds (pair / span) | E4 split (pair / span) | top-5 Jaccard |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 190 | 1.00 | 0.09 | 0.12 | 0.38 | 0.78 | 0.71 | 0.42 | 1.81 | 0.05 | 0.00 | 0.38 / 0.32 | 0.12 / 0.12 | 0.17 |
| 0.1 | 12.5 | 0.98 | 0.41 | 0.67 | 0.78 | 0.27 | 0.73 | 0.34 | 1.85 | 0.25 | 0.00 | 0.54 / 0.42 | 0.44 / 0.39 | 0.49 |
| 1 | 10.2 | 0.97 | 0.51 | 0.68 | 0.79 | 0.22 | 0.74 | 0.32 | 2.08 | 0.26 | 0.00 | 0.62 / 0.54 | 0.43 / 0.38 | 0.39 |

Per-factor counts: PR ≤ 5 — 2 / 6 / 7; top-1 completeness in [0.5, 1.5] — 2 / 6 / 10; |top-1 completeness| > 2 (cancelling pathways) — 2 / 3 / 3; raw neuron beats the factor — 15 / 21 / 21; cross-seed span match > 0.8 — 5 / 7 / **10 of 24** (ten pairs at 0.89–1.00: the same few factors recur in every seed); split-stable — 1 / 3 / 2 of 8.

### What the two runs say

1. **The PR drop replicates on AJ's data and shrinks off it.** AdvBench: 386 → 3 with 66–70% of the energy (AJ's "70–100%"). FineWeb: 190 → 10–12 with 97–98% of the energy. The 400× figure is specific to the prompt distribution (32-token AdvBench targets are mostly EOS padding at the three scored positions).
2. **The penalty concentrates about half the factors; the other half stay diffuse.** On AdvBench 10–11 of 24 penalised factors have PR ≤ 5 and top-1 completeness 0.5–0.75; the rest keep PR 15–130 with 70–95% of their interaction in the *source-block MLP*, outside the PR range (the scale-invariance loophole, live in half the dictionary). Medians (top-1 0.12–0.18) hide this bimodality.
3. **Attention carries the majority of every factor.** Freezing all attention in the span removes 52–59% (AdvBench) and 71–74% (FineWeb) of the interaction for penalised and unpenalised factors alike. The interactions the method finds are attention-mediated interactions that *also* pass through one MLP unit; PR ≈ 3 says nothing about the attention part. (The groups overlap: freezing everything gives exactly 1.00 while the parts sum to ~2.)
4. **Concentrated factors are partly neuron-aligned, and on FineWeb a raw neuron is the better factor.** Span alignment to the best unit is 0.32–0.47 against a null of 0.01. On AdvBench the best unit, scored as a factor with no optimisation, recovers 34–43% of the factor's energy; on FineWeb it recovers **1.8–2.1×** the factor's energy (21 of 24 factors are beaten by a single neuron). The fitting is not finding the best single-neuron factor even when it lands near one.
5. **Finite ablations agree with the derivative picture where the derivative is valid.** Patching the top unit to clean at 5% of the residual norm removes 0.4–0.65 of the finite interaction for concentrated factors and ~0 for random units; at 20% the response is non-quadratic (fractions from −15 to +2) and uninformative.
6. **Stability depends on the data more than on the penalty, and the fit is init-dominated.** On AdvBench no factor recurs across seeds (best pairwise span match 0.58; null 0.0001), at any weight — by the README's own table this makes per-factor retention numbers on that data uninterpretable. On FineWeb a handful of factors recur exactly (ten seed-pairs at ≥ 0.89 at w = 1) and the rest do not. The split test as specified (both halves from seed 0) is confounded with shared initialisation: split matches (0.4–0.8) exceed cross-seed matches (0.06–0.2) on AdvBench, the opposite of what data-stability would give. The energy traces are still rising at iteration 10 (e.g. 1.61 → 1.95 × 10⁻⁴ over the last three iterations at w = 0), so the fixed-point iteration has not converged at AJ's iteration count. **A 30-iteration run converges (energy flat over the last five iterations) and settles the point** (`results/conv30_bilinear_advbench.json`, fits only, AdvBench, seeds 0 1 2, w = 0 and 1):

| w | seed | median held-out PR | held-out energy | cross-seed span matches (all 3 pairs × 8 factors, sorted) |
|---|---|---|---|---|
| 0 | 0 / 1 / 2 | 384 / 137 / 471 | 2.33 / 2.23 / 2.23 × 10⁻⁴ | 1.00 1.00 1.00 0.60 0.54 0.51 0.48 0.25 … then ≤ 0.19 |
| 1 | 0 / 1 / 2 | **2.8 / 97.7 / 1.6** | 1.76 / 1.58 / 1.68 × 10⁻⁴ | 1.00 1.00 1.00 0.52 0.50 0.30 then ≤ 0.04 |

Exactly one factor recurs in every seed (three pairwise matches at 1.00, at both weights: the top-energy factor), and the other seven are init-dependent even after convergence. The penalty's effect is itself seed-dependent: with the same weight and scale, seed 1's dictionary stays at median PR 98 while seeds 0 and 2 reach 2–3, and its normalised-PR trace rises through the fit (0.004 → 0.011) as the energy term wins. Per-factor claims about "the" PR-regularised dictionary are therefore claims about one factor plus seven that depend on the random start.
7. **cos(l, r) = 1.00 for every factor at every weight** (median), including unpenalised ones at this scale: the asymmetric parametrisation is redundant (§1).

## 4. Verdict

_(pending the full runs)_
