# Handoff: do PR-regularized DCT factors find real sparse circuits?

## Context

AJ (ajskateboarder) extended Deep Causal Transcoders (DCTs) with a
participation-ratio (PR) penalty. DCTs find input directions (l, r) at a
source layer whose mixed second derivative drives an output direction u at a
target layer. The penalty asks that this interaction be carried by as few
intermediate MLP hidden units as possible.

- Notes: https://ajskateboarder.com/notes/shallow%20weight%20circuits
- Code: https://github.com/ajskateboarder/redesigned-octo-couscous
- DCT background: https://www.lesswrong.com/posts/fSRg5qs9TPbNy3sm5

**Claim to test:** on bilinear-MLP transformers, the penalty drops median PR
from ~300–400 to ~1, while many factors keep 70–100% of their causal strength.
This would mean interactions are routed through single neurons.

**Why it might not mean that:**

- **PR is scale-invariant**, and AJ's PR range excludes the source block's MLP
  and all attention. A factor can get PR ≈ 1 while almost none of its
  interaction flows through the measured units.
- **Bilinear hidden units are themselves rank-1 interactions.** PR ≈ 1 may just
  mean "the factor points at one neuron."
- **The evidence so far is thin.** Strength retention is measured on the
  objective, not by intervention, on 4 training and 8 held-out prompts.
  Nobody has checked whether the factors are stable across seeds or data.

See `BACKGROUND.md` for the reasoning and the toy demonstrations.

## Layout

```
circuit_checks/
  core.py        mixed_hessian, participation_ratio, FreezeSpec (freeze/patch units to clean)
  checks.py      E1 pathway_completeness, E2 neuron readoff/alignment,
                 E3 ablation_effect, E4 match_factors/jaccard
  toy.py         norm-free bilinear residual stack; exact weight-space formulas
  tensorgpt.py   adapter for AJ's TensorGPT (span interface, freezing, pullbacks)
tests/test_toy.py              7 sanity tests on the toy (CPU, seconds)
scripts/check_adapter_parity.py   adapter vs AJ's TensorMiddleSpan; E1 smoke test
scripts/run_tensorgpt_checks.py   full E1–E4 run on AJ's fitted factors -> JSON
```

Status: the tests pass. The adapter and the full runner were verified end to
end on a tiny random TensorGPT on CPU, where forward parity with AJ's
`TensorMiddleSpan` was exact. They have **not** been run on real checkpoints.
Step 0 below catches adapter problems.

## Setup

```bash
git clone https://github.com/ajskateboarder/redesigned-octo-couscous aj-repo
pip install torch einops huggingface_hub transformers scipy tqdm numpy
```

- **Data.** AJ's scripts read `harmful_behaviors.csv` from the repo root, but it
  isn't committed. It's AdvBench (`data/advbench/harmful_behaviors.csv` in the
  llm-attacks repo); the `target` column is used. Alternatively pass
  `--texts-file prompts.txt` (one prompt per line).
- **Second held-out set.** AJ's success criterion is generalization *outside*
  the training distribution. Also build a held-out set from a different
  distribution (plain web text, code) and run E1/E3 on it.
- **Don't edit AJ's files.** `pr_penalty_experiment.py` has a broken relative
  import (`from ..tensor_model import ...`); the runner patches it in memory.
- **Hardware.** Use a GPU for real runs. All forward-mode AD needs the SDPA
  MATH backend; the scripts set it.

## Step 0: validate the instruments

```bash
python tests/test_toy.py                                   # expect 7 PASS
python scripts/check_adapter_parity.py --repo aj-repo      # tiny model: parity err ~0
python scripts/check_adapter_parity.py --repo aj-repo \
    --hf Elriggs/gpt2-bilinear-18l-9h-1152embd --device cuda --source 8 --target 12
python scripts/run_tensorgpt_checks.py --repo aj-repo --tiny   # pipeline smoke test
```

Parity on the real model must be ~0 relative error, and "all MLPs + attention
frozen" must give a score of ~0. If either fails, fix `circuit_checks/tensorgpt.py`
before anything else. Likely culprits are dtype (the rotary caches are bf16) or
the `first_values` handling.

## Step 1: reproduce (E0)

1. Run the runner at AJ's scale first: `--train-contexts 4 --heldout-contexts 8
   --factors 4 --iterations 5 --penalty-weights 0 1 10 100`. Confirm the PR drop
   and energy retention replicate.
2. Scale up to the defaults (32 train, 64 held-out, 8 factors, 10 iterations,
   penalty weights 0 / 0.1 / 1, seeds 0 1 2).
3. Record whether the headline numbers survive. Check the convergence traces
   (`objective_values`, `normalized_pr_values` on AJ's dictionary objects); see
   note 5 below.

## The checks

All of these are computed per factor on held-out contexts, and all are written
by `run_tensorgpt_checks.py`.

**E1: pathway completeness.** Freeze a group of units at their clean values and
measure what fraction of u·H[l, r] disappears. Groups:

- top-k units by |mixed Hessian| in AJ's PR range
- k random units in that range (excluding the top ones)
- all MLPs in AJ's range
- the source-block MLP
- all attention

1 means the whole interaction flows through the group; ~0 means the group is
irrelevant. Values are signed and can fall outside [0, 1], because pathways can
cancel.

**E2: neuron readoff.** For bilinear models only (skipped automatically for
gated/SwiGLU):

- Pull each unit's Left/Right read vectors back to the source layer (J⊤a, J⊤b).
- Report how well each factor's (l, r) aligns with its best unit per layer,
  compared against `E2_null`, the alignment a random (l, r) pair gets.
- Score the top unit *as if it were a factor*, with no optimization, and report
  `energy_ratio`. If a raw neuron matches the factor's energy, the optimization
  isn't adding much.

**E3: finite ablations on held-out prompts.** Compute the finite-size
interaction f(αl+βr) − f(αl) − f(βr) + f(0), projected on u, with the top-k
units patched to clean versus k random units. Scales are set as a fraction of
the mean residual norm (`--ablation-scales`). This is the intervention version
of AJ's "retains causal strength."

**E4: stability.** Hungarian-match factors across seeds, and across fits on
disjoint halves of the training data. Compare against `random_null_mean`. Also
report the Jaccard overlap of matched factors' top-5 units.

**E5 (optional): weight-space version.** For a norm-free bilinear stack, the
data-averaged mixed Hessian depends on the data only through E[x] and E[xxᵀ]
(for two layers). `toy.expected_mixed_hessian_2layer` implements this, and
the tests verify it exactly. The real TensorGPT has an RMSNorm before every
MLP, so this is only exact for norm-free models. If one is available, fit the
DCT/PR objective using moments instead of datapoints, and compare the
resulting factors with the datapoint fits via E4 matching.

## Reading the results

These thresholds are heuristics; report the raw numbers regardless.

| Pattern | Interpretation |
|---|---|
| PR ≈ 1 but top-1 completeness low; source-MLP or attention completeness high | **Loophole.** The interaction lives outside the measured units, and PR is concentrating a small remainder. |
| PR ≈ 1, completeness high, E2 alignment near 1, `energy_ratio` ≈ 1 | **The factor is a neuron.** The method reduces to reading out individual bilinear units, which is less interesting unless the chain spans layers. |
| PR ≈ 1, completeness high, alignment near `E2_null` | Sparse mediation of inputs that aren't neuron-aligned. **This is the interesting case.** |
| Above, plus E3 top-k ≫ random on held-out (and on the off-distribution set), plus E4 stable | Strong evidence of real sparse circuits. |
| E4 matches near the null | Factors are optimization artifacts, so per-factor retention numbers aren't meaningful. |

Run the bilinear architecture first. Then run `swiglu`, and the `*-attn`
variants (AJ reports his method does best on softmax attention; the E1
attention group shows whether that's because attention is carrying the
interaction).

## Things noticed in AJ's code (confirm, don't assume)

1. **The PR range excludes block `source`'s MLP and all attention.** The
   perturbation enters at block `source`, so its MLP is the most direct place
   for a bilinear interaction. This is exactly where the scale-invariance
   loophole would bite.
2. **Very small defaults:** 4 train / 8 held-out prompts, 4 factors,
   5 iterations.
3. **The "exactly second order" claim needs qualifying.** RMSNorm precedes every
   MLP, so Lx⊙Rx is exactly second order in the MLP's normalized input, not in
   the residual at the source layer. The mixed Hessian is input-dependent even
   for a single block.
4. **PR is computed on hidden activations averaged over the last 3 positions.**
   Signed cancellation across positions can distort it. Consider also reporting
   per-position PR.
5. **The fitting update is a fixed-point iteration, not a gradient step.** It
   replaces U, L, R with the normalized gradient (β = 1), so the penalty's
   gradient enters through that normalization rather than as an ordinary
   Lagrangian term. Check convergence, and try β < 1 as a robustness check.
6. **Repo hygiene:** the relative-import bug in `pr_penalty_experiment.py`, and
   the missing CSV.

## Deliverable

Write a `RESULTS.md` containing:

- A table per architecture × penalty weight: median held-out PR; held-out
  energy vs. baseline; E1 (top-1, top-5, all-AJ-range, source MLP, attention);
  E2 (alignment vs. null, energy ratio); E3 (top-k vs. random, at each scale);
  E4 (seed and split matching vs. null, Jaccard).
- A short verdict using the table above.
- Negative results stated plainly.

Keep the raw JSON alongside it.
