# Hourly strategic review — 2026-08-29 23:20 UTC

## Bottom line

Two control-passing numerical experiments changed the shipped-program compression
picture.

First, a 133.8-second per-site mean-row screen found that table content in the current
36-site program is extremely back-loaded. Across the sampled MLPs, MLP16 and MLP17
account for roughly 96% of the summed single-site CE cost: replacing their tables by
mean rows costs about `0.366` and `0.823` nat on average, versus a combined `0.055` nat
for the eight sampled sites from MLP0 through MLP14. The preregistered claim that all
of MLP0/2/4 cost under 0.002 nat failed because MLP4 costs `0.0039--0.0049` nat.

Second, a 445.4-second rank curve at MLP16/17 passed every registered predicate and
control. Reducing rank 768 to 128 costs `0.00570` nat pooled at MLP16 and `0.00914` at
MLP17. Reducing both costs `0.01862`, about `0.0038` nat more than the sum. Rank 384 is
substantially better than rank 128, but still costs more CE than its storage saving is
worth under the frozen `0.010 nat per 100M values` price rule.

The conclusion is not “late layers are simple.” It is that the shipped compiler needs
a **causal, site-specific rank allocation**: high capacity late, possible truncation
early. Fit-table energy is the wrong allocator; its previous equal-price allocation
gave MLP16/17 rank 64 and lost `0.019--0.023` nat versus uniform rank 512.

## Fraction of the model explained

The strict whole-model ledger remains unchanged:

- structural intervention coverage: **36/36 sites**;
- certified storage removal: **5.348245316%**;
- named causal CE: **10.923302467%**;
- unexplained causal CE: **4.72714 nat / 89.077%**;
- terminal extracted/removed/OOD-certified actions: **0/68**.

The new results improve the compression strategy for an already lossy shipped table
program; they do not explain the missing 89.077% of native-model causal behavior. The
all-table program's CE is about `5.94` versus native `3.14` on these pooled roles, so
optimizing its ranks cannot substitute for recovering the missing interfaces.

## New computations

For site $j$ and rank $r$, the rank cost is

$$
\Delta_{j,r}=CE(\text{shipped program with table }j\text{ truncated to }r)
-CE(\text{rank-768 shipped program}).
$$

| Arm | skip7000 | skip11000 | skip1200 | pooled paired mean |
|---|---:|---:|---:|---:|
| MLP16 rank 128 | 0.005649 | 0.006078 | 0.005034 | 0.005697 |
| MLP16 rank 384 | 0.002589 | 0.001986 | 0.001247 | — |
| MLP17 rank 128 | 0.010063 | 0.008574 | 0.008438 | 0.009142 |
| MLP17 rank 384 | 0.002848 | 0.002882 | 0.001934 | — |
| both rank 128 | 0.019413 | 0.018514 | 0.017269 | 0.018624 |

The joint mixed remainder

$$
\Delta_{\{16,17\},128}-\Delta_{16,128}-\Delta_{17,128}
$$

is `0.003701 / 0.003862 / 0.003797` nat. This again shows why independent local rank
curves must be followed by a joint finite validation.

## Largest remaining gaps

1. **Native downstream interface metric.** Raw MLP2 write MSE still does not predict
   composition with MLP0-C512.
2. **Causal rank curves for the other sites.** We have a mean-row profile at ten MLPs
   and rank curves at only MLP2/16/17, not a full 36-site causal allocation.
3. **Missing live-consumer computation.** The shipped-table program removes the very
   attention interfaces that make MLP2 content worth 1.408 nat in partial compilation.
4. **Early composition.** No controlled C512×MLP1×MLP2 factorial has been closed.
5. **Semantic consumers and OOD.** Copy is partly localized, but capitalization,
   numeric formatting, syntax, and entity circuits are not verified consumers.
6. **Terminal editability.** No extracted behavior survives selective removal,
   collateral checks, and OOD transport end to end.

## Candidate pruning

- **Keep causal site-specific rank allocation.** It directly improves executable cost,
  has cheap falsifiers, and the depth/rank results show strong heterogeneity.
- **Keep the finite MLP2 error-Rayleigh pilot.** It targets native downstream
  consequences and whole-model composition rather than the already-compiled artifact.
- **Keep mixed-functional factorization.** The replicated MLP0×MLP2 and MLP16×MLP17
  interactions show that independent component curves do not compose automatically.
- **Keep the C512×MLP1×MLP2 factorial.** It distinguishes a specific MLP0→MLP2 fault
  from a general early-state interface failure.
- **Keep consumer quotient/Hankel work, after expanding consumers.** It remains the
  best route to semantic prediction and editing.
- **Prune fit-energy rank allocation.** Its equal-price real experiment already lost
  to uniform rank 512 on all roles and allocated least capacity to the most causal sites.
- **Prune uniform rank reduction.** Late-site rank 128 fails the frozen price rule.
- **Prune mean-row deletion as compression.** Even sampled early deletions cost much
  more CE than their saved storage is worth, and joint loss is super-additive.
- **Prune more unweighted local-MSE fitting and sparse document gates.** Both have
  receipt-backed failures.

## Ranked top five

1. **Execute the finite MLP2 error-Rayleigh validity pilot.** This has the highest
   whole-model information gain: it can validate or kill the proposed downstream
   metric before another fit and must predict the actual finite C512×MLP2 interaction.
2. **Measure causal marginal rank curves across the remaining shipped sites and solve
   a literal-price allocation.** Then validate the selected ranks jointly because
   MLP16×MLP17 is super-additive.
3. **Fit the mixed MLP0×MLP2 intervention functional if the Rayleigh pilot predicts
   finite effects.** Otherwise prune it before training.
4. **Run the controlled C512×MLP1×MLP2 factorial.** This is the smallest test of a
   composable early-layer quotient.
5. **Add verified capitalization/numeric/syntax/entity consumers, then test a causal
   quotient and finite Hankel realization on withheld consumers/compositions.**

## Executed action and CPU work

The shipped-program depth profile and the MLP16/17 rank curves both completed with
numerical artifacts and passing controls; neither is an unrun runner. During the GPU
rank job, the pure statistical core of the preregistered MLP2 error-Rayleigh pilot was
implemented and tested (`6 passed`): symmetric finite JVPs, exact categorical-Fisher
quadratic, teacher KL, attention-response energy, Spearman/tangent gates, held-out
predictor gates, and finite interaction gates. This code opens no model outcome.

The eight-hour plan expired at 12:00Z. Family F remains a fit negative; its E1--E4
receipt/pruning state is unchanged, and no historical checkbox is counted as progress.
