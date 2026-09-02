# Hourly strategic review — 2026-09-02 13:33 UTC

## Circuit targets and full goal

The program still seeks a smaller executable tensor program that is jointly predictive on fresh and shifted text,
composable when pieces are installed together, selectively manipulable, and simpler under literal storage, compute,
edge, state, and program prices. A useful decomposition must eventually provide:

1. a computational description of what is read, what operation is performed, what is written, and what later uses it;
2. grouping across heads/MLPs when downstream computation treats their parts as the same variable;
3. splitting within one native module when its parts serve different computations;
4. held-out and OOD prediction of activation and causal effect;
5. an executable extracted circuit or a precise interface plus background;
6. selective removal, swapping, or editing without unrelated damage, including redundancy and interaction checks;
7. stable identification under document/corpus splits, fitting restarts, and harmless coordinate changes; and
8. predictable composition and reuse across larger circuits.

Lower rank, fewer stored values, quantization, reconstruction error, and aggregate CE are not circuit discoveries.
They can price or implement an already identified object, or serve as matched-capacity controls, but cannot choose the
semantic pieces.

## What changed since 12:31

- The equality-query mirror test killed the proposed orthogonal geometric component: its complete interactions were
  around `10^-7`, five orders below the material boundary. The local channel is effectively one-dimensional; the
  intervention-coordinate disagreement is nonlinear magnitude recombination, not two distinct directions.
- Rung485 showed that neither MLP1's Left nor Right side alone predicts a complete MLP0 branch response. T and I have
  nearly identical average Left/Right/interaction profiles but per-example effects do not transfer under one scale.
  Current-token means also worsen held-out prediction.
- Rung486 exactly decomposed the direct MLP0 write, attention1 write, MLP1 write, and all interactions. The MLP1
  singleton is largest, not the direct×attention1 term; an initially published mask-order gloss was corrected.
  Previous×current-token means also worsen held-out prediction, closing the categorical token-table route at this
  downstream grain.
- Rung487 tested the exact quadratic secant `B(change, midpoint)`. T and I independently passed a bidirectional
  midpoint interchange in both discovery halves while T--C and C--I did not. This is direct evidence for distinct
  branch changes interpreted through a shared continuous MLP1 state. The run is not claimable because two deployed
  BF16 relative-squared checks exceeded the registered `1e-5` tolerance; its float32 identity, prefix replay, calls,
  injections, and end-to-end branch identity passed.
- Rung488 was preregistered and started. It preserves rung487 as failed, reruns discovery, and uses bounds derived
  from BF16 unit roundoff only for the two named checks. The original effect, graph, control, and held-out bars are
  unchanged.

## Is this still the highest-information route?

Yes, narrowly. Rung488 changes the status of a specific computational grouping: whether T and I share MLP1's
continuous multiplier on unseen documents. A pass moves a screen to held-out identification and licenses an exact
extraction/selective-swap experiment. A fail forces T and I back into separate within-branch finite-response models.
It is more informative than another product subset, token table, rank sweep, or descriptive similarity matrix.

The precision repair is scientifically acceptable only because it was separately registered, reruns discovery from
scratch, derives its changed bounds from BF16 unit roundoff, and leaves every scientific threshold fixed. Rung487 is
not retroactively passed.

## Confound audit

- **Precision floor:** the repaired limits apply only to two BF16 bookkeeping identities; float32 algebra and the
  physically deployed complete-branch identity retain their original strict limits.
- **Baseline mixing:** native, branch-absent, and injected arms are rebuilt within one process. Old and new causal
  effects are not numerically bridged across the known cross-process shift.
- **Post-selection:** the only allowed edge is frozen as T--I CONTEXT before the rerun. Extra T--C or C--I edges fail
  rather than expanding the graph.
- **Average-profile trap:** selection uses per-token physical CE effects in both directions and halves, not average
  seven-term shapes.
- **Position-insensitive similarity:** sixteen shifted-position controls must remain worse than the same-position
  write.
- **Nonlinear composition:** the intervention injects the exact finite MLP1 secant and recomputes layers2--17; it
  does not add singleton CE effects.
- **Shared token difficulty:** comparisons predict the same target effect under cross-factor swaps; no token lookup
  or circuit-member average selects the result.
- **Validation leakage:** documents500:1000 open only if all discovery gates independently pass.

## Ranked next moves and killing evidence

1. **Score rung488 and follow its frozen fork.** This directly advances cross-module grouping, held-out prediction,
   and selective factor interchange. It dies if discovery does not reproduce exactly the T--I context edge, either
   repaired instrument fails, or either validation quarter misses any original `.80/.50` or control clause.
2. **If rung488 passes, extract the shared T/I midpoint reader.** Replace the two branch-specific midpoint uses by a
   single explicit reader interface, test cross-document factor swaps, and measure unrelated T/C/I and equality
   effects. It dies if the installed shared interface cannot reproduce the two native effects or causes comparable C
   or off-target damage.
3. **Test the site-graded T/I filtration.** Delete only the T-minus-I component entering attention1 and separately at
   MLP1-total. The existing Gram calculation predicts large damage at attention1 but small damage after MLP1. It dies
   if the two observation points lose proportionally similar effects. This can distinguish a true depth-dependent
   merge from the current pairwise interchange result.
4. **If rung488 fails, fit separate within-branch integrated finite-response readers.** Couple each exact secant to
   the suffix response along that secant and test prediction on held-out documents. It dies if the state-conditioned
   model cannot beat branch-wise constant and shifted-state controls without using evaluation labels.
5. **Order-polytope scalar-law test on the equality-query circuit.** This CPU-only falsifier checks whether one
   additive latent scalar can reproduce each token's full eight-subset ordering. It is useful for composition, but is
   secondary while the T/I held-out decision is live. It dies if feasibility is near the independence/null rate.

## Live continuation

Rung488 is active in the managed GPU runner. Its result, not the attractive rung487 screen, chooses the successor.
The queue will not be allowed to drain without a scored receipt and an actually started result-dependent experiment.
The next three-hour mathematical review is due after the 13:10 review's three-hour interval, not at the older 11:14
clock.
