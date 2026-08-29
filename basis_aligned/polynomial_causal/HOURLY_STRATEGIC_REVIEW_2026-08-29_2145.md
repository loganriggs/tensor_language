# Strategic review — 2026-08-29 21:45 UTC

## New outcome

The matched paired-trajectory MLP2 fit is receipt-complete. It used the previously
unopened 192-document TRAIN role and never opened evaluation. Runtime was 109.38 s.

Frozen FULL512 was continued in two exactly matched ways:

- `CONTINUE512`: two independently sampled native-background half-batches;
- `ROBUST512`: one native and one MLP0-C512-background half-batch.

Both use 512 bilinear products, 1,770,624 float32 coefficients, Adam, 1,200 steps,
1,024 tokens per step, and identical checkpoint rules. This control was essential:
the parent had stopped at its step cap while still improving, so comparison only to
the frozen parent would have confounded trajectory exposure with extra optimization.

## Computation and result

The state handed to MLP2 changes by `0.15078` centered NRMSE when C512 replaces
MLP0. Evaluating the exact native MLP2 on those shifted states changes its write by
`0.16091` NRMSE. Thus there is a real distribution shift at the interface.

Normalized development MSE is ordinary squared MLP2-write error divided by the
centered energy of the exact native target, computed separately for each background.

| program | native dev MSE | C512 dev MSE | native NRMSE | C512 NRMSE |
|---|---:|---:|---:|---:|
| initial frozen FULL512 | 0.481077 | 0.472432 | 0.69360 | 0.68734 |
| `CONTINUE512` | 0.448338 | 0.440127 | 0.66958 | 0.66342 |
| `ROBUST512` | **0.448066** | **0.438886** | **0.66938** | **0.66248** |

Continued training improves both backgrounds by about 6.8%. Explicit C512 exposure
adds only `0.001241` normalized MSE improvement on C512 states (0.282% relative) and
`0.000272` on native states (0.061%). Both retained checkpoints occur at step 1,200.
The last 100-step improvement is below the preregistered 1% optimization-inconclusive
threshold, so the status is `fit_complete`, not visually adjudicated “still training.”

This is a local result, not the scientific composition answer. It lowers the prior
that simple local covariate coverage will remove the `+0.008739` CE interaction, but
small local changes can have large suffix effects. The frozen eight-arm physical
factorial on registry-fresh documents remains necessary.

## Other new interface evidence

The independent context-free program line now has a replicated empirical threshold:
across the tested early-MLP substitutions, arms that also compile both attention 5
and attention 6 fall in a `1.50–2.15` nat band, while omitting either produces
`2.56–10.94` nat. The separation replicates at 16,110 covered token types with a
`0.540–0.590` nat gap. About 90–95% of that gap occurs on covered inputs, so it is
not merely the uncovered-token fallback.

Several stronger stories—an unbroken chain, a monotone path, and a special frequency
gradient—were falsified by deletion controls. The robust claim is only that attention
5 and 6 mark a downstream interface condition in this program family; their semantic
computation and sufficiency are not yet known. This strengthens the case for learning
late consumers to define early representations, without promoting an unnamed
correlational threshold into a circuit.

## Honest project balance sheet

- `36/36` sites are structurally intervenable, not semantically explained.
- `5.348245316%` of storage has certified removal.
- `10.923302467%` of the measured causal CE gap is named and recovered.
- `4.72714` nat (`89.077%`) remains unexplained under that ledger.
- `0/68` terminal extraction/removal/OOD actions pass.

Family F is receipt-complete negative. E1, E2, and E3 are negative or prospectively
pruned under their tested grammars. The copy screen found a strong causal four-head
effect but failed selective removal because off-target CE is `0.024409` versus the
`0.01` bound; literal E4.1–E4.3 remain incomplete. The eight-hour deadline passed at
12:00 UTC, so plans and infrastructure are not counted as additional evidence cells.

## Largest gaps

1. **Fresh physical consequence of the two new programs.** Local MSE barely separates
   ROBUST from CONTINUE; only the suffix can tell whether that small difference matters.
2. **Operational meaning of the attention-5/6 interface.** It is replicated as a
   condition, but not identified as copying, syntax, entity, formatting, or another
   computation.
3. **A downstream-sensitive MLP2 objective.** Local write NRMSE remains about 0.66
   while final CE was much better, confirming severe metric misalignment.
4. **Cross-module composition beyond MLP0×MLP2.** MLP1 and the attention interfaces
   do not yet have independently priced composable ports.
5. **OOD and terminal utility.** No simplified coordinate supports verified selective
   extraction, removal, and transport.

## Pruned work

- More native-only steps without the matched continuation control.
- Declaring trajectory exposure successful from the 0.28% local improvement.
- More monotone attention-prefix sweeps interpreted as a mechanism; deletion controls
  already falsified three such stories.
- Flat shared dictionaries, large-budget shared/private RRR, native-channel selection,
  and HOSVD judged only by local reconstruction.
- Treating the four copy heads as a terminal circuit before collateral damage is fixed.

## Ranked top five

1. **Fresh eight-arm physical factorial for FULL512, CONTINUE512, and ROBUST512 with
   and without C512.** Highest information gain: it separates trajectory exposure
   from extra optimization and directly measures composability at fixed price.
2. **Identify attention-5/6 consumers using behavior-specific causal panels.** Test
   copy, capitalization, number formatting, syntax, and entities with sufficiency,
   necessity, shuffle, collateral, and OOD controls. This gives early layers an
   operational output vocabulary.
3. **Same-price suffix/Fisher-weighted MLP2 fit.** If ROBUST does not beat CONTINUE,
   optimize the metric the downstream model actually reads rather than local MSE.
4. **MLP1 independent rank/grammar frontier followed by MLP0×MLP1 composition.** This
   checks whether the measured interface brittleness is MLP2-specific.
5. **Structured copy-bundle replacement.** Replace the position mean with a small
   conditional program and require the off-target CE to fall below 0.01 before any
   terminal claim.

Priority 1 is next. It requires a new registry-fresh evaluation role and a source-
closed eight-arm evaluator; the programs themselves are now frozen. No user decision
is needed.

## Receipts

- Fit result SHA256: `c2d05281f8a5303fbb492f04d6eac1fff4c2ae20d0e07b4cd535b217a4e6e636`
- Fit bundle SHA256: `79d13685a1e0f53aecc3ea1d34e0c332a55149bf6de510daab519712b6ed5856`
- Authority SHA256: `5e22a810552c500cf0a0a1e5ff080888459f5351df55357531fdfbad0197706c`
- Receipt status: `fit_complete_receipt_last_evaluation_unopened`
- Independent audit SHA256: `e0b98c3074f89372095e3bc191bc269f732dbddaf1c1f576ee6b33b29bd9fdea`

