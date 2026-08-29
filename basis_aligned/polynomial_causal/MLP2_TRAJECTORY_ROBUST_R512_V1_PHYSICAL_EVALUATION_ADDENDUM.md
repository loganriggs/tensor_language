# MLP2 trajectory-robust rank-512 v1 — physical evaluation addendum

**Frozen before source audit, row selection, or evaluation access.**

This addendum instantiates the already-preregistered physical evaluation in
`MLP2_TRAJECTORY_ROBUST_R512_V1_PREREGISTRATION.md`; it does not change its arms,
contrasts, thresholds, bootstrap, or scientific claim.

## Fresh rows

The row freezer reserves a new 192-document `EVALUATION` role beginning at pinned
FineWeb document index 120,000.  Every source document, dataset
index, full token row, and 32-token prefix must be disjoint from the complete row
registry.  Rows have length 257 and positions 64--255 are scored.  The reservation is
outcome-blind and loads neither model nor program.

## Physical arms

The evaluator runs the exact eight arms already registered, in this order:

1. `NATIVE`
2. `C512`
3. `FULL512`
4. `C512_FULL512`
5. `CONTINUE512`
6. `C512_CONTINUE512`
7. `ROBUST512`
8. `C512_ROBUST512`

All arms use the same rows, batch boundaries, model bytes, and scoring positions.
Candidate MLPs replace the entire native residual write at their site.  C512 includes
its fitted intercept and the native MLP0 Down bias exactly once.  Each rank-512 MLP2
program includes its own complete fitted bias and makes no native MLP2 call.
Stored candidate coefficients are float32; deployment follows the established
`RankBilinear` contract and casts them to the BF16 residual-state dtype.  Each MLP2
candidate has exactly 512 products, 1,770,624 stored scalars, 7,082,496 stored bytes,
three dense matrix multiplies per token, and zero native MLP2 calls.  The unchanged
C512 program price and hashes are reported as a separate parent rather than hidden in
the MLP2 price.

## Prefix stability made explicit

The preregistered 48/96/192 summaries are all reported.  The inherited stability gate
passes only if every nonnative rank-512 MLP2 arm—standalone and with C512—has both
192-document dCE and KL within 0.01 nat of its 96-document value.  The 48-document
values remain diagnostic because treating both nested prefixes as independent gates
would double-use the same early documents without a registered multiplicity rule.

## Nine simultaneous contrasts

The evaluator uses common source-document bootstrap indices, 10,000 draws, seed
`2026082942`, and Bonferroni two-sided quantiles 0.0027777778/0.9972222222.
The exact nine quantities are:

1. fresh `FULL512` factorial interaction;
2. $0.5|I_{FULL}|-|I_{ROBUST}|$;
3. $|I_{ROBUST}|$;
4. combined-arm dCE improvement of ROBUST over FULL;
5. combined-arm KL improvement of ROBUST over FULL;
6. ROBUST standalone dCE noninferiority margin versus FULL, defined as
   $0.005+\Delta CE_{FULL}-\Delta CE_{ROBUST}$;
7. the analogous standalone KL noninferiority margin;
8. combined-arm dCE improvement of ROBUST over CONTINUE;
9. combined-arm KL improvement of ROBUST over CONTINUE, diagnostic only.

Absolute interactions are computed *inside each bootstrap draw* after averaging that
draw's document-level factorial effects.  Historical interaction values are never
inserted into the fresh contrasts.

Each document contributes equally to the bootstrap.  Within a document, dCE and KL
are totals divided by the fixed 192 scored positions.  Whole-population summaries sum
the same sufficient statistics and divide by their token count.  Scientific gates use
the strict comparisons written in the parent preregistration (`>` where a positive
lower bound is required, `>=` for the 0.005 improvement/noninferiority bars, and `<=`
for the absolute-interaction equivalence bar).

The numerical `optimization_inconclusive` rule is evaluated before assigning a
scientific rejection label.  It does not turn a failed physical gate into a pass and
cannot promote a program; it only distinguishes unfinished optimization from a
completed negative.  Integrity failures are always terminal implementation failures,
not scientific outcomes.

The sufficient-statistics ledger contains exactly one float64 `[192,9]` tensor per
arm: native NLL sum, candidate NLL sum, teacher KL sum, centered-logit error energy,
native centered-logit energy, top-1 agreement count, native-correct count,
candidate-correct count, and scored-token count.  It also contains the exact call
census and checkpoint identity.  No tokens or logits are published.

## Lifecycle

The freezer and evaluator must be committed and pushed, independently source-audited
with `outcome_access=false`, and hash-bound before row selection.  The evaluator then
publishes authority, sufficient-statistics ledger, result, and receipt last under a
create-only namespace.  A failure is preserved; no changed runner may reuse an opened
evaluation role.
