# MLP2 CMR v1 finite-validation findings

## Decision

The native-product K=512 compression grammar **fails validation**.  Replication is
not authorized.  The result is a receipt-backed held-out consequence test over 192
documents and 29,904 scored positions.

The original GPU run completed all 48 batches but its receipt-last check rejected a
SHA-256 metadata leaf named `rows`.  The immutable v1 failure is preserved.  A
separately audited CPU-only v1R finalizer then replayed the exact hash-pinned ledger,
recomputed every gate, and published the valid decision without another model run.

## What was replaced?

Native MLP2 has 4,608 bilinear product channels.  Each channel computes the product
of two linear functions of the residual stream, and the Down matrix maps those
products back to the 1,152-dimensional residual stream.  Every candidate retained
exactly 512 native channels and physically stored only their Left, Right, and Down
coefficients.  The average contribution of the omitted 4,096 channels was folded
into one bias.  Thus the candidate executed 512 products per token rather than all
4,608 and masking afterward.

SUFFIX chose the 512 channels whose infinitesimal edits most affected the downstream
suffix.  LOCAL and RMS used local output-size criteria; MASS used activation/output
mass; DERANGED and HASH_RANDOM were equal-price controls.  ZERO removed the complete
MLP2 write and was a damage baseline, not a compressed program.

## Main whole-model numbers

All values below are on the full 192-document role.  `dCE` is candidate minus native
cross-entropy, so smaller is better; teacher KL compares complete next-token
distributions; NRMSE is centered-logit relative error; agreement is the fraction of
positions with the same top-1 token as native.

| Arm | dCE (nat) | teacher KL | logit NRMSE | top-1 agreement |
|---|---:|---:|---:|---:|
| ZERO | 0.162349 | 0.169057 | 0.244575 | 77.735% |
| LOCAL | 0.265280 | 0.273132 | 0.319278 | 73.445% |
| RMS | 0.265491 | 0.272831 | 0.319107 | 73.545% |
| SUFFIX | 0.289200 | 0.297110 | 0.331499 | 72.596% |
| MASS | 0.307005 | 0.313543 | 0.337904 | 71.686% |
| HASH_RANDOM | 0.366290 | 0.374816 | 0.363301 | 69.098% |
| DERANGED | 0.396381 | 0.402987 | 0.373707 | 68.409% |

SUFFIX therefore fails every scientific faithfulness gate: `|dCE| <= 0.02`, KL
`<= 0.02`, NRMSE `<= 0.10`, agreement `>= 0.90`, the margin certificate, every-cell
CE, signed finite-edit geometry, and superiority over all equal-price controls.  It
passes the engineering gates: exact price/support, materialization, gauge replay,
call ledger, and float32/float64 precision.

The simultaneous document-bootstrap lower bound for SUFFIX's minimum relative KL
improvement over controls is `-0.10123`, below the required `+0.05`.  At the point
estimate, SUFFIX is 8.78% worse than LOCAL and 8.90% worse than RMS, although it is
better than MASS, HASH_RANDOM, and DERANGED.  The expensive suffix selector contains
real signal, but it is not the right selection rule for a faithful native-channel
program.

## Robustness to doubling the data

The central conclusion is stable at the nested 48/96/192-document prefixes:

| Arm | dCE at 48 | dCE at 96 | dCE at 192 |
|---|---:|---:|---:|
| ZERO | 0.165733 | 0.162466 | 0.162349 |
| LOCAL | 0.265920 | 0.262660 | 0.265280 |
| RMS | 0.266877 | 0.262534 | 0.265491 |
| SUFFIX | 0.294029 | 0.290443 | 0.289200 |

This is not a small-sample reversal.  ZERO remains much better than every K=512 arm,
and SUFFIX remains worse than LOCAL/RMS after two data doublings.

## The important structural result

Deleting MLP2 entirely is less harmful than keeping any tested 512-channel partial
write.  That means the native product channels cannot be treated as independently
valuable atoms.  The full MLP2 write depends on cancellations and coordinated groups:
retaining a subset can leave a large, systematically misbalanced write that is worse
than writing zero.

The preregistered singleton-additivity diagnostic agrees.  If omitted channels acted
independently, joint squared distortion `J` would be close to the sum of singleton
distortions `A`.  Instead,

$$
\frac{J}{A}=1.83497,
$$

far outside the accepted `[0.90,1.10]` range.  The signed finite-edit geometry is
also strongly nonlinear: the small-step/full-step cosine is only `0.261`, and the
central `+0.10` versus `-0.10` cosine is `0.0174`.  An infinitesimal downstream
tangent does not transport to deletion of 4,096 channels.

## A narrower semantic signal

SUFFIX is globally harmful, but on the preregistered copy-positive cell it changes
CE by `-0.00959` nat (a small improvement), with 92.79% native top-1 agreement.  The
same program has `+0.34697` nat CE on repeat-negative positions and `+0.29350` on
nonrepeat positions.  This suggests that the suffix selector concentrated a
copy-supporting subcomputation while discarding balancing/general-language functions.

That is potentially useful for **circuit extraction**, but it is not faithful MLP2
compression and does not yet prove selective removal or OOD transport.  A gated or
jointly balanced copy-specific program is now more plausible than treating SUFFIX as
a replacement for all of MLP2.

## What this prunes and what comes next

Pruned now:

- native-channel K sweeps as the primary strategy;
- composing a K512 MLP2 survivor with MLP0 C512, because no survivor exists;
- local MSE/energy, HOSVD, or SAE reconstruction without a finite consequence gate;
- treating tangent importance as a finite deletion certificate.

Highest-return next moves:

1. Factor **response-conditioned blocks**, allowing coordinated mixtures of native
   products and a refitted Down basis under final-logit/CE consequences.  This is the
   conditional branch preregistered for collective native-arm failure.
2. Use the ZERO result as the reference: add small balanced correction blocks to zero
   MLP2 and measure marginal final-logit value, rather than deleting channels from the
   full native write independently.
3. Test the SUFFIX coordinates as a copy-specific extracted program with an explicit
   gate and collateral controls, not as universal MLP2 compression.
4. Complete terminal extraction/removal/OOD testing of the already-localized copy
   edge.
5. Build a verified late-consumer bank so early factors can be jointly sparse across
   copy, capitalization, numeric, syntax, and entity-continuation functions.

The strict whole-model ledger does not move: this is a decisive pruning result, not
a certified simplification.

