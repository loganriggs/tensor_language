# Block-3 native-gate subset validation V1 — result and interpretation

## Result in one sentence

The activation-fitted K=256/512 shared native-gate programs contain real, stable
structure—especially in individual `uu` and `uv` replacements—but they are not faithful
all-term replacements for MLP3.  Under the frozen decision rule, family A stops and the
next experiment is the finite-suffix consequence-fitted family F.  No final rows open
and no global explanation ledger moves.

## What was tested

At Block 3, RMSNorm permits the exact decomposition

$$
z=u+v,
$$

and the bilinear MLP expands as

$$
B(z,z)=B(u,u)+B(u,v)+B(v,u)+B(v,v).
$$

One subset of K native product gates and one decoder were fitted jointly to the four
typed terms on the fit role.  Validation replaced the entire MLP3 bilinear write at all
256 positions and then reran Blocks 4--17 autonomously.  It used 192 rows from 79
source documents and a 2,000-draw source-document bootstrap.

The important metrics are:

- **summed local NRMSE:** root squared error of the complete replacement write divided
  by the native bias-free write energy; zero is exact and bias-only is one;
- **KL / bias-only KL:** final next-token distribution error divided by the error from
  removing all four bilinear terms while preserving the MLP bias; below one beats that
  omission stake and zero is exact;
- **CE delta:** candidate cross-entropy minus native-model cross-entropy, in nats per
  scored token; zero is exact and positive is worse;
- **native top-1 agreement:** fraction of positions where candidate and native choose
  the same most likely token.

## The simplicity-versus-faithfulness curve

| Program | Products/token | Exact bytes | Native storage | Local NRMSE | KL / bias-only KL | CE delta | Native top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| bias only | 0 | 4,608 | 0.0072% | 1.000 | 1.000 | +0.13028 | 79.24% |
| selected K=256 | 256 | 3,545,600 | 5.57% | 0.7603 | 1.3823 | +0.17998 | 76.34% |
| selected K=512 | 512 | 7,086,592 | 11.12% | 0.6842 | 0.7121 | +0.09138 | 82.27% |
| native K=4608 | 4,608 | 63,705,600 | 100% | 0 | 0 | 0 | 100% |

This is a genuine operational simplicity curve: it prices stored bytes and executed
products, then measures both the local port and the autonomous final behavior.  It is
not yet a useful compression frontier.  K=256 is worse than bias-only on final KL and
CE despite reconstructing more local write.  K=512 improves on bias-only, but remains
far from the registered fidelity thresholds.

## Frozen eligibility outcome

| Gate | K=256 | K=512 | Required |
|---|---:|---:|---:|
| summed local NRMSE | 0.7603 | 0.6842 | <= 0.20 |
| KL / bias-only KL, point | 1.3823 | 0.7121 | <= 0.20 |
| KL / bias-only KL, q95 | 1.4381 | 0.7355 | <= 0.35 |
| CE delta, point | +0.17998 | +0.09138 | — |
| CE delta, q95 | +0.18894 | +0.09761 | <= +0.01 |
| beats matched random on point KL ratio | no | yes | yes |
| beats label permutation | yes | yes | yes |
| positive recovery for every material singleton | yes | yes | yes |
| mirror point KL ratio | 0.2250 | 0.1980 | <= 0.35 |

Neither budget is validation-eligible.  K=512 also fails the frozen downstream-null
screen because candidate KL q95 is 0.736 rather than at most 0.35.  Therefore the
registered next action is exactly:

```text
stop_activation_family_and_preregister_finite_suffix_family
```

The full 16/15 mask cube and final role do not open for family A.

## What worked

### 1. Gate selection has real signal at K=512

Relative to the matched random K=512 subset, selection improves:

- local NRMSE by `0.02094`, paired-bootstrap 90% interval
  `[0.01992, 0.02200]` in improvement magnitude;
- KL/bias ratio by `0.03992`, interval `[0.02277, 0.05786]`;
- CE by `0.00379` nat point, although its interval for selected-minus-random
  `[-0.00819, +0.00027]` narrowly includes zero.

The permuted-label control is dramatically worse: KL ratios are 4.60 and 4.48.  Thus
the selected gates are not noise, even though the selected family is insufficient.

### 2. Larger K buys stable improvements

K=512 versus K=256 improves local NRMSE by `0.07607`, KL/bias ratio by `0.67022`, and
CE by `0.08860` nat.  Their paired-bootstrap intervals are respectively
`[0.07534,0.07671]`, `[0.62818,0.71412]`, and `[0.08207,0.09529]` in improvement
magnitude.  The curve is moving in the right direction, but K=512 remains too far from
the registered interface for extrapolation to be credible.

### 3. Individual typed pathways are much easier than replacing all four together

For K=512, replacing one fitted typed term while keeping the other three native terms
recovers the following fraction of that term's omission KL:

| Term | Meaning | Recovery point | 90% bootstrap interval | Omission KL / all-term-omission KL |
|---|---|---:|---:|---:|
| `uu` | residual × residual | 85.08% | [81.53%, 87.72%] | 268.2% |
| `uv` | residual × attention | 49.60% | [48.07%, 51.00%] | 5.70% |
| `vu` | attention × residual | 44.37% | [42.58%, 46.01%] | 4.88% |
| `vv` | attention × attention | 62.99% | [61.80%, 64.02%] | 3.61% |

`uu` and `uv` are material under the frozen 5% rule.  All four recoveries are positive
with tight document-bootstrap intervals.  The `uu` stake can exceed 100% of the
all-term stake because causal KL effects are not additive: removing all terms permits
cross-term cancellation and downstream nonlinear interaction.

This is the clearest constructive result.  The tensor decomposition identifies
partially replaceable pathways, while the failed all-term arm shows that independently
reasonable pieces do not automatically compose.

### 4. Downstream dynamics are structured, not simply error accumulation

For K=512 candidate replacement, native-state NRMSE changes from 0.429 after Block 3,
to 0.128 after Block 8, then rises to 0.164 after Block 17.  Relative squared-error
propagation falls to 0.141 at cut 8 but returns to 1.061 at cut 17.  Later computation
temporarily suppresses the perturbation and then makes it consequential again.  This
is why a cut-8-only reconstruction would give a misleadingly optimistic answer.

The K=512 mirror has KL/bias ratio 0.198, CE delta +0.02596, and 90.59% top-1
agreement, much better than the candidate.  The mirror is a diagnostic, not a deployable
compression: it computes `2*native-candidate` and therefore still requires the native
write.  Its asymmetry says the suffix treats the two signs of the same local error very
differently.  That is direct evidence that the next fit should optimize downstream
consequence, not symmetric local MSE alone.

## Is 79 documents enough for this decision?

It is enough to reject family A at these two budgets.  The document-bootstrap intervals
are narrow relative to the margins: K=512 local q95 is 0.686 versus a 0.20 threshold,
KL-ratio q95 is 0.736 versus 0.35, and CE q95 is 0.0976 versus 0.01.  Doubling the data
could refine estimates, but sampling error would need to shrink and the point estimates
would also need to move by factors of roughly 3--9 to reverse the decision.  The stable
singleton recoveries and K=512-versus-random KL improvement are also resolved across
source documents.  This is not a marginal statistical failure.

## What the mathematics contributed

- Exact RMS polarization defined four causal pathways rather than arbitrary neurons.
- Native product-gate coordinates and gauge balancing made the gate subset invariant
  to reciprocal rescaling of its two factors.
- The polynomial sum exposed a composition failure invisible in stacked-term error.
- Matched omissions quantified pathway materiality without treating coefficient energy
  as causal importance.
- The mirror intervention detected sign asymmetry in the nonlinear downstream suffix.
- Autonomous suffix cuts separated transient error suppression from final recovery.
- Literal tensor-program bytes and product counts turned “simple” into an executable
  cost rather than a visual or sparsity judgment.

## Next move

Family F keeps the same finite native-gate grammar and K=256/512 executable prices, but
fits gate scores against frozen Blocks 4--17 teacher consequences rather than four
stacked local writes.  Its implementation amendment must freeze optimizer, seed,
schedule, discretization, affine calibration, controls, and resource limits before any
teacher-logit fit.  Cheap scalar/constant calibration belongs as an explicit baseline,
because it changes almost no description or execution cost; it cannot be tuned on this
already observed validation role.
