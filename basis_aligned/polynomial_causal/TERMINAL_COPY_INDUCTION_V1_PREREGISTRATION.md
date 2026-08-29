# Terminal copy/induction v1 preregistration

Status: **source scaffold only; launch NO-GO**.  This document and its pure CPU
contract are frozen before opening an E4 row role, checkpoint, model, or outcome.
They do not authorize a model forward.

## Question and prior evidence

E4 asks whether a short behavior-anchored path can support extraction, selective
removal, and OOD transport.  Copy/continuation is the primary behavior because it has
an exact token-ID task and prior causal evidence for the six-head family

`L5H5, L7H3, L8H3, L8H4, L13H0, L14H7`.

The registered four-head prior minimal set is `L5H5, L7H3, L8H3, L8H4`.  These names
are fixed candidates, not a successful v1 result.  The late heads `L13H0` and
`L14H7` are the E4.1 short-path candidates.  Failure of the late pair while the
registered mid-layer set succeeds is a licensed negative: copy is behavior-anchored
but not terminally localized.

Capitalization is a frozen specificity-negative comparator.  The prior
boundary-capitalization arc had boundary/proper-noun specificity ratio 1.0, so a
generic capital booster cannot pass v1 merely by moving capital-token probability.
Number formatting is descriptive only in v1 and cannot select a program.

## Immutable row and label contract

Four document-disjoint roles must be created before any model forward:

1. `fit_natural`: fit ablation means, frequency bins, and any candidate program;
2. `selection_natural`: select the candidate and all thresholds;
3. `final_natural`: one-shot natural-text replication;
4. `ood_code`: separately frozen code/text-domain OOD replication.

Every row is exactly 257 tokenizer IDs.  The model receives columns `0:256`; targets
are columns `1:257`.  Only query positions `64:256` are scored.  Rows, ordered
document IDs, provenance, token tensor raw hash, tokenizer/checkpoint pins, and exact
role disjointness require receipt-last, create-only authorities.  Existing exposed
FineWeb slices, the suffix final role, E3 rows, and another experiment's held-out code
role do not authorize v1.

At query position `p`, let `q=x[p]` and `y=x[p+1]`.

- An all-positive has an earlier `k<p` with `x[k]=q` and `x[k+1]=y`.
- A negative candidate has an earlier occurrence of `q`, but no earlier occurrence
  followed by `y`.
- The registered distance is the nearest qualifying predecessor for a positive and
  the nearest same-query predecessor for a negative.
- The exact matching stratum is
  `(p//16, floor(log2(distance)), floor(log2(fit_count(q))),
  floor(log2(fit_count(y))))`.
- Within a stratum, positives and negatives are independently ordered by SHA256 of
  the frozen seed, document ID, and position, then paired without replacement.
  Unmatched positives are reported and receive no confirmatory credit.
- Off-target is every valid scored position that is neither an all-positive nor a
  selected matched negative.  Thus difficult unpaired copy opportunities cannot be
  relabeled as collateral controls.

Synthetic rows use unique token banks disjoint across fit/selection/final/OOD.  A
positive is `stem + sequence + sequence[:cut] + [sequence[cut]]`.  Its matched control
has identical length and token multiset: in the first occurrence only,
`sequence[cut]` is swapped with the first index not equal to `cut` or `cut-1`; the
second prefix and target stay fixed.  The control therefore retains the same current
query, target, positions, and marginal tokens while breaking the earlier
query-successor bigram.  No sequence-bank token may occur in its stem, preventing a
second accidental induction witness.  Token banks, stems, cuts, and their hashes must
be frozen in the row authority.

## Candidate and intervention bank

Candidate selection is fixed to the following nonadaptive bank:

- each of the six named heads;
- the named late pair `L13H0+L14H7`;
- the registered four-head set;
- the full named six-head family;
- hash-selected same-size head controls excluding all six named heads;
- no-op identity and fit-role per-position mean ablations;
- source-position shuffle, target shuffle, and length/multiset-matched synthetic null.

A reviewed attention adapter must compute the native two-QK, QK-normalized, RoPE,
unnormalized bilinear pattern and shared block-0 value route once, expose sealed
per-head writes, and prove that recomposing all nine heads through `c_proj` equals the
native attention write.  Means are fit-role only.  Hooks and tensor aliases may not
escape.

Late MLP sites 13--17 form a secondary E4.1 screen.  Candidates must be physical
`Left(x)*Right(x)` gates with `Down` columns and native bias, canonicalized for
scale/sign/permutation.  Output-SVD directions from earlier capitalization work are
controls, not product gates.  Same-K hash-random and canonical Down-derangement
controls are mandatory.  This secondary screen cannot launch until its source-closed
late-site gate adapter exists.

## Frozen stages and currencies

1. Fit means/programs and frequency strata on `fit_natural` only.
2. Score the complete fixed bank on `selection_natural` plus held-out synthetic
   templates.  Select by positive CE effect, then lower executable price, subject to
   every specificity/collateral gate below.
3. Freeze the selected executable program and all gates before final/OOD access.
4. Perform exactly one `final_natural` and one `ood_code` evaluation.

Keep these currencies separate by role and never pool them:

- task: positive target CE/log-probability and copy top-1 accuracy;
- specificity: matched-negative and off-target CE effects;
- distributional fidelity: native-to-candidate KL on each cell;
- extraction: fraction of native-versus-ablation positive-CE stake recovered by the
  sparse executable program;
- removal: agreement of extracted-program removal with native circuit ablation,
  plus intended positive damage and collateral negative/off-target damage;
- transplant: target-token/logit movement to the preregistered donor successor with
  negative/off-target CE and KL;
- OOD: the same quantities on `ood_code`, reported independently;
- price: actual serialized bytes, stored floats, products/token, multiplies/FLOPs,
  runtime, and peak memory.  Attention Q/K/Q2/K2/V/`c_proj` factors and all biases are
  charged; an index list alone is never the price.

Extraction recovery is `(CE_ablation - CE_extracted) /
(CE_ablation - CE_native)` on matched positives and is undefined unless the
denominator is positive.  Top-1 agreement alone never licenses extraction.

## Gates and claim boundary

The selected program must, on selection and independently on final and OOD:

- have positive document-bootstrap lower 95% bound for intended positive CE effect;
- beat every same-size and shuffled control under the frozen familywise rule;
- have matched-negative and off-target CE worsening at most 0.01 nat;
- have finite positive-cell KL and report it beside CE and accuracy;
- retain at least 75% extraction recovery on final and at least 50% on OOD;
- make removal reproduce at least 75% of native-ablation positive damage while each
  collateral cell remains at most 0.01 nat worse than native;
- pass native identity, all-head recomposition, hook cleanup, source closure,
  create-only publication, exact reload, and receipt-last lifecycle checks.

Before denominators and bootstrap details are frozen in a row/scorer authority, these
numeric thresholds are design targets rather than permission to run.  A pass licenses
only the measured copy circuit and prices.  It does not move whole-model explained CE,
named-behavior coverage, or current-ship executable recovery.

## Current launch blockers

The pure contract deliberately fails closed until all six flags are bound and replayed:
fresh row authority; checkpoint authority; source-closed per-head attention adapter;
source-closed physical eight-candidate dispatcher;
source-closed late product-gate adapter (or an explicit authority omitting that
secondary screen); frozen scorer/bootstrap authority; and an empty create-only result
namespace.  `terminal_copy_induction_v1.assert_launch_ready` is the executable NO-GO
gate.
