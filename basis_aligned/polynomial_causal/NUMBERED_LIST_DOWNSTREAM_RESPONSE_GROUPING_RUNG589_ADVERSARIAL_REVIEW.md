# R589 downstream-response grouping adversarial review

**Review date:** 2026-09-03 UTC

**Scope:** frozen R589 source, test, result, and its already-produced R584 FIT sufficient statistics; no model calls or new outcomes

**Verdict:** **numerical screen verified; negative grouping inference not licensed**

## What reproduced

The R584 input is byte-identical to R589's frozen source hash:
`7980753636fab422ed6c609a1afd054f99ed7f903e2bb3e61eddf0617316fdf6`.
Strict JSON parsing succeeds for both R584 and R589 without non-standard
`NaN`/`Infinity` values. R589 opens FIT only and reports zero forwards,
backwards, and weight updates.

The input has exactly 12 arms: four sites `MLP8/10/12/14` crossed with
`background_cross`, `contrast_self`, and `joint_response`. Every arm contains
the same 576 row IDs. Each arm has 96 rows for every one of the six conditions,
and 96 rows in every representation-by-source-level cell. All semantic row
metadata is identical across arms in the actual artifact.

The signed response was independently reconstructed from primitive saved
logits for all 6,912 arm-rows:

- `step_two`: native minus intervened
  `(arithmetic_logit - structural_logit)`;
- the other five conditions: native minus intervened
  `(answer_logit - max_other_candidate_logit)`.

Thus positive response consistently means that removal reduced the registered
preference. The stored derived margins agree with the primitive-logit
subtractions.

There are exactly

\[
\binom{4}{2}\times 3\times 3=54
\]

cross-site arm pairs. For all 54, an independent Pearson implementation
reproduces the overall correlation, six 96-row cell correlations, three
384-row leave-one-representation-out correlations, and two 288-row
leave-one-source-level-out correlations. Re-running the current analysis over
the frozen R584 JSON reproduces the complete R589 JSON exactly.

## What the result can and cannot say

No pair passes the stated conjunction: minimum cell correlation at least
`0.60`, minimum representation leave-out correlation at least `0.75`, and
minimum source-level leave-out correlation at least `0.75`. Therefore this
precise statement is licensed:

> No pair passed the recorded post-outcome discovery filter.

The stronger reading of the terminal label
`no_stable_cross_mlp_grouping_lead` is not licensed. The filter and thresholds
were chosen after R584 FIT outcomes were visible, so neither passing nor failing
them is confirmatory evidence. In particular, the top pair,
`MLP12 joint_response` with `MLP8 background_cross`, has overall correlation
`0.7781` and all six cell correlations positive with minimum `0.6324`. It misses
the representation leave-out threshold at `0.7317` and the source-level
leave-out threshold at `0.6440`. This is a threshold-dependent triage result,
not evidence that stable cross-MLP functional grouping is absent. It must not
close or materially down-rank the hypothesis without a prospectively frozen,
group-disjoint confirmation.

Even a passing centered correlation would remain only a screen. Correlation
ignores response scale and offset, can reflect shared row difficulty, and does
not test additivity, joint intervention, selective removal, or downstream
operational equivalence. The result's own `screen_only` and
`post_outcome_filter_not_a_registered_gate` labels correctly acknowledge this;
canonical summaries must preserve those qualifiers.

## Implementation hardening findings

The actual R584 rows have coherent metadata, but R589's validator checks only
row-ID membership plus a subset of arm fields. It does not verify that
representation, source level, condition, action, token IDs, answers, or group
ID match across arms. An adversarial test changes the representation of a row
in the lexicographically last arm; R589 accepts it and produces an identical
result because cell membership is taken only from `arm_a`. This does not alter
the present audited numbers, because the independent audit verified actual
metadata equality, but it is a fail-closed defect for reuse.

The result binds the R584 source bytes but does not contain the R589
implementation or test hash. Current observed hashes are:

- implementation: `bfb12061866b22a6a181d8133e04f326e7669f53be6a75cd8bafdea1922e8f97`;
- original test: `811a23a98ee64347c2b83db5f2b8d27173d9c60f7592661dcf798219b95d9335`;
- result: `794d857f673586e6d05c325753f6a9a760af04faa589e7450e7ed6709fafc02e`.

These separately observed hashes permit this review, but the missing bindings
should be fixed in future result schemas.

## Five-part handoff packet

1. **Dataset/manifest pattern.** Exact shared row IDs and balanced
   representation-by-source cells worked. The reusable validator must also
   require full semantic metadata equality across arms; row-ID equality alone
   is insufficient.
2. **Semantic-coordinate mapping.** Compare responses by registered row ID,
   representation, source level, and condition, not by array offset. Use the
   arithmetic-minus-structural preference only for `step_two`; use the
   registered answer-versus-other margin elsewhere.
3. **Smallest exact intervention term.** The screened objects are the twelve
   site/component removal arms, not a discovered shared term. The smallest next
   causal object is one prospectively chosen cross-site pair installed and
   removed jointly, with each native component still separately addressable.
4. **Active-control pattern.** A confirmation needs group-disjoint rows plus
   unrelated behaviors on which both arms are individually live. Test both
   singles, their joint intervention, an independently paired null, and whether
   joint behavior follows the frozen grouped prediction.
5. **Failure class and unresolved risk.** This is post-selection/negative-
   inference overreach, not a numerical failure. The unresolved scientific risk
   is that positive correlations arise from shared prompt difficulty or a broad
   common downstream effect rather than a reusable computation.

## Prompt improvement for the next wave

Require the author to state before implementation whether each threshold is
prospective or post-outcome, and require the result decision to use the literal
phrase `no_pair_passed_post_outcome_filter` for the latter. Also require an
adversarial cross-arm metadata-permutation test and implementation/test hashes
in every saved result.

## Checks

The separate adversarial suite recomputes the evidence and demonstrates the
metadata-validation defect:

```text
pytest -q basis_aligned/bilinear_quotient/ops/test_numbered_list_downstream_response_grouping_rung589_adversarial.py
10 passed
```

## Parent correction after review

The original committed R589 artifact remains in history. The parent accepted this review and prospectively corrected
the live screen without changing any R584 data or opening another split. The terminal label is now
`no_pair_passed_recorded_post_outcome_filter`; the licensed interpretation explicitly says that passing or failing the
filter is triage rather than evidence for or against shared computation. The validator now requires all semantic row
metadata to agree across arms, recomputes derived margins from primitive logits, and binds its implementation and
primary-test hashes in the result.

The combined original and adversarial suites now pass 16 tests. This addendum does not turn the post-outcome screen
into confirmation; it only repairs its evidence boundary and wording.
