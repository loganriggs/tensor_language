# Induction equality tensor: one-shot natural FINAL and code OOD

**Frozen before either outcome container was opened:** 2026-08-30 04:18 UTC.

## Candidate fixed by SELECT

The candidate is exactly the discovery candidate recorded in
`induction_equality_tensor_discovery.json` (SHA-256
`0b826952d227c6f2c9e8b0fadf19aeb28edcd4153a52e4b67777a587733e184b`).
No head, mask, tensor, null, threshold, or statistic may be selected here.

The heads are `L5H5`, `L7H3`, `L8H3`, and `L8H4`. Their retained interaction is

$$
M_{qk}=\langle e_{t_q},e_{t_{k-1}}\rangle\mathbf 1[1\leq k\leq q],
\qquad
z_q^{(h)}=\sum_k M_{qk}A_{qk}^{(h)}v_k^{(h)}.
$$

This sums every valid equality match. It contains no nearest-match, argmax, TopK,
`has_match`, or fitted router. The null is the already-fixed cyclic permutation of the
vocabulary identity tensor. The six arms and all four masks are unchanged from
discovery.

## Sealed roles and bindings

Open exactly once:

- `final_natural.pt`, SHA-256
  `1997026ce15d0524bd16540047799a6461bc94a57fbdd2812ef41ff36e8d5e3c`;
- `ood_code.pt`, SHA-256
  `6cf514e75dfd03399f223a9ba5f6ebe5f4b1315bcb839a515e1c19e7b5474bd9`.

Both descend from row receipt SHA-256
`aea52a94c643906ef822a7c6ddb37a371b4315507a1a0a79acd539a19ae7f5c8`.
An outcome-blind independent source audit and a source-bound GO authority are required
before execution. A result or precise failure is create-only.

## Statistics

For each role and arm, report cross-entropy (CE), native-to-arm KL, and top-1 changes
on positive, matched-negative, off-target, and all scoring positions. The unit of
resampling is a document, not a token. Use 20,000 shared bootstrap draws with the
frozen seed. Report:

- target damage: CE(remove equality) minus CE(native) on positive positions;
- specificity: target damage minus the corresponding matched-negative damage;
- off-target damage;
- extraction recovery: fraction of the full-head deletion damage recovered by the
  isolated equality tensor; and
- recovery of the cyclic-vocabulary null.

The SELECT anchors, fixed before this run, are target damage `0.5122487687` and
extraction recovery `0.9739717690`.

## Natural FINAL gates

All must pass:

1. target-damage and specificity 95% lower bounds are positive;
2. extraction recovery is at least 0.80 point and 0.60 lower bound;
3. off-target damage is at most 0.01 nat by 95% upper bound and at most 10% of target
   damage by point estimate;
4. the deranged recovery 95% upper bound is below half the extraction point estimate;
5. every named mask has at least 200 positions in at least 30 documents; and
6. replay and the zero-native-candidate-call census pass.

## Code OOD gates

OOD is a transport claim, so its gates are slightly less brittle but still fixed:

1. target-damage and specificity 95% lower bounds are positive;
2. target damage retains at least 50% of the SELECT point estimate;
3. extraction recovery is at least 0.60 point and 0.40 lower bound, and retains at
   least 50% of the SELECT point estimate;
4. off-target damage is at most 0.02 nat by 95% upper bound and at most 20% of target
   damage by point estimate;
5. the deranged recovery 95% upper bound is below half the extraction point estimate;
6. every named mask has at least 200 positions in at least 30 documents; and
7. replay and call-census gates pass.

Passing both roles certifies functional OOD transport for this bounded task and these
documents. It does not certify a cheaper parameterization, unique mechanism, or all
possible copy behavior.
