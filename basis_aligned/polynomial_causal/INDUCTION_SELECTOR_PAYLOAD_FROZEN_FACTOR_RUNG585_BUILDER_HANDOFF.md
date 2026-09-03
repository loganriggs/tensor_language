# R585 frozen selector × payload builder handoff

**Date:** 2026-09-03 UTC  
**Status:** model-facing implementation and CPU contract complete; no R585 model outcome opened

## Circuit hypothesis and counterfactuals

The narrow hypothesis is that the complete fixed set `L5H5`, `L7H3`,
`L8H3`, and `L8H4` carries an operational selector-by-payload computation.
At each site, the equality-gated continuous attention score selects a semantic
source role and the projected value writes its payload. The experiment tests
that claim as a four-site behavior-level factorization, not as a unique Q/K
basis or individual-head attribution.

R578 supplies four meaningful target constructions: selector-only changes,
payload-only changes, answer-preserving crossed selector/payload diagonals, and
answer-preserving selected-match breaks. Neutral-source, neutral-payload,
filler, and lag edits are active answer-preserving controls. Both physical
directions are retained. A/C are semantic roles read from row metadata even
when the A/C/N pairs appear in different physical prompt orders.

## Exact intervention

Every endpoint is run natively before any intervention. For every fixed site
and both roles A/C, the runner caches the native equality-gated score scalar
`e` and projected value vector `u`. It then materializes all four directed
role-summed terms before an intervention forward:

```text
replay  = sum_r e_recipient[r] * u_recipient[r]
score   = sum_r e_donor[r]     * u_recipient[r]
payload = sum_r e_recipient[r] * u_donor[r]
joint   = sum_r e_donor[r]     * u_donor[r]
```

During a directed forward, the hook recomputes the live equality term from the
current trajectory, removes it, and inserts the selected frozen term. L8H3 and
L8H4 are changed together from the same pre-attention state. Independent
checks compare the role sum with the hash-pinned canonical equality
contraction, reconstruct the native head and attention write, verify replay
logits against a separately scheduled native comparator, and check the actual
summed hook delta.

## Evidence bars

The runner preserves the amendment's 20 target cells, 32 control cells, 24
coverage keys, 88 eligible control-arm cells, and 124 bootstrap statistics per
split. Recovery is a ratio of whole-cell means plus a separately defined ratio
of medians; rowwise ratios are forbidden. Match-break movement is signed by
donor coherence. Selector, payload, and match transfer require recovery,
positive-effect, fraction-positive, and donor-CE gates. The crossed diagonal
requires both harmful single factors and positive joint interaction.

Control activity uses residual insertion norm. Control margin and
full-vocabulary effects use separate matched logit scales. At least two active
control families are required for every arm/direction/condition key, and every
group in an active cell is scored. FIT is completed and decided before SELECT;
FINAL_TEST and OOD remain closed. The exact ceiling is 459 FIT plus 231 SELECT,
with zero backwards and updates.

## Remaining ambiguity and risk

The scientific result is wholly unknown. The fixed four terms may lack factor
capacity, the proposed single-factor arms may fail to isolate selector from
payload after downstream nonlinear composition, or an active control may show
that the term is a broad contextual write. A held result would still be only a
behavior-level identification screen on FIT/SELECT. It would not establish a
unique weight feature, per-site necessity, selective native-term removal,
reuse across another behavior, or OOD generalization.

The implementation path itself remains untested against the live model in
this CPU-only wave. Runtime reconstruction tripwires are designed to turn any
hook/API mismatch into `invalid_instrument`, but a separate owner review should
inspect memory use before queueing: precomputed directed insertions and saved
live/delta vectors are intentionally literal and sizable.

## Five-part reusable packet

1. **Dataset pattern:** use multiple answer-changing edits plus crossed and
   answer-preserving controls, both physical directions, exact shared semantic
   metadata, and group-disjoint FIT/SELECT. Do not infer role from physical
   array order.
2. **Semantic mapper:** endpoint identity is the token-sequence hash; source,
   payload, query, answer, condition, and direction come from row metadata.
   `base_to_donor` has base as recipient; `donor_to_base` has donor as
   recipient.
3. **Intervention primitive:** cache native factors first, materialize every
   crossed term, then apply `frozen_inserted - live_removed` at a named semantic
   query coordinate.
4. **Control pattern:** establish nonzero intervention in residual units, judge
   behavior in margin/vocabulary-logit units, require multiple active control
   families, and retain structural zero arms only as instrument checks.
5. **Failure diagnosis:** distinguish invalid instrument, invalid native
   denominator/scale, insufficient factor capacity, failed factorization,
   insufficient active controls, and broad collateral write in fixed terminal
   precedence.

## Reusable helpers and tests

The R585 implementation exposes reusable CPU-side pieces for:

- strict finite-JSON result and receipt validation with exact types and hashes;
- parsed dependency-lock closure without weakening scientific nulls;
- semantic endpoint/direction joins and canonical manifest hashes;
- unequal-length exact-price capture/comparator schedules;
- SHA-defined group bootstrap traces and non-rowwise recovery summaries;
- four-way frozen factor materialization;
- three-unit control scale construction and active-family coverage;
- primitive-logit and vocabulary-RMS identities;
- deterministic terminal precedence and held/null/instrument fixtures; and
- deduplicated endpoint factors plus hash-bound binary/JSONL evidence files.

Owner tests exercise each helper adversarially, including tuple-valued decision
rejection, nonfinite JSON, forbidden split opening, price overflow, incorrect
terminal precedence, donor-coherence signs, physical role permutations, and
the full planted held/null paths.

## Transfer boundary

Transfers unchanged to another circuit:

- strict result/receipt and finite-JSON checks;
- authority and dependency hash closure;
- semantic-coordinate batching and padding tripwires;
- deterministic group bootstrap;
- FIT-first phase gating and terminal precedence;
- distinct activity/margin/vocabulary scales;
- active-control coverage; and
- evidence descriptors with dtype, shape, row order, and content hashes.

R585-specific and requiring replacement:

- the four layer/head sites;
- A/C equality-successor roles and the `e*u` algebra;
- R578 family/variant mappings and structural identities;
- donor-coherence match sign;
- selector/payload recovery and diagonal-interaction thresholds; and
- the 459/231 batching price.

## Prompt and tooling lessons for the next builder wave

1. Require a literal formula for every semantic-role sum and a unit test that
   would fail if one role were silently omitted.
2. Require all crossed terms to be materialized before intervention, not
   reconstructed from a trajectory already changed by an earlier site.
3. Require an exact unequal-length batch schedule and a model-free proof of the
   forward ceiling; `ceil(total/batch)` alone does not prove the comparator can
   exercise padding invariance.
4. Require three terminal fixtures—held, scientific null, invalid
   instrument—and strict scalar/list/dict type checks before model access.
5. Require result schemas to bind implementation, owner test, canonical term,
   dependency lock, ordered cell manifests, and binary evidence row order.

## Parent integration addendum

Parent review found that the first handed-off bytes passed the owner suite but
did not implement the repository runner's no-argument `BQLIB_DRYRUN=1`
preflight and did not expose the registered `pred_a` through `pred_c` names to
the static queue gate. Both requirements are now explicit in the runner and an
owner regression test. After the correction, the combined R578, manifest,
independent-specification, and runner suite passes 59 tests; `gate.py`, the
managed no-argument dry run, and `preflight.py` all pass without model access.

The deliberately literal caches are affordable on this machine. A held
FIT+SELECT run stores about 1.04 GB of native/replay final-position logits,
0.29 GB of endpoint factor tensors, at most 0.28 GB of frozen insertions for
the larger split, and about 0.62 GB of live/delta vectors before evidence
serialization. Serialization temporarily adds roughly another 0.91 GB and
writes under 1 GB. Against 25 GiB currently available CPU memory, 32 GB GPU
memory, and 8.8 GB free disk, this has several-fold headroom. The independent
implementation review must still reject unbounded growth or a changed census.
