# Rung 459: split equality terms into QK-score and value/output pieces

Status: prospective design, frozen after rung 458's registered strong null and before any factor-hybrid model run.
This is a natural-text circuit-splitting and interchange test, not compression, OOD confirmation, or adoption.

## Question

Rung 457 showed overlapping CE benefit between the early equality block and the layer-8 equality block. Rung 458
showed that no pair of complete terms produces a task-specific shared response at a later full attention/MLP write.
Test the finer hypothesis that an early and layer-8 term share **one algebraic part**—either the scalar attention
score used on equality edges or the residual-stream payload carried across those edges—even though their complete
terms differ.

The four source terms remain L5H5, L7H3, L8H3, and L8H4. Candidate causal directions are the four early-to-late
pairs L5H5→L8H3, L5H5→L8H4, L7H3→L8H3, and L7H3→L8H4. Only the early object may be transplanted into layer 8;
no future layer-8 state is injected backward into an earlier layer.

## Exact algebraic objects

For source term `h`, query `q`, and key position `k`, write its equality contribution in residual coordinates as

`T_h(q) = sum_k E(q,k) p_h(q,k) u_h(k)`.

`E(q,k)` is the exact token-equality-and-one-position-shift mask. `p_h` is the product of the head's two rotary QK
scores after the native normalization and causal mask. `u_h(k)` is that head's native mixed value after applying
only its 128→1152 slice of the output projection. Thus `p_h` is an actually used scalar attention function and
`u_h` is an actually written residual vector. They are invariant to internal Q/K and V/O changes of basis,
respectively. No rank, SAE, or learned basis defines either object.

For an early term `e` and late term `l`, remove `e` at its native site. Cache `p_e,u_e` immediately before its term
is subtracted. At layer 8, compute `p_l,u_l` under that same remove-early state. Define fitting-half RMS scales over
all-positive equality edges:

- `a_h = RMS(p_h)` over the scalar equality-edge entries;
- `b_h = RMS(u_h)` with each source payload replicated once per selected equality edge.

Require every scale finite and positive. Freeze the four pair-specific scale ratios from documents `0:96`. Then
construct at the late site:

- native late reference: `sum E p_l u_l`;
- score transplant: `sum E [(a_l/a_e) p_e] u_l`;
- payload transplant: `sum E p_l [(b_l/b_e) u_e]`; and
- relocated-whole-early diagnostic: `sum E [(a_l/a_e) p_e][(b_l/b_e) u_e]`.

Every analytical arm removes both `e` and `l`, then adds exactly one of these terms at `l`; the base adds none.
The reference is therefore equivalent to removing `e` while leaving `l` native, and its difference from the base is
the late term's marginal contribution when the proposed early provider is absent. RMS matching prevents a candidate
from winning only because one source has a larger arbitrary scale. The native terms and exact unscaled quantities
must also be reported.

## Rows, split, and closed outcomes

Use only rung 457's hash-bound 192-document `final_natural` role and frozen task masks. Documents `0:96` fit the
four pair-specific RMS ratios and select a pair, factor type, and downstream reader. Documents `96:192` validate
that frozen choice. Prior experiments exposed ordinary whole-term CE effects on these documents, but no score/payload
hybrid, hybrid-induced reader response, or hybrid CE outcome has been opened.

Do not load `ood_code`, SEALED attention0 confirmation, or another row role.

## Fitting screen

For each pair run base, native-late reference, score transplant, payload transplant, and relocated-whole-early
diagnostic. Capture every full attention and MLP write in layers 9–17. For reader `j` and task condition `c`, define

`R_ref = write_j(reference) - write_j(base)` and
`R_hyb = write_j(hybrid) - write_j(base)`.

Report cosine, reference-relative error `||R_hyb-R_ref||/||R_ref||`, and response RMS relative to the native reader
write. For each score or payload hybrid define

`task_margin = cosine_positive - max(cosine_matched_negative, cosine_off_target)`.

Also compute the fitting-half all-positive CE recovery

`recovery = [CE(base)-CE(hybrid)] / [CE(base)-CE(reference)]`,

requiring a positive reference stake. A candidate among `4 pairs × 2 factor types × 18 readers = 144` qualifies if:

- positive response cosine is at least `.75`;
- task margin is at least `.10`;
- positive response error is at most `.60`;
- both reference and hybrid response RMS values are at least `1e-4` of native write RMS;
- all-positive CE recovery is at least `.50`; and
- absolute off-target `CE(hybrid)-CE(reference)` is at most `.01 nat`.

Choose largest task margin, then lowest response error, then highest CE recovery, then lexical pair/factor/reader
identity. Direct score or payload cosine may be reported diagnostically but cannot select or identify a group by
itself. If no candidate qualifies, stop before validation and record the registered null.

## Held-out response and causal interchange

On documents `96:192`, recompute the frozen RMS ratios without fitting only as a reported calibration check; the
hybrid must use the ratios frozen on documents `0:96`. Require the frozen hybrid/reference reader comparison to have:

- positive cosine at least `.65`;
- task margin at least `.05`;
- positive relative response error at most `.70`;
- live reference and hybrid responses under the same `1e-4` rule; and
- positive cosine and CE-recovery signs in each 48-document validation half.

For all-positive CE require point recovery at least `.40`, a simultaneous 95% document-bootstrap lower bound above
`.20`, and positive reference stakes in every bootstrap draw. Require absolute off-target hybrid-minus-reference CE
at most `.01 nat`.

The frozen between-group control uses the other early term transplanted into the same late term with the same factor
type and its own fit-half RMS ratio. On each supported validation document compute the absolute CE discrepancy from
the native-late reference. Apply the shipped label-permutation statistic to selected-pair discrepancies as “within”
and the frozen other-early discrepancies as “between.” Require between/within mean discrepancy at least `2.0` and
`p <= .05`. Matched-negative and off-target discrepancies are reported but cannot choose the group.

## Registered predictions

### A. Instrument

All model/source/row/mask hashes, exact factor reconstruction, native replay, source-site dispatch, cache timing,
fit/validation split, RMS-scale, arm identity, and no-new-role clauses hold. For every term, `sum E p_h u_h` must
match the existing exact equality contribution with relative squared error at most `1e-10`; empty analytical replay
must have relative squared logit error at most `1e-12`.

### B. A factor-specific fitting candidate exists

At least one of the 144 score-or-payload hybrid/reader candidates clears every fitting bar.

### C. The factor-specific reader response transfers

The frozen pair, factor type, and reader clear every held-out response and two-wave sign bar without reselection.

### D. The hybrid preserves the late term's held-out causal task effect

The frozen hybrid clears the all-positive bootstrap recovery and off-target bars.

### E. The proposed shared factor passes the frozen between-group control

The selected early factor is materially more interchangeable with the late factor than the other early term under
the separation and permutation bars.

The strong null is instrument failure, no fitting candidate, held-out positive response cosine below `.30`, held-out
CE recovery at most `.10`, or between/within separation at most `1.2`.

## Claim boundary and next branch

B alone is a fitting screen. B+C is a stable factor-specific downstream response. Only B+C+D+E identifies a
natural-text shared score or shared payload candidate. It still does not license code OOD, compression, or replacing
the native heads. Deployed saving is zero; report exact forwards, wall time, peak memory, retained statistics, and
all 144 fitting candidates.

If no score transplant qualifies but a payload transplant does, keep the payload group and next split the QK product
into its two score branches. If a score transplant qualifies but payload does not, next decompose the shared score
into Q/Q2 and K/K2 feature vocabularies. If neither factor qualifies, the next object is context-conditioned mixtures
within each score or payload rather than another whole-head, whole-term, or rank sweep.

