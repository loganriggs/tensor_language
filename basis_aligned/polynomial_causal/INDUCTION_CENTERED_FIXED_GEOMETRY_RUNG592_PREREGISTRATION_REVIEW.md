# Independent pre-execution review: R592 fixed-geometry centered-factor preregistration

Date: 2026-09-04 UTC

Reviewed commit:
`cb81a22bf10fc46e2c851361d2a5de95dd5b7045`

Reviewed preregistration SHA-256:
`870fec55da7207a6e850e64ea705d4f9bb96b2cef40326b2cf59732466dd341a`

Verdict: **BLOCKED before implementation**

The row authority, centered algebra, fixed-width batching, phase price, FIT-first
opening rule, and v7 claim boundary are coherent. The preregistration is not yet an
executable specification, however. Four ambiguities at the R585-to-R592 boundary
permit scientifically different implementations while satisfying its prose.

This review used immutable Git blobs and small CPU fixtures only. It did not inspect
an R592 result namespace, load the model, import CUDA, use a GPU or queue, or edit
R590 or the R592 preregistration. The reviewed Git tree contains the R592
preregistration but no R592 producer, result, receipt, or evidence artifact.

## Exact authorities and reconstruction

All seven hashes cited by R592 match the immutable files at the reviewed commit:

| authority | SHA-256 |
|---|---|
| R585 replacement amendment | `98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf` |
| R585 manifest | `7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962` |
| R591 findings | `99c01c5efc03d3011dd562636a41a38ad181a7b35d4d9ab37a1da69ce26f425f` |
| v7 handoff | `595b43156117e0ba2e568972f76af81ac4e716ed5537861ac48f13b23d4ed9fd` |
| centered derivation | `afb816361603d880dea8dd5daa30b90e841f686d4935da8684ac78c3839a78ca` |
| independent centered review | `375ba4bb36655caf1807f978ff38b1aee0f85adc1e82f335663f30a00cf3eec0` |
| successor design | `075ae15bf31bc8ef4da625e3499ff125bc35225f6ab0846fa5bef45773876ad9` |

Independent reconstruction of the frozen R585 manifest gives:

| split | rows | endpoints | directions | endpoint × 4-site × 2-role operations | target/control cells | bootstrap cells |
|---|---:|---:|---:|---:|---:|---:|
| FIT | 1,872 | 1,728 | 3,744 | 13,824 | 20 / 32 | 124 |
| SELECT | 936 | 864 | 1,872 | 6,912 | 20 / 32 | 124 |

Each split also has 32 structural identities, 88 activity-eligible control-arm
cells, and 24 control-coverage keys. The legacy manifest's machine arm names are
exactly `score`, `payload`, and `joint`, and its bootstrap namespace is
`a8-r585-replacement-group-bootstrap-v1`.

The new call arithmetic is correct. Per directed chunk there are native, literal-zero
replay, coefficient, projected-content, and joint calls:

$$
\begin{aligned}
\mathrm{FIT}&=\lceil1728/32\rceil+5\lceil3744/32\rceil
=54+5(117)=639,\\
\mathrm{SELECT}&=\lceil864/32\rceil+5\lceil1872/32\rceil
=27+5(59)=322.
\end{aligned}
$$

Full execution is therefore exactly 961 forwards, zero backwards, and zero updates.
The 1,872 SELECT directions produce 58 full chunks and one 16-row chunk. The
preregistration correctly forbids filling that partial chunk.

## Algebra and licensed claim

For $B(E,U)=\sum_r e_ru_r$, the three centered changes and mixed difference are
correct:

$$
\begin{aligned}
\Delta_c&=B(E_y,U_x)-B(E_x,U_x),\\
\Delta_u&=B(E_x,U_y)-B(E_x,U_x),\\
\Delta_j&=B(E_y,U_y)-B(E_x,U_x),\\
\Delta_j-\Delta_c-\Delta_u&=B(E_y-E_x,U_y-U_x).
\end{aligned}
$$

The explicit zero replay, same-layer L8 transaction, semantic-role alignment, and
native-versus-replay pairing are also correct. The claim boundary follows v7: a held
result concerns a registered partial output-space factor only, not a complete
attention pattern, realizable query/key state, native remove-and-insert operation,
sufficiency, FINAL/OOD generalization, or a compiled model.

## Blocker 1: inherited gates contradict the repair

R592 says that all R585 cell definitions and numerical gates are unchanged and its
terminal precedence still lists a `canonical-term` or `padding` failure as
`invalid_instrument`. But R591 established that two old R585 instrument gates cannot
be inherited:

- $B(E_x,U_x)$ versus the differently contracted native equality term exceeds
  $10^{-5}$ at all four sites, reaching $9.1552734375\times10^{-5}$; and
- padded versus unpadded native logits exceed $10^{-5}$ by up to
  $4.9114227294921875\times10^{-5}$ over FIT.

Later R592 prose instead says that the first discrepancy only blocks the stronger
literal-removal claim and eliminates the padded/unpadded comparison by fixing width
30. Those are the correct repairs, but both readings remain licensed by the current
text. An implementation that inherits the old checks deterministically invalidates
R592; one that drops every `factor` or `canonical` check could silently weaken the
instrument.

Required repair: freeze a literal supersession table separating inherited
**scientific** gates from replaced **instrument** gates. It should state that:

1. complete native attention reconstruction, support, finiteness, role/site census,
   actual-hook-delta, and structural output identities remain hard $10^{-5}$ gates;
2. $B(E_x,U_x)-C_x$ is retained evidence and a hard failure only for the separately
   named literal-remove-and-insert level, not for R592's centered level;
3. the old padded/unpadded and length-sorted comparator gates are deleted and
   replaced by width-30 tensor-manifest and exact paired-geometry gates; and
4. `canonical-term failure`, `factor mismatch`, and `padding failure` are replaced
   by unambiguous predicate IDs and terminal mappings.

This does not loosen $10^{-5}$; it states which operational equality that threshold
tests.

## Blocker 2: the arm rename changes frozen bootstrap draws

The inherited manifest and all 248 bootstrap IDs use the strings `score`, `payload`,
and `joint`. R592 renames the first two to `coefficient` and `projected_content` while
also claiming that bootstrap cells remain unchanged. Because the arm string is part
of the SHA-defined cell ID, changing `score` to `coefficient` changes the bootstrap
draw matrix, not merely a display label. It also changes control-scale lookup keys,
structural identity rows, evidence joins, and failure-clause identities.

Required repair: freeze one exact bijection and one machine-ID policy. The minimal
policy is to retain the legacy machine IDs and bootstrap bytes while requiring the
v7 operational labels in prose and result metadata:

```text
score   -> registered_equality_factor_coefficient_swap
payload -> registered_projected_content_swap
joint   -> registered_joint_output_factor_swap
replay  -> literal_zero_centered_replay
```

Alternatively, a new manifest must materialize and hash every renamed cell, scale
lookup, structural identity, bootstrap ID, and draw matrix. A mixture of legacy and
renamed IDs must fail preflight.

## Blocker 3: capture-to-live transport checks the wrong invariant

R592 requires only cached-versus-live agreement of the self term $B(E_x,U_x)$. That
does not certify the two factors separately. A capture and a directed execution can
have equal self terms but different hybrids. For example,

$$
(E_x,U_x)=((1,0),(1,0)),\qquad
(E'_x,U'_x)=((0,1),(0,1))
$$

both give self term 1, while a donor $E_y=(2,3)$ gives
$B(E_y,U_x)=2$ and $B(E_y,U'_x)=3$. The coefficient intervention has changed even
though the registered cache/live gate passes. The analogous failure holds for the
projected-content hybrid.

Required repair: state whether every scientific delta uses only the frozen endpoint
cache or uses a live directed recipient factor, and freeze all three arm tensors
before any intervention call. If the cache defines the mediator, the directed native
pass must compare every operational hybrid needed for that direction/site under
capture and directed geometry:

$$
B(E_x,U_x),\quad B(E_y,U_x),\quad B(E_x,U_y),\quad B(E_y,U_y),
$$

or prove equivalent componentwise $E/U$ checks with explicitly typed tolerances.
Both directions do make every endpoint available as a recipient, but the check must
apply to every occurrence, not one representative endpoint. Nonfinite or
greater-than-$10^{-5}$ output-space drift is an instrument failure.

The inherited activity definition also cannot remain `inserted-live_removed`.
Centered R592 removes no native term. Its activity must be the norm of the actual
centered hook delta $B_{new}-B_x$. The evidence field should be renamed from
`live_removed` to `recipient_factor_baseline`; otherwise an implementation can
reintroduce exactly the $B_x-C_x$ term R591 rejected.

## Blocker 4: phase evidence and invalid-terminal closure are not frozen

The compact schema changes from one R585 row per direction/arm to one R592 row per
direction containing several conditions, but it gives no exact field schema, arm/site
array census, shapes, or phase-local row counts. It also requires an elementwise
native/replay full-vocabulary gate while the directed record omits the directed
native measurement. A replay-relative sum of squares cannot reconstruct an
elementwise maximum: two vectors can have identical RMS while only one exceeds
$10^{-5}$.

The same issue affects full-vocabulary structural arm identities. An online scalar
reported by the producer is not independently reconstructible from the stated saved
primitives.

Required repair:

- freeze separate directed native, replay, and three-arm sufficient-stat fields;
- freeze whether replay plus three centered deltas create four arm/site tensor rows
  per direction (59,904 FIT and 29,952 SELECT at four sites), including exact dtypes,
  shapes, ordering, and hashes;
- save enough raw difference evidence to independently reconstruct every paired
  native/replay and structural maximum, or explicitly narrow the audit claim and
  identify which predicates remain runtime-only;
- freeze the exact token-tensor hash census for 54/117 FIT and 27/59 SELECT chunks;
- define $A_g$ from the actual centered deltas and preserve the 124 bootstrap cells
  per opened split; and
- specify exact evidence presence and call counts for every terminal.

The last item matters because the preregistration simultaneously says complete raw
evidence is collected before cases 3--9 are decided and inherits R585's permission
to stop an invalid instrument below the phase ceiling. A valid implementation cannot
infer whether an invalid FIT artifact must have exactly 639 calls and complete FIT
evidence, may stop early with a recognized partial schema, or must hard-abort without
a scientific artifact. SELECT opening is otherwise clear: it occurs only after all
FIT instrument and scientific clauses pass, uses frozen FIT scales, and never opens
FINAL/OOD.

## Model-free adversarial packet

`test_induction_centered_fixed_geometry_rung592_prereg_review.py` pins the reviewed
preregistration and contains five checks:

1. exact authority-operation and 639/322/961 price arithmetic, including the
   16-row SELECT tail;
2. a bootstrap fixture showing that the renamed arm produces different SHA-defined
   draws;
3. equal self terms with unequal coefficient/content hybrids;
4. equal-RMS vectors on opposite sides of the elementwise $10^{-5}$ gate; and
5. a contraction discrepancy for which centered activity and legacy
   `inserted-live_removed` activity fall on opposite sides of $10^{-5}$.

These are specification attacks, not model outcomes.

## Decision

**BLOCK `cb81a22bf` from implementation.** The centered mathematical object and
fixed 639/322 execution design should be retained. A short prospective amendment can
close the block without changing rows, scientific thresholds, bootstrap draws,
forward price, or claim scope by freezing: the instrument-gate supersession table,
legacy-ID/operational-label mapping, exact frozen-versus-live hybrid policy, centered
activity definition, complete evidence schemas, and terminal-specific phase closure.

After that amendment is frozen, a different agent should review the amended spec
before implementation. No R592 model execution is licensed by this review.
