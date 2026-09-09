# Temporal `is`/`was` v23 source × destination cell atlas v3

## High-level question

The valid v2 atlas says that the four causal attention heads mostly share an effective-value
input program, while different token roles carry different amounts of the behavioral effect. It
does not say where each read is written. This prospective screen resolves the complete donor-minus-
native output of each head into the frozen 3 × 3 cross-product

`destination role × source role`,

where both axes are `changed`, `unchanged_prefix`, and `matched_suffix`. The screen asks whether a
changed temporal cue writes directly at the matched reporter, writes an intermediate prefix state,
or contributes through some other directed cell.

## Immutable authority and population

- Parent population: all 64 v23 rows, 16 in each of A1, A2, P, and C.
- Target score: the same 30 jointly capable A1/A2 rows used by the immutable v23 parent and v2.
- Routes: L8H1, L9H1, L9H4, and L11H3.
- Dependency: the exact v2 result is hash-bound and must have valid authority and replay predictions.
- v24 is forbidden.
- No row filtering, fitting, optimization, PCA, SAE, or coalition search occurs.

For a head, let `H_base[q]` and `H_donor[q]` be its complete pre-output-projection residual writes
constructed from the native and donor Q1/K1/Q2/K2/effective-value factors. Let `C[d,s,q]` be the
exact term for destination role `d` and source role `s`. The primitive must satisfy

`sum_(d,s) C[d,s,q] = H_donor[q] - H_base[q]`

at every covered causal query. For each of the nine cells, the screen runs both

- sufficiency: `H_base + C[d,s]`, and
- donor reset: `H_donor - C[d,s]`.

The reset's removed behavioral effect is `full_donor_delta - reset_delta`. One additional all-cell
sufficiency arm per head must reproduce that head's immutable v2 full intervention. This costs
`2 native + 6 distinct-side/layer captures + 4 × (9 sufficiency + 9 reset + 1 full replay) = 84`
model forwards, or 5,376 sequence evaluations.

## Frozen bars and predictions

Instrument validity requires tensor closure at `1e-4`, v2 full-arm behavioral replay at `2e-3`,
finite outputs, exact enumeration/price, and P/C control leak no larger than the registered bars.
A **reciprocal cell** has, in both target halves and overall, positive signed projection for both
sufficiency and removed-reset response; overall signed projection must be at least 0.15 in both
directions, cosine at least 0.60, direction fraction at least 0.60, and P/C leak at most 0.25.

Registered predictions:

A. Authority, v2 dependency, partitions, finiteness, enumeration, exact price, and no-v24 tripwire pass.
B. Exact cell closure and every all-cell head replay pass.
C. At least one `matched_suffix <- changed` cell is reciprocal.
D. At least three of four heads contain at least one reciprocal cell.
E. The same reciprocal cell label appears in at least two heads, supporting a shared directed role.

A or B failure is `invalid_instrument`. Otherwise the screen reports either
`shared_directed_cell_program_screen` when C--E all pass, or `resolved_cell_atlas` when they do not.
Neither terminal identifies a natural-language feature or downstream consumer. A directed cell is
eligible for a later writer-cell × reader intervention only after passing both reciprocal directions;
it is not an isolated executable circuit by itself.

## Opposing outcomes

- If changed cues directly produce reporter writes, `matched_suffix <- changed` should pass
  sufficiency and reset in both halves, potentially across multiple heads.
- If the four heads implement relay stages, reciprocal cells should occur at different destination
  roles or source roles per head.
- If singleton cells fail while the all-cell replay succeeds, the computation is interaction-
  distributed at this resolution; that is an honest atlas result, not permission to post-select a
  larger cell coalition.

