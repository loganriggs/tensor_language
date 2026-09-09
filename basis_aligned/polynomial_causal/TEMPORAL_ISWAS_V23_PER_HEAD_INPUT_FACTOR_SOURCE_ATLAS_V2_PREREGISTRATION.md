# Temporal is/was v23 per-head input factor/source atlas v2

**Status:** preregistered before model execution; conditionally licensed only by a valid v1 union atlas

## Circuit decision

The v1 atlas asks which Q1/K1/Q2/K2/effective-value factors and token roles matter to the four-head
union. That is insufficient for a computation story: it cannot tell whether `L8H1`, `L9H1`,
`L9H4`, and `L11H3` compute the same intermediate variable or implement distinct stages. V2 repeats
the exact finite games one head at a time while every other selected head stays native-base.

This directly tests computational specification and within/cross-module grouping. It neither fits a
subspace nor assumes a native head is a semantic unit. Similar profiles are evidence for a shared
operation only at screen level; different profiles license finer reciprocal edge tests.

## Frozen authority and population

- all 64 immutable v23 rows, 16 each in A1, A2, P, and C;
- target scoring on the 30 previously frozen jointly capable A1/A2 rows;
- controls unfiltered;
- routes `L8H1`, `L9H1`, `L9H4`, and `L11H3`;
- every query/source pair through the semantic endpoint;
- no v24 access, fitting, gradient, rank, threshold, route, or source selection;
- hard dependency: v1 must pass authority and union replay before v2 can execute.

The source-role partition and five exact attention factors are byte-identical to v1.

## Per-head games and measurements

For each head $$h$$, v2 evaluates all 32 factor coalitions and all 8 source-role coalitions. Empty
arms reuse the native base; full source arms reuse the head's full factor arm. Thus the model price is

$$
2\ \text{native references}
+6\ \text{captures}
+4(31+6)\ \text{interventions}
=156\ \text{forwards},
$$

or $$156\times64=9{,}984$$ sequence evaluations.

Each head's coalition effect is scored twice:

1. against the complete native donor-minus-base response, preserving comparability with the parent;
2. against that head's own all-factor effect, so a proper subset is asked to retain its writer rather
   than recover an impossible 50% of the entire four-head circuit.

A proper head-relative subset is selective when target signed projection is at least `.80`, cosine
and direction fraction are at least `.90`, and P/C response RMS is at most `.25` of that head's
target full-effect RMS. Split metrics use the two frozen reporter halves.

Factor-profile comparison uses each head's five Shapley allocations normalized by their L1 norm.
For two profiles $$a,b$$, distance is

$$
d_1(a,b)=\sum_i |a_i-b_i|.
$$

No profile pair or proper mask is selected before reporting all of them. The canonical minimal mask
for a head is the passing mask with smallest population count, breaking ties by integer mask.

## Registered predictions

1. authority, dependency, captures, closures, finiteness, enumeration, and exact price pass;
2. every head's full-factor arm replays its immutable parent singleton report within `2e-3`;
3. changed temporal tokens are the largest source-role Shapley allocation for at least three heads,
   allocate at least `.40` head-relative credit, and remain positive in both halves;
4. at least two heads have a proper selective factor subset under the head-relative bars;
5. the heads are not four copies of one input computation: at least one normalized factor-profile
   pair has L1 distance at least `.35`, and the same pair has distance at least `.20` in both halves.

Predictions 3--5 may fail without invalidating the instrument. If 3 fails, temporal evidence is
distributed through unchanged/suffix context for multiple heads. If 4 fails, each head's bilinear
input factors are coupled at this resolution. If 5 fails, a shared input operation is a better next
hypothesis than native-head-specific semantics. None alone identifies a directed upstream edge.

## Validity and nulls

- reconstructed and empty/full factor closures: maximum absolute error `1e-4`;
- parent singleton scalar replay: maximum absolute error `2e-3`;
- exactly 156 forwards, 9,984 sequence evaluations, zero backwards, updates, or fit parameters;
- v1 authority/replay must be true and v24 must remain unopened.

Any validity failure yields `invalid_instrument` and suppresses scientific interpretation. A valid
all-scientific-null result still advances the circuit: the four writers then require coupled factors,
distributed token context, and/or one shared operation rather than four factor-specific stories.
