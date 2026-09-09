# Record order versus full native output

The summary transport null does not imply native output dependence: later
computation could cancel positional details. Test the distinct function-only
abstraction: at fixed query entity and hop, full query distributions depend on
the represented bijection, with no record-order adapter. This concerns the
final query only; earlier positions see different prefixes and are not expected
to agree under reordering.

Use the opened QUERY_INITIALIZER_FACTORIZATION_V1_ROWS.pt cohort: all16 worlds,
24 entities and4 hops, with the same two fixed permutations already tested on
summaries (swap records0/1; cyclic shift by one). No newly held-out claim, fitting,
subset selection or native-weight modification. Original tokens/answers remain
fixed except the 24 complete records. Validate unchanged parsed function and
unchanged query suffix, plus identity permutation. Reject malformed records.

For each original/reordered pair p,q, an order-invariant surrogate has the same
distribution on both, so its best attainable mean teacher KL is JS(p,q), attained
by (p+q)/2. Reuse the existing entity-symmetry bound implementation, replacing
its label alignment with the identity output-label map. Report all cases,
per-population/hop means, maxima and argmax disagreements, both permutations
separately. Mean JS>.001 rules out the candidate's .001 paired query-KL target;
otherwise label it not ruled out by this bound, never confirmed invariant.

Mechanical prediction A: exported/native full-output agreement <=1e-9 absolute
and1e-10 relative RMS on original and reordered inputs, unchanged parsed function,
identity exactly preserved at token level, all logits finite. Prediction B:
mean paired JS<=.001 in every population/hop/permutation group. A failure is an
instrument problem; a B failure closes only a function-only output abstraction
without record-order inputs, not all semantic decomposition. No enlarged adapter
or easier permutation may rescue that fixed candidate in this receipt.

All387968 native export constants remain charged. No candidate is adopted from
this diagnostic. Execution cap1800s, B8 FP64,256MiB per new tensor; GPU only through
the managed enqueue helper. The token contract is implemented and CPU checked;
the managed full-output integration is the next unfinished action.
