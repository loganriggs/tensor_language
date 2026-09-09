# Does the extracted summary mediate record-order dependence?

Record-order output invariance fails. Keep that null. A different question is
which first-layer input to subsequent computation carries this dependence.
The already extracted final-query state is S(document,hop)+L(entity,hop).
Record permutations leave L identical. All other first-layer states form the
binding/context prefix P. Cross original/donor P and original/donor S in a
fixed 2x2 factorial, then execute all remaining layers with live normalization.

Cells are named by (prefix,summary): 00 original,01 summary-only transfer,
10 prefix-only transfer,11 reordered donor. Compute S directly from tokens;
do not use cached unexplained native activations in the exported executor.
The independent native oracle runs the prefix's original tokens and replaces
only its final-query first-layer state with the summary donor's native first-layer
final-query state. Identical suffix tokens make this exactly the stated S swap.
Earlier output positions must be unaffected by the summary bit.

Use all3072 opened original/reordered pairs from RECORD_ORDER_OUTPUT_INVARIANCE_V1:
16 worlds,24 queries,4 hops,two fixed permutations. No fresh-data claim or fit.
Prediction A: every full-output cell and saved00/11 query replay closes<=1e-9
absolute and1e-10 relative RMS; local terms equal; earlier summary-bit effects
<=1e-9; finite and live controls. Prediction B: summary-only01 reproduces donor11
query distribution at meanKL<=.001,p99<=.01 in every population/hop/permutation
group, and predicts full centered record-order logit effect11-00 with relative
RMS<=.01, denominator max(native effect RMS,1e-6). Report near-zero target norms.
Prediction C: compiled mixed term11-10-01+00 matches native mixed term at the
same1e-9/1e-10 bars. Report prefix-only mediation and interaction strengths as
diagnostics; neither becomes an alternate adopted candidate if B fails.

A B failure rejects summary-only mediation, without expanding the field or
selecting heads/tasks/scales. If B passes, confirm on fresh document/order cases
before identification. This partial causal decomposition keeps all387968 native
export constants and native prefix/suffix computation; no structural saving
or removal selectivity follows from this screen alone. B8FP64,1800s,
256MiB per new tensor. GPU through the managed runner only.
