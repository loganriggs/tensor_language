# Shared multiplication despite an inseparable query block

The coupled-block certificate says no proper norm-preserving linear split of
the fixed three-root query interface can preserve every reader. It does not
say there is no reusable operation. For example,
(z0²,z0*z1,z0*z2) has a common scalar gate z0, although the three symmetric
reader forms together have only a scalar commutant. Interactions and reuse
are compatible. This is the next explicit circuit hypothesis.

Test whether all58 centered I/J output forms from the fixed first context
can share one linear gate g(z), with output-specific linear readers h_k(z):
f_k(z)=g(z)*h_k(z). This could group work across the two routes and output
consumers; it is not a claim that native heads or input roots are independent.
All gate/reader coefficients and native context production remain priced.

Prior art: the reconstruction pilot checked literal proportionality of raw
MLP factors; those are closed. This test concerns already contracted
cross-layer query/endpoint route functions, where cancellations could expose
a factor not present as a repeated native weight row. No rank cutoff,
variance objective, native factor subset, or reopened MLP scan is used.

Source is QUERY_COMMON_SQUARED_CHANNELS_V1_COEFFICIENTS.pt, SHA256
3f195fe72cc86a62b96ba08733c89e4768a6462608b97a0cbf4483fe1b4e3151.
The context and coefficients are opened evidence. No prospective OOD claim.
This is a cheap necessary-condition falsifier, not a complete factor finder:
if a quadratic is a product of two linear forms, its symmetric coefficient
matrix is (uv^T+vu^T)/2 and has rank at most2. One exact nonzero3x3 determinant
therefore rejects a common linear gate even locally. Scan all58 forms in
their stored order and stop at the first nonzero determinant. If all vanish,
report INCONCLUSIVE, not identification or proof of a common factor. Later
factor discovery/held-out manipulation would require its own specification.

A controls: planted common-gate family passes the necessary condition even
though its commutant is scalar-only; exact arbitrary-amplitude evaluation
matches the factored circuit; independent and joint edits of the gate-producing
input z0 and the other consumer inputs z1/z2 match direct evaluation (the
first output uses z0 twice, so its two occurrences are edited together);
invertible nonorthogonal
congruence preserves the outcome. A full-rank quadratic rejects a linear
product. B: any exact determinant witness closes this single-gate grammar.
Fraction arithmetic on stored FP64 entries certifies the artifact only;
native equivalence retains its parent's numerical limitation.

CPU threads2, alarm180s, no native forward or GPU, no tensor>=256MiB. No
structural improvement follows from a negative witness or a planted circuit.
If rejected, do not add gate count or fit a selected output subset to rescue
this nomination. Keep the distinction between coupled operations and absent
reuse explicit in the report.
