# One private correction vector and a shared writer product

13 September 2026. This reuses the existing fixed-writer cache to simplify the
[complementary tail executor](POLYNOMIAL_TAIL_MLP10_V1_MATH.md).

For the three response vectors $v_0,v_1,v_2$, define
$P_{00}=K(v_0,v_0)$ and $P_{02}=K(v_0,v_2)$. The omitted degree-three and
degree-four coefficients in physical amplitude coordinates are

$$
N_3=-4\gamma\beta P_{00}-\gamma P_{02},
\qquad N_4=\gamma^2P_{00}.
$$

Therefore the omitted self numerator is

$$
E(a)=\frac{a^5}{2\rho(a)^2}
\left[-\gamma P_{02}+(a\gamma^2-4\gamma\beta)P_{00}\right].
$$

The linear response vector $v_1$ cancels from this correction. Only $v_2$
needs new Left/Right projections, followed by one Down write. The fixed writer
$v_0$, its two hidden projections and $P_{00}$ are shared across contexts using
`fixed_writer_products_v1.prepare_global`. This is the same function as the
previous two-tail executor, expressed without dimensionless scaling or a
duplicated fourth-degree vector. It is not a newly fitted approximation.

Actual checkpoint weights, Gaussian pristine contexts and signed amplitudes
$-2,-0.5,0,0.5,1,2$ give full-output and own-tail errors below
$4.00\times10^{-15}$. Checking the tail separately prevents its small size
relative to native output from hiding an inaccurate correction. A mismatched
writer cache raises an error.

Reused preparation takes 3.36/6.80/18.74 ms for context batches 1/16/64, versus
5.18/15.45/51.74 ms for the previous two-tail preparation: speedups
1.54/2.27/2.76×. All pass the declared 1.1× bar. One-time global preparation
took 4.05 ms in this run. It is excluded from reused timings and must be paid;
the single-context preparation saving covers it after roughly three contexts.
This setup timing is descriptive, not a stable latency guarantee.

The shared cache holds 11520 scalars and each context holds one 1152-vector.
The previous representation holds 2305 scalars per context. Thus total extra
state becomes smaller starting at ten contexts. Common upstream response
state and dense native maps remain charged on both sides. The global cache
could already be present for other consumers; no such reuse is assumed in
this standalone comparison. Execution still evaluates the native bilinear
layer, so this is a preparation/state improvement, not whole-model compression.

CPU receipt: `SHARED_POLYNOMIAL_TAIL_V1_RESULT.json`. Native implementation
replay is registered in `ops/run_shared_tail_native_v1.py`, with frozen binding
`SHARED_TAIL_NATIVE_V1_BINDING.json`: use the existing 48-template panel and
compare all prior scores within $10^{-5}$ while retaining the previous native
fidelity bars. This is not an independent fresh-text panel. Read
`SHARED_TAIL_NATIVE_V1_RESULT.json` for its authoritative terminal outcome.

The managed native replay completed in 5.09 seconds with all five predicates
true. Maximum score disagreement with the previous three-term program is
$3.81\times10^{-6}$. A separate relative-effect audit finds 0.0616% target and
0.2068% control disagreement in family 0, and exact effect agreement in the
other three families, with no material sign reversals. This checks the effect
scale rather than relying only on absolute score agreement. The native
prefix/background/suffix remain in use. See
`SHARED_TAIL_NATIVE_V1_RELATIVE_AUDIT.json`.

## Transfer to six new exact lexical contrasts

The frozen shared-tail executor now passes 48 rows covering
civilisation/civilization, modelling/modeling, criticised/criticized,
organisation/organization, metres/meters and colours/colors. Four instruction
styles are reused, each with paired British/American cues. The row builder
checks single-token endpoints, exact prefix-plus-completion tokenization, and
absence of the exact contrasts and token sequences from 54 local row inventories
containing 37 prior contrast pairs and 1085 sequences. Some roots or inflections
are related to earlier concepts: this is lexical transfer, not six independent
semantic mechanisms or corpus-wide OOD.

All four registered predicates pass in 5.15 seconds. Original-interaction
target errors range 0.122–0.278%; controls 0.530–0.828%. Incremental errors
against the exact five-bank program are 0.119–0.253% target and 0.280–0.556%
control. Native paired cue capability is positive in every style and there are
no material sign reversals. Live upstream generation and native prefix/suffix
remain dependencies; no fitting uses this panel.

Primary receipt: `SHARED_TAIL_LEXICAL_V1_RESULT.json`, rows and reproducible
builder `SHARED_TAIL_LEXICAL_V1_ROWS.json` / `shared_tail_lexical_rows_v1.py`.
The receipt's inherited phrase “six existing lexical contrasts” is a stale
scorer label: these six exact endpoint pairs are new under the stated inventory
check. Its opening scope and frozen rows describe the actual experiment.
The registered next countercheck removes the full residual-self term and
measures approximation error against that component's own effect, using
`SHARED_TAIL_LEXICAL_SELF_V1_RESULT.json` when terminal. This guards against
the surrounding circuit hiding approximation error.

The own-component countercheck has completed. Compression error relative to
the removed self term's own effect is 0.418–3.946% for targets and
0.967–2.218% for controls, passing 10% in every style. The separate prediction
that self removal changes at least 10% of the target interaction in every
style **fails**: the four fractions are 6.40%, 7.73%, 8.87% and 42.30%.
The compressed component is therefore less prominent in three styles than on
the earlier lexical panel, while its own effect is still preserved to the
registered accuracy. This does not establish semantic selectivity. A
descriptive leave-one-pair-out audit is saved in
`SHARED_TAIL_LEXICAL_SELF_V1_PAIR_AUDIT.json` to check lexical concentration.
