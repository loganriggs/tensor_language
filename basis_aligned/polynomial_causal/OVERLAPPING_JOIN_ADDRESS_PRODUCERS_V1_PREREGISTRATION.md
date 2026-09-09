# Does one native join head carry two address-producing paths?

Direction/gain diagnosis rejects direction-only dominance across both consumers,
but shows that directional key changes damage both. Test a specific within-head
split using existing producer traces, not another gain fit or port subset.

For a->b->c->d with b->c last, the backward L2H2 write reads the earlier a->b
binding. Define three exact additive write components:

E: direct embedding of its value token b, via the existing L2H2 source-value
column with native L2 pattern/RMS and the residual factor E/4.
O: existing L1 previous-key-position -> value-position path at a->b, then
the native L2H2 join. This path receives the native contextual state at key a;
it is not assumed to be a pure embedding or a successfully interchangeable
origin variable. The earlier origin-field swap null remains binding.
R: exact remainder of the full backward write after E and O.

Use existing producer_writes with only the source-value column for E and
origin_write for O. No new producer/rank/source search. Source support is fixed
to the same two destination positions. All512 opened requests,32 worlds,two
orders,two origins,four hops from overlapping_join_normalizer_reference.

Use the registered direction-only key interface: subtract each of the eight
E/O/R subsets from raw final K1/K2 inputs at the original source RMS gain.
Keep final queries, values and skip native, all four final heads. Thus this
diagnosis separates directional use from the now-demonstrated gain confound.
These are native projection-port interventions, not a residual edit with all
normalization held physically unchanged. Compare all-subset and empty-subset
query logits with the saved full-direction and native outputs from key-gain
rows SHA83f2ac14394f78c7b6e9e084328c617f9b631bd0c865fb347c29bcdf2fda23a7.

A: exact subset execution versus independent native projection hooks
abs<=1e-9/relative<=1e-10; saved full-direction/native replay; original producer
controls pass; same-query-fork write reuse. No component selected by outcome.

B two-address task nomination, separately in each population: E-only cut must
recover [.8,1.2] of the full-direction cut's signed forward-query hop3 gold
effect and at most .2 absolute ratio on the backward query. O-only must give
the converse. R-only absolute ratio must be<=.2 for both. Full-effect
denominators below .001 invalidate rather than filter a panel. These fixed
ratios are a diagnostic task bar, not the full-distribution success criterion.

C sufficient two-path direction interface is separate: E+O must predict full
E+O+R direction-cut centered query effect within1% relative RMS, floor1e-6,
in every population/query/hop group; query teacher KL mean<=.001,p99<=.01.
Report full factorial, signed interactions and maximum KL. Failure may not be
repaired by adding paths, changing sources, selecting a query/hop, or replacing
O by a larger Y1 field. No positive task panel can override the aggregate B/C.

CPU threads2,batch8 FP64,alarm180s,tensors below256MiB,no GPU/training.
Reuse existing producer, origin, key-direction, source-port and scoring modules.
All387968 native export constants, routing and contextual input to O remain
priced. A nomination would justify a fresh, explicitly scoped address-edit
test; no structural reduction or renewed origin-field interchange claim yet.

Additional pre-execution algebraic instrument: with source gain, V and query
fixed, the product of two affine key factors is quadratic in E/O/R removal
scales. The three-way finite difference must vanish. Compare sums
L7+L1+L2+L4 versus L3+L5+L6+L0 with the shared correspondence scorer,
so the relative tolerance uses live logits rather than division by zero.
Pair interactions may remain large; this is not an independence prediction.
