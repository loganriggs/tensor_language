# Composed query reuse and explicit source-column update

Joint Q1+Q2 baseline reuse at attention11 passes all16 opened native fidelity and selectivity cells. Worst number-effect error2.5433%, worst control-effect error2.2960%; minimum retention.93509 and maximum collateral ratio.08093. Full and both singleton effects replay exactly.12prefix40suffix calls,1.456s reported runtime. Prediction keys retain the old runner's q2 spelling, but this runner explicitly scores arm freeze_both; see its source and preregistration.

The joint-minus-single1-minus-single2+full effect reaches.10457% of full number effect, yet19.73% of the smaller singleton change. The mixed term is small under the registered full-effect measure, not uniformly negligible relative to the dropped pieces. Data are opened OODv2; this is prospective composition of fixed interventions, not fresh-input evidence. Per-input oracle selection and native source generation remain dependencies.

Important positive-result redteam: only one pre11 token changes. Queries therefore change only at that token's row, while keys/values change a causal source column reaching later destinations. Query-freeze success is consistent with weak feedback from the source row and does not show all model queries are redundant or context-independent.

## Exact next folding consequence, executed on CPU

For fixed normalized/rotated query fields and one changed source position s, let

    a_j[t,h] = (q1[t,h] dot k1_j[s,h]) * (q2[t,h] dot k2_j[s,h]) / head_dim^2
    delta_write[t] = sum_h O_h {a_1[t,h]*(v_1[h]-v_0[h]) + (a_1[t,h]-a_0[t,h])*v_0[h]}, t>=s.

For t<s the change is exactly zero. Each output-projected head writer can be computed before token broadcast; baseline writers may be cached across edits sharing context. This reuses the established attention8h2_rankone_edge_v1.rankone_write identity; the new object retains both changed keys and changed mixed value across all heads. It is cubic in independent source key/value ports with fixed queries, but raw-state normalization remains explicit upstream. No small sparse core or model-global polynomial is claimed.

fixed_query_source_column.py implements both cached and uncached execution. Independent dense causal attention subtraction agrees to4.63e-16 relative error; cached/uncached outputs exactly agree. Pre-source writes vanish. Deliberately omitting second-key changes produces89–109% error, so that interaction is live. These are synthetic ports; native installed replay is still required.

For d1152,T20,H9, static O remains1,327,104values. Baseline query ports46,080; baseline source K1/K2/V3456, new source ports3456, cached baseline writers10,368. Per edit, cached writer projection uses1,327,104 multiplies versus26,542,080 for a dense per-token O projection; uncached costs2,654,208. Add92,160 routing-dot and414,720 broadcast multiplies. Source projections/norms, baseline context, cache construction, remaining suffix and output materialization are separately charged. This is conditional repeated-edit compute reuse, not model parameter compression. V1 CPU receipt originally quoted the cached price before a cache input existed; V2 corrects that by implementing and testing cache reuse and recording both prices. Preserve V1 as an accounting correction, not a separate experiment.

Next native discriminator: install this source-column executor at attention11 and replay the joint-query-frozen path, then test a frozen prospective input panel before assigning reusable-circuit status. Same reader norms or exact local arithmetic cannot substitute for that check. Full goal remains active.

Primary native receipt: attention_query_composition_v1_result.json; CPU receipt FIXED_QUERY_SOURCE_COLUMN_CPU_V2.json. See ATTENTION_QUERY_COMPOSITION_V1_PREREGISTRATION.md and check_fixed_query_source_column.py.
