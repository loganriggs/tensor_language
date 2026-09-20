# Fresh query reuse fails; independent native reference confirms approximation failure

Frozen48textv4 test passes instrumentation and native capability in every cell. Fidelity passes12/16 (subjects8/8, attractors4/8); both full and compiled selector selectivity pass8/16 (subjects7/8, attractors1/8). Frozen bank tangent prediction passes0/16. Worst approximation number error17.47%, control error8.75%. Every cell and negative retained; no refit or capability exclusion. Counts18prefix48suffix6baseline attention,1.69s runtime; old installed/full replay0, zero-edit baseline4.29e-6.

Independent negative-result audit evaluates native joint-query-freezing at exactly the same edits. Explicit executor/native-frozen discrepancy<=.02618% of full number effect, below preregistered.1% instrument bar. Native freezing also fails the approximation gate. Thus this particular failure is supported as a failure of baseline-query reuse, not the tested source-column identity. Counts18prefix60suffix6baseline preparations plus12query projections;1.74s. Other bugs are not categorically excluded. The full frozen selector already fails eight cells without the replacement, so correcting the attention approximation cannot establish selector selectivity or independent prediction.

Freshness is now consumed: these v4 outcomes are opened. Subsequent correction on them is explanatory/repair evidence, not a new prospective OOD test.

## Exact next mathematical consequence, executed CPU

A one-token edit changes one query row and one key/value column. The previous executor handles the column with queries fixed. Complete the attention change by adding the source query row against the **updated** key/value fields:

    Delta A = column(q_old,K_new,V_new; K_old,V_old)
            + row_s(q_new,K_new,V_new) - row_s(q_old,K_new,V_new).

The row difference contains both QK products, all causal source positions j<=s, all heads, and the output projection. Updated source keys/values are required in the row correction; baseline keys/values would drop query-source mixed terms. This is an exact algebraic decomposition, not a higher Taylor order or per-case learned gate. It changes the object by restoring precisely the computation previously omitted.

source_query_row.py implements the row term from normalized/rotated query/key and mixed-value ports. check_source_query_row.py compares the row+column result to an independent dense causal two-QK attention evaluation with all five source factors changed. Relative errors1.71e-16 and3.03e-16; omitting query-source mixed terms causes21.4–29.4% relative error, a live negative control. These are synthetic ports, not yet native installed evidence. Source normalization, cached first-layer value, baseline context and output weights stay explicit.

With cached baseline ports this scans one source column and one query row, O(T*head_dim*heads), rather than recomputing T² pairs. It still needs all five changed-token projections, the dense O map for row and column writers, baseline production and the remaining suffix. Static weight count is unchanged; whole-model simplicity and circuit identification are not established. Native adapter currently freezes queries and must be extended separately before claiming this exact identity as an executable native replacement.

Next test: native row+column installed replay on openedv4 with unchanged selectors, then separately revisit the failed source/readout selector. Exact attention replacement cannot repair an already incorrect full-native selector. Positive subject fidelity remains a scoped finding; neither route meets the full goal.

Primary receipts: fixed_query_fresh_v4_result.json, fixed_query_fresh_v4_audit_result.json and managed logs; SOURCE_QUERY_ROW_CPU_V1.json. Preregistrations FIXED_QUERY_FRESH_V4_PREREGISTRATION.md and FIXED_QUERY_FRESH_V4_AUDIT_PREREGISTRATION.md. Current mathematical review1336 remains current; nextdue1636UTC.
