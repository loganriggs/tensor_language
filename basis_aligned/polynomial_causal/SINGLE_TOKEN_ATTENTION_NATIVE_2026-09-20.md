# Exact native row+column attention repair

The explicit single-token attention11 replacement passes the strict.1% effect-replay instrument and all16 openedv4 fidelity cells; worst relative output error.04821%, old full-native replay0, zero-edit baseline4.77e-6. It recomputes source Q1,Q2,K1,K2,V and combines a source-column update with a query-row correction against updated K/V. The baseline queries, all baseline source K/V, attention output, first-value cache and output weights remain explicit dependencies. No native attention11 call is made inside the edited replacement.

Native source/role selector remains unchanged and still passes only8/16selectivity cells. Exact replacement cannot fix the full-native selector. This is local extraction at a declared background interface and conditional arithmetic reuse, not an identified selective circuit, independent behavioral predictor, parameter-compressed model or new OOD success. v4 was opened by the failing query-freeze test before this repair.

Counts18prefix48suffix6baseline preparations;1.708s reported run. Independent source-port CPU proof tests and omitted query-source mixed-term negative control preceded native evaluation. Runtime does not establish a speedup: current adapter clones full K/V and uses FP64 summation. Literal cost includes five source projections, uncached old/new column writer projections, one row output projection, scans of a row and column, full baseline production and the remaining suffix.

SINGLE_TOKEN_ATTENTION_SCALE_AUDIT_V1.json records measured effect norms: subject1.17–7.63, attractor.01267–1.959; weakest unitB attractor norm.01349. Weak attractor denominators amplify numerical and model approximation errors, but strict replacement replay passes on those same denominators. No failed cell removed or normalized by a stronger unrelated output.

Next discriminating test is V4_SOURCE_ORACLE_AUDIT_V1_PREREGISTRATION.md: recipient-native derivatives with the same23ports and bounded LP. The current fixed-bank failure could be bad contextual selection, first-order/finite mismatch, or insufficient dictionary. A per-input oracle pass would rule out dictionary insufficiency for these edits but would not extract the oracle. All inputs are opened; fresh semantic prediction remains outstanding.

Primary receipt single_token_attention_native_v1_result.json and managed log; source native_single_token_attention.py, source_query_row.py and fixed_query_source_column.py. Full goal active.
