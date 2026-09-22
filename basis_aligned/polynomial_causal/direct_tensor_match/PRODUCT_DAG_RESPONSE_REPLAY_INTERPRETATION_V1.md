# Exported product graphs preserve source-candidate finite changes in float32

22 September 2026, 03:54 UTC. Both frozen graphs pass the registered numerical-preservation test on16,384openedstates and2,494fixedsame-token-position pairs. No fitting or additional graph edits were performed.

| Seed | Pooled value difference | Pooled response difference | Worst output value difference | Worst output response difference |
| --- | ---: | ---: | ---: | ---: |
| 1001 | 1.67e-7 | 3.96e-7 | 6.58e-7 | 8.86e-7 |
| 1002 | 1.88e-7 | 4.83e-7 | 9.23e-7 | 1.27e-6 |

Numbers are relative norms, not percentages. Differences compare the graph against its source reader-sharing candidate. Gates are1e-5pooled and1e-4for every original output, separately for values and finite changes. All pass. Source artifact and graph hashes are checked before execution. Both paths use the same float32 reader bank and physical output coefficients, with changed multiplication association. Results are computed onCPU in1,024-state batches; no GPU-backend claim or runtime advantage follows.

This closes the precision gap left by the256-state float64 export check. The graph compiler still inherits the source candidate's native reconstruction and removal failures. In particular, numerical agreement at1e-6does not turn roughly50%small-feature native response errors into accurate circuits. This is transformation fidelity, not native circuit identification or independent OOD evidence.

[Protocol](PRODUCT_DAG_RESPONSE_REPLAY_PLAN_V1.md) · [All16coordinate measurements](PRODUCT_DAG_RESPONSE_REPLAY_V1.json) · [Executor](audit_product_dag_response_replay.py) · [Rewrite and literal cost](QUARTIC_PRODUCT_REASSOCIATION_INTERPRETATION_V1.md).
