# Compiled mixed edge closes the conditional native join

The existing attention mixed-edge core, installed at the later edited query before MLP11, restores all8openedv4 weak-component prediction cells. Maximum output-effect discrepancy is.00025394 of the weaker number effect (.0254%), passing the stricter.1% instrument-level precision bar. Old singleton/joint/pre11 effects replay exactly. The uncorrected postattention join remains6/8; its two misses are preserved.12prefix16full20joined suffixes plus8explicitMLP11calls,1.541s reported runtime.

This is a composition result at a declared interface: baseline and singleton postattention states are still native, and the remaining MLP11/suffix12–17 are still native. The core supplies the missing local mixed state; it is not asserted to explain the entire downstream interaction by itself. Earlier failure of this edge as a standalone interaction ablation is not retracted. No new OOD data or independent full circuit is established.

## Independent CPU replay and explicit shared writers

Export v4_compiled_edge_join_v1_cores.pt contains per-context coefficient cores, predicted edge vectors and causal positions. CPU-only audit loads this artifact without model/checkpoint/tokenizer/prefix/suffix access. Saved GPU-edge replay error<=1.45e-15. Re-expression through27writer directions matches the original core at<=5.36e-15 over six amplitude pairs, including off-corner values; either zero axis gives exactly zero. These off-corner comparisons prove equivalence of the two conditional executors, not new native behavioral generalization.

The existing core has shape3x3 per head between rational query and key/value features. Split the output into three native-derived writers per head: value_base, value_linear and cached_value. For head routing contraction r(u,v) against the query-feature difference and the key polynomial features, let s(v) be the source input RMS scale. Writer amplitudes are

    beta_base = r(u,v)/s(v) - r(u,0)/s(0)
    beta_linear = v*r(u,v)/s(v)
    beta_cache = r(u,v)-r(u,0).

Thus edge(u,v)=W beta(u,v), W[1152,27], without losing shared norms, cache or either QK product. Implementation mixed_edge_writer_features.py. This exposes an explicit intermediate feature interface for contracting the following bilinear layer; it is not another independent compression of its three matrices.

Each full-output core contains31,296values/input, including31,104writer values and192scalar core/norm values. Writer matrices have numerical rank27 at relative1e-10 in all48inputs. This concerns unrestricted linear writer coordinates only; it is NOT a lower bound on the two-dimensional nonlinear amplitude manifold, total circuit complexity or semantic feature count. No rank reduction is claimed. All context-dependent coefficient generation, source/selector construction and singleton/suffix execution remain charged.

Next folding target: contract MLP11 Left/Right with this explicit27-writer interface, retain the background and exact quadratic RMS geometry, and compare joint tensor decompositions against the original implicit bilinear factors and canonical symmetric coefficients. An approximation must preserve signed native effect and controls on the actual feature manifold; coefficient compression alone cannot establish a circuit. This step must not import a downstream tangent reader as a free finite-effect predictor.

Primary result v4_compiled_edge_join_v1_result.json and managed log; portable coefficient artifact v4_compiled_edge_join_v1_cores.pt; independent receipt COMPILED_EDGE_CPU_REPLAY_V1.json and audit_compiled_edge_cpu_v1.py. Full goal remains active.
