# Prospective acceptance rubric: task14 FIT-localization-v2 physical compiler

**Frozen:** 2026-09-04 10:14 UTC. **Phase:** CPU-only review rubric.

This rubric was written while the compiler was still being built and before any localization result existed. It does
not modify the scientific authority, approve an implementation, authorize model or GPU access, or open SELECT, TEST,
or OOD. It fixes what a fresh reviewer must establish before the compiler can license an implementation.

The scientific authority is exact commit
`8f41f51cdf7e073063201cc48760622607ce91b9`, preregistration SHA-256
`3ea31387f611d0d095895dec6ed0859e1d99b2ad91a5d5adfb7be178bf127f59`. Its independent approval is exact commit
`2ffd6cf77998a6c7fb6af0c4e89c742bf1bbb923`, review SHA-256
`2905aeb040fad2d16062a22e3c4d32d9dd6953c468724ff51a80ab9fa849d384`.

## Acceptance conditions

A fresh reviewer must reject the compiler unless all of the following are true.

1. **Exact ancestry and inputs.** The compiler binds the two hashes above, the exact partition and donor bytes, all
   upstream FIT-authority hashes, and the fixed model/checkpoint identity without substituting current worktree files.
   The compiler itself imports no model, CUDA, activation, result, queue, or later-phase object.
2. **Complete conditional call graph.** Every possible FIT-only branch has a stable call or guard ID and explicit
   dependencies. The graph contains native logits/residual caches; all 38 site screens; full-state intervention
   ceilings; native gradients; rank-one joint fits at every eligible site for all five seeds; discovery-only H/Q,
   seed, and top-two-Q selection; the two family-only syntax fits; rank-two and rank-four falsifiers; locked
   validation; cellwise necessity; the conditional two-Q redundancy test; ordered H-to-Q reset/rescue; and terminal
   evaluation. Empty, ineligible, unhealthy, or failed branches terminate exactly as the authority says rather than
   silently falling through to a cheaper path.
3. **No result-dependent schedule invention.** Runtime results may activate only conditionals already present in the
   graph. They may not change batch membership, fitting data, objective weights, steps, seed aggregation, ranks,
   sites, donor relations, thresholds, controls, or terminal precedence. Discovery selects; validation never
   reselects or recalibrates.
4. **Exact physical units.** Every executable node fixes prompt and donor IDs, semantic position, boundary, rank,
   seed, objective, optimizer update range, physical batch, sequence length, dtype, cache input/output, and expected
   forward/backward multiplicity. Variable-length examples are either grouped into explicitly fixed equal-length
   batches or handled by a demonstrated numerically safe mask; a native and intervened comparison uses byte-identical
   physical batches.
5. **One computation, one accounting entry.** Shared native states, gradients, and full-state ceilings are computed
   once where mathematically identical and referenced by hash thereafter. Conversely, calls that differ in fitted
   state, donor, intervention order, or upstream activation are not incorrectly deduplicated. The reviewer must
   independently reconstruct the best-, ordinary-, and worst-case call counts from the graph rather than accepting a
   printed total.
6. **Resource and storage closure.** The manifest specifies retained arrays, shapes, dtypes, byte totals, temporary
   peak storage, optimizer state, forward/backward counts, and an explicit worst-case physical work bound. Execution
   must have a fail-closed eight-hour wall-clock guard that cannot serialize a partial scientific terminal as a null.
   A throughput estimate may inform feasibility, but cannot replace the hard guard or be reported as measured task-14
   runtime before execution.
7. **Intervention semantics.** Rank-$k$ swaps implement
   $r_x+UU^{\mathsf T}(r_d-r_x)$ at exactly one registered residual position. Full-state ceilings use the identity
   projector. Necessity uses the frozen discovery midpoint. The two-site redundancy arm applies the later edit to its
   current activation after the earlier edit. Reader reset and rescue use the distinct native-target,
   upstream-changed, and natural-donor baselines specified in the authority.
8. **Numerical and failure semantics.** Nonfinite quantities, small denominators, absent causal ceilings, collapsed
   coordinates, unhealthy seeds, sign reversals, overshoot, higher-rank rescue, and mutually exclusive
   necessity/redundancy outcomes map to the exact frozen gates. Float precision and reduction order are explicit.
   Instrument failure cannot be published as a scientific null.
9. **The spectral calculation stays a diagnostic.** If the compiler includes the task-conditioned symmetric operator
   $A=\operatorname{mean}_n\frac{\sigma_n}{2}(g_n\delta_n^{\mathsf T}+\delta_ng_n^{\mathsf T})$, it uses DISCOVERY
   objects only and may initialize or compare with finite DAS. Its eigenvalue, eigenvector, projector distance, or
   local-effect correlation cannot select a validation result, satisfy a causal gate, or replace any finite
   intervention.
10. **Immutable evidence lifecycle.** Output paths are creation-only and run-unique; pre-existing files, duplicate
    call IDs, missing calls, changed ordering, partial chunks, or wrong hashes fail before a terminal is interpreted.
    A deterministic root hash commits to the complete call graph and any chunk manifests. A completion receipt binds
    compiler, implementation, authority, graph root, result, evidence, and run log without overwriting an earlier
    receipt.
11. **Adversarial tests.** Focused tests must reject, at minimum: one dropped coordinated-subject arm; a donor crossing
    the discovery/validation boundary; validation-based selection; one missing seed; a rank-two branch promoted to
    success; swapped reset/rescue baselines; simultaneous single-site and redundancy success; changed intervention
    order; a disconnected diagnostic; a changed dtype/batch/call count; a shortened optimizer schedule; a stale or
    occupied output namespace; and an eight-hour overrun represented as a null.
12. **Independent reconstruction.** A different agent must review exact committed bytes, regenerate the graph and root
    hash from source, independently recompute censuses and resource totals, run the focused tests with model and CUDA
    disabled, and state a narrow verdict. Approval can license only construction of a separate FIT implementation;
    it is not model/GPU execution or enqueue approval.

## Why this is an interpretability experiment

The primary output is not a lower rank. Success would identify a causally transferable complete-subject-number state,
distinguish it from local noun morphology, locate where it is present, test whether one or two sites are necessary,
and test an ordered upstream-to-downstream handoff. Rank two and four are falsifiers of the claimed one-dimensional
state. The later weight translation remains a separate phase: only after finite interchange, necessity, and reader
evidence establish the operational variable should we fold the identified projector through the model's bilinear
weights and ask which exact weight terms implement it.
