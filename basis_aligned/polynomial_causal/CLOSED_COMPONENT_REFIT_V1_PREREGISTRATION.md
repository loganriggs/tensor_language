# Matched local refits after one graph connection changes

Moving the shared parent while freezing any of its consumers would silently change the background. The connected closure is groups **10, 11, 16, 19, 25**, with shared parents **1, 8, 9**. All other 59 groups remain fixed. The two arms retain original connections or add parent 1 to group 11. Both use the same original frozen background and original five-writer output span. Starting local parameters come from the original graph or the completed V2 edge proposal respectively.

Subtract the frozen groups from the complete native coefficient tensor, then project its output onto an orthonormal basis of those five original writers. The local problem is exactly equivalent up to a constant for all permitted changes: shared readers and private input spaces can move freely subject to the existing orthogonality constraints; writers can move within the fixed five-dimensional output span; all five quadratic cores are jointly eliminated at every evaluation. No text or model-body forward passes are used.

The legacy objective begins at 1 using the original native normalization $N$. Its local value is shifted from the full penalized objective by the fixed CPU-measured offset **-0.09978352475059615**. Shifted local capture/residual quantities must not be reported as whole-model capture. Initial and final full objectives are converted explicitly; independent full-graph objective-difference and graph-injection tests pass.

Each separately managed lane1 arm gets 1,200 soft seconds, at most 1,000 re-encoding cycles, 200 accepted L-BFGS steps per cycle, and a 1,800-second process alarm. Reuse the established bounded solver, full-U coefficient objective and 0.01 group-energy penalty. Stop early only when both fresh-coordinate gradient and recent progress criteria hold. The smaller component makes more local iterations feasible; neither a time limit nor earlier planted recovery failures imply convergence or absence of structure.

CPU preflight A: whole/local objective-difference, executor/injection and reduced-objective identities <=1e-9; directional finite-difference relative error <=1e-5. Actual maximum FD error is8.42e-8. CPU initial evaluations take0.13–0.31seconds.

Per-arm bars, frozen before GPU outcomes:

- A: CPU/GPU initial objective, reduced identity, same-function re-encoding, executor and full-graph injection errors <=1e-8; accepted objective increase <=1e-8; inner true relative normal residual <=1e-10; raw coefficient entries within1+1e-10.
- B: fresh maximum packed gradient <=1e-7 and relative change over20acceptedsteps <=1e-6. Each criterion is measured again in fresh coordinates.
- C: its own normalized penalized objective improves by at least1e-6. This is an optimization diagnostic, not acceptance of the graph change.

After both complete, the aggregate records A all numeric checks, B both converged, and C new-edge full objective minus matched-original objective <=1e-6, at least1,000floats saved, and parent1 named-node removal energy >=1% of each group's energy in all three consumers **11,16,25**. Report reader movement and prices. Even aggregate C without convergence is only a limited comparison. No positive weight result establishes OOD prediction, extraction, selective behavior or compositional semantics.

Sources and transitive local Python dependencies are content-bound in CLOSED_COMPONENT_REFIT_V1_BINDING.json. Both wrappers undergo standard model-free enqueue gates. New fits are named run_closed_component_original_v1 and run_closed_component_new_edge_v1. Existing sources and completed verdicts remain immutable.
