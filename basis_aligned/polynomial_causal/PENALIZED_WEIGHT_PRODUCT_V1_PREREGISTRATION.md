# Explicit component-energy bias: first native joint fit

The prior unpenalized 128-product fit stalled with cancellation ratio14264.96.
Equivalent-loss refinements failed the original1e-8 improvement bar. A fixed-reader
penalty path motivates lambda=0.01: retain80%of captured coefficient energy while
reducing cancellation to1.245. That exploratory choice is now frozen for this run.

Fit128 products against the same full50304-output coefficient tensor. Minimize
reconstruction plus lambda times sum of individual component energies, normalized
by native tensor energy. This is a changed inductive bias, not a silent ridge fix.
Report reconstruction, penalty, full-gradient stationarity and cancellation
separately. Conditional writers solve the exact regularized normal equation.

Initialize from the saved unpenalized best model, with underlying a/b rows reset
to unit norm (an exact function-preserving gauge change). All prior optimization
is charged as warm-start provenance; this is not an independent restart. New
Adam2000 updates and L-BFGS follow. L-BFGS numerical change tolerance is1e-16.
Best-model selection and plateau use optimization_loss, not reconstruction alone.
Convergence bars remain: five diagnostic checks with relative objective movement
<=1e-5, relative stationarity<=1e-4, max gradient<=1e-7. Twenty-five unchanged
L-BFGS calls terminate as line_search_stalled unless those convergence gates pass.
The first development test found a five-call stall cutoff premature for collecting
five diagnostics; the cutoff was corrected before any native run. Gradient bars
were unchanged. Planted-optimum/resume and nonstationary-stall controls must pass.

Managed chunk permits540 fitting seconds,900 whole-run alarm; resume saves full
optimizer state. 442368 fitted scalars (readers+writers),0native forwards,FP64.
Feature raw conditioning is descriptive; the regularized solved Gram must be
finite and condition<=1e12. Saved optimization-loss replay must be<=1e-10.

- pred_a_instrument: live source bindings, controls and saved-objective replay hold.
- pred_b_converged: registered plateau and full-gradient bars hold; false means
  unfinished or stalled optimizer, not absent structure.
- pred_c_stable_useful: cancellation<=10 and captured coefficient energy at least
  75%of the frozen unpenalized best capture, AND penalized objective at least1e-4
  below the frozen fixed-reader lambda.01 baseline. The extra gain requirement
  makes this a live reader-refitting test; the initializer already meets the
  first two descriptive bars. A pass is a numerical representation
  screen, not a circuit. A miss receives the standing red-team audit.

The full goal still requires OOD prediction, extraction, selective removal and
composition/reuse. Neither low error nor numerical stability substitutes for it.
