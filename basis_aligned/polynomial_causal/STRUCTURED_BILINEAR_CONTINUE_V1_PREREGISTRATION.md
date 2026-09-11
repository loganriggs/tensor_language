# Saved-state structured bilinear continuation

11 September 2026, registered after Native V2 results and before continuation.

Both starts stopped after about 302 joint seconds, at 1.4968% and 0.9388%
full-U coefficient capture. Gradients remain large and objectives are improving.
The original convergence and quality predictions failed; they remain failed.
This is not a structural negative. More optimization is warranted before
changing this representation or introducing data into discovery.

Resume the exact model and persistent L-BFGS state for seeds 0 and 937.
Same 276480 coefficients, 4608 products, mixed-radix wiring, exact full-U
coefficient objective, .01 individual component-energy penalty, history size20,
strong-Wolfe line search and original convergence criteria. No initialization,
corpus, activation weighting or model forwards. Native U and all background
parameters remain separately charged; this is not circuit identification.

Additional budget: 1800 soft joint seconds and at most 3000 outer steps per
seed, whichever occurs first. Save continuation state and progress receipt
every 300 seconds and at termination. Saved native checkpoints remain intact.
Periodic replacement applies only to this run's own latest checkpoint.

Predictions:

- A: exact point replay, loss and relative-stationarity error <=1e-8; the loaded
  optimizer n_iter, function-evaluation count and history length match the
  saved state before any step; no data access.
- B: both starts meet original relative stationarity <=1e-4, max gradient
  <=1e-7 and five-diagnostic relative objective change <=1e-5.
- C: both captures exceed 1.05 times the earlier block reference .08634383041327387.

Also report improvement over each starting checkpoint, cost and full learning
curves. These diagnostics cannot replace B/C if they fail. Stop early on
nonfinite values or ten consecutive effectively unchanged accepted-point diagnostics
(absolute difference <=1e-15), recording a stall rather than convergence.
Do not change thresholds to make a stationary poor fit pass.

Decision: if still improving and nonstationary, this family remains unresolved;
use measured progress and conditioning before deciding more time or a solver
change. If stationary and poor, inspect wiring/depth and restart dependence.
If quality holds, examine factor stability and reusable operations before frozen
FineWeb validation and intervention tests. Neither outcome establishes all four
circuit properties.
