# Is the calibration quadratic a reused vocabulary-spread computation?

2026-09-10, original handoff/pilot authority. Parent fixed q,w,Q,beta and its
conditional two-consumer/manipulation evidence; previous single source terms
and fitting-direction refinements failed. This tests an explicit operation,
not activation-variance preservation, a rank sweep or entropy estimation.

For the native unembedding U (V=50304 rows,d=1152), let
K=(U-rowmean(U))^T(U-rowmean(U))/V. Then E(u)=u^T K u equals the population
variance over virtual pre-softcap vocabulary scores Uu. Determine coefficients
from weights alone: remove traces to get Q0,K0; alpha=<Q0,K0>/||K0||²;
gamma=(trace(Q)-alpha*trace(K))/d. Define q_E=alpha*E(u)+gamma*||u||²+beta.
No fitting activations/labels or selection of coefficients after evaluation.
Keep native radius dependence explicit: epsilon prevents assuming ||u||²=d.

Evaluate both the global weight relation and approximation on the same reused
42FW/16Pile rows. A global coefficient mismatch does not by itself establish
a failure on native reachable states, so run the conditional test regardless.

Four readout arms: native q, complete q removal, q_E replacement, and frozen
FIT mean q replacement (from first stability result). Retain g=h-qw and native
RMS/softcap. Online MLP17 projection replacement first4rows percohort; independent
native facade first4FWrows. 15 evaluationbatch forwards +1facade +2online =18,
70length256 sequence instances. All original545902902 parameters retained.
Stored K adds1152² FP64 coefficients if retained; conceptual U reuse does not
hide its57950208 parameters or required upstream computation.

Predictions:

* A instrument:18forwards70seq, finite outputs and nondegenerateK0; savedQ/beta
  replay from nativeweights <=1e-10rel/max; direct vocabulary variance vsK
  contraction on16 fixedrandom normalized inputs <=1e-10rel; native formula,
  independent facade and onlineq_E replacement <=1e-3abs/1e-5rel.
* B numerical compatibility with global form: ||Q-alphaK-gammaI||/||Q0||<=1e-10.
  A pass is numerical compatibility, not an exact rational identity proof.
* C native sufficiency: q_E scalarrelativeerror<=.10, centered full-vocabulary
  replacement error / original complete-removal effect<=.10, and absolute mean
  next-token CE change<=.01nat, each on BOTHcorpora. No datafit if it fails.
* D frozen mean control replay: common/rare/all mean CEs for native, complete
  removal and frozen FIT mean equal first stability result within1e-10. This
  is an execution control, not new mean-replacement evidence. A/Dfailure invalid.

If B/C fail, close this weight-defined vocabulary-spread formula; do not label
the scalar confidence or entropy. Preserve conditional q positives. No shifted
vocabulary subset, new metric, coefficient tuning, output-axis refinement or
rank rescue. FineWeb document grouping unknown, Pile corpus shift only, evaluation
texts reused. <=900seconds managedruntime; CPU exact/planted/perturbed controls
already executed and bound. No full-logit dump; compactperrowloss and K artifact.
