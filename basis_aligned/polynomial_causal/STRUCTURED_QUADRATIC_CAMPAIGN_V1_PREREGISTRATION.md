# Convergence-controlled structural comparison, first wave

Authority: the user's request for a large list, lots of data and optimization,
explicit convergence, and red-team review of negative results. The24-hypothesis
campaign explanation defines the broader agenda. This wave compares four distinct
representations under two fitting metrics, four starts each:32 fixed configurations.

Representations, all with signed residual-space output writers and real factors:
- product:128 products,256 independently learned normalized input readers.
- square:256 signed square features,256 normalized input readers.
- shared_reader:64 learned dictionary readers;128 products of normalized dense
  combinations of that dictionary. Common input reads are computed once.
- block:32 output blocks, each a learned signed sum of8 squared readers.
  The output writer is shared across the8 squares. Not a sparse token-output model.

Metrics: (a) exact full-input coefficient Frobenius norm after all50304 unembedding
rows, evaluated implicitly; (b) mean squared unembedding-numerator error on all51200
training states in UNSUPERVISED_DATA_V2_STATES.pt. Decode scales in FP32 then fit in
FP64; subtract the native Down bias from target y and retain that bias separately
in a future executor. Validation/test states are not used for fitting or stopping.
All output writers solve unregularized least squares conditional on current input
factors. No linguistic labels, task-specific selection or sparse loading penalty.

Initializations: product seed0 uses128 native products with largest standalone
energy under its metric. Other configurations use fixed Gaussian starts; all seeds
and configurations are committed in STRUCTURED_QUADRATIC_CAMPAIGN_V1_CONFIGS.json.

Optimization:2000 Adam updates at.03, then L-BFGS with strong-Wolfe line search,
up to10 inner iterations per outer call, history8. Diagnostics every50 Adam updates
and5 L-BFGS calls. Track actual closure counts and L-BFGS iterations, not just caps.
No global-optimality claim. A managed chunk allows540 fit seconds,900hard seconds
including setup/checkpointing. It saves a resumable optimizer/model/history state.

Convergence requires five successive L-BFGS diagnostics with absolute loss change
relative to captured energy <=1e-5, full-gradient relative stationarity <=1e-4,
and maximum absolute gradient <=1e-7. Relative stationarity is the maximum over
parameter blocks of ||gradient||*max(||parameter||,1)/(captured-energy fraction).
Finite losses and Gram condition<=1e12 are instrument requirements. Singular or
ill-conditioned solves are optimizer issues, not scientific absence of structure.
Continuation chunks keep the same configuration and optimizer state. A time limit
is status optimization_unfinished; it does not reject a structural hypothesis.

pred_a_instrument: finite/conditioning checks and checkpoint function replay<=1e-10.
pred_b_converged: the convergence criteria above are actually met; false means
unfinished optimization, not a negative circuit verdict.
pred_c_checkpoint_replay: independently loaded checkpoint loss matches<=1e-10.

Controls: all4 representations under both metrics agree with explicit dense losses
and finite-difference gradients. Planted one-product random-start fit converges;
an unfinished checkpoint stays unfinished and resumes correctly. Stable-reader
rotation control demonstrates why individual-term matches are insufficient.

Evaluation after convergence: held-out objective/error; individual and block-level
stability across starts; literal reader/product/output-code cost; graded consumer
profiles; source folding and registered causal removal/interchange. Preserve all
native remainders/background. No fit alone establishes any of the four properties.
Red-team weak data, metric, capacity, gauges and optimization before interpreting
negatives. All32 configurations are defined; enqueue a small number of managed
chunks at a time to share the GPU and interpret outcomes before continuing.
