# Full folded objective through a shared sparse dictionary: native step

Use the two frozen learned OVERCOMPLETE_OLS_REENCODE_V1 programs, not the live
coupled-L1 caches. Both have 2304 shared input features and 128 coefficients
per native reader. Keep support indices, native product pairing, Down and bias.
Vary all dictionary directions and all signed sparse values together.

Unlike normalized-reader Lasso, optimize the complete folded coefficient error
directly, including the full unembedding metric and output cross-cancellations.
The target contains every native reader: the historical weight test split is
now included in the objective and must not be reported as held-out discovery
evidence. No activations, text, CE or distributional weights enter discovery.

Reuse chunked_bilinear_coefficient_v1 for exact product-factor loss/gradients
and folded_sparse_dictionary_v1 for the sparse/shared-feature chain rule.
Parameterize dictionary rows by normalized raw directions and code values
divided by their fixed native reader norms. This changes parameter scales,
not the physical function or the objective. No L1 penalty is added.

Measure one initial full-gradient call per start. Choose a descent direction
by separately scaling negative dictionary and code gradients to the norms of
their parameter blocks. Start step .05 and halve up to 12 times, accepting
the Armijo condition with coefficient1e-4. This is one controlled descent step,
not a converged native factorization or a general optimizer comparison.
Finite differences along the actual direction use h=1e-5. Save the accepted
program, replay its sparse execution, and independently compute its complete
CP tensor error using the existing inner-product implementation.

Predictions:

- A: initial parent capture replay and final independent CP capture replay
  <=1e-8; finite-difference versus analytic directional derivative error
  <=1e-6 relative to max(1, absolute derivative); sparse/dense output replay
  <=1e-8; finite scores; unit dictionary norms within1e-6; finite gradients.
- B: both starts accept a descent step, satisfy Armijo, and do not worsen
  the full coefficient objective by more than1e-10.
- C: both gain at least .005 absolute full coefficient capture in that step.
- D: each initial full-gradient evaluation takes at most30seconds on the
  managed GPU. Report actual timings including synchronization.

Report normalized-reader error before/after and both parameter gradient norms;
do not use reader scores to select the direction or step. The cancellation
fixture shows these objectives can conflict but does not predict their native
alignment. A failure invalidates the instrument; B failure with A held requires
step/conditioning analysis; C miss bounds only this single step; D miss directs
kernel optimization before a long run. No structural absence conclusion.

Matrix/support price remains9,142,272 coefficients,1,179,648indices and1152bias
values, with runtime COO indices/U/background also charged. Zero body passes
or text accesses. Managed lane1 after current coupled L1 run; whole-job600second
alarm. Save per-start receipts before final combined result. Do not infer any
of the four circuit properties from a passed gradient or reconstruction check.
