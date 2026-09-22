Streamed exact gradients for shared quadratic producer learning

22 September2026. Scope: implementation/control result, not native feature-learning result.

With fixed active rootpairs and shared quadratic factors U,V, define the mixed selfGram and teacher crossGram

$$
G=(G_G+\lambda G_F)/(1+\lambda),\qquad
X=(X_G+\lambda X_F)/(1+\lambda).
$$

G_G uses the fitted mean/covariance Gaussian; G_F matches symmetric coefficient tensors. The profiled regularized loss omits the constant teacher norm:

$$
L(U,V)=\min_C\ \operatorname{tr}(CGC^\top)-2\langle C,X\rangle+\rho\|C\|_F^2,
\qquad C=X(G+\rho I)^{-1}.
$$

At this exact minimizer the derivative through C cancels by stationarity. Hold C detached, compute rectangular Gram rowblocks, multiply by the corresponding blocks of C^T C, and accumulate gradients. Likewise stream teacher-cross blocks with weight -2C. Recreate the covariance-transformed features in every block and release its autograd graph before the next block. This changes memory use, not the objective or gradient. All rowblocks are included; do not halve offdiagonal contributions.

The new module is independent of the helpers used by the then-live native support exchange. Gaussian rectangular contractions use the same15partitions/four-quadratic cumulant identities already checked against independent degree8quadrature; coefficient rectangles use the symmetrized quartic trace formula. New tests compare disjoint rectangles to fullmatrices and streamed gradients to full autodiff for independent, sharedinput, sharedoutput, squares and cancellation families. Max gradient discrepancy1.38e-13; rectangular values match exactly in these fixtures.

Correction retained: the initial profiled finite-difference check used rho1e-6 and step1e-5 on a highly redundant, badly scaled cancellation fixture. It failed despite the exact-autodiff gradient matching. Using rho.01 for this numerical fixture and showing the step ladder1e-5,1e-6,1e-7 demonstrates truncation-error convergence: cancellation relativeerrors.00551,.0000551,.000000908. The larger step still fails and is retained. The five final smallest-step errors are all below1e-5. This fixture adjustment is not a change to the native objective, whose intended ridge remains1e-6. A temporary editing error during test repair caused NameError before any result file; the restored full test passes.

Before native optimization: profile one exact forward/gradient step at144quadratics,4terms,1152inputdimensions,512rootpairs,16outputs. Compare directional finite differences or fullgradients at a tractable reduced size, and check native scalar replay/stationarity. Select a bounded optimizer schedule from measured time/memory and existing Muon/Adam controls. Do not infer that passing gradients implies useful learnable circuits.
