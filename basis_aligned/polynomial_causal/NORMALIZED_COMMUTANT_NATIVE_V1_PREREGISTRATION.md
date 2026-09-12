# Native normalized commutant: spectral search registration

12 September2026. This job is prepared during the direct-rotation run and may be
submitted only after that run's terminal result is read. It evaluates a different
search formulation, not a repair to that run's bars.

Target: the same full centered-U pure MLP16/MLP17 path in the exact producer
metric; all1152input dimensions, symmetric operator dimension664,128.
The [relaxation note](NORMALIZED_COMMUTANT_RELAXATION_V1_MATH.md) defines
$\mathcal L,\mathcal M$. In whitened coordinates use

$$
\mathcal A=\tfrac12 I+\mathcal M^{-1/2}\Phi\mathcal M^{-1/2}
=I-\tfrac12\mathcal M^{-1/2}\mathcal L\mathcal M^{-1/2}.
$$

Remove the transformed identity before and after each action. On the remaining
space, $\mathcal A$ has spectrum in[0,1]: the commutator and anticommutator
squared-norm identities give $-\mathcal M\preceq2\Phi\preceq\mathcal M$.
This makes the existing largest-magnitude symmetric eigensolver appropriate.
Convert its largest eigenvalues $\mu$ to relaxed values $\lambda=2(1-\mu)$.

Rotate readers into the eigenbasis of $K$ once; mass whitening then divides each
entry by $\sqrt{k_i+k_j}$. No dense664,128square matrix is constructed. Existing
packed coordinates preserve Frobenius norms with off-diagonal square-root-two
weights. CPU controls reproduce both smallest dense generalized eigenvalues to
$3.1\times10^{-15}$, remove the trivial mode exactly, and recover planted blocks.

Two independent Lanczos starts:120440/120441. Each requests two eigenpairs,
32Lanczos vectors, relative tolerance1e-8, up to100ARPACK iterations and
600operator applications. The existing adapter returns an explicit action-limit
status rather than silently treating incomplete iterations as convergence.
Alarm900seconds for the job. Record actions and residuals, not only elapsed time.

Round each leading witness into a576/576projector by sorting its eigenvalues,
then compute the exact all-output normalized cut and incident-energy balance.
No local polishing or alternative ranks inside this comparison.

- A: both solvers return two converged pairs with relative residuals<=1e-7,
  exact transformed identity action norm<=1e-10, projector orthogonality<=1e-8,
  positive-definite $K$, finite values, and independent estimates agree<=1e-5.
- B: both rounded normalized cuts<=0.1, incident energy in[0.1,0.9], and
  at least20% lower than the prior matched random-basis control's mean cut.
- C: the two rounded partitions overlap>=0.9 allowing block exchange.

Also report the unrounded relaxed eigenvalues and eigengap. These are numerical
Ritz estimates, **not certified lower bounds**. Even A passing does not prove
global extremality or absence of smaller eigenvalues. Low relaxed value with
poor rounding is a relaxation gap, not a usable decomposition. Negative results
exclude neither nonorthogonal/overlapping arithmetic graphs nor other producer
metrics. A passing candidate still needs frozen native extraction, selective
removal, composition and OOD tests, with every adapter and native parent priced.
