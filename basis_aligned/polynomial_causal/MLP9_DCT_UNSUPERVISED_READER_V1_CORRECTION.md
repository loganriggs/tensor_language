# MLP9 DCT unsupervised-reader V1 implementation correction

The first execution (runner SHA-256
`1f0a4908456f7d05317dfda910d6d8dcf7a0796571d70417c913205a93e08727`)
wrote files now preserved as `INVALID_FIRST_RESULT` and
`INVALID_FIRST_ARTIFACT`.  Its analytic/autodiff tensor contraction check
passed, but the serialized factor reconstruction was not the reconstruction
used by the solver.

The serializer canonicalized each symmetric input factor's sign and also
flipped the paired output factor.  That is incorrect: the recovered term is
`alpha * u ⊗ v ⊗ v`, so replacing `v` by `-v` leaves the term unchanged and
must not change `u`.  The erroneous output flip can turn fitted terms into
their negatives and invalidate the solver-residual predicate.  The first
scientific terminal is therefore retracted rather than interpreted.

The corrected runner removes only the output-factor sign flip.  It retains the
original preregistered rows, seeds, solver, price, predictions, and thresholds,
receives a new hash, and binds this correction.  Subspace-only quantities are
mathematically sign-invariant, but all quantities are recomputed rather than
carried forward.
