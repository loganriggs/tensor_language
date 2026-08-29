# MLP2 CMR v1 FIT_MEAN collection addendum

Frozen before any MLP2 CMR model forward.

This stage may load the exact pinned bilin18 checkpoint and execute only the
`FIT_MEAN` role.  It may accumulate MLP2 native product counts, sums, squared sums,
means, variances, and weight-derived LOCAL/RMS/MASS controls.  It must not access
next-token targets, compute or retain losses/KL/accuracy, fit the SUFFIX selector,
open `FIT_SELECTOR`, `VALIDATION`, or `REPLICATION` in the model, or publish raw
activations, residual states, gradients, or logits.

The existing observed-model dispatcher executes the exact native prefix through the
MLP2 input and MLP2 `Left`/`Right` product.  A private control-flow exception then
stops the call before the MLP2 write, layers 3--17, final RMSNorm, and logits.  Those
operations cannot affect the already-computed MLP2 product moments, so executing them
would add cost without evidence.  Exactly 48 four-document prefix calls are allowed.
MLP2 products contribute only where the published `FIT_MEAN` eligibility mask is
true.  No logit tensor is constructed.

The fit bundle must contain:

- the exact eligible observation count;
- product mean, variance, and second moment for all 4,608 native products;
- squared norms of MLP2 `Left`, `Right`, and `Down` factors;
- LOCAL, RMS, and MASS scores and their frozen top-512 supports;
- one hash-random 512-support control;
- reciprocal-scale gauge replay for LOCAL/RMS/MASS and the materialized score ranks;
- complete checkpoint, source, token-parent, call, shape, dtype, and price receipts.

This stage cannot promote a simplification.  It is a required fit artifact for the
later SUFFIX-selector and finite-consequence transaction.
