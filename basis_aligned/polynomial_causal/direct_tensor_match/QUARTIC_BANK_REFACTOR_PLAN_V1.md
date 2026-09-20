# Refactor the four quadratic bank features jointly

The exact bank tensor is4x32x32 in the union of its32 weighted reader directions;
CPU replay is3.2e-15. Root-sensitivity weighting has condition38.2. Its input-mode
spectrum puts93.81% energy in two directions and99.56% in eight. This is not a
CP-rank bound, nor a guarantee on the composed quartic error.

Fit a shared-product representation q_g=sum_k C_gk(a_k x)(b_k x), widths8/12,
Adam/Muon,.005/.03, two seeds,500steps. Use exact4x32x32 weighted core loss with
normalized unit quadratic features and writer penalty.001 to discourage the
previous large cancellation failure. Preserve root mixing and output frame;
only replace16primitive bank products by8or12 shared products and bank writers.
Map through inverse root-sensitivity square root and saved input mapback.
Keep original Gaussian teacher mean via analytic constant correction. Select
withinwidth using penalized weight objective only. Full quartic evaluation on
existing diagnostic and fresh panels is mandatory: local sensitivity weighting
is a surrogate and ignores correlated higher-order composition error.

Predictions: width8 retains>=99% weighted bank tensor energy; its full fresh64
quartic error increases<=.01 absolute from.17440681; literal coefficients and
products decrease after charging the4xwidth bank writer. Width12 is a registered
capacity control. Audit component cancellation and actual DAG sharing, rather
than equating a good reconstruction with stable identified features.
