# Token readers compared as bilinear functions

10 September 2026. Original bilinear handoff and user's two unembedding views.
This is a weight-only candidate screen for reusable computations, not a rank,
variance, whole-layer replacement or behavioral success test.

Prior-art check: MLP17 dossier, readout-L15-17, failed four-quadratic compiler,
normalized response v184-v186, reader-energy and signed-reader-pair experiments,
and UNEMBEDDING_BACKWARD_VIEWS_V1. Those do not establish equivalence between
individual token quadratic readers. The old fixed hierarchy is not refitted.

At MLP17 let c_t=U_t D and Q_t=sym(L^T diag(c_t) R). Its contribution before
final normalization is x^T Q_t x/(mean(x^2)+epsilon)+U_t bias. For product
features H_i=sym(l_i r_i^T), their exact Frobenius Gram is

    K_ij = [(l_i·l_j)(r_i·r_j)+(l_i·r_j)(r_i·l_j)]/2.
    <Q_t,Q_s> = c_t K c_s^T.

This avoids materializing one 1152-square form per token. It is invariant to
exchanging bilinear halves, permutations, and compensating product rescalings.
The input-coordinate metric is the native Euclidean metric, not arbitrary GL
gauge invariant. Compare also trace-free forms Q°=Q-tr(Q)I/1152, so a shared
isotropic norm response alone cannot nominate a shared token computation.

Freeze the same 518 token IDs and 16 raw centroids from the saved hierarchy.
For every token, select its nearest OTHER token by absolute trace-free cosine,
ties by token ID. For that pair report best signed scalar alpha, relative
trace-free error sqrt(1-cos²), full-Q error under its own optimal alpha, and
raw-unembedding error under its own optimal alpha. This is weight-side search;
any subsequent causal experiment must freeze its candidates before new data.
No fitted change to the old hierarchy, model, rank, threshold or native task.

Predictions registered before execution:

* A instrument: CPU FP64 dense-form, half-exchange and rescaling controls
  <=1e-10 relative; trained first two token forms and centroid zero, explicitly
  materialized in FP64, agree with FP32 contracted Gram <=1e-5 relative;
  no negative squared norms beyond 1e-5 of largest norm; finite outputs.
* B sharing candidates: at least 16 of 518 token readers have a selected partner
  with BOTH full and trace-free relative error <=.10, while raw-unembedding
  optimal relative error >.50. This distinguishes a potentially new common
  bilinear computation from near-duplicate unembedding rows. Zero is the strong
  null; 1-15 is a smaller descriptive candidate set, not this prediction passing.

Also report fixed-centroid errors in U, Q and Q° without a new success threshold.
No extraction, OOD, removal, composition, semantic label or gain claim follows
from B. The residual skip, bias and final RMS/softcap remain separate consumers;
even exact Q equivalence would only identify this MLP contribution.

Price: zero model forwards, one managed GPU weight contraction job <=900s.
All native parameters remain; temporary 4608-square Gram and 534-reader
coefficients/Gram; no model replacement or saving. A trained FP64 control and
tiny CPU controls test the metric rather than use outcome-only plausibility.
Next CPU receipt will inspect candidate identities and raw-versus-function
neighbors; do not auto-promote or expand the sample if B fails.
