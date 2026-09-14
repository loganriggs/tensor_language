# Weights-only scalar refit inside the existing sparse support

The complete normalized objective ranks ordinary/balanced shared-output candidates
worse than sparse in total error at matched storage. Balanced improves paired
contrasts slightly but worsens total energy. Do not refit those completed graphs.

Let S be the decoded frozen sparse tensor, B the decoded balanced shared tensor,
and T the exact tensor. Express B-S in S's stored output/head coordinate frames
and zero entries outside S's frozen occupancy mask. Decode this direction D.
Fit only alpha in S+alpha D, minimizing the complete normalized conditional
mixed-error objective under the exact synthetic law used in PAIR_RANKING_V1.
The exact scalar least-squares solution is -mean(<e,d>)/mean(||d||²).
All three producer terms, both QK factors, inherited values, exact source sums,
shared background and supplied exact conditional RMS remain included.

Freeze two independent 8192-context training panels: seeds170217000..170217127
and170218000..170218127,64contexts perseed. Fit alpha independently; choose the
first panel's alpha in advance for serialization. One separate8192-context
validation panel: seeds170219000..170219127. No text, adaptive sampling or new
support selection. CPU FP64/two threads/120seconds; coefficients stored FP32.

A: directional quadratic reconstruction and stored FP32 executor/dense replay
relative errors<=1e-6; mask/adapters exactly unchanged; final program bytes within
1% of the original sparse artifact. Both native input frames and output constants
must be priced, but no second full program is retained at runtime.
B: frozen first-alpha candidate reduces validation total squared error >=5%
versus original sparse, with positive paired improvement >3standard errors.
C: validation paired-contrast squared error <=1.01times original and alpha fits
agree within0.1 absolute. All clauses must pass. Preserve misses as written.

The null is no worthwhile matched-cost improvement under this fixed directional
class/measure. Local scalar optimality is exact conditional on this direction
and sample, not a general sparse-graph optimum. A positive synthetic result
requires fresh native-port validation before any circuit or adoption claim.
Independent Gaussian raw edits are not the original circuit's intervention law;
that limitation must remain explicit even after a passing refit.
