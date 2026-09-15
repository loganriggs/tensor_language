# Setting2 regional attention9-head × MLP16 fold V2 correction

V1 is preserved as invalid. Its nine projected native head writes reproduce the
separately rounded float32 attention9 block output with relative error
$1.43\times10^{-7}$, inside the frozen $10^{-6}$ native bridge. V1 then used that
separately rounded block output as the direct reference for a stricter $10^{-8}$
folded algebra check, producing error $2.85\times10^{-8}$ and invalidating the run.

V2 defines the exact whole-head folding reference as the sum of the same nine
projected head writes. The native float32 block-output bridge remains separately
measured against the original $10^{-6}$ gate. This separates algebraic closure
from the already permitted native rounding boundary.

Rows, head values, native projections, propagation coefficients, MLP16 write,
MLP17 factors, denominator, unembedding readers, ranking, top-two selection,
scientific predictions, thresholds, price, and all outcome prohibitions are
unchanged. V1 rankings are not used to change any decision. V2 writes a new result
and binds the invalid V1 receipt plus this correction before execution.
