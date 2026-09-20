# Native retained quadratic spectral audit — 2026-09-20 17:32 UTC

Native hierarchy baseline completed:8roots heldout coefficient98.816%,Gaussian92.836%;256roots96.294%,Gaussian91.622%. Both improvement predictions:8rootbeatsCP held;128root<95% failed. Fullbank10.70M values at8roots. Training/evaluation gap increases withwidth despitecondition≤2.60.

Audit the16 quadratic matrices inside the exported8root model on CPU. Q=Sym(A^T diag(left_k) B), similarlyright. Report numerical positive/negative eigenvalue counts at relative thresholds1e-10/1e-6/1e-4/1e-2 and signed eigen-truncation errors at bilinear widths16/32/64/128/256/512. Retain the largest k positive and k negative eigenvalues; pair signs into products. Validate fullmatrix evaluation against the exported sharedbank before approximation.

Measure each compressed program against the exported8root program on8192 independently sampled coefficient tuples and256 Gaussian vectors (seed1732), never the original teacher. Original teacher errors remain from independent baseline receipt. Price standalone independent quadratic factors4*r*k*d+outputd*r; compare completebank price. Dense spectral directions need not be interpretable, and independently factoring each quadratic can lose bank reuse. This is approximation/pricing evidence, not native circuit adoption or certified global quartic error.
