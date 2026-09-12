# Spectral normalization dependency

For each native128x1152 map A, eigendecompose AA^T and obtain right singular
readers. Keep k=32/64 leading directions of A^TA; tail coefficient beta is
sum omitted eigenvalues/(1152-k). Approximate ||Ax||² by retained eigenvalue-
weighted readings squared plus beta times remaining orthogonal input norm².
Divide by128 and add nativeepsilon. Rank128 restores the exact norm quadratic.
No input samples, fitted scales or objective optimization are used in construction.
Native sharedsource numerator/parent/privatewriters stay frozen. Arms native,
query32,key32,both32,both64,both128. A native/priorcomponent and full128 replay
relative<=1e-5. B both32 write AND signed-prefix removal-effect error<=.1 in
EACHfamily; C both64 same. Separate query/key32diagnostics. Null: this isotropic
remainder still misses circuit effects. Spectrum convergence isn't circuit proof.
9bodybatches48rows13–19tokens7suffixarms180secmanagedcap,<4MB artifact. Compiled
norm readers cost4*9*k*1152 scalars plus eigenvalues/remainder coefficients,
versus4*9*128*1152 native norm maps. Numerator and upstream weights remain charged;
this runner retains native maps for reference, so no measured fullmodel saving.
