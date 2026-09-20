# Mixed midpoint tensor output spectrum — 2026-09-20 23:36 UTC

Target: complete previous-MLP-polynomial-dependent numerator at MLP17, with independent midpoint n (1152) and previous channel products p (4608). Teacher C=Ru Down17, L=Left17, R=Right17, D=lambda17[0] Down16. Ru is the actual unembedding QR frame. Bias and normalization remain explicit outside this numerator.

Measure exact output-mode Gram without forming the tensor. For M=DD^T, H=(LL^T)⊙(RMR^T)+(RR^T)⊙(LML^T)+(LR^T)⊙(RML^T)+(RL^T)⊙(LMR^T); Gram=CHC^T. Output rank-r imposes relative coefficient Frobenius error at least sqrt(sum eigenvalues beyond r / total). This constrains shared output directions, not number of semantic circuits or functional error on dependent native inputs.

Registered predictions: (a) toy Gram/eigenspectrum replay <1e-12, native PSD relative negative mass surrogate >-1e-10, trace vs independent teacher_cross self expression <1e-8; (b) output rank64 floor <0.5; (c) output rank256 floor <0.2. Null: broad output spectrum prevents these capacities from accurately matching isotropic coefficients, without ruling out covariance-weighted or native-input simplification. Widths 4/16/32/64/128/256/512/1024/1152 reported regardless of bars.

Price: native checkpoint load, QR, FP64 matrix contractions up to4608², output eigendecomposition1152²; zero text forwards and no fit. Scalar global normalization of each factor preserves relative spectrum. Circuit relevance: tests capacity feasibility for a target grouping all previous-source self and cross interactions, before optimization; no semantic identification claim.
