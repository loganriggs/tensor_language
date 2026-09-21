**Pairs of root quadratic forms can share products without refitting the function.**

The16-output program computes32 shared quadratic features q, then16 quadratic forms of q. Separate eigendecompositions used512 root squares. Reusing the existing real symmetric-pair compiler reduces this to256 root products, plus128 products for q itself.

| Program | Products | FP32 coefficients | Integer indices | Additions | Parameter/index bytes |
|---|---:|---:|---:|---:|---:|
|Unconstrained joint parent|656|903,168|0|901,856|3,612,672|
|16 separate spectral root forms|640|330,240|0|328,400|1,320,960|
|V1 paired roots, one fallback|416|323,072|672|321,636|1,297,664|
|V2 paired roots|384|322,048|768|320,666|1,294,336|

Each supported real1x1 block needs one square. A complex-conjugate2x2 block uses(u+v)(u-v)anduv; its two root outputs share those two products. Linear projections, additions, output coefficients and integer metadata are charged. The unembedding and required normalization/background remain external and are not a claimed whole-model speedup.

V1preserves the function but FAILS its<=384product price target because one pair falls back to separate spectral forms. Its matrix replay1.47e-10narrowly exceeds the unchanged1e-10bar. A preregistered CPU redteam changes only the orthogonal coordinates of that output pair. All three alternative fixed bases pass original-matrix replay below6.5e-13. V2tries those fixed bases before falling back; no error bar is relaxed and no activation target is fitted. V1failure remains archived.

V2passes the primary16-form replay and price predictions. Both opened-panel FP32replay errors are below6e-7; FP64replay below6.2e-14. Its16-form parent's native value errors were8.28261%/13.61492%; a triangle bound using the measured replay gives upper bounds8.28267%/13.61498%, preserving the earlier value-screen margins. This is a certificate on the same opened panels, not a new native forward test or OOD confirmation. The fitted parent's measured derivative errors remain21.29%/27.53%; the real-arithmetic rewrite preserves the function, but no separate FP32 derivative replay is claimed here.

The rank8fit remained the failed primary of its fitting study. Selecting16as the input to a separately registered exact rewrite does not erase that failure. Five compiler controls include repeated eigenvalues and a singular-base fallback. General defective or ill-conditioned pencils remain unsupported and must keep the fallback.

Next test the frozen paired16program in the existing native quartic-branch adapter, retaining the actual RMS denominator, residual/cross terms and attention background. Compare the full joint parent and inexpensive earlier programs. Passing polynomial reconstruction and graph pricing has not established semantic feature identity, selective removal, independent OOD prediction, or reusable causal circuits.

[V1failure](PAIRED_ROOT_NATIVE_V1.json) · [V2result](PAIRED_ROOT_NATIVE_V2.json) · [Coordinate audit](PAIR_BASIS_ROTATION_AUDIT_V1.json) · [Value-margin certificate](PAIRED_ROOT_VALUE_CERTIFICATE_V2.json) · [Compiler](paired_root_compiler.py).
