# Subject-number frozen embedding decoder fresh V1

Registered after freezing the decoder and then authoring a disjoint 16-pair lexical
authority. The panel contains six irregular and ten regular singular/plural pairs,
all distinct one-token forms absent from decoder discovery. No decoder score or
model weight was read while selecting the forms.

Apply the frozen normalized-embedding axis and threshold without refitting. Require
overall endpoint accuracy and pair-order accuracy at least `.90`, endpoint accuracy
at least `.80` separately for irregular and regular strata, and positive mean signed
margin in both strata.

As a matched null, reconstruct the 96 opened training embedding rows and draw 64
unit Gaussian axes. Each null threshold uses the same mean training-midpoint rule;
the frozen decoder must beat median fresh null endpoint accuracy by at least `.20`.

Instrumentation requires the bound authority and frozen decoder hashes, exactly 32
fresh and 96 training embedding rows, 64 random controls, finite scores, and zero
model forwards, gradients, fits, or updates. Passing establishes prospective lexical
OOD prediction for the native token-number node only; behavioral writer integration,
removal, and composition remain separate tests.
