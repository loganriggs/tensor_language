# Native attention8 write: forward response census

Freeze the native8 midpoint from TYPED_FACE_NATIVE8_SCOPE_V1. Same 40 opened
sequences and 240 endpoint rows; native plus native8 midpoint, 80 forwards/300s.
No direction, support, strength, or row changes. The destination mask excludes
final position; MLP8 is positionwise, so final residual at block8 remains native.
Transport attention/MLP9–17 finite responses through subsequent skip lambdas;
include final RMS normalization and per-token softcap secants explicitly.

pred_a: replay scope arms[0,2] <=1e-5 absolute; reconstructed final residual
relative error<=1e-4 and readout error<=1e-4 absolute/1e-3 relative; finite,
80 forwards. Assert every destination excludes the scored final position.
pred_b: attention9 response alone predicts full native8 effect <=.35 relative
L2 in every family. pred_c: attention9 plus MLP9,10,11 plus output normalization
predicts <=.35 every family. pred_d: line-break versus each other family's
paired response-allocation vector cosine>=.9.

Opposing prediction: the broader application engages a distributed downstream
response, so neither direct nor fixed early set suffices. These are response
attributions, not selective edits or evidence of an extracted suffix. Retain
negative terms and failed gates. The four behavioral properties and separate
simplicity requirement are unchanged. Any later suffix proposal must use this
census instead of the older conditional-only census.
